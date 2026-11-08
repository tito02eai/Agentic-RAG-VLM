"""BluPe shared-runner integration using its trusted ``run_session`` hook.

The upstream RunnerController owns queue admission, capability and Stop. This
provider adopts the already leased session; it never creates a second session.
"""
from dataclasses import replace
from pathlib import Path
import json
import time
import uuid
import threading

from ..clip_client import CLIPFeatureClient
from ..qwen_adapter import OpenAICompatibleQwenVL
from ..services import resolve_model
from ..experience import atomic_json
from ..task_planner import VLMTaskPlanner
from .blupe_client import BluPeSessionClient,BluPeAPIError,SessionHandle
from .config import load_platform_config
from .deployment import validate_profile,validate_profile_fields,validate_execution_binding
from .motion import YamMotionPlanner
from .transport import RemotePlatformTransport,PlatformUnavailableError
from .session import BluPeSession
from .commands import GraspActionMapper
from .verifier import SceneChangeVerifier
from .policy import paper_components,bind_observation


class RunnerBoundClient(BluPeSessionClient):
    """Use public runner methods for leased calls; credentials remain upstream."""
    def __init__(self,endpoint,api,context,*,cancelled=lambda:False):
        super().__init__(endpoint,max_retries=0)
        self.api=api;self.context=dict(context);self.cancelled=cancelled
        self._sleep=self._interruptible_sleep

    def _guard(self,handle=None):
        if self.cancelled():raise BluPeAPIError('runner_cancelled')
        if handle is not None and handle.session_id!=self.context['session_id']:
            raise BluPeAPIError('runner session identity mismatch')

    def _interruptible_sleep(self,seconds):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            self._guard();time.sleep(min(.1,max(0,deadline-time.monotonic())))

    def create_session(self,*a,**kw):raise BluPeAPIError('queue admission belongs to RunnerController')
    def stop_session(self,*a,**kw):raise BluPeAPIError('Stop belongs to RunnerController')
    def get_station_observation(self,robot_id):
        self._guard()
        if robot_id!=self.context['robot_id']:raise BluPeAPIError('runner robot identity mismatch')
        return dict(self.api.get_robot_observation(robot_id))
    def get_queue(self,robot_id):
        self._guard();return dict(self.api.get_queue_snapshot())
    def get_session(self,handle):
        self._guard(handle);return dict(self.api.get_session(handle.session_id))
    def get_episode_trace(self,handle):
        self._guard(handle);return dict(self.api.get_episode_trace(handle.session_id))
    def submit_trajectory_packet(self,handle,packet):
        self._guard(handle)
        if getattr(self.api,'supports_trajectories',False) is not True:
            raise BluPeAPIError('runner must explicitly enable supports_trajectories')
        current=self.get_session(handle)
        if current.get('cadence_hz',10.)!=10. or packet.get('cadence_hz')!=10.:
            raise BluPeAPIError('this adapter supports 10-Hz sessions; do not mix runner/trajectory cadence')
        if (current.get('episode_id')!=self.context['episode_id'] or not current.get('lease_id')
                or current.get('status')!='running' or current.get('active_trajectory') is not None):
            raise BluPeAPIError('runner episode/lease is unavailable or changed')
        handle.lease_id=current['lease_id']
        self._guard(handle)
        return dict(self.api.submit_trajectory(handle.session_id,self.context['episode_id'],handle.lease_id,
            packet['trajectory_id'],packet['cadence_hz'],packet['waypoints']))


class AgenticRAGVLMProvider:
    """Drop-in trusted provider for RunnerController's run_session hook.

    Load/register/prepare are non-actuating. ``run_session`` is invoked only by
    the runner for a user-created leased run. ``preview`` never uses its API.
    """
    provider_name='agentic-rag-vlm'
    model='Agentic RAG-VLM'

    def __init__(self,*,knowledge_path,registry,platform_config,profile=None,task=None,destinations=None,
                 endpoint='http://127.0.0.1:18074/v1/chat/completions',model=None,
                 clip_endpoint='http://127.0.0.1:18072',output='output/platform',client=None,encoder=None,candidate_provider=None):
        self.knowledge_path=Path(knowledge_path);self.registry=registry
        self.platform_config=platform_config;self.profile=profile;self.task=task
        self.destinations=dict(destinations or {});self.endpoint=endpoint;self.served_model=model
        self.clip_endpoint=clip_endpoint;self.output=Path(output);self.client=client;self.encoder=encoder
        self.candidate_provider=candidate_provider
        self._active=False;self._prepared=False;self.cancelled=lambda:False
        self._run_lock=threading.Lock()
        self.interaction_sink=None
        self._runtime_event=None

    def prepare(self):
        if self.client is None:
            self.served_model=resolve_model(self.endpoint,self.served_model)
            self.client=OpenAICompatibleQwenVL(self.endpoint,self.served_model)
        if self.encoder is None:self.encoder=CLIPFeatureClient(self.clip_endpoint)
        from ..paper import PaperAgent
        agent=PaperAgent(self.knowledge_path,self.client)
        if agent.retriever.store.embedding_space!=self.encoder.embedding_space:raise ValueError('encoder/store mismatch')
        self._prepared=True
        return self.public_config()

    def public_config(self):
        return {'provider':self.provider_name,'model':self.model,'profile':'paper_haa_clip_vlm_review',
            'prepared':self._prepared,'site_geometry_configured':self.profile is not None,
            'underlying_models':{'vlm_requested':self.served_model or getattr(self.client,'model',None),
                'visual_encoder':getattr(self.encoder,'embedding_space',None)},
            'queue_owner':'RunnerController','hardware_qualification':'not_claimed',
            'task_progress':{'stage':self._runtime_event['phase'] if self._runtime_event else 'ready','active':self._active}}

    def _runtime_notice(self,event):
        self._runtime_event=event
        if self.interaction_sink is not None:
            self.interaction_sink('run_notice','Agent: '+event['event'],phase=event['phase'],payload=event['payload'])

    def infer(self,prompt,observation):
        raise RuntimeError('This full task policy requires RunnerController.run_session; do not turn a task into one guessed joint command')

    def prepare_before_session(self,prompt):
        """Official runner hook: reject unusable execution before queue admission.

        Plain ``prepare`` and previews remain available without site geometry.
        This hook is called only when an operator actually requests a task run.
        """
        if self.cancelled():raise BluPeAPIError('runner_cancelled')
        if not isinstance(prompt,str) or not prompt.strip():raise ValueError('task prompt required')
        if self.profile is None:
            raise ValueError('Configure site geometry before joining a robot session; use preview for software-only validation')
        if self.profile.get('robot_id')!=self.platform_config.robot_id:
            raise ValueError('deployment profile robot mismatch')
        if self.task is not None and prompt.strip()!=self.task.instruction.strip():
            raise ValueError('configured task prompt differs')
        if self.task is None and not self.destinations:
            raise ValueError('named destinations required before joining a robot session')
        # Local validation catches missing/invalid profile fields. The live
        # calibration is still checked against the site immediately on adoption.
        validate_profile_fields(self.profile,robot_id=self.platform_config.robot_id,arm=self.profile.get('arm'))
        validate_execution_binding(self.profile,self.platform_config.endpoint_base)
        self.prepare()
        if self.cancelled():raise BluPeAPIError('runner_cancelled')

    def validate_session_admission(self,session_api):
        if self.cancelled():raise BluPeAPIError('runner_cancelled')
        if self.profile:validate_execution_binding(self.profile,self.platform_config.endpoint_base,session_api=session_api)
        if getattr(session_api,'supports_trajectories',False) is not True:
            raise BluPeAPIError('runner must explicitly enable supports_trajectories before queue admission')

    def preview(self,task,observation,calibration,geometry):
        self.prepare()
        runtime,perception=paper_components(self.knowledge_path,self.client,self.encoder,self.registry,calibration,geometry,candidate_provider=self.candidate_provider)
        runtime.event_sink=self._runtime_notice
        from .perception import PerceptionError
        try:objects,receipt=bind_observation(runtime,perception,observation)
        except PerceptionError as error:
            raw=Path(observation.image_paths[0]).parent/'grounded_vlm_response.json'
            return {'execution_performed':False,'profile':'paper_haa_clip_vlm_review','decision':'stop','action':None,
                'reason':'perception_not_grounded','message':str(error),'trace':[],
                'perception':json.loads(raw.read_text()) if raw.exists() else {},'reasoner_calls':[]}
        runtime.start(task)
        missing=set(task.protected_objects)-set(objects)
        if missing:runtime.safe_stop('protected_object_not_visible',missing_objects=sorted(missing))
        else:runtime.observe(objects,receipt={'observation_id':observation.observation_id})
        action=None
        from ..runtime import RuntimePhase
        if runtime.phase==RuntimePhase.PLAN:
            try:action=runtime.plan(objects)
            except ValueError as error:runtime.safe_stop('plan_not_approved',message=str(error))
        return {'execution_performed':False,'profile':'paper_haa_clip_vlm_review',
                'decision':'propose' if action is not None else 'stop',
                'action':action.to_dict() if action is not None else None,
                'perception':receipt.to_dict(),'trace':runtime.trace(),'reasoner_calls':runtime.pipeline.reasoner.calls}

    def run_with_transport(self,task,transport,calibration,geometry,*,require_release=True):
        self.prepare()
        runtime,perception=paper_components(self.knowledge_path,self.client,self.encoder,self.registry,calibration,geometry,candidate_provider=self.candidate_provider)
        runtime.event_sink=self._runtime_notice
        session=BluPeSession(runtime,transport,perception,GraspActionMapper(),SceneChangeVerifier(),
            destinations=self.destinations,require_release_confirmation=require_release)
        result=session.run(task)
        # Preserve VLM vetoes/errors too: an unapproved plan never reaches the
        # subgoal_planned trace event, so trace alone loses that raw response.
        result['reasoner_calls']=runtime.pipeline.reasoner.calls if runtime.pipeline.reasoner else []
        result['memory']=runtime.pipeline.memory.to_dict()
        return result

    def run_session(self,prompt,observation,session_api,context,cancelled):
        if self.profile is None:raise ValueError('Configure site geometry for execution; provider deployment/preview does not require a live robot')
        if context.get('robot_id')!=self.platform_config.robot_id:raise ValueError('runner robot mismatch')
        if not all(context.get(k) for k in ('session_id','episode_id','lease_id')):raise ValueError('leased runner context required')
        if not self._run_lock.acquire(blocking=False):raise RuntimeError('provider instance already has an active run')
        self._active=True
        folder=self.output/uuid.uuid4().hex
        result=None
        try:
            folder.mkdir(parents=True,exist_ok=False)
            self.prepare()
            if cancelled():
                atomic_json(folder/'result.json',{'outcome':'cancelled','execution_performed':False})
                return {'state':'cancelled','result_path':str(folder/'result.json')}
            client=RunnerBoundClient(self.platform_config.endpoint_base,session_api,context,cancelled=cancelled)
            validate_execution_binding(self.profile,self.platform_config.endpoint_base,session_api=session_api)
            calibration=client.get_calibration(context['robot_id'])
            validate_profile(self.profile,calibration,robot_id=context['robot_id'],arm=self.profile['arm'])
            transport=RemotePlatformTransport(self.platform_config,client=client,deployment_profile=self.profile,
                arm=self.profile['arm'],destinations=self.destinations,image_dir=folder/'images')
            transport.motion_planner=YamMotionPlanner.from_deployment(calibration,self.profile)
            transport.handle=SessionHandle(context['session_id'],'',context['robot_id'],'running','',
                episode_id=context['episode_id'],lease_id=context['lease_id'])
            step=observation.get('step_id',0)
            if type(step) is not int or not 0<=step<3000:raise ValueError('runner step_id must be a valid next contiguous waypoint ID')
            transport._step_id=step;transport._waypoints_submitted=step
            task=self.task
            if task is None:
                runtime,perception=paper_components(self.knowledge_path,self.client,self.encoder,self.registry,calibration,self.profile,candidate_provider=self.candidate_provider)
                first=transport.get_observation();objects,receipt=bind_observation(runtime,perception,first)
                atomic_json(folder/'initial_perception.json',receipt.to_dict())
                task,planning=VLMTaskPlanner(self.client).plan(folder.name,prompt,objects,self.destinations,
                    protected_objects=tuple(self.profile.get('protected_objects',())),images=first.image_paths)
                atomic_json(folder/'task_planning.json',planning)
            elif prompt.strip()!=task.instruction.strip():
                raise ValueError('configured task prompt differs; use task=None for validated natural-language planning')
            result=self.run_with_transport(task,transport,calibration,self.profile)
            result.update(framework_profile='paper_haa_clip_vlm_review',queue_owner='RunnerController',
                hardware_success=None,knowledge_role='retrieved_prior; unsupported force/synergy are not calibrated hardware controls')
            atomic_json(folder/'result.json',result)
            if cancelled():return {'state':'cancelled','result_path':str(folder/'result.json')}
            if result['outcome']!='complete':raise RuntimeError('Agent stopped; see '+str(folder/'result.json'))
            return {'state':'completed','result_path':str(folder/'result.json')}
        except Exception as error:
            import contextlib
            with contextlib.suppress(OSError):
                atomic_json(folder/'error.json',{'error_type':type(error).__name__,'message':str(error),
                    'queue_owner':'RunnerController','hardware_success':None})
            raise
        finally:
            self._active=False
            self._run_lock.release()


def load_provider(config_path):
    """Config paths resolve relative to the config file, not a developer home."""
    path=Path(config_path).resolve();data=json.loads(path.read_text());base=path.parent
    def file(name):
        value=data.get(name)
        if value is None:return None
        candidate=Path(value);return candidate if candidate.is_absolute() else base/candidate
    from scripts.agentic_framework import load_semantic_registry
    from scripts.run_platform_session import load_task
    task,destinations=(load_task(file('task')) if file('task') else (None,data.get('destinations',{})))
    if file('task'):
        raw=json.loads(file('task').read_text())
        if raw.get('coordinate_frame')!='arm_base' or raw.get('remote_execution_allowed') is False:
            raise ValueError('runner task must explicitly use arm_base; mock tasks cannot bind to a runner')
    profile=json.loads(file('deployment_profile').read_text()) if file('deployment_profile') else None
    candidates=None
    if file('candidate_config'):
        from .candidates import load_candidate_provider
        candidates=load_candidate_provider(file('candidate_config'))
    return AgenticRAGVLMProvider(knowledge_path=file('knowledge_store'),registry=load_semantic_registry(file('registry')),
        platform_config=load_platform_config(file('platform_config')),profile=profile,task=task,destinations=destinations,
        endpoint=data.get('vlm_endpoint','http://127.0.0.1:18074/v1/chat/completions'),model=data.get('vlm_model'),
        clip_endpoint=data.get('clip_endpoint','http://127.0.0.1:18072'),output=base/data.get('output','../output/platform'),
        candidate_provider=candidates)
