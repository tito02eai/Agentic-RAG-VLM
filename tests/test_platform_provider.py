"""Paper-agent platform contracts with explicit model/executor test doubles."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image

from agentic_rag_vlm.qwen_adapter import VLMResult
from agentic_rag_vlm.platform.runner_provider import AgenticRAGVLMProvider,RunnerBoundClient
from agentic_rag_vlm.platform.blupe_client import SessionHandle,BluPeAPIError
from agentic_rag_vlm.platform.config import PlatformConfig
from agentic_rag_vlm.platform.transport import PlatformObservation
from agentic_rag_vlm.experience import ExperienceStore
from agentic_rag_vlm.runtime import TaskSpec,Subgoal
from scripts.agentic_framework import load_semantic_registry
from tests.test_delivery import calibration

ROOT=Path(__file__).resolve().parents[1]
STORE=ROOT/'knowledge_base/yam_verified_diverse_r8.json'


def fixture(tmp_path):
    registry=load_semantic_registry(ROOT/'knowledge_base/blupe_semantic_registry.json')
    registry['cube2']=dict(registry['cube'])
    geometry={'arm':'right','coordinate_frame':'arm_base','support_plane':{'normal_base':[0,0,1],'offset_m':0.},
        'feature_heights_m':{'cube':.1,'cube2':.1,'vase':.1}}
    class Transport:
        def __init__(self):
            self.positions={'cube':[.1,0,.1],'cube2':[.2,0,.1],'vase':[.45,.1,.1]}
            self.commands=[];self.number=0
        def get_observation(self):
            self.number+=1;path=tmp_path/f'frame_{self.number}.png'
            Image.new('RGB',(640,360),(220,200,180)).save(path)
            return PlatformObservation(f'frame_{self.number}',(),(str(path),),{'calibration_id':'test-cal',
                'camera_role':'top','captured_at':float(self.number),'release_state':{'gripper_open':True,'retracted':True}})
        def send_command(self,command):
            self.commands.append(command)
            if command.command_type=='place':self.positions[command.target]=command.parameters['destination_xyz_m']
            return {'accepted':True,'execution_started':True}
    transport=Transport()
    class Client:
        def __init__(self):self.calls=[]
        def infer(self,images,prompt):
            self.calls.append(prompt)
            if prompt.startswith('Detect visible'):
                objects=[]
                for name,(x,y,z) in transport.positions.items():
                    objects.append({'name':name,'pixel_xy':[320+500*x/(1-z),180-500*y/(1-z)]})
                value={'objects':objects}
            else:value={'decision':'execute','summary':'test-double review'}
            return VLMResult(value,json.dumps(value),1.,'test-double','test-double',{})
    class Encoder:
        def __init__(self):self.store=ExperienceStore.load(STORE);self.embedding_space=self.store.embedding_space
        def encode(self,paths):return np.array([self.store.records[0]['visual_embedding'] for p in paths])
    client=Client();provider=AgenticRAGVLMProvider(knowledge_path=STORE,registry=registry,
        platform_config=PlatformConfig('blupe','unused','http://unused','yam-1'),client=client,encoder=Encoder(),
        destinations={'first':[-.2,.1,.1],'second':[-.25,-.1,.1]},output=tmp_path/'runs')
    task=TaskSpec('contract','sort cubes',(Subgoal('cube','first'),Subgoal('cube2','second')),('vase',))
    return provider,transport,geometry,task,client


def test_complete_paper_chain_through_platform_transport(tmp_path):
    provider,transport,geometry,task,client=fixture(tmp_path)
    result=provider.run_with_transport(task,transport,calibration(),geometry)
    assert result['outcome']=='complete'
    plans=[e['payload'] for e in result['trace'] if e['event']=='subgoal_planned']
    assert [p['action']['source'] for p in plans]==['haa_rag','episodic_memory']
    assert all(p['vlm_thought']['parsed']['decision']=='execute' for p in plans)
    assert all(p['evidence']['retrieval']['ranked_cards'] for p in plans)
    assert [c.command_type for c in transport.commands]==['pick','place','pick','place']
    assert all(p['vlm_thought']['images'] for p in plans)
    assert plans[0]['action']['strategy']['_execution_domain'].startswith('blupe:')
    binding=plans[0]['action']['strategy']['_execution_binding']
    assert binding['knowledge_domain']=='mjlab_yam_retention_v1'
    assert binding['force_control_calibrated'] is False
    assert not any('static_cards' in p['action']['source'] for p in plans)


def test_preview_uses_full_chain_without_dispatch(tmp_path):
    provider,transport,geometry,task,_=fixture(tmp_path)
    result=provider.preview(task,transport.get_observation(),calibration(),geometry)
    assert result['execution_performed'] is False and not transport.commands
    assert result['action']['source']=='haa_rag'
    assert result['perception']['vlm_result']['visual_embeddings']['cube']


def test_registration_and_public_config_do_not_require_robot(tmp_path):
    provider,_,_,_,_=fixture(tmp_path)
    assert provider.prepare()['prepared'] is True
    assert provider.public_config()['site_geometry_configured'] is False


def test_runner_never_owns_queue_and_stop():
    client=RunnerBoundClient('http://unused',None,{'session_id':'s','robot_id':'yam-1'})
    with pytest.raises(BluPeAPIError,match='RunnerController'):client.create_session()
    with pytest.raises(BluPeAPIError,match='RunnerController'):client.stop_session()


def test_runner_lease_binding_and_feedback_use_existing_sdk():
    calls=[]
    class API:
        supports_trajectories=True
        def get_session(self,sid):return {'episode_id':'episode','lease_id':'fresh_lease','status':'running'}
        def submit_trajectory(self,*args):calls.append(args);return {'accepted':True}
        def get_episode_trace(self,sid):return {'results':[{'trajectory_id':'trajectory','status':'completed'}]}
    client=RunnerBoundClient('http://unused',API(),{'session_id':'sid','episode_id':'episode','lease_id':'old','robot_id':'yam-1'})
    handle=SessionHandle('sid','','yam-1','running','',episode_id='episode',lease_id='old')
    client.submit_trajectory_packet(handle,{'trajectory_id':'trajectory','cadence_hz':10.,'waypoints':[{'step_id':1}]})
    assert calls[0][2]=='fresh_lease'
    assert client.wait_for_trajectory(handle,'trajectory',timeout_s=1)['status']=='completed'
    with pytest.raises(BluPeAPIError,match='identity'):client.get_session(replace(handle,session_id='other'))
    client.cancelled=lambda:True
    with pytest.raises(BluPeAPIError,match='cancelled'):client.get_session(handle)


def test_vlm_veto_stops_before_platform_dispatch(tmp_path):
    provider,transport,geometry,task,client=fixture(tmp_path)
    original=client.infer
    def infer(images,prompt):
        if not prompt.startswith('Detect visible'):
            return VLMResult({'decision':'stop'},'{"decision":"stop"}',1.,'test-double','test-double',{})
        return original(images,prompt)
    client.infer=infer
    result=provider.run_with_transport(task,transport,calibration(),geometry)
    assert result['outcome']=='safe_stop' and transport.commands==[]


def test_encoder_identity_mismatch_rejected_at_prepare(tmp_path):
    provider,*_=fixture(tmp_path);provider.encoder.embedding_space='openai/clip-vit-large-patch14@wrong'
    with pytest.raises(ValueError,match='mismatch'):provider.prepare()


def test_official_run_session_hook_adopts_one_session_and_closes_loop(tmp_path,monkeypatch):
    import agentic_rag_vlm.platform.runner_provider as module
    from tests.test_delivery import measured_test_profile,native_fixture
    provider,transport,geometry,task,_=fixture(tmp_path)
    profile=measured_test_profile();profile['feature_heights_m']=geometry['feature_heights_m']
    profile['grasp_offsets_base_m']={n:[0,0,0] for n in profile['feature_heights_m']}
    provider.profile=profile;provider.task=task
    cal=calibration();cal['kinematics']=native_fixture()['kinematics']
    monkeypatch.setattr(module.RunnerBoundClient,'get_calibration',lambda *a:cal)
    monkeypatch.setattr(module,'RemotePlatformTransport',lambda *a,**kw:transport)
    class API:
        def create_session(self,*a,**kw):pytest.fail('duplicate queue admission')
        def stop_session(self,*a,**kw):pytest.fail('provider must not own Stop')
    context={'session_id':'existing','episode_id':'episode','lease_id':'lease','robot_id':'yam-1'}
    result=provider.run_session(task.instruction,{},API(),context,lambda:False)
    saved=json.loads(Path(result['result_path']).read_text())
    assert result['state']=='completed' and saved['framework_profile']=='paper_haa_clip_vlm_review'
    assert saved['hardware_success'] is None and saved['queue_owner']=='RunnerController'
    assert len(transport.commands)==4 and not provider._active


def test_cancelled_runner_run_never_dispatches(tmp_path):
    provider,transport,geometry,task,_=fixture(tmp_path)
    provider.profile=geometry;provider.task=task
    result=provider.run_session(task.instruction,{},None,
        {'session_id':'existing','episode_id':'episode','lease_id':'lease','robot_id':'yam-1'},lambda:True)
    assert result['state']=='cancelled' and transport.commands==[] and not provider._active
