"""Full paper-profile composition shared by the CLI and BluPe runner provider.

Constructing a policy/preview never creates a BluPe session or commands a robot.
"""
from pathlib import Path
from dataclasses import replace
from ..paper import PaperAgent
from ..runtime import AgenticRuntime,RuntimeConfig
from .geometry import PlaneProjector
from .perception import GroundedVLMPerception


class _ExecutionStoreView:
    def __init__(self,store,domain):self._store=store;self.domain=domain
    def __getattr__(self,name):return getattr(self._store,name)


class ExecutorBoundRetriever:
    """Retain knowledge provenance while isolating actual executor memory.

    Retrieved force/synergy are semantic recommendations, not commands supported
    by the current BluPe joint/gripper API. Their non-application is explicit.
    """
    action_source='haa_rag'
    def __init__(self,retriever,domain):
        self.retriever=retriever;self.store=_ExecutionStoreView(retriever.store,domain)
        self.binding={'knowledge_domain':retriever.store.domain,'execution_domain':domain,
            'knowledge_role':'retrieved_prior','unsupported_control_fields':['force_n','synergy'],
            'force_control_calibrated':False}
    def retrieve(self,*args,**kwargs):
        result=self.retriever.retrieve(*args,**kwargs)
        return replace(result,strategy={**result.strategy,'_execution_binding':self.binding},
            stages={**result.stages,'execution_binding':self.binding})


def paper_components(knowledge_path,client,encoder,registry,calibration,profile,*,maximum_replans=3,candidate_provider=None):
    agent=PaperAgent(Path(knowledge_path),client)
    if encoder.embedding_space!=agent.retriever.store.embedding_space:
        raise ValueError('platform encoder differs from the knowledge embedding space')
    if profile.get('coordinate_frame')!='arm_base':raise ValueError('arm_base geometry required')
    domain=f"blupe:{profile.get('robot_id','preview')}:{profile['arm']}:{calibration['calibration_id']}"
    agent.retriever=ExecutorBoundRetriever(agent.retriever,domain)
    plane=profile['support_plane']
    from .camera_contract import projection_calibration
    projector=PlaneProjector(projection_calibration(calibration,profile),profile['arm'],plane_normal_base=plane['normal_base'],plane_offset_m=plane['offset_m'])
    perception=GroundedVLMPerception(client,registry,projector,profile['feature_heights_m'],visual_encoder=encoder,candidate_provider=candidate_provider)
    runtime=AgenticRuntime(agent,RuntimeConfig(maximum_replans=maximum_replans))
    return runtime,perception


def bind_observation(runtime,perception,observation):
    """Same image/embedding/context binding as a full session, for plan preview."""
    objects,receipt=perception.perceive(observation);evidence=receipt.vlm_result or {}
    runtime.pipeline.bind_observation(visual_embeddings=evidence.get('visual_embeddings',{}),
        embedding_space=evidence.get('embedding_space'),scene_context=evidence.get('scene_context',{}),
        images=[Path(p) for p in observation.image_paths])
    runtime.pipeline.reasoner.target_images={g['name']:[Path(g['crop_path'])]
        for g in evidence.get('grounding',[]) if g.get('crop_path')}
    return objects,receipt
