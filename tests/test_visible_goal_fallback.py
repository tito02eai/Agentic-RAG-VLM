"""Bounded progress cannot invent the identity/relationship of unseen goals."""
from dataclasses import replace
import pytest
from tests.test_visibility_replanning import runtime_scene
from agentic_rag_vlm.runtime import RuntimePhase


def scene():
    runtime,objects=runtime_scene()
    runtime.config=replace(runtime.config,visible_goal_fallback=True)
    runtime.pipeline.scene_context={
        'visible_geometry':[{'target':'visible_upper','center_xyz_m':[.2,.1,.064],
            'known_half_height_m':.014,'observation_id':'obs1',
            'source':'public_rgbd_and_vlm_identified_target'}],
        'support_plane':{'z_m':0.,'source':'calibrated_testbed_table'},
        'visible_goal_stability':{'previous_observation_id':'obs0','current_observation_id':'obs1',
            'previous_capture_time':1.,'current_capture_time':2.,
            'position_tolerance_m':.004,'stable_visible_targets':['protected','visible_upper']}}
    return runtime,objects


def test_stable_elevated_goal_progress_preserves_unknown_goal():
    runtime,objects=scene()
    runtime.observe(objects,receipt={'observation_id':'obs1'})
    assert runtime.phase==RuntimePhase.PLAN
    assert runtime.pending_targets==['visible_upper','hidden_base']
    assert 'hidden_base' not in objects
    event=runtime.trace()[-1]['payload']
    assert event['policy']=='visible_target_progress_then_reobserve'
    assert event['support_identity']=='not_inferred'
    runtime.completed_targets.append('visible_upper');runtime.phase=RuntimePhase.OBSERVE
    runtime.observe(objects,receipt={'observation_id':'obs2'})
    assert runtime.phase==RuntimePhase.SAFE_STOP and runtime.pending_targets==['hidden_base']


@pytest.mark.parametrize('problem',['disabled','order','budget','missing_protected','unstable_target',
    'unstable_protected','same_id','stale_id','same_time','nan_time','wrong_tolerance','no_pair',
    'no_plane','on_table','wrong_geometry','fragile','blocked'])
def test_progress_rejects_unsafe_or_unverified_context(problem):
    runtime,objects=scene();context=runtime.pipeline.scene_context
    pair=context['visible_goal_stability'];geom=context['visible_geometry'][0]
    if problem=='disabled':runtime.config=replace(runtime.config,visible_goal_fallback=False)
    elif problem=='order':runtime.task=replace(runtime.task,reorderable=False)
    elif problem=='budget':runtime.config=replace(runtime.config,maximum_replans=0)
    elif problem=='missing_protected':objects.pop('protected')
    elif problem=='unstable_target':pair['stable_visible_targets'].remove('visible_upper')
    elif problem=='unstable_protected':pair['stable_visible_targets'].remove('protected')
    elif problem=='same_id':pair['previous_observation_id']='obs1'
    elif problem=='stale_id':geom['observation_id']='old'
    elif problem=='same_time':pair['current_capture_time']=1.
    elif problem=='nan_time':pair['current_capture_time']=float('nan')
    elif problem=='wrong_tolerance':pair['position_tolerance_m']=.1
    elif problem=='no_pair':context.pop('visible_goal_stability')
    elif problem=='no_plane':context.pop('support_plane')
    elif problem=='on_table':
        objects['visible_upper']=replace(objects['visible_upper'],center_xyz_m=(.2,.1,.014))
        geom['center_xyz_m']=[.2,.1,.014]
    elif problem=='wrong_geometry':geom['center_xyz_m'][0]+=.01
    elif problem=='fragile':objects['visible_upper']=replace(objects['visible_upper'],fragility=.8)
    elif problem=='blocked':
        objects['on_upper']=replace(objects['visible_upper'],name='on_upper',center_xyz_m=(.2,.1,.1))
        context['relations']=[{'source':'visible_upper','target':'on_upper','relation':'supports'}]
    runtime.observe(objects,receipt={'observation_id':'obs1'})
    assert runtime.phase==RuntimePhase.SAFE_STOP
    assert runtime.completed_targets==[]


@pytest.mark.parametrize('decision',['execute','stop'])
def test_progress_tool_retains_real_reasoner_veto_contract(decision):
    from agentic_rag_vlm.reasoning import VLMReasoner
    from agentic_rag_vlm.pipeline import GraspAction
    from agentic_rag_vlm.qwen_adapter import VLMResult
    import json
    class Client:
        def infer(self,images,prompt):
            self.prompt=prompt;value={'decision':decision,'summary':'bounded progress review'}
            return VLMResult(value,json.dumps(value),1,'test','test',{})
    client=Client();reasoner=VLMReasoner(client)
    evidence={'visibility_replanning':{'policy':'visible_target_progress_then_reobserve',
        'selected_target':'visible_upper','support_identity':'not_inferred','deferred_targets':['hidden_base']}}
    action=GraspAction('visible_upper',(.2,.1,.064),{'force_n':.8},{},'haa_rag')
    if decision=='stop':
        with pytest.raises(ValueError,match='did not approve'):reasoner.think(action,evidence,[],0)
    else:assert reasoner.think(action,evidence,[],0)['parsed']['decision']=='execute'
    assert 'No identity or support/occlusion relationship is inferred' in client.prompt


@pytest.mark.parametrize('change',['stable','drift','stale','missing_protected'])
def test_session_captures_second_observation_before_progress(change):
    from agentic_rag_vlm.platform.session import BluPeSession
    from agentic_rag_vlm.platform.commands import GraspActionMapper
    from agentic_rag_vlm.platform.verifier import SceneChangeVerifier
    from agentic_rag_vlm.platform.transport import PlatformObservation
    runtime,initial=scene();seen=[];runtime.phase=RuntimePhase.IDLE
    session=BluPeSession(runtime,None,None,GraspActionMapper(),SceneChangeVerifier(),
        destinations={'base_zone':(.1,.2,.014),'upper_zone':(.2,.2,.014)})
    def perceive():
        i=len(seen);objects=dict(initial)
        if i and change=='drift':objects['visible_upper']=replace(initial['visible_upper'],center_xyz_m=(.21,.1,.064))
        if i and change=='missing_protected':objects.pop('protected')
        session._last_observation=PlatformObservation('obs'+str(i),(),receipt={'captured_at':1. if change=='stale' else 1.+i})
        runtime.pipeline.scene_context={'visible_geometry':[{'target':n,'center_xyz_m':list(o.center_xyz_m),
            'known_half_height_m':.014,'observation_id':'obs'+str(i),
            'source':'public_rgbd_and_vlm_identified_target'} for n,o in objects.items()],
            'support_plane':{'z_m':0.,'source':'calibrated_testbed_table'}}
        seen.append(objects);return objects
    session._perceive=perceive
    def plan(*a,**kw):runtime.safe_stop('test_stop_before_motion')
    runtime.plan=plan
    result=session.run(runtime.task)
    assert len(seen)==2
    pairs=[e for e in result['receipts'] if e['kind']=='visibility_stability_check']
    assert len(pairs)==1
    events=[e for e in result['trace'] if e['event']=='visibility_replan']
    assert bool(events)==(change=='stable')
    assert not any(e['kind']=='command' for e in result['receipts'])
