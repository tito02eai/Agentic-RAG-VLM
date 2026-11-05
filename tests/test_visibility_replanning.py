"""Partial visibility must not fabricate identities, skip goals, or weaken safety."""
from dataclasses import replace

import pytest

from agentic_rag_vlm.pipeline import AgenticPipeline
from agentic_rag_vlm.runtime import AgenticRuntime, RuntimeConfig, RuntimePhase, Subgoal, TaskSpec
from scripts.agentic_framework import PublicObject


def runtime_scene(*, enabled=True, reorderable=True, budget=1):
    upper=PublicObject('visible_upper',(.2,.1,.064),'cube','rigid_body','graspable_body','rigid',.1,'body',1.)
    vase=replace(upper,name='protected',center_xyz_m=(.5,.1,.04),fragility=1.)
    pipe=AgenticPipeline([])
    pipe.scene_context={'unresolved_supports':[{
        'upper_target':'visible_upper','support_candidate_id':'opaque_candidate',
        'support_top_xyz_m':[.201,.1,.05],'upper_half_height_m':.014,
        'upper_bbox_xyxy':[10,10,25,25],'support_bbox_xyxy':[9,9,27,27],
        'observation_id':'obs1','source':'public_rgbd_surface_plus_known_upper_dimensions',
        'support_identity':'unresolved'}]}
    runtime=AgenticRuntime(pipe,RuntimeConfig(maximum_replans=budget,visibility_replanning=enabled))
    task=TaskSpec('sort','move both independently',
        (Subgoal('hidden_base','base_zone'),Subgoal('visible_upper','upper_zone')),('protected',),reorderable)
    runtime.start(task)
    return runtime,{'visible_upper':upper,'protected':vase}


def test_partial_visibility_reorders_visible_upper_without_inventing_base():
    runtime,objects=runtime_scene()
    runtime.observe(objects,receipt={'observation_id':'obs1'})
    assert runtime.phase==RuntimePhase.PLAN
    assert runtime.current_subgoal.target=='visible_upper'
    assert runtime.pending_targets==['visible_upper','hidden_base']
    assert runtime.completed_targets==[] and 'hidden_base' not in objects
    event=runtime.trace()[-1]
    assert event['event']=='visibility_replan'
    assert event['payload']['support_identity']=='unresolved'
    assert event['payload']['deferred_targets']==['hidden_base']
    assert runtime.replan_count==1


@pytest.mark.parametrize('enabled,reorderable,budget',[(False,True,1),(True,False,1),(True,True,0)])
def test_partial_visibility_requires_opt_in_order_permission_and_budget(enabled,reorderable,budget):
    runtime,objects=runtime_scene(enabled=enabled,reorderable=reorderable,budget=budget)
    runtime.observe(objects,receipt={'observation_id':'obs1'})
    assert runtime.phase==RuntimePhase.SAFE_STOP


@pytest.mark.parametrize('problem',['protected_missing','stale','unsupported_gap','no_overlap','named_support','blocked_upper'])
def test_partial_visibility_rejects_unverified_or_unsafe_removal(problem):
    runtime,objects=runtime_scene()
    witness=runtime.pipeline.scene_context['unresolved_supports'][0]
    if problem=='protected_missing':objects.pop('protected')
    elif problem=='stale':witness['observation_id']='old'
    elif problem=='unsupported_gap':witness['support_top_xyz_m'][2]=.01
    elif problem=='no_overlap':witness['support_bbox_xyxy']=[30,30,40,40]
    elif problem=='named_support':witness['support_identity']='hidden_base'
    elif problem=='blocked_upper':
        objects['on_upper']=replace(objects['visible_upper'],name='on_upper',center_xyz_m=(.2,.1,.09))
        runtime.pipeline.scene_context['relations']=[{'source':'visible_upper','target':'on_upper','relation':'supports'}]
    runtime.observe(objects,receipt={'observation_id':'obs1'})
    assert runtime.phase==RuntimePhase.SAFE_STOP and runtime.completed_targets==[]


def test_missing_goal_remains_pending_after_visible_target_is_completed():
    runtime,objects=runtime_scene()
    runtime.observe(objects,receipt={'observation_id':'obs1'})
    # Simulate the already verified upper subgoal, without claiming a physics
    # experiment. The unresolved base must still prevent task completion.
    runtime.completed_targets.append('visible_upper')
    runtime.phase=RuntimePhase.OBSERVE
    runtime.observe(objects,receipt={'observation_id':'obs2'})
    assert runtime.phase==RuntimePhase.SAFE_STOP
    assert runtime.pending_targets==['hidden_base']


def test_unidentified_or_fragile_upper_cannot_be_selected():
    runtime,objects=runtime_scene()
    objects['visible_upper']=replace(objects['visible_upper'],fragility=.9)
    runtime.observe(objects,receipt={'observation_id':'obs1'})
    assert runtime.phase==RuntimePhase.SAFE_STOP


def test_perception_records_anonymous_support_without_relabeling_it(tmp_path):
    import json
    from pathlib import Path
    from agentic_rag_vlm.simulation.extended import SupportRGBDPerception
    from agentic_rag_vlm.platform.transport import PlatformObservation
    from agentic_rag_vlm.qwen_adapter import VLMResult
    from scripts.agentic_framework import load_semantic_registry
    class Client:
        def infer(self,images,prompt):
            labels={'c0':'unknown','c1':'cube2','c2':'vase'}
            return VLMResult(labels,json.dumps(labels),1,'test','test',{})
    candidates=[]
    for key,xyz,box in [('c0',[.2,.1,.036],[9,9,27,27]),('c1',[.2,.1,.064],[10,10,25,25]),('c2',[.5,.1,.08],[40,40,50,50])]:
        path=tmp_path/(key+'.png');path.write_bytes(b'client does not open the fixture image')
        candidates.append({'candidate_id':key,'top_xyz_m':xyz,'bbox_xyxy':box,'crop_path':str(path)})
    root=Path(__file__).resolve().parents[1]
    registry=load_semantic_registry(root/'knowledge_base/blupe_semantic_registry.json')
    registry['cube2']=dict(registry['cube'])
    perception=SupportRGBDPerception(Client(),registry,{'cube':.018,'cube2':.014,'vase':.04})
    objects,receipt=perception.perceive(PlatformObservation('fresh',(),receipt={'rgbd_candidates':candidates}))
    assert set(objects)=={'cube2','vase'} and 'cube' not in objects
    witness=receipt.vlm_result['scene_context']['unresolved_supports'][0]
    assert witness['support_identity']=='unresolved' and witness['support_candidate_id']=='c0'
    assert witness['upper_target']=='cube2' and witness['observation_id']=='fresh'
    assert receipt.vlm_result['parsed']['c0']=='unknown'


@pytest.mark.parametrize('decision',['execute','stop'])
def test_vlm_reviews_public_visibility_tool_and_retains_veto(decision):
    import json
    from agentic_rag_vlm.reasoning import VLMReasoner
    from agentic_rag_vlm.pipeline import GraspAction
    from agentic_rag_vlm.qwen_adapter import VLMResult
    class Client:
        def infer(self,images,prompt):
            self.prompt=prompt
            value={'decision':decision,'summary':'review of supplied public tool evidence'}
            return VLMResult(value,json.dumps(value),1,'test','test',{})
    client=Client();reasoner=VLMReasoner(client)
    evidence={'visibility_replanning':{'selected_target':'visible_upper',
        'support_identity':'unresolved','deferred_targets':['hidden_base']}}
    action=GraspAction('visible_upper',(.2,.1,.05),{'force_n':.8},{},'haa_rag')
    if decision=='stop':
        with pytest.raises(ValueError,match='did not approve'):
            reasoner.think(action,evidence,[],0)
    else:
        assert reasoner.think(action,evidence,[],0)['parsed']['decision']=='execute'
    assert '"support_identity": "unresolved"' in client.prompt
    assert 'do not assign it an identity' in client.prompt
