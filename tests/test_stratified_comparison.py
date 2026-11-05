"""Physical-factor isolation, strong geometric baseline and KB integrity."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from agentic_rag_vlm.experience import ExperienceStore,validate_command_experience
from agentic_rag_vlm.retrieval import FixedSkillRetriever
from agentic_rag_vlm.runtime import AgenticRuntime,RuntimeConfig,RuntimePhase,Subgoal,TaskSpec
from agentic_rag_vlm.pipeline import AgenticPipeline
from scripts.agentic_framework import PublicObject

ROOT=Path(__file__).resolve().parents[1]


def test_occlusion_factor_changes_only_upper_footprint(tmp_path):
    pytest.importorskip('mujoco')
    from agentic_rag_vlm.simulation.robust import varied_scene
    paths=[tmp_path/'wide.xml',tmp_path/'narrow.xml']
    for path,width in zip(paths,[.012,.017]):
        varied_scene(path,seed=891,mode='stacked_sort',variation='mixed_contact',mixed_shapes=True,upper_half_width=width)
    a,b=[ET.parse(p).getroot() for p in paths]
    first=a.find('.//geom[@name="target_geom2"]');second=b.find('.//geom[@name="target_geom2"]')
    assert first.get('mass')==second.get('mass') and first.get('friction')==second.get('friction')
    assert first.get('size')=='.012 .012 .014' or first.get('size')=='0.012 0.012 .014'
    second.set('size',first.get('size'))
    assert ET.tostring(a)==ET.tostring(b)


def test_geometric_baseline_can_clear_visible_upper_with_missing_lower():
    pipeline=AgenticPipeline([])
    runtime=AgenticRuntime(pipeline,RuntimeConfig(planning_policy='geometric',replanning_enabled=False))
    task=TaskSpec('sort','independent sorting',(Subgoal('base','b'),Subgoal('upper','u')),('vase',),True)
    runtime.start(task)
    upper=PublicObject('upper',(.2,0,.05),'cube','rigid_body','graspable_body','rigid',.1,'body',1.)
    vase=replace(upper,name='vase',center_xyz_m=(.5,0,.08),fragility=1.)
    runtime.observe({'upper':upper,'vase':vase})
    assert runtime.phase==RuntimePhase.PLAN and runtime.current_subgoal.target=='upper'
    assert runtime.pending_targets==['upper','base'] and runtime.completed_targets==[]
    assert runtime.replan_count==0
    assert runtime.trace()[-1]['event']=='geometric_schedule'


def test_geometric_baseline_cannot_reorder_explicitly_ordered_task():
    runtime=AgenticRuntime(AgenticPipeline([]),RuntimeConfig(planning_policy='geometric'))
    runtime.start(TaskSpec('ordered','ordered work',(Subgoal('base','b'),Subgoal('upper','u'))))
    upper=PublicObject('upper',(.2,0,.05),'cube','rigid_body','graspable_body','rigid',.1,'body',1.)
    runtime.observe({'upper':upper})
    assert runtime.phase==RuntimePhase.SAFE_STOP


def test_geometric_baseline_refreshes_moved_pending_target_without_agent_l3():
    runtime=AgenticRuntime(AgenticPipeline([]),RuntimeConfig(planning_policy='geometric',replanning_enabled=False))
    runtime.start(TaskSpec('sort','sort',(Subgoal('base','b'),),reorderable=True))
    runtime.phase=RuntimePhase.MONITOR
    assert runtime.monitor({'stale_targets':['base']}) is False
    assert runtime.phase==RuntimePhase.OBSERVE and runtime.replan_count==0


def test_fixed_skill_baseline_uses_nominal_force_not_generic_15n_fallback():
    store=ExperienceStore.load(ROOT/'knowledge_base/yam_retention_observed.json')
    pipeline=AgenticPipeline([],retriever=FixedSkillRetriever(store))
    obj=PublicObject('target',(.2,0,.018),'cube','rigid_body','graspable_body','rigid',.1,'body',1.)
    action,evidence=pipeline.plan(obj,{'target':obj})
    assert action.source=='fixed_skill' and action.strategy['force_n']==.8
    assert evidence['retrieval']['retrieval_enabled'] is False
    assert evidence['retrieval']['ranked_cards']==()


@pytest.fixture
def imported_record():
    store=ExperienceStore.load(ROOT/'knowledge_base/yam_verified_diverse_r8.json')
    record=next(r for r in store.records if r['category']=='cylinder' and 'source_command_id' in r['provenance'])
    result=json.loads((ROOT/'knowledge_base'/record['provenance']['source_result_path']).read_text())
    return record,result


def test_observed_store_keeps_actual_final_force_and_portable_sources(imported_record):
    record,result=imported_record
    validate_command_experience(record,result)
    assert record['strategy']['force_n']>.8
    assert not any('position' in key for key in record['strategy'])
    store=ExperienceStore.load(ROOT/'knowledge_base/yam_verified_diverse_r8.json')
    store.validate_sources(ROOT/'knowledge_base')
    assert len(store.records)==14 and sum(r['outcome']==0 for r in store.records)==2


@pytest.mark.parametrize('fault',['initial_force','wrong_embedding','wrong_crop','wrong_observation'])
def test_import_validation_rejects_misbound_execution_evidence(imported_record,fault):
    record,result=imported_record;record=deepcopy(record)
    if fault=='initial_force':record['strategy']['force_n']=.8
    elif fault=='wrong_embedding':record['visual_embedding'][0]+=.1
    elif fault=='wrong_crop':record['provenance']['image_sha256']='f'*64
    else:record['provenance']['source_observation_id']='another_observation'
    with pytest.raises(ValueError):validate_command_experience(record,result)


def test_successful_episode_does_not_make_its_rejected_attempt_successful():
    store=ExperienceStore.load(ROOT/'knowledge_base/yam_verified_diverse_r8.json')
    rejected=next(r for r in store.records if r['outcome']==0)
    result=json.loads((ROOT/'knowledge_base'/rejected['provenance']['source_result_path']).read_text())
    assert result['physical_success'] is True
    forged=deepcopy(rejected);forged['outcome']=1
    with pytest.raises(ValueError,match='positive experience'):
        validate_command_experience(forged,result)


def test_motor_audit_does_not_confuse_transient_contact_force_with_motor_output():
    from scripts.audit_extended_benchmark import verify_motor_force
    # Saved source rejection had 1.646 N contact normal sum under a 0.96 N
    # actuator cap. The actual motor channel must determine the limit check.
    sample={'time':1.,'bilateral':True,'normal_force_n':{'left':.845,'right':.801},'actuator_force_n':-.96}
    assert verify_motor_force(sample,.96,require_telemetry=True)==(True,None)
    sample['actuator_force_n']=-1.1
    assert verify_motor_force(sample,.96,require_telemetry=True)[0] is False
    sample.pop('actuator_force_n')
    assert verify_motor_force(sample,.96,require_telemetry=True)[0] is False
    valid,warning=verify_motor_force(sample,.96)
    assert valid and warning['actual_motor_output_verified'] is False
