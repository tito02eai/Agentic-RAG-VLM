"""Contact-dynamics counterexamples for recovery and within-task memory."""
import json,os,subprocess,sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('force,friction,held',[(.8,1.,True),(.8,.7,False),(.96,.7,True)])
def test_measured_force_friction_changes_retention(tmp_path,force,friction,held):
    pytest.importorskip('mujoco');os.environ.setdefault('MUJOCO_GL','egl')
    from agentic_rag_vlm.simulation.extended import ExtendedPhysics,RetentionTransport
    from agentic_rag_vlm.platform.commands import PlatformCommand
    s=ExtendedPhysics(tmp_path/'world',mode='single_shift',friction=friction)
    try:
        s.move([.2,-.2,.15],grip=.0375,stage='initial_view_clearance')
        transport=RetentionTransport(s,destinations_by_target=s.setup['destinations'])
        receipt=transport.send_command(PlatformCommand('test','pick','cube','test',
            {'approach':'top','position_xyz_m':s.detect()['cube'],'force_n':force,'aperture_m':.075}))
        assert receipt['accepted'] is held
        if held:
            tactile=s.tactile()
            assert tactile['bilateral']
            assert sum(tactile['normal_force_n'].values())==pytest.approx(force,rel=.03)
            assert tactile['downward_load_n']==pytest.approx(.06*9.81,rel=.03)
            assert s.data.body('target_cube').xpos[2]>.12  # evaluator only
        else:
            assert receipt['failure_type']=='slip' and receipt['release_confirmed']
            assert all(not sample['bilateral'] for sample in receipt['retention']['retention_samples'])
    finally:s.renderer.close()


@pytest.mark.parametrize('disabled,attempts,hits',[(False,4,2),(True,6,0)])
def test_stack_removal_real_slip_recovery_and_memory(tmp_path,disabled,attempts,hits):
    pytest.importorskip('mujoco')
    command=[sys.executable,str(ROOT/'scripts/run_extended_sim.py'),'--diagnostic','--mode','stacked_sort',
             '--output',str(tmp_path/'run')]
    if disabled:command+=['--disable-memory']
    p=subprocess.run(command,env={**os.environ,'MUJOCO_GL':'egl'},capture_output=True,text=True,timeout=120)
    assert p.returncode==0,p.stdout+p.stderr
    result=json.loads((tmp_path/'run/result.json').read_text())
    assert result['physical_success'] and len(result['attempts'])==attempts and result['memory_hits']==hits
    assert result['trace'][-1]['payload']['completed_targets']==['cube2','cube','cube3']
    assert len([a for a in result['attempts'] if not a['retained']])==(3 if disabled else 1)
    assert result['max_target_table_penetration_m']<.003


def test_disabling_recovery_does_not_override_physical_failure(tmp_path):
    pytest.importorskip('mujoco')
    command=[sys.executable,str(ROOT/'scripts/run_extended_sim.py'),'--diagnostic','--mode','single_shift',
        '--disable-recovery','--output',str(tmp_path/'run')]
    run=subprocess.run(command,env={**os.environ,'MUJOCO_GL':'egl'},capture_output=True,text=True,timeout=90)
    assert run.returncode==1
    result=json.loads((tmp_path/'run/result.json').read_text())
    assert not result['physical_success'] and len(result['attempts'])==1
    assert not result['attempts'][0]['retained'] and not result['memory']


def test_contact_retry_cannot_bypass_release_confirmation():
    from agentic_rag_vlm.runtime import AgenticRuntime,TaskSpec,Subgoal
    from agentic_rag_vlm.pipeline import AgenticPipeline
    from scripts.agentic_framework import PublicObject
    cards=json.loads((ROOT/'knowledge_base/blupe_affordance_cards.json').read_text())['cards']
    runtime=AgenticRuntime(AgenticPipeline(cards));runtime.start(TaskSpec('t','move',(Subgoal('cube','z'),)))
    obj=PublicObject('cube',(.3,0,.018),'cube','rigid_body','graspable_body','rigid',.1,'body',1)
    runtime.observe({'cube':obj});action=runtime.plan({'cube':obj})
    with pytest.raises(RuntimeError):runtime.replace_after_safe_release(action,evidence={'release_confirmed':False,'scene_stable':True})


def test_new_low_force_knowledge_has_observed_sources():
    from agentic_rag_vlm.experience import ExperienceStore
    path=ROOT/'knowledge_base/yam_retention_observed.json';store=ExperienceStore.load(path)
    store.validate_sources(path.parent)
    assert len(store.records)==3 and store.domain=='mjlab_yam_retention_v1'
    assert all(e['strategy']['force_n']==.8 and e['outcome']==1 for e in store.records)
