"""Regression failures and independent physical counterexamples for r5."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_final_trace_flush_does_not_advance_or_change_physics(tmp_path):
    pytest.importorskip('mujoco');os.environ.setdefault('MUJOCO_GL','egl')
    from agentic_rag_vlm.simulation.yam import YamPhysics
    sim=YamPhysics(tmp_path)
    sim.step(1)
    before=sim.data.qpos.copy();time=float(sim.data.time)
    assert sim.private_trace[-1]['time']<time
    sim.save({'scope':'log_flush_test'})
    raw=json.loads((tmp_path/'private_evaluator.jsonl').read_text().splitlines()[-1])
    assert raw['time']==time and raw['final_snapshot']
    assert np.array_equal(sim.data.qpos,before)
    assert float(sim.data.time)==time


def test_impact_guard_stops_before_continuing_invalid_contact(tmp_path):
    mujoco=pytest.importorskip('mujoco');os.environ.setdefault('MUJOCO_GL','egl')
    from agentic_rag_vlm.simulation.robust import RobustPhysics
    from agentic_rag_vlm.simulation.yam import PhysicsSafetyError
    sim=RobustPhysics(tmp_path,mode='single_shift')
    try:
        # Test-only falling initial condition; no policy can reset object poses.
        joint=sim.model.joint('target_free');q=int(joint.qposadr[0]);v=int(joint.dofadr[0])
        sim.data.qpos[q+2]=.08;sim.data.qvel[v+2]=-2.
        sim.data.qacc_warmstart[:]=0;mujoco.mj_forward(sim.model,sim.data)
        with pytest.raises(PhysicsSafetyError,match='target/table penetration'):
            sim.step(300)
        assert .003<sim.private_max_table_penetration<.02
        assert sim.data.body('target_cube').xpos[2]>-.02
    finally:sim.renderer.close()


def test_bound_aware_refinement_keeps_original_tolerances(tmp_path):
    mujoco=pytest.importorskip('mujoco');os.environ.setdefault('MUJOCO_GL','egl')
    from agentic_rag_vlm.simulation.robust import RobustPhysics
    from agentic_rag_vlm.simulation.yam import _rotation_log
    sim=RobustPhysics(tmp_path)
    try:
        before=sim.data.qpos.copy()
        # Measured r4 seed-101 failure; the regularized DLS stalled at 0.7 mm.
        initial=np.array([-.71574368,1.69689786,1.52851946,-1.40241753,0.,-.71574379])
        target=np.array([.34424992,-.17171753,.15])
        q=sim.solve(target,seed=initial)
        check=mujoco.MjData(sim.model);check.qpos[sim.qadr]=q;mujoco.mj_forward(sim.model,check)
        site=check.site(sim.site)
        assert np.linalg.norm(target-site.xpos)<.0005
        assert np.linalg.norm(_rotation_log(np.diag([-1.,1.,-1.])@site.xmat.reshape(3,3).T))<.015
        limits=sim.model.jnt_range[sim.joints]
        assert np.all(q>=limits[:,0]+.001) and np.all(q<=limits[:,1]-.001)
        assert np.array_equal(before,sim.data.qpos)  # planning cannot teleport the arm
        assert any(e['event']=='ik_bound_refinement' for e in sim.public_trace)
        with pytest.raises(RuntimeError,match='IK failed'):
            sim.solve([2.,2.,2.])
    finally:sim.renderer.close()


@pytest.mark.parametrize('force,fragility',[(.8,.9),(1.5,.1),(float('nan'),.1)])
def test_feedback_force_policy_refuses_unvalidated_increases(force,fragility):
    from agentic_rag_vlm.simulation.robust import feedback_force_recovery
    from agentic_rag_vlm.recovery import FailureType
    strategy={'force_n':force,'_semantic_signature':['cube','graspable_body','rigid',fragility,'body']}
    result=feedback_force_recovery(strategy,FailureType.SLIP,1)
    assert result.safe_stop


def test_visible_support_gate_cannot_release_airborne_object(tmp_path):
    pytest.importorskip('mujoco');os.environ.setdefault('MUJOCO_GL','egl')
    from agentic_rag_vlm.simulation.robust import RobustPhysics,RobustRetentionTransport
    from agentic_rag_vlm.platform.commands import PlatformCommand
    sim=RobustPhysics(tmp_path,mode='single_shift')
    try:
        sim.move([.2,-.2,.15],grip=.0375,stage='initial_view_clearance')
        initial=np.asarray(sim.detect()['cube'])
        transport=RobustRetentionTransport(sim,destinations_by_target=sim.setup['destinations'])
        receipt=transport.send_command(PlatformCommand('held','pick','cube','test',
            {'approach':'top','position_xyz_m':initial,'force_n':.96,'aperture_m':.075}))
        assert receipt['accepted']
        baseline=sim.tactile()['downward_load_n']
        assert not transport._guarded_unload('cube',initial,baseline)
        assert sim.data.ctrl[6]==0 and sim.tactile()['bilateral']
    finally:sim.renderer.close()


def _diagnostic(tmp_path,*extra):
    pytest.importorskip('mujoco')
    command=[sys.executable,str(ROOT/'scripts/run_extended_sim.py'),'--diagnostic',
             '--controller-profile','robust','--output',str(tmp_path/'run'),*extra]
    process=subprocess.run(command,env={**os.environ,'MUJOCO_GL':'egl'},capture_output=True,text=True,timeout=150)
    assert process.returncode==0,process.stdout+process.stderr
    folder=tmp_path/'run'
    return json.loads((folder/'result.json').read_text()),[json.loads(s) for s in (folder/'public_trace.jsonl').read_text().splitlines()]


def test_partial_load_placement_regression_needs_visual_and_unload_evidence(tmp_path):
    result,events=_diagnostic(tmp_path,'--seed','103','--mode','stacked_sort','--disable-memory')
    assert result['physical_success'] and len(result['attempts'])==6
    # Solver/rendering builds can reach the direct, stronger load-transfer
    # gate before needing an unload probe. Verify the actual release evidence
    # for each target, rather than requiring a particular recovery branch.
    loads=[e for e in events if e['event']=='placement_load']
    targets={e['target'] for e in loads}
    assert len(targets)==3
    for target in targets:
        final=next(e for e in reversed(loads) if e['target']==target)
        if all(load<final['reference_load_n']*.5 for load in final['observed_loads_n'][-3:]):
            continue
        probes=[e for e in events if e['event']=='placement_unload_probe' and e['target']==target]
        assert probes and all(e['visible_support_stable'] for e in probes)
        assert abs(probes[-1]['normal_load_after_n'])<.15*final['reference_load_n']
        assert any(e['event']=='placement_visual_support' and e['target']==target
                   and e['matched_surface_xyz_m'] is not None for e in events)
    assert result['max_target_table_penetration_m']<.003


@pytest.mark.parametrize('stable,remaining_load,expected',[
    (True,.1,True), (False,.1,False), (True,.4,False)])
def test_partial_unload_requires_both_stable_visual_support_and_load_transfer(stable,remaining_load,expected):
    """Deterministic branch counterexamples; synthetic feedback, not physics evidence."""
    from types import SimpleNamespace
    from agentic_rag_vlm.simulation.robust import RobustRetentionTransport
    sim=SimpleNamespace(data=SimpleNamespace(ctrl=np.zeros(7),time=0.),public_trace=[],stage='test')
    def step(count):sim.data.time+=count*.002
    sim.step=step
    sim.tactile=lambda:{'left_finger_q_m':.01,'downward_load_n':remaining_load}
    transport=object.__new__(RobustRetentionTransport);transport.sim=sim
    visible_calls=[]
    def visible(target,destination):
        visible_calls.append(target)
        if not stable and len(visible_calls)>2:return np.array([.3,.1,.036])
        return np.array([.2,.1,.036])
    transport._visible_support=visible
    assert transport._guarded_unload('cube',np.array([.2,.1,.018]),1.) is expected
    assert sim.data.ctrl[6]<.0375  # A probe never performs the full release.
    if not expected:
        assert sim.public_trace
        assert any(not p['visible_support_stable'] for p in sim.public_trace) or remaining_load>=.15


def test_heterogeneous_contact_and_shape_reuse_invalidates_failed_memory(tmp_path):
    result,events=_diagnostic(tmp_path,'--seed','20','--mode','stacked_sort',
        '--scene-variation','mixed_contact','--mixed-shapes','--recovery-policy','force_ladder')
    assert result['physical_success'] and result['mixed_shapes']
    assert max(a['motor_force_cap_n'] for a in result['attempts'])>.96
    failures=[e['payload']['evidence'] for e in result['trace'] if e['event']=='contact_failure_recovery']
    assert any(e['memory_invalidated'] is not None for e in failures)
    assert all(e['release_confirmed'] and e['scene_stable'] for e in failures)
    assert any(e['event']=='placement_load' and e['tcp_z_m']<.024 for e in events)
    assert all(result['checks'].values())
