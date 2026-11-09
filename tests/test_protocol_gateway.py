"""Production-path regressions discovered by the protocol-physics integration."""
from copy import deepcopy
from types import SimpleNamespace
import pytest

from agentic_rag_vlm.platform.blupe_client import BluPeSessionClient,SessionHandle
from agentic_rag_vlm.platform.deployment import validate_profile_fields,validate_execution_binding
from agentic_rag_vlm.platform.geometry import GeometryError
from tests.test_delivery import measured_test_profile
from agentic_rag_vlm.platform.transport import RemotePlatformTransport
from agentic_rag_vlm.platform.config import PlatformConfig


@pytest.mark.parametrize('arm', ['left', 'right'])
@pytest.mark.parametrize('final_gripper', [0., 1.])
def test_fractional_trajectory_roundtrip_keeps_exact_endpoints_and_idle_arm(arm, final_gripper):
    import json
    import os
    import numpy as np
    from agentic_rag_vlm.platform.local_runner import import_official_runner
    from agentic_rag_vlm.platform.motion import MotionPlan, MotionKeyframe
    from agentic_rag_vlm.platform.blupe_client import TrajectoryBuilder
    checkout = os.environ.get('BLUPE_RUNNER_CHECKOUT')
    if not checkout:
        pytest.skip('set BLUPE_RUNNER_CHECKOUT for official wire validation')
    _, session = import_official_runner(checkout)
    start = (13.1031001, -47.2876712, 68.7123931, -.007001, .003912, 1.123456)
    goal = (-7.003129, -35.998171, 74.187654, -.119111, .013001, 2.819101)
    idle = tuple(value/3 for value in start)
    plan = MotionPlan(arm, [MotionKeyframe('start', start, .481923),
                            MotionKeyframe('finish', goal, final_gripper)], idle, .61123)
    builder = TrajectoryBuilder(arm)
    for step, (joints, gripper) in enumerate(plan.interpolated_steps(max_joint_delta_deg_per_step=3.)):
        builder.add(step_id=step, joints_deg=joints, gripper=gripper,
                    other_joints_deg=idle, other_gripper=.61123)
    packet = json.loads(json.dumps(builder.packets(episode_id='e', lease_id='l', prefix='fractional')[0]))
    points = session._trajectory_waypoints(packet['waypoints'])
    other = 'right' if arm == 'left' else 'left'
    assert points[-1][arm+'_gripper'] == final_gripper
    assert points[-1][arm+'_joints_deg'] == list(goal)
    assert all(point[other+'_joints_deg'] == list(idle) and point[other+'_gripper'] == .61123 for point in points)
    assert np.max(np.abs(np.diff([point[arm+'_joints_deg'] for point in points], axis=0))) <= 3.
    invalid = [dict(points[-1], **{arm+'_gripper': -1e-12})]
    with pytest.raises(ValueError, match='between 0 and 1'):
        session._trajectory_waypoints(invalid)


@pytest.mark.parametrize('nested',[False,True])
def test_terminal_trace_result_cannot_be_hidden_by_accepted(nested):
    client=BluPeSessionClient('http://127.0.0.1:1')
    accepted={'trajectory_id':'packet','status':'accepted'};completed={'trajectory_id':'packet','status':'completed'}
    entries=[accepted,completed]
    if nested:entries=[{'type':'trajectory_result','payload':e} for e in entries]
    client.get_episode_trace=lambda h:{'events':entries}
    handle=SessionHandle('session','','robot','running','')
    assert client.find_trajectory_result(handle,'packet')['status']=='completed'


def test_accepted_collection_cannot_mask_terminal_collection():
    client=BluPeSessionClient('http://127.0.0.1:1')
    client.get_episode_trace=lambda h:{'trajectory_results':[{'trajectory_id':'x','status':'accepted'}],
        'events':[{'payload':{'trajectory_id':'x','status':'completed'}}]}
    assert client.find_trajectory_result(None,'x')['status']=='completed'


def simulated_profile():
    profile=measured_test_profile();profile['execution_environment']='simulation'
    profile['physical_validation']={'passed':False}
    profile['simulation_validation']={'passed':True,'record':'synthetic test only'}
    return profile


def test_simulator_geometry_does_not_claim_hardware_validation():
    profile=simulated_profile();validate_profile_fields(profile,robot_id='yam-1',arm='right')
    profile['physical_validation']={'passed':True,'record':'not allowed'}
    with pytest.raises(GeometryError,match='must not claim'):validate_profile_fields(profile,robot_id='yam-1',arm='right')


@pytest.mark.parametrize('endpoint,api',[
    ('https://public.example',SimpleNamespace(local_simulation_only=True,_base_url='https://public.example')),
    ('http://127.0.0.1:18792',SimpleNamespace(_base_url='http://127.0.0.1:18792')),
    ('http://127.0.0.1:18792',SimpleNamespace(local_simulation_only=True,_base_url='https://public.example')),
])
def test_simulation_profile_never_binds_public_or_mismatched_sdk(endpoint,api):
    with pytest.raises(GeometryError):validate_execution_binding(simulated_profile(),endpoint,session_api=api)


@pytest.mark.parametrize('arm',['left','right'])
def test_production_transport_reads_other_gripper_without_degree_suffix(arm):
    station={'left_joints_deg':[1.]*6,'right_joints_deg':[2.]*6,'left_gripper':.4,'right_gripper':.8}
    client=SimpleNamespace(get_station_observation=lambda robot:station)
    transport=RemotePlatformTransport(PlatformConfig('blupe','NONE','http://unused','yam-1'),
        client=client,arm=arm,deployment_profile=measured_test_profile())
    state=transport._current_arm_state()
    assert state[1:4:2]==((.4,.8) if arm=='left' else (.8,.4))


def test_rgb_proposals_merge_disconnected_faces_without_assigning_identity(tmp_path):
    import numpy as np
    from PIL import Image
    from agentic_rag_vlm.simulation.protocol_gateway import color_region_candidates
    from agentic_rag_vlm.platform.transport import PlatformObservation
    rgb=np.full((60,80,3),100,dtype=np.uint8)
    rgb[10:13,10:26]=[220,10,10];rgb[14:30,10:26]=[220,10,10]
    rgb[12:30,50:64]=[10,200,20];rgb[40:50,20:40]=[10,20,220]
    image=tmp_path/'frame.png';Image.fromarray(rgb).save(image)
    candidates=color_region_candidates(PlatformObservation('x',(),(str(image),),{}))
    assert len(candidates)==2
    assert all(set(c)=={'candidate_id','bbox_xyxy','pixel_xy'} for c in candidates)
    assert any(c['bbox_xyxy']==[10,10,26,30] for c in candidates)


def test_yellow_surface_stays_one_candidate_across_channel_dominance(tmp_path):
    import numpy as np
    from PIL import Image
    from agentic_rag_vlm.simulation.protocol_gateway import color_region_candidates
    from agentic_rag_vlm.platform.transport import PlatformObservation
    rgb=np.full((60,80,3),100,dtype=np.uint8)
    rgb[10:30,10:20]=[220,190,10]
    rgb[10:30,20:30]=[190,220,10]
    rgb[10:30,50:64]=[10,200,20]
    path=tmp_path/'frame.png';Image.fromarray(rgb).save(path)
    regions=color_region_candidates(PlatformObservation('x',(),(str(path),),{}))
    assert len(regions)==2
    assert any(c['bbox_xyxy']==[10,10,30,30] for c in regions)
    assert all('name' not in c and 'category' not in c for c in regions)


@pytest.fixture
def finalization_worker(tmp_path):
    import numpy as np
    from agentic_rag_vlm.simulation.protocol_gateway import PhysicsWorker

    names = ['table', 'target_geom', 'target_geom2', 'lf_collision']
    positions = {
        'target_cube': np.array([.26, .12, .018]),
        'target_cube2': np.array([.19, .17, .018]),
    }
    calls = []
    worker = object.__new__(PhysicsWorker)
    worker.output = tmp_path
    worker.finalized = False
    worker._capture = lambda: calls.append('capture')
    worker.sim = SimpleNamespace(
        step=lambda count: calls.append(('step', count)),
        save=lambda result: calls.append('save'),
        setup={'destination': positions['target_cube'].tolist()},
        data=SimpleNamespace(
            body=lambda name: SimpleNamespace(xpos=positions[name]),
            contact=[SimpleNamespace(geom1=0, geom2=1, dist=0.)],
        ),
        model=SimpleNamespace(geom=lambda index: SimpleNamespace(name=names[index])),
        bilateral_lift_seconds=1.,
        lift_seconds_by_object={'target_cube': 1., 'target_cube2': 1.},
        max_protected_displacement=0.,
        protected_contacts=[],
        forbidden_contacts=[],
        self_contacts=[],
        max_target_penetration=0.,
    )
    return worker, calls


def test_repeated_finalization_preserves_physics_and_saved_result(finalization_worker):
    worker, calls = finalization_worker
    first = worker.finalize()
    assert first['simulation_success']
    second = worker.finalize()
    assert second == first
    second['simulation_success'] = False
    assert worker.finalize() == first
    assert calls == [('step', 500), 'capture', 'save']


def test_queued_capture_after_finalization_keeps_last_observation():
    from agentic_rag_vlm.simulation.protocol_gateway import PhysicsWorker

    worker = object.__new__(PhysicsWorker)
    worker.finalized = True
    worker.cache = {'top': b'final jpeg', 'captured_at': 123.}
    worker.sim = SimpleNamespace(rgbd=lambda: pytest.fail('renderer already closed'))
    worker._capture()
    assert worker.cache == {'top': b'final jpeg', 'captured_at': 123.}


def test_first_target_requires_its_own_bilateral_lift(finalization_worker):
    worker, _ = finalization_worker
    worker.sim.bilateral_lift_seconds = 2.
    worker.sim.lift_seconds_by_object['target_cube'] = 0.
    result = worker.finalize()
    assert not result['checks']['bilateral_lift']
    assert result['bilateral_lift_seconds'] == 0.
    assert not result['simulation_success']


@pytest.mark.parametrize('still_held', [False, True])
def test_second_target_requires_release_from_robot(finalization_worker, still_held):
    from scripts.run_blupe_robustness_case import FaultWorker

    worker, _ = finalization_worker
    fault_worker = object.__new__(FaultWorker)
    fault_worker.__dict__.update(worker.__dict__)
    fault_worker.multiple = True
    fault_worker.faults = []
    fault_worker.executed_waypoints = []
    fault_worker.sim.data.contact.append(SimpleNamespace(geom1=0, geom2=2, dist=0.))
    if still_held:
        fault_worker.sim.data.contact.append(SimpleNamespace(geom1=2, geom2=3, dist=0.))
    result = fault_worker.finalize()
    assert result['second_target']['released'] is (not still_held)
    assert result['simulation_success'] is (not still_held)
