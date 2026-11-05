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
