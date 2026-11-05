"""Stream/calibration mismatches must not silently enter metric grounding."""
from copy import deepcopy
import io
import json
import time
import numpy as np
import pytest
from agentic_rag_vlm.platform.camera_contract import projection_calibration,decode_paired_rgb,fetch_paired_top
from agentic_rag_vlm.platform.geometry import GeometryError,PlaneProjector
from agentic_rag_vlm.platform.transport import RemotePlatformTransport,PlatformUnavailableError
from agentic_rag_vlm.platform.config import PlatformConfig
from tests.test_delivery import calibration,measured_test_profile


def factory_calibration():
    value=calibration();camera=value['cameras']['top'];K=camera['K'];size=camera['image_size']
    camera.update(distortion_model='realsense_factory',distortion=[0.]*5,serial='SYNTHETIC',
        depth_intrinsics={'width':size[0],'height':size[1],'fx':K[0][0],'fy':K[1][1],
            'ppx':K[0][2],'ppy':K[1][2],'model':'distortion.inverse_brown_conrady','coeffs':[0.]*5},
        depth={'bundle_url':'/v1/robots/yam-1/cameras/top.rgbd.npz'})
    return value


def packet(cal=None,**updates):
    cal=cal or factory_calibration();width,height=cal['cameras']['top']['image_size']
    meta={'robot_id':'yam-1','camera':'top','calibration_id':cal['calibration_id'],
        'color_captured_at':time.time(),'factory_color_intrinsics':deepcopy(cal['cameras']['top']['depth_intrinsics']),
        'sensor_serial':'SYNTHETIC','depth_units':'meters','depth_aligned_to':'color','depth_semantics':'optical_z'}
    meta.update(updates)
    raw=io.BytesIO();np.savez_compressed(raw,rgb=np.zeros((height,width,3),np.uint8),
        depth_m=np.ones((height,width),np.float32),metadata=np.array(json.dumps(meta)))
    return raw.getvalue()


def test_zero_distortion_factory_projection_requires_paired_rgb_and_keeps_raw_calibration():
    cal=factory_calibration()
    with pytest.raises(GeometryError,match='independent JPEG'):projection_calibration(cal,{'top_rgb_source':'jpeg'})
    normalized=projection_calibration(cal,{'top_rgb_source':'paired_rgbd'})
    assert normalized['calibration_id']==cal['calibration_id']
    assert cal['cameras']['top']['distortion_model']=='realsense_factory'
    assert normalized['cameras']['top']['distortion_model']=='none'
    projector=PlaneProjector(normalized,'right',plane_normal_base=[0,0,1],plane_offset_m=0)
    assert np.isfinite(projector.project_base([320,180],height_m=.1)).all()


def test_nonzero_factory_distortion_is_not_silently_treated_as_pinhole():
    cal=factory_calibration();cal['cameras']['top']['depth_intrinsics']['coeffs'][0]=.02
    with pytest.raises(GeometryError,match='nonzero'):projection_calibration(cal,{'top_rgb_source':'paired_rgbd'})


@pytest.mark.parametrize('updates',[
    {'robot_id':'other'}, {'camera':'left'}, {'calibration_id':'changed'}, {'sensor_serial':'other'},
    {'color_captured_at':None}, {'depth_units':'millimeters'}])
def test_paired_capture_identity_and_timestamp_must_match(updates):
    with pytest.raises(GeometryError):decode_paired_rgb(packet(**updates),factory_calibration(),'yam-1')


def test_paired_capture_rejects_changed_intrinsics():
    rawcal=factory_calibration();changed=deepcopy(rawcal['cameras']['top']['depth_intrinsics']);changed['fx']+=2
    with pytest.raises(GeometryError,match='profile changed'):
        decode_paired_rgb(packet(factory_color_intrinsics=changed),rawcal,'yam-1')


def test_empty_rgbd_is_unavailable_not_a_successful_observation():
    with pytest.raises(GeometryError,match='no fresh'):decode_paired_rgb(b'',factory_calibration(),'yam-1')


def test_transport_uses_paired_rgb_instead_of_rectified_jpeg_and_checks_age(tmp_path):
    cal=factory_calibration()
    class Client:
        endpoint_base='http://127.0.0.1:1'
        def get_calibration(self,robot):return cal
        def get_station_observation(self,robot):
            return {'source':'hardware','homed':True,'settled':True,'observed_at':time.time(),
                'left_joints_deg':[0.]*6,'right_joints_deg':[0.]*6,'left_gripper':.3,'right_gripper':.4}
        def download_observation_images(self,*a):raise AssertionError('independent JPEG must not be used')
        def _request(self,method,path,**kw):
            assert method=='GET' and path.endswith('/top.rgbd.npz')
            self.last_binary_headers={'x-calibration-id':cal['calibration_id']}
            return packet(color_captured_at=self.captured)
    client=Client();client.captured=time.time()
    profile=measured_test_profile();profile['top_rgb_source']='paired_rgbd'
    transport=RemotePlatformTransport(PlatformConfig('blupe','NONE',client.endpoint_base,'yam-1'),
        client=client,deployment_profile=profile,image_dir=tmp_path)
    observation=transport.get_observation()
    assert observation.image_paths[0].endswith('top.png')
    assert observation.receipt['top_rgb_source']=='paired_rgbd'
    assert list(tmp_path.glob('*/paired_rgbd_metadata.json'))
    client.captured=time.time()-10
    with pytest.raises(PlatformUnavailableError,match='stale'):
        transport.get_observation()
