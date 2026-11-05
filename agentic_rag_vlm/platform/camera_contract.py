"""Bind projection intrinsics to the actual RGB stream, without invented geometry."""
from copy import deepcopy
import io
import json
from pathlib import Path
from urllib.parse import urlsplit,quote
from zipfile import ZipFile
import numpy as np
from PIL import Image
from .geometry import GeometryError
from ..experience import atomic_json


def projection_calibration(calibration,profile):
    """Only admit the zero-distortion factory profile observed at Robo-house.

    Paired raw RGB is mandatory: its factory K must not be applied to the
    separately rectified public JPEG. Nonzero/unknown factory models fail closed.
    """
    camera=calibration['cameras']['top']
    if camera.get('distortion_model')!='realsense_factory':return calibration
    if profile.get('top_rgb_source')!='paired_rgbd':
        raise GeometryError('RealSense factory intrinsics require paired_rgbd RGB; independent JPEG is rectified')
    intr=camera.get('depth_intrinsics',{})
    if intr.get('model') not in ('distortion.none','distortion.inverse_brown_conrady','distortion.brown_conrady'):
        raise GeometryError('unsupported RealSense factory model')
    coeff=np.asarray(intr.get('coeffs'),dtype=float)
    distortion=np.asarray(camera.get('distortion'),dtype=float)
    if coeff.shape!=(5,) or distortion.shape!=(5,) or not np.isfinite(coeff).all() or not np.isfinite(distortion).all() or np.any(coeff!=0) or np.any(distortion!=0):
        raise GeometryError('nonzero RealSense factory distortion is not qualified by this adapter')
    K=np.array([[intr['fx'],0,intr['ppx']],[0,intr['fy'],intr['ppy']],[0,0,1.]])
    if [intr['width'],intr['height']]!=camera['image_size'] or not np.allclose(K,camera['K'],atol=1e-6,rtol=0):
        raise GeometryError('factory intrinsics do not match the advertised RGB calibration')
    value=deepcopy(calibration);value['cameras']['top']['distortion_model']='none'
    value['projection_adapter']={'source_model':'realsense_factory','rgb_source':'paired_rgbd',
        'scope':'zero-distortion factory pinhole equivalence only; no new physical calibration'}
    return value


def decode_paired_rgb(raw,calibration,robot_id,headers=None):
    if not raw:raise GeometryError('no fresh paired RGB-D capture')
    if len(raw)>16*1024*1024:raise GeometryError('RGB-D transfer exceeds supported bound')
    with ZipFile(io.BytesIO(raw)) as archive:
        if sum(i.file_size for i in archive.infolist())>48*1024*1024:
            raise GeometryError('RGB-D expanded arrays exceed supported bound')
    with np.load(io.BytesIO(raw),allow_pickle=False) as arrays:
        rgb=arrays['rgb'];metadata=json.loads(arrays['metadata'].item())
        # Depth is retained in the original packet but not used by the current
        # measured-plane controller; do not call this a depth-guided grasp.
        depth_shape=arrays['depth_m'].shape;depth_dtype=arrays['depth_m'].dtype
    camera=calibration['cameras']['top'];width,height=camera['image_size']
    if rgb.dtype!=np.uint8 or rgb.shape!=(height,width,3) or depth_shape!=(height,width) or depth_dtype!=np.float32:
        raise GeometryError('RGB-D array size/type differs from calibrated RGB')
    if (metadata.get('robot_id')!=robot_id or metadata.get('camera')!='top'
            or metadata.get('calibration_id')!=calibration['calibration_id']):
        raise GeometryError('RGB-D robot/camera/calibration mismatch')
    if (metadata.get('depth_units')!='meters' or metadata.get('depth_aligned_to')!='color'
            or metadata.get('depth_semantics')!='optical_z'):
        raise GeometryError('unexpected paired RGB-D units or alignment')
    intr=metadata.get('factory_color_intrinsics',{})
    packaged=camera.get('depth_intrinsics',{})
    for key in ('width','height','fx','fy','ppx','ppy','model','coeffs'):
        if intr.get(key)!=packaged.get(key):raise GeometryError('RGB-D factory profile changed: '+key)
    if camera.get('serial') and metadata.get('sensor_serial')!=camera['serial']:
        raise GeometryError('RGB-D sensor identity mismatch')
    captured=metadata.get('color_captured_at',metadata.get('captured_at'))
    if type(captured) not in (int,float) or not np.isfinite(captured):raise GeometryError('RGB capture timestamp missing')
    headers={k.lower():v for k,v in (headers or {}).items()}
    if headers.get('x-calibration-id') not in (None,calibration['calibration_id']):
        raise GeometryError('RGB-D response calibration header mismatch')
    return rgb,metadata,captured


def fetch_paired_top(client,robot_id,calibration,folder):
    camera=calibration['cameras']['top'];url=camera.get('depth',{}).get('bundle_url')
    parsed=urlsplit(url or '');base=urlsplit(client.endpoint_base)
    if (not url or parsed.query or parsed.fragment or parsed.username or parsed.password
            or parsed.netloc and (parsed.scheme,parsed.netloc)!=(base.scheme,base.netloc)
            or parsed.path!=f'/v1/robots/{quote(robot_id,safe="")}/cameras/top.rgbd.npz'):
        raise GeometryError('paired RGB-D URL must match this robot and Session API origin')
    raw=client._request('GET',url,is_json=False)
    rgb,metadata,captured=decode_paired_rgb(raw,calibration,robot_id,getattr(client,'last_binary_headers',{}))
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    path=folder/'top.png';Image.fromarray(rgb).save(path)
    (folder/'top.rgbd.npz').write_bytes(raw);atomic_json(folder/'paired_rgbd_metadata.json',metadata)
    if not hasattr(client,'image_metadata'):client.image_metadata={}
    client.image_metadata[str(path)]={'x-captured-at':str(captured),'x-calibration-id':calibration['calibration_id']}
    return {'top':str(path)}
