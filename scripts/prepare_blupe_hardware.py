"""Generate pending site files and capture public hardware data using GET only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agentic_rag_vlm.experience import atomic_json
from agentic_rag_vlm.platform.blupe_client import BluPeSessionClient
from agentic_rag_vlm.platform.config import load_platform_config


def initialize_site(output, platform_config, *, code_root=ROOT):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    code_root=Path(code_root).resolve()
    atomic_json(output/'platform.local.json',platform_config.to_dict())
    for source,dest in [('blupe_deployment.template.json','deployment.local.json'),
                        ('blupe_measured_task.template.json','task.local.json'),
                        ('blupe_candidates.template.json','candidates.local.json')]:
        data=json.loads((ROOT/'configs'/source).read_text())
        if source.startswith('blupe_deployment'):
            data.update(robot_id=platform_config.robot_id,execution_environment='hardware',
                arm=None,max_joint_delta_deg_per_step=None,lift_height_m=None,top_rgb_source=None,
                observation_park_xyz_m=None,
                retention_observation={'minimum_blocked_aperture':None,
                    'maximum_blocked_aperture':None,'minimum_tcp_lift_m':None})
        atomic_json(output/dest,data)
    registry=json.loads((ROOT/'knowledge_base/blupe_semantic_registry.json').read_text())
    registry['objects']={'cube':dict(registry['objects']['cube'],visual_description='SITE_OBJECT_DESCRIPTION_REQUIRED')}
    atomic_json(output/'registry.local.json',registry)
    provider=json.loads((ROOT/'configs/blupe_provider.example.json').read_text())
    provider.update(knowledge_store=os.path.relpath(code_root/'knowledge_base/yam_verified_diverse_r8.json',output),
        registry='registry.local.json',platform_config='platform.local.json',output='../runs',
        deployment_profile=None,task=None,destinations={})
    atomic_json(output/'provider.monitor.local.json',provider)
    provider.update(deployment_profile='deployment.local.json',task='task.local.json',candidate_config='candidates.local.json')
    atomic_json(output/'provider.hardware.local.json',provider)
    atomic_json(output/'site_handoff.json',{
        'physical_validation_passed':False,'hardware_session_created':False,
        'required_from_site':[
            'robot and operating arm; initialization/Home workflow',
            'fresh independently checked camera/tool geometry with calibration ID and evidence',
            'joint limits, workspace and allowed joint step/speed',
            'table plane, object dimensions, top-feature height and grasp offset',
            'destination observable-feature coordinates in selected arm base',
            'gripper aperture and normalized measured feedback/blocked aperture semantics',
            'lift height, post-release observation pose and stop/operator procedure',
            'actual object appearance and original-image ROI/HSV thresholds'],
        'note':'Pending fields are intentional. Never mark physical validation passed using simulator evidence.'})


def collect_public(output, config):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    robot=quote(config.robot_id,safe='')
    paths={'robots':'/v1/robots','queue':f'/v1/robots/{robot}/queue',
        'observation':f'/v1/robots/{robot}/observation','calibration':f'/v1/robots/{robot}/calibration'}
    paths.update({role:f'/v1/robots/{robot}/cameras/{role}.jpg' for role in ('top','left','right')})
    def fetch(item):
        name,path=item
        client=BluPeSessionClient(config.endpoint_base,request_timeout_s=8,max_retries=0)
        try:
            binary=name in ('top','left','right')
            value=client._request('GET',path,is_json=not binary)
            if binary:
                filename=name+'.jpg';(output/filename).write_bytes(value)
                headers={k.lower():v for k,v in client.last_binary_headers.items()}
                atomic_json(output/(name+'_headers.json'),headers)
                return name,{'ok':True,'file':filename,'captured_at':headers.get('x-captured-at'),
                    'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()}
            atomic_json(output/(name+'.json'),value)
            return name,{'ok':True,'file':name+'.json'}
        except Exception as error:
            return name,{'ok':False,'error_type':type(error).__name__,'message':str(error)}
    with ThreadPoolExecutor(max_workers=7) as pool:results=dict(pool.map(fetch,paths.items()))
    summary={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'robot_id':config.robot_id,
        'http_methods':['GET'],'session_created':False,'motion_submitted':False,'resources':results}
    for name,keys in [('observation',('source','mode','homed','settled','queue_ready','safety')),
                      ('calibration',('calibration_id','quality'))]:
        if results[name]['ok']:
            data=json.loads((output/(name+'.json')).read_text());summary[name]={k:data.get(k) for k in keys}
    summary['note']='Snapshot only; disabled standby is not proof of station failure. Home may occur during session preparation. No site measurements are certified.'
    atomic_json(output/'snapshot_summary.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--platform-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--capture',action='store_true',help='also GET public metadata and RGB images; never join a queue')
    args=p.parse_args();config=load_platform_config(args.platform_config)
    args.output.mkdir(parents=True,exist_ok=False)
    initialize_site(args.output/'site',config)
    result=collect_public(args.output/'snapshot',config) if args.capture else {'session_created':False,'motion_submitted':False}
    print(json.dumps(result,ensure_ascii=False,indent=2))
