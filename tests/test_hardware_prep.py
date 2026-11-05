"""Local preparation contracts. Synthetic geometry never qualifies hardware."""
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest

from agentic_rag_vlm.platform.candidates import ColorRegionCandidates
from agentic_rag_vlm.platform.config import load_platform_config
from agentic_rag_vlm.platform.hardware_runner import ReadOnlySessionAPI,validate_hardware_provider,make_hardware_playground
from agentic_rag_vlm.platform.transport import PlatformObservation
from agentic_rag_vlm.platform.deployment import validate_profile_fields
from agentic_rag_vlm.platform.geometry import GeometryError
from scripts.prepare_blupe_hardware import initialize_site,collect_public
from tests.test_delivery import measured_test_profile

ROOT=Path(__file__).resolve().parents[1]


def color_config():
    return {'kind':'color_regions','image_size_px':[100,80],'roi_xyxy':[5,5,95,75],
        'hsv_bands':[{'hue_min':240,'hue_max':15,'saturation_min':120,'value_min':60},
                     {'hue_min':70,'hue_max':110,'saturation_min':120,'value_min':60}],
        'minimum_area_px':20,'maximum_area_px':1000}


def test_rgb_proposals_only_expose_pixels_and_respect_site_roi(tmp_path):
    rgb=np.full((80,100,3),100,dtype=np.uint8)
    rgb[10:30,10:30]=[230,20,10];rgb[40:60,60:80]=[10,200,20]
    rgb[:3,:3]=[230,20,10]
    image=tmp_path/'image.png';Image.fromarray(rgb).save(image)
    proposals=ColorRegionCandidates(color_config())(PlatformObservation('x',(),(str(image),)))
    assert len(proposals)==2
    assert proposals[0]['bbox_xyxy']==[10,10,30,30]
    assert all(set(c)=={'candidate_id','bbox_xyxy','pixel_xy'} for c in proposals)
    Image.new('RGB',(200,160)).save(image)
    with pytest.raises(ValueError,match='resolution changed'):
        ColorRegionCandidates(color_config())(PlatformObservation('x',(),(str(image),)))


def test_pending_color_template_fails_without_inventing_dimensions():
    with pytest.raises(ValueError,match='pending'):
        ColorRegionCandidates(json.loads((ROOT/'configs/blupe_candidates.template.json').read_text()))


def test_default_monitor_denies_all_write_methods():
    calls=[]
    api=SimpleNamespace(get_queue_snapshot=lambda:calls.append('GET'),create_session=lambda **kw:calls.append('POST'))
    readonly=ReadOnlySessionAPI(api)
    readonly.get_queue_snapshot()
    for method in ('create_session','submit_trajectory','stop_session'):
        with pytest.raises(RuntimeError,match='Read-only'):getattr(readonly,method)()
    assert calls==['GET']


def test_site_generation_keeps_measurements_pending_and_portable(tmp_path):
    config=load_platform_config(ROOT/'configs/blupe_platform.example.json')
    initialize_site(tmp_path/'site',config)
    profile=json.loads((tmp_path/'site/deployment.local.json').read_text())
    assert profile['physical_validation']['passed'] is False
    assert profile['support_plane']['offset_m'] is None
    assert profile['max_joint_delta_deg_per_step'] is None
    provider=json.loads((tmp_path/'site/provider.monitor.local.json').read_text())
    assert provider['deployment_profile'] is None and provider['task'] is None
    assert (tmp_path/'site'/provider['knowledge_store']).resolve().is_file()
    with pytest.raises(FileExistsError):initialize_site(tmp_path/'site',config)


def test_provider_loader_wires_site_candidates_without_starting_models(tmp_path):
    from agentic_rag_vlm.platform.runner_provider import load_provider
    initialize_site(tmp_path/'site',load_platform_config(ROOT/'configs/blupe_platform.example.json'))
    site=tmp_path/'site'
    (site/'candidates.local.json').write_text(json.dumps(color_config()))
    task=json.loads((site/'task.local.json').read_text());task['destinations']['tray_zone']=[.2,.1,.04]
    (site/'task.local.json').write_text(json.dumps(task))
    provider=load_provider(site/'provider.hardware.local.json')
    assert isinstance(provider.candidate_provider,ColorRegionCandidates)
    assert provider.client is None  # loading alone performs no model or robot call


def test_read_only_capture_never_invokes_a_write_method(tmp_path,monkeypatch):
    import scripts.prepare_blupe_hardware as module
    calls=[]
    class Client:
        def __init__(self,*a,**kw):self.last_binary_headers={'X-Captured-At':'123'}
        def _request(self,method,path,**kw):
            calls.append((method,path));assert method=='GET'
            return b'JPEG TEST FIXTURE' if path.endswith('.jpg') else {}
    monkeypatch.setattr(module,'BluPeSessionClient',Client)
    report=collect_public(tmp_path/'capture',load_platform_config(ROOT/'configs/blupe_platform.example.json'))
    assert len(calls)==7 and all(method=='GET' for method,_ in calls)
    assert not report['session_created'] and not report['motion_submitted']


def test_hardware_admission_rejects_simulation_and_missing_measurements():
    provider=SimpleNamespace(profile={'execution_environment':'simulation'})
    with pytest.raises(ValueError,match='simulator'):validate_hardware_provider(provider)
    profile=measured_test_profile();profile['max_joint_delta_deg_per_step']=9
    with pytest.raises(GeometryError,match='3 degrees'):
        validate_profile_fields(profile,robot_id='yam-1',arm='right')


def test_official_monitor_run_route_is_blocked_before_queue_creation(tmp_path):
    checkout=os.environ.get('BLUPE_RUNNER_CHECKOUT')
    if not checkout:pytest.skip('set BLUPE_RUNNER_CHECKOUT for official monitor route test')
    initialize_site(tmp_path/'site',load_platform_config(ROOT/'configs/blupe_platform.example.json'))
    # Deliberately use pending HARDWARE files: monitoring must work before they
    # can be loaded as an executing provider.
    app=make_hardware_playground(checkout,tmp_path/'site/provider.hardware.local.json',tmp_path/'runs')
    async def run():
        cookie='';csrf=''
        async def call(path,payload):
            nonlocal cookie,csrf
            raw=json.dumps(payload).encode();sent=[]
            async def receive():return {'type':'http.request','body':raw,'more_body':False}
            async def send(value):sent.append(value)
            headers={'host':'127.0.0.1:18797','origin':app.origin,'cookie':cookie,
                'content-type':'application/json','x-yam-runner-token':csrf}
            await app({'type':'http','method':'POST','path':path,'query_string':b'',
                       'headers':[(k.encode(),v.encode()) for k,v in headers.items()]},receive,send)
            h=dict(sent[0]['headers'])
            if b'set-cookie' in h:cookie=h[b'set-cookie'].decode().split(';')[0]
            result=json.loads(b''.join(s.get('body',b'') for s in sent[1:]))
            if 'csrf' in result:csrf=result['csrf']
            return sent[0]['status'],result
        status,data=await call('/api/session',{})
        assert status==200
        status,data=await call('/api/run',{'provider':'agentic-rag-vlm','model':'Agentic RAG-VLM',
            'prompt':'test','share_conversation':False})
        assert status==409 and 'Read-only' in str(data)
    asyncio.run(run())
    assert not app.hardware_control
    with pytest.raises(RuntimeError,match='Read-only'):app.api_factory().create_session('test')
