"""Official BluPe Runner imports and strictly local recorded-image preview.

This preview adapter exercises the real runner task hook with real models, but
does not replace the production AgenticRAGVLMProvider execution adapter.
"""
import importlib
import json
import re
from pathlib import Path
import sys
import threading

from ..experience import atomic_json
from .http_policy import preview_request


def import_official_runner(checkout):
    root=Path(checkout).resolve()/'codex-runner'
    if not (root/'src/remote_yam/controller.py').is_file():
        raise ValueError('--runner-checkout must name the official repository root')
    existing=sys.modules.get('remote_yam.controller')
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(root):
        raise ValueError('another remote_yam checkout is already imported')
    sys.path[:0]=[str(root),str(root/'src')]
    controller=importlib.import_module('remote_yam.controller')
    session=importlib.import_module('remote_yam.session')
    return controller,session


class RecordedPreviewAdapter:
    """Only admits a marked local fixture; never submits robot commands."""
    provider_name='agentic-rag-vlm'
    model='Agentic RAG-VLM'
    trajectory_cadence_hz=10

    def __init__(self,provider,payload,output,*,lock=None):
        self.provider=provider;self.payload=payload;self.output=Path(output)
        self.cancelled=lambda:False;self.interaction_sink=None
        self._lock=lock or threading.Lock()
        self.stage='ready';self.result=None

    def _stage(self,stage,message):
        self.stage=stage
        if callable(self.interaction_sink):
            self.interaction_sink('run_notice',message,stage=stage,execution_mode='local_recorded_image_preview')

    def public_config(self):
        return {**self.provider.public_config(),'execution_mode':'local_recorded_image_preview',
                'task_progress':{'stage':self.stage,'execution_performed':False,
                                 'decision':self.result.get('decision') if self.result else None}}

    def prepare_before_session(self,prompt):
        if prompt.strip()!=self.payload['task']['instruction'].strip():
            raise ValueError('This local preview uses the displayed recorded-image task only')
        self._stage('preparing','Checking local VLM, CLIP and knowledge before the local fixture starts')
        self.provider.prepare()
        if self.cancelled():raise RuntimeError('local preview cancelled during preparation')
        self._stage('ready','Models ready; local recorded-image preview only')

    def validate_session_admission(self,api):
        if getattr(api,'local_validation_only',False) is not True:
            raise ValueError('Recorded preview requires the local mock API; public sessions are forbidden')

    def infer(self,*args):
        raise RuntimeError('Use the official full task hook, not individual joint inference')

    def run_session(self,prompt,observation,session_api,context,cancelled):
        self.validate_session_admission(session_api)
        if cancelled():return {'state':'cancelled'}
        self._stage('waiting_for_model','Waiting for the local model slot; no robot command is sent')
        while not self._lock.acquire(timeout=.1):
            if cancelled():return {'state':'cancelled'}
        try:
            if cancelled():return {'state':'cancelled'}
            self._stage('reasoning','Running image classification, CLIP, HAA-RAG, constraints and VLM review')
            self.result=preview_request(self.provider,self.payload,self.output/'previews')
            atomic_json(self.output/'latest_preview.json',self.result)
            if cancelled():
                self._stage('cancelled','Preview cancelled; no action was executed')
                return {'state':'cancelled'}
            self._stage('preview_complete','Preview decision: '+self.result['decision']+'; no action was executed')
            return {'state':'completed','execution_performed':False,
                    'decision':self.result['decision'],'result_path':str(self.output/'latest_preview.json')}
        finally:self._lock.release()


def make_local_playground(checkout,provider,payload,output,*,port=18791):
    _,session=import_official_runner(checkout)
    from playground import HostedRunner,RequestError
    from PIL import Image
    import base64,io
    raw=base64.b64decode(payload['image']['data_base64'],validate=True)
    image=io.BytesIO()
    with Image.open(io.BytesIO(raw)) as decoded:decoded.convert('RGB').save(image,format='PNG')
    png=image.getvalue();lock=threading.Lock()
    preview_js=b'''async function agenticLastPreview(){try{const r=await(await fetch('/agentic/result')).json();
const el=document.getElementById('agenticPreviewResult');if(r.decision){el.textContent=
'Last preview: '+r.decision+' | Target: '+(r.action?.target||'none')+' | Retrieval: '+(r.action?.source||'none')+
' | VLM action reviews: '+(r.reasoner_calls?.length||0)+' | Executed: NO';}}
catch(e){}}agenticLastPreview();setInterval(agenticLastPreview,1500);'''
    preview_css=(b'#agenticRecorded{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;'
                 b'background:#151719;z-index:20;pointer-events:none}'
                 b'#liveViewer .liveStopwatch,#liveViewer #viewerCue{z-index:25}'
                 b'#agenticLocalBanner{background:#fff2c6;padding:12px;text-align:center;color:#222}'
                 b'#agenticPreviewResult{font-size:13px;margin-top:6px;font-weight:600}')
    class LocalFixture(session.MockSessionAPI):
        local_validation_only=True
        def get_robot_observation(self,robot):
            value=super().get_robot_observation(robot)
            value['images']={'top':{'url':'/agentic/recorded.png'}}
            return value
    class Playground(HostedRunner):
        def page(self):
            html=super().page()
            banner=(b'<div id="agenticLocalBanner">'
                    b'LOCAL VALIDATION | Recorded simulation image | Real Qwen + CLIP | '
                    b'No physical robot or public queue | <a href="/agentic/result">Latest result</a>'
                    b'<div id="agenticPreviewResult">No preview result yet.</div></div>')
            html=html.replace(b'<body class="playgroundConsole">',b'<body class="playgroundConsole">'+banner,1)
            html=html.replace(b'</head>',b'<link rel="stylesheet" href="/agentic/local.css">'
                b'<script src="/agentic/local.js" defer></script></head>')
            return re.sub(b'(<section id="liveViewer"[^>]*>)',
                rb'\1<img id="agenticRecorded" src="/agentic/recorded.png" alt="Recorded top-camera model input">',html,count=1)
        def validate_launch(self,visitor,request,paid):
            if request.get('provider')!='agentic-rag-vlm':
                raise RequestError(400,'Only the local Agentic RAG-VLM preview is enabled')
            if request.get('prompt','').strip()!=payload['task']['instruction'].strip():
                raise RequestError(400,'Use the configured recorded-image task shown in the model selector')
            if request.get('share_conversation') is not False:
                raise RequestError(400,'Local validation requires conversation sharing disabled')
        async def session_extension(self,cookies,visitor):
            setup,headers=await super().session_extension(cookies,visitor)
            return {**setup,'local_runner':True,'default_provider':'agentic-rag-vlm',
                    'configured_providers':{'agentic-rag-vlm':{
                'label':'Agentic RAG-VLM — local preview','model':'Agentic RAG-VLM',
                'help':'Recorded simulation image; model reasoning only, no physical execution.',
                'task':payload['task']['instruction'],'default_run_duration_s':300}}},headers
        async def http(self,scope,receive,send):
            path=scope['path']
            if path.startswith('/agentic/'):
                headers={k.decode().lower():v.decode() for k,v in scope.get('headers',[])}
                if scope['method']!='GET' or headers.get('host')!=self.authority:
                    raise RequestError(403,'Local GET request required')
                if path=='/agentic/recorded.png':
                    await send({'type':'http.response.start','status':200,
                        'headers':[(b'content-type',b'image/png'),(b'content-length',str(len(png)).encode())]})
                    await send({'type':'http.response.body','body':png});return
                if path in {'/agentic/local.js','/agentic/local.css'}:
                    content,mime=(preview_js,b'application/javascript') if path.endswith('.js') else (preview_css,b'text/css')
                    await send({'type':'http.response.start','status':200,'headers':[
                        (b'content-type',mime),(b'content-length',str(len(content)).encode())]})
                    await send({'type':'http.response.body','body':content});return
                if path=='/agentic/result':
                    file=Path(output)/'latest_preview.json'
                    result=json.loads(file.read_text()) if file.exists() else {'status':'not_run','execution_performed':False}
                    return await self.json(send,200,result)
                raise RequestError(404,'Unknown local preview route')
            return await super().http(scope,receive,send)
    origin=f'http://127.0.0.1:{port}'
    def factory(model,directory):
        if model!='Agentic RAG-VLM':raise RequestError(400,'Unknown approach')
        return RecordedPreviewAdapter(provider,payload,output,lock=lock)
    return Playground(public_origin=origin,session_api=None,camera_origin=origin,development=True,
        hardware_control=False,api_factory=LocalFixture,share_conversation=False,
        default_provider='agentic-rag-vlm',extra_providers={'agentic-rag-vlm':factory},
        camera_names=('top',),policy_camera_names=('top',),recording_root=Path(output)/'runner_recordings',
        robots=[{'id':'yam-1','name':'YAM — LOCAL FIXTURE','url':origin+'/'}])
