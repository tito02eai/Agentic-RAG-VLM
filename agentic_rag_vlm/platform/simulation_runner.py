"""Execute complete local MuJoCo tasks through the official BluPe task hook.

The unmodified controller owns only a LOCAL mock session. A child process runs
the existing contact-dynamics executor and actual Agent models, never BluPe's
public robot API. This is not a calibrated BluPe digital twin.
"""
from dataclasses import dataclass
import json
import html
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import urlsplit

from ..experience import atomic_json
from .local_runner import import_official_runner


CASES={
    'agentic-flat':{'label':'Agentic RAG-VLM — three-object sorting','model':'Agentic RAG-VLM',
        'seed':980,'mode':'flat_sort','policy':'agentic',
        'task':'Sort the red, yellow and magenta objects into their individual zones; protect the green vessel.'},
    'agentic-stack':{'label':'Agentic RAG-VLM — unstack and sort','model':'Agentic RAG-VLM',
        'seed':981,'mode':'stacked_sort','policy':'agentic',
        'task':'Unstack and sort all three objects into their individual zones; protect the green vessel.'},
    'geometric-stack':{'label':'Geometric baseline — matched stacked scene','model':'Geometric baseline',
        'seed':981,'mode':'stacked_sort','policy':'geometric',
        'task':'Unstack and sort all three objects into their individual zones; protect the green vessel.'},
}


def read_json(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default


def simulation_command(root,case,folder,provider,python=sys.executable):
    """Fixed scenarios/flags; browser prompts cannot inject CLI or coordinates."""
    root=Path(root);folder=Path(folder)
    cmd=[str(python),str(root/'scripts/run_extended_sim.py'),'--output',str(folder/'simulation'),
        '--store',str(root/'knowledge_base/yam_verified_diverse_r8.json'),'--mode',case['mode'],
        '--seed',str(case['seed']),'--friction','.7','--controller-profile','guarded',
        '--scene-variation','mixed_contact','--recovery-policy','force_ladder','--mixed-shapes',
        '--grounding-repair','--multiccd','--record','--live-output',str(folder/'live'),
        '--endpoint',provider.endpoint,'--model',provider.served_model,'--clip-endpoint',provider.clip_endpoint]
    if case['mode']=='stacked_sort':cmd+=['--upper-half-width','.017']
    if case['policy']=='geometric':
        cmd+=['--retrieval-profile','fixed','--planning-policy','geometric','--disable-memory','--disable-scene-graph']
    else:cmd+=['--visibility-replanning','--visible-goal-fallback']
    return cmd


class SimulationState:
    def __init__(self,output):
        self.output=Path(output).resolve();self.output.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock();self.state_lock=threading.RLock()
        self.current=read_json(self.output/'state.json',{'stage':'ready','execution_mode':'local_mujoco_closed_loop'})
    def update(self,**values):
        with self.state_lock:
            self.current.update(values);atomic_json(self.output/'state.json',self.current)
    def status(self):
        with self.state_lock:result=dict(self.current)
        if result.get('run_id'):
            folder=self.output/result['run_id']
            result['motion']=read_json(folder/'live/motion.json',{})
            result['runtime']=read_json(folder/'live/runtime.json',{})
            result['video_ready']=(folder/'simulation/rollout.mp4').is_file()
            try:
                events=[json.loads(line) for line in (folder/'live/runtime_events.jsonl').read_text().splitlines()]
            except (OSError,ValueError):events=[]
            result['last_plan']=next((e['payload'] for e in reversed(events) if e['event']=='subgoal_planned'),None)
            result['verified_targets']=[e['payload']['target'] for e in events
                if e['event']=='subgoal_verified' and e['payload'].get('success')]
            result['recent_events']=[{'event':e['event'],'phase':e['phase']} for e in events[-8:]]
        return result
    def runs(self):
        return [read_json(p) for p in sorted(self.output.glob('*/receipt.json')) if read_json(p)]


@dataclass
class SimulationTaskProvider:
    provider:object
    source_root:Path
    state:SimulationState
    case_key:str
    python:str=sys.executable
    timeout_s:float=600.

    def __post_init__(self):
        if self.case_key not in CASES:raise ValueError('unknown local simulation case')
        self.case=CASES[self.case_key]
        self.provider_name='agentic-rag-vlm' if self.case['policy']=='agentic' else 'geometric-baseline'
        self.model=self.case['model'];self.cancelled=lambda:False;self.interaction_sink=None
    def public_config(self):
        status=self.state.status()
        return {**self.provider.public_config(),'provider':self.provider_name,'model':self.model,
            'execution_mode':'local_mujoco_closed_loop','hardware_qualification':'not_claimed',
            'task_progress':{'task':self.case['task'],'stage':status.get('stage'),
                'run_id':status.get('run_id'),'scenario':self.case_key,'simulation_success':status.get('simulation_success')}}
    def prepare_before_session(self,prompt):
        if prompt.strip()!=self.case['task']:raise ValueError('use the configured simulation task')
        self.provider.prepare()
        if not (Path(self.source_root)/'scripts/run_extended_sim.py').is_file():raise ValueError('simulation source missing')
        if self.cancelled():raise RuntimeError('simulation preparation cancelled')
    def validate_session_admission(self,api):
        if getattr(api,'local_validation_only',False) is not True:
            raise ValueError('local simulation cannot adopt a public robot session')
    def infer(self,*args):raise RuntimeError('complete simulation uses run_session')
    def _notice(self,message,**details):
        if callable(self.interaction_sink):self.interaction_sink('run_notice',message,**details)
    def run_session(self,prompt,observation,session_api,context,cancelled):
        self.validate_session_admission(session_api)
        if cancelled():return {'state':'cancelled'}
        if not self.state.lock.acquire(blocking=False):raise RuntimeError('another local simulation is active')
        run_id=uuid.uuid4().hex;folder=self.state.output/run_id;folder.mkdir(exist_ok=False)
        process=None;started=time.time();was_cancelled=False;timed_out=False
        try:
            cmd=simulation_command(self.source_root,self.case,folder,self.provider,self.python)
            receipt={'run_id':run_id,'scenario':self.case_key,'case':self.case,'command':cmd,
                'execution_mode':'local_mujoco_closed_loop','started_at':started,
                'physical_robot_contacted':False,'public_queue_joined':False,'result':'running'}
            atomic_json(folder/'receipt.json',receipt)
            self.state.update(run_id=run_id,scenario=self.case_key,stage='starting_simulator',simulation_success=None,receipt=None)
            self._notice('Starting actual MuJoCo physics and real VLM/CLIP; local execution only',run_id=run_id)
            with (folder/'process.log').open('w') as log:
                if cancelled():was_cancelled=True
                else:
                    process=subprocess.Popen(cmd,cwd=self.source_root,env={**os.environ,'MUJOCO_GL':'egl'},
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    self.state.update(stage='running_simulation')
                    last_event=None
                    while process.poll() is None:
                        if cancelled() or time.time()-started>self.timeout_s:
                            was_cancelled=cancelled();timed_out=not was_cancelled
                            os.killpg(process.pid,signal.SIGTERM)
                            try:process.wait(timeout=3)
                            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=3)
                            break
                        event=read_json(folder/'live/runtime.json',{})
                        if event and event.get('sequence')!=last_event:
                            last_event=event['sequence']
                            self._notice('Agent event: '+event['event'],phase=event['phase'],payload=event['payload'])
                        time.sleep(.2)
            result=read_json(folder/'simulation/result.json')
            success=bool(result and result.get('physical_success') is True)
            stage='cancelled' if was_cancelled else 'timed_out' if timed_out else 'completed' if success else 'failed'
            receipt.update(result=stage,ended_at=time.time(),wall_elapsed_s=time.time()-started,
                exit_code=process.returncode if process else None,simulation_success=success,
                framework_outcome=result.get('outcome') if result else None,
                attempts=len(result.get('attempts',[])) if result else 0,
                failed_attempts=sum(not a['retained'] for a in result.get('attempts',[])) if result else 0,
                memory_hits=result.get('memory_hits',0) if result else 0,
                checks=result.get('checks',{}) if result else {},
                final_objects=result.get('final_objects',{}) if result else {},
                vlm_perception_calls=len(list((folder/'simulation').glob('observation_*/vlm_response.json'))),
                vlm_review_calls=len(result.get('reasoner_calls',[])) if result else 0)
            atomic_json(folder/'receipt.json',receipt)
            self.state.update(stage=stage,simulation_success=success,receipt=receipt)
            self._notice('Local simulation '+stage,simulation_success=success,run_id=run_id)
            if was_cancelled:return {'state':'cancelled','run_id':run_id}
            if stage!='completed':raise RuntimeError('Local simulation '+stage+'; all evidence retained in '+str(folder))
            return {'state':'completed','run_id':run_id,'simulation_success':True,'hardware_success':None}
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=3)
            self.state.lock.release()


def make_simulation_playground(checkout,provider,source_root,output,*,port=18791,python=sys.executable):
    _,session=import_official_runner(checkout)
    from playground import HostedRunner,RequestError
    state=SimulationState(output)
    class LocalFixture(session.MockSessionAPI):local_validation_only=True
    class Playground(HostedRunner):
        def page(self):
            html=super().page()
            banner=(b'<div id="agenticLocalBanner"><strong>LOCAL MUJOCO CLOSED LOOP</strong> | '
                b'Actual Qwen + CLIP + contact grasping | No public robot connection<br>'
                b'<span id="agenticSimulationStatus">Ready</span> | '
                b'<a href="/experiment/result" target="_blank">Full result</a> | '
                b'<a href="/experiment/video" target="_blank">Grasp video</a> | '
                b'<a href="/experiment/report" target="_blank">Results and all videos</a></div>')
            html=html.replace(b'<body class="playgroundConsole">',b'<body class="playgroundConsole">'+banner,1)
            html=html.replace(b'</head>',b'<link rel="stylesheet" href="/experiment/local.css">'
                b'<script src="/experiment/local.js" defer></script></head>')
            html=html.replace(b'<section id="liveConversationPanel"',
                b'<aside id="agenticEvidence"><h3>Framework execution evidence</h3><pre id="agenticEvidenceText">Waiting for a task</pre></aside><section id="liveConversationPanel"',1)
            return re.sub(b'(<section id="liveViewer"[^>]*>)',
                rb'\1<img id="agenticSimulationFrame" alt="Current local MuJoCo frame" src="/experiment/frame">',html,count=1)
        def validate_launch(self,visitor,payload,paid):
            case=CASES.get(payload.get('provider'))
            if case is None or payload.get('prompt','').strip()!=case['task']:
                raise RequestError(400,'Choose a configured local simulation task')
            if payload.get('share_conversation') is not False:raise RequestError(400,'Local-only conversation required')
            if state.lock.locked():raise RequestError(409,'A simulation is active; wait or Stop it')
        async def session_extension(self,cookies,visitor):
            setup,headers=await super().session_extension(cookies,visitor)
            return {**setup,'local_runner':True,'default_provider':'agentic-stack',
                'configured_providers':{name:{**{k:spec[k] for k in ('label','model','task')},
                    'help':'Real local contact simulation; live image and saved video; no physical robot.',
                    'default_run_duration_s':600} for name,spec in CASES.items()}},headers
        async def http(self,scope,receive,send):
            parsed=urlsplit(scope['path']);path=parsed.path
            if path.startswith('/experiment/'):
                headers={k.decode().lower():v.decode() for k,v in scope.get('headers',[])}
                if scope['method']!='GET' or headers.get('host')!=self.authority:raise RequestError(403,'Local GET required')
                if path=='/experiment/status':return await self.json(send,200,state.status())
                if path=='/experiment/runs':return await self.json(send,200,state.runs())
                if path=='/experiment/report':
                    rows=[]
                    for run in sorted(state.runs(),key=lambda r:r['started_at']):
                        rid=run['run_id'];label=html.escape(run['case']['label'])
                        rows.append(f'<tr><td>{label}</td><td>{html.escape(run["result"])}</td>'
                            f'<td>{run.get("attempts","—")}</td><td>{run.get("failed_attempts","—")}</td>'
                            f'<td>{run.get("memory_hits","—")}</td><td><a href="/experiment/run/{rid}/video">Video</a> · '
                            f'<a href="/experiment/run/{rid}/result">Evidence</a></td></tr>')
                    page='''<!doctype html><meta charset="utf-8"><title>Local Agentic RAG-VLM results</title>
<link rel="stylesheet" href="/experiment/local.css"><main class="experimentReport"><h1>Local platform task results</h1>
<p>Actual MuJoCo contact execution and real VLM/CLIP, launched through official BluPe Runner. No physical robot connection.</p>
<p>Three declared showcase cases; all results retained. This small set does not establish benchmark superiority.</p>
<table><thead><tr><th>Approach / task</th><th>Result</th><th>Grasp attempts</th><th>Failed attempts</th><th>Memory hits</th><th>Artifacts</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table>
<p>The geometric baseline uses the same real VLM identity perception and low-level feedback controller; it disables retrieval-based task reasoning and memory.</p><a href="/">Back to Playground</a></main>'''
                    return await self.bytes(send,page.encode(),b'text/html; charset=utf-8')
                if path in {'/experiment/local.css','/experiment/local.js'}:
                    if path.endswith('.css'):
                        data=(b'#agenticLocalBanner{padding:12px;background:#e1f3e9;text-align:center;color:#123}'
                            b'#agenticSimulationFrame{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;'
                            b'background:#151719;z-index:20;pointer-events:none}'
                            b'#liveViewer .liveStopwatch,#liveViewer #viewerCue{z-index:25}'
                            b'#agenticEvidence{background:#f6faf8;color:#123;padding:16px;border-radius:12px;margin:8px 0}'
                            b'#agenticEvidenceText{white-space:pre-wrap;font:13px/1.6 monospace}'
                            b'.experimentReport{max-width:1050px;margin:50px auto;font:16px/1.6 sans-serif;color:#16342b}'
                            b'.experimentReport table{border-collapse:collapse;width:100%}'
                            b'.experimentReport th,.experimentReport td{padding:14px;border:1px solid #b4c9bf;text-align:left}');mime=b'text/css'
                    else:
                        data=br'''async function agenticProgress(){try{const s=await(await fetch('/experiment/status')).json();
document.querySelector('#agenticSimulationStatus').textContent=(s.scenario||'Select a task')+' | '+s.stage+
' | motion: '+(s.motion?.stage||'waiting')+' | Agent: '+(s.runtime?.event||'preparing')+
(s.receipt?' | result: '+s.receipt.result+' | attempts: '+s.receipt.attempts+' | memory hits: '+s.receipt.memory_hits:'');
if(s.run_id)document.querySelector('#agenticSimulationFrame').src='/experiment/frame?t='+Date.now();
const p=s.last_plan,a=p?.action;const lines=['Verified: '+(s.verified_targets||[]).join(', '),
'Target: '+(a?.target||'waiting'),'Strategy source: '+(a?.source||'waiting'),
'Motor force cap: '+(a?.strategy?.force_n??'waiting')+' N (simulation)',
'Constraint: '+(a?.constraint?.reason||'none'),
'VLM review: '+(p?.vlm_thought?.parsed?.decision||'not available'),
'Recent: '+(s.recent_events||[]).map(e=>e.event).join(' -> ')];
document.querySelector('#agenticEvidenceText').textContent=lines.join('\n');
}catch(e){}}agenticProgress();setInterval(agenticProgress,750);''';mime=b'application/javascript'
                    return await self.bytes(send,data,mime)
                current=state.status();run_id=current.get('run_id')
                selected=re.fullmatch(r'/experiment/run/([0-9a-f]{32})/(result|video)',path)
                if selected:run_id=selected[1];path='/experiment/'+selected[2]
                if not run_id:raise RequestError(404,'No simulation has started')
                folder=state.output/run_id
                if path=='/experiment/result':return await self.json(send,200,read_json(folder/'simulation/result.json',current))
                if path=='/experiment/frame':
                    file=folder/'live/frame.jpg'
                    if not file.exists():raise RequestError(404,'Simulator is preparing its first frame')
                    return await self.bytes(send,file.read_bytes(),b'image/jpeg')
                if path=='/experiment/video':
                    file=folder/'simulation/rollout.mp4'
                    if not file.exists():raise RequestError(404,'Video is written after task completion')
                    return await self.video(send,file,headers.get('range'))
                raise RequestError(404,'Unknown local experiment route')
            return await super().http(scope,receive,send)
        async def bytes(self,send,data,mime):
            await send({'type':'http.response.start','status':200,'headers':[
                (b'content-type',mime),(b'content-length',str(len(data)).encode()),(b'cache-control',b'no-store')]})
            await send({'type':'http.response.body','body':data})
        async def video(self,send,path,range_header):
            size=path.stat().st_size;start=0;end=size-1;status=200
            if range_header:
                match=re.fullmatch(r'bytes=(\d+)-(\d*)',range_header)
                if match:
                    start=int(match[1]);end=min(size-1,int(match[2])) if match[2] else size-1;status=206
                if not match or not 0<=start<=end<size:raise RequestError(416,'Unsupported video range')
            headers=[(b'content-type',b'video/mp4'),(b'content-length',str(end-start+1).encode()),(b'accept-ranges',b'bytes')]
            if status==206:headers.append((b'content-range',f'bytes {start}-{end}/{size}'.encode()))
            await send({'type':'http.response.start','status':status,'headers':headers})
            with path.open('rb') as stream:
                stream.seek(start);remaining=end-start+1
                while remaining:
                    chunk=stream.read(min(256*1024,remaining));remaining-=len(chunk)
                    await send({'type':'http.response.body','body':chunk,'more_body':remaining>0})
    origin=f'http://127.0.0.1:{port}'
    def factory(case_key):
        def create(model,directory):
            if model!=CASES[case_key]['model']:raise RequestError(400,'Wrong approach')
            return SimulationTaskProvider(provider,Path(source_root),state,case_key,python)
        return create
    app=Playground(public_origin=origin,session_api=None,camera_origin=origin,development=True,
        hardware_control=False,api_factory=LocalFixture,share_conversation=False,
        default_provider='agentic-stack',extra_providers={name:factory(name) for name in CASES},
        recording_root=state.output/'runner_recordings',
        robots=[{'id':'yam-1','name':'YAM — LOCAL MUJOCO','url':origin+'/'}])
    return app,state
