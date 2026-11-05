"""Opt-in tests against an unmodified, locally checked out BluPe runner.

Set BLUPE_RUNNER_CHECKOUT to the official repository root. All sessions here
are local fixtures; no BluPe endpoint, capability, queue or robot is contacted.
"""
import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

import pytest

from agentic_rag_vlm.platform.runner_provider import RunnerBoundClient,load_provider
from agentic_rag_vlm.platform.blupe_client import SessionHandle,BluPeAPIError
from tests.test_platform_provider import fixture as agent_fixture
from tests.test_delivery import measured_test_profile,calibration,native_fixture


@pytest.fixture(scope='module')
def upstream():
    path=os.environ.get('BLUPE_RUNNER_CHECKOUT')
    if not path:pytest.skip('set BLUPE_RUNNER_CHECKOUT for official-runner integration tests')
    root=Path(path).resolve()/'codex-runner'
    from scripts.check_runner_compatibility import inspect_runner
    assert inspect_runner(root/'src/remote_yam')['compatible']
    sys.path[:0]=[str(root),str(root/'src')]
    import remote_yam.controller as controller
    import remote_yam.session as session
    assert Path(controller.__file__).resolve().is_relative_to(root)
    return controller.RunnerController,session


def run_controller(controller,provider,prompt,timeout=8):
    controller.join_and_run(provider,prompt)
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        worker=controller._worker
        if worker is not None and not worker.is_alive():return controller.status()
        time.sleep(.01)
    controller.stop('local_test_timeout')
    pytest.fail('official runner worker did not terminate within test deadline')


def configured_agent(tmp_path,monkeypatch):
    import agentic_rag_vlm.platform.runner_provider as module
    provider,transport,geometry,task,vlm=agent_fixture(tmp_path)
    profile=measured_test_profile();profile['feature_heights_m']=geometry['feature_heights_m']
    profile['grasp_offsets_base_m']={n:[0,0,0] for n in geometry['feature_heights_m']}
    provider.profile=profile;provider.task=task
    cal=calibration();cal['kinematics']=native_fixture()['kinematics']
    monkeypatch.setattr(module.RunnerBoundClient,'get_calibration',lambda *args:cal)
    monkeypatch.setattr(module,'RemotePlatformTransport',lambda *args,**kwargs:transport)
    return provider,transport,task,vlm


def test_official_controller_runs_complete_agent_and_owns_stop(upstream,tmp_path,monkeypatch):
    Controller,session=upstream
    api=session.MockSessionAPI()
    provider,transport,task,vlm=configured_agent(tmp_path,monkeypatch)
    controller=Controller(api,hardware_control_enabled=False,share_conversation=False)
    state=run_controller(controller,provider,task.instruction)
    assert state['error'] is None and state['status']=='stopped'
    assert controller._last_stop_reason=='policy_complete'
    assert len(api.create_requests)==1 and len(api._sessions)==1
    assert not api.action_log and not api.trajectory_log  # Commands below are transport doubles.
    assert [c.command_type for c in transport.commands]==['pick','place','pick','place']
    result=json.loads(next((tmp_path/'runs').glob('*/result.json')).read_text())
    assert result['outcome']=='complete' and result['hardware_success'] is None
    planned=[e['payload'] for e in result['trace'] if e['event']=='subgoal_planned']
    assert [p['action']['source'] for p in planned]==['haa_rag','episodic_memory']
    assert len(vlm.calls)>2 and not provider._active


@pytest.mark.parametrize('failure',['no_geometry','invalid_geometry','model_failure','no_trajectories'])
def test_official_runner_rejects_before_queue(upstream,tmp_path,monkeypatch,failure):
    Controller,session=upstream
    api=session.MockSessionAPI()
    provider,transport,task,_=configured_agent(tmp_path,monkeypatch)
    if failure=='no_geometry':provider.profile=None
    elif failure=='invalid_geometry':provider.profile['joint_limits_deg']=[[10,-10]]*6
    elif failure=='model_failure':
        def unavailable():raise RuntimeError('injected model unavailable')
        provider.prepare=unavailable
    else:api.supports_trajectories=False
    state=run_controller(Controller(api,hardware_control_enabled=False,share_conversation=False),provider,task.instruction)
    assert state['error'] and state['status']=='failed'
    assert not api.create_requests and not api.action_log and not transport.commands


def test_official_runner_stop_cancels_task_before_any_dispatch(upstream,tmp_path,monkeypatch):
    Controller,session=upstream
    provider,transport,task,_=configured_agent(tmp_path,monkeypatch)
    entered=threading.Event();cancelled=threading.Event()
    def waiting(prompt,observation,api,context,is_cancelled):
        entered.set()
        deadline=time.monotonic()+4
        while not is_cancelled() and time.monotonic()<deadline:time.sleep(.01)
        assert is_cancelled();cancelled.set()
        return {'state':'cancelled'}
    provider.run_session=waiting
    api=session.MockSessionAPI()
    controller=Controller(api,hardware_control_enabled=False,share_conversation=False)
    controller.join_and_run(provider,task.instruction)
    assert entered.wait(3)
    controller.stop('user_requested')
    controller._worker.join(3)
    assert cancelled.is_set() and not controller._worker.is_alive()
    assert len(api.create_requests)==1 and not api.action_log and not transport.commands
    assert controller.status()['error'] is None


@contextmanager
def local_gateway():
    """Gateway-shaped HTTP fixture; not a robot/physics simulator.

    Deliberately implements lease_id and terminal trace results, which the
    upstream in-memory mock does not expose in its HTTP snapshot/trace shape.
    """
    state={'calls':[],'status':'running','episode_id':'local_episode','lease_id':'lease_new',
           'cadence_hz':10.,'capability':'LOCAL_FIXTURE_TOKEN','results':[]}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def handle_request(self):
            raw=self.rfile.read(int(self.headers.get('Content-Length',0)))
            body=json.loads(raw) if raw else None
            authorized=self.headers.get('Authorization')=='Bearer '+state['capability']
            # Record presence only; never serialize even the synthetic token.
            state['calls'].append({'method':self.command,'path':self.path,'authorized':authorized,'body':body})
            private=self.path.startswith('/v1/sessions/local_session')
            if private and not authorized:code,result=401,{'error':'unauthorized'}
            elif self.path=='/v1/sessions' and self.command=='POST':
                code,result=200,{'session_id':'local_session','session_capability':state['capability'],
                    'status':'running','episode_id':state['episode_id'],'lease_id':'lease_old'}
            elif self.path=='/v1/sessions/local_session' and self.command=='GET':
                code,result=200,{k:state[k] for k in ('status','episode_id','lease_id','cadence_hz')}
                if 'active_trajectory' in state:result['active_trajectory']=state['active_trajectory']
            elif self.path.endswith('/trajectories') and self.command=='POST':
                assert body['lease_id']==state['lease_id']
                assert body['episode_id']==state['episode_id']
                state['results'].append({'trajectory_id':body['trajectory_id'],'status':'completed'})
                code,result=202,{'accepted':True,'trajectory_id':body['trajectory_id']}
            elif self.path.endswith('/episode/trace'):code,result=200,{'trajectory_results':state['results']}
            else:code,result=404,{'error':'not_found'}
            data=json.dumps(result).encode();self.send_response(code)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)))
            self.end_headers();self.wfile.write(data)
        do_GET=handle_request
        do_POST=handle_request
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield 'http://127.0.0.1:'+str(server.server_port),state
    finally:server.shutdown();server.server_close();thread.join(3)


def sdk_context(session,url):
    api=session.HttpSessionAPI(url,supports_trajectories=True,websocket_factory=lambda *a,**k:None)
    created=api.create_session('local transport contract')
    context={k:created[k] for k in ('session_id','episode_id','lease_id')};context['robot_id']='yam-1'
    client=RunnerBoundClient(url,api,context)
    handle=SessionHandle('local_session','','yam-1','running','',episode_id=context['episode_id'],lease_id=context['lease_id'])
    packet={'trajectory_id':'local_trajectory','cadence_hz':10.,'waypoints':[
        {'step_id':0,'left_joints_deg':[0.]*6,'right_joints_deg':[0.]*6,'left_gripper':1.,'right_gripper':1.}]}
    return api,client,handle,packet


def test_official_http_sdk_sends_rotated_lease_and_reads_completion(upstream):
    _,session=upstream
    with local_gateway() as (url,state):
        api,client,handle,packet=sdk_context(session,url)
        assert client.submit_trajectory_packet(handle,packet)['accepted']
        assert client.wait_for_trajectory(handle,'local_trajectory',timeout_s=1)['status']=='completed'
        writes=[c for c in state['calls'] if c['path'].endswith('/trajectories')]
        assert len(writes)==1 and writes[0]['body']['lease_id']=='lease_new' and writes[0]['authorized']
        assert handle.capability==''
        assert 'LOCAL_FIXTURE_TOKEN' not in json.dumps(state['calls'])


@pytest.mark.parametrize('failure',['episode','cadence','packet_cadence','stopped','preparing','active_trajectory','cancelled'])
def test_official_http_sdk_rejects_invalid_context_before_write(upstream,failure):
    _,session=upstream
    with local_gateway() as (url,state):
        api,client,handle,packet=sdk_context(session,url)
        if failure=='episode':state['episode_id']='other_episode'
        elif failure=='cadence':state['cadence_hz']=20.
        elif failure=='packet_cadence':packet['cadence_hz']=20.
        elif failure=='stopped':state['status']='stopped'
        elif failure=='preparing':state['status']='preparing'
        elif failure=='active_trajectory':
            # Included by the HTTP fixture when present.
            state['active_trajectory']={'trajectory_id':'previous_in_flight'}
        else:client.cancelled=lambda:True
        with pytest.raises(BluPeAPIError):client.submit_trajectory_packet(handle,packet)
        assert not any(c['path'].endswith('/trajectories') for c in state['calls'])


def test_official_playground_registers_our_provider_without_external_session(upstream,tmp_path,monkeypatch):
    # Calls the official ASGI app/session/Run routes rather than a recreated UI.
    _,session=upstream
    from playground import HostedRunner
    provider,transport,task,_=configured_agent(tmp_path,monkeypatch)
    apis=[]
    def api_factory():
        api=session.MockSessionAPI();apis.append(api);return api
    app=HostedRunner(public_origin='http://127.0.0.1:18791',session_api=None,
        camera_origin='http://127.0.0.1:18791',development=True,hardware_control=False,
        api_factory=api_factory,share_conversation=False,
        extra_providers={'agentic-rag-vlm':lambda model,directory:provider})
    async def requests():
        cookie='';csrf=''
        async def call(path,payload=None):
            nonlocal cookie,csrf
            messages=[]
            async def receive():return {'type':'http.request','body':json.dumps(payload or {}).encode()}
            async def send(message):messages.append(message)
            headers={'host':'127.0.0.1:18791','origin':app.origin,'cookie':cookie,
                     'content-type':'application/json','x-yam-runner-token':csrf}
            await app({'type':'http','method':'POST' if payload is not None else 'GET','path':path,
                       'query_string':b'','headers':[(k.encode(),v.encode()) for k,v in headers.items()]},receive,send)
            head=messages[0];h=dict(head['headers'])
            if b'set-cookie' in h:cookie=h[b'set-cookie'].decode().split(';')[0]
            result=json.loads(b''.join(m.get('body',b'') for m in messages[1:]))
            if 'csrf' in result:csrf=result['csrf']
            return head['status'],result
        try:
            code,setup=await call('/api/session',{})
            assert code==200 and setup['simulation'] is True
            code,result=await call('/api/run',{'provider':'agentic-rag-vlm','model':'Agentic RAG-VLM',
                'prompt':task.instruction,'runner_name':'Local integration fixture','share_conversation':False})
            assert code==200,result
            controller=next(iter(app.visitors.values())).controller
            await asyncio.to_thread(controller._worker.join,8)
            code,status=await call('/api/status')
            assert code==200 and status['provider']['model']=='Agentic RAG-VLM'
            assert status['error'] is None and status['status']=='stopped'
            assert len(transport.commands)==4
        finally:
            for visitor in app.visitors.values():await asyncio.to_thread(visitor.close)
            import shutil
            shutil.rmtree(app.root,ignore_errors=True)
    asyncio.run(requests())


@pytest.mark.parametrize('case',['single','protected'])
def test_actual_models_through_official_controller_preview(upstream,tmp_path,case):
    path=os.environ.get('BLUPE_RECORDED_PREVIEW' if case=='single' else 'BLUPE_RECORDED_PROTECTED')
    if not path:pytest.skip('set recorded preview paths for real-model recorded-image validation')
    Controller,session=upstream
    payload=json.loads(Path(path).read_text())
    from agentic_rag_vlm.platform.local_runner import RecordedPreviewAdapter
    config=Path(__file__).resolve().parents[1]/'configs/blupe_provider.example.json'
    provider=load_provider(config);provider.prepare()
    adapter=RecordedPreviewAdapter(provider,payload,tmp_path);api=session.MockSessionAPI()
    api.local_validation_only=True
    controller=Controller(api,hardware_control_enabled=False,share_conversation=False)
    state=run_controller(controller,adapter,payload['task']['instruction'],timeout=180)
    assert state['error'] is None
    if case=='single':
        assert adapter.result['decision']=='propose'
        assert adapter.result['action']['source']=='haa_rag' and adapter.result['reasoner_calls']
    else:
        assert adapter.result['decision']=='stop' and adapter.result['action'] is None
        assert not adapter.result['reasoner_calls']
    assert adapter.result['execution_performed'] is False
    assert not api.action_log and not api.trajectory_log
    output=os.environ.get('BLUPE_VALIDATION_OUTPUT')
    if output:
        folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
        (folder/f'official_controller_real_{case}_preview.json').write_text(json.dumps(adapter.result,indent=2))
        (folder/f'official_controller_real_{case}_status.json').write_text(json.dumps(state,indent=2))


def test_recorded_preview_rejects_unmarked_session_before_queue(upstream,tmp_path):
    Controller,session=upstream
    from agentic_rag_vlm.platform.local_runner import RecordedPreviewAdapter
    provider,_,_,task,_=agent_fixture(tmp_path)
    adapter=RecordedPreviewAdapter(provider,{'task':{'instruction':task.instruction}},tmp_path)
    api=session.MockSessionAPI()  # Deliberately lacks the local-only marker.
    state=run_controller(Controller(api,share_conversation=False),adapter,task.instruction)
    assert state['status']=='failed' and 'local mock API' in state['error']
    assert not api.create_requests and not api.action_log
