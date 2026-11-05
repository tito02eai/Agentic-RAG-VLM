"""Local simulation task ownership and read-only progress contracts."""
import json
import asyncio
import shutil
import subprocess
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import numpy as np
import pytest

from agentic_rag_vlm.platform.simulation_runner import SimulationTaskProvider,SimulationState,CASES
from agentic_rag_vlm.simulation.live import LiveSimulationObserver
from agentic_rag_vlm.runtime import AgenticRuntime,TaskSpec,Subgoal
from tests.test_platform_provider import fixture as agent_fixture
from tests.test_official_runner import upstream


def test_runtime_observer_receives_copy_without_mutating_decisions(tmp_path):
    provider,*_=agent_fixture(tmp_path)
    from agentic_rag_vlm.paper import PaperAgent
    writer=LiveSimulationObserver(tmp_path/'live')
    def observe(event):
        writer.event(event)
        event['payload']['task']['subgoals'][0]['target']='tampered'
    runtime=AgenticRuntime(PaperAgent(provider.knowledge_path,provider.client),event_sink=observe)
    runtime.start(TaskSpec('test','test',(Subgoal('cube','tray'),)))
    assert runtime.trace()[0]['payload']['task']['subgoals'][0]['target']=='cube'
    assert json.loads((tmp_path/'live/runtime.json').read_text())['payload']['task']['subgoals'][0]['target']=='cube'


def test_live_frames_are_readonly_and_atomically_available(tmp_path):
    writer=LiveSimulationObserver(tmp_path)
    frame=np.zeros((40,50,3),dtype=np.uint8);original=frame.copy()
    writer.frame(frame,'lift',1.)
    assert np.array_equal(frame,original)
    from PIL import Image
    with Image.open(tmp_path/'frame.jpg') as image:assert image.size==(50,40)
    assert json.loads((tmp_path/'motion.json').read_text())['stage']=='lift'


def test_simulation_cannot_adopt_public_sdk(tmp_path):
    provider,*_=agent_fixture(tmp_path)
    adapter=SimulationTaskProvider(provider,Path(__file__).resolve().parents[1],SimulationState(tmp_path/'runs'),'agentic-flat')
    with pytest.raises(ValueError,match='public robot'):adapter.validate_session_admission(SimpleNamespace())


def test_stop_terminates_local_child_and_never_records_success(tmp_path,monkeypatch):
    import agentic_rag_vlm.platform.simulation_runner as module
    provider,*_=agent_fixture(tmp_path)
    state=SimulationState(tmp_path/'runs')
    adapter=SimulationTaskProvider(provider,Path(__file__).resolve().parents[1],state,'agentic-flat')
    started=threading.Event();stop=threading.Event()
    def command(*args):
        started.set()
        return [sys.executable,'-c','import time;time.sleep(20)']
    monkeypatch.setattr(module,'simulation_command',command)
    results=[]
    thread=threading.Thread(target=lambda:results.append(adapter.run_session(CASES['agentic-flat']['task'],{},
        SimpleNamespace(local_validation_only=True),{},stop.is_set)))
    thread.start();assert started.wait(2)
    deadline=time.monotonic()+2
    while state.status()['stage']!='running_simulation' and time.monotonic()<deadline:time.sleep(.01)
    stop.set();thread.join(5)
    assert not thread.is_alive() and results[0]['state']=='cancelled'
    receipt=state.runs()[0]
    assert receipt['simulation_success'] is False and receipt['result']=='cancelled'
    assert not state.lock.locked()


def test_child_failure_retains_error_instead_of_becoming_task_success(tmp_path,monkeypatch):
    import agentic_rag_vlm.platform.simulation_runner as module
    provider,*_=agent_fixture(tmp_path)
    state=SimulationState(tmp_path/'runs')
    adapter=SimulationTaskProvider(provider,Path(__file__).resolve().parents[1],state,'agentic-stack')
    monkeypatch.setattr(module,'simulation_command',lambda *args:[sys.executable,'-c','raise RuntimeError("injected simulator failure")'])
    with pytest.raises(RuntimeError,match='failed'):
        adapter.run_session(CASES['agentic-stack']['task'],{},SimpleNamespace(local_validation_only=True),{},lambda:False)
    receipt=state.runs()[0]
    assert receipt['exit_code']!=0 and not receipt['simulation_success']
    assert 'injected simulator failure' in (state.output/receipt['run_id']/'process.log').read_text()


def test_official_app_serves_valid_progress_script_and_report(upstream,tmp_path):
    from agentic_rag_vlm.platform.simulation_runner import make_simulation_playground
    import os
    provider,*_=agent_fixture(tmp_path)
    app,_=make_simulation_playground(os.environ['BLUPE_RUNNER_CHECKOUT'],provider,
        Path(__file__).resolve().parents[1],tmp_path/'site',port=18792)
    async def request(path):
        messages=[]
        async def receive():return {'type':'http.request','body':b''}
        async def send(message):messages.append(message)
        await app({'type':'http','method':'GET','path':path,'query_string':b'',
            'headers':[(b'host',b'127.0.0.1:18792')]},receive,send)
        assert messages[0]['status']==200
        return b''.join(m.get('body',b'') for m in messages[1:])
    javascript=asyncio.run(request('/experiment/local.js'))
    if shutil.which('node'):
        subprocess.run(['node','--check','-'],input=javascript,capture_output=True,check=True)
    report=asyncio.run(request('/experiment/report')).decode()
    assert 'Local platform task results' in report and 'No physical robot connection' in report
