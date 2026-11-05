"""Recommended protected-neighbor task through the PRODUCTION BluPe provider.

Official Runner -> official HTTP/WebSocket SDK -> loopback Session API -> actual
MuJoCo joint controls -> camera/verification. No transport/planner/model doubles.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
from urllib.request import urlopen
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agentic_rag_vlm.platform.local_runner import import_official_runner
from agentic_rag_vlm.platform.runner_provider import load_provider
from agentic_rag_vlm.platform.config import PlatformConfig
from agentic_rag_vlm.runtime import TaskSpec,Subgoal
from agentic_rag_vlm.experience import atomic_json
from agentic_rag_vlm.simulation.protocol_gateway import PhysicsWorker,make_gateway,recommended_provider


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runner-checkout',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidate-config',type=Path,help='optional RGB candidate config for local production-path qualification')
    p.add_argument('--seed',type=int,default=1000);p.add_argument('--port',type=int,default=18792)
    args=p.parse_args();os.environ.setdefault('MUJOCO_GL','egl')
    output=args.output.resolve()
    if output.exists():p.error('use a new output directory; all attempts are retained')
    worker=PhysicsWorker(output,seed=args.seed)
    origin=f'http://127.0.0.1:{args.port}'
    app,backend,requests=make_gateway(args.runner_checkout,worker,origin)
    import uvicorn
    server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=args.port,log_level='warning'))
    thread=threading.Thread(target=server.run,daemon=True);thread.start()
    for _ in range(100):
        if server.started:break
        time.sleep(.05)
    else:raise RuntimeError('local gateway startup failed')
    controller_module,session_module=import_official_runner(args.runner_checkout)
    sdk=session_module.HttpSessionAPI(origin,supports_trajectories=True,robot_id='yam-local-1')
    sdk.local_simulation_only=True
    provider=recommended_provider(args.config,worker,origin);task=provider.task
    if args.candidate_config:
        from agentic_rag_vlm.platform.candidates import load_candidate_provider
        provider.candidate_provider=load_candidate_provider(args.candidate_config)
    controller=controller_module.RunnerController(sdk,robot_id='yam-local-1',hardware_control_enabled=False,
        share_conversation=False,recording_root=output/'runner_recordings')
    print('Starting real production provider through official HTTP/WebSocket SDK; local physics only',flush=True)
    controller.join_and_run(provider,task.instruction)
    deadline=time.monotonic()+600;last=None
    while controller._worker.is_alive() and time.monotonic()<deadline:
        state=controller.status();stage=(state['status'],state.get('error'),len(backend.trajectory_log))
        if stage!=last:print(json.dumps({'controller':stage[0],'error':stage[1],'trajectory_packets':stage[2]}),flush=True);last=stage
        time.sleep(.3)
    if controller._worker.is_alive():controller.stop('local_validation_timeout');controller._worker.join(5)
    state=controller.status();server.should_exit=True;thread.join(5)
    physical=worker.rpc(worker.finalize,timeout=60)
    results=list((output/'agent_runs').glob('*/result.json'))
    agent=json.loads(results[-1].read_text()) if results else None
    receipt={'task':task.instruction,'seed':args.seed,'uses_production_provider':True,'uses_official_http_websocket_sdk':True,
        'transport_replaced':False,'native_kinematics_replaced':False,'model_test_double':False,
        'framework_outcome':agent.get('outcome') if agent else None,'controller_error':state.get('error'),
        'trajectory_packets':len(backend.trajectory_log),'trajectory_waypoints':sum(len(p['waypoints']) for p in backend.trajectory_log),
        'physical':physical,'public_robot_contacted':False,'execution_environment':'simulation',
        'success':bool(agent and agent['outcome']=='complete' and physical['simulation_success'] and not state.get('error'))}
    atomic_json(output/'receipt.json',receipt);atomic_json(output/'controller_status.json',state)
    atomic_json(output/'gateway_requests.json',requests);atomic_json(output/'trajectory_packets.json',backend.trajectory_log)
    atomic_json(output/'trajectory_events.json',backend.trace_events)
    print(json.dumps(receipt,indent=2));return 0 if receipt['success'] else 1


if __name__=='__main__':raise SystemExit(main())
