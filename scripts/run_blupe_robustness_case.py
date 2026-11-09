"""Local production-path fault trials, never a public BluPe robot client.

Faults affect simulated actuator commands, physical external forces or real
loopback HTTP/WS responses. The production provider/IK/model remain unchanged.
Each invocation is a fresh process/session and retains every result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from agentic_rag_vlm.experience import atomic_json
from agentic_rag_vlm.platform.local_runner import import_official_runner
from agentic_rag_vlm.runtime import TaskSpec, Subgoal
from agentic_rag_vlm.simulation.protocol_gateway import PhysicsWorker, make_gateway, recommended_provider


CASES = ('control', 'empty_grasp', 'slip', 'memory_moved', 'memory_failed_reuse',
         'ack_lost_once', 'ack_lost_persistent', 'feedback_outage', 'ws_reconnect', 'operator_stop')


class FaultWorker(PhysicsWorker):
    """Fault/evaluator instrumentation; no simulator truth reaches the Agent."""
    def __init__(self, output, *, seed, case):
        self.case = case
        self.packet_count = 0
        self.faults = []
        self.executed_waypoints = []
        self.packet_active = False
        super().__init__(output, seed=seed, multiple=case.startswith('memory_'))
        self.rpc(self.install_faults)

    def install_faults(self):
        if self.multiple:
            self.profile['feature_heights_m']['cube2'] = .036
            self.profile['grasp_offsets_base_m']['cube2'] = [.014, 0., -.012]
            atomic_json(self.output/'profile.json', self.profile)
        original_step = self.sim.step
        self.grip_fault = False

        def step(count):
            for _ in range(count):
                eligible = self.packet_active and (
                    self.case == 'empty_grasp' and self.packet_count == 1
                    or self.case == 'memory_failed_reuse' and self.packet_count == 3)
                if eligible and self.sim.data.ctrl[6] < .02:
                    if not self.grip_fault:
                        self.faults.append({'kind': 'actuator_forced_open', 'packet': self.packet_count,
                                            'simulation_time_s': float(self.sim.data.time)})
                    self.grip_fault = True
                if (self.case == 'slip' and self.packet_active and self.packet_count == 1
                        and not self.grip_fault and self.sim.bilateral_lift_seconds >= .15):
                    self.grip_fault = True
                    self.faults.append({'kind': 'gripper_opens_after_observed_bilateral_lift',
                        'simulation_time_s': float(self.sim.data.time),
                        'bilateral_lift_seconds_before_fault': self.sim.bilateral_lift_seconds,
                        'target_z_before_fault_m': float(self.sim.data.body('target_cube').xpos[2])})
                if self.packet_active and self.grip_fault:
                    self.sim.data.ctrl[6] = .0375
                original_step(1)
        self.sim.step = step

    def execute(self, waypoints, progress):
        self.packet_count += 1
        self.packet_active = True
        def recorded(step):
            self.executed_waypoints.append({'packet': self.packet_count, 'step_id': step})
            progress(step)
            if self.case == 'operator_stop':
                time.sleep(.01)
        try:
            result = super().execute(waypoints, recorded)
            if self.case == 'memory_moved' and self.packet_count == 2:
                # External disturbance acts through force + integration, not qpos.
                # It moves the next target after the first place, before verify.
                body = self.sim.model.body('target_cube2').id
                before = self.sim.data.body(body).xpos.copy()
                self.sim.data.xfrc_applied[body, :3] = [0., .9, 0.]
                self.sim.step(50)
                self.sim.data.xfrc_applied[body] = 0.
                self.sim.step(400)
                after = self.sim.data.body(body).xpos.copy()
                self.faults.append({'kind': 'external_force_moves_pending_object',
                    'body': 'target_cube2', 'before_xyz_m': before.tolist(), 'after_xyz_m': after.tolist(),
                    'displacement_m': float(np.linalg.norm(after-before)),
                    'force_n': [0., .9, 0.], 'duration_s': .1, 'qpos_assigned': False})
                self._capture()
            return result
        finally:
            self.packet_active = False

    def finalize(self):
        result = super().finalize()
        second = None
        if self.multiple:
            sim = self.sim
            xyz = sim.data.body('target_cube2').xpos.copy()
            contacts = [set((sim.model.geom(c.geom1).name, sim.model.geom(c.geom2).name)) for c in sim.data.contact]
            error = float(np.linalg.norm(xyz-np.array([.19, .17, .018])))
            second = {'position_xyz_m': xyz.tolist(), 'destination_error_m': error,
                'destination': error < .025,
                'supported': {'table', 'target_geom2'} in contacts,
                'released': not any('target_geom2' in contact and any(
                    name and name.endswith('collision') for name in contact) for contact in contacts),
                'bilateral_lift_seconds': sim.lift_seconds_by_object['target_cube2']}
        if second is not None:
            result['second_target'] = second
            result['simulation_success'] &= bool(second['destination'] and second['supported'] and second['released']
                                                   and second['bilateral_lift_seconds'] > .5)
            atomic_json(self.output/'physical_result.json', result)
        atomic_json(self.output/'faults.json', self.faults)
        atomic_json(self.output/'executed_waypoints.json', self.executed_waypoints)
        return result


class FaultGateway:
    """ASGI wire faults after real gateway handling; no fabricated successes."""
    def __init__(self, app, case):
        self.app, self.case = app, case
        self.requests = []
        self.faults = []
        self.first_tid = None
        self.ws_count = 0
        self.submitted = False

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'websocket':
            self.ws_count += 1
            if self.case == 'ws_reconnect' and self.ws_count == 1:
                await receive()
                await send({'type': 'websocket.accept'})
                await send({'type': 'websocket.close', 'code': 1012, 'reason': 'declared local fault'})
                self.faults.append({'kind': 'websocket_closed_before_initial_observation'})
                return
            return await self.app(scope, receive, send)
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        path, method = scope['path'], scope['method']
        chunks = []
        while True:
            chunk = await receive()
            chunks.append(chunk)
            if not chunk.get('more_body'):
                break
        raw = b''.join(c.get('body', b'') for c in chunks)
        payload = json.loads(raw) if raw else {}
        tid = payload.get('trajectory_id')
        if tid:
            self.submitted = True
            if self.first_tid is None:
                self.first_tid = tid
        entry = {'method': method, 'path': path, 'trajectory_id': tid,
                 'body_sha256': hashlib.sha256(raw).hexdigest()}
        self.requests.append(entry)
        if (self.case == 'feedback_outage' and self.submitted and method == 'GET'
                and '/v1/sessions/' in path and not path.endswith('/calibration')):
            return await self.unavailable(send, 'feedback_read_unavailable')
        cursor = iter(chunks)
        async def replay_receive():
            return next(cursor)
        responses = []
        async def capture_send(message):
            responses.append(message)
        await self.app(scope, replay_receive, capture_send)
        lose_ack = tid == self.first_tid and tid is not None and (
            self.case == 'ack_lost_persistent'
            or self.case == 'ack_lost_once' and not self.faults)
        if lose_ack:
            self.faults.append({'kind': 'accepted_packet_reply_lost', 'trajectory_id': tid,
                                'body_sha256': entry['body_sha256']})
            return await self.unavailable(send, 'accepted_packet_reply_lost', record=False)
        for response in responses:
            await send(response)

    async def unavailable(self, send, kind, record=True):
        if record:
            self.faults.append({'kind': kind})
        raw = json.dumps({'error': {'code': 'declared_local_fault', 'message': kind, 'status': 503}}).encode()
        await send({'type': 'http.response.start', 'status': 503,
                    'headers': [(b'content-type', b'application/json'), (b'content-length', str(len(raw)).encode())]})
        await send({'type': 'http.response.body', 'body': raw})


def assess(case, worker, gateway, backend, result, physical, state, stop_record):
    trace = result.get('trace', [])
    receipts = result.get('receipts', [])
    commands = [r['payload'] for r in receipts if r['kind'] == 'command']
    planned = [e['payload'] for e in trace if e['event'] == 'subgoal_planned']
    complete = result.get('outcome') == 'complete'
    checks = {'no_public_robot': physical.get('physical_robot_contacted') is False,
              'agent_result_saved': bool(result), 'model_calls_saved': bool(result.get('reasoner_calls'))}
    for key in ('protected_stable', 'no_protected_robot_contact', 'no_arm_table_contact', 'no_self_contact'):
        checks[key] = physical['checks'][key]
    detail = {}
    if case in ('control', 'ack_lost_once', 'ws_reconnect', 'memory_moved'):
        checks.update(task_completed=complete, independent_physics_passed=physical['simulation_success'])
    else:
        checks.update(agent_stopped=not complete and result.get('outcome') == 'safe_stop',
                      no_task_completed_event=not any(e['event'] == 'task_completed' for e in trace))
    if case in ('empty_grasp', 'slip', 'memory_failed_reuse'):
        failed = [c for c in commands if c['receipt'].get('reason') == 'retention_unconfirmed']
        checks['retention_failure_observed'] = bool(failed)
        checks['physical_fault_triggered'] = bool(worker.faults)
        checks['no_place_after_failed_pick'] = len(backend.trajectory_log) == (3 if worker.multiple else 1)
        if case == 'slip':
            checks['fault_after_real_lift'] = any(f.get('bilateral_lift_seconds_before_fault', 0) >= .15 for f in worker.faults)
    if case.startswith('memory_'):
        hits = [p for p in planned if p['action']['source'] == 'episodic_memory']
        checks['verified_first_task_before_reuse'] = bool(hits) and any(
            e['event'] == 'subgoal_verified' and e['payload'].get('success') for e in trace)
        checks['second_action_uses_memory'] = bool(hits)
        detail['memory_actions'] = hits
        if case == 'memory_moved':
            moved = [f for f in worker.faults if f['kind'] == 'external_force_moves_pending_object']
            checks['pending_object_physically_moved'] = bool(moved and moved[0]['displacement_m'] > .03)
            checks['scene_change_replanned'] = any(e['event'] == 'l3_replan_started' for e in trace)
            if moved and hits:
                xyz = np.array(hits[0]['action']['position_xyz_m'])
                fresh = np.array(moved[0]['after_xyz_m']); old = np.array(moved[0]['before_xyz_m'])
                fresh_error = float(np.linalg.norm(xyz[:2]-fresh[:2]))
                old_error = float(np.linalg.norm(xyz[:2]-old[:2]))
                checks['uses_fresh_position_not_old_position'] = fresh_error < .015 and old_error > .03
                detail.update(fresh_position_error_m=fresh_error, previous_position_error_m=old_error)
        else:
            checks['failed_memory_invalidated'] = 'cube' not in result.get('memory', {}) and any(
                e['event'] == 'memory_invalidation' for e in trace)
    if case.startswith('ack_lost'):
        posts = [r for r in gateway.requests if r['trajectory_id'] == gateway.first_tid]
        checks['retry_exercised'] = len(posts) >= 2
        checks['retry_same_id_and_body'] = len({r['body_sha256'] for r in posts}) == 1
        checks['no_duplicate_physical_execution'] = len([p for p in backend.trajectory_log if p['trajectory_id'] == gateway.first_tid]) == 1
        detail['first_packet_submission_attempts'] = len(posts)
        if case == 'ack_lost_persistent':
            checks['no_second_packet_under_uncertainty'] = len(backend.trajectory_log) == 1
    if case == 'feedback_outage':
        checks['feedback_fault_triggered'] = bool(gateway.faults)
        checks['no_second_packet_without_completion'] = len(backend.trajectory_log) == 1
    if case == 'ws_reconnect':
        checks['websocket_actually_reconnected'] = gateway.ws_count >= 2 and bool(gateway.faults)
        checks['same_single_session'] = len(backend._sessions) == 1
    if case == 'operator_stop':
        checks['stop_during_real_waypoints'] = stop_record.get('executed_before_stop', 0) > 0
        checks['no_packet_after_stop'] = len(backend.trajectory_log) == stop_record.get('packets_before_stop') == 1
        checks['physical_executor_cancelled'] = worker.cancel.is_set() and any(
            e['payload']['status'] == 'aborted' for e in backend.trace_events)
        checks['controller_stopped'] = state['status'] == 'stopped'
    return {'passed': all(checks.values()), 'checks': checks, 'details': detail,
            'case': case, 'agent_outcome': result.get('outcome'),
            'task_success': bool(complete and physical['simulation_success']),
            'trajectory_packets': len(backend.trajectory_log), 'faults': worker.faults+gateway.faults,
            'scope': 'local production-path engineering fault qualification; not hardware/benchmark success rate',
            'physical_robot_contacted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner-checkout', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--candidate-config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--seed', type=int, default=1010)
    parser.add_argument('--port', type=int, default=18795)
    args = parser.parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    output = args.output.resolve()
    if output.exists():
        parser.error('use a fresh directory; never overwrite failures')
    worker = FaultWorker(output, seed=args.seed, case=args.case)
    origin = f'http://127.0.0.1:{args.port}'
    app, backend, requests = make_gateway(args.runner_checkout, worker, origin)
    gateway = FaultGateway(app, args.case)
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(gateway, host='127.0.0.1', port=args.port, log_level='warning'))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic()+15
    while not server.started and time.monotonic() < deadline:
        time.sleep(.05)
    if not server.started:
        raise RuntimeError('local gateway did not start')
    controller_module, session_module = import_official_runner(args.runner_checkout)
    sdk = session_module.HttpSessionAPI(origin, supports_trajectories=True, robot_id='yam-local-1', timeout_s=3.)
    sdk.local_simulation_only = True
    provider = recommended_provider(args.config, worker, origin)
    if args.candidate_config:
        from agentic_rag_vlm.platform.candidates import load_candidate_provider
        provider.candidate_provider = load_candidate_provider(args.candidate_config)
    if worker.multiple:
        provider.registry['cube2'] = dict(provider.registry['cube'], visual_description='yellow rigid cube or block')
        provider.task = TaskSpec('memory_two_distinct_objects',
            'Move the red cube to the first blue area, then the yellow cube to the second blue area. Do not touch or move the green vessel.',
            (Subgoal('cube', 'tray'), Subgoal('cube2', 'tray2')), ('vase',))
        provider.destinations['tray2'] = [.19, .17, .036]
    controller = controller_module.RunnerController(sdk, robot_id='yam-local-1', hardware_control_enabled=False,
        share_conversation=False, recording_root=output/'runner_recordings')
    atomic_json(output/'case_protocol.json', {'case': args.case, 'seed': args.seed,
        'expected_completion': args.case in ('control','memory_moved','ack_lost_once','ws_reconnect'),
        'task': provider.task.instruction, 'destinations': provider.destinations,
        'all_model_calls_real': True, 'public_robot_contacted': False})
    started = time.monotonic()
    controller.join_and_run(provider, provider.task.instruction)
    stop_record = {}
    deadline = time.monotonic()+420
    last = None
    while controller._worker.is_alive() and time.monotonic() < deadline:
        state = controller.status()
        progress = (state['status'], len(backend.trajectory_log))
        if progress != last:
            print(json.dumps({'case': args.case, 'controller': progress[0], 'packets': progress[1]}), flush=True)
            last = progress
        if args.case == 'operator_stop' and len(worker.executed_waypoints) >= 3 and not stop_record:
            stop_record = {'packets_before_stop': len(backend.trajectory_log),
                           'executed_before_stop': len(worker.executed_waypoints)}
            controller.stop('user_requested')
        time.sleep(.01 if args.case == 'operator_stop' else .1)
    if controller._worker.is_alive():
        controller.stop('qualification_deadline')
    controller._worker.join(15)
    # Finalize through the physics thread after any active execution unwinds.
    physical = worker.rpc(worker.finalize, timeout=60)
    state = controller.status()
    result_paths = sorted((output/'agent_runs').glob('*/result.json'))
    result = json.loads(result_paths[-1].read_text()) if result_paths else {}
    receipt = assess(args.case, worker, gateway, backend, result, physical, state, stop_record)
    receipt['wall_time_s'] = time.monotonic()-started
    receipt['controller_error'] = state.get('error')
    receipt['controller_worker_exited'] = not controller._worker.is_alive()
    receipt['passed'] &= receipt['controller_worker_exited']
    for name, value in [('receipt', receipt), ('controller_status', state), ('gateway_requests', requests),
                        ('fault_gateway_requests', gateway.requests), ('trajectory_packets', backend.trajectory_log),
                        ('trajectory_events', backend.trace_events), ('stop_record', stop_record)]:
        atomic_json(output/(name+'.json'), value)
    server.should_exit = True
    thread.join(5)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
