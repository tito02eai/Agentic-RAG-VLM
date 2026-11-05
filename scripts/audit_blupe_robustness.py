"""Read-only audit of frozen robustness cases and original execution evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def audit(folder):
    folder = Path(folder)
    read = lambda p: json.loads(p.read_text())
    protocol, summary = read(folder/'protocol.json'), read(folder/'summary.json')
    errors, rows = [], []
    digest = hashlib.sha256((folder/'source.zip').read_bytes()).hexdigest()
    if digest != protocol['source_sha256']:
        errors.append('source_archive_changed')
    with ZipFile(folder/'source.zip') as archive:
        for name, expected in json.loads(archive.read('SOURCE_SHA256.json')).items():
            if hashlib.sha256((folder/'source'/name).read_bytes()).hexdigest() != expected:
                errors.append('frozen_source_changed:'+name)
    if not summary.get('complete') or Counter(r['case'] for r in summary['cases']) != Counter(protocol['cases']):
        errors.append('declared_cases_incomplete')
    for case in protocol['cases']:
        path = folder/case
        receipt = read(path/'receipt.json')
        result = read(next((path/'agent_runs').glob('*/result.json')))
        physical = read(path/'physical_result.json')
        profile = read(path/'profile.json')
        packets = read(path/'trajectory_packets.json')
        events = read(path/'trajectory_events.json')
        wire = read(path/'fault_gateway_requests.json')
        executed = read(path/'executed_waypoints.json')
        trace, commands = result['trace'], [r['payload'] for r in result['receipts'] if r['kind']=='command']
        local = []
        if profile.get('execution_environment') != 'simulation' or profile['physical_validation']['passed'] is not False:
            local.append('incorrect_execution_provenance')
        if physical.get('physical_robot_contacted') is not False or receipt.get('physical_robot_contacted') is not False:
            local.append('incorrect_hardware_claim')
        calls = result.get('reasoner_calls', []) + [read(f) for f in path.glob('agent_runs/*/images/*/grounded_vlm_response.json')]
        if not calls or any(c.get('response_model') != protocol['model_health']['vlm']['model'] or not c.get('raw_text') for c in calls):
            local.append('raw_model_reply_or_identity_missing')
        for key in ('protected_stable','no_protected_robot_contact','no_arm_table_contact','no_self_contact'):
            if physical['checks'].get(key) is not True:
                local.append('physical_safety_violation:'+key)
        ids = [p['trajectory_id'] for p in packets]
        if len(set(ids)) != len(ids):
            local.append('duplicate_physical_packet')
        steps = [w['step_id'] for p in packets for w in p['waypoints']]
        if steps != list(range(len(steps))):
            local.append('noncontiguous_command_steps')
        executed_ids = [w['step_id'] for w in executed]
        if executed_ids != steps[:len(executed_ids)] or len(set(executed_ids)) != len(executed_ids):
            local.append('duplicate_or_reordered_execution')
        terminal = {e['payload']['trajectory_id']:e['payload']['status'] for e in events if e['payload']['status']!='accepted'}
        complete_expected = case in ('control','memory_moved','ack_lost_once','ws_reconnect')
        if complete_expected:
            if result['outcome']!='complete' or not physical['simulation_success']:
                local.append('expected_task_did_not_complete')
            if executed_ids!=steps or any(terminal.get(t)!='completed' for t in ids):
                local.append('missing_physical_terminal_execution')
            if not all(physical['checks'].values()):
                local.append('independent_placement_failed')
        elif result['outcome']!='safe_stop' or any(e['event']=='task_completed' for e in trace) or receipt['task_success']:
            local.append('false_task_completion_under_fault')
        if case in ('empty_grasp','slip','memory_failed_reuse'):
            failures = [c for c in commands if c['receipt'].get('reason')=='retention_unconfirmed']
            expected_packets = 3 if case.startswith('memory_') else 1
            if not failures or not receipt['faults'] or len(packets)!=expected_packets:
                local.append('retention_failure_or_stop_not_observed')
            if case=='slip' and not any(f.get('bilateral_lift_seconds_before_fault',0)>=.15 for f in receipt['faults']):
                local.append('slip_not_after_real_lift')
        if case.startswith('memory_'):
            verified = [e['sequence'] for e in trace if e['event']=='subgoal_verified' and e['payload']['success']]
            reused = [e for e in trace if e['event']=='subgoal_planned' and e['payload']['action']['source']=='episodic_memory']
            if not reused or not verified or min(verified)>=reused[0]['sequence']:
                local.append('memory_reuse_without_preceding_verified_action')
            if case=='memory_moved':
                moved = [f for f in receipt['faults'] if f['kind']=='external_force_moves_pending_object']
                from math import dist
                if not moved or moved[0]['qpos_assigned'] or moved[0]['displacement_m']<=.03:
                    local.append('physical_displacement_not_exercised')
                elif reused:
                    xyz = reused[0]['payload']['action']['position_xyz_m'][:2]
                    if dist(xyz,moved[0]['after_xyz_m'][:2])>=.015 or dist(xyz,moved[0]['before_xyz_m'][:2])<=.03:
                        local.append('memory_reused_stale_position')
                if not any(e['event']=='l3_replan_started' for e in trace):
                    local.append('missing_scene_change_replanning')
                second = physical.get('second_target',{})
                if not second.get('destination') or not second.get('supported') or second.get('bilateral_lift_seconds',0)<=.5:
                    local.append('second_target_physical_verification_failed')
            else:
                if 'cube' in result.get('memory',{}):
                    local.append('failed_memory_not_invalidated')
                if not any(e['event']=='memory_invalidation' for e in trace):
                    local.append('memory_eviction_runtime_event_missing')
        if case.startswith('ack_lost'):
            submissions = [r for r in wire if r['trajectory_id']==ids[0]] if ids else []
            if len(submissions)<2 or len({r['body_sha256'] for r in submissions})!=1:
                local.append('stable_id_body_retry_not_exercised')
            if case=='ack_lost_persistent' and len(packets)!=1:
                local.append('motion_continued_after_uncertain_ack')
        if case=='feedback_outage' and (not any(f['kind']=='feedback_read_unavailable' for f in receipt['faults']) or len(packets)!=1):
            local.append('feedback_outage_not_exercised_or_motion_continued')
        if case=='ws_reconnect' and not receipt['checks'].get('websocket_actually_reconnected'):
            local.append('ws_reconnect_not_exercised')
        if case=='operator_stop':
            stop = read(path/'stop_record.json')
            if stop.get('executed_before_stop',0)<3 or len(packets)!=1 or not any(v=='aborted' for v in terminal.values()):
                local.append('active_executor_stop_not_exercised')
        if not (path/'physics/rollout.mp4').is_file():
            local.append('video_missing')
        if receipt['passed'] != (all(receipt['checks'].values()) and receipt['controller_worker_exited']):
            local.append('receipt_internally_inconsistent')
        errors.extend(case+':'+e for e in local)
        rows.append({'case':case, 'criterion_passed':receipt['passed'], 'agent_outcome':result['outcome'],
            'task_completed':result['outcome']=='complete', 'trajectory_packets':len(packets),
            'actual_model_replies':len(calls), 'audit_errors':local})
    return {'integrity_passed':not errors,'all_acceptance_criteria_passed':all(r['criterion_passed'] for r in rows),
        'errors':errors,'cases':rows,'actual_model_replies':sum(r['actual_model_replies'] for r in rows),
        'source_sha256':digest,'physical_robot_contacted':False,
        'scope':'functional engineering checks; intentional stops are not successful grasp tasks or a success rate'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('campaign',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(args.campaign)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['integrity_passed'] and result['all_acceptance_criteria_passed'] else 1)
