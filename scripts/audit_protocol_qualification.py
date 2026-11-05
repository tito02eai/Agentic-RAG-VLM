"""Audit a frozen production-provider qualification, retaining failed tasks."""
import argparse,collections,hashlib,json,math
from pathlib import Path
import zipfile


def audit(folder):
    folder=Path(folder);summary=json.loads((folder/'summary.json').read_text());protocol=summary['protocol']
    errors=[];cases=[]
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(folder/'source.zip')!=protocol['source_sha256']:errors.append('source_archive_changed')
    with zipfile.ZipFile(folder/'source.zip') as archive:
        manifest=json.loads(archive.read('SOURCE_SHA256.json'))
        for name,expected in manifest.items():
            if digest(folder/'source'/name)!=expected:errors.append('frozen_source_changed:'+name)
    if digest(folder/'source/knowledge_base/yam_verified_diverse_r8.json')!=protocol['knowledge_sha256']:errors.append('knowledge_changed')
    expected=[(c['seed'],c['repetition']) for c in protocol['cases']] if 'cases' in protocol else [
        (s,r) for s in protocol['seeds'] for r in range(1,protocol['repetitions']+1)]
    if collections.Counter((r['seed'],r['repetition']) for r in summary['cases'])!=collections.Counter(expected):errors.append('case_count_mismatch')
    for row in summary['cases']:
        directory=folder/row['directory'];receipt=json.loads((directory/'receipt.json').read_text())
        path=next((directory/'agent_runs').glob('*/result.json'));agent=json.loads(path.read_text())
        profile=json.loads((directory/'profile.json').read_text());packets=json.loads((directory/'trajectory_packets.json').read_text())
        events=json.loads((directory/'trajectory_events.json').read_text());calls=list(agent.get('reasoner_calls',[]))
        calls += [json.loads(p.read_text()) for p in (directory/'agent_runs').glob('*/images/*/grounded_vlm_response.json')]
        local=[]
        if profile.get('execution_environment')!='simulation' or profile.get('physical_validation',{}).get('passed') is not False:local.append('incorrect_execution_provenance')
        if any(c.get('response_model')!=protocol['vlm']['model'] or not c.get('raw_text') for c in calls):local.append('inference_identity_or_raw_response_missing')
        physical=json.loads((directory/'physical_result.json').read_text())
        if receipt['success']:
            if agent['outcome']!='complete' or not all(physical['checks'].values()):local.append('false_task_success')
            if len(packets)<2:local.append('missing_pick_place_trajectories')
            terminal={e['payload']['trajectory_id']:e['payload']['status'] for e in events if e['payload']['status']!='accepted'}
            if any(terminal.get(p['trajectory_id'])!='completed' for p in packets):local.append('trajectory_not_completed')
            steps=[w['step_id'] for p in packets for w in p['waypoints']]
            if steps!=list(range(len(steps))):local.append('noncontiguous_waypoints')
            commands=[e['payload'] for e in agent['receipts'] if e['kind']=='command']
            pick=next(e for e in commands if e['sent']['command_type']=='pick')
            place=next(e for e in commands if e['sent']['command_type']=='place')
            if pick['receipt'].get('retention_observation',{}).get('passed') is not True:local.append('retention_proxy_missing')
            frames=place['receipt']['keyframes']
            if any(f['gripper']!=.02 for f in frames[:3]):local.append('gripper_preload_not_held')
            lines=(directory/'physics/private_evaluator.jsonl').read_text().splitlines();last=json.loads(lines[-1])
            error=math.dist(last['target_xyz'],[.26,.12,.018])
            support=any(set(c[:2])=={'table','target_geom'} for c in last['contacts'])
            if error>=.025 or not support:local.append('independent_placement_check_failed')
            if not (directory/'physics/rollout.mp4').is_file():local.append('missing_video')
        elif packets:
            # Failures after motion are retained and must not be reported as
            # successful; only the known no-motion segmentation failure is
            # expected in the first batch.
            local.append('failed_trial_requires_motion_review')
        errors += [row['directory']+':'+e for e in local]
        planned=[e['payload'] for e in agent['trace'] if e['event']=='subgoal_planned']
        cases.append({'seed':row['seed'],'repetition':row['repetition'],'cohort':row.get('cohort','initial_qualification'),
            'success':receipt['success'],'framework_outcome':agent['outcome'],
            'stop_reason':agent['trace'][-1]['payload'] if agent['outcome']!='complete' else None,
            'trajectory_packets':len(packets),'waypoints':sum(len(p['waypoints']) for p in packets),
            'actual_model_replies':len(calls),'destination_error_m':physical['destination_error_m'],
            'constraint_active':bool(planned and planned[0]['action']['constraint']['active']),
            'grasp_aperture_retention_checked':bool(receipt['success']),
            'directory':row['directory'],'audit_errors':local})
    cohorts={}
    for row in cases:
        value=cohorts.setdefault(row['cohort'],{'tasks':0,'successes':0});value['tasks']+=1;value['successes']+=row['success']
    return {'integrity_passed':not errors,'errors':errors,'dataset_complete':summary['dataset_complete'],
        'cohorts':cohorts,'cases':cases,'actual_model_replies':sum(c['actual_model_replies'] for c in cases),
        'public_robot_contacted':False,'note':'Audit integrity and task success are separate. Do not pool regression and new-layout cohorts.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(args.campaign);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('integrity_passed','errors','cohorts','actual_model_replies')},indent=2))
    raise SystemExit(0 if result['integrity_passed'] else 1)
if __name__=='__main__':main()
