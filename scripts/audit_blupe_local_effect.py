"""Audit a declared local Playground/MuJoCo batch; never contacts hardware."""
import argparse,collections,hashlib,json,math,subprocess
from pathlib import Path
import zipfile


def audit(folder):
    folder=Path(folder).resolve();protocol=json.loads((folder/'frozen_protocol.json').read_text())
    errors=[];cases=[];scene_hashes={}
    if hashlib.sha256((folder/'source.zip').read_bytes()).hexdigest()!=protocol['source_sha256']:errors.append('source_archive_changed')
    with zipfile.ZipFile(folder/'source.zip') as archive:
        manifest=json.loads(archive.read('SOURCE_SHA256.json'))
        for name,digest in manifest.items():
            if hashlib.sha256((folder/'source'/name).read_bytes()).hexdigest()!=digest:errors.append('frozen_source_changed:'+name)
    kb=folder/'source/knowledge_base/yam_verified_diverse_r8.json'
    if hashlib.sha256(kb.read_bytes()).hexdigest()!=protocol['knowledge_sha256']:errors.append('knowledge_changed')
    declared={c['key']:c for c in protocol['cases']}
    receipts=[(p,json.loads(p.read_text())) for p in folder.glob('runs/*/receipt.json')]
    keys=[r['scenario'] for _,r in receipts]
    if collections.Counter(keys)!=collections.Counter(declared.keys()):errors.append('declared_case_count_mismatch')
    for path,receipt in receipts:
        key=receipt['scenario'];sim=path.parent/'simulation';local=[]
        result=json.loads((sim/'result.json').read_text()) if (sim/'result.json').exists() else {}
        if not result:
            errors.append(key+':missing_final_result')
            cases.append({'scenario':key,'run_id':receipt['run_id'],'simulation_success':False,
                'outcome':'incomplete','audit_errors':['missing_final_result']})
            continue
        if receipt['result']!='completed':local.append('incomplete_or_failed_run')
        checks=result.get('checks',{})
        if not checks or not all(v is True for v in checks.values()):local.append('physical_checks_not_all_true')
        if receipt.get('simulation_success') is not True or result.get('physical_success') is not True:local.append('simulation_success_not_established')
        if result.get('uses_vlm') is not True:local.append('real_vlm_path_not_enabled')
        if result.get('object_attachment_used') is not False or result.get('object_pose_reset_after_initialization') is not False:local.append('unsupported_attachment_or_pose_reset')
        raw=[json.loads(p.read_text()) for p in sim.glob('observation_*/vlm_response.json')]
        identity=[json.loads(p.read_text()) for p in sim.glob('observation_*/identity_review_*.json')]
        calls=raw+identity+result.get('reasoner_calls',[])
        if not raw:local.append('missing_raw_perception_calls')
        if any(c.get('requested_model')!=protocol['vlm']['requested_model'] for c in calls):local.append('requested_model_changed')
        if any(not c.get('raw_text') or c.get('parse_error') for c in calls):local.append('raw_response_or_parse_issue')
        expected_model=protocol['vlm']['model']
        if any(c.get('response_model')!=expected_model for c in calls):local.append('response_model_changed')
        last=json.loads((sim/'private_evaluator.jsonl').read_text().splitlines()[-1])
        # Check the independent last physical sample, not the Agent's completion text.
        independent={}
        for i,name in enumerate(('cube','cube2','cube3')):
            body=('target_cube','target_cube2','target_cube3')[i]
            geom=('target_geom','target_geom2','target_geom3')[i]
            xyz=last['objects'][body];dst=result['destinations'][name]
            error=math.dist(xyz,dst)
            support=any(set(c[:2])=={'table',geom} for c in last['contacts'])
            if error>=.025 or not support:local.append('independent_final_state_failed:'+name)
            independent[name]={'destination_error_m':error,'table_supported':support}
        scene_hashes[key]=hashlib.sha256((sim/'scene.xml').read_bytes()).hexdigest()
        events=collections.Counter(e['event'] for e in result.get('trace',[]))
        plan=[e['payload'] for e in result.get('trace',[]) if e['event']=='subgoal_planned']
        video=sim/'rollout.mp4'
        if not video.is_file():local.append('missing_video');video_info={}
        else:
            video_info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration,size','-of','json',str(video)],text=True))['format']
        row={'scenario':key,'run_id':receipt['run_id'],'simulation_success':result.get('physical_success',False),
            'outcome':result.get('outcome'),'attempts':len(result.get('attempts',[])),
            'failed_attempts':sum(not a['retained'] for a in result.get('attempts',[])),
            'force_adjustments':sum(len(a.get('force_adjustments',[])) for a in result.get('attempts',[])),
            'memory_hits':result.get('memory_hits',0),'l3_replans':events['l3_replan_started'],
            'visibility_fallback_events':sum(v for k,v in events.items() if 'visibility' in k),
            'plan_order':[p['target'] for p in plan],'strategy_sources':[p['action']['source'] for p in plan],
            'perception_calls':len(raw),'identity_review_calls':len(identity),'action_review_calls':len(result.get('reasoner_calls',[])),
            'actual_model_calls':len(calls),'response_models':sorted({c.get('response_model','') for c in calls}),
            'physical_checks':checks,'independent_final_state':independent,
            'scene_sha256':scene_hashes[key],'video_relative':str(video.relative_to(folder)),
            'video_info':video_info,'wall_elapsed_s':receipt.get('wall_elapsed_s'),'audit_errors':local}
        if key=='agentic-stack' and (not {'cube','cube2'}.issubset(row['plan_order']) or
                row['plan_order'].index('cube2')>row['plan_order'].index('cube')):local.append('wrong_unstack_order')
        if key=='geometric-stack' and (row['memory_hits'] or row['action_review_calls']):local.append('baseline_not_disabled')
        errors.extend(key+':'+e for e in local);cases.append(row)
    paired=scene_hashes.get('agentic-stack')==scene_hashes.get('geometric-stack')
    if not paired:errors.append('paired_scene_xml_different')
    return {'passed':not errors,'errors':errors,'expected_cases':3,'completed_cases':len(cases),
        'dataset_complete':len(cases)==3,'paired_scene_xml_identical':paired,'cases':sorted(cases,key=lambda r:list(declared).index(r['scenario'])),
        'physical_robot_contacted':False,'public_queue_joined':False,
        'interpretation':'All three local tasks completed. No success-rate advantage established; failure recovery/visibility fallback were not exercised. Compare force-adjustment traces descriptively only.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(args.campaign);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'errors':result['errors'],'cases':len(result['cases']),'paired_scene_xml_identical':result['paired_scene_xml_identical']},indent=2))
    raise SystemExit(0 if result['passed'] else 1)

if __name__=='__main__':main()
