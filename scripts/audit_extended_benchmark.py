"""Independent source/config, tactile, physical-result and retry audit."""
import argparse,hashlib,json,math
from pathlib import Path
from zipfile import ZipFile


def verify_endpoint(sample, final_time, reported_xyz, goal_xyz, geom):
    """Do not compare unequal timestamps as if they were exact snapshots."""
    body={'target_geom':'target_cube','target_geom2':'target_cube2','target_geom3':'target_cube3'}[geom]
    xyz=sample['objects'][body];gap=final_time-sample['time'];difference=math.dist(xyz,reported_xyz)
    if abs(gap)<1e-8:
        return difference<1e-5,None
    supported=any(set(c[:2])=={'table',geom} for c in sample['contacts'])
    valid=(0<gap<=.020001 and goal_xyz is not None and math.dist(xyz,goal_xyz)<.025
           and math.dist(reported_xyz,goal_xyz)<.025 and supported)
    return valid,{'kind':'sampled_endpoint_task_verification','time_gap_s':gap,
        'position_difference_m':difference,'exact_final_pose_verified':False,
        'note':'task destination/support checked at last <=20-ms sample; exact final-time equality not asserted'}


def verify_motor_force(sample, cap, *, require_telemetry=False):
    """Contact forces include inertia/passive effects; they are not motor force."""
    value=sample.get('actuator_force_n')
    if value is not None:
        return math.isfinite(value) and abs(value)<=cap+1e-6,None
    if require_telemetry:
        return False,{'kind':'missing_actuator_force_telemetry'}
    contact=sum(sample['normal_force_n'].values())
    warning=({'kind':'legacy_contact_force_not_actuator_force','time':sample['time'],
              'contact_normal_sum_n':contact,'motor_force_cap_n':cap,
              'actual_motor_output_verified':False}
             if sample['bilateral'] and abs(contact-cap)>=.1 else None)
    return True,warning


def audit(root):
    summary=json.loads((root/'summary.json').read_text());config=json.loads((root/'frozen_config.json').read_text())
    errors=[];warnings=[]
    def check(ok,label):
        if not ok:errors.append(label)
    check(hashlib.sha256((root/'source.zip').read_bytes()).hexdigest()==summary['source_sha256'],'source hash')
    check(hashlib.sha256((root/'frozen_config.json').read_bytes()).hexdigest()==summary['config_sha256'],'config hash')
    with ZipFile(root/'source.zip') as archive:
        manifest=json.loads(archive.read('SOURCE_SHA256.json'))
        for name,digest in manifest.items():check(hashlib.sha256(archive.read(name)).hexdigest()==digest,f'source:{name}')
        if 'knowledge_store' in summary:
            check(manifest.get(summary['knowledge_store'])==summary['knowledge_store_sha256'],'selected knowledge store hash')
            check(config.get('knowledge_store',summary['knowledge_store'])==summary['knowledge_store'],'selected knowledge store config')
    expected={(c['name'],s) for c in config['conditions'] for s in config['seeds']}
    seen=[(c['condition'],c['seed']) for c in summary['cases']]
    check(set(seen)==expected and len(seen)==len(expected),'declared case coverage')
    conditions={c['name']:c for c in config['conditions']};counts={};paired_scenes={}
    health=json.loads((root/'model_health.json').read_text()) if (root/'model_health.json').exists() else None
    if config.get('model_health_required'):
        check(health is not None,'model health metadata missing')
        if health:check(health.get('server_source_sha256')==manifest.get('scripts/serve_local_vlm.py'),'inference server source fingerprint')
    for case in summary['cases']:
        folder=root/case['directory'];r=json.loads((folder/'result.json').read_text());label=case['directory']
        c=conditions[case['condition']]
        mode=config.get('mode_by_seed',{}).get(str(case['seed']),c['mode'])
        check(r['physical_success']==all(r['checks'].values())==case['physical_success'],label+': verdict')
        check(r['seed']==case['seed'] and r['mode']==mode and r['friction']==c['friction'],label+': condition')
        for key in ('controller_profile','scene_variation','recovery_policy','mixed_shapes','grounding_repair','multiccd','visibility_replanning','visible_goal_fallback','retrieval_profile','planning_policy'):
            if key in c:check(r.get(key)==c[key],label+': '+key)
        if config.get('iteration_order')=='paired_seed':
            scene_hash=hashlib.sha256((folder/'scene.xml').read_bytes()).hexdigest()
            paired_scenes.setdefault(case['seed'],scene_hash)
            check(paired_scenes[case['seed']]==scene_hash,label+': paired scene differs')
        check(not r['object_attachment_used'] and not r['object_pose_reset_after_initialization'],label+': manipulation flags')
        check(r['disable_memory']==(not c['memory']) and r['disable_recovery']==(not c['recovery']),label+': switches')
        if 'scene_graph' in c:check(r.get('disable_scene_graph')==(not c['scene_graph']),label+': scene graph switch')
        width=config.get('upper_half_width_by_seed',{}).get(str(case['seed']),c.get('upper_half_width_m'))
        if width is not None:
            check(r.get('upper_half_width_m')==width,label+': physical occlusion condition')
            import xml.etree.ElementTree as ET
            shape=[float(v) for v in ET.parse(folder/'scene.xml').find('.//geom[@name="target_geom2"]').get('size').split()]
            check(abs(shape[0]-width)<1e-9 and abs(shape[1]-width)<1e-9 and abs(shape[2]-.014)<1e-9,label+': actual upper footprint')
        if c.get('retrieval_profile')=='fixed':
            planned=[e['payload']['action'] for e in r['trace'] if e['event']=='subgoal_planned']
            check(all(a['source'] in {'fixed_skill','episodic_memory'} for a in planned),label+': fixed skill mislabeled as HAA-RAG')
            check(all(e['payload']['evidence']['retrieval']['retrieval_enabled'] is False for e in r['trace'] if e['event']=='subgoal_planned'),label+': fixed skill performed retrieval')
        if c.get('planning_policy')=='geometric':
            check(not r['reasoner_calls'] and not r['memory'],label+': geometric baseline used agent reasoning/memory')
            check(not any(e['event'] in {'visibility_replan','l3_replan_started'} for e in r['trace']),label+': geometric baseline used agent task replanning')
        check(not r['memory'] if not c['memory'] else True,label+': disabled memory')
        check(r['uses_vlm'] and not r['original_paper_trials'],label+': experiment provenance')
        check(case['attempts']==len(r['attempts']),label+': attempt summary count')
        check(case['failed_attempts']==sum(not a['retained'] for a in r['attempts']),label+': failure summary count')
        memory_hits=sum(e['event']=='subgoal_planned' and e['payload']['action']['source']=='episodic_memory' for e in r['trace'])
        check(case['memory_hits']==r['memory_hits']==memory_hits,label+': memory summary count')
        for attempt in r['attempts']:
            check(len(attempt['retention_samples'])==10,label+': retention sample coverage')
            times=[s['time'] for s in attempt['retention_samples']]
            check(len(times)==10 and abs(times[-1]-times[0]-.36)<1e-5,label+': retention sample duration')
            sensed=all(s['bilateral'] and s['left_finger_q_m']>.004 for s in attempt['retention_samples'])
            if attempt.get('retention_contract') == 'low_height_proof_v1':
                sensed = sensed and attempt['proof_passed'] and attempt['lift_abort_reason'] is None
                samples = attempt['proof_samples']
                if attempt['proof_passed']:
                    check(len(samples)==10 and all(s['bilateral'] and s['downward_load_n']>.03 for s in samples),label+': proof load/contact')
                    check(max(s['left_finger_q_m'] for s in samples)-min(s['left_finger_q_m'] for s in samples)<.0006,label+': proof aperture stability')
                    normal=min(sum(s['normal_force_n'].values()) for s in samples)
                    load=max(s['downward_load_n'] for s in samples)
                    check(normal>=1.7*load,label+': proof reserve margin')
                    vision=attempt['visual_proof']
                    valid=[cam for cam in set(vision['before']) & set(vision['first']) & set(vision['second'])
                        if min(vision['first'][cam],vision['second'][cam])-vision['before'][cam]>=.006
                        and abs(vision['first'][cam]-vision['second'][cam])<.0008]
                    check(bool(valid) and set(vision['valid_cameras'])<=set(valid),label+': proof visual motion')
                updates=attempt['force_adjustments'];force=attempt['initial_motor_force_cap_n']
                check(len(updates)<=3,label+': preventive force budget')
                for update in updates:
                    check(abs(update['before_n']-force)<1e-8 and abs(update['after_n']-force*1.2)<1e-8 and update['after_n']<=1.6,label+': preventive force bounds')
                    force=update['after_n']
                check(abs(force-attempt['motor_force_cap_n'])<1e-8,label+': final applied force')
            check(attempt['retained']==sensed,label+': sensed retention')
            for sample in attempt['retention_samples']:
                valid,warning=verify_motor_force(sample,attempt['motor_force_cap_n'],
                    require_telemetry=r.get('tactile_telemetry_version',1)>=2)
                check(valid,label+': actual motor-force limit/telemetry')
                if warning:warnings.append({'case':label,'target':attempt['target'],**warning})
            if r.get('tactile_telemetry_version',1)>=2:
                for sample in [attempt['initial_tactile'],*attempt.get('lift_monitor_samples',[]),*attempt.get('proof_samples',[])]:
                    time=sample.get('time',attempt['initial_tactile_time_s'])
                    cap=attempt.get('initial_motor_force_cap_n',attempt['motor_force_cap_n'])
                    for update in attempt.get('force_adjustments',[]):
                        if update['time']<=time:cap=update['after_n']
                    valid,_=verify_motor_force(sample,cap,require_telemetry=True)
                    check(valid,label+': motor-force limit during lift/proof')
        recoveries=[e for e in r['trace'] if e['event']=='contact_failure_recovery']
        check(c['recovery'] or not recoveries,label+': disabled recovery invoked')
        for event in recoveries:
            evidence=event['payload']['evidence']
            check(evidence['release_confirmed'] and evidence['scene_stable'],label+': unsafe retry')
            check(len(set(evidence['public_observation_ids']))==2,label+': repeated retry observation')
            if evidence['action']=='bounded_force_escalation':
                check(0<evidence['corrected_strategy']['force_n']<=1.6,label+': adaptive force bound')
        public=[json.loads(line) for line in (folder/'public_trace.jsonl').read_text().splitlines()]
        for probe in (e for e in public if e['event']=='placement_unload_probe'):
            matches=[e for e in public if e['event']=='placement_visual_support' and e['target']==probe['target']
                     and e['matched_surface_xyz_m'] is not None and 0<=probe['time']-e['time']<2.1]
            check(len({e['time'] for e in matches})>=2,label+': unload lacks fresh visual support')
            check(probe['opening_increment_m']<=.0005 and probe['step']<=3,label+': unload exceeds movement bound')
        if r['physical_success']:
            raw=[json.loads(line) for line in (folder/'private_evaluator.jsonl').read_text().splitlines()]
            for name,body in (('cube','target_cube'),('cube2','target_cube2'),('cube3','target_cube3')):
                if name not in r['final_objects']:continue
                xyz=raw[-1]['objects'][body]
                geom={'cube':'target_geom','cube2':'target_geom2','cube3':'target_geom3'}[name]
                valid,warning=verify_endpoint(raw[-1],r['simulated_time_s'],r['final_objects'][name]['xyz'],r.get('destinations',{}).get(name),geom)
                check(valid,label+': endpoint evidence inconsistent')
                if warning:warnings.append({'case':label,'object':name,**warning})
                check(max(e['objects'][body][2] for e in raw)>.1,label+': no raw lift')
            if mode=='stacked_sort':
                order=r['summary']['completed_targets'];check(order.index('cube2')<order.index('cube'),label+': support order')
        perception=[e for e in r['receipts'] if e['kind']=='perception']
        visibility_events=[e for e in r['trace'] if e['event']=='visibility_replan']
        check(c.get('visibility_replanning',False) or not visibility_events,label+': disabled visibility policy invoked')
        if visibility_events:
            observation_map={};observation_receipts={};last_id=None
            for receipt in r['receipts']:
                if receipt['kind']=='observation':
                    last_id=receipt['payload']['observation_id'];observation_receipts[last_id]=receipt['payload']
                elif receipt['kind']=='perception':observation_map[last_id]=receipt['payload']['vlm_result']
            initial_task=next(e['payload']['task'] for e in r['trace'] if e['event']=='task_started')
            check(initial_task.get('reorderable') is True,label+': ordered task was changed')
            check(len(visibility_events)<=3,label+': visibility replan budget')
            for entry in visibility_events:
                event=entry['payload']
                witness=event['witness'];value=observation_map.get(witness['observation_id'],{})
                known={g['name'] for g in value.get('grounding',[])}
                check(event['selected_target'] in known and set(event['deferred_targets']).isdisjoint(known),label+': visibility used ungrounded target or invented missing labels')
                if event.get('policy')=='visible_target_progress_then_reobserve':
                    check(c.get('visible_goal_fallback',False),label+': fallback not enabled')
                    check(witness in value.get('scene_context',{}).get('visible_geometry',[]),label+': fallback lacks original public geometry')
                    check(event['support_identity']=='not_inferred',label+': fallback invented a support identity')
                    plane=event.get('support_plane',{})
                    check(plane==value.get('scene_context',{}).get('support_plane'),label+': fallback changed calibrated plane')
                    check(witness['center_xyz_m'][2]-witness['known_half_height_m']-plane.get('z_m',float('inf'))>.01,label+': fallback target not elevated')
                    pair=event.get('observation_pair',{})
                    check(any(e['kind']=='visibility_stability_check' and e['payload']==pair for e in r['receipts']),label+': missing stability receipt')
                    old_id=pair.get('previous_observation_id');new_id=pair.get('current_observation_id')
                    old_time=observation_receipts.get(old_id,{}).get('receipt',{}).get('captured_at')
                    new_time=observation_receipts.get(new_id,{}).get('receipt',{}).get('captured_at')
                    check(old_id!=new_id and new_id==witness['observation_id'] and old_time is not None and new_time is not None and new_time>old_time,label+': stale fallback observations')
                    check(old_time==pair.get('previous_capture_time') and new_time==pair.get('current_capture_time') and pair.get('position_tolerance_m')==.004,label+': altered stability times/tolerance')
                    def centers(v):return {g['target']:g['center_xyz_m'] for g in v.get('scene_context',{}).get('visible_geometry',[])}
                    old_positions=centers(observation_map.get(old_id,{}));new_positions=centers(value)
                    required=set(initial_task['protected_objects'])|{event['selected_target']}
                    check(required<=set(pair.get('stable_visible_targets',[])),label+': unstable fallback targets')
                    check(all(n in old_positions and n in new_positions and math.dist(old_positions[n],new_positions[n])<=.004 for n in required),label+': fallback stability not supported by RGB-D')
                else:
                    check(witness in value.get('scene_context',{}).get('unresolved_supports',[]),label+': visibility lacks original perception witness')
                    check(event['support_identity']=='unresolved' and witness['support_identity']=='unresolved',label+': anonymous support was assigned an identity')
                check(set(initial_task['protected_objects'])<=known,label+': visibility ignored unseen protected objects')
                plans=[e['payload'] for e in r['trace'] if e['event']=='subgoal_planned' and e['sequence']>entry['sequence']]
                if r['physical_success'] or plans:
                    check(bool(plans) and plans[0]['target']==event['selected_target'],label+': visibility selection differs from executed plan')
                    if plans:
                        thought=plans[0].get('vlm_thought') or {}
                        check(plans[0]['evidence'].get('visibility_replanning',{}).get('witness')==witness,label+': visibility evidence not delivered to planner')
                        check(thought.get('parsed',{}).get('decision')=='execute' and '"visibility_replanning"' in thought.get('prompt',''),label+': visibility change lacks real VLM review')
            if r['physical_success']:
                check(set(r['summary']['completed_targets'])=={g['target'] for g in initial_task['subgoals']},label+': deferred subgoal was skipped')
        saved=sorted(folder.glob('observation_*/vlm_response.json'))
        extra=len(saved)-len(perception)
        failed_perception=(r['outcome']=='safe_stop' and any(e['kind']=='platform_error'
            and e['payload'].get('error_type')=='PerceptionError' for e in r['receipts']))
        check(extra==0 or (extra==1 and failed_perception),label+': unaccounted raw perception responses')
        reviews=sorted(folder.glob('observation_*/identity_review_*.json'))
        raw_calls=len(saved)+len(reviews)+len(r['reasoner_calls'])
        expected_calls=raw_calls if summary.get('inference_count_contract')=='saved_responses_v2' else len(perception)+len(r['reasoner_calls'])
        check(case['vlm_calls']==expected_calls,label+': declared VLM count contract')
        for path in folder.glob('observation_*/grounding_decision.json'):
            decision=json.loads(path.read_text())
            check(hashlib.sha256((path.parent/'vlm_response.json').read_bytes()).hexdigest()==decision['initial_response_sha256'],label+': initial grounding response hash')
            calls=[json.loads((path.parent/name).read_text()) for name in decision['call_files']]
            check(len(calls)==decision['review_calls']<=6,label+': identity review budget')
            labels=dict(decision['initial_labels']) if isinstance(decision['initial_labels'],dict) else {}
            labels={key:labels.get(key,'unknown') if isinstance(labels.get(key,'unknown'),str) else 'unknown' for key in decision['effective_labels']}
            allowed=set(r['final_objects'])|{'vase','unknown'}
            for call in calls:
                image=path.parent/(call['candidate_id']+'.png')
                check(image.resolve().is_relative_to(path.parent.resolve()),label+': unsafe review image path')
                check(hashlib.sha256(image.read_bytes()).hexdigest()==call['image_sha256'],label+': review image hash')
                value=call['parsed'].get('label') if isinstance(call['parsed'],dict) else None
                labels[call['candidate_id']]=value if isinstance(value,str) and value in allowed else 'unknown'
            check(labels==decision['effective_labels'],label+': effective labels lack model evidence')
        for receipt in perception:
            check('clip-vit-large-patch14@sha256:' in receipt['payload']['vlm_result'].get('embedding_space',''),label+': real embedding namespace')
        if health:
            for response in [json.loads(path.read_text()) for path in saved+reviews]+r['reasoner_calls']:
                check(response['response_model']==health['model'],label+': model identity changed')
        for event,path in zip(perception,saved):
            check(event['payload']['vlm_result']['raw_text']==json.loads(path.read_text())['raw_text'],label+': raw model text')
            value=event['payload']['vlm_result']
            if 'identity_review' in value:
                decision=json.loads((path.parent/'grounding_decision.json').read_text())
                check(value['identity_review']==decision,label+': grounding receipt differs from audit record')
                check(all(value['effective_identity_labels'][g['candidate_id']]==g['name'] for g in value['grounding']),label+': grounded objects differ from reviewed labels')
        counts.setdefault(case['condition'],{'cases':0,'successes':0,'attempts':0,'failed_attempts':0,'memory_hits':0,'vlm_calls':0,'rejected_perception_responses':0,'identity_review_calls':0})
        out=counts[case['condition']];out['cases']+=1;out['successes']+=r['physical_success']
        for k in ('attempts','failed_attempts','memory_hits'):out[k]+=case[k]
        out['vlm_calls']+=raw_calls;out['rejected_perception_responses']+=extra;out['identity_review_calls']+=len(reviews)
    return {'passed':not errors,'checked_cases':len(seen),'source_files':len(manifest),'conditions':counts,'errors':errors,'warnings':warnings,
        'scope':'artifact consistency plus sampled independent evaluator/tactile evidence; not real hardware certification'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);args=p.parse_args()
    result=audit(args.directory);print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)
