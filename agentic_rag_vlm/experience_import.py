"""Import source-bound successes AND measured rejections from real rollouts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from .experience import ExperienceStore


def import_rollouts(store, folders, *, output):
    """Keep original source files; no synthetic labels, embeddings or forces."""
    output=Path(output)
    if output.exists():raise ValueError('experience output is immutable; choose a new file')
    sources=output.parent/(output.stem+'_sources')
    sources.mkdir(parents=True,exist_ok=False)
    count=0
    for index,folder in enumerate(folders):
        folder=Path(folder);raw=(folder/'result.json').read_bytes();result=json.loads(raw)
        if not result.get('uses_vlm'):raise ValueError('diagnostic identities cannot populate this observed store')
        episode=sources/f'episode_{index:02d}_{folder.name}';episode.mkdir()
        (episode/'result.json').write_bytes(raw)
        plans={e['payload']['target']:e['payload'] for e in result['trace'] if e['event']=='subgoal_planned'}
        observation=None;perception=None
        for event in result['receipts']:
            if event['kind']=='observation':observation=event['payload'];perception=None;continue
            if event['kind']=='perception':perception=event['payload']['vlm_result'];continue
            if event['kind']!='command':continue
            command=event['payload'];sent=command['sent'];receipt=command['receipt']
            if sent['command_type']!='pick' or not observation or not perception:continue
            retention=receipt.get('retention',{})
            positive=(result.get('physical_success') and receipt.get('accepted') is True and retention.get('retained') is True)
            negative=(receipt.get('accepted') is False and receipt.get('execution_started') is True
                      and receipt.get('reason')=='grasp_not_retained' and retention.get('retained') is False)
            if not positive and not negative:continue
            target=sent['target'];plan=plans[target];parameters=sent['parameters']
            signature=plan['action']['strategy']['_semantic_signature']
            category,affordance,material,fragility,region=signature
            if plan['action']['strategy']['_execution_domain']!=store.domain:
                raise ValueError('rollout executor domain differs from store')
            if perception.get('embedding_space')!=store.embedding_space:
                raise ValueError('rollout embedding namespace differs from store')
            ground=[g for g in perception['grounding'] if g['name']==target]
            if len(ground)!=1:raise ValueError('source pick target is not uniquely model-grounded')
            candidate=next(c for c in observation['receipt']['rgbd_candidates'] if c['candidate_id']==ground[0]['candidate_id'])
            # Relocate by observation folder and basename so prior evidence
            # remains importable after copying a rollout directory.
            image=folder/Path(candidate['crop_path']).parent.name/Path(candidate['crop_path']).name
            attachment=episode/(sent['command_id']+'_crop.png');shutil.copy2(image,attachment)
            scale=float(sent.get('constraint',{}).get('force_scale',1.))
            force=float(retention['motor_force_cap_n'])/scale
            strategy={key:deepcopy(parameters[key]) for key in ('synergy','aperture_m','approach','velocity_scale')}
            intrinsic=float(parameters.get('approach_height_delta_m',0))-float(sent.get('constraint',{}).get('approach_height_delta_m',0))
            strategy.update(force_n=force,base_force_n=force,approach_height_delta_m=intrinsic,intrinsic_approach_height_m=intrinsic)
            node=next(n for n in plan['evidence']['scene_graph']['nodes'] if n['name']==target)
            store.add({'id':f'rollout_{index:02d}_{folder.name}_{sent["command_id"]}',
                'category':category,'superclass':node['superclass'],'affordance_type':affordance,
                'material':material,'fragility':fragility,'region':region,'strategy':strategy,'outcome':int(positive),
                'visual_embedding':perception['visual_embeddings'][target],
                'provenance':{'domain':store.domain,'source_episode':folder.name,
                    'evidence_kind':'mujoco_contact_dynamics','original_paper_experience':False,
                    'label_scope':'verified_complete_grasp_place' if positive else 'observed_retention_rejection',
                    'observation_image':str(attachment.relative_to(output.parent)),
                    'image_sha256':hashlib.sha256(attachment.read_bytes()).hexdigest(),
                    'source_result_path':str((episode/'result.json').relative_to(output.parent)),
                    'source_result_sha256':hashlib.sha256(raw).hexdigest(),
                    'source_command_id':sent['command_id'],'source_target':target,
                    'source_observation_id':observation['observation_id'],'source_candidate_id':candidate['candidate_id'],
                    'force_semantics':'actual simulated motor cap, including measured preventive updates; not fingertip force'}})
            count+=1
    if not count:raise ValueError('no measured outcomes imported')
    store.validate_sources(output.parent)
    store.save(output)
    return {'imported':count,'records':len(store.records),
            'positive':sum(r['outcome']==1 for r in store.records),'negative':sum(r['outcome']==0 for r in store.records)}
