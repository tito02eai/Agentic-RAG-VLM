"""Freeze code/data and run every declared extended case once, failures retained."""
import argparse,hashlib,json,os,subprocess,sys,time
from urllib.request import urlopen
from pathlib import Path
from zipfile import ZipFile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agentic_rag_vlm.clip_client import CLIPFeatureClient
from agentic_rag_vlm.experience import ExperienceStore,atomic_json
from agentic_rag_vlm.services import vlm_endpoint,clip_endpoint,resolve_model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=ROOT/'configs/extended_sim_qualification.json')
    p.add_argument('--store',type=Path,default=ROOT/'knowledge_base/yam_retention_observed.json')
    p.add_argument('--endpoint',default=vlm_endpoint())
    p.add_argument('--model',help='served model; otherwise AGENTIC_VLM_MODEL or /health identity')
    p.add_argument('--clip-endpoint',default=clip_endpoint());args=p.parse_args()
    if args.output.exists():p.error('new output required')
    try:args.model=resolve_model(args.endpoint,args.model)
    except ValueError as error:p.error(str(error))
    store=ExperienceStore.load(args.store);store.validate_sources(args.store.parent)
    if CLIPFeatureClient(args.clip_endpoint).embedding_space!=store.embedding_space:p.error('CLIP weights changed')
    config_bytes=args.config.read_bytes();config=json.loads(config_bytes)
    store_relative=str(args.store.resolve().relative_to(ROOT))
    if config.get('knowledge_store',store_relative)!=store_relative:p.error('configuration specifies a different frozen knowledge store')
    health_url=args.endpoint.rsplit('/v1/chat/completions',1)[0]+'/health'
    model_health=None
    if config.get('model_health_required'):
        with urlopen(health_url,timeout=10) as response:model_health=json.load(response)
        if model_health.get('requested_model')!=args.model:p.error('service checkpoint differs from requested model')
    args.output.mkdir(parents=True);(args.output/'frozen_config.json').write_bytes(config_bytes)
    if model_health is not None:atomic_json(args.output/'model_health.json',model_health)
    archive=args.output/'source.zip'
    subprocess.run([sys.executable,str(ROOT/'scripts/build_delivery_bundle.py'),'--output',str(archive)],check=True,capture_output=True)
    frozen=args.output.resolve()/'source'
    with ZipFile(archive) as z:z.extractall(frozen)
    frozen_store=frozen/args.store.resolve().relative_to(ROOT)
    rows=[];started=time.time()
    pairs=([(condition,seed) for seed in config['seeds'] for condition in config['conditions']]
        if config.get('iteration_order')=='paired_seed' else
        [(condition,seed) for condition in config['conditions'] for seed in config['seeds']])
    for condition,seed in pairs:
            if model_health is not None:
                with urlopen(health_url,timeout=10) as response:current=json.load(response)
                if current!=model_health:raise RuntimeError('inference service changed during frozen evaluation')
            folder=(args.output/f"{condition['name']}_seed_{seed}").resolve()
            mode=config.get('mode_by_seed',{}).get(str(seed),condition['mode'])
            command=[sys.executable,str(frozen/'scripts/run_extended_sim.py'),'--output',str(folder),'--store',str(frozen_store),
                '--mode',mode,'--friction',str(condition['friction']),'--seed',str(seed),
                '--endpoint',args.endpoint,'--model',args.model,'--clip-endpoint',args.clip_endpoint]
            for key in ('controller_profile','scene_variation','recovery_policy','retrieval_profile','planning_policy'):
                if key in condition:command+=['--'+key.replace('_','-'),condition[key]]
            if condition.get('mixed_shapes'):command+=['--mixed-shapes']
            if condition.get('grounding_repair'):command+=['--grounding-repair']
            if condition.get('multiccd'):command+=['--multiccd']
            if condition.get('visibility_replanning'):command+=['--visibility-replanning']
            if condition.get('visible_goal_fallback'):command+=['--visible-goal-fallback']
            width=config.get('upper_half_width_by_seed',{}).get(str(seed),condition.get('upper_half_width_m'))
            if width is not None:command+=['--upper-half-width',str(width)]
            if not condition['memory']:command+=['--disable-memory']
            if not condition['recovery']:command+=['--disable-recovery']
            if not condition.get('scene_graph',True):command+=['--disable-scene-graph']
            if seed in config.get('record_seeds',[config['seeds'][0]]):command+=['--record']
            case_started=time.time()
            process=subprocess.run(command,env={**os.environ,'MUJOCO_GL':'egl'},capture_output=True,text=True)
            wall_elapsed=time.time()-case_started
            folder.mkdir(exist_ok=True);(folder/'process.log').write_text(process.stdout+process.stderr)
            result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else {}
            perceptions=len(list(folder.glob('observation_*/vlm_response.json')))
            identity_reviews=len(list(folder.glob('observation_*/identity_review_*.json')))
            reasonings=len(result.get('reasoner_calls',[]))
            rows.append({'condition':condition['name'],'seed':seed,'physical_success':result.get('physical_success',False),
                'mode':mode,'controller_profile':result.get('controller_profile','legacy'),
                'recovery_policy':result.get('recovery_policy','paper'),
                'scene_variation':result.get('scene_variation','fixed'),'mixed_shapes':result.get('mixed_shapes',False),
                'planning_policy':result.get('planning_policy','agentic'),'retrieval_profile':result.get('retrieval_profile','haa'),
                'upper_half_width_m':result.get('upper_half_width_m'),'wall_elapsed_s':wall_elapsed,
                'outcome':result.get('outcome','process_error'),'exit_code':process.returncode,
                'attempts':len(result.get('attempts',[])),'failed_attempts':sum(not a['retained'] for a in result.get('attempts',[])),
                'completed_targets':result.get('summary',{}).get('completed_targets',[]),
                'memory_hits':result.get('memory_hits',0),'perception_calls':perceptions,'reasoner_calls':reasonings,
                'identity_review_calls':identity_reviews,
                'vlm_calls':perceptions+reasonings+identity_reviews,'simulated_time_s':result.get('simulated_time_s'),
                'final_objects':result.get('final_objects',{}),'directory':folder.name})
            summary={'expected_cases':len(config['seeds'])*len(config['conditions']),'completed_cases':len(rows),
                'dataset_complete':len(rows)==len(config['seeds'])*len(config['conditions']),
                'source_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
                'knowledge_store':store_relative,'knowledge_store_sha256':hashlib.sha256(args.store.read_bytes()).hexdigest(),
                'config_sha256':hashlib.sha256(config_bytes).hexdigest(),'embedding_space':store.embedding_space,
                'evidence_kind':'mujoco_contact_dynamics','inference_count_contract':'saved_responses_v2',
                'elapsed_wall_s':time.time()-started,'cases':rows}
            atomic_json(args.output/'summary.json',summary)
            print(json.dumps(rows[-1]),flush=True)
            if not result:return 1
    return 0

if __name__=='__main__':raise SystemExit(main())
