"""Read-only local simulation/model admission; never joins a hardware queue."""
import argparse
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agentic_rag_vlm.services import vlm_endpoint,clip_endpoint,resolve_model,health_url
from agentic_rag_vlm.experience import ExperienceStore,atomic_json


def inspect(args):
    report={'ready_for_simulation':False,'hardware_requests_made':False,
        'platform_paper_chain_integrated':True,'hardware_trial_performed':False,'blockers':[],'packages':{}}
    for name,expected in [('mujoco','3.2.2'),('numpy','1.26.4'),('Pillow','12.3.0'),
                          ('scipy','1.14.0'),('mediapy','1.2.0')]:
        try:
            actual=importlib.metadata.version(name);report['packages'][name]=actual
            if actual!=expected:report['blockers'].append(f'{name}: expected {expected}, found {actual}; install configs/extended_sim_dependencies.txt')
        except importlib.metadata.PackageNotFoundError:report['blockers'].append(f'{name}: not installed')
    if not shutil.which('ffmpeg'):report['blockers'].append('ffmpeg missing; required for recording')
    try:
        store=ExperienceStore.load(args.store);store.validate_sources(args.store.parent)
        if store.domain!='mjlab_yam_retention_v1':raise ValueError('wrong simulation executor domain')
        report['knowledge']={'records':len(store.records),'domain':store.domain,'embedding_space':store.embedding_space}
        model=resolve_model(args.endpoint,args.model)
        with urlopen(health_url(args.endpoint),timeout=10) as response:health=json.load(response)
        report['vlm']=health
        if health.get('requested_model')!=model:raise ValueError('configured VLM differs from served model')
        with urlopen(args.clip_endpoint.rstrip('/')+'/health',timeout=10) as response:clip=json.load(response)
        report['clip']=clip
        if clip.get('embedding_space')!=store.embedding_space or clip.get('dimension')!=store.dimension:
            raise ValueError('CLIP weights/dimension differ from the knowledge store; do not relabel embeddings')
    except Exception as error:report['blockers'].append(f'{type(error).__name__}: {error}')
    report['ready_for_simulation']=not report['blockers']
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint',default=vlm_endpoint());p.add_argument('--model')
    p.add_argument('--clip-endpoint',default=clip_endpoint())
    p.add_argument('--store',type=Path,default=ROOT/'knowledge_base/yam_verified_diverse_r8.json')
    p.add_argument('--output',type=Path)
    args=p.parse_args();report=inspect(args)
    if args.output:atomic_json(args.output,report)
    print(json.dumps(report,indent=2));return 0 if report['ready_for_simulation'] else 2


if __name__=='__main__':raise SystemExit(main())
