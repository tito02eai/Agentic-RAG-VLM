"""Add actual command-level outcomes to a new, provenance-checked KB file."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agentic_rag_vlm.experience import ExperienceStore
from agentic_rag_vlm.experience_import import import_rollouts

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rollouts',type=Path,nargs='+',required=True);args=p.parse_args()
    if args.base.resolve().parent!=args.output.resolve().parent:
        p.error('base and output must share their portable attachment directory')
    store=ExperienceStore.load(args.base);store.validate_sources(args.base.parent)
    print(json.dumps(import_rollouts(store,args.rollouts,output=args.output),indent=2))
