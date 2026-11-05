"""Check model backend and optional official runner source; no robot admission."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agentic_rag_vlm.platform.runner_provider import load_provider
from scripts.check_runner_compatibility import inspect_runner


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--runner-source',type=Path);p.add_argument('--output',type=Path);args=p.parse_args()
    result={'ready_for_software_integration':False,'hardware_session_created':False,'errors':[]}
    try:
        result['provider']=load_provider(args.config).prepare()
        if args.runner_source:
            result['runner']=inspect_runner(args.runner_source)
            if not result['runner']['compatible']:raise ValueError('runner interface incompatible')
        result['ready_for_software_integration']=True
    except Exception as error:result['errors'].append({'type':type(error).__name__,'message':str(error)})
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['ready_for_software_integration'] else 2)
