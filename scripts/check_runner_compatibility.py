"""Inspect an installed BluPe runner contract without importing or running it."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def inspect_runner(root):
    root=Path(root)
    required={'session.py':{'get_robot_observation','get_queue_snapshot','get_session','submit_trajectory','get_episode_trace'},
              'providers.py':{'public_config','infer'}}
    checks={};hashes={}
    for name,methods in required.items():
        text=(root/name).read_text();tree=ast.parse(text)
        found={n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        checks[name]=sorted(methods-found);hashes[name]=hashlib.sha256(text.encode()).hexdigest()
    text=(root/'controller.py').read_text();tree=ast.parse(text)
    has_hook=any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='getattr'
        and len(n.args)>1 and isinstance(n.args[1],ast.Constant) and n.args[1].value=='run_session' for n in ast.walk(tree))
    hashes['controller.py']=hashlib.sha256(text.encode()).hexdigest()
    return {'compatible':has_hook and not any(checks.values()),'run_session_hook':has_hook,
        'missing_methods':checks,'source_sha256':hashes,'execution_performed':False,
        'scope':'static interface check; runtime contract covered by local bridge tests; no robot/model call'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runner-source',type=Path,required=True)
    p.add_argument('--output',type=Path);args=p.parse_args();result=inspect_runner(args.runner_source)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['compatible'] else 2)
