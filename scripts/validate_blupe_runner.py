"""Reproduce local official-runner integration checks without a hardware session."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.check_runner_compatibility import inspect_runner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runner-checkout',type=Path,required=True)
    p.add_argument('--recorded-request',type=Path)
    p.add_argument('--protected-request',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();checkout=args.runner_checkout.resolve();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    compatibility=inspect_runner(checkout/'codex-runner/src/remote_yam')
    (output/'compatibility.json').write_text(json.dumps(compatibility,indent=2))
    if not compatibility['compatible']:raise SystemExit('Official runner contract changed; inspect compatibility.json')
    env=os.environ.copy();env['BLUPE_RUNNER_CHECKOUT']=str(checkout);env['BLUPE_VALIDATION_OUTPUT']=str(output)
    env.pop('BLUPE_RECORDED_PREVIEW',None)
    env.pop('BLUPE_RECORDED_PROTECTED',None)
    if args.recorded_request:env['BLUPE_RECORDED_PREVIEW']=str(args.recorded_request.resolve())
    if args.protected_request:env['BLUPE_RECORDED_PROTECTED']=str(args.protected_request.resolve())
    cmd=[sys.executable,'-m','pytest','-q','tests/test_official_runner.py',
         '--junitxml='+str(output/'tests.xml'),'--tb=short']
    with (output/'tests.log').open('w') as log:
        completed=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    suite=ET.parse(output/'tests.xml').getroot().find('testsuite')
    counts={key:int(suite.get(key,0)) for key in ('tests','failures','errors','skipped')}
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
    report={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'passed':completed.returncode==0,
        'official_repository':'https://github.com/andlyu/blupe-remote-bots','official_revision':revision,
        'upstream_modified':bool(subprocess.check_output(['git','status','--porcelain'],cwd=checkout,text=True).strip()),
        'tests':counts,'real_model_recorded_preview_requested':bool(args.recorded_request),
        'physical_session_created':False,'public_queue_joined':False,'partner_deployment_performed':False,
        'scope':'Official controller + SDK + Playground runtime tests; local fixture commands are not physical success',
        'source_sha256':{str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob('*.py')) if not any(x in path.parts for x in ('output','.venv','__pycache__'))}}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
    raise SystemExit(completed.returncode)


if __name__=='__main__':main()
