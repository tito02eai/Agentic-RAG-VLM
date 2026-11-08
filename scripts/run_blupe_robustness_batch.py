"""Freeze and run a bounded, local production-path robustness qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentic_rag_vlm.experience import atomic_json
from scripts.run_blupe_robustness_case import CASES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runner-checkout', type=Path, required=True)
    p.add_argument('--port', type=int, default=18795)
    p.add_argument('--candidate-config', type=Path)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(code/'scripts/build_delivery_bundle.py'), '--output', str(out/'source.zip')], check=True)
    with ZipFile(out/'source.zip') as z:
        z.extractall(out/'source')
    candidate_config = None
    if args.candidate_config:
        candidate_config = out/'candidates.json'
        atomic_json(candidate_config, json.loads(args.candidate_config.read_text()))
    model_health = {}
    for name, origin in [('vlm','http://127.0.0.1:18074'), ('clip','http://127.0.0.1:18072')]:
        with urlopen(origin+'/health', timeout=10) as r:
            model_health[name] = json.load(r)
    protocol = {'declared_at_unix': time.time(), 'cases': list(CASES), 'seed': 1010,
        'source_sha256': hashlib.sha256((out/'source.zip').read_bytes()).hexdigest(),
        'model_health': model_health, 'known_layout': True,
        'scope': 'functional fault qualification; each case once, no reliability/success-rate estimate',
        'physical_robot_contacted': False, 'development_excluded': True,
        'pass_conditions': {
            'control': 'complete and independent lift/release/support/destination/protected checks',
            'empty_grasp': 'physical open-actuator fault, retention rejection, no place or false completion',
            'slip': 'fault after actual bilateral lift, retention rejection, no place or false completion',
            'memory_moved': 'two distinct same-category targets; force moves pending target >3cm; verified first action; actual memory hit and L3; fresh position; physical placements',
            'memory_failed_reuse': 'verified first action; actual memory reuse fails retention; memory evicted; no subsequent place or false completion',
            'ack_lost_once': 'accepted response replaced by 503 once; SDK retries identical ID/body; no duplicate execution; task completes',
            'ack_lost_persistent': 'all first-packet acknowledgements replaced by 503; same-ID/body retries; one physical packet; stop without false completion',
            'feedback_outage': '503 on session reads after dispatch; no new motion without completion; stop without false completion',
            'ws_reconnect': 'real WS close before initial observation; reconnect same session; complete once',
            'operator_stop': 'Stop after >=3 real wire waypoints; executor aborts; no new packet or false completion'},
        'all_cases_require': 'raw model evidence, no protected-object/arm-table/self-contact violation; no public hardware'}
    protocol['candidate_config_sha256'] = (
        hashlib.sha256(candidate_config.read_bytes()).hexdigest() if candidate_config else None)
    atomic_json(out/'protocol.json', protocol)
    statuses = []
    for case in CASES:
        started = time.time()
        command = [sys.executable, '-u', str(out/'source/scripts/run_blupe_robustness_case.py'),
            '--runner-checkout', str(args.runner_checkout.resolve()),
            '--config', str(out/'source/configs/blupe_provider.example.json'),
            '--output', str(out/case), '--case', case, '--seed', '1010', '--port', str(args.port)]
        if candidate_config:
            command.extend(['--candidate-config', str(candidate_config)])
        print('START '+case, flush=True)
        with (out/(case+'.log')).open('w') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        receipt_path = out/case/'receipt.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        statuses.append({'case': case, 'passed': receipt.get('passed') is True and process.returncode == 0,
            'returncode': process.returncode, 'agent_outcome': receipt.get('agent_outcome'),
            'task_success': receipt.get('task_success'), 'wall_time_s': time.time()-started})
        summary = {'cases': statuses, 'complete': len(statuses) == len(CASES),
            'checks_passed': sum(c['passed'] for c in statuses), 'total_cases': len(CASES),
            'scope': protocol['scope'], 'physical_robot_contacted': False}
        atomic_json(out/'summary.json', summary)
        print(json.dumps(statuses[-1]), flush=True)
    return 0 if all(s['passed'] for s in statuses) else 1


if __name__ == '__main__':
    raise SystemExit(main())
