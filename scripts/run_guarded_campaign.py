"""Detached, auditable r6 campaign: test, freeze, known regressions, new pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentic_rag_vlm.experience import atomic_json
from agentic_rag_vlm.services import vlm_endpoint,clip_endpoint,resolve_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/guarded_sim_qualification.json')
    parser.add_argument('--known-seeds',type=int,nargs='*',default=[502,603])
    parser.add_argument('--known-visibility-replanning',action='store_true')
    parser.add_argument('--known-visible-goal-fallback',action='store_true')
    parser.add_argument('--known-upper-half-width',type=float)
    parser.add_argument('--store',type=Path)
    parser.add_argument('--endpoint',default=vlm_endpoint())
    parser.add_argument('--model',help='served model; otherwise AGENTIC_VLM_MODEL or /health identity')
    parser.add_argument('--clip-endpoint',default=clip_endpoint())
    args = parser.parse_args()
    try:args.model=resolve_model(args.endpoint,args.model)
    except ValueError as error:parser.error(str(error))
    configuration=json.loads(args.config.read_text())
    knowledge=args.store or ROOT/configuration.get('knowledge_store','knowledge_base/yam_retention_observed.json')
    output = args.output.resolve()
    output.mkdir(exist_ok=False, parents=True)
    started = time.time()
    state = {'status': 'running', 'phase': 'starting', 'started_unix': started,
             'hardware_run': False, 'original_paper_trials': False, 'steps': []}

    def status(phase, **fields):
        state.update(phase=phase, elapsed_wall_s=time.time()-started, **fields)
        atomic_json(output/'status.json', state)
        print(json.dumps({'phase': phase, **fields}), flush=True)

    def run(name, command, *, cwd, timeout=10800):
        status(name)
        with (output/(name+'.log')).open('w') as log:
            process = subprocess.Popen(command, cwd=cwd, env={**os.environ, 'MUJOCO_GL': 'egl'},
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
                raise RuntimeError(f'{name} exceeded its declared wall-clock limit; stopped own process group')
        state['steps'].append({'name': name, 'return_code': code, 'command': command})
        return code

    try:
        archive = output/'source.zip'
        if run('freeze_source', [sys.executable, str(ROOT/'scripts/build_delivery_bundle.py'),
                '--output', str(archive)], cwd=ROOT, timeout=120):
            raise RuntimeError('source archive failed')
        frozen = output/'source'
        with ZipFile(archive) as z:
            manifest = json.loads(z.read('SOURCE_SHA256.json'))
            if any(hashlib.sha256(z.read(name)).hexdigest()!=digest for name, digest in manifest.items()):
                raise RuntimeError('source manifest mismatch')
            z.extractall(frozen)
        status('source_frozen', source_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
               source_files=len(manifest))
        if run('tests', [sys.executable, '-m', 'pytest', '-q', '--junitxml='+str(output/'tests.xml')],
               cwd=frozen, timeout=900):
            raise RuntimeError('frozen extracted-source tests failed; benchmark not started')
        known = output/'known_regressions'; known.mkdir()
        for seed in args.known_seeds:
            directory = known/f'guarded_seed_{seed}'
            command = [sys.executable, str(frozen/'scripts/run_extended_sim.py'), '--output', str(directory),
                '--store', str(frozen/knowledge.resolve().relative_to(ROOT)), '--seed', str(seed),
                '--mode', 'stacked_sort', '--controller-profile', 'guarded', '--scene-variation', 'mixed_contact',
                '--mixed-shapes', '--recovery-policy', 'force_ladder', '--grounding-repair', '--multiccd', '--record',
                '--endpoint',args.endpoint,'--model',args.model,'--clip-endpoint',args.clip_endpoint]
            if args.known_visibility_replanning:
                command+=['--visibility-replanning']
            if args.known_visible_goal_fallback:command+=['--visible-goal-fallback']
            if args.known_upper_half_width is not None:command+=['--upper-half-width',str(args.known_upper_half_width)]
            code = run(f'known_{seed}', command, cwd=frozen, timeout=1200)
            result_path = directory/'result.json'
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            state.setdefault('known_regressions', []).append({'seed': seed,
                'physical_success': result.get('physical_success', False), 'return_code': code,
                'excluded_from_new_seed_score': True})
            if not result.get('physical_success'):
                raise RuntimeError(f'known regression {seed} failed under real VLM; new cohort not started')
        batch = output/'qualification_v1'
        code = run('new_seed_pairs', [sys.executable, str(frozen/'scripts/run_extended_benchmark.py'),
            '--output', str(batch), '--config', str(frozen/args.config.resolve().relative_to(ROOT)),
            '--store', str(frozen/knowledge.resolve().relative_to(ROOT)),
            '--endpoint',args.endpoint,'--model',args.model,'--clip-endpoint',args.clip_endpoint], cwd=frozen)
        if code:
            raise RuntimeError('new cohort runner did not finish; partial cases retained')
        code = run('artifact_audit', [sys.executable, str(frozen/'scripts/audit_extended_benchmark.py'), str(batch)], cwd=frozen)
        audit = json.loads((output/'artifact_audit.log').read_text())
        atomic_json(batch/'artifact_audit.json', audit)
        if code or not audit['passed']:
            raise RuntimeError('artifact audit failed; results require review')
        for name, script in [('report', 'summarize_robust_benchmark.py'), ('viewer', 'build_extended_viewer.py')]:
            if run(name, [sys.executable, str(frozen/'scripts'/script), str(batch)], cwd=frozen):
                raise RuntimeError(f'{name} generation failed')
        metrics = json.loads((batch/'metrics.json').read_text())
        status('complete', status='complete', conditions=metrics['conditions'],
               formal_model_replies=metrics['vlm_calls_total'], formal_cases=metrics['formal_case_count'])
        return 0
    except Exception as error:
        status('needs_attention', status='needs_attention', error_type=type(error).__name__, error=str(error))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
