"""Plot saved raw physics and tactile samples, without rerunning or smoothing."""
import argparse
import json
from pathlib import Path


def plot(root,seed=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = json.loads((root / 'summary.json').read_text())
    config = json.loads((root / 'frozen_config.json').read_text())
    seed = config['seeds'][0] if seed is None else seed
    condition='complex_shift_full' if any(c['condition']=='complex_shift_full' for c in summary['cases']) else 'mixed_enhanced'
    case = next(c for c in summary['cases'] if c['condition'] == condition and c['seed'] == seed)
    folder = root / case['directory']
    result = json.loads((folder / 'result.json').read_text())
    raw = [json.loads(line) for line in (folder / 'private_evaluator.jsonl').read_text().splitlines()]
    colors = {'cube2': '#d49b00', 'cube': '#d33939', 'cube3': '#a231ad'}
    names = {'cube2': 'Yellow block', 'cube': 'Red block', 'cube3': 'Magenta cylinder' if result.get('mixed_shapes') else 'Magenta block'}
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), gridspec_kw={'height_ratios': [2, 1.4, 1.3]})
    for name, body in [('cube2', 'target_cube2'), ('cube', 'target_cube'), ('cube3', 'target_cube3')]:
        axes[0].plot([e['time'] for e in raw], [e['objects'][body][2] * 1000 for e in raw],
                     color=colors[name], label=names[name], linewidth=1.3)
    axes[0].axhline(100, color='#6b7280', linestyle=':', linewidth=.8, label='Lift threshold')
    axes[0].set(ylabel='Object center height (mm)', ylim=(0, 175),
                title=f'Frozen complex task, seed {seed}: raw simulation evidence')
    axes[0].legend(loc='upper right', fontsize=8, ncol=2)
    for index, attempt in enumerate(result['attempts']):
        start, end = attempt['start_time_s'], attempt['end_time_s']
        axes[0].axvspan(start, end, color=colors[attempt['target']], alpha=.07)
        values = [{'time': attempt['initial_tactile_time_s'], **attempt['initial_tactile']},
                  *attempt.get('lift_monitor_samples',[]),*attempt.get('proof_samples',[]),*attempt['retention_samples']]
        axes[1].scatter([s['time'] for s in values], [sum(s['normal_force_n'].values()) for s in values],
                        color=colors[attempt['target']], s=10)
        changes=[(start,attempt.get('initial_motor_force_cap_n',attempt['motor_force_cap_n']))]
        changes += [(e['time'],e['after_n']) for e in attempt.get('force_adjustments',[])]
        changes += [(end,attempt['motor_force_cap_n'])]
        axes[1].step([x[0] for x in changes],[x[1] for x in changes],where='post',color='#6b7280',linestyle='--',linewidth=1)
        axes[1].text(start, attempt['motor_force_cap_n'] + .07, f"{attempt['motor_force_cap_n']:.2f} N", fontsize=8)
        y = {'cube2': 2, 'cube': 1, 'cube3': 0}[attempt['target']]
        axes[2].barh(y, end-start, left=start, color=colors[attempt['target']], height=.35,
                     alpha=1 if attempt['retained'] else .35, edgecolor='black', linewidth=.5)
        label = {'haa_rag': 'RAG', 'recovery_l1': 'L1 recovery', 'episodic_memory': 'Memory',
                 **{f'recovery_force_ladder_{i}':f'Repair {i}' for i in (1,2,3)}}.get(attempt['source'], attempt['source'])
        axes[2].text(start if not attempt['retained'] else end + .7,
                     y + .29 if not attempt['retained'] else y,
                     label + (' / not verified' if not attempt['retained'] else ' / held'),
                     va='center', fontsize=8)
    axes[1].set(ylabel='Total normal force (N)', ylim=(-.03, max([1.2]+[a['motor_force_cap_n']*1.3 for a in result['attempts']])),
                title='Dots: saved tactile samples; dashed line: commanded motor force cap', xlabel='')
    axes[2].set(yticks=[0, 1, 2], yticklabels=[names[k] for k in ('cube3','cube','cube2')],
                xlabel='Simulation time (s)', title='Pick attempts and actual recorded decision source')
    for ax in axes:
        ax.set_xlim(0, result['simulated_time_s'])
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.15)
    fig.text(.02, .012, 'Raw simulation-time samples; inference pauses excluded. Heights are private evaluation only. No attachment / pose reset.', fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, 1))
    for suffix in ('png', 'pdf'):
        fig.savefig(root / f'contact_recovery_evidence.{suffix}', dpi=160)
    plt.close(fig)
    print(root / 'contact_recovery_evidence.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--seed',type=int)
    args=parser.parse_args();plot(args.directory,args.seed)
