"""Summarize all frozen cases; compare memory only on paired completions."""
import argparse
import csv
import json
import statistics
from pathlib import Path

NAMES = {
    'complex_nominal_full': '三物体原摩擦 / 完整框架',
    'complex_shift_full': '三物体较滑表面 / 完整框架',
    'complex_shift_no_memory': '三物体较滑表面 / 关闭记忆',
    'complex_shift_no_recovery': '三物体较滑表面 / 关闭恢复',
    'single_shift_full': '单物体较滑表面 / 完整框架',
}


def summarize(root):
    summary = json.loads((root / 'summary.json').read_text())
    config = json.loads((root / 'frozen_config.json').read_text())
    if not summary['dataset_complete']:
        raise ValueError('Formal batch is incomplete; do not publish a final report.')
    groups, result_map = {}, {}
    for case in summary['cases']:
        groups.setdefault(case['condition'], []).append(case)
        result_map[case['condition'], case['seed']] = json.loads(
            (root / case['directory'] / 'result.json').read_text())
    totals = {}
    for name, cases in groups.items():
        results = [result_map[name, c['seed']] for c in cases]
        totals[name] = {
            'n': len(cases), 'completed': sum(c['physical_success'] for c in cases),
            **{key: sum(c[key] for c in cases) for key in
               ('attempts', 'failed_attempts', 'memory_hits', 'vlm_calls')},
            'recovery_events': sum(sum(e['event'] == 'contact_failure_recovery'
                                       for e in r['trace']) for r in results),
            'mean_simulation_seconds': statistics.mean(c['simulated_time_s'] for c in cases),
            'max_protected_motion_m': max(r['max_protected_displacement_m'] for r in results),
            'max_target_table_penetration_m': max(r['max_target_table_penetration_m'] for r in results),
        }
    full = {c['seed']: c for c in groups['complex_shift_full']}
    no_memory = {c['seed']: c for c in groups['complex_shift_no_memory']}
    pairs = [(full[s], no_memory[s]) for s in config['seeds']
             if full[s]['physical_success'] and no_memory[s]['physical_success']]
    comparison = {'declared_pairs': len(config['seeds']),
                  'both_completed_pairs': len(pairs),
                  'excluded_seeds': [s for s in config['seeds']
                                     if not (full[s]['physical_success'] and no_memory[s]['physical_success'])]}
    for key in ('attempts', 'failed_attempts', 'vlm_calls', 'simulated_time_s'):
        if pairs:
            a, b = (statistics.mean(c[i][key] for c in pairs) for i in (0, 1))
            comparison[key] = {'full_mean': a, 'no_memory_mean': b,
                               'reduction_fraction': (b - a) / b if b else None}
    failures = []
    for case in summary['cases']:
        if case['physical_success']:
            continue
        r = result_map[case['condition'], case['seed']]
        stop = next((e['payload'] for e in reversed(r['trace']) if e['event'] == 'safe_stop'), {})
        failures.append({'condition': case['condition'], 'seed': case['seed'],
                         'completed_targets': case['completed_targets'], 'stop': stop,
                         'failed_checks': [k for k, v in r['checks'].items() if not v]})
    models, parse_errors = set(), 0
    for r in result_map.values():
        for call in r['reasoner_calls']:
            models.add(call.get('response_model', 'unknown'))
            parse_errors += bool(call.get('parse_error'))
        for receipt in r['receipts']:
            if receipt['kind'] == 'perception':
                value = receipt['payload']['vlm_result']
                models.add(value.get('response_model', 'unknown'))
                parse_errors += bool(value.get('parse_error'))
    metrics = {'conditions': totals, 'paired_memory': comparison, 'failed_cases': failures,
               'response_models': sorted(models), 'parse_errors': parse_errors,
               'vlm_calls_total': sum(c['vlm_calls'] for c in summary['cases']),
               'source_sha256': summary['source_sha256'],
               'config_sha256': summary['config_sha256']}
    (root / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n')
    with (root / 'cases.csv').open('w', newline='') as stream:
        keys = ('condition', 'seed', 'physical_success', 'outcome', 'attempts',
                'failed_attempts', 'memory_hits', 'vlm_calls', 'simulated_time_s', 'directory')
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader(); writer.writerows(summary['cases'])
    lines = ['# Agentic RAG-VLM：多物体、恢复与记忆仿真结果', '',
             '运行日期：2026-11-02。以下是本次冻结方案的新实验，不是论文原始数据。', '',
             f"共 {summary['completed_cases']} 个已声明案例；每个条件使用种子 "
             + ', '.join(map(str, config['seeds'])) + '，每个案例只运行一次，失败计入结果。', '',
             '| 条件 | 完成 | 抓取尝试 | 失败尝试 | 恢复事件 | 记忆命中 | VLM 调用 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, t in totals.items():
        lines.append(f"| {NAMES[name]} | {t['completed']}/{t['n']} | {t['attempts']} | "
                     f"{t['failed_attempts']} | {t['recovery_events']} | {t['memory_hits']} | {t['vlm_calls']} |")
    lines += ['', '## 任务与闭环', '',
              '机器人需将三个自由物体分别抓起、搬运、放到三个指定位置。其中黄色块叠在红色块上；'
              '目标队列最初要求红色在先，框架根据 RGB-D 支撑关系先移走黄色块，再处理红色块和独立品红块。'
              '绿色容器是保护物，不是抓取目标。', '',
              '初始夹持参数取自另外三个原摩擦成功样本。较滑条件只改变物体摩擦；'
              '首次接触后物体未保持在夹爪中，由实际接触力和夹爪开度判失败。'
              '恢复前先释放、撤离，并以两次新 RGB-D 观测确认稳定；L1 将电机力上限从 0.80 调到 0.96 N。'
              '后续目标只有在已验证抓放成功后才可复用这一会话经验。', '',
              '## 记忆对照', '',
              f"完整框架与关闭记忆有 {comparison['both_completed_pairs']}/{comparison['declared_pairs']} 对同时完成，"
              f"未同时完成的种子：{comparison['excluded_seeds']}。以下尝试数/调用数比较仅使用同时完成的配对，"
              '所有案例的完成情况均保留在上表。']
    if pairs:
        for key, label in [('attempts', '每任务抓取尝试'), ('failed_attempts', '每任务失败尝试'),
                           ('vlm_calls', '每任务 VLM 调用'), ('simulated_time_s', '每任务仿真时间（秒）')]:
            v = comparison[key]
            reduction = '不适用' if v['reduction_fraction'] is None else f"{v['reduction_fraction']:.1%}"
            lines.append(f"- {label}：完整框架 {v['full_mean']:.2f}，关闭记忆 {v['no_memory_mean']:.2f}；减少 {reduction}。")
    lines += ['', '## 未完成案例（全部保留）', '']
    for failure in failures:
        lines.append(f"- {failure['condition']} / seed {failure['seed']}：已完成 {failure['completed_targets']}；"
                     + json.dumps(failure['stop'], ensure_ascii=False) + '。')
    if not failures:
        lines.append('本批次没有未完成案例。')
    lines += ['', '## 可以与不能据此说明什么', '',
              '- 支持关系改变执行顺序；失败恢复处理真实接触保持失败；记忆对照衡量同会话兼容物体的参数复用。',
              '- 完成率包含低层 IK 与放置技能的影响，不能把关闭记忆组的所有未完成归因于记忆推理。'
              '同完成配对上的尝试数和调用数更直接说明本任务中的复用收益；不作统计显著性或总体优越性声明。',
              '- 使用真实 Qwen3.5-4B 感知/决策调用和 CLIP ViT-L/14 检索；低层为固定 IK/接触技能，未训练 VLA。',
              '- 这是受控工程验证：已知形状尺寸、颜色候选分割、位置仅小幅扰动、理想重力补偿、近似碰撞几何；'
              '每条件只有五个种子。摩擦变化是主动设计的压力条件，不能推出开放世界泛化或总体成功率。',
              '- 物理验证覆盖当前平行夹爪的 top 抓取与 L1 夹持修正；不等同于灵巧手、所有失败类型或 L2/L3 的物理验证。',
              '- 夹持读数来自 MuJoCo 接触力，不代表平台真机已提供触觉/力控接口；未执行真机操作。',
              '- 没有物体绑定或初始化后位姿重置。全部条件固定使用 elliptic cone / 10 NoSlip iterations，'
              '用于减小数值接触漂移；不是实物摩擦标定。', '',
              '## 原始证据', '',
              '- [逐案例表](cases.csv)、[结构化指标](metrics.json)、[完整汇总](summary.json)。',
              '- [同步视频回放](viewer.html)：每个条件固定展示第一个声明种子，保留失败；播放为仿真时间，不含模型等待。',
              '- [接触与恢复测量图](contact_recovery_evidence.png)：原始物体高度、保存的接触力采样及动作来源。',
              '- 每个案例目录保存感知原图、模型原始输出、工具/动作记录、触觉采样和独立物理评估。',
              '- [独立审计](artifact_audit.json)需与本报告一同查看；审计不通过时不得称为已验收。', '',
              f"冻结源码 SHA256：`{summary['source_sha256']}`。", '',
              f"冻结配置 SHA256：`{summary['config_sha256']}`。", '']
    (root / 'REPORT_CN.md').write_text('\n'.join(lines))
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    print(json.dumps(summarize(parser.parse_args().directory), ensure_ascii=False, indent=2))
