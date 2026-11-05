"""Descriptive paired analysis of r5; retain all stops and infrastructure faults."""
import argparse
import csv
import json
from pathlib import Path
import statistics

NAMES={'mixed_enhanced':'增强方案','mixed_paper':'原固定恢复规则（同一改进底层）','mixed_no_memory':'增强方案关闭记忆',
       'mixed_strict_grounding':'单次识别（关闭身份冲突复核）','mixed_no_proof':'r5 抬升与中心估计（同一接触求解）',
       'mixed_strict_visibility':'关闭可见性重规划（其余相同）','mixed_fixed_skill':'关闭知识检索（标称技能）',
       'mixed_geometric':'RGB-D 高度排序 + 同一抓取控制'}
MODES={'stacked_sort':'堆叠整理','flat_sort':'平铺整理'}


def summarize(root):
    summary=json.loads((root/'summary.json').read_text())
    if not summary['dataset_complete']:raise ValueError('batch incomplete; no final report')
    cases=summary['cases'];details={};groups={};failures=[]
    config=json.loads((root/'frozen_config.json').read_text())
    modes=[mode for mode in MODES if any(c['mode']==mode for c in cases)]
    for case in cases:
        label=case['directory'];folder=root/label
        result=json.loads((folder/'result.json').read_text())
        public=[json.loads(s) for s in (folder/'public_trace.jsonl').read_text().splitlines()]
        raw_perceptions=len(list(folder.glob('observation_*/vlm_response.json')))
        identity_reviews=len(list(folder.glob('observation_*/identity_review_*.json')))
        invalidations=sum(e['event']=='contact_failure_recovery' and bool(e['payload']['evidence'].get('memory_invalidated')) for e in result['trace'])
        stop=next((e['payload'] for e in reversed(result['trace']) if e['event']=='safe_stop'),None)
        infrastructure=bool(stop and stop.get('error_type') in {'URLError','HTTPError','TimeoutError','ConnectionError'})
        extra={'runner_reported_vlm_calls':case['vlm_calls'],
               'occlusion_level':config.get('occlusion_level_by_seed',{}).get(str(case['seed']),'original'),
               'attempts_started':result.get('attempts_started',len(result['attempts'])),
               'preventive_force_adjustments':sum(e['event']=='retention_force_adjustment' for e in public),
               'visibility_replans':sum(e['event']=='visibility_replan' for e in result['trace']),
               'vlm_calls':raw_perceptions+identity_reviews+len(result['reasoner_calls']),
               'identity_review_calls':identity_reviews,
               'rejected_perception_responses':raw_perceptions-sum(e['kind']=='perception' for e in result['receipts']),
               'memory_invalidations':invalidations,'ik_refinements':sum(e['event']=='ik_bound_refinement' for e in public),
               'unload_probe_steps':sum(e['event']=='placement_unload_probe' for e in public),
               'fine_place_steps':sum(e['event']=='placement_load' and e['tcp_z_m']<.024-1e-8 for e in public),
               'infrastructure_failure':infrastructure}
        details[case['condition'],case['seed']]={**case,**extra}
        groups.setdefault(case['condition'],[]).append(details[case['condition'],case['seed']])
        if not case['physical_success']:
            failures.append({'condition':case['condition'],'seed':case['seed'],'mode':case['mode'],
                'completed_targets':case['completed_targets'],'stop':stop,'infrastructure_failure':infrastructure,
                'failed_checks':[key for key,value in result['checks'].items() if not value],
                'max_target_table_penetration_m':result['max_target_table_penetration_m']})
    totals={}
    for name,rows in groups.items():
        totals[name]={'n':len(rows),'completed':sum(r['physical_success'] for r in rows),
            **{key:sum(r[key] for r in rows) for key in ('attempts','failed_attempts','memory_hits','vlm_calls',
                'memory_invalidations','ik_refinements','unload_probe_steps','fine_place_steps','infrastructure_failure',
                'identity_review_calls','rejected_perception_responses')},
            'by_layout':{mode:{'n':sum(r['mode']==mode for r in rows),
                'completed':sum(r['physical_success'] and r['mode']==mode for r in rows)} for mode in modes}}
        for key in ('attempts_started','preventive_force_adjustments','visibility_replans'):
            totals[name][key]=sum(r[key] for r in rows)
    paired={}
    for control in [name for name in groups if name!='mixed_enhanced']:
        seeds=sorted(s for c,s in details if c=='mixed_enhanced')
        both=[s for s in seeds if details['mixed_enhanced',s]['physical_success'] and details[control,s]['physical_success']]
        comparison={'declared_pairs':len(seeds),'both_completed_pairs':len(both),'included_seeds':both,
            'excluded_seeds':[s for s in seeds if s not in both]}
        for key in ('attempts','failed_attempts','vlm_calls','simulated_time_s','preventive_force_adjustments'):
            if both:
                a=statistics.mean(details['mixed_enhanced',s][key] for s in both)
                b=statistics.mean(details[control,s][key] for s in both)
                comparison[key]={'enhanced_mean':a,'control_mean':b,'reduction_fraction':(b-a)/b if b else None}
        paired[control]=comparison
    metrics={'study':config.get('study','controller_recovery'),'conditions':totals,'paired':paired,'failed_cases':failures,
        'vlm_calls_total':sum(c['vlm_calls'] for c in details.values()),
        'runner_reported_vlm_calls_total':sum(c['vlm_calls'] for c in cases),'formal_case_count':len(cases),
        'counting_note':'All saved responses counted, including failed perception before a receipt was created; original summary unmodified.',
        'source_sha256':summary['source_sha256'],'config_sha256':summary['config_sha256'],
        'model_health':json.loads((root/'model_health.json').read_text())}
    if config.get('occlusion_level_by_seed'):
        levels=list(dict.fromkeys(config['occlusion_level_by_seed'].values()))
        metrics['by_occlusion']={name:{level:{'n':sum(r['occlusion_level']==level for r in rows),
            'completed':sum(r['occlusion_level']==level and r['physical_success'] for r in rows)}
            for level in levels} for name,rows in groups.items()}
    if (root/'artifact_audit.json').exists():
        audit=json.loads((root/'artifact_audit.json').read_text())
        metrics['audit_passed']=audit['passed'];metrics['audit_warnings']=audit.get('warnings',[])
    (root/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    with (root/'cases.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['condition','seed','mode','physical_success','outcome','attempts',
            'failed_attempts','memory_hits','memory_invalidations','vlm_calls','ik_refinements','unload_probe_steps',
            'fine_place_steps','infrastructure_failure','identity_review_calls','rejected_perception_responses',
            'runner_reported_vlm_calls','directory','occlusion_level','upper_half_width_m','planning_policy',
            'retrieval_profile','wall_elapsed_s','preventive_force_adjustments','visibility_replans'],extrasaction='ignore')
        writer.writeheader();writer.writerows(details.values())
    stratified=config.get('study')=='stratified_occlusion'
    progress=config.get('study')=='visible_goal_progress'
    visibility=config.get('study')=='visibility_replanning' or stratified or progress
    guarded=config.get('study') in {'guarded_retention','visibility_replanning','stratified_occlusion','visible_goal_progress'}
    lines=[('# r9：已识别目标继续推进与新种子配对验证' if progress else '# r8：遮挡分层、检索与几何基线' if stratified else '# r7：部分可见场景中的任务重规划' if visibility else '# r6：试提、执行反馈与记忆复用' if guarded else '# r5：'+('身份冲突复核独立对照' if config.get('study')=='grounding_review' else '混合物体、接触恢复与记忆复用')), '',
        ('日期：2026-11-03。' if guarded else '日期：2026-11-02。')+'全部条件共用接触求解；这是新工程验证，不是原论文数据或完整 benchmark 成绩。', '',
        '| 条件 | 完整完成 | 抓取尝试总数 | 失败尝试 | 记忆命中 | 记忆失效 | VLM 调用 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,t in totals.items():
        lines.append(f"| {NAMES[name]} | {t['completed']}/{t['n']} | {t['attempts']} | {t['failed_attempts']} | {t['memory_hits']} | {t['memory_invalidations']} | {t['vlm_calls']} |")
    if guarded:
        lines+=['','留存记录可能不含物理保护中断前未返回的抓取；实际启动数另列，不把中断漏算。',
                '预防调节属于同次抓取内控制，不计为新抓取；它不等同于已经发生的滑脱。',
                '| 条件 | 实际启动抓取 | 预防调节 |','|---|---:|---:|']
        for name,t in totals.items():
            lines.append(f"| {NAMES[name]} | {t['attempts_started']} | {t['preventive_force_adjustments']} |")
    if visibility:
        lines+=['','## 可见性重规划触发次数','',
                '缺失目标保持待办与未识别状态；只搬移已识别上层目标，再重新感知。']
        for name,t in totals.items():
            lines.append(f"- {NAMES[name]}：{t['visibility_replans']} 次。")
    lines+=['','## 分布局结果','','| 条件 | '+' | '.join(MODES[m] for m in modes)+' |',
            '|---|'+'---:|'*len(modes)]
    for name,t in totals.items():
        cells=[f"{t['by_layout'][mode]['completed']}/{t['by_layout'][mode]['n']}" for mode in modes]
        lines.append(f"| {NAMES[name]} | {' | '.join(cells)} |")
    if 'by_occlusion' in metrics:
        lines+=['','## 物理遮挡分层','',
                '通过改变上层方块的水平占地改变可见边缘；高度、接触参数和底层控制配对相同。']
        for name,levels in metrics['by_occlusion'].items():
            lines.append(f"- {NAMES[name]}："+'；'.join(f"{level} {x['completed']}/{x['n']}" for level,x in levels.items()))
    lines+=['','## 配对效率（仅双方完整完成）','',
        '未完成和接口故障仍计入上表；效率比较列出所有排除种子，不把提前停止当作更高效率。']
    for name,comparison in paired.items():
        lines+=['',f"与 {NAMES[name]}：双方完成 {comparison['both_completed_pairs']}/{comparison['declared_pairs']} 对；"
            f"纳入 {comparison['included_seeds']}，排除 {comparison['excluded_seeds']}。"]
        for key,label in [('attempts','每任务尝试'),('failed_attempts','每任务失败尝试'),('vlm_calls','每任务 VLM 调用'),('simulated_time_s','每任务仿真秒数'),('preventive_force_adjustments','每任务预防调节')]:
            if key in comparison:
                v=comparison[key];change='不适用' if v['reduction_fraction'] is None else f"{v['reduction_fraction']:.1%}"
                lines.append(f"- {label}：增强 {v['enhanced_mean']:.2f}，对照 {v['control_mean']:.2f}，减少 {change}。")
    lines+=['','## 未完成案例（全部保留）','']
    for f in failures:
        lines.append(f"- {f['condition']} / {f['mode']} / seed {f['seed']}："
            f"已完成 {f['completed_targets']}；接口故障={f['infrastructure_failure']}；"
            +json.dumps(f['stop'],ensure_ascii=False)+f"；未通过判据 {f['failed_checks']}，"
            f"最大目标/桌面穿透 {f['max_target_table_penetration_m']*1000:.3f} mm。")
    if not failures:lines.append('本轮没有未完成案例。')
    lines+=['','## 适用范围','',
        f"- {len(config['seeds'])} 个新种子，每个 {len(config['conditions'])} 个配对条件；每个案例从空记忆开始、仅运行一次，不以成功重跑替换。",
        '- 两个方块加圆柱，独立位置/目的地扰动、独立质量 55–75 g、物体摩擦 0.6–1.0。'
        '指垫摩擦 0.7，MuJoCo 合并后有效滑动摩擦至少 0.7；真实参数未输入策略。',
        ('- r6 区分实际滑脱与预防性余量不足；显式授权且观测低脆弱性、双侧接触和正向载荷时，'
         '同次抓取内最多三次 20% 调节，上限 1.6 N。不是原论文 L2 手型切换。'
         if guarded else '- 接触反馈调参仅用于已确认 slip、低脆弱性、支持的夹爪控制；三次以内、力上限 <=1.6 N。'
         '这不是声称执行了原论文的 L2 手型切换。'),
        ('- r9 共用感知、相机、物理求解、试提和有界反馈重试；完整框架保留检索、记忆、场景约束与 VLM 任务审核，几何基线按可见高度排序。这是组合比较，不能隔离每个模块；已知回归与新种子分开报告。' if progress else
         '- r8 分别比较可见性重规划开关、知识检索开关及较强几何基线。几何基线能按可见高度排序，使用相同 Qwen 身份识别、抓取控制与有界接触重试，但不使用 Agent 任务审核、检索或记忆。组合基线差异不能全部归给某个模块。' if stratified else
         '- r7 两组共用相同可重排整理任务、检索、记忆、真实模型、身份复核、试提和物理求解；仅可见性重规划开关不同，不与旧批次混算。' if visibility else
         '- r6 对照共用多点接触、相机和真实模型；增强条件组合了试提、受限调节和倾斜补偿，不能隔离每项贡献，也不与 r5 混算。' if guarded else
         '- 该追加对照只改变身份复核开关，共用底层、反馈恢复和记忆；不与先前 30 例混算总体成功率。'
         if config.get('study')=='grounding_review' else
         '- 原规则对照共用改进底层，比较固定恢复与支持的反馈策略；不代表完整原论文基线或超越原论文。'),
        '- 记忆失效是观测到复用失败后删除旧参数，再通过完整成功验证后写入新经验；'
        '并不保证复用在所有异质对象上都有正收益。',
        '- 保持最终放置、抬升、支撑顺序、碰撞/位移与穿透判据。未独立检验 HAA-RAG 相对其他检索器，'
        '未验证开放世界分割、长期跨会话记忆、灵巧手或真机。',
        '- 机器人理想重力补偿、近似碰撞几何、elliptic cone / 10 NoSlip 的接触模型假设仍在；'
        '触觉来自仿真，不是平台实机接口。',
        '- 仿真视频包含失败，播放时间不含推理等待；不能把视频时长当作端到端吞吐量。', '',
        '## 证据入口','',
        '- [同步回放](viewer.html)、[逐案例表](cases.csv)、[完整指标](metrics.json)、[审计](artifact_audit.json)。',
        f"- 固定录制种子 {config['record_seeds']}，每条件各一段，不按结果选片。",
        '- 每例包含原始 RGB-D / VLM 回答、接触力、失败恢复和记忆事件、独立物理轨迹；'
        '微量松指额外保存支持检查图像、深度和相机参数。',
        f"- 冻结源码 SHA256：`{summary['source_sha256']}`。",
        f"- 模型身份：`{metrics['model_health']['model']}`。",'']
    lines+=['## 推理计数审计','',
        f"原运行器汇总 {metrics['runner_reported_vlm_calls_total']} 次；按全部保存回答计数 {metrics['vlm_calls_total']} 次。"
        '旧版在感知校验失败前尚未建立有效回执，存在漏计；原 summary.json 未修改，'
        '本报告及 cases.csv 使用原始回答文件的完整计数，包含额外身份复核。','']
    if metrics.get('audit_warnings'):
        count=len({w['case'] for w in metrics['audit_warnings']})
        lines+=['## 末态采样限制','',f'{count} 个完成案例的末态落在 20 ms 日志间隙。'
            '审计只复核最后采样的目的地/支撑判据，并列出坐标差；未声称精确最终时刻坐标已逐位核验。'
            '原始轨迹未补写，最新代码另增加实际末态快照。','']
    (root/'REPORT_CN.md').write_text('\n'.join(lines))
    return metrics


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    print(json.dumps(summarize(parser.parse_args().directory),ensure_ascii=False,indent=2))
