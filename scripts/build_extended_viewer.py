"""Local evidence viewer: unedited video plus synchronized recorded events."""
import argparse,json,subprocess
from pathlib import Path

HTML='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Agentic RAG-VLM 仿真证据</title>
<style>body{margin:0;background:#111827;color:#e5e7eb;font:16px system-ui}main{max-width:1250px;margin:32px auto;padding:0 24px}h1{font-size:27px}p{color:#aebccc;line-height:1.6}select,button{padding:10px;border:1px solid #475569;border-radius:8px;background:#1e293b;color:white}section{display:grid;grid-template-columns:2fr 1fr;gap:22px;margin:22px 0}video{width:100%;background:black;border-radius:12px}.card{background:#1e293b;border-radius:12px;padding:20px}dl{display:grid;grid-template-columns:100px 1fr;gap:14px}dt{color:#94a3b8}dd{margin:0}.badge{color:#60d9be;font-weight:600}table{width:100%;border-collapse:collapse}td,th{padding:12px;text-align:left;border-bottom:1px solid #334155}a{color:#7dd3fc}button{margin:5px}small{color:#aebccc}@media(max-width:800px){section{grid-template-columns:1fr}}</style>
<main><h1>Agentic RAG-VLM · 实验回放与证据</h1><p>原始 MuJoCo 画面，右侧状态直接读取保存的事件。红色、黄色和品红色为操作目标，绿色容器受保护；可在下拉菜单切换任务和条件。播放时间为仿真时间，不含模型等待。</p>
<select id="select"></select><section><div><video id="video" controls></video><div id="jump"></div></div><div class="card"><h2 id="title"></h2><dl><dt>当前阶段</dt><dd id="stage"></dd><dt>当前对象</dt><dd id="target"></dd><dt>动作来源</dt><dd class="badge" id="source"></dd><dt>电机力上限</dt><dd id="force"></dd><dt>本次抬升检查</dt><dd id="retained"></dd><dt>最近任务调整</dt><dd id="decision"></dd></dl><p id="result"></p><small>力值是仿真夹爪电机上限；不是实机指尖力。保留失败，不据此推断广泛任务上的成功率。</small><p><a id="raw" target="_blank">查看原始结果 JSON</a></p></div></section>
<div class="card"><h2>所有已声明案例</h2><p id="progress"></p><table><thead><tr><th>条件</th><th>完成/总数</th><th>抓取尝试总数</th><th>失败尝试总数</th><th>记忆命中总数</th><th>模型调用总数</th></tr></thead><tbody id="summary"></tbody></table><p><a href="REPORT_CN.md">完整结果与适用范围</a> · <a href="artifact_audit.json">独立审计</a></p></div><div class="card" style="margin-top:22px"><h2>失败与恢复的原始测量</h2><p>高度曲线只供独立评估；执行决策未读取这些物体真实坐标。虚线是电机力上限，散点是已保存的接触力采样。</p><img src="contact_recovery_evidence.png" style="width:100%" alt="物体高度、接触力和动作来源的原始采样"></div></main>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),video=document.getElementById('video'),select=document.getElementById('select');
const names={complex_nominal_full:'原摩擦条件 · 完整框架',complex_shift_full:'较滑表面 · 完整框架',complex_shift_no_memory:'较滑表面 · 关闭记忆',complex_shift_no_recovery:'较滑表面 · 关闭恢复',single_shift_full:'单物体 · 失败恢复',mixed_enhanced:'混合物体 · 增强方案',mixed_paper:'混合物体 · 原固定恢复规则',mixed_no_memory:'混合物体 · 增强方案关闭记忆',mixed_strict_grounding:'混合物体 · 关闭身份冲突复核',mixed_no_proof:'混合物体 · 关闭试提工具',mixed_strict_visibility:'混合物体 · 关闭可见性重规划'};
Object.assign(names,{mixed_fixed_skill:'关闭检索 · 标称技能',mixed_geometric:'RGB-D 高度排序与相同抓取控制'});
const modes={stacked_sort:'堆叠整理',flat_sort:'平铺整理',single_shift:'单物体',calibration:'标称样本'};
const objects={cube:'红色方块',cube2:'黄色方块',cube3:'品红色物体'},sources={haa_rag:'HAA-RAG 检索',episodic_memory:'会话记忆',recovery_l1:'L1 反思修正',recovery_l2:'L2 方法切换',recovery_l3:'L3 重新规划',recovery_force_ladder_1:'接触反馈修正 1',recovery_force_ladder_2:'接触反馈修正 2',recovery_force_ladder_3:'接触反馈修正 3'};
const labels={initial_view_clearance:'移开空手，准备观察',pregrasp:'接近目标',descend:'下降',close:'闭合夹爪',proof_lift:'低高度试提',proof_lift_recheck:'调整夹持后重新试提',proof_return_to_support:'试提后返回支撑面',lift:'抬升与夹持检查',release_failed_grasp:'验收未通过后松手',recovery_retreat:'撤离并重新观察',settle_final:'稳定观察',transport:'搬运',place_compliant:'承载感应放置',placement_unload_probe:'视觉与微量松指检查',release:'释放',retract:'撤离',park:'移开手臂'};
Object.assign(sources,{fixed_skill:'固定标称技能（无知识检索）'});
for(const [index,c] of data.clips.entries()){let option=document.createElement('option');option.value=index;option.textContent=(names[c.condition]||c.condition)+' / '+(modes[c.mode]||c.mode)+' / seed '+c.seed+(c.upper_half_width_m?' / 上层半宽 '+(c.upper_half_width_m*1000).toFixed(0)+' mm':'');select.append(option)}
let current;
function load(){
 current=data.clips[+select.value];video.src=current.video;video.poster=current.poster;
 document.getElementById('title').textContent=names[current.condition]||current.condition;
 document.getElementById('raw').href=current.raw;
 document.getElementById('result').textContent=`本例：${current.success?'完整完成':'未完整完成'}；${current.attempts.length}次抓取尝试，${current.hits}次记忆命中，${current.invalidations}次失败记忆失效，${current.identity_reviews}次身份复核。${current.stop_reason||''}`;
 let jump=document.getElementById('jump');jump.replaceChildren();
 for(const [i,a] of current.attempts.entries()){
  let b=document.createElement('button');b.textContent=`尝试${i+1} · ${objects[a.target]||a.target}`;
  b.onclick=()=>{video.currentTime=Math.max(0,a.start_time_s-current.first_frame_time_s);video.play()};jump.append(b)
 }update()
}
function update(){
 if(!current)return;const t=video.currentTime+current.first_frame_time_s;
 let stage=current.stages.findLast(x=>x.time<=t);document.getElementById('stage').textContent=stage?(labels[stage.stage]||stage.stage):'初始化';
 let a=current.attempts.findLast(x=>x.start_time_s<=t);
 document.getElementById('target').textContent=a?(objects[a.target]||a.target):'—';
 document.getElementById('source').textContent=a?(sources[a.source]||a.source):'—';
 let force=a?(a.initial_motor_force_cap_n??a.motor_force_cap_n):null;
 if(a)for(const adjustment of a.force_adjustments||[])if(adjustment.time<=t)force=adjustment.after_n;
 document.getElementById('force').textContent=force===null?'—':force.toFixed(2)+' N';
 document.getElementById('retained').textContent=!a?'—':t<a.end_time_s?'尚未完成检查':a.retained?'夹持留存验收通过':'夹持留存验收未通过';
 let decision=(current.agent_decisions||[]).findLast(x=>x.time<=t);
 document.getElementById('decision').textContent=decision?(decision.policy==='visible_target_progress_then_reobserve'?`连续两次观察确认目标稳定，先完成已识别的高处目标${objects[decision.target]||decision.target}，随后重新观察；未推断未知物体的支撑关系。`:`下层身份尚未确认，先搬移已识别的${objects[decision.target]||decision.target}，随后重新观察。`):'—';
}
select.onchange=load;video.ontimeupdate=update;select.value=String(Math.max(0,data.clips.findIndex(c=>['complex_shift_full','mixed_enhanced'].includes(c.condition)&&c.seed===data.default_seed)));
document.getElementById('progress').textContent=`已运行 ${data.completed}/${data.expected} 个案例。表中计数为各条件总计，模型调用含校验失败前保存的回答；视频使用预先声明的录制种子。`;
for(const [key,c] of Object.entries(data.summary)){let row=document.createElement('tr');for(const value of [names[key]||key,c.success+'/'+c.n,c.attempts,c.failed,c.hits,c.calls]){let td=document.createElement('td');td.textContent=value;row.append(td)}document.getElementById('summary').append(row)}load();
</script></html>'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--default-seed',type=int);args=p.parse_args()
    summary=json.loads((args.directory/'summary.json').read_text());config=json.loads((args.directory/'frozen_config.json').read_text())
    data={'clips':[],'summary':{},'completed':summary['completed_cases'],'expected':summary['expected_cases'],
          'default_seed':args.default_seed if args.default_seed is not None else config['seeds'][0]}
    for case in summary['cases']:
        totals=data['summary'].setdefault(case['condition'],dict(n=0,success=0,attempts=0,failed=0,hits=0,calls=0))
        folder=args.directory/case['directory']
        identity_reviews=len(list(folder.glob('observation_*/identity_review_*.json')))
        calls=len(list(folder.glob('observation_*/vlm_response.json')))+identity_reviews+case['reasoner_calls']
        totals['n']+=1;totals['success']+=case['physical_success'];totals['attempts']+=case['attempts'];totals['failed']+=case['failed_attempts'];totals['hits']+=case['memory_hits'];totals['calls']+=calls
        if case['seed'] not in config.get('record_seeds',[config['seeds'][0]]) or not (folder/'rollout.mp4').exists():continue
        poster=folder/'preview.png'
        if not poster.exists():
            subprocess.run(['ffmpeg','-v','error','-i',str(folder/'rollout.mp4'),'-frames:v','1','-y',str(poster)],check=True)
        result=json.loads((folder/'result.json').read_text());stages=[];previous=None
        observation_times={r['payload']['observation_id']:r['payload']['receipt'].get('captured_at',0)
            for r in result['receipts'] if r['kind']=='observation'}
        decisions=[{'time':observation_times[e['payload']['witness']['observation_id']],
                    'target':e['payload']['selected_target'],'deferred_targets':e['payload']['deferred_targets'],'policy':e['payload'].get('policy')}
                   for e in result['trace'] if e['event']=='visibility_replan']
        stop=next((e['payload'] for e in reversed(result['trace']) if e['event']=='safe_stop'),None)
        stop_reason=('停止原因：'+str(stop)) if stop else ('' if result['physical_success'] else
            '独立物理评估未通过：'+', '.join(k for k,v in result['checks'].items() if not v))
        for line in (folder/'private_evaluator.jsonl').read_text().splitlines():
            e=json.loads(line)
            if e['stage']!=previous:stages.append({'time':e['time'],'stage':e['stage']});previous=e['stage']
        data['clips'].append({'condition':case['condition'],'seed':case['seed'],'success':case['physical_success'],
            'video':case['directory']+'/rollout.mp4','poster':case['directory']+'/preview.png',
            'raw':case['directory']+'/result.json','stages':stages,
            'attempts':result['attempts'],'hits':case['memory_hits'],'mode':result['mode'],'first_frame_time_s':.04,
            'agent_decisions':decisions,
            'upper_half_width_m':result.get('upper_half_width_m'),
            'identity_reviews':identity_reviews,'stop_reason':stop_reason,
            'invalidations':sum(e['event']=='contact_failure_recovery' and bool(e['payload']['evidence'].get('memory_invalidated')) for e in result['trace'])})
    output=args.directory/'viewer.html';output.write_text(HTML.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')))
    print(output)

if __name__=='__main__':main()
