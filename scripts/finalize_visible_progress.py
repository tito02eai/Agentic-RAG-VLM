"""Assemble a portable local delivery from completed, immutable r8/r9 evidence.

Never reruns a case, deletes a failure, sends email, or touches hardware.
"""
import argparse
import hashlib
import html
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_extended_benchmark import audit
from scripts.plot_extended_evidence import plot


def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def physics_scene_fingerprint(path):
    """Normalize only the mesh directory, after binding actual mesh contents.

    Separate frozen checkouts have different absolute meshdir strings. Never
    ignore geometry, dynamics or asset-content changes to make them match.
    """
    tree=ET.fromstring(path.read_text());compiler=tree.find('compiler')
    directory=Path(compiler.get('meshdir','.'))
    if not directory.is_absolute():directory=path.parent/directory
    assets={mesh.get('file'):digest(directory/mesh.get('file')) for mesh in tree.findall('.//mesh') if mesh.get('file')}
    compiler.set('meshdir','ASSETS_VERIFIED_BY_CONTENT')
    normalized=ET.tostring(tree)+json.dumps(assets,sort_keys=True).encode()
    return {'raw_xml_sha256':digest(path),'physics_and_assets_sha256':hashlib.sha256(normalized).hexdigest(),
        'mesh_assets':assets,'normalization':'only absolute compiler.meshdir; all referenced mesh bytes included'}


def finalize(campaign,previous,output):
    state=read(campaign/'status.json')
    if state['status']!='complete':raise ValueError('campaign not complete')
    batch=campaign/'qualification_v1';known=campaign/'known_regressions'
    # Correct an explanatory scope sentence in the frozen report generator;
    # keep its original output and record both hashes. No rollout is changed.
    original_report=batch/'REPORT_FROZEN_GENERATOR_CN.md'
    if not original_report.exists():shutil.copy2(batch/'REPORT_CN.md',original_report)
    subprocess.run([sys.executable,str(ROOT/'scripts/summarize_robust_benchmark.py'),str(batch)],check=True)
    write(batch/'REPORT_REVISION.json',{'kind':'post_run_analysis_wording_correction',
        'before_sha256':digest(original_report),'after_sha256':digest(batch/'REPORT_CN.md'),
        'analysis_script_sha256':digest(ROOT/'scripts/summarize_robust_benchmark.py'),
        'reason':'r9 compares full Agent and geometric policy, not only a visibility switch; counts recomputed from identical raw results'})
    for folder in (previous,batch):
        report=audit(folder)
        if not report['passed']:raise ValueError(report['errors'])
    # The known-case wrapper is generated after execution for uniform auditing.
    # Its seed list comes from the frozen campaign launch, not outcome selection.
    expected=[928,929,930,931]
    rows=[];comparison=[]
    for seed in expected:
        label=f'guarded_seed_{seed}';folder=known/label;r=read(folder/'result.json')
        if r['seed']!=seed or not r['physical_success']:raise ValueError('known regression incomplete')
        old=previous/f'mixed_enhanced_seed_{seed}'
        before=physics_scene_fingerprint(old/'scene.xml');after=physics_scene_fingerprint(folder/'scene.xml')
        same=before['physics_and_assets_sha256']==after['physics_and_assets_sha256']
        if not same:raise ValueError('known comparison changed physical scene')
        replies=len(list(folder.glob('observation_*/vlm_response.json')))+len(list(folder.glob('observation_*/identity_review_*.json')))+len(r['reasoner_calls'])
        rows.append({'condition':'known_visible_progress','directory':label,'seed':seed,
            'physical_success':r['physical_success'],'attempts':len(r['attempts']),
            'failed_attempts':sum(not a['retained'] for a in r['attempts']),
            'memory_hits':r['memory_hits'],'vlm_calls':replies})
        comparison.append({'seed':seed,'before_success':read(old/'result.json')['physical_success'],
            'after_success':r['physical_success'],'same_physics_and_assets':same,
            'before_scene':before,'after_scene':after,'excluded_from_new_seed_score':True,
            'visibility_policies':[e['payload']['policy'] for e in r['trace'] if e['event']=='visibility_replan']})
    conf=read(batch/'frozen_config.json');condition={**conf['conditions'][0],'name':'known_visible_progress'}
    known_config={'seeds':expected,'conditions':[condition],'knowledge_store':conf['knowledge_store'],
        'model_health_required':True,'iteration_order':'paired_seed',
        'purpose':'Post-run known-regression audit wrapper; not a new-seed study. Seeds were declared in campaign launch.'}
    write(known/'frozen_config.json',known_config)
    shutil.copy2(campaign/'source.zip',known/'source.zip')
    shutil.copy2(batch/'model_health.json',known/'model_health.json')
    with ZipFile(known/'source.zip') as z:manifest=json.loads(z.read('SOURCE_SHA256.json'))
    write(known/'summary.json',{'cases':rows,'source_sha256':digest(known/'source.zip'),
        'config_sha256':digest(known/'frozen_config.json'),'knowledge_store':conf['knowledge_store'],
        'knowledge_store_sha256':manifest[conf['knowledge_store']],
        'inference_count_contract':'saved_responses_v2','expected_cases':4,'completed_cases':4,'dataset_complete':True})
    known_audit=audit(known);write(known/'artifact_audit.json',known_audit)
    if not known_audit['passed']:raise ValueError(known_audit['errors'])
    write(campaign/'known_comparison.json',comparison)
    plot(batch)
    tests=ET.parse(campaign/'tests.xml').getroot()
    suites=[tests] if tests.tag=='testsuite' else list(tests)
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts['failures'] or counts['errors']:raise ValueError('frozen test failure')
    output.mkdir(parents=True,exist_ok=False)
    source=output/'Agentic_RAG_VLM_source.zip'
    subprocess.run([sys.executable,str(ROOT/'scripts/build_delivery_bundle.py'),'--output',str(source)],check=True)
    # Documentation may advance; executed modules/scripts must match the frozen run.
    with ZipFile(source) as latest, ZipFile(batch/'source.zip') as frozen:
        runtime_names=[n for n in frozen.namelist() if n.startswith(('agentic_rag_vlm/','scripts/')) and n.endswith('.py')
                       and n!='scripts/summarize_robust_benchmark.py']
        changed=[n for n in runtime_names if latest.read(n)!=frozen.read(n)]
        if changed:raise ValueError('delivery differs from executed code: '+str(changed))
    evidence=output/'evidence';evidence.mkdir()
    ignore=shutil.ignore_patterns('source','__pycache__','.pytest_cache')
    shutil.copytree(previous,evidence/'r8',ignore=ignore)
    shutil.copytree(campaign,evidence/'r9',ignore=ignore)
    shutil.copytree(ROOT/'docs',output/'docs')
    metrics=read(batch/'metrics.json')
    write(output/'DELIVERY_STATUS.json',{'status':'simulation_qualified','tests':counts,
        'known_regressions':comparison,'new_seed_metrics':metrics,'known_audit':known_audit,
        'source_sha256':digest(source),'executed_python_files_byte_identical':len(runtime_names),
        'post_run_analysis_revision':read(batch/'REPORT_REVISION.json'),
        'hardware_run':False,'original_paper_trials':False})
    intro='''# Agentic RAG-VLM 交付入口（2026-11-03）

先打开 index.html：完整新种子视频、修复前后回放、结果和审计均在一处。
Agentic_RAG_VLM_source.zip 包含代码、14 条有来源的知识、测试、运行配置及机器人资产。
解压源码后按 docs/CURRENT_DELIVERY.md 安装运行；模型权重需另备。

evidence/r8 保留原 48 例（完整 Agent 8/12，强几何基线 12/12）。
evidence/r9/known_regressions 是已知失败 928–931 的回归，单列；
其 frozen_config 是按已声明种子在事后生成的统一审计封装，不是新种子预注册。
evidence/r9/qualification_v1 是事先冻结的 960–963 新种子配对实验；
所有失败保留，不与旧批次混算。新批和回归有各自逐案例审计。

真实模型回答、RGB-D、指令、接触、独立物理评估与预声明种子视频包含在 evidence。
无需网络即可播放视频和查看审计；JSON 中历史绝对路径是来源记录，图像同时按相对目录保存。
复核：在源码目录执行 `python scripts/audit_extended_benchmark.py /解压目录/evidence/r9/qualification_v1`；
对 r8、r9/known_regressions 也适用。SH A256 清单（SHA256SUMS.json）覆盖本目录文件。

适用范围：已知类别、尺寸、颜色候选和桌面标定，三个刚体整理、保护邻居、平行夹爪。
此次交付不包含真机执行证明、易碎物破坏模型、原论文丢失日志或完整 benchmark 成绩。
向 BluPe 提供软件与演示不等于已部署到硬件。没有自动联系对方。
'''.replace('SH A256','SHA256')
    (output/'README.md').write_text(intro)
    full=metrics['conditions']['mixed_enhanced'];geo=metrics['conditions']['mixed_geometric']
    new_video_ok=read(batch/'mixed_enhanced_seed_960/result.json')['physical_success']
    showcase=('evidence/r9/qualification_v1/mixed_enhanced_seed_960/rollout.mp4' if new_video_ok
              else 'evidence/r9/known_regressions/guarded_seed_928/rollout.mp4')
    showcase_label='新种子 960' if new_video_ok else '已知回归 928（新种子失败仍保留）'
    def score(m):return f"{m['completed']}/{m['n']}"
    def a(href,label):return f'<a href="{html.escape(href)}">{html.escape(label)}</a>'
    cards=''.join(f'<tr><td>{html.escape(name)}</td><td>{score(m)}</td><td>{m["attempts"]}</td><td>{m["vlm_calls"]}</td></tr>'
        for name,m in [('完整框架（含新工具）',full),('强几何基线（同一控制器）',geo)])
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Agentic RAG-VLM · 当前交付</title>
<style>body{margin:0;background:#0e1728;color:#e7edf7;font:16px system-ui;line-height:1.7}main{max-width:1180px;margin:32px auto;padding:0 24px}h1{font-size:30px}h2{font-size:22px}p{color:#bccadd}a{color:#7fd9ff}section{padding:22px;background:#1b2941;border-radius:14px;margin:22px 0}video{width:100%;max-height:570px;background:black}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #42516a;text-align:left}.tag{color:#82e2bb}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>
<main><h1>Agentic RAG-VLM · 当前交付</h1><p>真实 Qwen3.5-4B + CLIP、知识检索、约束推理、接触执行、验证和记忆。2026-11-03 · 完整物理抓放闭环。</p>
<section><h2>先看完整任务：__SHOWCASE_LABEL__</h2><p>移走堆叠上层的黄色目标，重新观察并完成红色方块与品红圆柱的搬运；绿色容器保持稳定。原始画面未剪辑，播放时间不包含模型等待。</p>
<video controls preload="metadata" src="__SHOWCASE__"></video>
<p>__VIEWER__ · __SOURCE__ · __README__</p></section>
<section><h2>本轮验收</h2><p class="tag">__TESTS__ 项测试通过；10 项历史资产测试跳过。已知失败回归 4/4 完整完成；全部物理检查和来源审计通过。</p>
<table><tr><th>新种子配对（960–963）</th><th>完整完成</th><th>抓取尝试</th><th>真实模型回答</th></tr>__CARDS__</table>
<p>4 对新种子仅作工程验收。几何基线共用识别、接触控制和反馈重试。不能据此声称 Agent 在广泛任务上更优。__AUDIT__ · __REPORT__</p></section>
<section><h2>为什么这次改动必要</h2><p>旧版在窄边遮挡下，因下层身份不明而直接停止。新工具连续两次确认已知高处目标和保护物稳定，再由 VLM 审核先完成已知目标；没有给未知物体补身份或编造支撑关系。未知目标保持待办。</p>
<div class="pair"><div><h3>修复前：seed 928 停止</h3><video controls preload="metadata" src="evidence/r8/mixed_enhanced_seed_928/rollout.mp4"></video></div><div><h3>修复后：同一场景完整执行</h3><video controls preload="metadata" src="evidence/r9/known_regressions/guarded_seed_928/rollout.mp4"></video></div></div>
<p>这是已知回归，排除在新种子成绩之外；场景物理配置与网格内容一致。XML 中冻结目录的绝对路径分别保留。__KNOWN__</p></section>
<section><h2>保留旧结果与适用范围</h2><p>旧 r8 完整方案 8/12，几何基线 12/12；全部 48 例保留。当前验证是已知尺寸、颜色候选及桌面标定下的刚体整理，使用平行夹爪。脆弱性是语义限制，没有破裂模型；没有真机结果。</p><p>__OLD__ · __STATUS__ · __FRAMEWORK__</p></section></main></html>'''
    substitutions={'__VIEWER__':a('evidence/r9/qualification_v1/viewer.html','同步决策回放与全部新案例'),
        '__SHOWCASE__':showcase,'__SHOWCASE_LABEL__':showcase_label,
        '__SOURCE__':a(source.name,'源码与知识库 ZIP'),'__README__':a('README.md','交接说明'),
        '__TESTS__':str(counts['tests']-counts['skipped']),'__CARDS__':cards,
        '__AUDIT__':a('evidence/r9/qualification_v1/artifact_audit.json','新种子审计'),
        '__REPORT__':a('evidence/r9/qualification_v1/REPORT_CN.md','详细结果'),
        '__KNOWN__':a('evidence/r9/known_comparison.json','逐例修复记录'),
        '__OLD__':a('evidence/r8/viewer.html','旧批完整回放'),
        '__STATUS__':a('DELIVERY_STATUS.json','机器可读交付状态'),
        '__FRAMEWORK__':a('docs/CURRENT_DELIVERY.md','框架与复现命令')}
    for key,value in substitutions.items():page=page.replace(key,value)
    (output/'index.html').write_text(page)
    hashes={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}
    write(output/'SHA256SUMS.json',hashes)
    archive=output.with_suffix('.zip')
    with ZipFile(archive,'x',compression=ZIP_DEFLATED,compresslevel=3) as z:
        for path in sorted(output.rglob('*')):
            if path.is_file():z.write(path,str(path.relative_to(output.parent)))
    with ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('delivery ZIP CRC failure')
    write(output.parent/(archive.name+'.sha256.json'),{'file':archive.name,'sha256':digest(archive),'bytes':archive.stat().st_size})
    print(json.dumps({'entry':str(output/'index.html'),'archive':str(archive),'source':str(source),'files':len(hashes)},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    finalize(args.campaign.resolve(),args.previous.resolve(),args.output.resolve())
