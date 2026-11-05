"""Known-case before/after evidence page; videos and model replies remain raw."""
import argparse
import html
import json
from pathlib import Path
import subprocess


def build(root):
    before=Path('qualification_v1/mixed_enhanced_seed_500')
    after=Path('grounding_regression_v2_500')
    evidence=json.loads((root/'grounding_regression_evidence.json').read_text())
    if not evidence['passed']:raise ValueError('initial observations or model output differ')
    for folder in (before,after):
        preview=root/folder/'preview.png'
        if not preview.exists():
            subprocess.run(['ffmpeg','-v','error','-i',str(root/folder/'rollout.mp4'),'-frames:v','1',str(preview)],check=True)
    def video(folder):
        return f'<video controls preload="metadata" poster="{folder}/preview.png"><source src="{folder}/rollout.mp4" type="video/mp4"></video>'
    def code(value):return html.escape(json.dumps(value,ensure_ascii=False,indent=2))
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Agentic RAG-VLM · 身份冲突恢复</title>
<style>body{{margin:0;background:#111827;color:#e5e7eb;font:16px system-ui}}main{{max-width:1280px;margin:36px auto;padding:0 24px}}h1{{font-size:28px}}p{{color:#afbed0;line-height:1.8}}section{{display:grid;grid-template-columns:1fr 1fr;gap:22px}}article{{background:#1e293b;padding:20px;border-radius:12px}}video{{width:100%;border-radius:8px;background:black}}pre{{white-space:pre-wrap;color:#9de4d3;line-height:1.6}}a{{color:#7dd3fc}}.images{{display:flex;gap:20px;align-items:center;margin:24px 0}}img{{width:150px;border-radius:8px}}small{{color:#a6b3c5}}@media(max-width:800px){{section{{grid-template-columns:1fr}}}}</style>
<main><h1>Agentic RAG-VLM · 从身份冲突到完整抓放</h1>
<p>已知失败案例 seed 500 的回归演示。初始 RGB-D、四张候选图、首个提示词和模型原始回答均一致；开启复核后，模型逐个重新检查冲突候选，再执行三物体任务。此例不计入新的留出种子实验。</p>
<section><article><h2>单次识别：执行前停止</h2>{video(before)}<p>模型给红色遮挡边缘和绿色容器都分配 vase；实例身份重复，框架停止。</p><pre>{code(evidence['original_labels'])}</pre><a href="{before}/observation_001/vlm_response.json">首次原始模型回答</a></article>
<article><h2>有界复核：完整抓放</h2>{video(after)}<p>{evidence['review_calls']} 次真实单图复核消除冲突，随后完成抓放、接触失败恢复和会话记忆复用。</p><pre>{code(evidence['reviewed_labels'])}</pre><a href="{after}/observation_001/grounding_decision.json">有效标签与复核来源</a> · <a href="{after}/result.json">完整执行结果</a></article></section>
<div class="images"><img src="{before}/observation_001/candidate_0.png"><p>原始红色候选只露出边缘。复核使用这张相同图像；有效身份来自模型重新判断。代码没有按颜色或候选序号填写身份。</p></div>
<p><a href="grounding_regression_evidence.json">查看相同输入和首次回答的校验证据</a> · <a href="qualification_v1/viewer.html">30 案例回放</a> · <a href="grounding_qualification_v1/viewer.html">身份复核新种子对照</a></p>
<small>原始仿真时间视频，保留失败，不含模型等待。已知案例用于回归说明；整体表现以两个独立冻结批次的全部结果为准。</small></main></html>'''
    output=root/'grounding_regression_comparison.html';output.write_text(page);print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    build(parser.parse_args().directory)
