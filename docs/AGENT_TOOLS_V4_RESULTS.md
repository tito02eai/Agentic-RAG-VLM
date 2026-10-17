# Agentic RAG-VLM Challenge v4：模型显式工具路由

## 一句话结论

新版不再根据 G1/G2/G3/G4 实验编号自动分派工具，而是先让 Qwen 根据图像、任务和工具接口
显式选择工具，再由工具检查公开输入并生成可执行约束。10 个冻结种子上，Full Agent 四类任务
均为 10/10，四类工具选择准确率均为 100%。

## 这次具体修正了什么

旧版运行时虽然不读取 evaluator truth，但工具调用类型由场景编号触发，容易被质疑为实验脚本
提前知道该调用哪个模块。v4 将流程改为：

`Public RGB-D → Qwen Tool Router → Qwen Planner → Tool Preconditions → Executed Decision`

- 路由回合只返回 `requested_tools`，不生成任务答案；
- 规划回合独立生成抓法、安全参数或恢复候选；
- 工具注册表根据候选中的输出绑定与公开输入检查能否执行；
- 未知工具、缺少公开输入或字段不匹配都会被拒绝并留下收据；
- 场景编号仅保留在私有评估器中，不参与工具分派；
- 原始路由、原始候选、无语义改写的 JSON 结构归一化、执行结果与拒绝原因全部写入
  `public_trace.jsonl`。

## 正式实验

- 10 个冻结种子，100 个方法 rollout 的固定种子配对设计；
- 246 次真实本地 Qwen3.5-4B 多模态调用，最终 endpoint error 为 0；
- 260 个公开事件通过 public/private 隔离审计；
- 40 个 Full-Agent rollout 共执行 60 次工具调用，原始模型路由、请求字段与执行收据逐项一致；
- 41 项自动测试通过。

| 场景 | Full Agent | 主要对照 | 对照结果 | 直接说明 |
|---|---:|---|---:|---|
| G1 可供性检索与手型路由 | 10/10 | VLM-only | 1/10 | 只看图直接猜抓法不稳定，检索卡与手型路由有效 |
| G2 受保护关系与安全动作 | 10/10 | no-graph | 0/10 | 能识别风险不等于能生成方向正确、幅值足够的动作约束 |
| G3 变化归因与任务记忆 | 10/10 | no-memory / no-replan | 0/10 / 5/10 | memory 防止回滚；replan 只在待完成目标变化时需要 |
| G4 联合任务 | 10/10 | local / fixed skill | 0/10 / 1/10 | 抓法、安全、变化和记忆必须同时满足，单点能力不能替代闭环 |

Full Agent 在 target-move、no-change、irrelevant-move 三种事件切片中均全部通过，
`false-replan=0`。因此恢复结果不是通过“任何变化都重规划”获得的。

## 配对统计

- G1 Full vs VLM-only：McNemar exact `p=0.00390625`；
- G2 Full vs no-graph：`p=0.00195312`；
- G3 Full vs no-memory：`p=0.00195312`；
- G3 Full vs no-replan：`p=0.0625`；
- G4 Full vs local：`p=0.00195312`；
- G4 Full vs fixed skill：`p=0.00390625`。

10 个种子的 Full 通过率 Wilson 95% CI 为 `[0.72, 1.00]`。这些统计支持冻结协议内的模块
差异，不应外推为任意环境下 100% 成功。

## 视频证据升级

当前推荐主线是 `video/vlabench_defense_reel/` 下的 84 秒 VLABench 总览片工程；其最终 MP4
尚未渲染，且化学任务的接触/安全收尾 P0 仍需处理。具体状态见 `../KIRO_HANDOFF.md` 和
`../video/vlabench_defense_reel/DEFENSE_GUIDE.md`。

以下 67 秒视频是 Challenge v4 阶段的历史机制成片，不再作为当前完整操作主片：
`output/defense_suite/agentic_rag_vlm_full_defense_reel_v9.mp4`

视频为 67.0 秒、1920×1080、30 fps，包含三类容易讲清的优势证据：

1. 同一颜色、不同几何的对象由 HAA-RAG 路由到不同手型；
2. 黄色物体明确作为“受保护的易碎约束”：固定腕向虽未穿透，但 1.5 mm 净空低于 3 mm
   安全门槛；关系条件化候选保留 +51.9 mm；
3. 蓝色目标移动 80.0 mm 后，旧路点因对齐误差被门禁拒绝，Agentic 方法只重规划尚未完成的目标；
   E3 两侧各 417 帧均通过可见网格审计，不以手指穿模制造失败。

配对失败由可测净空或对齐门限触发，不使用脚本化掉落制造差异。视频验证报告与 SHA-256
位于同目录的 `agentic_rag_vlm_full_defense_reel_v9.validation.json`。

## 结论边界

本实验验证真实多模态高层决策、模型工具选择、公开观测约束执行、任务记忆和失败恢复。连续运输
仍使用明确披露的运动学/contact-assisted 技能代理；G0-B 接触动力学抓取保持
`NOT_ADMITTED`，不据此宣称实机或 sim-to-real 成功。

## 证据位置

- 正式结果：`output/qwen35_challenge_v4_agent_routing_main/CHALLENGE_V4_REPORT.md`
- 逐运行公开轨迹：`output/qwen35_challenge_v4_agent_routing_main/runs/*/public_trace.jsonl`
- 隔离审计：`output/qwen35_challenge_v4_agent_routing_main/validation_receipt.json`
- 工具路由审计：`output/qwen35_challenge_v4_agent_routing_main/agent_tool_validation_receipt.json`
- 当前 VLABench 视频工程：`video/vlabench_defense_reel/`
- 历史机制视频：`output/defense_suite/agentic_rag_vlm_full_defense_reel_v9.mp4`
- 简历证据审计：`output/resume_evidence_v1/receipt.json`
