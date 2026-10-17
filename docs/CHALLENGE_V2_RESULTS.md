# 光华 Agentic RAG-VLM Challenge v2 结果

## 结论

Challenge v2 已完成 10 个冻结 seed、100 个方法 rollout、192 次真实 Qwen3.5-4B
多模态调用。实验加入可供性几何变体、易碎物距离/方位、target/no-change/irrelevant
三类事件、指令改写，以及 G4 的位姿、质量和摩擦联合随机化。所有最终 endpoint 调用
成功；100 个 run、100 行结果和 260 条公开事件通过隔离审计。

| 场景 | Full | 对照 | 关键解释 |
|---|---:|---:|---|
| G1 Affordance | 10/10 | VLM-only 0/10 | Full 在四种几何组合中正确使用 shape-conditioned RAG 卡 |
| G2 Safety Graph | 4/10 | No-Graph 1/10 | Full 提高可执行 offset，但仍有方位推理失败，差异未显著 |
| G3 Recovery | 7/10 | No-Memory 0/10；No-Replan 7/10 | Full 通过全部 target-move；No-Replan 只通过无需恢复事件 |
| G4 Combined | 2/10 | Local 0/10；Skill-only 0/10 | Full 平均完成 0.60 的模块，仍受组合错误和 40.75 s P50 延迟限制 |

配对 McNemar exact：G1 Full/VLM-only `p=0.00195312`；G2 Full/No-Graph
`p=0.375`；G3 Full/No-Memory `p=0.015625`；G3 Full/No-Replan `p=1`；
G4 的两个比较均为 `p=0.5`。因此只能对 G1 和 G3-memory 做显著差异表述，G2/G4
只能报告正向趋势和能力边界。

## 事件条件结果

- G3 Full：target-move 3/3，no-change 4/4，irrelevant-fragile-move 0/3；
- G3 No-Replan：target-move 0/3，no-change 4/4，irrelevant-fragile-move 3/3；
- G4 Full：target-move 0/3，no-change 1/4，irrelevant-fragile-move 1/3。

所以 G3 的两个 7/10 不能解释为 Full 和 No-Replan 等价：Full 具有真正的目标位移
恢复能力，但当前 4B Planner 会把无关易碎物变化错误升级为重规划，false-replan 为
3/10。该错误作为自然失败保留，没有通过提示词针对 seed 修掉。

## 证据入口

- 主报告：`output/qwen35_challenge_v2_main/CHALLENGE_V2_REPORT.md`；
- 原始表：`output/qwen35_challenge_v2_main/challenge_results.csv`；
- 冻结配置：`output/qwen35_challenge_v2_main/frozen_config.json`；
- 完整性审计：`output/qwen35_challenge_v2_main/validation_receipt.json`；
- 每条运行：`output/qwen35_challenge_v2_main/runs/<scene_seed_condition>/`；
- 可复现实验协议：`configs/guanghua_challenge_v2.json`。

## 声明边界

这些结果验证真实 VLM、RAG、场景图、执行记忆、变化监控、Agentic 分解和有界修复，
但仍属于高层机制与校准运动技能代理实验，不是接触动力学抓取成功率。答辩中的完整
成功视频继续用于定性系统案例，不能拿单视频替代本表的 100-rollout 定量结果。
