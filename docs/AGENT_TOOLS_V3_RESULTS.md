# 光华 Agentic RAG-VLM Challenge v3：正式结果

## 结论

Challenge v3 验证的不是“VLM 看见物体后直接输出 IK”，而是三个可执行、可审计的
Agent 工具契约：

1. `retrieval_to_dexterous_skill_router_v1`：将公开 RGB-D 几何与 HAA-RAG 卡片转成
   灵巧手 synergy；
2. `protected_relation_to_motion_constraint_v1`：将黄色 Protected Glass 的场景图关系
   转成 away-offset、高度和力限值；
3. `role_aware_change_and_memory_verifier_v1`：按公开位移阈值、对象角色和 pending-target
   memory 决定是否重规划。

正式主集固定 seeds 301–310，共 100 个方法 rollout、195 次真实本地
Qwen3.5-4B 多模态调用。Full Agent 在四个场景均为 10/10；最终 endpoint error 为 0。

## 冻结实验矩阵

| 场景 | 被验证的能力 | Full | 对照/消融 | 配对统计 |
|---|---|---:|---:|---:|
| G1 | RAG 卡片是否真正改变手型技能 | 10/10 | VLM-only 0/10 | McNemar p=0.00195 |
| G2 | 保护关系是否成为可执行运动约束 | 10/10 | no-graph 0/10 | McNemar p=0.00195 |
| G3 | 目标变化、无变化、无关变化的归因与恢复 | 10/10 | no-memory 0/10；no-replan 5/10 | p=0.00195；p=0.0625 |
| G4 | 可供性、安全关系、记忆的联合随机化 | 10/10 | local 0/10；skill-only 1/10 | p=0.00195；p=0.00391 |

每个 10/10 的 Wilson 95% CI 为 [0.72, 1.00]，因此答辩中应说“冻结的 10 个主种子
全部通过”，不应外推为任意布局 100% 成功。

## 为什么它比“传统视觉检测 + IK”更难

- G1 改变物体几何和正确手型，禁止把 red 固定映射到 power、blue 固定映射到 pinch；
- G2 只给公开关系和测量，不把正确 offset 作为输入；
- G3 混合 3 个 target-move、4 个 no-change 和 3 个 irrelevant-fragile-move，避免把
  “总是重规划”误判为恢复能力；
- G4 同时改变可供性、邻居关系、事件、质量、摩擦、位姿、yaw 和指令改写。

传统 T1 在简单已知几何任务中可以与 Agentic 方法打平，这一点应主动承认。框架的优势
出现在语义角色、关系安全、变化归因以及多个约束必须同时成立时，而不是声称检测器或 IK
本身更强。

## G3 失败纠正的证据

Full Agent 在三种事件中分别达到 target-move 3/3、no-change 4/4、
irrelevant-fragile-move 3/3，false-replan 为 0。无重规划版本在 target-move 为 0/3，
但在 irrelevant move 为 3/3；这说明部分基线成功来自“无需纠正”的负对照，而非人为压低。

现有 L3 MuJoCo 视频还提供实际蓝色目标位移证据；L1/L2 仍明确标记为 controlled verifier
fault injection，不能包装成自然物理失败。

## 审计与复现

正式输出：`output/qwen35_challenge_v3_main_final/`

- `CHALLENGE_V3_REPORT.md`：Wilson CI、配对检验、事件/难度切片和成本；
- `challenge_summary.json` / `challenge_results.csv`：机器可读汇总与逐次结果；
- `validation_receipt.json`：100 个 run、260 个 public event 及 evaluator-only 字段隔离，PASS；
- `agent_tool_validation_receipt.json`：100 个 run、60 次工具调用，PASS；
- `frozen_config.json`：知识卡 SHA-256、感知适配器版本、条件和种子。

```bash
python scripts/summarize_qwen_vlm_challenge.py output/qwen35_challenge_v3_main_final
python scripts/validate_agentic_artifacts.py output/qwen35_challenge_v3_main_final
python scripts/validate_agent_tool_contracts.py output/qwen35_challenge_v3_main_final
```

## 结论边界

本结果支持“真实多模态高层决策 + public-observation Agent 工具执行 + 运动学技能代理”的
任务级机制主张。G0-B 纯接触动力学闭合/提升未通过准入，因此不得把 10/10 解释为接触抓取
成功率，也不声明端到端 VLA 或 sim-to-real。G0-B 负结果和无接触稳定控制见
`output/g0b_contact_admission_audit_v1/REPORT.md`。
