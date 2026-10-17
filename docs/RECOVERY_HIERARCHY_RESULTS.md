# Failure-correction hierarchy

## 目的

该实验检查 Agentic 执行器能否依据验证证据和预算选择不同粒度的恢复，而不是把“再试一次”
当作统一策略。三种方法使用相同的任务状态与 verifier 接口：

- **T0 no recovery**：失败后不恢复；
- **T1 fixed retry**：只允许一次不更换方法的参数重试；
- **A2 hierarchical**：按 L1 参数重试、L2 方法切换、L3 全量重规划逐级升级，每级预算为 1。

## 冻结场景与结果

| 场景 | 正确行为 | T0 | T1 | A2 |
|---|---|---:|---:|---:|
| F0 无故障负控制 | 不触发恢复 | PASS | PASS | PASS |
| F1 位置偏差 | L1 重观测/参数重试 | FAIL | PASS | PASS |
| F2 抓法族错误 | L2 POWER→PINCH | FAIL | FAIL | PASS |
| F3 场景计划过期 | L3 保留记忆并重规划 | FAIL | FAIL | PASS |
| F4 超时控制 | 预算截止并安全停止 | PASS | PASS | PASS |

在三个可恢复故障 F1–F3 上，T0、T1、A2 的恢复成功率分别为 **0/3、1/3、3/3**；
机制选择正确率（包含 F0/F4 控制）分别为 **40%、60%、100%**。A2 在 F0 中没有产生多余
恢复，F4 中执行安全停止。

## 证据边界

F1 与 F2 是受控 verifier fault injection，用来隔离验证恢复状态机，不代表 MuJoCo 已经发生
真实接触失败。F3 另有实际 MuJoCo 蓝色目标外部位移视频：传统开放环继续使用旧路点，A2
检测到 pending target 变化后执行一次有界 L3 重规划，并保留已完成红色目标的记忆。

本实验不验证接触动力学闭环抓取成功率。完整 grasp/lift success rate 必须等 G0-B 接触闭合与
提升门禁通过后才能报告。

## 复现与视频

```bash
python scripts/run_recovery_hierarchy_experiment.py
python scripts/render_paired_comparisons.py --experiments recovery
```

结构化结果位于 `output/recovery_hierarchy_v1/results.json`，L3 成对 MuJoCo 视频位于
`output/paired_comparison_videos_v1/E3_target_change_paired.mp4`。36 秒答辩证据片位于
`video/recovery_evidence_reel/renders/guanghua_recovery_evidence_reel_v1.mp4`，其可编辑
HyperFrames 工程位于 `video/recovery_evidence_reel/`。
