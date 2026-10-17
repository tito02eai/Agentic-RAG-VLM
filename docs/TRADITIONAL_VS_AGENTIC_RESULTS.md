# Traditional RGB-D + IK vs Agentic RAG-VLM

## 为什么有黄色物体

黄色物体现在明确表示半透明的 **Protected Glass（受保护易碎玻璃器皿）**，不是第三个待抓
目标。它用于检验系统能否把自然语言中的“禁止接触/易碎”转换成：

1. `protected_glass` 语义节点；
2. 与当前抓取目标的邻接关系；
3. 腕部候选角、接近路径、速度/力度和安全停止约束；
4. 手到玻璃器皿的最小净空这一可量化指标。

传统检测器仍能给出黄色区域和质心，但若没有预先硬编码，就不知道它是“禁止接触的玻璃
器皿”，也不会把该角色用于抓取朝向和任务级恢复。视频中使用半透明杯体、上下边缘和桌面
keep-out 光环来避免其意义仅靠口头说明。

## 公平的三层方法

- **T0 Traditional RGB-D + fixed IK**：与 Agentic 相同的公开 RGB-D 质心，固定 power 抓法、
  固定腕角、开放环执行。
- **T1 Reactive geometry + IK**：更强的传统基线；方块/圆柱几何启发式选择 power/pinch，
  腕角在冻结布局全集上离线调优，并对任何视觉变化重规划。它没有开放语义、受保护关系和
  通用任务记忆。
- **A2 Agentic RAG-VLM**：复用完全相同的检测与 IK 执行器，只增加检索策略、语义场景图、
  已完成子目标记忆和与 pending target 相关的有界重规划。

## 冻结结果

| 实验 | T0 fixed RGB-D+IK | T1 reactive geometry+IK | Agentic RAG-VLM |
|---|---:|---:|---:|
| E1 已知物体抓法兼容性（12） | 50.0% | 100.0% | 100.0% |
| E2 八方位 protected-glass 净空门禁（8） | 62.5% | 62.5% | 87.5% |
| E3 target/no-change/irrelevant-change 判断（12） | 66.7% | 66.7% | 100.0% |
| E4 四项机制联合通过（8） | 0.0% | 50.0% | 87.5% |

E2 中 T1 的单一腕角是在全部冻结布局上选择的最优固定角，不是随意挑选的弱参数。
Agentic 的平均保护净空为 25.2 mm；仍有一个方位只有 6.8 mm，低于 10 mm 门槛，因此如实
记为不通过，而不是强行执行。E3 中 T1 对目标移动召回率为 1，但在四次无关玻璃器皿移动上
均误重规划；Agentic 的 target-change precision/recall 均为 1。

E1 也保留了一个重要负结论：在两个已知规则几何体上，手工 shape heuristic 可以与 Agentic
打平。因此答辩时不应声称 Agentic 改善了所有低层 IK；优势主要来自开放任务语义、关系安全
和变化归因。

## 证据边界

上述结果是校准运动技能代理下的机制/运动学安全实验，不是 MuJoCo 接触动力学抓取成功率。
原始逐试验记录为 `output/traditional_vs_agentic_v1/paired_trials.csv`，聚合结果为
`output/traditional_vs_agentic_v1/summary.json`。成对仿真视频位于
`output/paired_comparison_videos_v1/`。
