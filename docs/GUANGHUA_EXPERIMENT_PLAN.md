# 光华机器人 Agentic RAG-VLM 物理验证方案

## 1. 实验结论边界

本实验验证 Agentic RAG-VLM 在新机器人本体上的高层决策能力，不训练 VLA，也不
声称获得端到端学习策略。低层动作由右臂 IK、关节位置控制和三种固定手部 synergy
完成。研究变量是 HAA-RAG、场景图、执行记忆和有界恢复，而不是低层控制器。

中期阶段的主要结论表述为：在相同光华机器人模型和相同公开 RGB-D 观测下，完整
Agentic RAG-VLM 是否比对应消融更可靠地触发可供性选择、易碎邻居约束和场景变化
重规划。等 Admission-B 通过后，再把机制结论升级为完整桌面整理任务成功率。

## 2. 代表性任务

自然语言指令：

> 将红色方块放入红色区域，将蓝色圆柱放入蓝色区域，并且不要碰黄色易碎物。

该任务包含两个具有不同可供性的目标：方块使用 power grasp，细圆柱使用 pinch
grasp。黄色圆柱是受保护的易碎代理物体。任务不是按抓起次数计分，而是按最终目标
关系、受保护物体接触、重复动作和恢复结果计分。

## 3. 场景族与因果比较

| 场景 | 物理变化 | 主要机制 | 成对比较 |
|---|---|---|---|
| G0 Nominal | 两个目标无遮挡，易碎物远离路径 | 基础执行能力 | Skill-only / Full |
| G1 Affordance | 方块与圆柱交换位置并随机朝向 | HAA-RAG 选择 power/pinch | Full / No-RAG |
| G2 Fragile Neighbor | 易碎物紧邻红色方块 | 场景图生成远离邻居的 waypoint | Full / No-Graph |
| G3 Displacement | 完成第一个物体后移动未完成目标 | 执行记忆、视觉变化检测和 L3 | Full / No-Memory / No-Replan |
| G4 Combined | 位姿、摩擦和质量联合随机化 | 完整闭环鲁棒性 | Skill-only / Local / Full |

G3 的位移属于外部环境事件：它改变 MuJoCo 物理状态，但不把位移对象或位移量传给
Agent。Agent 必须从下一帧 RGB-D 中发现旧计划失效。主结果不使用控制器内部抓取
偏差来人为制造恢复成功。

## 4. 系统条件

- `B0_skill_only`：固定顺序、固定通用抓取方式，不使用 Agentic 模块；
- `B1_rag_only`：使用 HAA-RAG，但无场景图和恢复；
- `C1_local`：公共观测 Monitor 加一次固定重试，隔离确定性监督收益；
- `C2_full`：HAA-RAG、场景图、执行记忆、L1/L2/L3 和安全停止；
- `A_no_rag`：Full 使用通用抓取参数；
- `A_no_graph`：Full 只能直线接近；
- `A_no_memory`：不保存已完成子任务与成功参数；
- `A_no_replan`：检测到目标变化后不能进行 L3 重规划。

中期答辩主表只放机制匹配的成对比较，不把所有条件平均成一个难以解释的总分。

## 5. 公共观测与私有评价器

Agent 可使用：前视/Agent 视角 RGB-D、腕部 RGB、关节与手部状态、由公开观测计算
的 lift/place/scene-change 证据、任务指令和经验记忆。

Agent 不可使用：MuJoCo 物体坐标、接触真值、目标区判定、扰动元数据和终局成功
标记。这些只写入独立的 `private_evaluator.json`。规划 trace 中若出现上述字段，整条
rollout 判定为协议违规，而不是成功样本。

## 6. 主要指标

第一层是任务结果：完整订单成功率、完成子目标数、易碎物接触率、条件自然恢复成功
率。第二层是机制指标：HAA-RAG Top-1、抓取 synergy 选择、约束满足、旧计划检测
precision/recall、重复动作、错误目标和安全停止。第三层单独报告成本：VLM 调用、
动作数、路径长度、P50/P95 规划延迟和 episode wall time。

成功率报告 Wilson 95% 置信区间；相同 seed 的 Full/消融使用配对差值和 McNemar
exact test。VLM 延迟使用 median/P95，不能混进低层控制延迟。

## 7. 执行规模与停止规则

1. Admission-A：G0、seed 7，验证复位、相机、RGB-D、IK、安全预抓取和私有评价器；
2. Mechanism Pilot：每个 G1-G3 条件 5 个冻结 seed，验证设计是否隔离对应机制；
3. Admission-B：同一 seed 验证柔顺闭合、双侧持续接触、公开视觉抬升和无数值复位；
4. Physical Main：Admission-B 通过后，每个场景/条件运行 20 个成对 seed；
5. Headline：只对效果最大的 Full/关键消融增加到 30 个新 seed。

Mechanism Pilot 不需要等待完整抓取标定：G1/G3 使用冻结前后 RGB-D，G2 在动力学接触
前冻结场景快照。这样可以先验证 Agentic 因果链，同时不会把未通过的接触模型写成
物理任务成功。G3 的“红方块已完成”是公开 verifier contract fixture，必须在 trace
中显式标出，不能伪装成已执行的抓取。

如果 G1 中通用抓取与 HAA-RAG 都稳定成功，就停止统计，不宣称检索优势，而是调整
物体尺寸/摩擦使两类物体的有效抓取区间真正不同。如果 G2 的 No-Graph 不产生安全
代价，则说明邻居布局无效。如果 G3 的移动无法从 RGB-D 稳定检测，则不能继续做
L3 成功率。

## 8. 每条 rollout 的证据

每条实验必须保存冻结配置、实际场景参数、公共观测与决策 trace、独立 evaluator、
前视视频、关键帧和真实模型调用耗时。视频必须在运行前按场景 ID 预先指定，不能只
挑成功样本。失败和安全停止同样保留。

## 9. 中期答辩建议展示

演示 G3 的 Agentic trace：公开 verifier 宣告红色子目标完成；蓝色圆柱在规划后发生
外部位移；前后 RGB-D 触发 `target_displaced`；系统保留已完成子任务并重建蓝色目标
计划。随后并排展示同 seed 的 No-Memory 和 No-Replan 输出。画面文字必须标注
“机制演示/非完整抓取”，不把 contract fixture 剪成真实抓取视频。

主表展示 G1、G2、G3 三组成对消融。G4 作为后续正式论文扩展，不阻塞中期答辩。

## 10. 当前 Admission 状态

G0 seed 7 已通过 Admission-A“RGB-D 定位到物理预抓取”门槛：颜色/深度点云估计
红色方块位置，右臂沿桌外安全走廊经过五个 waypoint 到达手指向下的预抓取位置；
加入连续 IK 和重力前馈后，末端物理误差降到毫米级，且没有非支撑易碎物接触。

Admission-B 尚未通过。多组张开余量、有限 power 闭合、柔顺限力和接触参数标定中，
第一次指尖接触仍可能触发导入物体自由关节的 MuJoCo 数值复位。因此当前结果只证明
感知、IK、安全轨迹和证据边界已连通，绝不计为抓取或任务成功。具体失败边界与已排除
因素见 `GUANGHUA_CALIBRATION_STATUS.md`。

G1-G3 的真实 Qwen3.5-4B NF4 5-seed Pilot 已完成，共 35 次多模态调用。Full 在三个
对应机制判据均为 5/5；VLM-Only、No-Graph、No-Memory、No-Replan 的对应综合判据
均为 0/5。35 条回答全部可解析，endpoint 错误为 0；自动审计检查了 80 条公开事件，
未发现 evaluator-only 字段泄漏。四个 5-pair McNemar exact 均为 p=0.0625，所以这是
判据 pilot 而非显著性主实验。结果见 `QWEN_VLM_RESULTS.md`；不能与论文 78.3% 物理
任务成功率直接比较。

20-seed Main 也已完成，共 140 次真实多模态调用。三个 Full 条件均为 20/20；四个
对应消融均为 0/20；140 条回答全部可解析，endpoint 错误为 0。四个配对的双侧
McNemar exact 均为 `p=1.90735e-6`，Wilson 95% CI 分别为 Full `[0.84, 1.00]`、
消融 `[0.00, 0.16]`。自动审计检查了 140 个 run 和 320 条公开事件，无私有字段泄漏。
Main 应作为中期答辩主表，5-seed Pilot 只说明实验冻结与调参过程。

## 11. Challenge v2 升级结果（答辩主表）

旧 Main 的 20/20 对 0/20 主要作为机制单元测试保留。新的 Challenge v2 已完成
10 个冻结 seed、100 个方法 rollout 和 192 次真实 Qwen3.5-4B 多模态调用：

- G1：Full 10/10，VLM-only 0/10；物体几何与正确 synergy 随 seed 改变；
- G2：Full 4/10，No-Graph 1/10；Graph 只给公开关系/测量，不给候选 offset；
- G3：Full 7/10，No-Memory 0/10，No-Replan 7/10；Full 通过全部 target-move，
  No-Replan 只通过无需恢复的 no-change/irrelevant-move；
- G4：Full 2/10，Local 0/10，Skill-only 0/10；平均模块分数为 0.60/0.27/0.37。

G3/G4 均混合 target-move、no-change 和 irrelevant-fragile-move，避免只用正变化
样本。G4 的 Full 采用三个有审计凭据的 Agentic 子规划器组合；平均 4.4 次 VLM 调用，
P50 累计规划时延 40.75 s，说明组合可靠性和成本仍是明确局限。

完整性审计检查 100 个 run、100 行 CSV 和 260 条公开事件，结果 PASS，endpoint 错误
为 0。正式数字、事件条件表和难度分层见 `docs/CHALLENGE_V2_RESULTS.md`。中期答辩应把
Challenge v2 作为主定量结果，把旧 140-call Main 标成“机制单元测试”。
