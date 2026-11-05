# Agentic RAG-VLM 代码核查与补齐进度

核查日期：2026-11-02。本文记录当前本机代码实况，历史冻结实验结果不随工程修改重算。

本日后续更新：目录已归并，新增实现、最新验收与外部阻塞以
`../DELIVERY_README.md` 和工作区 `../README.md` 为准。下文保留首次检查时的缺口清单，
不表示后续仍未修复。特别更正：VLABench 运行脚本已在工作区
`实验数据/benchmarks/vlabench/` 找到；首次仅搜索独立代码目录导致漏查。

## 目录与接续入口

- 当前实现：`/home/admin1/ct/agentic-rag-vlm`。该目录本机没有 `.git` 元数据；不能凭 README 声称与远程仓库 HEAD 一致。
- 研究交接：`/home/admin1/ct/Agentic-Policy/交接.md`、`README.md`、`Agentic RAG-VLM/BluPe对接/往来记录.md`。
- 本地平台合同：`/home/admin1/ct/agentic-rag-vlm-apidocs/`（api.md、contracts.json、标定、观测、URDF）。
- 仿真资产工作区：`/home/admin1/ct/simulation`，本轮没有更改。
- `Agentic-Policy` 的 BluPe 说明仍是 10 月 26 日的占位 transport 状态；独立代码目录今天已含 Session API、YAM FK/IK、运动规划与 remote transport。旧交接中的 23 项测试、等待实现接口等不能当作当前代码状态。
- 本轮没有联系平台、启动真机会话、发邮件、推送仓库或更改历史实验记录。

## 实际实现与缺口

| 模块 | 当前实现 | 尚需补齐 |
|---|---|---|
| HAA-RAG | 类别门控、四维兼容评分、top-3 及确定性排序 | visual score 是长宽比距离，不是 CLIP；缺兼容筛选阈值与跨类别无匹配拒绝协议；原始 116 条库不在工作区 |
| 场景图 | 平面邻接关系、易碎邻居对应的偏移/高度/力修正 | containment/support/occlusion 的观测与动作约束合同；运动层实际遵守约束的验证 |
| VLM | OpenAI-compatible 图像调用、原文/解析结果审计、平台物体感知 | 平台任务目标和子目标仍来自任务 JSON；独立模型工具路由主要在旧 challenge 脚本；RGB 猜测米制坐标缺乏可验证的尺度依据 |
| 单目标 pipeline | 七因子质量模型、14 类故障、策略级 L1/L2/L3 标签 | L3 标记没有触发完整重新观察与规划；位置/高度恢复字段未全部落实到动作或运动层 |
| 平台 runtime | 多子目标、成功后记忆、场景变化触发有界重规划 | 验证失败直接停止，未接平台 L1/L2；不能称真机三级恢复已实现 |
| 记忆 | 类别索引、质量择优、JSON 存取 | 相同类别不同对象/场景的兼容校验、保质期和执行条件；避免重复使用已施加场景缩放的策略 |
| 平台运动 | 固定技能、YAM FK/IK、关键帧插值、包级回执 | RGB 几何定位、站点标定、碰撞/桌面净空与夹爪释放验证；关节插值界不等于无碰撞路径 |
| VLABench | 卡片库、结果说明和部分演示材料 | 当前独立目录缺对应完整任务执行脚本；文档中的历史运行结果不等于本目录可直接复跑 |

## 本轮已补齐

1. `BluPeSession` 必须收到明确的目的地坐标表。缺失/非法目的地在发送动作前停止，CLI 将 mock 场景或 remote 任务坐标表传入 session。
2. place 命令与 verifier 使用同一份目的地坐标，增加三维到达误差与容差（默认 5 cm）的审计字段。仅移动到错误位置不再记为完成。
3. 抓取或放置回执未明确 `accepted=True` 时停止，不再继续后续动作、不更新成功记忆。
4. 受保护对象在操作前缺失则停止；验证要求前后均可见，同时检查三维位移，含竖直方向。
5. 非有限目标/邻居坐标在动作前拒绝；非有限验证坐标拒绝通过；任务要求移动受保护对象时拒绝执行。
6. `SceneChangeVerifier.verify()` 的直接调用仍可省略目的地作为历史位移诊断；此时回执明确标记 `verification_scope=displacement_only`。平台 session 使用目的地校验。

目的地容差是工程配置，不是测得的感知精度。单次观测位置变化仍不能证明稳定接触、夹爪释放、无碰撞或真实物体身份正确。

## 验证

使用本机 `/home/admin1/miniconda3/envs/RoboDojo/bin/python`；base Python 没有 pytest。

```bash
cd /home/admin1/ct/agentic-rag-vlm
/home/admin1/miniconda3/envs/RoboDojo/bin/python -m pytest -q tests
/home/admin1/miniconda3/envs/RoboDojo/bin/python scripts/run_platform_session.py \
  --task configs/blupe_stage1_task.example.json \
  --cards knowledge_base/blupe_affordance_cards.json \
  --registry knowledge_base/blupe_semantic_registry.json \
  --mock --output /tmp/agentic-rag-vlm-audit-20261102
```

修改前：75 passed、3 skipped；修改后：87 passed、3 skipped。本轮新增反例验证覆盖错放、消失的邻居、竖直碰撞位移、NaN/Inf、拒绝回执、受保护目标冲突。
mock CLI：`outcome=complete completed=['cup'] receipts=7`。这是离线协议验证，不是真机抓取结果。

## 下一步实施顺序

用户随后明确：优先完成 CMU/BluPe 合作对接。近期范围以
[BluPe 合作首轮交付](BLUPE_FIRST_DELIVERY.md) 为准；下列是完整框架的后续清单，
其中 VLABench 入口恢复和完整视觉检索升级暂不作为首轮合作交付的前置。

1. 补齐输入与感知合同：稳定对象 ID、数值/坐标系检查、相机选择与时间戳；用已有 K、T 和经测量的支撑平面/深度完成几何定位，避免从 RGB 凭空估计公制位置。
2. 接通恢复执行语义：区分可恢复抓取失败、目标丢失、保护约束违反与基础设施错误；让 L1/L2 的变化真正进入动作，L3 重新观察和规划，所有路径有预算和明确停止条件。
3. 恢复 VLABench 运行入口和版本依赖，先单任务完整抓取/放置/释放/撤离，再做多种子与冻结配对实验。
4. 补视觉 embedding 检索、场景图关系与上下文记忆；保留现有轻量实现作为明确标记的基线，不覆盖历史结果。
5. 几何、运动约束、离线反例全部通过后再安排 BluPe 单物体联调；真实硬件执行需在具体任务、标定和会话范围明确后开展。

代码补齐产生的新实验应使用新输出目录与来源说明，不能当作论文原始日志或原始经验库。
