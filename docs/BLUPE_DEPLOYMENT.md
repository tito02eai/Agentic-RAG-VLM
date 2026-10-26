# BluPe 远程真机平台部署方案

更新日期：2026-10-26。本文件是 Agentic RAG-VLM 对接 BluPe 远程双臂平台（YAM）的唯一部署
说明，覆盖架构边界、当前完成状态、待填项与分阶段计划。API 合同到手前，远程 transport
保持 fail-fast 占位，不猜测接口细节。

## 1. 部署边界（与论文 claim boundary 一致）

- **只对接感知 + 推理层**：框架输出目标选择、抓取参数（synergy/force/aperture/approach）、
  空间约束与纠错决策，驱动 BluPe 平台自带运动原语执行。类比 VLABench 实验中
  “任务级 Agent + IK/技能低层”的结构。
- **不做端到端 VLA 控制**：不训练、不微调策略模型，不直接输出关节/末端连续控制量。
- **验证以公开观测为准**：成功判据来自平台回传的物体位置变化（目标位移 + 受保护物零位移），
  全程 receipts 审计，与仓库既有 public/private 证据纪律一致。
- 不因对接平台而放宽论文边界：不声称 benchmark-wide success、real-robot certification、
  sim-to-real；本目录现有冻结实验协议与结果不受部署改动影响。

## 2. 架构与代码映射

```
BluPe 平台 API (待合同)                      本仓库
─────────────────                      ─────────────────────────────
相机观测/物体元数据  ──▶ PlatformTransport ──▶ PlatformObservation
                                          │
                                          ▼
                              RegistryMetadataPerception（确定性）
                              或 VLMObjectPerception（图像+Qwen）
                                          │  PublicObject
                                          ▼
                    AgenticRuntime: observe→plan→execute→verify→monitor→replan
                    AgenticPipeline: HAA-RAG + 场景图 + memory + recovery
                                          │  GraspAction
                                          ▼
                              GraspActionMapper ──▶ PlatformCommand(pick/place)
                                          │
                                          ▼
                    SceneChangeVerifier（目标位移 + 受保护物判据）
```

- `agentic_rag_vlm/platform/`：config / transport / perception / commands / verifier / session。
- `agentic_rag_vlm/platform/transport.py`：`RemotePlatformTransport` 为 fail-fast 占位，
  API 合同到位后只在此文件实现 HTTP 细节，其余层不动。
- `scripts/run_platform_session.py`：会话 CLI；`--mock` 为离线确定性 dry-run。
- `knowledge_base/blupe_semantic_registry.json`：三阶段物体的冻结语义注册表。
- `knowledge_base/blupe_affordance_cards.json`：BluPe 物体卡片库（参数为建议值，
  由平台原语翻译解释，不是标定声明）。
- `configs/blupe_platform.example.json`、`configs/blupe_stage1_task.example.json`：示例配置。

## 3. 分阶段计划（起始方案）

| 阶段 | 场景 | 验证目标 |
|---|---|---|
| Stage 1 | 单物体四件套：cup / bottle / banana / cube（bin picking 优先） | 感知→检索→抓取→放置全链路跑通，receipts 完整 |
| Stage 2 | 易碎邻居：cup + vase | 场景图约束生效，vase 位移判据通过 |
| Stage 3 | 多物体清理 | 多子目标顺序执行、memory 与重规划 |

每阶段先 mock dry-run，再远程真机；真机阶段遵守平台配额与安全限制。

## 4. 待填项（BluPe API 四问，合同到手后落实）

1. **观测格式**：RGB-D 还是 RGB-only；相机视角列表；是否提供物体元数据（名称/位置/类别）。
   → 决定 `RemotePlatformTransport.get_observation` 的归一化实现与感知路径选择
   （元数据走 `RegistryMetadataPerception`，纯图像走 `VLMObjectPerception`）。
2. **控制原语层级**：命名技能（pick/place/…）还是笛卡尔/关节目标；目的地如何表达
   （命名区域 or 坐标）。→ 决定 `PlatformCommand` 到平台调用的翻译与
   place 目的地解析（当前 mock 用命名区域坐标表模拟）。
3. **安全限制与中断语义**：力/位姿限幅、急停、人在环要求。→ 决定 session 的
   safe-stop 触发面与 `SceneChangeVerifier` 阈值标定。
4. **配额、限流与开源要求**：调用配额、速率限制、代码/数据是否要求开源。→ 决定
   重试/限流策略与仓库发布边界（当前为私有仓库，不自动转公开）。

## 5. 安装与运行

```bash
# 核心框架（无 MuJoCo，适用于部署机）
pip install -e .

# 离线确定性 dry-run（无网络）
python scripts/run_platform_session.py \
  --task configs/blupe_stage1_task.example.json \
  --cards knowledge_base/blupe_affordance_cards.json \
  --registry knowledge_base/blupe_semantic_registry.json \
  --mock --output output/blupe_session_dryrun

# 远程模式（API 合同到位后；当前 fail-fast 退出并给出说明）
python scripts/run_platform_session.py \
  --task configs/blupe_stage1_task.example.json \
  --cards knowledge_base/blupe_affordance_cards.json \
  --registry knowledge_base/blupe_semantic_registry.json \
  --platform-config configs/blupe_platform.example.json \
  --output output/blupe_session_run
```

鉴权：配置文件只写环境变量名（`auth_token_env`），令牌从环境变量读取，
不进仓库、不进日志。VLM 感知模式：`--vlm-endpoint`/`--vlm-model` 指向
OpenAI 兼容服务（如本地 Qwen），复用 `agentic_rag_vlm/qwen_adapter.py`。

## 6. 当前状态与验证

- `pytest -q tests`：新增 `tests/test_platform.py` 覆盖注册表、配置、映射、感知
  （含 VLM 归一化与解析失败）、验证判据、mock transport 与完整会话
  （成功 / place 失败 safe-stop / 受保护物）；全部离线运行。
- CLI dry-run 实测：`outcome=complete completed=['cup'] receipts=7`。
- 远程 transport、真实平台参数标定、VLM 感知真机联调：**未开始**，等 API 合同。
