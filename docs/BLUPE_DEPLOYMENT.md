# BluPe 远程真机平台部署方案

更新日期：2026-11-02。本文件是 Agentic RAG-VLM 对接 BluPe 远程双臂平台（YAM）的唯一部署
说明。BluPe Session API v1 合同已公开（2026-10-25 校验，
`https://blupe.io/docs/robot-api/`），platform 部署层已按合同完成实现：
API 客户端、YAM 6-DOF 运动学、确定性运动规划、远程 transport 与 CLI 远程模式，
离线测试套件全绿，公开只读端点实测通过。剩余工作为真机联调（见 §8）。

## 1. 部署边界（与论文 claim boundary 一致）

- **只对接感知 + 推理层**：框架输出目标选择、抓取参数（synergy/force/aperture/approach）、
  空间约束与纠错决策，驱动平台执行。类比 VLABench 实验中
  “任务级 Agent + IK/技能低层”的结构。
- **不做端到端 VLA 控制**：不训练、不微调策略模型。BluPe v1 合同只接受关节轨迹
  （无命名技能/ 笛卡尔接口），因此部署层包含一个**确定性几何 IK 运动适配器**
  （`kinematics.py` + `motion.py`）：URDF 转录的运动学 + 阻尼最小二乘 IK + 关键帧插值，
  无任何学习成分、无反应式控制回路，每个关键帧可审计——与 VLABench 实验中
  IK/技能层的性质相同。此为对“不直接输出关节控制量”字面边界的一次经批准的修订
  （2026-11-02），不改变“不做端到端 VLA”的本意。
- **验证以公开观测为准**：成功判据来自平台回传的物体位置变化（目标位移 + 受保护物零位移），
  全程 receipts 审计，与仓库既有 public/private 证据纪律一致。
- 不因对接平台而放宽论文边界：不声称 benchmark-wide success、real-robot certification、
  sim-to-real；本目录现有冻结实验协议与结果不受部署改动影响。

## 2. 架构与代码映射

```
BluPe Session API v1 (已实现)                    本仓库
──────────────────────────                ─────────────────────────────
GET /observation (站观测+三相机 JPEG)  ──▶ BluPeSessionClient ──▶ RemotePlatformTransport
GET /calibration (T_base_camera)              │        │
POST /v1/sessions (capability)                ▼        ▼
POST .../trajectories (10Hz 关节包)    PlatformObservation  YamMotionPlanner + YamKinematics
GET .../episode/trace (执行结果)       (JPEG + 站状态)     (IK: 相机系目标 → 关节关键帧)
                                             │                      ▲
                                             ▼                      │ 基座系目标
                                 VLMObjectPerception（图像+Qwen，    │
                                   输出 top 相机光学系 center_xyz_m）  │
                                             │  PublicObject    _camera_to_base()
                                             ▼                  (标定变换)
                    AgenticRuntime: observe→plan→execute→verify→monitor→replan
                    AgenticPipeline: HAA-RAG + 场景图 + memory + recovery
                                             │  GraspAction
                                             ▼
                                 GraspActionMapper ──▶ PlatformCommand(pick/place)
                                             │
                                             ▼
                    SceneChangeVerifier（目标位移 + 受保护物判据）
```

- `agentic_rag_vlm/platform/blupe_client.py`：Session API 客户端（仅标准库）。
  只读端点、会话生命周期、202-accepted 后跟 `episode/trace` 轮询至终态、
  4xx 不重试 / 5xx 与网络错误退避重试。
- `agentic_rag_vlm/platform/kinematics.py`：YAM 6-DOF FK/IK（numpy）。
  关节链转录自 I2RT 官方 `yam.urdf`（flange convention v1）；home 位姿精确锚定
  I2RT README（q=0 时 mount=[0.110598, 0, 0.173501]）。含 `base_yaw_deg` /
  `joint_signs` / `joint_offsets_deg` 站点修正参数（首次真机标定用，默认复现 URDF）。
- `agentic_rag_vlm/platform/motion.py`：确定性运动规划。pick = 5 关键帧
  （当前观测位姿 → pre-grasp → 下降 → 闭合驻留 → 提升），place 同构
  （当前 → 目标上方 → 下降 → 释放驻留 → 撤离）。关节插值步进 ≤3°/步（30°/s @10Hz），
  夹爪线性过渡；**非操作臂全程冻结在观测关节**。计划 ≤1200 waypoints
  （会话 3000 预算留余量）。
- `agentic_rag_vlm/platform/transport.py`：`RemotePlatformTransport`
  （安全门 → 规划 → 打包提交 → 跟踪结果，receipts 全程留痕）与
  `MockPlatformTransport`（离线确定性 dry-run）。
- `scripts/run_platform_session.py`：会话 CLI；`--mock` 离线，`--platform-config` 远程。
- `knowledge_base/blupe_semantic_registry.json`：三阶段物体的冻结语义注册表。
- `knowledge_base/blupe_affordance_cards.json`：BluPe 物体卡片库（参数为建议值，
  由运动层翻译解释，不是标定声明）。
- `configs/blupe_platform.example.json`、`configs/blupe_stage1_task.example.json`：
  示例配置（v2：真实 endpoint、destinations 命名区域坐标表）。

## 3. 分阶段计划（不变）

| 阶段 | 场景 | 验证目标 |
|---|---|---|
| Stage 1 | 单物体四件套：cup / bottle / banana / cube（bin picking 优先） | 感知→检索→抓取→放置全链路跑通，receipts 完整 |
| Stage 2 | 易碎邻居：cup + vase | 场景图约束生效，vase 位移判据通过 |
| Stage 3 | 多物体清理 | 多子目标顺序执行、memory 与重规划 |

每阶段先 mock dry-run，再远程真机；真机阶段遵守平台配额与安全限制。

## 4. API 合同要点（原“四问”，已全部落地到代码）

1. **观测格式**：RGB-only（三相机 top/left/right，640×360 JPEG，URL 随观测下发）；
   深度当前 available:false；**无物体元数据**。
   → 感知走 `VLMObjectPerception`：VLM 输出 top 相机光学系（x 右 / y 下 / z 前）
   `center_xyz_m`（提示词见 `perception.py` 的 `REMOTE_VLM_PERCEPTION_PROMPT`，
   典型深度 0.3–1.2 m），transport 用标定 `T_base_camera[arm]` 变换到臂基座系。
2. **控制原语层级**：仅关节轨迹。`cadence_hz` 锁 10.0；每臂 6 关节（度）+ 夹爪 0–1
   （0 闭合）；1–300 waypoints/包，单包激活；`step_id` 0–12299 跨包连续；
   3000 waypoints/会话；重试需同 `trajectory_id` 同 body。
   → `TrajectoryBuilder` 打包 + `YamMotionPlanner` IK 规划。
3. **安全限制与中断语义**：观测含 `safety{ok,estop_engaged,reason}`、`homed`、
   `settled`、`mode`；会话生命周期 queued→preparing→running→complete→resetting
   （终态含 stopped/timed_out/disconnected/safety_aborted）。
   → `_safety_gate`：estop / safety 不 ok / 首命令未 homed+settled 即拒发；
   每包提交后轮询 trace 至 completed/aborted/rejected；会话终态即中止。
4. **鉴权与配额**：公开监测与建会话无需凭据；`POST /v1/sessions` 返回
   `session_capability` 作为该会话 Bearer。capability 仅存内存，
   不落盘、不进 URL/日志/receipts（`SessionHandle.to_public_dict()` 审计视图）。
   配额见第 2 条 waypoint 预算。

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

# 远程模式（需 VLM 感知：远程站无物体元数据）
export BLUPE_VLM_ENDPOINT=http://<openai-compatible-endpoint>/v1
export BLUPE_VLM_MODEL=<model-name>
python scripts/run_platform_session.py \
  --task configs/blupe_stage1_task.example.json \
  --cards knowledge_base/blupe_affordance_cards.json \
  --registry knowledge_base/blupe_semantic_registry.json \
  --platform-config configs/blupe_platform.example.json \
  --vlm-endpoint "$BLUPE_VLM_ENDPOINT" --vlm-model "$BLUPE_VLM_MODEL" \
  --arm right --run-duration-s 300 --queue-timeout-s 600 \
  --output output/blupe_session_run
```

- 远程模式启动即 `health_check`（robots / queue / calibration），未注册、未连接即退出；
  排队等待至 running 后执行，结束或异常均 `stop_session`。
- `--arm {left,right}` 选操作臂（默认 right），`--no-wait` 跳过排队等待。
- 鉴权：`auth_token_env` 为可选的预共享令牌（当前公开 API 不需要）；
  会话 capability 见 §4.4。
- 坐标系：task 的 `destinations` 与 VLM 感知输出均为 top 相机光学系；
  site 修正（基座偏航 / 关节符号与偏置）在 `YamKinematics` 构造参数中配置，
  首次真机标定后固化到部署配置。

## 6. 安全设计摘要

- 首命令要求 homed+settled；estop / safety 不 ok 一律拒发（fail-closed）。
- 轨迹首关键帧 = 当前观测位姿，保证从实际位姿出发的有界步进（≤3°/步）。
- 非操作臂冻结；夹爪命令域 [0.02, 1.0]，避免硬限位碰撞。
- 202-accepted ≠ 已执行：每包跟踪 trace 至终态，超时/会话终态即报错中止。
- 会话 waypoint 预算（3000）与 step_id 上限（12299）在打包层强制。

## 7. 测试与验证状态

- `pytest -q tests`：**75 passed, 3 skipped**（RoboDojo env，全部离线）。新增
  `test_kinematics.py`（FK 锚点 / IK 往返 / top-grasp / 限位 / 夹爪映射）、
  `test_blupe_client.py`（协议：URL、capability 头、4xx/5xx 重试、分包、
  lease 刷新、trace 轮询）、`test_motion.py`（关键帧结构 / FK 几何 / 插值界 /
  预算）、`test_remote_transport.py`（FakeClient + 真实运动学全流程：
  生命周期 / 观测下载 / pick-place 执行 / step_id 连续性 / 安全门 / 坐标变换 /
  失败回执）。
- CLI dry-run 实测：`outcome=complete completed=['cup'] receipts=7`。
- **公开只读端点实测（2026-11-02，本机直连）**：`/v1/robots`（6 机器人，yam-1 在列）、
  `/v1/robots/yam-1/queue`、`/observation`（真实关节状态与三相机 JPEG 解析）、
  `/calibration`（top K + 双臂 T_base_camera）、三相机 JPEG 下载（23–31 KB，合法 JPEG）
  全部通过。`/camera-poses` 当前 503（需硬件 settled），非关键路径
  （用 calibration 的 T_base_camera）。
- 测试期间修复两个实现 bug：motion.py 方向常量拼写（NameError）、
  计划缺首关键帧导致的首步关节跳变（现从观测位姿有界出发）。

## 8. 剩余工作（真机联调前置）

1. **站点可用性**：实测时 yam-1 处于 `mode=DISABLED, safety.reason=hardware_not_initialized,
   homed=false, settled=false`（远程站未上电初始化）。站恢复后 safety gate 放行。
2. **VLM 感知联调**：远程模式必须 `--vlm-endpoint/--vlm-model`；建议先用
   top 相机实测图验证 VLM 输出的相机系坐标量级（深度 0.3–1.2 m）。
3. **首次真机标定**：用小幅度 pick 计划对比 FK 预测与回传关节/图像，
   必要时调 `YamKinematics` 的 site 修正参数并固化。
4. **配额演练**：确认单会话 waypoint 预算（默认计划 ≤1200/命令）满足 Stage 1 任务。
