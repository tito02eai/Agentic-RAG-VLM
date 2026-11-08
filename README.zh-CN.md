# Agentic RAG-VLM

**融合可供性感知检索、场景约束与自反思规划的机器人抓取框架**

论文已被 IROS 2026 接收。第一作者为复旦大学 Tao Chen。

[English](README.md) · [论文 PDF](paper/main_final.pdf) · [启动指南](docs/GETTING_STARTED.md) · [演示视频](examples/blupe/README.md)

## 项目介绍

框架通过“观察—检索与规划—执行—验证—监测—重规划”闭环完成机器人操作。VLM 负责语义识别与动作审查，知识库提供可供性兼容的经验，场景图把保护对象和空间关系转为执行约束，记忆模块复用已验证策略并清除失效经验。低层动作由 IK 和机器人技能生成。

| 模块 | 作用 | 主要实现 |
| --- | --- | --- |
| HAA-RAG | 类别匹配，按可供性、材质、脆弱性和可抓区域筛选，再用 CLIP 排序 | `retrieval.py`、`experience.py` |
| 场景推理 | 根据空间关系和物体状态约束动作与任务顺序 | `scene.py`、`constraints.py` |
| 智能体闭环 | 规划、模型审查、执行验证、监测与有限次数的重规划 | `paper.py`、`runtime.py`、`platform/session.py` |
| 任务记忆 | 复用已验证策略，重新观察位置，清除失败经验 | `memory.py` |
| 执行接口 | 原生 YAM IK、关节与夹爪轨迹、官方 Runner/SDK | `platform/` |
| 物理仿真 | MuJoCo 接触执行、图像和执行反馈 | `simulation/` |

当前工程知识库有 14 条带来源的经验及关联图像。论文历史结果与重建代码的工程验收分别保存，详情见[验证说明](docs/VALIDATION.md)。

## 如何启动

### 1. 安装并运行离线测试

使用 Linux、Python 3.10。系统需要 EGL/OpenGL 和 `ffmpeg`。

```bash
git clone https://github.com/tito02eai/Agentic-RAG-VLM.git
cd Agentic-RAG-VLM
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r configs/extended_sim_dependencies.txt
python -m pip install -e '.[runner,dev]'
git clone https://github.com/andlyu/blupe-remote-bots.git third_party/blupe-runner
git -C third_party/blupe-runner checkout e6040394c5ed62efbcc7d627acc0dfc8ccc100a6
export BLUPE_RUNNER_CHECKOUT="$PWD/third_party/blupe-runner"
export MUJOCO_GL=egl
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m pytest -q tests
```

### 2. 启动 Qwen 与 CLIP

两个模型服务分别运行在 `18074` 和 `18072`。建议与固定版本的仿真环境分开安装，具体依赖、权重路径和命令见[启动指南](docs/GETTING_STARTED.md#model-services)。本次验证使用 Qwen3.5-4B NF4 和 CLIP ViT-L/14。

### 3. 执行完整抓放

```bash
python scripts/check_platform_provider.py \
  --config configs/blupe_provider.example.json \
  --runner-source "$BLUPE_RUNNER_CHECKOUT/codex-runner/src/remote_yam"

python scripts/run_blupe_protocol_task.py \
  --runner-checkout "$BLUPE_RUNNER_CHECKOUT" \
  --config configs/blupe_provider.example.json \
  --candidate-config configs/blupe_local_candidates.example.json \
  --output output/first_grasp --seed 1040
```

运行结果保存为 `receipt.json`、原始模型回复、动作数据和 `physics/rollout.mp4`。每次运行应使用新输出目录。

### 4. 打开浏览器界面

```bash
python scripts/serve_blupe_recommended.py \
  --runner-checkout "$BLUPE_RUNNER_CHECKOUT" \
  --config configs/blupe_provider.example.json \
  --output output/browser_grasp --seed 1040 --port 18793
```

打开 `http://127.0.0.1:18793/`。该入口使用正式 provider 与官方 SDK，后端为本地 MuJoCo。界面中运行一次完整任务后可查看结果与视频，新任务需要新输出目录。更多实验见[仿真说明](docs/SIMULATION.md)。

## 当前演示与验收

- [完整抓放视频，16.08 秒](examples/blupe/local_pick_place.mp4)展示抓取、抬升、放置、释放和退让，同时保留邻近保护对象。
- [两物体记忆与重规划视频，29.80 秒](examples/blupe/local_memory_replan.mp4)展示第一物体完成后，待执行的第二物体受外力移动，框架重新观察、重规划并复用策略。
- 最新工程验收包含 3 个布局、2 种渲染的 6 次正常抓放，一次 ROI 截断拒绝和 10 项故障/记忆标准，保留了 60 次真实 Qwen 回复。故障批次有 4 项任务完成和 6 项预期停止。
- 验收提交 `440a40e` 的本地 GPU、Mesa 和 GitHub CI 均为 379 通过、12 跳过。[原始范围与回执](examples/blupe/pre_hardware_20261108.json)保持独立保存。

视频按仿真时间录制，推理等待单独记录。当前视觉入口依赖注册语义、受控图像候选、已知特征高度和实测桌面平面。首次真实硬件任务仍需现场标定、目的地、抓取偏移和操作者流程确认。

## 文件结构

| 目录 | 内容 |
| --- | --- |
| `agentic_rag_vlm/` | 核心框架、平台适配和仿真 |
| `scripts/` | 启动、模型服务、验收和审计入口 |
| `configs/` | 本地可运行示例与待填写的硬件模板 |
| `knowledge_base/` | 工程经验、关联图像和来源记录 |
| `tests/` | 单元、接口、物理执行及兼容回归 |
| `examples/blupe/` | 两段最新演示及验收摘要 |
| `paper/` | 录用论文 PDF、LaTeX、参考文献与插图 |
| `docs/` | 项目架构、启动、仿真、联调和复现说明 |
| `local/` | 本地私有配置、沟通材料、日志和归档索引，Git 忽略 |

旧交付包、旧演示、历史论文草稿和大型原始实验已转入本地归档。仍被核心模块或回归测试调用的兼容代码继续保留，其用途见[scripts 索引](scripts/README.md)。

## 进一步阅读

[架构与论文对应](docs/ARCHITECTURE.md) · [BluPe 联调](docs/BLUPE.md) · [验证范围](docs/VALIDATION.md) · [论文引用](paper/README.md)
