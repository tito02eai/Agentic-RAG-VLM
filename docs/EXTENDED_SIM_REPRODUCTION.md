# 多物体、接触恢复和记忆复用：运行说明

当前入口是 `scripts/run_extended_benchmark.py`。它启动独立的 MuJoCo
YAM 仿真进程，不创建真机队列、不调用真机运动接口。
实验定义见 [EXTENDED_SIM_PROTOCOL.md](EXTENDED_SIM_PROTOCOL.md)。

## 环境

正式运行使用 Python 3.10.21、MuJoCo 3.2.2；精确依赖记录在
`configs/extended_sim_dependencies.txt`。服务器需要 EGL 和 ffmpeg。

```bash
python -m pip install -r configs/extended_sim_dependencies.txt
python -m pip install -e .
MUJOCO_GL=egl python -m pytest -q tests
```

模型推理是独立服务，权重不在源码包内。本机配置如下，可通过命令行替换：

- Qwen OpenAI-compatible completions：`http://127.0.0.1:18070/v1/chat/completions`。
- 请求的模型：`/home/admin1/g2_multimodal_agent/models/Qwen3.5-4B`。
  验证各条推理回执中的 `response_model`，不要只根据 `/v1/models` 的别名判断模型。
- CLIP：`http://127.0.0.1:18072`，模型为 `openai/clip-vit-large-patch14`。
  `/health` 返回包括权重 SHA256 的 embedding namespace；与知识库不符则拒绝运行。

CLIP 服务示例（需要另装 `.[vision]`，使用本地完整权重）：

```bash
python scripts/serve_clip.py --model-path /absolute/path/to/clip-vit-large-patch14 \
  --device cpu --port 18072
```

`knowledge_base/yam_retention_observed.json` 包含三个独立标称摩擦成功样本，
附有原始图像、结果、动作回执及校验值。新的物理执行域是
`mjlab_yam_retention_v1`，不与之前 `yam_observed_clip.json` 的执行参数混用。

## 完整批次

```bash
MUJOCO_GL=egl python scripts/run_extended_benchmark.py \
  --output /absolute/path/to/new_batch \
  --endpoint http://127.0.0.1:18070/v1/chat/completions \
  --model /absolute/path/to/Qwen3.5-4B \
  --clip-endpoint http://127.0.0.1:18072
python scripts/audit_extended_benchmark.py /absolute/path/to/new_batch
python scripts/summarize_extended_benchmark.py /absolute/path/to/new_batch
python scripts/plot_extended_evidence.py /absolute/path/to/new_batch
python scripts/build_extended_viewer.py /absolute/path/to/new_batch
```

输出目录必须新建。运行器先冻结配置与包含知识库的源码 ZIP，
再从该快照依次启动 25 个案例；不依据前一案例的结果修改后续条件。
每个案例重新建立记忆，**不会把其他种子的成功经验提前带入**。
完整框架只在同一连续任务内复用已经验证的兼容类别经验。
关闭记忆/恢复的条件与完整框架共享场景、低层技能及评估标准。

`summary.json` 的 `dataset_complete` 表示所有案例均已产生结果，
不是“全部成功”。`physical_success` 才是每个案例的完整物理判据。
批次运行结束码 0 也不表示各消融条件成功，必须查看逐案例表。
进程异常未产出结果时批次会中止；保留该目录和日志，不能删除后当作未运行。

`audit_extended_benchmark.py` 把 JSON 审计结果打印到标准输出；交付时另存为
`artifact_audit.json`，只有 `passed: true` 才能称为完成证据一致性审计。
汇总脚本拒绝把不完整批次发布为最终结果。

## 单案例和诊断

```bash
MUJOCO_GL=egl python scripts/run_extended_sim.py \
  --store knowledge_base/yam_retention_observed.json \
  --mode stacked_sort --friction 0.7 --seed 200 \
  --record --output /absolute/path/to/new_single_case
```

可用 `--disable-memory` 或 `--disable-recovery` 做新案例对照。
`--diagnostic` 是独立的颜色/几何身份诊断，不是完整 VLM 框架，
不能与 `--store` 同用，其结果也不能混入正式全框架数据。
这轮冻结批次之后新增的便携性复测同样独立保存，不加入五种子结果。

## 证据阅读与复现边界

- `result.json`：目标完成判据、实际抓取尝试、恢复与记忆事件、模型原始输出。
- `observation_*/`：真实渲染 RGB-D、候选图像、VLM 输出。候选分割使用已声明的颜色规则；
  物体身份由模型判断，空间定位来自深度与已知尺寸。
- `public_trace.jsonl`：动作参数、抬升检查和放置承载力采样。
- `private_evaluator.jsonl`：仿真真值物体高度、关节与接触，供离线评估，未输入决策。
- `rollout.mp4`：首个声明种子的原始 25 fps 仿真视频，包含失败；模型推理期间仿真暂停，
  因此视频时长不能当作系统墙钟吞吐量。

五个种子只扰动物体初始位置数毫米，物体种类、摩擦与质量范围较窄。
Qwen 请求温度为 0，但服务端实现/硬件和数值环境仍可能导致重跑输出不同。
固定种子与保存原始回执用于可追溯，不保证逐位相同或所有重跑均成功。
要扩大结论，需要另外冻结更广的物体、材质、布局和扰动范围，再运行新的批次。

本轮没有验证接触后的 L2/L3 物理执行、跨会话长期经验迁移、灵巧手或真机力控。
现有远程硬件路径默认没有启用仿真专用的接触恢复回调。
