# Agentic RAG-VLM 可复用复现说明

## 复现范围

本目录依据 `paper/Agentic-RAG-VLM/main_final.tex` 重建了论文描述的核心软件结构：

- HAA-RAG：类别、affordance、视觉三级评分，最终权重为 0.2/0.4/0.4；
- 场景图：对象节点、10 cm 邻接边和易碎邻居约束；
- Agentic ReAct：Thought → Action → Observation → Reflection；
- 七因子质量模型：论文给出的 0.20/0.15/0.10/0.15/0.15/0.10/0.15 权重；
- 14 类失败以及 L1 参数调整、L2 方法切换、L3 完整重规划；
- 按对象类别索引的 session episodic memory；
- 可替换的 perception、robot executor、public verifier 和 private evaluator 接口。

## 无法逐数值复现的部分

原工作区只有论文，没有论文声称使用的 116 条经验库、Qwen3-VL prompt、CLIP embedding、
七个质量因子的具体计算式、原始 12-task trial 日志或随机种子。因此：

- `knowledge_base/affordance_cards.json` 是 8 条可审计的小型示例库；
- 已接入本地 Qwen3.5-4B 多模态模型；正式 pilot 使用 bnb-NF4、temperature 0，完整保存
  图像、prompt、原始回答、解析 JSON、token 和延迟；
- `deterministic_color_geometry_adapter_v1` 只保留为无 GPU 的快速协议诊断，不再作为
  主要 VLM 实验结果；
- 七因子加权公式与恢复规则可复现，但不能宣称重现论文 78.3% 数字；
- Qwen adapter 已实现；后续替换为论文同款 Qwen3-VL-8B 或 CLIP 向量召回时，
  Planner/Memory/Recovery 与评价协议不变。

## 目录与入口

- `agentic_rag_vlm/pipeline.py`：Agentic 主循环；
- `agentic_rag_vlm/quality.py`：七因子质量模型；
- `agentic_rag_vlm/recovery.py`：14 类失败与三级恢复；
- `agentic_rag_vlm/memory.py`：episodic memory；
- `agentic_rag_vlm/adapters.py`：外部 VLM、真实机器人和评价器接口；
- `agentic_rag_vlm/qwen_adapter.py`：OpenAI-compatible Qwen 图像调用与 JSON 解析；
- `scripts/agentic_framework.py`：光华 RGB-D semantic adapter、HAA-RAG 和场景图；
- `scripts/run_agentic_pilot.py`：G1–G3 成对机制实验；
- `scripts/run_qwen_vlm_pilot.py`：支持 5-seed pilot 与 20-seed main 的真实 Qwen 成对实验；
- `scripts/run_qwen_vlm_challenge.py`：G1–G4 困难场景、负样本、Agentic 分解与有界修复；
- `scripts/summarize_qwen_vlm_pilot.py`：真实模型统计、token 与延迟报告；
- `scripts/summarize_qwen_vlm_challenge.py`：事件条件、难度分层、配对统计和调用成本；
- `scripts/validate_agentic_artifacts.py`：检查产物完整性与 public/private 隔离；
- `scripts/run_g0_admission.py`：光华本体 RGB-D 到安全预抓取；
- `scripts/render_complete_task_demo.py`：读取真实 Qwen 凭据并生成完整双物体任务轨迹；
- `scripts/render_answer_video.sh`：在不改动仿真帧的前提下生成答辩字幕成片；
- `scripts/validate_complete_task_demo.py`：核验成片参数、评价阈值、真实 Qwen 凭据和代理披露；
- `tests/test_agentic_framework.py`：核心算法契约测试。

## 运行

```bash
cd Agentic-RAG-VLM
python scripts/prepare_assets.py
MUJOCO_GL=egl python scripts/run_qwen_vlm_challenge.py \
  --output output/qwen35_challenge_v2_main --seed-split main \
  --endpoint http://127.0.0.1:18070/v1/chat/completions \
  --model /home/admin1/g2_multimodal_agent/models/Qwen3.5-4B \
  --deployment-label qwen3.5-4b-bf16-local
python scripts/summarize_qwen_vlm_challenge.py output/qwen35_challenge_v2_main
python scripts/validate_agentic_artifacts.py output/qwen35_challenge_v2_main

# 旧 v1 机制诊断
MUJOCO_GL=egl python scripts/run_qwen_vlm_pilot.py \
  --output output/qwen35_nf4_main --seed-split main \
  --endpoint http://127.0.0.1:18110/v1/chat/completions \
  --model /home/admin1/g2_multimodal_agent/models/Qwen3.5-4B \
  --deployment-label qwen3.5-4b-bnb-nf4-local
python scripts/summarize_qwen_vlm_pilot.py output/qwen35_nf4_main
python scripts/validate_agentic_artifacts.py output/qwen35_nf4_main

# 可选：无模型服务的快速协议诊断
MUJOCO_GL=egl python scripts/run_agentic_pilot.py --output output/agentic_pilot
python scripts/summarize_agentic_pilot.py output/agentic_pilot
python scripts/validate_agentic_artifacts.py output/agentic_pilot
MUJOCO_GL=egl python scripts/run_g0_admission.py --seed 7 --output output/g0_admission_seed7

# 真实 Qwen 凭据驱动的完整 Agentic 任务与答辩成片
MUJOCO_GL=egl python scripts/render_complete_task_demo.py \
  --seed 101 --camera frontview --output output/complete_task_demo_v2_seed101
bash scripts/render_answer_video.sh
python scripts/validate_complete_task_demo.py output/complete_task_demo_v2_seed101
pytest -q tests
```

Challenge/Pilot 的每条 run 都分别保存 `public_trace.jsonl`、`private_evaluator.json` 和
`scene_spec.json`。正式实验不得把 private evaluator 字段输入 Planner。

v2 完整演示在校准运动技能代理下通过：红/蓝目标关系均为真、易碎物位移为 0 mm、
最小指尖—桌面净距为 1.10 mm，最大末端 IK 误差小于 0.001 mm。旧 v1 轨迹存在可见
穿模和欠约束姿态，只保留作失败基线。v2 验证 Agentic 高层闭环，不替代尚未准入的
G0-B 接触动力学抓取实验。完整证据与答辩措辞见 `docs/COMPLETE_TASK_DEMO.md`。

## 后续扩展接口

1. 将完整经验记录追加到 affordance cards，并替换为 CLIP/向量库召回；
2. 在同一冻结协议下比较 Qwen3.5-4B、Qwen3-VL-4B/8B 与远程 VLM；
3. 用真实机器人 SDK 实现 `RobotExecutor`，保持 Planner、Memory 和 Recovery 不变；
4. Admission-B 通过后，把当前机制判据接到完整 pick-place executor，再运行 20 个冻结 main seeds；
5. 若转向 CARVE-VLA，将 Agent 输出的结构化子目标作为 VLA 的高层条件，不必重写评价协议。
