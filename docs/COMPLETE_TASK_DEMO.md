# 光华机器人完整任务演示

## 交付物

- 答辩成片：`output/complete_task_demo_v2_seed101/guanghua_agentic_rag_vlm_complete_success.mp4`
- 无字幕前视轨迹：`output/complete_task_demo_v2_seed101/guanghua_complete_task_raw.mp4`
- 无字幕俯视复核：`output/complete_task_demo_v2_seed101_birdview/guanghua_complete_task_raw.mp4`
- 私有评价：`output/complete_task_demo_v2_seed101/private_evaluator.json`
- 公共 Agent 轨迹：`output/complete_task_demo_v2_seed101/public_trace.jsonl`
- 分阶段运动记录：`output/complete_task_demo_v2_seed101/timeline.json`
- Qwen 与视频运行凭据：`output/complete_task_demo_v2_seed101/runtime_receipt.json`
- 冻结 v2 配置：`configs/complete_task_demo_v2.json`

成片为 1280×720、30 fps、21.3 s 的连续 MuJoCo 轨迹，未用剪切隐藏失败段。
旧的 `complete_task_demo_seed101` 含有欠约束 IK 和可见穿模，只保留作失败基线，禁止用于答辩。

## 演示任务

1. 使用真实本地 Qwen3.5-4B NF4 的 G1/G2/G3 seed 101 响应，选择红方块的
   `POWER` 和蓝圆柱的 `PINCH` 技能。
2. 通过场景图发现红方块旁的黄色易碎物，增加 30 mm 外偏并降低速度。
3. 红方块放入红色目标区后，外部将蓝圆柱平移约 29 mm。
4. Agent 保留已完成的红色子任务，触发 L3 重规划并从新观测位置完成蓝色子任务。
5. 私有评价器只在执行结束后检查两个目标误差和易碎物位移。

v2 另外修复了运动层：同时约束末端局部 X/Y 轴；依据进口手指网格标定 POWER/PINCH
抓取中心；将任务目标移入右臂舒适工作空间；以 30 mm 必要净空运输；对桌面净距、关节
限位余量、航点变化、末端误差和抓取对齐误差设置自动门禁。零位到任务准备位不属于
episode，视频从声明的碰撞检查任务准备姿态开始。

## 冻结结果

| 指标 | 结果 | 阈值 | 判定 |
|---|---:|---:|---|
| 红方块到目标中心误差 | 0 mm | ≤10 mm | PASS |
| 蓝圆柱到目标中心误差 | 0 mm | ≤10 mm | PASS |
| 黄色易碎物位移 | 0 mm | ≤2 mm | PASS |
| 最小指尖—桌面净距 | 1.10 mm | ≥-2 mm | PASS |
| 最大末端 IK 误差 | 0.00088 mm | ≤5 mm | PASS |
| 最大抓取中心对齐误差 | 0.00033 mm | ≤3 mm | PASS |
| 最小关节限位余量 | 0.1598 rad | ≥0.14 rad | PASS |
| 最大单关节航点变化 | 0.4156 rad | ≤1.5 rad | PASS |
| 视觉运动门禁 | true | 全部门禁通过 | PASS |
| 任务成功（声明的技能代理下） | true | 目标、易碎物与运动门禁同时通过 | PASS |

## 结论边界

此视频验证的是 **Agentic RAG-VLM 高层闭环**：多模态决策、经验检索、场景图约束、
完成状态记忆、变化检测和 L3 重规划。物体运输使用公开记录的
`calibrated kinematic grasp-skill proxy`，答辩字幕全程显示“校准运动技能代理”。

它不能被表述为“MuJoCo 接触动力学稳定抓取已经通过”。当前 G0-B 接触闭合与抬升仍未
准入；真实接触成功率需要在 G0-B 通过后重新运行冻结种子实验。这个边界不影响视频作为
Agentic 框架机制演示，但必须在答辩陈述中保留。

## 一键复现

完整 rollout 读取 `output/qwen35_nf4_main/runs/` 中已保存的真实 Qwen 响应：

```bash
cd Agentic-RAG-VLM
MUJOCO_GL=egl python scripts/render_complete_task_demo.py \
  --seed 101 --camera frontview --output output/complete_task_demo_v2_seed101
bash scripts/render_answer_video.sh
python scripts/validate_complete_task_demo.py output/complete_task_demo_v2_seed101

# 可选：同一轨迹的俯视穿模复核
MUJOCO_GL=egl python scripts/render_complete_task_demo.py \
  --seed 101 --camera birdview --output output/complete_task_demo_v2_seed101_birdview
```

核验结果：

```bash
python -m json.tool output/complete_task_demo_v2_seed101/private_evaluator.json
python scripts/validate_complete_task_demo.py output/complete_task_demo_v2_seed101
ffprobe -v error \
  -show_entries format=duration,size:stream=codec_name,width,height,r_frame_rate,pix_fmt \
  -of json output/complete_task_demo_v2_seed101/guanghua_agentic_rag_vlm_complete_success.mp4
```

当前答辩成片 SHA-256：
`2a37fb7a5c64b2ca0f76e64ab924c810ba7aa996837cbc01fe3db2fb47170052`。
