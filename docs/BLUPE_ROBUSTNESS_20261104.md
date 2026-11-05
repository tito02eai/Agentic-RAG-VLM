# BluPe 生产接口链路：故障、记忆与停止验收（0.3.4）

2026-11-04 完成。本地 MuJoCo + 官方 Runner/真实 HTTP/WebSocket SDK +
生产 AgenticRAGVLMProvider/原生 IK + 真实 Qwen/CLIP。
**没有连接 BluPe 公开硬件，也不是 BluPe 官方云端仿真环境。**

## 最终结果

**10 项验收条件全部通过：4 项完成任务，6 项按预期停止。**
这不是“10 次抓取全部成功”，也不是硬件成功率。每项仅在已知布局 seed 1010
运行一次，用于验证指定行为；不估计总体可靠性，不与之前的新布局结果混合。

| 案例 | 实际扰动与要求 | 结果 |
|---|---|---|
| 正常抓放 | 红方块放到蓝区，保护绿色容器，独立检查接触、抬起、释放与支撑 | 完成 |
| 抓空 | 执行器故障令夹爪保持打开；必须发现保持失败 | 停止；无放置，无成功误报 |
| 抬起后滑落 | 双侧接触抬升持续 0.15 秒后令夹爪打开，物体依重力落下 | 停止；无放置，无成功误报 |
| 记忆复用且目标移动 | 第一块完成后，对第二块施加 0.9 N、0.1 秒外力 | 移动约 5.55 cm；L3 重规划；复用经验并使用新位置，两块均完成 |
| 复用后失败 | 第一块验证成功后，第二块使用记忆；第二次抓取注入执行器故障 | 保持失败后删除记忆，记录事件并停止 |
| 确认响应丢失一次 | 实际接收轨迹后，将第一次确认响应替换为 HTTP 503 | 相同 ID/内容重试一次；轨迹不重复执行，任务完成 |
| 持续丢失确认 | 首个已接收轨迹的所有确认响应均返回 503 | 相同 ID/内容重试；只执行一个轨迹包，停止 |
| 执行反馈中断 | 下发第一包后，会话与轨迹反馈 GET 持续返回 503 | 无完成证据时不继续动作，停止 |
| WebSocket 断开 | 真实事件连接在初始观测前关闭 | 重连同一会话，完成一次任务 |
| 运动中 Stop | 实际执行至少 3 个轨迹采样点后调用官方 Stop | 返回 aborted；无后续轨迹包，无成功误报 |

故障是为验证边界而明确注入的实验条件，不伪装为自然失败。没有修改模型回复、
规划器输出、物体坐标或成功标签。物体移动使用外力和物理积分，没有抓取粘附或
初始化后的物体位置重置。四类保护物/机器人接触安全检查均通过。

独立审计零错误。最终批次保存 **36 次真实 Qwen3.5-4B 回复**、轨迹、执行事件、
原始图像、物理评估、记忆快照和每例视频。完整代码回归 **335 通过、12 跳过**：
10 项缺失的历史学校资产测试，以及 2 项未配置的旧回放图像模型测试。
官方 Runner 集成测试通过；本轮真实模型调用不依赖被跳过的旧回放测试。

## 修复与保留的失败

1. 多物体开发首轮在运动前停止：黄色表面在红/绿主通道间变化，产生重复候选。
   修复为统一黄色色带连通区域，不赋予候选物体身份，身份仍由 VLM 判别。
   新增回归覆盖该问题，开发失败独立保留，不计入最终验收。
2. 第一轮冻结的严格验收 **9/10**。失败记忆已删除，命令回执也有记录，但统一
   runtime trace 缺少 `memory_invalidation` 事件。该次严格验收失败保持原样。
   将执行阶段和验证阶段的失效处理统一接入 runtime 事件后，重新冻结运行全部
   十项，最终全部通过，没有改写第一轮结果。

最终移动目标案例中，第二次动作距新位置约 **1.3 mm（平面误差）**，距旧位置约
**5.66 cm**，验证该场景下复用经验而不沿用旧坐标。它是同一会话中两个不同同类
物体间的复用，不是跨会话长期记忆验证。

## 复现与证据

使用 Python 3.10、`configs/extended_sim_dependencies.txt` 的固定依赖，启动本地
18074（Qwen）和 18072（CLIP）。官方 Runner revision：
`e6040394c5ed62efbcc7d627acc0dfc8ccc100a6`。

```bash
python scripts/run_blupe_robustness_batch.py \
  --output /new/robustness_batch \
  --runner-checkout /path/to/blupe-remote-bots
python scripts/audit_blupe_robustness.py /new/robustness_batch \
  --output /new/robustness_batch/audit.json
BLUPE_RUNNER_CHECKOUT=/path/to/blupe-remote-bots python -m pytest -q
```

- `scripts/run_blupe_robustness_case.py`：逐例真实链路运行，本地故障注入与物理评估。
- `scripts/run_blupe_robustness_batch.py`：运行前冻结代码、知识和模型身份，保留所有结果。
- `scripts/audit_blupe_robustness.py`：从原始模型、轨迹、执行采样和物理记录重新核验。
- `runtime.py`、`platform/session.py`：统一记忆失效事件，包含保持失败和放置命令失败。
- `simulation/protocol_gateway.py`：可选双物体测试场景及 RGB 区域修复。

原始证据位于项目根目录 `仿真实验/blupe_robustness_20261104/`：
`qualification_v1`、`qualification_v2` 各自包含冻结源码、协议、案例和审计。
最终执行源码 ZIP SHA256：
`8d5191641cb1fe3bbb648b443f829f588bcd31d529df028995d77000b346ca6d`。
当前代码与冻结执行版本的全部 Python 文件逐个校验一致；后补说明另行打包。

评估文件分层：底层 `physics/result.json` 是基类对红色主目标的评估；多物体案例
以案例根目录的 `physical_result.json` 为完整物理评估，它还要求第二目标完成。
因此“复用后失败”中，第一目标完成可以使底层评估为真，但完整物理评估为假，
Agent 也明确停止。`receipt.json` 分别记录验收条件是否满足与任务是否完成。

## 当前判断与边界

本轮针对性本地验收完成，可以冻结版本并推进真实相机预览、实机参数配置和首次
BluPe 硬件联调。**接触后失败默认保守停止，没有在实机路径开放自动重抓**，
不能称为“故障后自动恢复抓取成功”。

场景仍是已知物体、受控颜色候选、已知尺寸与水平桌面。一个活动仿真机械臂搭配
静止虚拟另一臂通道，不是经过标定的 BluPe 双臂数字孪生。夹爪保持判据是
关节/开口反馈代理，不能替代硬件触觉或已校准力控制。视频按仿真时钟播放，
不包含推理等待时间，每例墙钟时间另存。

实机仍需确认现场标定准确度、物体/目标区几何、夹爪反馈语义及工作站就绪状态。
本轮没有创建公开机器人会话、发布公共模型或发送合作邮件。

## Partner-facing summary

The production provider passed ten declared local functional checks through the
official BluPe HTTP/WebSocket runner and actuator-driven MuJoCo physics. Four
cases completed tasks; six intentionally required a stop. Coverage includes
grasp loss, moved-target memory reuse, failed-memory eviction, stable-ID retries,
missing feedback, event-stream reconnect and operator cancellation. Original
failures and raw evidence are retained. This qualifies the local integration
path, not hardware performance. Site setup and a first hardware trial remain.
