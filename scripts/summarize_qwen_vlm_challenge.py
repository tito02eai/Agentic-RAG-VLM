#!/usr/bin/env python3
"""Create a compact report for the Guanghua Agent-tool challenge."""

from __future__ import annotations

import argparse
import csv
import json
from math import comb, sqrt
from pathlib import Path

import numpy as np


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def mcnemar_exact(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, k) for k in range(min(b, c) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def metric(rows: list[dict[str, str]], key: str) -> str:
    values = [float(row[key]) for row in rows if row.get(key) not in {None, "", "None"}]
    return "—" if not values else f"{np.mean(values):.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("challenge_dir", type=Path)
    args = parser.parse_args()
    config = json.loads((args.challenge_dir / "frozen_config.json").read_text(encoding="utf-8"))
    with (args.challenge_dir / "challenge_results.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    lines = [
        f"# 光华 Agentic RAG-VLM {config['experiment_id']}",
        "",
        f"部署：`{config['deployment_label']}`；冻结 seeds：{config['seeds']}；共 {len(rows)} 个方法 rollout。",
        "本实验报告真实多模态高层机制与技能代理结果，不声明接触动力学抓取成功率。",
        "",
        "| 场景 | 条件 | 机制通过 | Wilson 95% CI | JSON有效 | 工具选择 | 机制分数 | 抓取准确 | 安全正确 | 变化检测 | 重规划 | 记忆 | 错误重规划 | 平均VLM调用 | P50延迟 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    pairs: list[str] = []
    for scene, conditions in config["conditions"].items():
        for condition in conditions:
            subset = [row for row in rows if row["scene"] == scene and row["condition"] == condition]
            successes = sum(float(row["mechanism_success"]) == 1.0 for row in subset)
            low, high = wilson(successes, len(subset))
            latencies = [float(row["latency_ms"]) / 1000.0 for row in subset]
            lines.append(
                f"| {scene} | {condition} | {successes}/{len(subset)} | [{low:.2f}, {high:.2f}] | "
                f"{metric(subset, 'parse_valid')} | {metric(subset, 'tool_selection_correct')} | {metric(subset, 'mechanism_score')} | {metric(subset, 'grasp_accuracy')} | "
                f"{metric(subset, 'safety_decision_correct')} | {metric(subset, 'change_detection_correct')} | "
                f"{metric(subset, 'replan_decision_correct')} | {metric(subset, 'memory_preserved')} | "
                f"{metric(subset, 'false_replan')} | {metric(subset, 'vlm_calls')} | {np.median(latencies):.2f}s |"
            )
        full = conditions[0]
        for baseline in conditions[1:]:
            b = c = 0
            shared_seeds = sorted({int(row["seed"]) for row in rows if row["scene"] == scene})
            for seed in shared_seeds:
                full_row = next(row for row in rows if row["scene"] == scene and row["condition"] == full and int(row["seed"]) == seed)
                baseline_row = next(row for row in rows if row["scene"] == scene and row["condition"] == baseline and int(row["seed"]) == seed)
                full_success = float(full_row["mechanism_success"]) == 1.0
                baseline_success = float(baseline_row["mechanism_success"]) == 1.0
                b += int(full_success and not baseline_success)
                c += int(not full_success and baseline_success)
            pairs.append(f"- {scene} `{full}` vs `{baseline}`：b={b}, c={c}, McNemar exact p={mcnemar_exact(b, c):.6g}。")

    event_lines = [
        "| 场景 | 条件 | 事件 | 通过 | 平均机制分数 | False-replan |",
        "|---|---|---|---:|---:|---:|",
    ]
    for scene in ("G3", "G4"):
        if scene not in config["conditions"]:
            continue
        for condition in config["conditions"][scene]:
            for event in ("target_move", "no_change", "irrelevant_fragile_move"):
                subset = [row for row in rows if row["scene"] == scene and row["condition"] == condition and row["event_type"] == event]
                if not subset:
                    continue
                successes = sum(float(row["mechanism_success"]) == 1.0 for row in subset)
                event_lines.append(
                    f"| {scene} | {condition} | {event} | {successes}/{len(subset)} | "
                    f"{metric(subset, 'mechanism_score')} | {metric(subset, 'false_replan')} |"
                )

    g2_lines = [
        "| 条件 | 易碎物距离 | 通过 | 可执行 offset |",
        "|---|---:|---:|---:|",
    ]
    if "G2" in config["conditions"]:
        seed_distance: dict[int, float] = {}
        for seed in config["seeds"]:
            scene_path = args.challenge_dir / "runs" / f"G2_seed{seed}_{config['conditions']['G2'][0]}" / "scene_spec.json"
            if scene_path.is_file():
                seed_distance[int(seed)] = float(json.loads(scene_path.read_text(encoding="utf-8"))["fragile_distance_m"])
        for condition in config["conditions"]["G2"]:
            for distance in sorted(set(seed_distance.values())):
                subset = [
                    row for row in rows
                    if row["scene"] == "G2" and row["condition"] == condition
                    and seed_distance.get(int(row["seed"])) == distance
                ]
                successes = sum(float(row["mechanism_success"]) == 1.0 for row in subset)
                g2_lines.append(f"| {condition} | {distance:.2f} m | {successes}/{len(subset)} | {metric(subset, 'offset_executable')} |")

    call_errors = 0
    total_calls = sum(int(float(row["vlm_calls"])) for row in rows)
    total_usage: dict[str, int] = {}
    tool_counts: dict[str, int] = {}
    tool_runs = 0
    for private_path in (args.challenge_dir / "runs").glob("*/private_evaluator.json"):
        private = json.loads(private_path.read_text(encoding="utf-8"))
        call_errors += int(bool(private.get("call_error")))
        trace = json.loads(private_path.with_name("public_trace.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        receipts = trace.get("agent_tool_receipts", [])
        tool_runs += int(bool(receipts))
        for receipt in receipts:
            name = str(receipt["tool"])
            tool_counts[name] = tool_counts.get(name, 0) + 1
        for key, value in trace["result"].get("usage", {}).items():
            if isinstance(value, int):
                total_usage[key] = total_usage.get(key, 0) + value
    lines.extend([
        "",
        "## 配对统计",
        "",
        *pairs,
        "",
        "## 事件条件结果",
        "",
        *event_lines,
        "",
        "事件分层用于区分真正的 target-move 恢复与 no-change / irrelevant-move 负对照，",
        "避免只看聚合成功率而把错误重规划误判为恢复能力。",
        "",
        "## G2 难度分层",
        "",
        *g2_lines,
        "",
        "## 调用与审计",
        "",
        f"- 方法 rollout：{len(rows)}；实际 VLM 调用：{total_calls}；最终 endpoint 错误：{call_errors}。",
        f"- Token：{total_usage.get('total_tokens', 0)}（prompt {total_usage.get('prompt_tokens', 0)}，completion {total_usage.get('completion_tokens', 0)}）。",
        f"- Agent 工具执行覆盖：{tool_runs} 个 rollout；工具调用：{sum(tool_counts.values())}。",
        *[f"  - `{name}`：{count} 次。" for name, count in sorted(tool_counts.items())],
        "- `validation_receipt.json` 审计 run/CSV/公开事件完整性以及 evaluator-only 字段隔离。",
        "",
        "## 设计升级",
        "",
        "- G1 在冻结 seed 间改变物体几何与正确 synergy，而不是固定 red=power、blue=pinch。",
        "- G2 只提供公开图关系和测量，不提供候选动作 offset。",
        "- G3/G4 混合目标移动、无变化和无关易碎物移动，报告 false-replan。",
        "- 所有 VLM 条件共享一次有界 schema repair，额外调用计入成本。",
        "- Full Agent 先由独立 Qwen 路由回合显式选择工具，再将候选交给三个确定性执行契约：RAG→手型、场景图→运动约束、变化+记忆→恢复；实验编号不参与工具分派。",
        "- 路由选择、原始候选、结构归一化、工具输入/输出和拒绝原因均写入 public trace；工具只读公开观测。",
        "- G4 联合可供性、邻居关系、变化事件、质量、摩擦、位姿和指令改写。",
    ])
    version = int(config.get("experiment_id", "v3").split("_v")[-1].split("_")[0]) if "_v" in config.get("experiment_id", "") else 3
    output = args.challenge_dir / f"CHALLENGE_V{version}_REPORT.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
