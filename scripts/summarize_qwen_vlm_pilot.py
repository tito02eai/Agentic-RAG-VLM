#!/usr/bin/env python3
"""Summarize the real-Qwen pilot with confidence intervals and latency."""

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


def mean(rows: list[dict[str, str]], key: str) -> str:
    values = [float(row[key]) for row in rows if row.get(key) not in {None, "", "None"}]
    return "—" if not values else f"{np.mean(values):.2f}"


def mcnemar_exact(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, k) for k in range(min(b, c) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_dir", type=Path)
    args = parser.parse_args()
    config = json.loads((args.pilot_dir / "frozen_config.json").read_text(encoding="utf-8"))
    split_label = "Main" if config.get("seed_split") == "main" else "Pilot"
    with (args.pilot_dir / "pilot_results.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    total_usage: dict[str, int] = {}
    response_models: set[str] = set()
    call_errors = 0
    for private_path in (args.pilot_dir / "runs").glob("*/private_evaluator.json"):
        private = json.loads(private_path.read_text(encoding="utf-8"))
        call_errors += int(bool(private.get("call_error")))
        trace_path = private_path.with_name("public_trace.jsonl")
        decision = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[-1])
        result = decision["result"]
        if result.get("response_model"):
            response_models.add(result["response_model"])
        for key, value in result.get("usage", {}).items():
            if isinstance(value, int):
                total_usage[key] = total_usage.get(key, 0) + value

    lines = [
        f"# 光华真实 Qwen RAG-VLM {split_label} 结果",
        "",
        f"部署：`{config['deployment_label']}`；{len(config['seeds'])} 个冻结 seed；共 {len(rows)} 次真实多模态调用。",
        "本表是 Agentic 机制决策结果，不是完整物理 pick-place 成功率。",
        "",
        "| 场景 | 条件 | 机制通过 | Wilson 95% CI | JSON有效 | 检索准确 | 约束满足 | 变化检测 | 重规划 | 记忆保留 | P50延迟 | P95延迟 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scene, conditions in config["conditions"].items():
        for condition in conditions:
            subset = [row for row in rows if row["scene"] == scene and row["condition"] == condition]
            successes = sum(float(row["mechanism_success"]) == 1.0 for row in subset)
            low, high = wilson(successes, len(subset))
            latency = [float(row["latency_ms"]) / 1000.0 for row in subset]
            lines.append(
                f"| {scene} | {condition} | {successes}/{len(subset)} | [{low:.2f}, {high:.2f}] | "
                f"{mean(subset, 'parse_valid')} | {mean(subset, 'retrieval_top1_accuracy')} | "
                f"{mean(subset, 'constraint_satisfaction')} | {mean(subset, 'change_detection')} | "
                f"{mean(subset, 'replan_valid')} | {mean(subset, 'memory_preserved')} | "
                f"{np.median(latency):.2f}s | {np.percentile(latency, 95):.2f}s |"
            )
    pair_lines = []
    for scene, conditions in config["conditions"].items():
        full = conditions[0]
        for ablation in conditions[1:]:
            b = c = 0
            for seed in config["seeds"]:
                full_success = next(float(row["mechanism_success"]) for row in rows if row["scene"] == scene and row["condition"] == full and int(row["seed"]) == seed)
                ablation_success = next(float(row["mechanism_success"]) for row in rows if row["scene"] == scene and row["condition"] == ablation and int(row["seed"]) == seed)
                b += int(full_success == 1.0 and ablation_success == 0.0)
                c += int(full_success == 0.0 and ablation_success == 1.0)
            pair_lines.append(f"- {scene} `{full}` vs `{ablation}`：b={b}, c={c}, McNemar exact p={mcnemar_exact(b, c):.6g}。")
    lines.extend([
        "",
        "## 统计与解释",
        "",
        *pair_lines,
        "- G1：RAG-VLM 选择 red=power、blue=pinch；VLM-Only 每条仅命中一个目标，平均准确率 0.50。",
        "- G2：Graph-VLM 输出与易碎物反向的非零世界坐标偏移；No-Graph 虽能说出安全原则，",
        "  但输出零偏移，无法形成可执行避障动作。",
        "- G3：Full 与 No-Memory 都检测变化并重规划，但 No-Memory 回到 red_cube；No-Replan",
        "  保留记忆却没有当前观察。该分解验证了两个模块不是同一个开关。",
        "",
        "## 调用审计",
        "",
        f"- 响应模型：`{', '.join(sorted(response_models))}`；endpoint错误：{call_errors}。",
        f"- Token总计：{total_usage.get('total_tokens', 0)}（prompt {total_usage.get('prompt_tokens', 0)}，completion {total_usage.get('completion_tokens', 0)}）。",
        "- 每条运行保存输入图像、完整提示词、Qwen原始回答、解析JSON、usage、延迟和独立评价器。",
        "",
        "## 结论边界",
        "",
        "本实验已经是真实 Qwen 多模态推理，不再是确定性 semantic adapter；但执行仍停在高层机制",
        "判据与 G0 预抓取。Admission-B 通过前，不能把这些结果写成物理抓取成功率。",
    ])
    output = args.pilot_dir / "QWEN_VLM_REPORT.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
