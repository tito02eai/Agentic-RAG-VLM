#!/usr/bin/env python3
"""Create a compact Markdown report from the frozen Agentic pilot CSV."""

from __future__ import annotations

import argparse
import csv
from math import sqrt
from pathlib import Path


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with (args.pilot_dir / "pilot_results.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pairs = []
    for scene in ("G1", "G2", "G3"):
        conditions = sorted({row["condition"] for row in rows if row["scene"] == scene}, key=lambda value: (value != "C2_full", value))
        for condition in conditions:
            subset = [row for row in rows if row["scene"] == scene and row["condition"] == condition]
            successes = sum(float(row["mechanism_success"]) == 1.0 for row in subset)
            low, high = wilson(successes, len(subset))
            def mean(key: str) -> str:
                values = [float(row[key]) for row in subset if row[key] not in {"", "None"}]
                return "—" if not values else f"{sum(values) / len(values):.2f}"
            pairs.append((scene, condition, successes, len(subset), low, high, mean("retrieval_top1_accuracy"), mean("constraint_satisfaction"), mean("change_detection"), mean("replan_valid"), mean("memory_preserved")))
    lines = [
        "# Guanghua Agentic RAG-VLM 冻结 Pilot 结果",
        "",
        "本表报告机制级判据，不是完整物理抓取成功率。每个条件使用 5 个冻结 seed，",
        "Agent 只读取渲染 RGB-D；MuJoCo 坐标只由独立 evaluator 使用。",
        "",
        "| 场景 | 条件 | 机制通过 | Wilson 95% CI | 检索 Top-1 | 约束满足 | 变化检测 | 重规划 | 记忆保留 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scene, condition, success, total, low, high, retrieval, constraint, change, replan, memory in pairs:
        lines.append(f"| {scene} | {condition} | {success}/{total} | [{low:.2f}, {high:.2f}] | {retrieval} | {constraint} | {change} | {replan} | {memory} |")
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- G1：Full 为方块选择 power、为细圆柱选择 pinch；No-RAG 使用通用 power，因此仅 50% synergy 正确。",
            "- G2：Full 从公开场景图识别 10 cm 内易碎邻居并生成远离方向、+30 mm 高度和 0.8 倍力度；No-Graph 不生成约束。",
            "- G3：两种条件都能从前后 RGB-D 检测约 29 mm 位移；No-Memory 会重复已完成红方块，No-Replan 保留旧蓝圆柱目标。",
            "",
            "5-seed pilot 只用于确认场景能隔离对应机制。正式论文统计需要冻结实现后再运行 20 个 main seeds；",
            "当前结果不能与论文中缺失原始代码/经验库的 78.3% 物理任务成功率直接比较。",
        ]
    )
    output = args.pilot_dir / "PILOT_REPORT.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
