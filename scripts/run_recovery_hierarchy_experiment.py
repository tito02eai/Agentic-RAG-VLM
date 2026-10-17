#!/usr/bin/env python3
"""Exercise the paper's bounded L1/L2/L3 correction policy with receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agentic_rag_vlm.pipeline import AgenticPipeline, PipelineConfig  # noqa: E402
from scripts.agentic_framework import PublicObject, load_cards  # noqa: E402


CONFIG = PROJECT_ROOT / "configs" / "recovery_hierarchy_v1.json"
CARDS = PROJECT_ROOT / "knowledge_base" / "affordance_cards.json"
METHODS = ("T0_no_recovery", "T1_fixed_retry", "A2_hierarchical")


def target() -> PublicObject:
    return PublicObject(
        "blue_cylinder", (-0.18, -0.03, 0.861), "cylinder", "rigid_body", "pinchable",
        "rigid", 0.2, "body", 2.1,
    )


def objects() -> dict[str, PublicObject]:
    return {
        "blue_cylinder": target(),
        "fragile_proxy": PublicObject(
            "fragile_proxy", (0.05, 0.10, 0.876), "cylinder", "container", "fragile",
            "glass", 1.0, "body", 2.2,
        ),
    }


def quality(failed_factor: str | None) -> dict[str, float]:
    values = {
        "position_accuracy": 0.95,
        "width_compatibility": 0.95,
        "force_appropriateness": 0.95,
        "grasp_type_match": 0.95,
        "approach_clearance": 0.95,
        "object_difficulty": 0.95,
        "grip_security": 0.95,
    }
    if failed_factor is not None:
        values[failed_factor] = 0.1
    return values


SCENARIO_OBSERVATIONS: dict[str, list[dict[str, Any]]] = {
    "F0_no_failure": [{"quality_factors": quality(None)}],
    "F1_position_bias": [
        {"failure_type": "position_error", "quality_factors": quality("position_accuracy")},
        {"quality_factors": quality(None)},
    ],
    "F2_wrong_grasp_family": [
        {"failure_type": "orientation", "quality_factors": quality("grasp_type_match")},
        {"failure_type": "orientation", "quality_factors": quality("grasp_type_match")},
        {"quality_factors": quality(None)},
    ],
    "F3_stale_scene_plan": [
        {"failure_type": "unreachable_pose", "quality_factors": quality("object_difficulty")},
        {"failure_type": "unreachable_pose", "quality_factors": quality("object_difficulty")},
        {"failure_type": "unreachable_pose", "quality_factors": quality("object_difficulty")},
        {"quality_factors": quality(None)},
    ],
    "F4_timeout": [
        {"failure_type": "timeout", "quality_factors": quality("object_difficulty")},
    ],
}


class ScriptedExecutor:
    def __init__(self, observations: list[dict[str, Any]]) -> None:
        self.observations = observations
        self.index = 0

    def __call__(self, action: object) -> dict[str, Any]:
        index = min(self.index, len(self.observations) - 1)
        self.index += 1
        payload = dict(self.observations[index])
        payload["executor_receipt"] = {
            "scripted_attempt": index,
            "action_source": getattr(action, "source", "unknown"),
            "fault_role": "private_verifier_injection",
        }
        return payload


def baseline_result(method: str, scenario: str) -> dict[str, object]:
    observations = SCENARIO_OBSERVATIONS[scenario]
    maximum_attempts = 1 if method == "T0_no_recovery" else 2
    used = observations[:maximum_attempts]
    success = any(all(value >= 0.4 for value in item["quality_factors"].values()) for item in used)
    safe_stop = scenario == "F4_timeout"
    return {
        "success": success,
        "safe_stop": safe_stop,
        "attempts": len(used),
        "recovery_levels": [] if method == "T0_no_recovery" else ([1] if len(used) > 1 else []),
        "trace": [
            {"attempt": index, "observation": item, "method_unchanged": True}
            for index, item in enumerate(used)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output" / "recovery_hierarchy_v1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = []
    traces: dict[str, object] = {}
    for scenario, spec in frozen["scenarios"].items():
        for method in METHODS:
            if method == "A2_hierarchical":
                pipeline = AgenticPipeline(
                    load_cards(CARDS),
                    PipelineConfig(max_retries=3, recovery_enabled=True),
                )
                result = pipeline.run(
                    f"{scenario}_{method}", target(), objects(), ScriptedExecutor(SCENARIO_OBSERVATIONS[scenario])
                )
                levels = [
                    int(event["reflection"]["level"])
                    for event in result["trace"] if "reflection" in event
                ]
                safe_stop = any(
                    event.get("reflection", {}).get("safe_stop", False) for event in result["trace"]
                )
                normalized = {
                    "success": bool(result["success"]),
                    "safe_stop": bool(safe_stop),
                    "attempts": int(result["attempts"]),
                    "recovery_levels": levels,
                    "trace": result["trace"],
                }
            else:
                normalized = baseline_result(method, scenario)
            expected = str(spec["expected"])
            expected_level = {"L1_parameter_retry": 1, "L2_method_switch": 2, "L3_full_replan": 3}.get(expected)
            mechanism_correct = (
                (expected == "no_recovery" and normalized["success"] and not normalized["recovery_levels"])
                or (expected == "safe_stop" and normalized["safe_stop"])
                or (
                    expected_level is not None
                    and normalized["success"]
                    and expected_level in normalized["recovery_levels"]
                )
            )
            rows.append({
                "scenario": scenario,
                "method": method,
                "expected": expected,
                "success": normalized["success"],
                "safe_stop": normalized["safe_stop"],
                "attempts": normalized["attempts"],
                "recovery_levels": normalized["recovery_levels"],
                "mechanism_correct": mechanism_correct,
            })
            traces[f"{scenario}_{method}"] = normalized["trace"]

    summary = {}
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        recovery = [row for row in subset if row["scenario"] in {"F1_position_bias", "F2_wrong_grasp_family", "F3_stale_scene_plan"}]
        summary[method] = {
            "recovery_success_rate": sum(bool(row["success"]) for row in recovery) / len(recovery),
            "mechanism_correct_rate": sum(bool(row["mechanism_correct"]) for row in subset) / len(subset),
            "safe_stop_on_timeout": next(bool(row["safe_stop"]) for row in subset if row["scenario"] == "F4_timeout"),
            "unnecessary_recovery_on_no_failure": bool(next(row["recovery_levels"] for row in subset if row["scenario"] == "F0_no_failure")),
        }
    payload = {
        "experiment_id": frozen["experiment_id"],
        "claim_boundary": frozen["claim_boundary"],
        "rows": rows,
        "summary": summary,
        "traces": traces,
    }
    (args.output / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "frozen_config.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

