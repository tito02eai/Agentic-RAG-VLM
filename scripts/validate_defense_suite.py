#!/usr/bin/env python3
"""Audit defense-suite videos, Agentic traces, and private success gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_REPLANS = {"nominal": 0, "fragile": 0, "recovery": 1, "showcase": 1}
EXPECTED_GRAPH = {"nominal": False, "fragile": True, "recovery": False, "showcase": True}


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "output" / "defense_suite")
    args = parser.parse_args()
    failures: list[str] = []
    rows = []
    manifest = read_json(args.root / "suite_manifest.json")
    for scenario, expected_replans in EXPECTED_REPLANS.items():
        run_dir = args.root / scenario
        evaluator = read_json(run_dir / "private_evaluator.json")
        receipt = read_json(run_dir / "runtime_receipt.json")
        timeline = read_json(run_dir / "timeline.json")
        scene = read_json(run_dir / "scene_spec.json")
        trace = [json.loads(line) for line in (run_dir / "agentic_runtime_trace.jsonl").read_text(encoding="utf-8").splitlines()]
        video = run_dir / f"guanghua_agentic_rag_vlm_{scenario}.mp4"
        probe = json.loads(subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-show_entries", "stream=width,height,nb_frames,r_frame_rate", "-of", "json", str(video),
        ], check=True, capture_output=True, text=True).stdout)
        stream = probe["streams"][0]
        runtime = receipt["agentic_runtime"]
        checks = {
            "success": evaluator["task_success_under_declared_grasp_proxy"] is True,
            "runtime_complete": runtime["phase"] == "complete" and evaluator["runtime_complete"] is True,
            "two_targets": runtime["completed_targets"] == ["red_cube", "blue_cylinder"],
            "expected_replans": runtime["replan_count"] == expected_replans,
            "expected_graph": scene["scenario_spec"]["scene_graph_enabled"] is EXPECTED_GRAPH[scenario],
            "trace_completed": trace[-1]["event"] == "task_completed",
            "motion_gate": evaluator["visual_motion_admitted"] is True,
            "fragile_protected": evaluator["fragile_shift_m"] <= 0.002,
            "video_contract": (
                stream["width"] == 1280
                and stream["height"] == 720
                and stream["r_frame_rate"] == "30/1"
                and int(stream["nb_frames"]) == int(timeline["frames"])
                and abs(float(probe["format"]["duration"]) - float(timeline["duration_s"])) < 0.02
            ),
        }
        for name, passed in checks.items():
            if not passed:
                failures.append(f"{scenario}: {name}")
        rows.append({"scenario": scenario, "checks": checks, "video": str(video), "replans": runtime["replan_count"]})
    if set(manifest["scenarios"]) != set(EXPECTED_REPLANS):
        failures.append("manifest scenario set")
    report = {"passed": not failures, "failures": failures, "scenarios": rows}
    (args.root / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
