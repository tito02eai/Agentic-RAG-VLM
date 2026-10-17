#!/usr/bin/env python3
"""Validate the complete Agentic demo, video properties, and claim boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = PROJECT_ROOT / "output" / "corrected_motion_mesh_gate_final_v5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path, default=DEFAULT_RUN)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    errors: list[str] = []
    required = {
        "raw_video": run_dir / "guanghua_complete_task_raw.mp4",
        "evaluator": run_dir / "private_evaluator.json",
        "trace": run_dir / "public_trace.jsonl",
        "timeline": run_dir / "timeline.json",
        "receipt": run_dir / "runtime_receipt.json",
    }
    for name, path in required.items():
        if not path.is_file():
            errors.append(f"missing {name}: {path}")

    evaluator: dict[str, Any] = {}
    timeline: dict[str, Any] = {}
    receipt: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    if required["evaluator"].is_file():
        evaluator = read_json(required["evaluator"])
        if evaluator.get("task_success_under_declared_grasp_proxy") is not True:
            errors.append("declared-proxy task success is not true")
        if evaluator.get("contact_dynamics_admitted") is not False:
            errors.append("contact_dynamics_admitted must remain false")
        if evaluator.get("visual_motion_admitted") is not True:
            errors.append("visual motion gate is not admitted")
        if float(evaluator.get("red_target_error_m", 1.0)) > 0.01:
            errors.append("red target error exceeds 10 mm")
        if float(evaluator.get("blue_target_error_m", 1.0)) > 0.01:
            errors.append("blue target error exceeds 10 mm")
        if float(evaluator.get("fragile_shift_m", 1.0)) > 0.002:
            errors.append("fragile shift exceeds 2 mm")

    if required["timeline"].is_file():
        timeline = read_json(required["timeline"])
        phase_names = {phase.get("name") for phase in timeline.get("phases", [])}
        expected_phases = {
            "haa_rag_and_scene_graph",
            "red_subgoal_verified",
            "external_blue_displacement",
            "public_change_detected",
            "blue_release_to_preshape",
            "task_success",
        }
        missing_phases = sorted(expected_phases - phase_names)
        if missing_phases:
            errors.append(f"missing timeline phases: {missing_phases}")
        if timeline.get("fps") != 30 or timeline.get("frames") != 798:
            errors.append("unexpected timeline frame contract")
        motion = timeline.get("motion_safety", {})
        if float(motion.get("minimum_table_clearance_m", -1.0)) < -0.002:
            errors.append("fingertip-table clearance gate failed")
        if float(motion.get("maximum_terminal_position_error_m", 1.0)) > 0.005:
            errors.append("terminal IK error gate failed")
        if float(motion.get("maximum_grasp_alignment_error_m", 1.0)) > 0.003:
            errors.append("grasp alignment gate failed")
        if float(motion.get("minimum_visual_table_clearance_m", -1.0)) < -0.0005:
            errors.append("visible hand-table mesh gate failed")
        if float(motion.get("minimum_visual_hand_object_clearance_m", -1.0)) < -0.0005:
            errors.append("visible hand-object mesh gate failed")
        if float(motion.get("minimum_protected_object_clearance_m", -1.0)) < 0.003:
            errors.append("protected-object clearance gate failed")
        if float(motion.get("minimum_joint_limit_margin_rad", -1.0)) < 0.14:
            errors.append("joint-limit margin gate failed")

    if required["receipt"].is_file():
        receipt = read_json(required["receipt"])
        expected_receipts = {
            "G1_seed101_R2_rag_vlm",
            "G2_seed101_R2_graph_vlm",
            "G3_seed101_R2_agentic_vlm",
        }
        if set(receipt.get("qwen_receipts", [])) != expected_receipts:
            errors.append("real-Qwen receipt set does not match the frozen demo")

    if required["trace"].is_file():
        events = [json.loads(line) for line in required["trace"].read_text(encoding="utf-8").splitlines()]
        qwen_events = [event for event in events if event.get("event", "").startswith("qwen_")]
        proxy_events = [event for event in events if "grasp_skill_proxy" in event.get("event", "")]
        if len(qwen_events) != 3:
            errors.append(f"expected 3 Qwen receipts, found {len(qwen_events)}")
        if any(event.get("uses_privileged_simulator_state") is not False for event in qwen_events):
            errors.append("Qwen/Planner receipt incorrectly uses privileged simulator state")
        if len(proxy_events) != 2:
            errors.append(f"expected 2 calibrated proxy events, found {len(proxy_events)}")
        for event in proxy_events:
            if event.get("privileged_state_used_by_executor") is not True:
                errors.append(f"proxy executor privilege is not disclosed: {event.get('event')}")
            if event.get("privileged_state_exposed_to_planner") is not False:
                errors.append(f"proxy state leaked to Planner: {event.get('event')}")

    video_probe: dict[str, Any] = {}
    if required["raw_video"].is_file():
        command = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration,size:stream=codec_name,width,height,r_frame_rate,pix_fmt",
            "-of", "json", str(required["raw_video"]),
        ]
        video_probe = json.loads(subprocess.check_output(command, text=True))
        stream = video_probe.get("streams", [{}])[0]
        video_format = video_probe.get("format", {})
        if (stream.get("width"), stream.get("height")) != (1280, 720):
            errors.append("answer video is not 1280x720")
        if stream.get("r_frame_rate") != "30/1":
            errors.append("answer video is not 30 fps")
        if abs(float(video_format.get("duration", 0.0)) - 26.6) > 0.02:
            errors.append("answer video duration differs from the continuous rollout")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "run_dir": str(run_dir),
        "task_success_under_declared_grasp_proxy": evaluator.get("task_success_under_declared_grasp_proxy"),
        "contact_dynamics_admitted": evaluator.get("contact_dynamics_admitted"),
        "qwen_receipts": receipt.get("qwen_receipts", []),
        "public_events_checked": len(events),
        "timeline_phases_checked": len(timeline.get("phases", [])),
        "answer_video_probe": video_probe,
        "errors": errors,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
