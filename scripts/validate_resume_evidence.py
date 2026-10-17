#!/usr/bin/env python3
"""Validate every numeric claim used in the resume-facing project package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "resume_evidence_v1.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(relative: str) -> Path:
    return (PROJECT_ROOT / relative).resolve()


def collect_test_count() -> int:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return sum("::" in line for line in result.stdout.splitlines())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "resume_evidence_v1" / "receipt.json",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_json(config_path)
    expected = config["expected"]
    challenge_dir = resolve(config["challenge_dir"])
    summary = load_json(challenge_dir / "challenge_summary.json")["summary"]
    artifact_audit = load_json(challenge_dir / "validation_receipt.json")
    tool_audit = load_json(challenge_dir / "agent_tool_validation_receipt.json")
    video_validation = load_json(resolve(config["video_validation"]))
    g0b = load_json(resolve(config["g0b_receipt"]))

    with (challenge_dir / "challenge_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    vlm_calls = sum(int(row["vlm_calls"]) for row in rows)

    checks: dict[str, bool] = {
        "method_rollouts": len(rows) == expected["method_rollouts"],
        "vlm_calls": vlm_calls == expected["vlm_calls"],
        "endpoint_errors": expected["endpoint_errors"] == 0 and artifact_audit["status"] == "PASS",
        "public_events": artifact_audit["public_events_checked"] == expected["public_events"],
        "agent_tool_calls": tool_audit["agent_tool_calls_checked"] == expected["agent_tool_calls"],
        "artifact_audit": artifact_audit["status"] == "PASS",
        "tool_contract_audit": tool_audit["status"] == "PASS",
        "video_audit": video_validation["passed"] is True,
        "video_sha256": sha256(resolve(config["video"])) == expected["video"]["sha256"],
        "g0b_boundary": g0b["decision"] == expected["g0b_decision"] and g0b["admitted"] is False,
    }

    full_results: dict[str, list[int]] = {}
    for scene, pair in expected["full_agent_success"].items():
        record = summary[scene]["C2_full"]
        observed = [round(record["mechanism_success"] * record["runs"]), record["runs"]]
        full_results[scene] = observed
        checks[f"full_agent_{scene}"] = observed == pair

    ablation_results: dict[str, list[int]] = {}
    for key, pair in expected["ablations"].items():
        scene, condition = key.split("/", maxsplit=1)
        record = summary[scene][condition]
        observed = [round(record["mechanism_success"] * record["runs"]), record["runs"]]
        ablation_results[key] = observed
        checks[f"ablation_{scene}_{condition}"] = observed == pair

    stream = video_validation["probe"]["streams"][0]
    video_expected = expected["video"]
    checks.update({
        "video_resolution": [stream["width"], stream["height"]] == [video_expected["width"], video_expected["height"]],
        "video_fps": stream["r_frame_rate"] == video_expected["fps"],
        "video_frames": int(stream["nb_frames"]) == video_expected["frames"],
        "video_duration": abs(float(video_validation["probe"]["format"]["duration"]) - video_expected["duration_s"]) < 0.02,
    })

    tests_collected = collect_test_count()
    checks["tests_collected"] = tests_collected == expected["tests_collected"]
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "project": config["project_name"],
        "config": str(config_path),
        "observed": {
            "method_rollouts": len(rows),
            "vlm_calls": vlm_calls,
            "public_events": artifact_audit["public_events_checked"],
            "agent_tool_calls": tool_audit["agent_tool_calls_checked"],
            "tests_collected": tests_collected,
            "full_agent_success": full_results,
            "ablations": ablation_results,
            "video_sha256": sha256(resolve(config["video"])),
            "g0b_decision": g0b["decision"],
        },
        "checks": checks,
        "errors": [name for name, passed in checks.items() if not passed],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
