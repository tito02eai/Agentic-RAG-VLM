#!/usr/bin/env python3
"""Audit Agentic pilot/challenge artifacts for completeness and isolation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PRIVATE_ONLY_KEYS = {
    "role",
    "expected_synergies",
    "correct",
    "red_fragile_distance_m",
    "hazard_expected",
    "constraint_satisfied",
    "public_measured_shift_m",
    "replan_valid",
    "memory_preserved",
    "duplicate_red_action",
    "mechanism_metrics",
    "metrics",
    "expected_public_constraint",
    "result_row",
    "perturbation_receipt",
}


def walk_keys(value: object) -> set[str]:
    """Collect mapping keys recursively from a JSON-compatible value."""
    if isinstance(value, dict):
        keys = set(value)
        for child in value.values():
            keys.update(walk_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(walk_keys(child))
        return keys
    return set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.artifact_dir.resolve()
    config = json.loads((root / "frozen_config.json").read_text(encoding="utf-8"))
    result_path = root / ("challenge_results.csv" if (root / "challenge_results.csv").is_file() else "pilot_results.csv")
    with result_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    comparisons = config.get("scenes") or config.get("conditions")
    if not isinstance(comparisons, dict):
        raise ValueError("frozen_config must define scenes or conditions")
    expected_runs = sum(len(conditions) for conditions in comparisons.values()) * len(config["seeds"])
    errors: list[str] = []
    checked_events = 0
    run_dirs = sorted(path for path in (root / "runs").glob("*") if path.is_dir())
    if len(rows) != expected_runs:
        errors.append(f"CSV has {len(rows)} rows; expected {expected_runs}")
    if len(run_dirs) != expected_runs:
        errors.append(f"runs/ has {len(run_dirs)} directories; expected {expected_runs}")

    for run_dir in run_dirs:
        for filename in ("public_trace.jsonl", "private_evaluator.json", "scene_spec.json"):
            if not (run_dir / filename).is_file():
                errors.append(f"{run_dir.name}: missing {filename}")
        trace_path = run_dir / "public_trace.jsonl"
        private_path = run_dir / "private_evaluator.json"
        if private_path.is_file():
            private = json.loads(private_path.read_text(encoding="utf-8"))
            if private.get("call_error"):
                errors.append(f"{run_dir.name}: endpoint call failed: {private['call_error']}")
        if not trace_path.is_file():
            continue
        for line_number, line in enumerate(trace_path.read_text(encoding="utf-8").splitlines(), 1):
            event = json.loads(line)
            checked_events += 1
            leaked = sorted(walk_keys(event) & PRIVATE_ONLY_KEYS)
            if leaked:
                errors.append(f"{run_dir.name}: public line {line_number} contains private keys {leaked}")
            if event.get("uses_privileged_simulator_state") is not False:
                errors.append(f"{run_dir.name}: public line {line_number} lacks explicit non-privileged receipt")

    receipt = {
        "status": "PASS" if not errors else "FAIL",
        "artifact_dir": str(root),
        "expected_runs": expected_runs,
        "csv_rows": len(rows),
        "run_directories": len(run_dirs),
        "public_events_checked": checked_events,
        "private_only_keys_checked": sorted(PRIVATE_ONLY_KEYS),
        "errors": errors,
    }
    output = root / "validation_receipt.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
