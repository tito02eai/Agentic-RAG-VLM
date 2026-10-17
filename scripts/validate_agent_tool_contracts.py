#!/usr/bin/env python3
"""Validate Agent tool routing and public receipts in Challenge artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {
    "G1": {"retrieval_to_dexterous_skill_router_v1"},
    "G2": {"protected_relation_to_motion_constraint_v1"},
    "G3": {"role_aware_change_and_memory_verifier_v1"},
    "G4": {
        "retrieval_to_dexterous_skill_router_v1",
        "protected_relation_to_motion_constraint_v1",
        "role_aware_change_and_memory_verifier_v1",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()
    root = args.artifact_dir.resolve()
    errors: list[str] = []
    checked = 0
    tool_calls = 0
    for run_dir in sorted((root / "runs").glob("*")):
        if not run_dir.is_dir():
            continue
        trace_path = run_dir / "public_trace.jsonl"
        if not trace_path.is_file():
            errors.append(f"{run_dir.name}: missing public trace")
            continue
        planner = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[-1])
        scene = run_dir.name.split("_", 1)[0]
        condition = str(planner.get("condition"))
        receipts = planner.get("agent_tool_receipts", [])
        router_attempts = [
            attempt
            for attempt in planner.get("semantic_attempts", [])
            if attempt.get("phase") == "tool_router"
        ]
        routed_by_model: set[str] = set()
        if scene == "G4":
            latest_by_component: dict[str, dict] = {}
            for attempt in router_attempts:
                latest_by_component[str(attempt.get("component"))] = attempt
            selected_attempts = latest_by_component.values()
        else:
            selected_attempts = router_attempts[-1:]
        for attempt in selected_attempts:
            parsed = attempt.get("result", {}).get("parsed") or {}
            routed_by_model.update(str(tool) for tool in parsed.get("requested_tools", []))
        actual = {
            str(receipt.get("tool"))
            for receipt in receipts
            if receipt.get("status") == "executed"
        }
        expected = EXPECTED[scene] if condition == "C2_full" else set()
        if routed_by_model != expected:
            errors.append(
                f"{run_dir.name}: raw VLM router selection {sorted(routed_by_model)} != {sorted(expected)}"
            )
        executed_request = set((planner.get("executed_decision") or {}).get("requested_tools", []))
        if executed_request != expected:
            errors.append(
                f"{run_dir.name}: executed decision request {sorted(executed_request)} != {sorted(expected)}"
            )
        if actual != expected:
            errors.append(
                f"{run_dir.name}: tool routing {sorted(actual)} != {sorted(expected)}"
            )
        for receipt in receipts:
            if receipt.get("status") != "executed":
                errors.append(f"{run_dir.name}: requested tool was rejected: {receipt.get('tool')} ({receipt.get('reason')})")
            if receipt.get("routing_source") != "vlm_requested_tools":
                errors.append(f"{run_dir.name}: tool receipt was not selected by the VLM router")
            if receipt.get("uses_privileged_simulator_state") is not False:
                errors.append(f"{run_dir.name}: tool receipt lacks non-privileged declaration")
            if not isinstance(receipt.get("inputs"), dict):
                errors.append(f"{run_dir.name}: tool receipt has no inputs")
            if not isinstance(receipt.get("executed_fields"), dict):
                errors.append(f"{run_dir.name}: tool receipt has no executed fields")
        if receipts and planner.get("executed_decision") is None:
            errors.append(f"{run_dir.name}: tools ran without an executed decision")
        checked += 1
        tool_calls += sum(receipt.get("status") == "executed" for receipt in receipts)
    receipt = {
        "status": "PASS" if not errors else "FAIL",
        "artifact_dir": str(root),
        "runs_checked": checked,
        "agent_tool_calls_checked": tool_calls,
        "errors": errors,
    }
    (root / "agent_tool_validation_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
