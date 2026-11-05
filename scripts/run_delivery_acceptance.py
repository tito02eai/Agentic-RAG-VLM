"""Synthetic offline acceptance cases; never reports hardware success."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentic_rag_vlm.pipeline import AgenticPipeline
from agentic_rag_vlm.runtime import AgenticRuntime, TaskSpec, Subgoal
from agentic_rag_vlm.platform.session import BluPeSession
from agentic_rag_vlm.platform.transport import MockPlatformTransport
from agentic_rag_vlm.platform.perception import RegistryMetadataPerception
from agentic_rag_vlm.platform.commands import GraspActionMapper
from agentic_rag_vlm.platform.verifier import SceneChangeVerifier
from scripts.agentic_framework import load_semantic_registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cards = json.loads((ROOT / "knowledge_base/blupe_affordance_cards.json").read_text())["cards"]
    registry = load_semantic_registry(ROOT / "knowledge_base/blupe_semantic_registry.json")
    records = []
    cases = [(f"single_{name}", [name], [], "complete") for name in ("cup", "bottle", "banana", "cube")]
    cases += [("fragile_neighbor", ["cup"], ["vase"], "complete"),
              ("two_objects", ["cup", "cube"], [], "complete"),
              ("missing_protected", ["cup"], ["vase"], "safe_stop"),
              ("rejected_pick", ["cup"], [], "safe_stop"),
              ("precontact_retry", ["cup"], [], "complete"),
              ("wrong_destination", ["cup"], [], "safe_stop")]
    for name, targets, protected, expected in cases:
        scene = {obj: (.30, .1 + i*.2, .8) for i, obj in enumerate(targets)}
        if name == "fragile_neighbor":
            scene["vase"] = (.34, .15, .8)
        destinations = {f"tray_{obj}": (-.25, .1+i*.2, .8) for i, obj in enumerate(targets)}
        transport = MockPlatformTransport(scene, destinations=destinations)
        original = transport.send_command
        attempts = [0]
        def send(command):
            from dataclasses import replace
            if command.command_type == "pick":
                attempts[0] += 1
                if name == "rejected_pick":
                    return {"accepted": False, "reason": "synthetic_rejection"}
                if name == "precontact_retry" and attempts[0] == 1:
                    return {"accepted": False, "execution_started": False,
                            "reason": "motion_plan_failed", "detail": "synthetic IK failed"}
            if name == "wrong_destination" and command.command_type == "place":
                command = replace(command, parameters={"destination_xyz_m": (.15, .1, .8)})
            return original(command)
        transport.send_command = send
        runtime = AgenticRuntime(AgenticPipeline(cards))
        session = BluPeSession(runtime, transport, RegistryMetadataPerception(registry),
            GraspActionMapper(command_prefix=name), SceneChangeVerifier(), destinations=destinations)
        task = TaskSpec(name, "synthetic protocol acceptance", tuple(Subgoal(obj, f"tray_{obj}") for obj in targets), tuple(protected))
        result = session.run(task)
        result.update(evidence_kind="synthetic_mock", hardware_success=None)
        (args.output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        memory_ok = result["outcome"] == "complete" or not runtime.pipeline.memory.to_dict()
        records.append({"case": name, "expected": expected, "actual": result["outcome"],
                        "passed": result["outcome"] == expected and memory_ok})
    summary = {"evidence_kind": "synthetic_mock", "hardware_success": None,
               "passed": all(record["passed"] for record in records), "cases": records}
    (args.output / "acceptance.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
