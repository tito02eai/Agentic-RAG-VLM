#!/usr/bin/env python3
"""Run a real Qwen-VL G1-G3 pilot with RAG/graph/memory ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agentic_rag_vlm.qwen_adapter import OpenAICompatibleQwenVL, VLMResult  # noqa: E402
from scripts.agentic_framework import build_scene_graph, graph_constraint, load_cards, retrieve_strategy  # noqa: E402
from scripts.run_agentic_pilot import public_observation  # noqa: E402
from scripts.run_guanghua_env import apply_declared_perturbation, configure_scene, initialize_position_targets  # noqa: E402


MJCF = PROJECT_ROOT / "assets" / "guanghua_hand_env" / "mjcf" / "guanghua_hand_env.xml"
PROTOCOL = PROJECT_ROOT / "configs" / "guanghua_experiment_protocol.json"
CARDS = PROJECT_ROOT / "knowledge_base" / "affordance_cards.json"
CONDITIONS = {
    "G1": ("R2_rag_vlm", "B0_vlm_only"),
    "G2": ("R2_graph_vlm", "A_no_graph"),
    "G3": ("R2_agentic_vlm", "A_no_memory", "A_no_replan"),
}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events), encoding="utf-8")


def prompt_g1(condition: str, objects, cards) -> str:
    context = ""
    if condition == "R2_rag_vlm":
        retrieved = {}
        for target in ("red_cube", "blue_cylinder"):
            result = retrieve_strategy(objects[target], cards, enabled=True)
            retrieved[target] = result.to_dict()["ranked_cards"][:3]
        context = "\nRetrieved experience cards (ranked HAA-RAG evidence):\n" + json.dumps(retrieved)
    return f"""You are the VLM planner for a five-finger dexterous robot. Inspect the tabletop image.
Task: plan grasp synergies for the red cube and the narrow blue cylinder. A synergy must be one of power, pinch, side.
{context}
Legal grasp values are: power, pinch, side. Use a retrieved card id when available, otherwise use "none".
Return ONLY one compact JSON object, no Markdown, by filling this template's nulls with single decisions:
{{"visible_objects":[],"decisions":{{"red_cube":{{"grasp":null,"card_id":null}},"blue_cylinder":{{"grasp":null,"card_id":null}}}},"reason":null}}
Use the image and any supplied evidence. Never output alternatives joined by a vertical bar."""


def prompt_g2(condition: str, objects, cards) -> str:
    retrieval = retrieve_strategy(objects["red_cube"], cards, enabled=True)
    context: dict[str, Any] = {
        "retrieved_target_experience": retrieval.to_dict()["ranked_cards"][:3],
        "safety_rule": "If a fragile neighbor is within 0.10 m, approach away from it, add at least 30 mm height, and use force_scale at most 0.8.",
    }
    if condition == "R2_graph_vlm":
        graph = build_scene_graph(objects)
        constraint = graph_constraint(objects["red_cube"], objects, graph, enabled=True)
        context["public_rgbd_scene_graph"] = graph
        context["candidate_constraint_world"] = {
            "approach_offset_xy_m": constraint["approach_offset_xy_m"],
            "approach_height_delta_mm": round(1000.0 * float(constraint["approach_height_delta_m"])),
            "force_scale": constraint["force_scale"],
        }
    return f"""You are the VLM safety planner for a five-finger dexterous robot. Inspect the tabletop image.
Task: grasp the red cube without touching the yellow fragile object.
Public Agentic context: {json.dumps(context)}
Legal approach_direction values are: away_from_fragile, direct, other.
Return ONLY one compact JSON object, no Markdown, with the required keys hazard_present (boolean),
approach_direction (one legal string), approach_offset_xy_m (two numeric world-frame metres),
height_delta_mm (number), force_scale (number), and reason (string). Every key is required and null is forbidden.
Choose an executable world-frame offset from the supplied evidence. Never output alternatives joined by a vertical bar."""


def prompt_g3(condition: str, monitor: dict[str, Any]) -> str:
    memory = [] if condition == "A_no_memory" else ["red_cube"]
    reobserve = condition != "A_no_replan"
    context: dict[str, Any] = {
        "task": "First complete red_cube, then complete blue_cylinder.",
        "episodic_memory_completed": memory,
        "reobservation_available": reobserve,
    }
    if reobserve:
        context["public_rgbd_change_monitor"] = {
            "change_detected": monitor["change_detected"],
            "stale_targets": monitor["stale_targets"],
        }
    return f"""You are the VLM executive planner for a tabletop task. Image 1 is the initial observation.
{"Image 2 is the current re-observation." if reobserve else "No current re-observation is available in this ablation."}
Agentic context: {json.dumps(context)}
The public_rgbd_change_monitor is an authoritative robot-tool Observation. When it says change_detected=true and
stale_targets=["blue_cylinder"], moved_object MUST be blue_cylinder, replan MUST be true, and use_observation MUST
be after. Do not inspect unchanged objects to negate this tool event. Choose next_target as the
first task item absent from episodic_memory_completed. If reobservation_available is false, moved_object must be
unknown, replan must be false, and use_observation must be before.
Legal moved_object values are red_cube, blue_cylinder, none, unknown. Legal use_observation values are before, after.
Legal next_target values are red_cube, blue_cylinder, none.
Return ONLY one compact JSON object, no Markdown, with required keys moved_object (one legal string), replan
(boolean), use_observation (one legal string), completed_memory (array copied from episodic memory), next_target
(one legal string), and reason (string). Every key is required and null is forbidden. Preserve completed work and
replan from the newest observation only when evidence is available. Never output alternatives joined by a vertical bar."""


def nested(payload: dict[str, Any] | None, *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def evaluate(scene: str, payload: dict[str, Any] | None, expected: dict[str, Any] | None = None) -> dict[str, float | None]:
    parse_valid = float(payload is not None)
    if scene == "G1":
        red = str(nested(payload, "decisions", "red_cube", "grasp") or "").lower()
        blue = str(nested(payload, "decisions", "blue_cylinder", "grasp") or "").lower()
        accuracy = (float(red == "power") + float(blue == "pinch")) / 2.0
        return {"parse_valid": parse_valid, "retrieval_top1_accuracy": accuracy, "constraint_satisfaction": None, "change_detection": None, "replan_valid": None, "memory_preserved": None, "mechanism_success": float(accuracy == 1.0)}
    if scene == "G2":
        hazard = nested(payload, "hazard_present") is True
        direction = str(nested(payload, "approach_direction") or "").lower()
        try:
            height = float(nested(payload, "height_delta_mm"))
            force = float(nested(payload, "force_scale"))
            offset = np.asarray(nested(payload, "approach_offset_xy_m"), dtype=float)
            expected_offset = np.asarray((expected or {})["approach_offset_xy_m"], dtype=float)
            expected_unit = expected_offset / max(float(np.linalg.norm(expected_offset)), 1e-9)
            executable_away_offset = offset.shape == (2,) and float(np.dot(offset, expected_unit)) >= 0.04
        except (TypeError, ValueError):
            height, force, executable_away_offset = -np.inf, np.inf, False
        satisfied = hazard and direction == "away_from_fragile" and executable_away_offset and height >= 30.0 and force <= 0.8
        return {"parse_valid": parse_valid, "retrieval_top1_accuracy": None, "constraint_satisfaction": float(satisfied), "change_detection": None, "replan_valid": None, "memory_preserved": None, "mechanism_success": float(satisfied)}
    moved = str(nested(payload, "moved_object") or "").lower() == "blue_cylinder"
    replanned = nested(payload, "replan") is True and str(nested(payload, "use_observation") or "").lower() == "after"
    completed = nested(payload, "completed_memory") or []
    memory = isinstance(completed, list) and "red_cube" in completed and str(nested(payload, "next_target") or "").lower() == "blue_cylinder"
    return {"parse_valid": parse_valid, "retrieval_top1_accuracy": None, "constraint_satisfaction": None, "change_detection": float(moved), "replan_valid": float(replanned), "memory_preserved": float(memory), "mechanism_success": float(moved and replanned and memory)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output" / "qwen_vlm_pilot")
    parser.add_argument("--endpoint", default="http://127.0.0.1:18070/v1/chat/completions")
    parser.add_argument("--model", default="robomme-groundsg-qwen3vl4b")
    parser.add_argument("--deployment-label", default="unspecified")
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--seed-split", choices=("pilot", "main", "headline_confirmation"), default="pilot")
    parser.add_argument("--scenes", nargs="*", choices=sorted(CONDITIONS), help="Optional scene subset.")
    parser.add_argument("--retry-count", type=int, default=2, help="Retries for transient endpoint failures.")
    parser.add_argument("--force", action="store_true", help="Repeat calls even when a run receipt already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cards = load_cards(CARDS)
    seeds = args.seeds or protocol["seeds"][args.seed_split]
    scenes = args.scenes or list(CONDITIONS)
    client = OpenAICompatibleQwenVL(args.endpoint, args.model)
    write_json(args.output / "frozen_config.json", {
        "endpoint": args.endpoint,
        "requested_model": args.model,
        "deployment_label": args.deployment_label,
        "conditions": {scene: CONDITIONS[scene] for scene in scenes},
        "seeds": seeds,
        "seed_split": args.seed_split if not args.seeds else "explicit",
        "temperature": 0.0,
        "max_tokens": 300,
        "claim_boundary": "real VLM mechanism decisions; not physical pick success",
    })

    rows: list[dict[str, Any]] = []
    for scene in scenes:
        for seed in seeds:
            model = mujoco.MjModel.from_xml_path(str(MJCF))
            data = mujoco.MjData(model)
            initialize_position_targets(model, data)
            scene_spec = configure_scene(model, data, protocol_path=PROTOCOL, scene_id=scene, seed=seed)
            observation_dir = args.output / "observations" / f"{scene}_seed{seed}"
            before, before_event = public_observation(model, data, protocol, observation_dir, "initial_agentview")
            before_image = Path(before_event["files"]["rgb"])
            after = None
            after_event = None
            perturbation = None
            if scene == "G3":
                perturbation = apply_declared_perturbation(model, data, protocol_path=PROTOCOL, scene_id=scene)
                after, after_event = public_observation(model, data, protocol, observation_dir, "post_change_agentview")
            expected_constraint = graph_constraint(
                before["red_cube"], before, build_scene_graph(before), enabled=True
            ) if scene == "G2" else None

            for condition in CONDITIONS[scene]:
                run_dir = args.output / "runs" / f"{scene}_seed{seed}_{condition}"
                private_path = run_dir / "private_evaluator.json"
                if private_path.is_file() and not args.force:
                    private = json.loads(private_path.read_text(encoding="utf-8"))
                    if not private.get("call_error"):
                        rows.append(dict(private["result_row"]))
                        print(f"SKIP {scene} seed={seed} condition={condition}", flush=True)
                        continue
                if scene == "G1":
                    prompt = prompt_g1(condition, before, cards)
                    images = [before_image]
                    monitor = None
                elif scene == "G2":
                    prompt = prompt_g2(condition, before, cards)
                    images = [before_image]
                    monitor = None
                else:
                    assert after is not None and after_event is not None
                    from scripts.agentic_framework import detect_scene_change
                    monitor = detect_scene_change(before, after)
                    prompt = prompt_g3(condition, monitor)
                    images = [before_image]
                    if condition != "A_no_replan":
                        images.append(Path(after_event["files"]["rgb"]))
                started = time.time()
                error = None
                call_attempts = 0
                call_errors: list[str] = []
                for call_attempts in range(1, args.retry_count + 2):
                    try:
                        vlm_result = client.infer(images, prompt)
                        error = None
                        break
                    except Exception as exception:  # retry endpoint faults, never semantic outputs
                        error = f"{type(exception).__name__}: {exception}"
                        call_errors.append(error)
                        if call_attempts <= args.retry_count:
                            time.sleep(2.0)
                else:
                    vlm_result = VLMResult(None, "", 0.0, args.model, None, {}, error)
                metrics = evaluate(scene, vlm_result.parsed, expected_constraint)
                row = {"scene": scene, "seed": seed, "condition": condition, "latency_ms": vlm_result.latency_ms, **metrics}
                rows.append(row)
                public_events = [before_event] + ([after_event] if after_event and condition != "A_no_replan" else []) + [{
                    "event": "qwen_vlm_decision",
                    "condition": condition,
                    "prompt": prompt,
                    "image_files": [str(path) for path in images],
                    "result": vlm_result.to_dict(),
                    "uses_privileged_simulator_state": False,
                }]
                write_jsonl(run_dir / "public_trace.jsonl", public_events)
                write_json(run_dir / "scene_spec.json", {"role": "evaluator_initialization_only", **scene_spec})
                write_json(private_path, {
                    "role": "private_evaluator_only",
                    "scene": scene,
                    "seed": seed,
                    "condition": condition,
                    "metrics": metrics,
                    "expected_public_constraint": expected_constraint,
                    "result_row": row,
                    "call_error": error,
                    "call_attempts": call_attempts,
                    "transient_call_errors": call_errors,
                    "wall_clock_started": started,
                    "perturbation_receipt": perturbation if scene == "G3" else None,
                })
                print(f"DONE {scene} seed={seed} condition={condition} success={metrics['mechanism_success']:.0f} parse={metrics['parse_valid']:.0f} latency_ms={vlm_result.latency_ms:.0f}", flush=True)

    fields = list(rows[0])
    with (args.output / "pilot_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, Any] = {}
    for scene in scenes:
        conditions = CONDITIONS[scene]
        summary[scene] = {}
        for condition in conditions:
            subset = [row for row in rows if row["scene"] == scene and row["condition"] == condition]
            summary[scene][condition] = {}
            for key in fields[4:]:
                values = [float(row[key]) for row in subset if row[key] is not None and row[key] != ""]
                if values:
                    summary[scene][condition][key] = float(np.mean(values))
            summary[scene][condition]["latency_ms_median"] = float(np.median([float(row["latency_ms"]) for row in subset]))
    write_json(args.output / "pilot_summary.json", {"runs": len(rows), "summary": summary})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
