#!/usr/bin/env python3
"""Run G1-G3 Agentic mechanism pilots from public Guanghua RGB-D observations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agentic_framework import (  # noqa: E402
    build_scene_graph,
    condition_flags,
    detect_scene_change,
    graph_constraint,
    load_cards,
    retrieve_strategy,
    semantic_adapter,
)
from scripts.guanghua_perception import capture_rgbd, estimate_colored_objects, save_rgbd  # noqa: E402
from scripts.run_guanghua_env import (  # noqa: E402
    apply_declared_perturbation,
    configure_scene,
    initialize_position_targets,
)


MJCF = PROJECT_ROOT / "assets" / "guanghua_hand_env" / "mjcf" / "guanghua_hand_env.xml"
PROTOCOL = PROJECT_ROOT / "configs" / "guanghua_experiment_protocol.json"
CARDS = PROJECT_ROOT / "knowledge_base" / "affordance_cards.json"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, events: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")


def truth_xy(model: mujoco.MjModel, data: mujoco.MjData, body: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    return data.xpos[body_id, :2].copy()


def public_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    protocol: dict[str, object],
    directory: Path,
    stem: str,
) -> tuple[dict[str, object], dict[str, object]]:
    rgb, depth = capture_rgbd(model, data, camera="agentview")
    files = save_rgbd(rgb, depth, directory, stem)
    rest_z = {name: float(spec["rest_z_m"]) for name, spec in protocol["objects"].items()}
    estimates = estimate_colored_objects(
        model,
        data,
        rgb,
        depth,
        rest_z_by_object=rest_z,
        camera="agentview",
    )
    objects = semantic_adapter(estimates)
    event = {
        "event": "public_scene_observation",
        "adapter": "deterministic_color_geometry_adapter_v1",
        "camera": "agentview_rgbd",
        "files": files,
        "objects": {name: item.to_dict() for name, item in objects.items()},
        "uses_privileged_simulator_state": False,
    }
    return objects, event


def evaluate_g1(objects, cards, condition: str) -> tuple[list[dict[str, object]], dict[str, object], dict[str, float]]:
    flags = condition_flags(condition)
    decisions = []
    correct = 0
    expected = {"red_cube": "power", "blue_cylinder": "pinch"}
    for target in ("red_cube", "blue_cylinder"):
        result = retrieve_strategy(objects[target], cards, enabled=flags["rag"])
        chosen = result.strategy["synergy"]
        correct += int(chosen == expected[target])
        decisions.append({"event": "haa_rag_decision", **result.to_dict(), "uses_privileged_simulator_state": False})
    accuracy = correct / 2.0
    private = {"role": "private_evaluator_only", "expected_synergies": expected, "correct": correct, "total": 2}
    metrics = {"retrieval_top1_accuracy": accuracy, "constraint_satisfaction": None, "change_detection": None, "replan_valid": None, "memory_preserved": None, "mechanism_success": float(accuracy == 1.0)}
    return decisions, private, metrics


def evaluate_g2(model, data, objects, cards, condition: str) -> tuple[list[dict[str, object]], dict[str, object], dict[str, float]]:
    flags = condition_flags(condition)
    retrieval = retrieve_strategy(objects["red_cube"], cards, enabled=flags["rag"])
    graph = build_scene_graph(objects)
    constraint = graph_constraint(objects["red_cube"], objects, graph, enabled=flags["graph"])
    red_truth = truth_xy(model, data, "cube")
    fragile_truth = truth_xy(model, data, "fragile_proxy")
    distance = float(np.linalg.norm(red_truth - fragile_truth))
    hazard_expected = distance <= 0.10
    offset = np.asarray(constraint["approach_offset_xy_m"], dtype=float)
    away = red_truth - fragile_truth
    away /= max(float(np.linalg.norm(away)), 1e-9)
    satisfied = bool(
        (not hazard_expected)
        or (
            constraint["active"]
            and float(np.dot(offset, away)) >= 0.04
            and float(constraint["approach_height_delta_m"]) >= 0.03
            and float(constraint["force_scale"]) <= 0.8
        )
    )
    events = [
        {"event": "haa_rag_decision", **retrieval.to_dict(), "uses_privileged_simulator_state": False},
        {"event": "scene_graph", "graph": graph, "constraint": constraint, "uses_privileged_simulator_state": False},
    ]
    private = {"role": "private_evaluator_only", "red_fragile_distance_m": distance, "hazard_expected": hazard_expected, "constraint_satisfied": satisfied}
    metrics = {"retrieval_top1_accuracy": float(retrieval.strategy["synergy"] == "power"), "constraint_satisfaction": float(satisfied), "change_detection": None, "replan_valid": None, "memory_preserved": None, "mechanism_success": float(satisfied)}
    return events, private, metrics


def evaluate_g3(before, after, cards, condition: str) -> tuple[list[dict[str, object]], dict[str, object], dict[str, float]]:
    flags = condition_flags(condition)
    completed = ["red_cube"] if flags["memory"] else []
    retrieval = retrieve_strategy(before["blue_cylinder"], cards, enabled=flags["rag"])
    monitor = detect_scene_change(before, after)
    stale_blue = "blue_cylinder" in monitor["stale_targets"]
    replanned_target = list(after["blue_cylinder"].center_xyz_m) if stale_blue and flags["replan"] else list(before["blue_cylinder"].center_xyz_m)
    public_shift = float(np.linalg.norm(np.asarray(after["blue_cylinder"].center_xyz_m[:2]) - np.asarray(before["blue_cylinder"].center_xyz_m[:2])))
    replan_valid = bool(stale_blue and flags["replan"] and np.allclose(replanned_target[:2], after["blue_cylinder"].center_xyz_m[:2]))
    memory_preserved = "red_cube" in completed
    next_targets = [name for name in ("red_cube", "blue_cylinder") if name not in completed]
    events = [
        {"event": "verified_subgoal_fixture", "completed": "red_cube", "source": "public_verifier_contract", "uses_privileged_simulator_state": False},
        {"event": "initial_blue_plan", **retrieval.to_dict(), "target_xyz_m": list(before["blue_cylinder"].center_xyz_m), "uses_privileged_simulator_state": False},
        {"event": "public_monitor", **monitor, "uses_privileged_simulator_state": False},
        {"event": "l3_replan", "enabled": flags["replan"], "target_xyz_m": replanned_target, "completed_memory": completed, "next_targets": next_targets, "uses_privileged_simulator_state": False},
    ]
    private = {"role": "private_evaluator_only", "public_measured_shift_m": public_shift, "change_detected": stale_blue, "replan_valid": replan_valid, "memory_preserved": memory_preserved, "duplicate_red_action": "red_cube" in next_targets}
    mechanism = stale_blue and replan_valid and memory_preserved
    metrics = {"retrieval_top1_accuracy": float(retrieval.strategy["synergy"] == "pinch"), "constraint_satisfaction": None, "change_detection": float(stale_blue), "replan_valid": float(replan_valid), "memory_preserved": float(memory_preserved), "mechanism_success": float(mechanism)}
    return events, private, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output" / "agentic_pilot")
    parser.add_argument("--seeds", type=int, nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cards = load_cards(CARDS)
    seeds = args.seeds or protocol["seeds"]["pilot"]
    comparisons = {scene: protocol["paired_comparisons"][scene] for scene in ("G1", "G2", "G3")}
    write_json(args.output / "frozen_config.json", {"protocol": str(PROTOCOL), "cards": str(CARDS), "scenes": comparisons, "seeds": seeds, "claim_boundary": "mechanism metrics, not physical pick success", "semantic_adapter": "deterministic_color_geometry_adapter_v1"})
    rows = []
    start_all = time.perf_counter()
    for scene_id, conditions in comparisons.items():
        for seed in seeds:
            model = mujoco.MjModel.from_xml_path(str(MJCF))
            data = mujoco.MjData(model)
            initialize_position_targets(model, data)
            scene_spec = configure_scene(model, data, protocol_path=PROTOCOL, scene_id=scene_id, seed=seed)
            observation_dir = args.output / "observations" / f"{scene_id}_seed{seed}"
            before, before_event = public_observation(model, data, protocol, observation_dir, "initial_agentview")
            after = None
            after_event = None
            perturbation = None
            if scene_id == "G3":
                perturbation = apply_declared_perturbation(model, data, protocol_path=PROTOCOL, scene_id=scene_id)
                after, after_event = public_observation(model, data, protocol, observation_dir, "post_change_agentview")
            for condition in conditions:
                run_dir = args.output / "runs" / f"{scene_id}_seed{seed}_{condition}"
                if scene_id == "G1":
                    decisions, private, metrics = evaluate_g1(before, cards, condition)
                elif scene_id == "G2":
                    decisions, private, metrics = evaluate_g2(model, data, before, cards, condition)
                else:
                    assert after is not None
                    decisions, private, metrics = evaluate_g3(before, after, cards, condition)
                public_events = [before_event] + ([after_event] if after_event else []) + decisions
                write_jsonl(run_dir / "public_trace.jsonl", public_events)
                write_json(run_dir / "private_evaluator.json", {**private, "scene": scene_id, "seed": seed, "condition": condition, "mechanism_metrics": metrics, "perturbation_receipt": perturbation if scene_id == "G3" else None})
                write_json(run_dir / "scene_spec.json", {"role": "evaluator_initialization_only", **scene_spec})
                rows.append({"scene": scene_id, "seed": seed, "condition": condition, **metrics})
    fields = list(rows[0])
    with (args.output / "pilot_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for scene_id, conditions in comparisons.items():
        summary[scene_id] = {}
        for condition in conditions:
            subset = [row for row in rows if row["scene"] == scene_id and row["condition"] == condition]
            condition_summary = {}
            for key in fields[3:]:
                values = [float(row[key]) for row in subset if row[key] is not None]
                if values:
                    condition_summary[key] = float(np.mean(values))
            summary[scene_id][condition] = condition_summary
    write_json(args.output / "pilot_summary.json", {"episodes": len(rows), "elapsed_s": time.perf_counter() - start_all, "summary": summary})
    print(json.dumps({"output": str(args.output.resolve()), "episodes": len(rows), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
