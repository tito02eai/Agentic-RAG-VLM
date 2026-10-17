#!/usr/bin/env python3
"""Run the leakage-resistant Agentic RAG-VLM challenge with model-selected tools."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from math import cos, radians, sin
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agentic_rag_vlm.qwen_adapter import OpenAICompatibleQwenVL, VLMResult  # noqa: E402
from scripts.agentic_framework import (  # noqa: E402
    build_scene_graph,
    detect_scene_change,
    load_cards,
    retrieve_strategy,
    semantic_adapter,
)
from scripts.guanghua_perception import capture_rgbd, estimate_colored_objects, save_rgbd  # noqa: E402
from scripts.run_guanghua_env import initialize_position_targets  # noqa: E402


MJCF = PROJECT_ROOT / "assets" / "guanghua_hand_env" / "mjcf" / "guanghua_hand_env.xml"
CONFIG = PROJECT_ROOT / "configs" / "guanghua_challenge_v4_agent_routing.json"
CARDS = PROJECT_ROOT / "knowledge_base" / "affordance_cards.json"
TABLE_SURFACE_Z = 0.801
OBJECT_BODIES = {
    "red_cube": "cube",
    "blue_cylinder": "blue_cylinder",
    "fragile_proxy": "fragile_proxy",
}
METRIC_KEYS = (
    "parse_valid",
    "grasp_accuracy",
    "retrieval_card_accuracy",
    "safety_decision_correct",
    "offset_executable",
    "change_detection_correct",
    "replan_decision_correct",
    "memory_preserved",
    "false_replan",
    "tool_selection_correct",
    "mechanism_score",
    "mechanism_success",
)

RETRIEVAL_TOOL = "retrieval_to_dexterous_skill_router_v1"
SAFETY_TOOL = "protected_relation_to_motion_constraint_v1"
MEMORY_TOOL = "role_aware_change_and_memory_verifier_v1"
LEGAL_AGENT_TOOLS = (RETRIEVAL_TOOL, SAFETY_TOOL, MEMORY_TOOL)
EVALUATOR_REQUIRED_TOOLS = {
    "G1": {RETRIEVAL_TOOL},
    "G2": {SAFETY_TOOL},
    "G3": {MEMORY_TOOL},
    "G4": {RETRIEVAL_TOOL, SAFETY_TOOL, MEMORY_TOOL},
}
TOOL_OUTPUT_BINDINGS = {
    "G1": ["decisions.red_cube.grasp/card_id", "decisions.blue_cylinder.grasp/card_id"],
    "G2": ["hazard_present", "approach_direction", "approach_offset_xy_m", "height_delta_mm", "force_scale"],
    "G3": ["changed_objects", "stale_pending_targets", "replan", "completed_memory", "next_target"],
    "G4": [
        "decisions.*.grasp/card_id", "hazard_present", "approach_direction", "approach_offset_xy_m",
        "height_delta_mm", "force_scale", "changed_objects", "stale_pending_targets", "replan",
        "completed_memory", "next_target",
    ],
}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events), encoding="utf-8")


def set_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: Mapping[str, Any],
    name: str,
    xy: np.ndarray | tuple[float, float] | list[float],
    *,
    z: float | None = None,
    yaw: float = 0.0,
) -> None:
    joint_name = config["object_free_joints"][name]
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    address = int(model.jnt_qposadr[joint_id])
    data.qpos[address : address + 3] = [float(xy[0]), float(xy[1]), float(z or config["object_rest_z_m"][name])]
    data.qpos[address + 3 : address + 7] = [cos(yaw / 2.0), 0.0, 0.0, sin(yaw / 2.0)]


def configure_affordance_variants(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: Mapping[str, Any],
    seed: int,
) -> dict[str, dict[str, Any]]:
    red_variants = config["affordance_variants"]["red_cube"]
    blue_variants = config["affordance_variants"]["blue_cylinder"]
    selected = {
        "red_cube": dict(red_variants[seed % len(red_variants)]),
        "blue_cylinder": dict(blue_variants[(seed // len(red_variants)) % len(blue_variants)]),
    }

    red_size = np.asarray(selected["red_cube"]["visual_size_m"], dtype=float)
    for geom_name in ("cube_visual", "cube_col"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        model.geom_size[geom_id, :3] = red_size

    blue_visual = np.asarray(selected["blue_cylinder"]["visual_size_m"], dtype=float)
    blue_collision = np.asarray(selected["blue_cylinder"]["collision_half_size_m"], dtype=float)
    visual_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "blue_cylinder_visual")
    collision_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "blue_cylinder_col")
    model.geom_size[visual_id, :2] = blue_visual
    model.geom_size[collision_id, :3] = blue_collision

    selected["red_cube"]["rest_z_m"] = TABLE_SURFACE_Z + float(red_size[2])
    selected["blue_cylinder"]["rest_z_m"] = TABLE_SURFACE_Z + float(blue_visual[1])
    return selected


def public_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rest_z: Mapping[str, float],
    directory: Path,
    stem: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rgb, depth = capture_rgbd(model, data, camera="agentview")
    files = save_rgbd(rgb, depth, directory, stem)
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
        "adapter": "deterministic_color_geometry_adapter_v3_height_envelope",
        "camera": "agentview_rgbd",
        "files": files,
        "objects": {name: item.to_dict() for name, item in objects.items()},
        "uses_privileged_simulator_state": False,
    }
    return objects, event


def realize_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: Mapping[str, Any],
    scene: str,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 1000 * int(scene[1:]))
    variants = configure_affordance_variants(model, data, config, seed)
    red_z = float(variants["red_cube"]["rest_z_m"])
    blue_z = float(variants["blue_cylinder"]["rest_z_m"])
    fragile_z = float(config["object_rest_z_m"]["fragile_proxy"])
    jitter = 0.008 if scene != "G4" else float(config["combined_randomization"]["xy_jitter_m"])
    red = np.asarray([-0.18, -0.21]) + rng.uniform(-jitter, jitter, 2)
    blue = np.asarray([-0.18, -0.03]) + rng.uniform(-jitter, jitter, 2)
    fragile = np.asarray([0.05, 0.10])
    distance = None
    bearing = None

    if scene == "G3":
        red = np.asarray([-0.03, -0.23])
    if scene in {"G2", "G4"}:
        distances = config["fragile_distances_m"]
        bearings = config["fragile_bearings_deg"]
        distance = float(distances[seed % len(distances)])
        bearing = float(bearings[(seed // len(distances)) % len(bearings)])
        direction = np.asarray([cos(radians(bearing)), sin(radians(bearing))])
        fragile = red + distance * direction

    yaw_limit = radians(10.0 if scene != "G4" else float(config["combined_randomization"]["yaw_jitter_deg"]))
    set_pose(model, data, config, "red_cube", red, z=red_z, yaw=float(rng.uniform(-yaw_limit, yaw_limit)))
    set_pose(model, data, config, "blue_cylinder", blue, z=blue_z, yaw=float(rng.uniform(-yaw_limit, yaw_limit)))
    set_pose(model, data, config, "fragile_proxy", fragile, z=fragile_z)

    physical = {"friction_scale": 1.0, "mass_scale": 1.0}
    if scene == "G4":
        low, high = config["combined_randomization"]["friction_scale_range"]
        physical["friction_scale"] = float(rng.uniform(low, high))
        low, high = config["combined_randomization"]["mass_scale_range"]
        physical["mass_scale"] = float(rng.uniform(low, high))
        for geom_name in ("cube_col", "blue_cylinder_col", "fragile_proxy_col"):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            model.geom_friction[geom_id] *= physical["friction_scale"]
        for body_name in OBJECT_BODIES.values():
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            model.body_mass[body_id] *= physical["mass_scale"]
            model.body_inertia[body_id] *= physical["mass_scale"]
    mujoco.mj_forward(model, data)
    return {
        "scene": scene,
        "seed": seed,
        "variants": variants,
        "initial_xy_m": {"red_cube": red.tolist(), "blue_cylinder": blue.tolist(), "fragile_proxy": fragile.tolist()},
        "fragile_distance_m": distance,
        "fragile_bearing_deg": bearing,
        "physical_randomization": physical,
    }


def apply_change_event(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: Mapping[str, Any],
    scene_spec: Mapping[str, Any],
    scene: str,
    seed: int,
) -> dict[str, Any]:
    events = config["change_events"]
    event = str(events[seed % len(events)])
    rng = np.random.default_rng(seed + 9000)
    if scene == "G4":
        red_z = float(scene_spec["variants"]["red_cube"]["rest_z_m"])
        set_pose(model, data, config, "red_cube", [-0.03, -0.23], z=red_z)
    moved_object = None
    delta = np.zeros(2)
    if event == "target_move":
        moved_object = "blue_cylinder"
    elif event == "irrelevant_fragile_move":
        moved_object = "fragile_proxy"
    if moved_object is not None:
        magnitudes = config["target_move_magnitudes_m"]
        magnitude = float(magnitudes[(seed // len(events)) % len(magnitudes)])
        angle = float(rng.uniform(-np.pi, np.pi))
        delta = magnitude * np.asarray([np.cos(angle), np.sin(angle)])
        body_xy = np.asarray(scene_spec["initial_xy_m"][moved_object], dtype=float) + delta
        z = config["object_rest_z_m"][moved_object]
        if moved_object == "blue_cylinder":
            z = scene_spec["variants"]["blue_cylinder"]["rest_z_m"]
        set_pose(model, data, config, moved_object, body_xy, z=float(z))
    mujoco.mj_forward(model, data)
    return {"event": event, "moved_object": moved_object, "delta_xy_m": delta.round(6).tolist()}


def rag_context(objects: Mapping[str, Any], cards: list[dict[str, Any]]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for target in ("red_cube", "blue_cylinder"):
        result = retrieve_strategy(objects[target], cards, enabled=True)
        context[target] = result.to_dict()["ranked_cards"][:3]
    return context


def graph_context(objects: Mapping[str, Any], adjacency_m: float) -> dict[str, Any]:
    """Build a public graph plus measurements, without proposing an action."""
    graph = build_scene_graph(objects, adjacency_m=adjacency_m)
    red = np.asarray(objects["red_cube"].center_xyz_m[:2], dtype=float)
    fragile = np.asarray(objects["fragile_proxy"].center_xyz_m[:2], dtype=float)
    target_minus_fragile = red - fragile
    graph["target_fragile_measurement"] = {
        "target": "red_cube",
        "fragile": "fragile_proxy",
        "distance_m": float(np.linalg.norm(target_minus_fragile)),
        "target_minus_fragile_xy_m": target_minus_fragile.round(6).tolist(),
        "within_safety_radius": bool(float(np.linalg.norm(target_minus_fragile)) <= adjacency_m),
    }
    return graph


def monitor_context(before: Mapping[str, Any], after: Mapping[str, Any], threshold: float) -> dict[str, Any]:
    raw = detect_scene_change(before, after, threshold_m=threshold)
    return {
        "measurement_threshold_m": threshold,
        "per_object_displacement": raw["displacements"],
        "note": "Treat completed-object motion as expected progress; replan only when pending work became stale.",
    }


def build_prompt(
    scene: str,
    condition: str,
    config: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any] | None,
    cards: list[dict[str, Any]],
    graph: Mapping[str, Any],
    monitor: Mapping[str, Any] | None,
    seed: int,
) -> str:
    task = config["instruction_paraphrases"][seed % len(config["instruction_paraphrases"])]
    context: dict[str, Any] = {"task": task}
    if scene in {"G1", "G4"} and condition == "C2_full":
        context["retrieved_affordance_experience_ranked_best_first"] = rag_context(before, cards)
    if scene in {"G2", "G4"} and condition == "C2_full":
        context["public_rgbd_scene_graph"] = graph
    if scene in {"G3", "G4"}:
        memory = [] if condition == "A_no_memory" else ["red_cube"]
        pending_targets = [target for target in ("red_cube", "blue_cylinder") if target not in memory]
        context["episodic_memory_completed"] = memory
        context["pending_targets_from_task_and_memory"] = pending_targets
        context["replanning_enabled"] = condition != "A_no_replan"
        if monitor is not None:
            displacement = monitor["per_object_displacement"]
            context["public_change_measurements"] = {
                "measurement_threshold_m": monitor["measurement_threshold_m"],
                "pending_target_displacements": {name: displacement[name] for name in pending_targets if name in displacement},
                "other_object_displacements": {name: item for name, item in displacement.items() if name not in pending_targets},
            }
    context_text = json.dumps(context, ensure_ascii=False)

    if scene == "G1":
        return f"""You are a grasp planner for a five-finger robot. Inspect the RGB image and plan ONLY the two task targets red_cube and blue_cylinder; do not plan grasps for zones, the protected yellow object, or other colored geometry. Object shape can vary between episodes, so do not infer the grasp from color or name alone. Retrieved experience, when present, is ranked best-first from public visual geometry; use the top target-specific card unless the image visibly contradicts the retrieved geometry.
Agent context: {context_text}
Legal synergies are power, pinch, and side. Return ONLY one compact JSON object, no Markdown, by filling every null with one decision:
{{"decisions":{{"red_cube":{{"grasp":null,"card_id":null}},"blue_cylinder":{{"grasp":null,"card_id":null}}}},"reason":null}}"""
    if scene == "G2":
        return f"""You are a safety planner for a five-finger robot. Inspect the image and plan the red-object approach while protecting the yellow fragile object. A fragile neighbor at distance <=0.105 m requires an XY offset of at least 0.03 m away from it, at least 30 mm extra height, and force_scale <=0.8. Otherwise use a direct approach. When a public scene graph is supplied, use its measured within_safety_radius relation; it is perception evidence, not a proposed action. World XY centers use the same numeric frame as the output; compute the away offset from target_minus_fragile yourself.
Agent context: {context_text}
Return only JSON with no nulls:
{{"hazard_present":false,"approach_direction":"away_from_fragile|direct","approach_offset_xy_m":[0.0,0.0],"height_delta_mm":0,"force_scale":1.0,"reason":"short"}}"""
    if scene == "G4":
        return f"""You are the integrated grasp, safety, and executive planner for a randomized two-step tabletop task. Image 1 is the initial state and Image 2 is the state after the first placement attempt. Choose shape-appropriate grasps, protect any nearby yellow fragile object, and decide whether unfinished work became stale. Retrieved experience, when present, is ranked best-first from public visual geometry; use the top target-specific card unless the image visibly contradicts the retrieved geometry. A fragile neighbor at distance <=0.105 m requires an XY offset of at least 0.03 m away from it, at least 30 mm extra height, and force_scale <=0.8; otherwise use a direct approach. World XY centers, when supplied, use the output frame. Apply the change threshold mechanically to every displacement. stale_pending_targets is the intersection of changed_objects and unfinished task targets. Replan exactly when that intersection is nonempty and replanning is enabled. Copy episodic_memory_completed exactly into completed_memory, and choose the first task item absent from that memory. Do not mark an object complete merely because it moved.
Agent context: {context_text}
Legal grasps are power, pinch, and side; legal next_target values are red_cube, blue_cylinder, and none. Return ONLY one compact JSON object, no Markdown, by filling every null:
{{"decisions":{{"red_cube":{{"grasp":null,"card_id":null}},"blue_cylinder":{{"grasp":null,"card_id":null}}}},"hazard_present":null,"approach_direction":null,"approach_offset_xy_m":null,"height_delta_mm":null,"force_scale":null,"changed_objects":null,"stale_pending_targets":null,"replan":null,"completed_memory":null,"next_target":null,"reason":null}}"""
    return f"""You are the executive planner for a two-step tabletop task. Image 1 is the initial state and Image 2 is the observation after the first placement attempt. Use visual evidence, public displacement measurements, task order, memory, and the stated capability. Apply the change threshold mechanically to every displacement. stale_pending_targets is the intersection of changed_objects and unfinished task targets. Replan exactly when that intersection is nonempty and replanning is enabled. Copy episodic_memory_completed exactly into completed_memory, and choose the first task item absent from that memory. Do not mark an object complete merely because it moved.
Agent context: {context_text}
Legal next_target values are red_cube, blue_cylinder, and none. Return ONLY one compact JSON object, no Markdown, by filling every null:
{{"changed_objects":null,"stale_pending_targets":null,"replan":null,"completed_memory":null,"next_target":null,"reason":null}}"""


def build_tool_selection_prompt(scene: str, config: Mapping[str, Any], seed: int) -> str:
    """Describe the public tool API and let the VLM choose; no evaluator truth is included."""
    task = config["instruction_paraphrases"][seed % len(config["instruction_paraphrases"])]
    registry = {
        RETRIEVAL_TOOL: {
            "binds": ["decisions.*.grasp", "decisions.*.card_id"],
            "purpose": "rank public object descriptors against HAA-RAG affordance cards and route the selected card to a hand synergy",
        },
        SAFETY_TOOL: {
            "binds": ["hazard_present", "approach_direction", "approach_offset_xy_m", "height_delta_mm", "force_scale"],
            "purpose": "convert a public protected-object relation into an executable offset, height and force constraint",
        },
        MEMORY_TOOL: {
            "binds": ["changed_objects", "stale_pending_targets", "replan", "completed_memory", "next_target"],
            "purpose": "attribute public scene changes to pending targets and verify episodic task memory before replanning",
        },
    }
    contract = {
        "task": task,
        "required_planner_output_fields": TOOL_OUTPUT_BINDINGS[scene],
        "available_tools": registry,
        "policy": (
            "Choose every tool whose declared binding fields are required by this planner output, and no unrelated tool. "
            "Do not solve the task or emit decision fields in this routing step."
        ),
    }
    return (
        "You are the tool-routing step of a robotic agent. Inspect the supplied public image observation and API contract. "
        "Return ONLY one compact JSON object with the exact selected tool names and no Markdown.\n"
        f"Routing contract: {json.dumps(contract, ensure_ascii=False)}\n"
        '{"requested_tools":["exact_tool_name"]}'
    )


def skill_only_payload() -> dict[str, Any]:
    return {
        "requested_tools": [],
        "decisions": {
            "red_cube": {"grasp": "power", "card_id": "none"},
            "blue_cylinder": {"grasp": "power", "card_id": "none"},
        },
        "hazard_present": False,
        "approach_direction": "direct",
        "approach_offset_xy_m": [0.0, 0.0],
        "height_delta_mm": 0,
        "force_scale": 1.0,
        "changed_objects": [],
        "stale_pending_targets": [],
        "replan": False,
        "completed_memory": ["red_cube"],
        "next_target": "blue_cylinder",
        "reason": "fixed skill-only policy",
    }


def nested(payload: Mapping[str, Any] | None, *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def schema_errors(scene: str, payload: Mapping[str, Any] | None) -> list[str]:
    errors: list[str] = []
    if payload is None:
        return ["response is not a JSON object"]
    if "requested_tools" in payload:
        requested_tools = nested(payload, "requested_tools")
        if not isinstance(requested_tools, list) or not all(isinstance(item, str) for item in requested_tools):
            errors.append("requested_tools must be an array of tool-name strings")
        elif len(requested_tools) != len(set(requested_tools)):
            errors.append("requested_tools must not contain duplicates")
    legal_grasps = {"power", "pinch", "side"}
    if scene in {"G1", "G4"}:
        for target in ("red_cube", "blue_cylinder"):
            if str(nested(payload, "decisions", target, "grasp") or "").lower() not in legal_grasps:
                errors.append(f"decisions.{target}.grasp is missing or illegal")
            if not isinstance(nested(payload, "decisions", target, "card_id"), str):
                errors.append(f"decisions.{target}.card_id must be a string")
    if scene in {"G2", "G4"}:
        if not isinstance(nested(payload, "hazard_present"), bool):
            errors.append("hazard_present must be boolean")
        if str(nested(payload, "approach_direction") or "").lower() not in {"away_from_fragile", "direct"}:
            errors.append("approach_direction is missing or illegal")
        offset = nested(payload, "approach_offset_xy_m")
        if not isinstance(offset, list) or len(offset) != 2 or not all(isinstance(value, (int, float)) for value in offset):
            errors.append("approach_offset_xy_m must contain two numbers")
        for key in ("height_delta_mm", "force_scale"):
            if not isinstance(nested(payload, key), (int, float)):
                errors.append(f"{key} must be numeric")
    if scene in {"G3", "G4"}:
        for key in ("changed_objects", "stale_pending_targets", "completed_memory"):
            if not isinstance(nested(payload, key), list):
                errors.append(f"{key} must be an array")
        if not isinstance(nested(payload, "replan"), bool):
            errors.append("replan must be boolean")
        if str(nested(payload, "next_target") or "").lower() not in {"red_cube", "blue_cylinder", "none"}:
            errors.append("next_target is missing or illegal")
    return errors


def public_consistency_errors(
    scene: str,
    condition: str,
    payload: Mapping[str, Any] | None,
    monitor: Mapping[str, Any] | None,
) -> list[str]:
    """Deployable consistency gate using only task, memory, and public measurements."""
    if scene != "G3" or payload is None or monitor is None:
        return []
    threshold = float(monitor["measurement_threshold_m"])
    expected_changed = {
        name for name, item in monitor["per_object_displacement"].items()
        if float(item["distance_m"]) >= threshold
    }
    memory = set() if condition == "A_no_memory" else {"red_cube"}
    pending = [target for target in ("red_cube", "blue_cylinder") if target not in memory]
    expected_stale = expected_changed & set(pending)
    replanning_enabled = condition != "A_no_replan"
    errors: list[str] = []
    changed = nested(payload, "changed_objects")
    stale = nested(payload, "stale_pending_targets")
    completed = nested(payload, "completed_memory")
    if isinstance(changed, list) and set(changed) != expected_changed:
        errors.append("changed_objects is inconsistent with mechanically applying the public displacement threshold to every object")
    if isinstance(stale, list) and set(stale) != expected_stale:
        errors.append("stale_pending_targets is not the intersection of changed objects and pending task targets")
    if isinstance(nested(payload, "replan"), bool) and nested(payload, "replan") != bool(expected_stale and replanning_enabled):
        errors.append("replan is inconsistent with stale pending targets and replanning_enabled")
    if isinstance(completed, list) and set(completed) != memory:
        errors.append("completed_memory does not exactly copy episodic_memory_completed")
    expected_next = next((target for target in ("red_cube", "blue_cylinder") if target not in memory), "none")
    if str(nested(payload, "next_target") or "").lower() != expected_next:
        errors.append("next_target is not the first task item absent from episodic memory")
    return errors


def merge_vlm_results(first: VLMResult, final: VLMResult) -> VLMResult:
    usage: dict[str, Any] = {}
    for result in (first, final):
        for key, value in result.usage.items():
            if isinstance(value, int):
                usage[key] = int(usage.get(key, 0)) + value
    return VLMResult(
        parsed=final.parsed,
        raw_text=final.raw_text,
        latency_ms=first.latency_ms + final.latency_ms,
        requested_model=final.requested_model,
        response_model=final.response_model,
        usage=usage,
        parse_error=final.parse_error,
    )


def canonicalize_planner_payload(scene: str, result: VLMResult) -> VLMResult:
    """Normalize harmless JSON nesting drift without changing any decision value."""
    if result.parsed is None:
        return result
    payload = json.loads(json.dumps(result.parsed))
    if scene in {"G1", "G4"} and not isinstance(payload.get("decisions"), Mapping):
        if all(isinstance(payload.get(target), Mapping) for target in ("red_cube", "blue_cylinder")):
            payload["decisions"] = {
                target: payload.pop(target)
                for target in ("red_cube", "blue_cylinder")
            }
    return VLMResult(
        parsed=payload,
        raw_text=result.raw_text,
        latency_ms=result.latency_ms,
        requested_model=result.requested_model,
        response_model=result.response_model,
        usage=result.usage,
        parse_error=result.parse_error,
    )


def infer_with_schema_repair(
    client: OpenAICompatibleQwenVL,
    scene: str,
    images: list[Path],
    prompt: str,
    retry_count: int,
    consistency_gate: Callable[[Mapping[str, Any] | None], list[str]] | None = None,
) -> tuple[VLMResult, str | None, int, list[str], list[dict[str, Any]], list[str]]:
    result, error, attempts, transient_errors = call_vlm(client, images, prompt, retry_count)
    semantic_attempts = [result.to_dict()]
    result = canonicalize_planner_payload(scene, result)
    errors_before_repair = schema_errors(scene, result.parsed)
    if consistency_gate is not None:
        errors_before_repair.extend(consistency_gate(result.parsed))
    if error is None and errors_before_repair:
        repair_prompt = (
            prompt
            + "\nYour previous response failed the output schema: "
            + "; ".join(errors_before_repair)
            + "\nPrevious response:\n"
            + result.raw_text
            + "\nReturn ONLY the corrected compact JSON object. Preserve decisions that already satisfy the schema. Null is forbidden. "
            + "Every grasp must be exactly power, pinch, or side; top and lateral are approach names, not grasps. "
            + "approach_direction must be exactly away_from_fragile or direct. "
            + "For executive outputs, next_target must always be exactly one of red_cube, blue_cylinder, or none, including while replanning."
        )
        repaired, repair_error, repair_attempts, repair_transient = call_vlm(client, images, repair_prompt, retry_count)
        semantic_attempts.append(repaired.to_dict())
        repaired = canonicalize_planner_payload(scene, repaired)
        result = merge_vlm_results(result, repaired)
        attempts += repair_attempts
        transient_errors.extend(repair_transient)
        error = repair_error
    return result, error, attempts, transient_errors, semantic_attempts, errors_before_repair


def combine_component_results(results: Mapping[str, VLMResult]) -> VLMResult:
    payload: dict[str, Any] = {"component_reasons": {}, "requested_tools": []}
    usage: dict[str, int] = {}
    latency = 0.0
    raw: dict[str, str] = {}
    response_model = None
    for component, result in results.items():
        latency += result.latency_ms
        response_model = response_model or result.response_model
        raw[component] = result.raw_text
        for key, value in result.usage.items():
            if isinstance(value, int):
                usage[key] = usage.get(key, 0) + value
        parsed = result.parsed or {}
        for tool_name in parsed.get("requested_tools", []):
            if isinstance(tool_name, str) and tool_name not in payload["requested_tools"]:
                payload["requested_tools"].append(tool_name)
        payload["component_reasons"][component] = parsed.get("reason")
        if component == "G1":
            payload["decisions"] = parsed.get("decisions")
        elif component == "G2":
            for key in ("hazard_present", "approach_direction", "approach_offset_xy_m", "height_delta_mm", "force_scale"):
                payload[key] = parsed.get(key)
        else:
            for key in ("changed_objects", "stale_pending_targets", "replan", "completed_memory", "next_target"):
                payload[key] = parsed.get(key)
    return VLMResult(
        parsed=payload,
        raw_text=json.dumps(raw, ensure_ascii=False),
        latency_ms=latency,
        requested_model=next(iter(results.values())).requested_model,
        response_model=response_model,
        usage=usage,
        parse_error=None,
    )


def evaluate(
    scene: str,
    condition: str,
    payload: Mapping[str, Any] | None,
    scene_spec: Mapping[str, Any],
    expected_constraint: Mapping[str, Any] | None,
    event_receipt: Mapping[str, Any] | None,
    config: Mapping[str, Any],
    agent_tool_receipts: list[Mapping[str, Any]] | None = None,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {key: None for key in METRIC_KEYS}
    metrics["parse_valid"] = float(not schema_errors(scene, payload))
    components: list[bool] = []

    if scene in {"G1", "G4"}:
        correct = []
        card_correct = []
        for target in ("red_cube", "blue_cylinder"):
            expected = scene_spec["variants"][target]
            correct.append(str(nested(payload, "decisions", target, "grasp") or "").lower() == expected["expected_grasp"])
            card = str(nested(payload, "decisions", target, "card_id") or "none")
            card_correct.append(card in {expected["expected_card"], "none"})
        metrics["grasp_accuracy"] = float(np.mean(correct))
        metrics["retrieval_card_accuracy"] = float(np.mean(card_correct)) if condition == "C2_full" else None
        components.append(all(correct))

    if scene in {"G2", "G4"}:
        assert expected_constraint is not None
        hazard_expected = bool(expected_constraint["hazard_expected"])
        hazard_output = nested(payload, "hazard_present") is True
        direction = str(nested(payload, "approach_direction") or "").lower()
        try:
            offset = np.asarray(nested(payload, "approach_offset_xy_m"), dtype=float)
            height = float(nested(payload, "height_delta_mm"))
            force = float(nested(payload, "force_scale"))
        except (TypeError, ValueError):
            offset, height, force = np.zeros(0), -np.inf, np.inf
        executable = False
        if offset.shape == (2,):
            norm = float(np.linalg.norm(offset))
            if hazard_expected:
                away = np.asarray(expected_constraint["away_unit_xy"], dtype=float)
                cosine = float(np.dot(offset, away) / max(norm, 1e-9))
                executable = norm >= float(config["evaluation"]["minimum_avoidance_offset_m"]) and cosine >= float(config["evaluation"]["minimum_away_cosine"])
            else:
                executable = norm <= 0.02
        safety = hazard_output == hazard_expected
        if hazard_expected:
            safety = safety and direction == "away_from_fragile" and height >= 30.0 and force <= 0.8
        else:
            safety = safety and direction == "direct"
        metrics["safety_decision_correct"] = float(safety)
        metrics["offset_executable"] = float(executable)
        components.append(bool(safety and executable))

    if scene in {"G3", "G4"}:
        assert event_receipt is not None
        target_changed = event_receipt["event"] == "target_move"
        changed = nested(payload, "changed_objects") or []
        stale = nested(payload, "stale_pending_targets") or []
        reported_blue = isinstance(changed, list) and "blue_cylinder" in changed
        stale_blue = isinstance(stale, list) and "blue_cylinder" in stale
        requested_replan = nested(payload, "replan") is True
        replan = requested_replan and condition != "A_no_replan"
        completed = nested(payload, "completed_memory") or []
        memory = isinstance(completed, list) and "red_cube" in completed and str(nested(payload, "next_target") or "").lower() == "blue_cylinder"
        detection = reported_blue == target_changed and stale_blue == target_changed
        replan_correct = replan == target_changed
        metrics["change_detection_correct"] = float(detection)
        metrics["replan_decision_correct"] = float(replan_correct)
        metrics["memory_preserved"] = float(memory)
        metrics["false_replan"] = float(not target_changed and requested_replan)
        components.append(bool(detection and replan_correct and memory))

    if condition == "C2_full" and agent_tool_receipts is not None:
        requested = nested(payload, "requested_tools") or []
        requested_set = {str(tool) for tool in requested} if isinstance(requested, list) else set()
        executed = {
            str(receipt.get("tool"))
            for receipt in agent_tool_receipts
            if receipt.get("status") == "executed"
        }
        selection_correct = requested_set == EVALUATOR_REQUIRED_TOOLS[scene] and executed == requested_set
        metrics["tool_selection_correct"] = float(selection_correct)
        components.append(selection_correct)

    metrics["mechanism_score"] = float(np.mean(components)) if components else 0.0
    metrics["mechanism_success"] = float(bool(components) and all(components))
    return metrics


def expected_constraint(objects: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    red = np.asarray(objects["red_cube"].center_xyz_m[:2], dtype=float)
    fragile = np.asarray(objects["fragile_proxy"].center_xyz_m[:2], dtype=float)
    away = red - fragile
    distance = float(np.linalg.norm(away))
    unit = away / max(distance, 1e-9)
    return {
        "hazard_expected": distance <= float(config["evaluation"]["fragile_threshold_m"]),
        "public_distance_m": distance,
        "away_unit_xy": unit.round(6).tolist(),
    }


def apply_agent_tools(
    _scene: str,
    condition: str,
    payload: Mapping[str, Any] | None,
    constraint: Mapping[str, Any] | None,
    monitor: Mapping[str, Any] | None,
    config: Mapping[str, Any],
    objects: Mapping[str, Any] | None = None,
    cards: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Execute deterministic tools on public observations after VLM reasoning.

    The VLM supplies semantic proposals; numerical geometry and task-state
    invariants are enforced by auditable tools.  These tools never receive the
    randomized evaluator label (expected grasp/event), MuJoCo body state, or
    success metric.
    """
    if payload is None:
        decision: dict[str, Any] = {}
    else:
        decision = json.loads(json.dumps(payload))
    receipts: list[dict[str, Any]] = []
    if condition != "C2_full":
        return decision if payload is not None else None, receipts

    requested = decision.get("requested_tools", [])
    if not isinstance(requested, list):
        requested = []
    seen: set[str] = set()
    requested_tools: list[str] = []
    for value in requested:
        name = str(value)
        if name not in seen:
            requested_tools.append(name)
            seen.add(name)

    def reject(tool: str, reason: str) -> None:
        receipts.append({
            "tool": tool,
            "status": "rejected",
            "routing_source": "vlm_requested_tools",
            "reason": reason,
            "inputs": {},
            "executed_fields": {},
            "uses_privileged_simulator_state": False,
        })

    for tool_name in requested_tools:
        if tool_name not in LEGAL_AGENT_TOOLS:
            reject(tool_name, "unknown_tool")
            continue

        if tool_name == RETRIEVAL_TOOL:
            if not isinstance(decision.get("decisions"), Mapping):
                reject(tool_name, "missing_decision_output_binding")
                continue
            if objects is None or cards is None or any(target not in objects for target in ("red_cube", "blue_cylinder")):
                reject(tool_name, "missing_public_object_descriptors_or_cards")
                continue
            updates: dict[str, Any] = {}
            retrieval_inputs: dict[str, Any] = {}
            for target in ("red_cube", "blue_cylinder"):
                result = retrieve_strategy(objects[target], cards, enabled=True)
                updates[target] = {
                    "grasp": str(result.strategy["synergy"]),
                    "card_id": result.selected_card_id,
                }
                retrieval_inputs[target] = {
                    "public_descriptor": objects[target].to_dict(),
                    "selected_card_id": result.selected_card_id,
                    "selected_strategy": result.strategy,
                }
            proposal_before = decision.get("decisions")
            decision["decisions"] = updates
            receipts.append(
                {
                    "tool": RETRIEVAL_TOOL,
                    "status": "executed",
                    "routing_source": "vlm_requested_tools",
                    "inputs": retrieval_inputs,
                    "proposal_before_tool": proposal_before,
                    "executed_fields": updates,
                    "uses_privileged_simulator_state": False,
                }
            )

        elif tool_name == SAFETY_TOOL:
            if not all(key in decision for key in ("hazard_present", "approach_direction", "approach_offset_xy_m", "height_delta_mm", "force_scale")):
                reject(tool_name, "missing_safety_output_binding")
                continue
            if constraint is None:
                reject(tool_name, "missing_public_protected_relation")
                continue
            hazard = bool(constraint["hazard_expected"])
            if hazard:
                away = np.asarray(constraint["away_unit_xy"], dtype=float)
                offset_norm = max(
                    0.04,
                    float(config["evaluation"]["minimum_avoidance_offset_m"]) + 0.005,
                )
                updates = {
                    "hazard_present": True,
                    "approach_direction": "away_from_fragile",
                    "approach_offset_xy_m": (offset_norm * away).round(6).tolist(),
                    "height_delta_mm": 40,
                    "force_scale": 0.7,
                }
            else:
                updates = {
                    "hazard_present": False,
                    "approach_direction": "direct",
                    "approach_offset_xy_m": [0.0, 0.0],
                    "height_delta_mm": 0,
                    "force_scale": 1.0,
                }
            before = {key: decision.get(key) for key in updates}
            decision.update(updates)
            receipts.append(
                {
                    "tool": SAFETY_TOOL,
                    "status": "executed",
                    "routing_source": "vlm_requested_tools",
                    "inputs": {
                        "public_red_fragile_distance_m": constraint["public_distance_m"],
                        "public_away_unit_xy": constraint["away_unit_xy"],
                        "safety_radius_m": config["evaluation"]["fragile_threshold_m"],
                    },
                    "proposal_before_tool": before,
                    "executed_fields": updates,
                    "uses_privileged_simulator_state": False,
                }
            )

        elif tool_name == MEMORY_TOOL:
            if not all(key in decision for key in ("changed_objects", "stale_pending_targets", "replan", "completed_memory", "next_target")):
                reject(tool_name, "missing_memory_output_binding")
                continue
            if monitor is None:
                reject(tool_name, "missing_public_change_measurements")
                continue
            threshold = float(monitor["measurement_threshold_m"])
            changed = sorted(
                name
                for name, item in monitor["per_object_displacement"].items()
                if float(item["distance_m"]) >= threshold
            )
            completed = ["red_cube"]
            pending = [name for name in ("red_cube", "blue_cylinder") if name not in completed]
            stale = sorted(set(changed) & set(pending))
            updates = {
                "changed_objects": changed,
                "stale_pending_targets": stale,
                "replan": bool(stale),
                "completed_memory": completed,
                "next_target": pending[0] if pending else "none",
            }
            before = {key: decision.get(key) for key in updates}
            decision.update(updates)
            receipts.append(
                {
                    "tool": MEMORY_TOOL,
                    "status": "executed",
                    "routing_source": "vlm_requested_tools",
                    "inputs": {
                        "public_per_object_displacement": monitor["per_object_displacement"],
                        "measurement_threshold_m": threshold,
                        "task_order": ["red_cube", "blue_cylinder"],
                        "episodic_memory_completed": completed,
                    },
                    "proposal_before_tool": before,
                    "executed_fields": updates,
                    "uses_privileged_simulator_state": False,
                }
            )
    return decision, receipts


def call_vlm(client: OpenAICompatibleQwenVL, images: list[Path], prompt: str, retry_count: int) -> tuple[VLMResult, str | None, int, list[str]]:
    errors: list[str] = []
    result: VLMResult | None = None
    error = None
    attempts = 0
    for attempts in range(1, retry_count + 2):
        try:
            result = client.infer(images, prompt)
            error = None
            break
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
            errors.append(error)
            if attempts <= retry_count:
                time.sleep(2.0)
    if result is None:
        result = VLMResult(None, "", 0.0, client.model, None, {}, error)
    return result, error, attempts, errors


def tool_selection_errors(payload: Mapping[str, Any] | None) -> list[str]:
    if payload is None:
        return ["tool router response is not a JSON object"]
    requested = payload.get("requested_tools")
    if not isinstance(requested, list) or not all(isinstance(item, str) for item in requested):
        return ["requested_tools must be an array of tool-name strings"]
    if len(requested) != len(set(requested)):
        return ["requested_tools must not contain duplicates"]
    if any(item not in LEGAL_AGENT_TOOLS for item in requested):
        return ["requested_tools contains an unknown tool name"]
    return []


def infer_tool_selection(
    client: OpenAICompatibleQwenVL,
    images: list[Path],
    prompt: str,
    retry_count: int,
) -> tuple[VLMResult, list[str], str | None, int, list[str], list[dict[str, Any]], list[str]]:
    """Run a standalone model tool-routing turn with schema-only repair."""
    result, error, attempts, transient_errors = call_vlm(client, images, prompt, retry_count)
    semantic_attempts = [result.to_dict()]
    errors_before_repair = tool_selection_errors(result.parsed)
    if error is None and errors_before_repair:
        repair_prompt = (
            prompt
            + "\nYour previous routing response failed the JSON schema: "
            + "; ".join(errors_before_repair)
            + "\nPrevious response:\n"
            + result.raw_text
            + "\nReturn ONLY {\"requested_tools\":[...]} using exact legal names from available_tools."
        )
        repaired, repair_error, repair_attempts, repair_transient = call_vlm(client, images, repair_prompt, retry_count)
        semantic_attempts.append(repaired.to_dict())
        result = merge_vlm_results(result, repaired)
        attempts += repair_attempts
        transient_errors.extend(repair_transient)
        error = repair_error
    final_errors = tool_selection_errors(result.parsed)
    selected = list(result.parsed.get("requested_tools", [])) if result.parsed is not None and not final_errors else []
    return result, selected, error, attempts, transient_errors, semantic_attempts, errors_before_repair


def attach_tool_selection(planner: VLMResult, router: VLMResult | None, selected: list[str]) -> VLMResult:
    """Attach the independent model routing decision to the planner proposal."""
    payload = json.loads(json.dumps(planner.parsed)) if planner.parsed is not None else {}
    payload["requested_tools"] = selected
    if router is None:
        return VLMResult(
            parsed=payload,
            raw_text=planner.raw_text,
            latency_ms=planner.latency_ms,
            requested_model=planner.requested_model,
            response_model=planner.response_model,
            usage=planner.usage,
            parse_error=planner.parse_error,
        )
    merged = merge_vlm_results(router, planner)
    return VLMResult(
        parsed=payload,
        raw_text=json.dumps({"tool_router": router.raw_text, "planner": planner.raw_text}, ensure_ascii=False),
        latency_ms=merged.latency_ms,
        requested_model=planner.requested_model,
        response_model=planner.response_model or router.response_model,
        usage=merged.usage,
        parse_error=planner.parse_error or router.parse_error,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output" / "qwen35_challenge_v2")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18070/v1/chat/completions")
    parser.add_argument("--model", default="/home/admin1/g2_multimodal_agent/models/Qwen3.5-4B")
    parser.add_argument("--deployment-label", default="qwen3.5-4b-bf16-local")
    parser.add_argument("--seed-split", choices=("admission", "main"), default="admission")
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--scenes", nargs="*", choices=("G1", "G2", "G3", "G4"))
    parser.add_argument(
        "--conditions",
        nargs="*",
        choices=("C2_full", "B0_vlm_only", "A_no_graph", "A_no_memory", "A_no_replan", "C1_local", "B0_skill_only"),
        help="Optional condition subset for admission diagnostics; unsupported scene-condition pairs are skipped.",
    )
    parser.add_argument("--retry-count", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    cards = load_cards(CARDS)
    seeds = args.seeds or config["seeds"][args.seed_split]
    scenes = args.scenes or list(config["conditions"])
    args.output.mkdir(parents=True, exist_ok=True)
    client = OpenAICompatibleQwenVL(args.endpoint, args.model, timeout_s=120.0, max_tokens=500)
    selected_conditions = set(args.conditions or [])
    conditions = {
        scene: [condition for condition in config["conditions"][scene] if not selected_conditions or condition in selected_conditions]
        for scene in scenes
    }
    write_json(args.output / "frozen_config.json", {
        "experiment_id": config["experiment_id"],
        "protocol": str(args.config.resolve()),
        "knowledge_cards": str(CARDS.resolve()),
        "knowledge_cards_sha256": hashlib.sha256(CARDS.read_bytes()).hexdigest(),
        "perception_adapter": "deterministic_color_geometry_adapter_v3_height_envelope",
        "agent_tool_contracts": config.get("agent_tool_contracts", {}),
        "endpoint": args.endpoint,
        "requested_model": args.model,
        "deployment_label": args.deployment_label,
        "conditions": conditions,
        "seeds": seeds,
        "seed_split": args.seed_split if not args.seeds else "explicit",
        "temperature": 0.0,
        "max_tokens": 500,
        "skill_only_uses_vlm": False,
        "claim_boundary": config["claim_boundary"],
    })

    rows: list[dict[str, Any]] = []
    for scene in scenes:
        for seed in seeds:
            model = mujoco.MjModel.from_xml_path(str(MJCF))
            data = mujoco.MjData(model)
            initialize_position_targets(model, data)
            scene_spec = realize_scene(model, data, config, scene, seed)
            rest_z = dict(config["object_rest_z_m"])
            rest_z["red_cube"] = scene_spec["variants"]["red_cube"]["rest_z_m"]
            rest_z["blue_cylinder"] = scene_spec["variants"]["blue_cylinder"]["rest_z_m"]
            observation_dir = args.output / "observations" / f"{scene}_seed{seed}"
            before, before_event = public_observation(model, data, rest_z, observation_dir, "before")
            after = None
            after_event = None
            event_receipt = None
            if scene in {"G3", "G4"}:
                event_receipt = apply_change_event(model, data, config, scene_spec, scene, seed)
                after, after_event = public_observation(model, data, rest_z, observation_dir, "after")
            graph = graph_context(before, adjacency_m=float(config["evaluation"]["fragile_threshold_m"]))
            monitor = monitor_context(before, after, float(config["evaluation"]["change_threshold_m"])) if after is not None else None
            constraint = expected_constraint(before, config) if scene in {"G2", "G4"} else None

            for condition in conditions[scene]:
                run_dir = args.output / "runs" / f"{scene}_seed{seed}_{condition}"
                private_path = run_dir / "private_evaluator.json"
                if private_path.is_file() and not args.force:
                    private = json.loads(private_path.read_text(encoding="utf-8"))
                    if not private.get("call_error"):
                        rows.append(private["result_row"])
                        print(f"SKIP {scene} seed={seed} condition={condition}", flush=True)
                        continue

                prompt = build_prompt(scene, condition, config, before, after, cards, graph, monitor, seed)
                before_image = Path(before_event["files"]["rgb"])
                images = [before_image]
                if after_event is not None:
                    images.append(Path(after_event["files"]["rgb"]))
                started = time.time()
                if condition == "B0_skill_only":
                    vlm_result = VLMResult(skill_only_payload(), json.dumps(skill_only_payload()), 0.0, "none", None, {}, None)
                    error, attempts, transient_errors = None, 0, []
                    inference_mode = "deterministic_skill_baseline"
                    semantic_attempts = [vlm_result.to_dict()]
                    errors_before_repair: list[str] = []
                elif scene == "G4":
                    component_conditions = {
                        "C2_full": {"G1": "C2_full", "G2": "C2_full", "G3": "C2_full"},
                        "C1_local": {"G1": "B0_vlm_only", "G2": "A_no_graph", "G3": "C2_full"},
                    }[condition]
                    component_results: dict[str, VLMResult] = {}
                    component_prompts: dict[str, Any] = {}
                    semantic_attempts = []
                    errors_before_repair = []
                    transient_errors = []
                    attempts = 0
                    error = None
                    for component, component_condition in component_conditions.items():
                        component_prompt = build_prompt(component, component_condition, config, before, after, cards, graph, monitor, seed)
                        component_images = [before_image] if component in {"G1", "G2"} else images
                        router_result: VLMResult | None = None
                        selected_tools: list[str] = []
                        routing_prompt: str | None = None
                        if condition == "C2_full":
                            routing_prompt = build_tool_selection_prompt(component, config, seed)
                            router_result, selected_tools, router_error, router_calls, router_transient, router_attempts, router_errors = infer_tool_selection(
                                client, component_images, routing_prompt, args.retry_count
                            )
                            semantic_attempts.extend(
                                {"component": component, "phase": "tool_router", "result": item}
                                for item in router_attempts
                            )
                            errors_before_repair.extend(f"{component} router: {item}" for item in router_errors)
                            transient_errors.extend(router_transient)
                            attempts += router_calls
                            error = error or router_error
                        component_result, component_error, component_call_attempts, component_transient, component_attempts, component_errors = infer_with_schema_repair(
                            client,
                            component,
                            component_images,
                            component_prompt,
                            args.retry_count,
                            lambda payload, component=component, component_condition=component_condition: public_consistency_errors(
                                component, component_condition, payload, monitor
                            ),
                        )
                        component_result = attach_tool_selection(component_result, router_result, selected_tools)
                        component_results[component] = component_result
                        component_prompts[component] = {"tool_router": routing_prompt, "planner": component_prompt}
                        semantic_attempts.extend({"component": component, "phase": "planner", "result": item} for item in component_attempts)
                        errors_before_repair.extend(f"{component}: {item}" for item in component_errors)
                        transient_errors.extend(component_transient)
                        attempts += component_call_attempts
                        error = error or component_error
                    vlm_result = combine_component_results(component_results)
                    prompt = json.dumps({"agentic_decomposition": component_prompts}, ensure_ascii=False)
                    inference_mode = "decomposed_agentic_multimodal_qwen_with_explicit_tool_router"
                else:
                    router_result = None
                    selected_tools = []
                    routing_prompt = None
                    routing_attempts: list[dict[str, Any]] = []
                    routing_errors: list[str] = []
                    router_error = None
                    router_calls = 0
                    router_transient: list[str] = []
                    if condition == "C2_full":
                        routing_prompt = build_tool_selection_prompt(scene, config, seed)
                        router_result, selected_tools, router_error, router_calls, router_transient, routing_attempts, routing_errors = infer_tool_selection(
                            client, images, routing_prompt, args.retry_count
                        )
                    planner_result, planner_error, planner_calls, planner_transient, planner_attempts, planner_errors = infer_with_schema_repair(
                        client,
                        scene,
                        images,
                        prompt,
                        args.retry_count,
                        lambda payload: public_consistency_errors(scene, condition, payload, monitor),
                    )
                    vlm_result = attach_tool_selection(planner_result, router_result, selected_tools)
                    error = router_error or planner_error
                    attempts = router_calls + planner_calls
                    transient_errors = [*router_transient, *planner_transient]
                    semantic_attempts = [
                        *({"phase": "tool_router", "result": item} for item in routing_attempts),
                        *({"phase": "planner", "result": item} for item in planner_attempts),
                    ]
                    errors_before_repair = [
                        *(f"router: {item}" for item in routing_errors),
                        *(f"planner: {item}" for item in planner_errors),
                    ]
                    if routing_prompt is not None:
                        prompt = json.dumps({"tool_router": routing_prompt, "planner": prompt}, ensure_ascii=False)
                    inference_mode = "real_multimodal_qwen_with_explicit_tool_router" if condition == "C2_full" else "real_multimodal_qwen"
                executed_decision, agent_tool_receipts = apply_agent_tools(
                    scene,
                    condition,
                    vlm_result.parsed,
                    constraint,
                    monitor,
                    config,
                    before,
                    cards,
                )
                metrics = evaluate(
                    scene,
                    condition,
                    executed_decision,
                    scene_spec,
                    constraint,
                    event_receipt,
                    config,
                    agent_tool_receipts,
                )
                row = {
                    "scene": scene,
                    "seed": seed,
                    "condition": condition,
                    "event_type": event_receipt["event"] if event_receipt else "none",
                    "latency_ms": vlm_result.latency_ms,
                    "vlm_calls": 0 if condition == "B0_skill_only" else len(semantic_attempts),
                    **metrics,
                }
                rows.append(row)
                public_events = [before_event]
                if after_event is not None:
                    public_events.append(after_event)
                public_events.append({
                    "event": "planner_decision",
                    "condition": condition,
                    "inference_mode": inference_mode,
                    "prompt": prompt,
                    "image_files": [str(path) for path in images],
                    "result": vlm_result.to_dict(),
                    "executed_decision": executed_decision,
                    "agent_tool_receipts": agent_tool_receipts,
                    "semantic_attempts": semantic_attempts,
                    "schema_errors_before_repair": errors_before_repair,
                    "uses_privileged_simulator_state": False,
                })
                write_jsonl(run_dir / "public_trace.jsonl", public_events)
                write_json(run_dir / "scene_spec.json", {"role": "evaluator_initialization_only", **scene_spec})
                write_json(private_path, {
                    "role": "private_evaluator_only",
                    "scene": scene,
                    "seed": seed,
                    "condition": condition,
                    "expected_constraint": constraint,
                    "change_event_receipt": event_receipt,
                    "metrics": metrics,
                    "result_row": row,
                    "call_error": error,
                    "call_attempts": attempts,
                    "semantic_calls": 0 if condition == "B0_skill_only" else len(semantic_attempts),
                    "schema_errors_before_repair": errors_before_repair,
                    "transient_call_errors": transient_errors,
                    "wall_clock_started": started,
                })
                print(f"DONE {scene} seed={seed} condition={condition} score={metrics['mechanism_score']:.2f} success={metrics['mechanism_success']:.0f} latency_ms={vlm_result.latency_ms:.0f}", flush=True)

    fieldnames = ["scene", "seed", "condition", "event_type", "latency_ms", "vlm_calls", *METRIC_KEYS]
    with (args.output / "challenge_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, Any] = {}
    for scene, scene_conditions in conditions.items():
        summary[scene] = {}
        for condition in scene_conditions:
            subset = [row for row in rows if row["scene"] == scene and row["condition"] == condition]
            summary[scene][condition] = {
                key: float(np.mean([float(row[key]) for row in subset if row[key] is not None]))
                for key in METRIC_KEYS
                if any(row[key] is not None for row in subset)
            }
            summary[scene][condition]["runs"] = len(subset)
            summary[scene][condition]["latency_ms_median"] = float(np.median([float(row["latency_ms"]) for row in subset]))
            summary[scene][condition]["vlm_calls_mean"] = float(np.mean([float(row["vlm_calls"]) for row in subset]))
    write_json(args.output / "challenge_summary.json", {"runs": len(rows), "summary": summary})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
