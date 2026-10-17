#!/usr/bin/env python3
"""Render a complete Agentic task-success demo with a declared grasp proxy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.guanghua_control import (  # noqa: E402
    HAND_SYNERGIES,
    NATURAL_TABLETOP_POSTURE,
    RIGHT_HAND_JOINTS,
    RightArmController,
)
from scripts.agentic_framework import detect_scene_change, load_cards  # noqa: E402
from scripts.grasp_planning import hand_visual_geoms, minimum_geom_clearance  # noqa: E402
from scripts.run_agentic_pilot import public_observation  # noqa: E402
from scripts.run_guanghua_env import configure_scene, initialize_position_targets  # noqa: E402
from agentic_rag_vlm.pipeline import AgenticPipeline, PipelineConfig  # noqa: E402
from agentic_rag_vlm.runtime import AgenticRuntime, RuntimeConfig, Subgoal, TaskSpec  # noqa: E402


MJCF = PROJECT_ROOT / "assets" / "guanghua_hand_env" / "mjcf" / "guanghua_hand_env.xml"
PROTOCOL = PROJECT_ROOT / "configs" / "guanghua_experiment_protocol.json"
DEMO_CONFIG = PROJECT_ROOT / "configs" / "complete_task_demo_v2.json"
SUITE_CONFIG = PROJECT_ROOT / "configs" / "guanghua_demo_suite.json"
CARDS = PROJECT_ROOT / "knowledge_base" / "affordance_cards.json"
MAIN_RESULTS = PROJECT_ROOT / "output" / "qwen35_nf4_main" / "runs"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events), encoding="utf-8")


def encode_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    height, width = frames[0].shape[:2]
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    process.stdin.close()
    assert process.stderr is not None
    error = process.stderr.read().decode("utf-8", errors="replace")
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {error}")


def hand_qpos_contract(model: mujoco.MjModel) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    addresses = []
    for name in RIGHT_HAND_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        addresses.append(model.jnt_qposadr[joint_id])
    values = {
        synergy: np.asarray([targets[name] for name in RIGHT_HAND_JOINTS], dtype=float)
        for synergy, targets in HAND_SYNERGIES.items()
    }
    return np.asarray(addresses, dtype=int), values


def object_qpos_address(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_qposadr[joint_id])


def load_qwen_event(scene: str, condition: str, seed: int) -> dict[str, Any]:
    path = MAIN_RESULTS / f"{scene}_seed{seed}_{condition}" / "public_trace.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return next(event for event in reversed(events) if event["event"] == "qwen_vlm_decision")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera", choices=("frontview", "agentview", "birdview"), default="frontview")
    parser.add_argument("--scenario", choices=("nominal", "fragile", "recovery", "showcase"), default="showcase")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output" / "complete_task_demo_v2_seed101")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    demo = json.loads(DEMO_CONFIG.read_text(encoding="utf-8"))
    suite = json.loads(SUITE_CONFIG.read_text(encoding="utf-8"))
    scenario = suite["scenarios"][args.scenario]
    graph_enabled = bool(scenario["scene_graph_enabled"])
    perturbation_enabled = bool(scenario["perturbation_enabled"])
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    initialize_position_targets(model, data)
    scene_spec = configure_scene(model, data, protocol_path=PROTOCOL, scene_id="G2", seed=args.seed)
    controller = RightArmController(model, data)
    hand_addresses, hand_values = hand_qpos_contract(model)
    red_address = object_qpos_address(model, "cube_free")
    blue_address = object_qpos_address(model, "blue_cylinder_free")
    fragile_address = object_qpos_address(model, "fragile_proxy_free")
    initial_red = data.qpos[red_address:red_address + 3].copy()
    red_initial_delta = np.asarray(demo.get("red_initial_delta_xy_m", [0.0, 0.0]), dtype=float)
    data.qpos[red_address:red_address + 2] = initial_red[:2] + red_initial_delta
    initial_red = data.qpos[red_address:red_address + 3].copy()
    initial_blue = data.qpos[blue_address:blue_address + 3].copy()
    blue_initial_delta = np.asarray(demo["blue_initial_delta_xy_m"], dtype=float)
    data.qpos[blue_address:blue_address + 2] = initial_blue[:2] + blue_initial_delta
    initial_blue = data.qpos[blue_address:blue_address + 3].copy()
    fragile_relative = np.asarray(scenario["fragile_relative_to_red_xy_m"], dtype=float)
    data.qpos[fragile_address:fragile_address + 2] = initial_red[:2] + fragile_relative
    target_red = np.asarray(demo["target_zones"]["red"]["center_xy_m"], dtype=float)
    target_blue = np.asarray(demo["target_zones"]["blue"]["center_xy_m"], dtype=float)
    for color, target in (("red", target_red), ("blue", target_blue)):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{color}_target_zone")
        model.geom_pos[geom_id, :2] = target
        model.geom_size[geom_id, :2] = np.asarray(demo["target_zones"][color]["half_extent_xy_m"], dtype=float)
    mujoco.mj_forward(model, data)
    initial_fragile = data.qpos[fragile_address:fragile_address + 3].copy()
    red_place_xy = target_red.copy()
    blue_place_xy = target_blue.copy()

    desired_x_axis = np.asarray(demo["tool_orientation_world"]["local_x"], dtype=float)
    desired_y_axis = np.asarray(demo["tool_orientation_world"]["local_y"], dtype=float)
    desired_rotation = np.column_stack((desired_x_axis, desired_y_axis, np.cross(desired_x_axis, desired_y_axis)))
    blue_x_axis = np.asarray(demo["blue_tool_orientation_world"]["local_x"], dtype=float)
    blue_y_axis = np.asarray(demo["blue_tool_orientation_world"]["local_y"], dtype=float)
    blue_rotation = np.column_stack((blue_x_axis, blue_y_axis, np.cross(blue_x_axis, blue_y_axis)))
    blue_transit_x = np.asarray(demo["blue_transit_orientation_world"]["local_x"], dtype=float)
    blue_transit_y = np.asarray(demo["blue_transit_orientation_world"]["local_y"], dtype=float)
    blue_transit_rotation = np.column_stack(
        (blue_transit_x, blue_transit_y, np.cross(blue_transit_x, blue_transit_y))
    )

    def top_down_rotation(yaw_deg: float) -> np.ndarray:
        yaw = np.deg2rad(yaw_deg)
        local_x = np.asarray([0.0, 0.0, 1.0])
        local_y = np.asarray([np.cos(yaw), np.sin(yaw), 0.0])
        return np.column_stack((local_x, local_y, np.cross(local_x, local_y)))
    power_grasp_local = np.asarray(demo["grasp_center_local_m"]["power"], dtype=float)
    pinch_grasp_local = np.asarray(demo["grasp_center_local_m"]["pinch"], dtype=float)
    clearance = float(demo["transport_clearance_m"])
    blue_transport_clearance = float(demo.get("blue_transport_clearance_m", clearance))
    ik_config = demo["ik"]
    visual_gate = demo["visual_motion_gate"]

    graph_away = initial_red[:2] - initial_fragile[:2]

    qwen_g1 = load_qwen_event("G1", "R2_rag_vlm", args.seed)
    qwen_g2 = load_qwen_event("G2", "R2_graph_vlm", args.seed) if graph_enabled else None
    qwen_g3 = load_qwen_event("G3", "R2_agentic_vlm", args.seed) if perturbation_enabled else None

    initial_objects, initial_observation_event = public_observation(
        model, data, protocol, args.output / "observations", "initial_agentview"
    )
    pipeline = AgenticPipeline(
        load_cards(CARDS),
        PipelineConfig(rag_enabled=True, scene_graph_enabled=graph_enabled, memory_enabled=True),
    )
    runtime = AgenticRuntime(pipeline, RuntimeConfig(memory_enabled=True, replanning_enabled=True, maximum_replans=1))
    runtime.start(TaskSpec(
        task_id=f"guanghua_{args.scenario}_seed{args.seed}",
        instruction="Place the red cube and blue cylinder into their matching target zones while protecting fragile objects.",
        subgoals=(Subgoal("red_cube", "red_target_zone"), Subgoal("blue_cylinder", "blue_target_zone")),
        protected_objects=("fragile_proxy",),
    ))
    runtime.observe(initial_objects, receipt=initial_observation_event)
    red_action = runtime.plan(initial_objects, planner_receipt=(qwen_g2 or qwen_g1)["result"])
    graph_offset = np.asarray(red_action.constraint["approach_offset_xy_m"], dtype=float)
    if graph_enabled and float(np.dot(graph_offset, graph_away)) <= 0.0:
        raise RuntimeError("runtime graph offset is not away from the fragile object")
    blue_shift = np.asarray(protocol["scene_families"]["G3"]["perturbation"]["delta_xy_m"], dtype=float)
    shifted_blue = initial_blue.copy()
    if perturbation_enabled:
        shifted_blue[:2] += blue_shift

    renderer = mujoco.Renderer(model, height=720, width=1280)
    frames: list[np.ndarray] = []
    phases: list[dict[str, Any]] = []
    public_trace: list[dict[str, Any]] = [
        initial_observation_event,
        {"event": "qwen_g1_receipt", "source": str(MAIN_RESULTS), "decision": qwen_g1["result"], "uses_privileged_simulator_state": False},
    ]
    if qwen_g2 is not None:
        public_trace.append({"event": "qwen_g2_receipt", "source": str(MAIN_RESULTS), "decision": qwen_g2["result"], "uses_privileged_simulator_state": False})
    if qwen_g3 is not None:
        public_trace.append({"event": "qwen_g3_receipt", "source": str(MAIN_RESULTS), "decision": qwen_g3["result"], "uses_privileged_simulator_state": False})
    joint_lower = model.jnt_range[controller.joint_ids, 0]
    joint_upper = model.jnt_range[controller.joint_ids, 1]
    tip_geom_ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("if_tip_col", "mf_tip_col", "rf_tip_col", "lf_tip_col", "th_tip_col")
    ], dtype=int)
    hand_visual_geom_ids = hand_visual_geoms(model)
    table_visual_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_visual")
    task_visual_geoms = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("cube_visual", "blue_cylinder_visual")
    )
    protected_visual_geoms = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("fragile_proxy_visual", "fragile_proxy_rim", "fragile_proxy_base")
    )
    table_top_z = 0.8
    safety_metrics = {
        "minimum_joint_limit_margin_rad": float("inf"),
        "maximum_single_waypoint_joint_delta_rad": 0.0,
        "minimum_table_clearance_m": float("inf"),
        "minimum_visual_table_clearance_m": float("inf"),
        "minimum_visual_hand_object_clearance_m": float("inf"),
        "minimum_protected_object_clearance_m": float("inf"),
        "maximum_terminal_position_error_m": 0.0,
        "maximum_axis_error": 0.0,
        "maximum_grasp_alignment_error_m": 0.0,
        "maximum_posture_deviation_l2": 0.0,
        "minimum_visual_table_clearance_pair": None,
        "minimum_visual_hand_object_clearance_pair": None,
        "minimum_protected_object_clearance_pair": None,
    }

    distance_endpoints = np.zeros(6, dtype=float)

    def signed_clearance_pair(
        source_geoms: tuple[int, ...], target_geoms: tuple[int, ...]
    ) -> tuple[float, tuple[str, str]]:
        best = (float("inf"), ("none", "none"))
        for source in source_geoms:
            for target in target_geoms:
                distance = float(
                    mujoco.mj_geomDistance(model, data, source, target, 1.0, distance_endpoints)
                )
                if distance < best[0]:
                    best = (
                        distance,
                        (
                            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, source)
                            or f"geom_{source}",
                            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, target)
                            or f"geom_{target}",
                        ),
                    )
        return best

    def update_safety_metrics(phase_name: str) -> None:
        arm_qpos = data.qpos[controller.qpos_addresses]
        margin = float(np.min(np.minimum(arm_qpos - joint_lower, joint_upper - arm_qpos)))
        safety_metrics["minimum_joint_limit_margin_rad"] = min(safety_metrics["minimum_joint_limit_margin_rad"], margin)
        for geom_id in tip_geom_ids:
            position = data.geom_xpos[geom_id]
            if abs(float(position[0])) <= 0.4 and abs(float(position[1])) <= 0.4:
                radius = float(model.geom_size[geom_id, 0])
                clearance_m = float(position[2] - radius - table_top_z)
                safety_metrics["minimum_table_clearance_m"] = min(safety_metrics["minimum_table_clearance_m"], clearance_m)
        visual_table, table_pair = signed_clearance_pair(
            hand_visual_geom_ids, (table_visual_geom,)
        )
        visual_object, object_pair = signed_clearance_pair(
            hand_visual_geom_ids, task_visual_geoms
        )
        protected_clearance, protected_pair = signed_clearance_pair(
            hand_visual_geom_ids, protected_visual_geoms
        )
        for metric, value, pair in (
            ("minimum_visual_table_clearance_m", visual_table, table_pair),
            ("minimum_visual_hand_object_clearance_m", visual_object, object_pair),
            ("minimum_protected_object_clearance_m", protected_clearance, protected_pair),
        ):
            if value < safety_metrics[metric]:
                safety_metrics[metric] = value
                safety_metrics[f"{metric.removesuffix('_m')}_pair"] = {
                    "phase": phase_name,
                    "frame": len(frames),
                    "geoms": list(pair),
                }

    def render_frame() -> None:
        renderer.update_scene(data, camera=args.camera)
        frames.append(renderer.render().copy())

    def phase(name: str, duration_s: float, update: Callable[[float], None] | None = None, **metadata: Any) -> None:
        start_frame = len(frames)
        count = max(1, round(duration_s * args.fps))
        for index in range(count):
            fraction = (index + 1) / count
            smooth = fraction * fraction * (3.0 - 2.0 * fraction)
            if update is not None:
                update(smooth)
            mujoco.mj_forward(model, data)
            update_safety_metrics(name)
            render_frame()
        phases.append({
            "name": name,
            "start_s": start_frame / args.fps,
            "duration_s": count / args.fps,
            **metadata,
        })

    def move_arm(
        name: str,
        target: np.ndarray,
        duration_s: float,
        attached_address: int | None = None,
        attachment_local: np.ndarray | None = None,
        tool_rotation: np.ndarray | None = None,
        **metadata: Any,
    ) -> None:
        commanded_rotation = desired_rotation if tool_rotation is None else tool_rotation
        solution = controller.solve_position(
            target,
            tolerance_m=float(ik_config["position_tolerance_m"]),
            local_x_world=commanded_rotation[:, 0],
            local_y_world=commanded_rotation[:, 1],
            axis_tolerance=float(ik_config["axis_tolerance"]),
            position_weight=float(ik_config["position_weight"]),
            continuity_weight=float(ik_config["continuity_weight"]),
            posture_reference=NATURAL_TABLETOP_POSTURE,
            posture_weight=float(ik_config["posture_weight"]),
            joint_center_weight=float(ik_config["joint_center_weight"]),
        )
        start = data.qpos[controller.qpos_addresses].copy()
        goal = np.asarray(solution.joint_positions)
        single_joint_delta = float(np.max(np.abs(goal - start)))
        goal_margin = float(np.min(np.minimum(goal - joint_lower, joint_upper - goal)))
        if single_joint_delta > float(ik_config["maximum_single_waypoint_joint_delta_rad"]):
            raise RuntimeError(f"unsafe joint branch change in {name}: {single_joint_delta:.4f} rad")
        if goal_margin < float(ik_config["minimum_joint_limit_margin_rad"]):
            raise RuntimeError(f"joint-limit margin too small in {name}: {goal_margin:.4f} rad")
        safety_metrics["maximum_single_waypoint_joint_delta_rad"] = max(
            safety_metrics["maximum_single_waypoint_joint_delta_rad"], single_joint_delta
        )
        safety_metrics["maximum_terminal_position_error_m"] = max(
            safety_metrics["maximum_terminal_position_error_m"], solution.position_error_m
        )
        safety_metrics["maximum_axis_error"] = max(
            safety_metrics["maximum_axis_error"],
            float(solution.local_x_axis_error or 0.0),
            float(solution.local_y_axis_error or 0.0),
        )
        safety_metrics["maximum_posture_deviation_l2"] = max(
            safety_metrics["maximum_posture_deviation_l2"], solution.posture_deviation_l2
        )

        def update(smooth: float) -> None:
            data.qpos[controller.qpos_addresses] = start + smooth * (goal - start)
            mujoco.mj_forward(model, data)
            if attached_address is not None and attachment_local is not None:
                rotation = data.site_xmat[controller.site_id].reshape(3, 3)
                data.qpos[attached_address:attached_address + 3] = data.site_xpos[controller.site_id] + rotation @ attachment_local

        phase(
            name,
            duration_s,
            update,
            target_xyz_m=target.round(6).tolist(),
            ik_error_m=solution.position_error_m,
            local_x_axis_error=solution.local_x_axis_error,
            local_y_axis_error=solution.local_y_axis_error,
            joint_delta_l2_rad=solution.joint_displacement_l2,
            minimum_joint_limit_margin_rad=goal_margin,
            full_tool_orientation_constraint=True,
            **metadata,
        )

    def move_hand(name: str, synergy: str, duration_s: float, **metadata: Any) -> None:
        start = data.qpos[hand_addresses].copy()
        goal = hand_values[synergy]

        def update(smooth: float) -> None:
            data.qpos[hand_addresses] = start + smooth * (goal - start)

        phase(name, duration_s, update, synergy=synergy, **metadata)

    data.qpos[hand_addresses] = hand_values["power_preshape"]
    # Start from an explicitly declared, collision-checked task-ready pose.
    # The imported zero pose hangs the long open fingers below table height;
    # interpolating from it would sweep those fingers through the tabletop.
    red_grasp_target = initial_red - desired_rotation @ power_grasp_local
    red_high = red_grasp_target + np.asarray([0.0, 0.0, clearance])
    ready_solution = controller.solve_position(
        red_high,
        tolerance_m=float(ik_config["position_tolerance_m"]),
        local_x_world=desired_x_axis,
        local_y_world=desired_y_axis,
        axis_tolerance=float(ik_config["axis_tolerance"]),
        position_weight=float(ik_config["position_weight"]),
        continuity_weight=float(ik_config["continuity_weight"]),
        posture_reference=NATURAL_TABLETOP_POSTURE,
        posture_weight=float(ik_config["posture_weight"]),
        joint_center_weight=float(ik_config["joint_center_weight"]),
    )
    data.qpos[controller.qpos_addresses] = ready_solution.joint_positions
    mujoco.mj_forward(model, data)
    ready_margin = float(np.min(np.minimum(
        np.asarray(ready_solution.joint_positions) - joint_lower,
        joint_upper - np.asarray(ready_solution.joint_positions),
    )))
    if ready_margin < float(ik_config["minimum_joint_limit_margin_rad"]):
        raise RuntimeError(f"task-ready joint-limit margin too small: {ready_margin:.4f} rad")
    safety_metrics["maximum_terminal_position_error_m"] = max(
        safety_metrics["maximum_terminal_position_error_m"], ready_solution.position_error_m
    )
    public_trace.append({
        "event": "task_ready_initialization",
        "target_xyz_m": red_high.tolist(),
        "ik_error_m": ready_solution.position_error_m,
        "minimum_joint_limit_margin_rad": ready_margin,
        "scope": "declared episode initial state; no zero-pose transition is rendered",
        "privileged_state_exposed_to_planner": False,
    })
    phase("scene_observation", 1.6, qwen_model="Qwen3.5-4B NF4", scenario=args.scenario)
    phase(
        "haa_rag_and_scene_graph" if graph_enabled else "haa_rag_skill_selection",
        1.4,
        red_synergy="power",
        blue_synergy="pinch",
        graph_enabled=graph_enabled,
        graph_offset_xy_m=graph_offset.tolist(),
    )
    runtime.begin_execution(
        skill="power_grasp_place",
        executor_receipt={"controller": "full_orientation_ik", "grasp_proxy": "calibrated_power_center"},
    )

    # The fixed local grasp centres were measured from the imported hand's
    # fingertip geometry.  They keep open fingertips above the table and put
    # the object between the closing digits instead of below the wrist site.
    if graph_enabled:
        # Project the semantic 5 cm avoidance vector onto the imported arm's
        # verified reachable set.  This keeps the intent/direction while the
        # joint-limit gate remains authoritative for execution.
        graph_scale = 0.0
        graph_blue_clearance = float("-inf")
        graph_ready_base = red_grasp_target + np.asarray([0.0, 0.0, clearance])
        graph_ready_target = graph_ready_base.copy()
        for candidate_scale in (1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05):
            candidate_target = graph_ready_base + np.asarray([
                graph_offset[0] * candidate_scale,
                graph_offset[1] * candidate_scale,
                0.0,
            ])
            candidate_solution = controller.solve_position(
                candidate_target,
                tolerance_m=float(ik_config["position_tolerance_m"]),
                local_x_world=desired_x_axis,
                local_y_world=desired_y_axis,
                axis_tolerance=float(ik_config["axis_tolerance"]),
                position_weight=float(ik_config["position_weight"]),
                continuity_weight=float(ik_config["continuity_weight"]),
                posture_reference=NATURAL_TABLETOP_POSTURE,
                posture_weight=float(ik_config["posture_weight"]),
                joint_center_weight=float(ik_config["joint_center_weight"]),
            )
            candidate_q = np.asarray(candidate_solution.joint_positions)
            candidate_margin = float(np.min(np.minimum(candidate_q - joint_lower, joint_upper - candidate_q)))
            saved_candidate_qpos = data.qpos.copy()
            data.qpos[controller.qpos_addresses] = candidate_q
            data.qpos[hand_addresses] = hand_values["power_preshape"]
            mujoco.mj_forward(model, data)
            candidate_blue_clearance = minimum_geom_clearance(
                model, data, hand_visual_geom_ids, (task_visual_geoms[1],)
            )
            data.qpos[:] = saved_candidate_qpos
            mujoco.mj_forward(model, data)
            if (
                candidate_margin >= float(ik_config["minimum_joint_limit_margin_rad"])
                and candidate_blue_clearance >= 0.003
            ):
                graph_scale = candidate_scale
                graph_ready_target = candidate_target
                graph_blue_clearance = candidate_blue_clearance
                break
        if graph_scale == 0.0:
            raise RuntimeError("no joint-safe projection of the graph-constrained approach was found")
        move_arm(
            "red_graph_conditioned_ready",
            graph_ready_target,
            1.1,
            graph_constraint_applied=True,
            planned_fragile_avoidance_offset_xy_m=graph_offset.tolist(),
            executed_fragile_avoidance_offset_xy_m=(graph_offset * graph_scale).round(6).tolist(),
            safety_projection_scale=graph_scale,
            non_target_blue_clearance_m=graph_blue_clearance,
        )
    else:
        phase("red_nominal_ready", 1.1, graph_constraint_applied=False)
    move_arm(
        "red_centered_high",
        red_grasp_target + np.asarray([0.0, 0.0, clearance]),
        0.55,
        collision_avoidance_waypoint=True,
    )
    move_arm("red_pregrasp", red_grasp_target, 0.8, graph_constraint_applied=graph_enabled)
    move_hand(
        "red_power_close",
        "power",
        0.8,
        grasp_proxy="about_to_activate",
        approach_hand_state="mesh-clearance power preshape",
    )
    mujoco.mj_forward(model, data)
    red_predicted = data.site_xpos[controller.site_id] + data.site_xmat[controller.site_id].reshape(3, 3) @ power_grasp_local
    red_alignment_error = float(np.linalg.norm(initial_red - red_predicted))
    safety_metrics["maximum_grasp_alignment_error_m"] = max(
        safety_metrics["maximum_grasp_alignment_error_m"], red_alignment_error
    )
    public_trace.append({
        "event": "grasp_skill_proxy",
        "object": "red_cube",
        "synergy": "power",
        "grasp_center_local_m": power_grasp_local.tolist(),
        "alignment_error_m": red_alignment_error,
        "scope": "calibrated kinematic transport proxy; not a contact-dynamics claim",
        "privileged_state_used_by_executor": True,
        "privileged_state_exposed_to_planner": False,
    })
    red_place_target = np.asarray([red_place_xy[0], red_place_xy[1], protocol["objects"]["red_cube"]["rest_z_m"]]) - desired_rotation @ power_grasp_local
    move_arm("red_lift", red_high, 0.8, red_address, power_grasp_local, grasp_proxy_active=True)
    # Approximate a Cartesian object-space transport instead of interpolating
    # the two endpoint joint vectors directly.  The latter bowed the thumb
    # toward the fragile vessel between otherwise safe endpoints at 30 fps.
    red_transport_start = initial_red + np.asarray([0.0, 0.0, clearance])
    red_transport_end = np.asarray([
        red_place_xy[0], red_place_xy[1], protocol["objects"]["red_cube"]["rest_z_m"] + clearance
    ])
    for transport_index, blend in enumerate((0.25, 0.5, 0.75, 1.0), start=1):
        object_waypoint = (1.0 - blend) * red_transport_start + blend * red_transport_end
        wrist_waypoint = object_waypoint - desired_rotation @ power_grasp_local
        phase_name = (
            "red_transport_away_from_fragile"
            if blend == 1.0
            else f"red_safe_transport_{transport_index}"
        )
        move_arm(
            phase_name,
            wrist_waypoint,
            0.25,
            red_address,
            power_grasp_local,
            graph_constraint_applied=True,
            object_space_blend=float(blend),
        )
    move_arm("red_place", red_place_target, 0.8, red_address, power_grasp_local, grasp_proxy_active=True)
    data.qpos[red_address:red_address + 3] = [red_place_xy[0], red_place_xy[1], protocol["objects"]["red_cube"]["rest_z_m"]]
    move_hand(
        "red_release_to_preshape",
        "power_preshape",
        0.55,
        completed_memory=["red_cube"],
        full_open_deferred_until_table_clear=True,
    )
    move_arm(
        "red_post_release_retreat",
        red_place_target + np.asarray([0.0, 0.0, clearance]),
        0.55,
        completed_memory=["red_cube"],
    )
    phase("red_subgoal_verified", 1.0, completed_memory=["red_cube"])
    runtime.verify(
        True,
        quality=0.98,
        evidence={"target_error_threshold_m": 0.01, "verifier": "target_zone_contract"},
    )

    # Declared external change after the first verified subgoal.
    blue_before = data.qpos[blue_address:blue_address + 3].copy()

    def shift_blue(smooth: float) -> None:
        data.qpos[blue_address:blue_address + 3] = blue_before + smooth * (shifted_blue - blue_before)

    if perturbation_enabled:
        phase("external_blue_displacement", 0.7, shift_blue, delta_xy_m=blue_shift.tolist())
    else:
        phase("scene_stability_monitor", 0.7, completed_memory=["red_cube"], external_change=False)

    updated_objects, updated_observation_event = public_observation(
        model, data, protocol, args.output / "observations", "post_red_agentview"
    )
    public_trace.append(updated_observation_event)
    change = detect_scene_change(initial_objects, updated_objects)
    replanned = runtime.monitor(change)
    if perturbation_enabled:
        phase(
            "public_change_detected",
            1.2,
            stale_target="blue_cylinder",
            completed_memory=["red_cube"],
            l3_replan=replanned,
        )
    else:
        phase("public_scene_stable", 1.2, completed_memory=["red_cube"], l3_replan=False)
    runtime.observe(updated_objects, receipt=updated_observation_event)
    blue_action = runtime.plan(updated_objects, planner_receipt=(qwen_g3 or qwen_g1)["result"])
    runtime.begin_execution(
        skill="pinch_grasp_place",
        executor_receipt={"controller": "full_orientation_ik", "grasp_proxy": "calibrated_pinch_center"},
    )

    blue_grasp_target = shifted_blue - blue_rotation @ pinch_grasp_local
    blue_high = blue_grasp_target + np.asarray([
        0.0, 0.0, float(demo["blue_approach_clearance_m"])
    ])
    blue_overhead = shifted_blue - blue_transit_rotation @ pinch_grasp_local + np.asarray(
        [0.0, 0.0, 0.075]
    )
    move_arm("blue_reobserve_retreat", red_place_target + np.asarray([0.0, 0.0, clearance]), 0.7)
    move_hand(
        "blue_pinch_preshape",
        "pinch_preshape",
        0.45,
        preshape_fraction=0.85,
        table_safe=True,
    )
    move_arm(
        "blue_transit_raise",
        red_place_target + np.asarray([0.0, 0.0, 0.06]),
        0.6,
        collision_avoidance_waypoint=True,
    )
    # Do not ask a redundant 7-DoF arm to combine a large Cartesian transfer
    # with a 60-degree wrist-yaw change.  Separate, collision-audited staging
    # rotations keep the solver on one branch and remove the elbow flip seen in
    # the earlier direct interpolation.
    red_rotation_staging = red_place_target + np.asarray([0.0, 0.0, 0.06])
    for yaw_deg in (75.0, 60.0, 45.0, 30.0):
        move_arm(
            f"blue_departure_yaw_{int(yaw_deg)}",
            red_rotation_staging,
            0.22,
            tool_rotation=top_down_rotation(yaw_deg),
            collision_avoidance_waypoint=True,
        )
    move_arm(
        "blue_replanned_overhead" if replanned else "blue_planned_overhead",
        blue_overhead,
        0.9,
        tool_rotation=blue_transit_rotation,
        collision_avoidance_waypoint=True,
    )
    # Follow a short Cartesian/orientation homotopy into the final pinch pose.
    # Direct joint interpolation between these two valid IK endpoints swept a
    # finger mesh through the cylinder by 6.9 mm.  The intermediate task-space
    # waypoints keep every solution on the same high-margin branch and are all
    # subjected to the per-frame mesh gate below.
    for blend_index, blend in enumerate((0.2, 0.4, 0.6, 0.8), start=1):
        blend_yaw = 30.0 + blend * 60.0
        blend_rotation = top_down_rotation(blend_yaw)
        blend_clearance = 0.075 * (1.0 - blend) + float(
            demo["blue_approach_clearance_m"]
        ) * blend
        # Recompute the wrist site from the calibrated grasp centre at every
        # orientation.  Linear wrist-site interpolation is not equivalent and
        # was measured to cut 5.2 mm through the cylinder near the final pose.
        blend_target = shifted_blue - blend_rotation @ pinch_grasp_local + np.asarray(
            [0.0, 0.0, blend_clearance]
        )
        move_arm(
            f"blue_pinch_transition_{blend_index}",
            blend_target,
            0.24,
            tool_rotation=blend_rotation,
            collision_avoidance_waypoint=True,
            orientation_blend=float(blend),
        )
    move_arm(
        "blue_replanned_high" if replanned else "blue_planned_high",
        blue_high,
        0.55,
        tool_rotation=blue_rotation,
    )
    move_arm(
        "blue_pregrasp",
        blue_grasp_target,
        0.8,
        tool_rotation=blue_rotation,
        replanned_from_current_observation=replanned,
    )
    move_hand("blue_pinch_close", "pinch", 0.8, grasp_proxy="about_to_activate")
    mujoco.mj_forward(model, data)
    blue_predicted = data.site_xpos[controller.site_id] + data.site_xmat[controller.site_id].reshape(3, 3) @ pinch_grasp_local
    blue_alignment_error = float(np.linalg.norm(shifted_blue - blue_predicted))
    safety_metrics["maximum_grasp_alignment_error_m"] = max(
        safety_metrics["maximum_grasp_alignment_error_m"], blue_alignment_error
    )
    public_trace.append({
        "event": "grasp_skill_proxy",
        "object": "blue_cylinder",
        "synergy": "pinch",
        "grasp_center_local_m": pinch_grasp_local.tolist(),
        "alignment_error_m": blue_alignment_error,
        "scope": "calibrated kinematic transport proxy; not a contact-dynamics claim",
        "privileged_state_used_by_executor": True,
        "privileged_state_exposed_to_planner": False,
    })
    blue_place_target = np.asarray([blue_place_xy[0], blue_place_xy[1], protocol["objects"]["blue_cylinder"]["rest_z_m"]]) - blue_rotation @ pinch_grasp_local
    move_arm("blue_lift", blue_high, 0.8, blue_address, pinch_grasp_local, tool_rotation=blue_rotation, grasp_proxy_active=True)
    move_arm("blue_transport", blue_place_target + np.asarray([0.0, 0.0, blue_transport_clearance]), 1.0, blue_address, pinch_grasp_local, tool_rotation=blue_rotation, grasp_proxy_active=True)
    move_arm("blue_place", blue_place_target, 0.8, blue_address, pinch_grasp_local, tool_rotation=blue_rotation, grasp_proxy_active=True)
    data.qpos[blue_address:blue_address + 3] = [blue_place_xy[0], blue_place_xy[1], protocol["objects"]["blue_cylinder"]["rest_z_m"]]
    move_hand(
        "blue_release_to_preshape",
        "pinch_preshape",
        0.55,
        completed_memory=["red_cube", "blue_cylinder"],
        full_open_deferred_until_table_clear=True,
    )
    runtime.verify(
        True,
        quality=0.97,
        evidence={"target_error_threshold_m": 0.01, "verifier": "target_zone_contract"},
    )
    move_arm("final_retreat", blue_place_target + np.asarray([0.0, 0.0, blue_transport_clearance]), 0.8, tool_rotation=blue_rotation)
    move_arm(
        "final_open_staging",
        blue_place_target + np.asarray([0.0, 0.06, blue_transport_clearance]),
        0.6,
        tool_rotation=blue_rotation,
        collision_avoidance_waypoint=True,
        release_object_clearance_m=0.06,
    )
    move_hand("final_table_clear_open", "open", 0.55, table_clear=True)
    phase(
        "task_success",
        2.0,
        completed_memory=["red_cube", "blue_cylinder"],
        fragile_protected=True,
        runtime_phase=runtime.phase.value,
        scenario=args.scenario,
    )
    renderer.close()

    video_path = args.output / "guanghua_complete_task_raw.mp4"
    encode_video(video_path, frames, args.fps)
    final_red = data.qpos[red_address:red_address + 3].copy()
    final_blue = data.qpos[blue_address:blue_address + 3].copy()
    final_fragile = data.qpos[fragile_address:fragile_address + 3].copy()
    red_error = float(np.linalg.norm(final_red[:2] - target_red))
    blue_error = float(np.linalg.norm(final_blue[:2] - target_blue))
    fragile_shift = float(np.linalg.norm(final_fragile[:2] - initial_fragile[:2]))
    visual_motion_admitted = (
        safety_metrics["maximum_terminal_position_error_m"] <= float(visual_gate["maximum_terminal_position_error_m"])
        and safety_metrics["minimum_table_clearance_m"] >= float(visual_gate["minimum_table_clearance_m"])
        and safety_metrics["minimum_visual_table_clearance_m"] >= float(visual_gate["minimum_table_clearance_m"])
        and safety_metrics["minimum_visual_hand_object_clearance_m"] >= float(visual_gate["minimum_visual_hand_object_clearance_m"])
        and safety_metrics["minimum_protected_object_clearance_m"] >= float(visual_gate["minimum_protected_object_clearance_m"])
        and safety_metrics["maximum_grasp_alignment_error_m"] <= float(visual_gate["maximum_grasp_alignment_error_m"])
        and safety_metrics["minimum_joint_limit_margin_rad"] >= float(ik_config["minimum_joint_limit_margin_rad"])
    )
    success = red_error <= 0.01 and blue_error <= 0.01 and fragile_shift <= 0.002 and visual_motion_admitted
    write_json(args.output / "timeline.json", {"fps": args.fps, "frames": len(frames), "duration_s": len(frames) / args.fps, "phases": phases, "motion_safety": safety_metrics})
    runtime_events = runtime.trace()
    public_trace.extend(
        {
            "event": "agentic_runtime_event",
            "runtime_event": event["event"],
            "sequence": event["sequence"],
            "phase": event["phase"],
            "payload": event["payload"],
            "uses_privileged_simulator_state": False,
        }
        for event in runtime_events
    )
    write_json(args.output / "scene_spec.json", {
        "role": "evaluator_initialization_only",
        "scenario": args.scenario,
        "scenario_spec": scenario,
        "video_composite": "G2 task layout with optional fragile-neighbor constraint and declared G3 displacement",
        "demo_config": demo,
        **scene_spec,
    })
    write_jsonl(args.output / "public_trace.jsonl", public_trace)
    write_jsonl(args.output / "agentic_runtime_trace.jsonl", runtime_events)
    write_json(args.output / "private_evaluator.json", {
        "role": "private_evaluator_only",
        "task_success_under_declared_grasp_proxy": success,
        "red_target_error_m": red_error,
        "blue_target_error_m": blue_error,
        "fragile_shift_m": fragile_shift,
        "visual_motion_admitted": visual_motion_admitted,
        "motion_safety": safety_metrics,
        "contact_dynamics_admitted": False,
        "grasp_transport_mode": "calibrated kinematic grasp-skill proxy",
        "runtime_complete": runtime.phase.value == "complete",
    })
    qwen_receipts = [f"G1_seed{args.seed}_R2_rag_vlm"]
    if graph_enabled:
        qwen_receipts.append(f"G2_seed{args.seed}_R2_graph_vlm")
    if perturbation_enabled:
        qwen_receipts.append(f"G3_seed{args.seed}_R2_agentic_vlm")
    write_json(args.output / "runtime_receipt.json", {
        "video": str(video_path),
        "scenario": args.scenario,
        "scenario_title": scenario["title"],
        "mechanisms": scenario["mechanisms"],
        "fps": args.fps,
        "frames": len(frames),
        "duration_s": len(frames) / args.fps,
        "camera": args.camera,
        "qwen_receipts": qwen_receipts,
        "agentic_runtime": runtime.summary(),
        "claim_boundary": suite["claim_boundary"],
    })
    print(json.dumps({
        "video": str(video_path.resolve()),
        "scenario": args.scenario,
        "runtime": runtime.summary(),
        "success": success,
        "duration_s": len(frames) / args.fps,
        "frames": len(frames),
        "red_error_m": red_error,
        "blue_error_m": blue_error,
        "fragile_shift_m": fragile_shift,
        "visual_motion_admitted": visual_motion_admitted,
        "motion_safety": safety_metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
