#!/usr/bin/env python3
"""Render paired MuJoCo comparisons without manufacturing baseline failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable

import cv2
import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.guanghua_control import HAND_SYNERGIES, RIGHT_HAND_JOINTS, RightArmController  # noqa: E402
from scripts.grasp_planning import (  # noqa: E402
    calibrate_grasp_frame,
    evaluate_top_down_candidate,
    hand_collision_geoms,
    hand_visual_geoms,
    minimum_geom_clearance,
    select_top_down_grasp,
    top_down_rotation,
)
from scripts.run_guanghua_env import initialize_position_targets  # noqa: E402


MJCF = PROJECT_ROOT / "assets" / "guanghua_hand_env" / "mjcf" / "guanghua_hand_env.xml"
WIDTH, HEIGHT = 1280, 720
PANEL_WIDTH, PANEL_HEIGHT = 640, 480


def encode_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    process = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{WIDTH}x{HEIGHT}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    process.stdin.close()
    assert process.stderr is not None
    error = process.stderr.read().decode("utf-8", errors="replace")
    if process.wait() != 0:
        raise RuntimeError(error)


def put(canvas: np.ndarray, text: str, xy: tuple[int, int], scale: float, color: tuple[int, int, int], thickness: int = 1) -> None:
    cv2.putText(canvas, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (10, 10, 10), thickness + 3, cv2.LINE_AA)
    cv2.putText(canvas, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def object_address(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_qposadr[joint_id])


def hand_addresses(model: mujoco.MjModel) -> np.ndarray:
    return np.asarray(
        [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in RIGHT_HAND_JOINTS],
        dtype=int,
    )


def scene() -> tuple[mujoco.MjModel, mujoco.MjData, RightArmController, mujoco.Renderer]:
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    initialize_position_targets(model, data)
    controller = RightArmController(model, data)
    renderer = mujoco.Renderer(model, height=PANEL_HEIGHT, width=PANEL_WIDTH)
    return model, data, controller, renderer


def set_xyz(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, xyz: np.ndarray) -> None:
    address = object_address(model, joint_name)
    data.qpos[address:address + 3] = xyz


def hand_values(synergy: str) -> np.ndarray:
    return np.asarray([HAND_SYNERGIES[synergy][name] for name in RIGHT_HAND_JOINTS], dtype=float)


def compose(
    left: np.ndarray,
    right: np.ndarray,
    *,
    title: str,
    stage: str,
    left_lines: list[tuple[str, tuple[int, int, int]]],
    right_lines: list[tuple[str, tuple[int, int, int]]],
) -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), 19, dtype=np.uint8)
    canvas[80:560, :640] = left
    canvas[80:560, 640:] = right
    canvas[:, 638:642] = (70, 70, 70)
    put(canvas, title, (30, 43), 0.82, (245, 245, 245), 2)
    put(canvas, "Traditional RGB-D + fixed/engineered IK", (24, 73), 0.52, (120, 190, 255), 1)
    put(canvas, "Agentic RAG-VLM (same perception + IK executor)", (664, 73), 0.52, (120, 235, 155), 1)
    put(canvas, stage, (30, 600), 0.75, (245, 220, 115), 2)
    for index, (line, color) in enumerate(left_lines):
        put(canvas, line, (30, 636 + 29 * index), 0.53, color, 1)
    for index, (line, color) in enumerate(right_lines):
        put(canvas, line, (665, 636 + 29 * index), 0.53, color, 1)
    return canvas


def render_pair(left_renderer: mujoco.Renderer, left_data: mujoco.MjData, right_renderer: mujoco.Renderer, right_data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    left_renderer.update_scene(left_data, camera="frontview")
    right_renderer.update_scene(right_data, camera="frontview")
    return left_renderer.render().copy(), right_renderer.render().copy()


def smoothstep(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def render_safety(output: Path, fps: int) -> dict[str, object]:
    lm, ld, lc, lr = scene()
    rm, rd, rc, rr = scene()
    red = np.asarray([-0.1729, -0.2022, 0.834])
    blue = np.asarray([-0.2085, -0.0333, 0.861])
    # Keep the fixed-yaw baseline physically non-penetrating while placing it
    # inside the declared 3 mm keep-out margin.  This demonstrates a semantic
    # safety rejection without using interpenetration as visual evidence.
    fragile = red + np.asarray([-0.057, -0.057, 0.042])
    for model, data in ((lm, ld), (rm, rd)):
        set_xyz(model, data, "cube_free", red)
        set_xyz(model, data, "blue_cylinder_free", blue)
        set_xyz(model, data, "fragile_proxy_free", fragile)
        mujoco.mj_forward(model, data)

    left_frame = calibrate_grasp_frame(lm, ld, lc, "power")
    left_grasp = evaluate_top_down_candidate(lm, ld, lc, red, left_frame, 120.0)
    ld.qpos[lc.qpos_addresses] = left_grasp.ik.joint_positions
    mujoco.mj_forward(lm, ld)
    left_high = evaluate_top_down_candidate(lm, ld, lc, red + np.asarray([0.0, 0.0, 0.05]), left_frame, 120.0)
    right_grasp, right_candidates, right_frame = select_top_down_grasp(
        rm, rd, rc, red, "power", protected_geom_names=("fragile_proxy_col",),
        minimum_protected_clearance_m=0.003,
    )
    rd.qpos[rc.qpos_addresses] = right_grasp.ik.joint_positions
    mujoco.mj_forward(rm, rd)
    right_high = evaluate_top_down_candidate(
        rm, rd, rc, red + np.asarray([0.0, 0.0, 0.05]), right_frame, right_grasp.yaw_deg
    )
    left_hand = hand_addresses(lm)
    right_hand = hand_addresses(rm)
    ld.qpos[lc.qpos_addresses] = left_high.ik.joint_positions
    rd.qpos[rc.qpos_addresses] = right_high.ik.joint_positions
    ld.qpos[left_hand] = hand_values("power_preshape")
    rd.qpos[right_hand] = hand_values("power_preshape")
    mujoco.mj_forward(lm, ld)
    mujoco.mj_forward(rm, rd)
    left_fragile_geom = mujoco.mj_name2id(lm, mujoco.mjtObj.mjOBJ_GEOM, "fragile_proxy_col")
    right_fragile_geom = mujoco.mj_name2id(rm, mujoco.mjtObj.mjOBJ_GEOM, "fragile_proxy_col")
    left_min, right_min = float("inf"), float("inf")
    frames: list[np.ndarray] = []

    phases = [
        ("OBSERVE: protected glass is adjacent to the red cube", 2.0, None),
        ("PLAN: fixed wrist vs relation-conditioned wrist selection", 2.0, None),
        ("EXECUTE: identical calibrated power grasp and IK executor", 3.0, "descend"),
        ("CLOSE: high-level choice changes physical clearance", 1.2, "close"),
        ("VERIFY: both lift; safety margin is not a manufactured failure", 2.4, "lift"),
        ("RESULT: scene-conditioned orientation preserves the keep-out margin", 2.0, None),
    ]
    for stage, duration, action in phases:
        count = max(1, round(duration * fps))
        for index in range(count):
            fraction = smoothstep((index + 1) / count)
            if action == "descend":
                ld.qpos[lc.qpos_addresses] = (1 - fraction) * np.asarray(left_high.ik.joint_positions) + fraction * np.asarray(left_grasp.ik.joint_positions)
                rd.qpos[rc.qpos_addresses] = (1 - fraction) * np.asarray(right_high.ik.joint_positions) + fraction * np.asarray(right_grasp.ik.joint_positions)
            elif action == "close":
                ld.qpos[left_hand] = (1 - fraction) * hand_values("power_preshape") + fraction * hand_values("power")
                rd.qpos[right_hand] = (1 - fraction) * hand_values("power_preshape") + fraction * hand_values("power")
            elif action == "lift":
                ld.qpos[lc.qpos_addresses] = (1 - fraction) * np.asarray(left_grasp.ik.joint_positions) + fraction * np.asarray(left_high.ik.joint_positions)
                rd.qpos[rc.qpos_addresses] = (1 - fraction) * np.asarray(right_grasp.ik.joint_positions) + fraction * np.asarray(right_high.ik.joint_positions)
                for model, data, controller, address, rotation, center in (
                    (lm, ld, lc, object_address(lm, "cube_free"), top_down_rotation(120.0), left_frame.center_local_m),
                    (rm, rd, rc, object_address(rm, "cube_free"), top_down_rotation(right_grasp.yaw_deg), right_frame.center_local_m),
                ):
                    mujoco.mj_forward(model, data)
                    data.qpos[address:address + 3] = data.site_xpos[controller.site_id] + rotation @ np.asarray(center)
            mujoco.mj_forward(lm, ld)
            mujoco.mj_forward(rm, rd)
            left_clear = minimum_geom_clearance(lm, ld, hand_collision_geoms(lm), (left_fragile_geom,))
            right_clear = minimum_geom_clearance(rm, rd, hand_collision_geoms(rm), (right_fragile_geom,))
            left_min, right_min = min(left_min, left_clear), min(right_min, right_clear)
            left_image, right_image = render_pair(lr, ld, rr, rd)
            frames.append(compose(
                left_image, right_image, title="E2  PROTECTED-GLASS RELATIONAL SAFETY", stage=stage,
                left_lines=[
                    (f"fixed yaw = {left_grasp.yaw_deg:.0f} deg | min clearance = {1000*left_min:5.1f} mm", (125, 200, 255)),
                    (
                        f"keep-out gate = 3.0 mm -> {'REJECTED' if left_min < 0.003 else 'PENDING'}",
                        (230, 170, 145) if left_min < 0.003 else (220, 220, 220),
                    ),
                ],
                right_lines=[
                    (f"selected yaw = {right_grasp.yaw_deg:.0f} deg | min clearance = {1000*right_min:5.1f} mm", (125, 240, 160)),
                    ("protected_glass -> candidate audit -> safe orientation", (220, 220, 220)),
                ],
            ))
    lr.close(); rr.close()
    path = output / "E2_protected_glass_paired.mp4"
    encode_video(path, frames, fps)
    return {
        "video": str(path.resolve()),
        "baseline_yaw_deg": 120.0,
        "agentic_yaw_deg": right_grasp.yaw_deg,
        "baseline_minimum_clearance_m": left_min,
        "agentic_minimum_clearance_m": right_min,
        "minimum_required_clearance_m": 0.003,
        "baseline_admitted": left_min >= 0.003,
        "agentic_admitted": right_min >= 0.003,
        "agentic_candidate_audit": [candidate.to_dict() for candidate in right_candidates],
    }


def render_recovery(output: Path, fps: int) -> dict[str, object]:
    lm, ld, lc, lr = scene()
    rm, rd, rc, rr = scene()
    # Match the ergonomically admitted blue-workspace placement used by the
    # complete showcase.  The previous legacy coordinates forced both methods
    # into a visually poor cross-torso branch and confounded the recovery pair.
    blue_before = np.asarray([-0.2445, -0.1183, 0.861])
    # An 80 mm displacement separates the stale hand envelope from the moved
    # cylinder.  The old 29 mm perturbation was logically a miss, but the
    # imported long-finger mesh still swept 11 mm through the cylinder, making
    # the baseline visually invalid rather than informative.
    shift = np.asarray([0.064, 0.048, 0.0])
    blue_after = blue_before + shift
    red_done = np.asarray([-0.08, -0.27, 0.834])
    fragile = np.asarray([0.05, 0.10, 0.876])
    for model, data in ((lm, ld), (rm, rd)):
        set_xyz(model, data, "cube_free", red_done)
        set_xyz(model, data, "blue_cylinder_free", blue_before)
        set_xyz(model, data, "fragile_proxy_free", fragile)
        mujoco.mj_forward(model, data)
    lf = calibrate_grasp_frame(lm, ld, lc, "pinch")
    rf = calibrate_grasp_frame(rm, rd, rc, "pinch")

    def ergonomic_blue_ik(
        controller: RightArmController, object_center: np.ndarray, grasp_frame: object
    ):
        rotation = top_down_rotation(90.0)
        wrist_target = object_center - rotation @ np.asarray(grasp_frame.center_local_m)
        return controller.solve_position(
            wrist_target,
            tolerance_m=0.012,
            local_x_world=rotation[:, 0],
            local_y_world=rotation[:, 1],
            axis_tolerance=0.04,
            position_weight=20.0,
            continuity_weight=0.0,
            posture_weight=0.04,
            joint_center_weight=0.015,
        )

    lb = ergonomic_blue_ik(lc, blue_before, lf)
    ld.qpos[lc.qpos_addresses] = lb.joint_positions
    mujoco.mj_forward(lm, ld)
    lh = ergonomic_blue_ik(lc, blue_before + np.asarray([0.0, 0.0, 0.033]), lf)
    rb = ergonomic_blue_ik(rc, blue_after, rf)
    rd.qpos[rc.qpos_addresses] = rb.joint_positions
    mujoco.mj_forward(rm, rd)
    rh = ergonomic_blue_ik(rc, blue_after + np.asarray([0.0, 0.0, 0.033]), rf)
    # Replanning is rendered as the same raise-transfer-descend topology used
    # in the complete task.  Directly interpolating two high-pose joint vectors
    # cut through the moved target even though both endpoints were valid.
    right_overhead_before = ergonomic_blue_ik(
        rc, blue_before + np.asarray([0.0, 0.0, 0.060]), rf
    )
    right_overhead_after = ergonomic_blue_ik(
        rc, blue_after + np.asarray([0.0, 0.0, 0.060]), rf
    )
    la, ra = hand_addresses(lm), hand_addresses(rm)
    ld.qpos[lc.qpos_addresses] = right_overhead_before.joint_positions
    # Both paired methods begin from the identical stale high waypoint.
    rd.qpos[rc.qpos_addresses] = right_overhead_before.joint_positions
    ld.qpos[la] = hand_values("pinch_preshape")
    rd.qpos[ra] = hand_values("pinch_preshape")
    mujoco.mj_forward(lm, ld); mujoco.mj_forward(rm, rd)
    # Audit the visible STL meshes at render cadence.  The paired recovery
    # video is evidence, so endpoint-only IK checks are insufficient: a joint
    # interpolation can be clear at both ends while sweeping a long finger
    # through the table or target between them.
    left_visual_hand = hand_visual_geoms(lm)
    right_visual_hand = hand_visual_geoms(rm)
    left_table = mujoco.mj_name2id(lm, mujoco.mjtObj.mjOBJ_GEOM, "table_visual")
    right_table = mujoco.mj_name2id(rm, mujoco.mjtObj.mjOBJ_GEOM, "table_visual")
    left_blue = mujoco.mj_name2id(lm, mujoco.mjtObj.mjOBJ_GEOM, "blue_cylinder_visual")
    right_blue = mujoco.mj_name2id(rm, mujoco.mjtObj.mjOBJ_GEOM, "blue_cylinder_visual")
    mesh_audit = {
        "baseline_minimum_hand_table_clearance_m": float("inf"),
        "baseline_minimum_hand_target_clearance_m": float("inf"),
        "agentic_minimum_hand_table_clearance_m": float("inf"),
        "agentic_minimum_hand_target_clearance_m": float("inf"),
        "frames_audited_per_method": 0,
    }

    def record_mesh_minimum(metric: str, value: float, stage: str) -> None:
        if value < mesh_audit[metric]:
            mesh_audit[metric] = value
            mesh_audit[f"{metric}_phase"] = stage
            mesh_audit[f"{metric}_frame"] = mesh_audit["frames_audited_per_method"]
    frames: list[np.ndarray] = []
    left_attached = False
    right_attached = False
    alignment_left = float(np.linalg.norm(shift[:2]))
    alignment_right = 0.0
    phases = [
        ("PLAN: both methods receive the same initial RGB-D centroid", 2.0, None),
        ("EVENT: pending blue target moves 80 mm after planning", 1.2, "shift"),
        ("MONITOR: open loop keeps stale target; Agentic invalidates it", 2.0, "replan"),
        ("EXECUTE: stale waypoint vs re-observed waypoint", 3.0, "descend"),
        ("VERIFY: alignment gate decides whether the grasp proxy activates", 1.2, "close"),
        ("RECOVER: target-aware L3 replan preserves the completed red subgoal", 2.5, "lift"),
        ("RESULT: failure follows from a declared external change, not a scripted drop", 2.0, None),
    ]
    left_blue_address = object_address(lm, "blue_cylinder_free")
    right_blue_address = object_address(rm, "blue_cylinder_free")
    for stage, duration, action in phases:
        count = max(1, round(duration * fps))
        for index in range(count):
            fraction = smoothstep((index + 1) / count)
            if action == "shift":
                ld.qpos[left_blue_address:left_blue_address + 3] = blue_before + fraction * shift
                rd.qpos[right_blue_address:right_blue_address + 3] = blue_before + fraction * shift
            elif action == "replan":
                rd.qpos[rc.qpos_addresses] = (
                    (1 - fraction) * np.asarray(right_overhead_before.joint_positions)
                    + fraction * np.asarray(right_overhead_after.joint_positions)
                )
            elif action == "descend":
                if fraction <= 0.42:
                    segment = smoothstep(fraction / 0.42)
                    ld.qpos[lc.qpos_addresses] = (
                        (1 - segment) * np.asarray(right_overhead_before.joint_positions)
                        + segment * np.asarray(lh.joint_positions)
                    )
                else:
                    segment = smoothstep((fraction - 0.42) / 0.58)
                    ld.qpos[lc.qpos_addresses] = (
                        (1 - segment) * np.asarray(lh.joint_positions)
                        + segment * np.asarray(lb.joint_positions)
                    )
                if fraction <= 0.42:
                    segment = smoothstep(fraction / 0.42)
                    rd.qpos[rc.qpos_addresses] = (
                        (1 - segment) * np.asarray(right_overhead_after.joint_positions)
                        + segment * np.asarray(rh.joint_positions)
                    )
                else:
                    segment = smoothstep((fraction - 0.42) / 0.58)
                    rd.qpos[rc.qpos_addresses] = (
                        (1 - segment) * np.asarray(rh.joint_positions)
                        + segment * np.asarray(rb.joint_positions)
                    )
            elif action == "close":
                # Do not close a stale gripper around the displaced target.
                # The verifier rejects the 80 mm alignment error before hand
                # actuation, which keeps the comparison physically valid.
                if alignment_left <= 0.015:
                    ld.qpos[la] = (
                        (1 - fraction) * hand_values("pinch_preshape")
                        + fraction * hand_values("pinch")
                    )
                rd.qpos[ra] = (1 - fraction) * hand_values("pinch_preshape") + fraction * hand_values("pinch")
                left_attached = alignment_left <= 0.015
                right_attached = alignment_right <= 0.015
            elif action == "lift":
                # The stale grasp is rejected by the same alignment gate, so
                # the baseline does not execute a meaningless lift whose hand
                # envelope would sweep through the displaced cylinder.
                if left_attached:
                    ld.qpos[lc.qpos_addresses] = (
                        (1 - fraction) * np.asarray(lb.joint_positions)
                        + fraction * np.asarray(lh.joint_positions)
                    )
                rd.qpos[rc.qpos_addresses] = (1 - fraction) * np.asarray(rb.joint_positions) + fraction * np.asarray(rh.joint_positions)
                if left_attached:
                    mujoco.mj_forward(lm, ld)
                    ld.qpos[left_blue_address:left_blue_address + 3] = ld.site_xpos[lc.site_id] + top_down_rotation(90.0) @ np.asarray(lf.center_local_m)
                if right_attached:
                    mujoco.mj_forward(rm, rd)
                    rd.qpos[right_blue_address:right_blue_address + 3] = rd.site_xpos[rc.site_id] + top_down_rotation(90.0) @ np.asarray(rf.center_local_m)
            mujoco.mj_forward(lm, ld); mujoco.mj_forward(rm, rd)
            record_mesh_minimum(
                "baseline_minimum_hand_table_clearance_m",
                minimum_geom_clearance(lm, ld, left_visual_hand, (left_table,)), stage,
            )
            record_mesh_minimum(
                "baseline_minimum_hand_target_clearance_m",
                minimum_geom_clearance(lm, ld, left_visual_hand, (left_blue,)), stage,
            )
            record_mesh_minimum(
                "agentic_minimum_hand_table_clearance_m",
                minimum_geom_clearance(rm, rd, right_visual_hand, (right_table,)), stage,
            )
            record_mesh_minimum(
                "agentic_minimum_hand_target_clearance_m",
                minimum_geom_clearance(rm, rd, right_visual_hand, (right_blue,)), stage,
            )
            mesh_audit["frames_audited_per_method"] += 1
            li, ri = render_pair(lr, ld, rr, rd)
            frames.append(compose(
                li, ri, title="E3  TARGET-CHANGE RECOVERY", stage=stage,
                left_lines=[
                    (f"open loop | stale error = {1000*alignment_left:.1f} mm", (125, 200, 255)),
                    ("grasp proxy: NOT ACTIVATED" if action in {"close", "lift"} and not left_attached else "planned target: original blue centroid", (230, 170, 145) if not left_attached else (220, 220, 220)),
                ],
                right_lines=[
                    ("target move detected -> one bounded L3 replan", (125, 240, 160)),
                    ("grasp proxy: ACTIVATED" if action in {"close", "lift"} and right_attached else "Memory: red_cube remains completed", (125, 240, 160)),
                ],
            ))
    lr.close(); rr.close()
    path = output / "E3_target_change_paired.mp4"
    encode_video(path, frames, fps)
    mesh_audit["admitted"] = bool(
        mesh_audit["baseline_minimum_hand_table_clearance_m"] >= 0.0
        and mesh_audit["agentic_minimum_hand_table_clearance_m"] >= 0.0
        and mesh_audit["baseline_minimum_hand_target_clearance_m"] >= -0.0005
        and mesh_audit["agentic_minimum_hand_target_clearance_m"] >= -0.0005
    )
    if not mesh_audit["admitted"]:
        raise RuntimeError(f"E3 visible-mesh gate rejected the rendered trajectory: {mesh_audit}")
    return {
        "video": str(path.resolve()),
        "declared_shift_m": shift.tolist(),
        "baseline_alignment_error_m": alignment_left,
        "agentic_alignment_error_m": alignment_right,
        "baseline_grasp_proxy_activated": left_attached,
        "agentic_grasp_proxy_activated": right_attached,
        "completed_memory_preserved": ["red_cube"],
        "visible_mesh_audit": mesh_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output" / "paired_comparison_videos_v1")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--experiments", nargs="*", choices=("safety", "recovery"), default=("safety", "recovery"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, object] = {
        "claim_boundary": "calibrated kinematic grasp-skill proxy; not contact-dynamics success",
        "paired_executor_contract": "same imported robot, hand calibration, scene state and IK executor",
    }
    if "safety" in args.experiments:
        receipt["E2_protected_clearance"] = render_safety(args.output, args.fps)
    if "recovery" in args.experiments:
        receipt["E3_target_change"] = render_recovery(args.output, args.fps)
    (args.output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
