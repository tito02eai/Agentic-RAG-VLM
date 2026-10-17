#!/usr/bin/env python3
"""Run the G0 RGB-D-to-pregrasp admission rollout with auditable outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import mujoco
import numpy as np

from guanghua_control import RightArmController, command_hand_synergy
from guanghua_perception import capture_rgbd, estimate_colored_objects, save_rgbd
from run_guanghua_env import configure_scene, initialize_position_targets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MJCF = PROJECT_ROOT / "assets" / "guanghua_hand_env" / "mjcf" / "guanghua_hand_env.xml"
PROTOCOL = PROJECT_ROOT / "configs" / "guanghua_experiment_protocol.json"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_mp4(path: Path, frames: list[np.ndarray], *, fps: int) -> None:
    if not frames:
        raise ValueError("cannot encode an empty video")
    height, width = frames[0].shape[:2]
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    process.stdin.close()
    assert process.stderr is not None
    error_output = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {error_output}")


def _contact_pairs(model: mujoco.MjModel, data: mujoco.MjData) -> list[list[str | None]]:
    pairs = []
    for contact in data.contact:
        pairs.append(
            [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1),
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2),
            ]
        )
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "g0_admission_seed7",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    initialize_position_targets(model, data)
    scene_spec = configure_scene(
        model,
        data,
        protocol_path=PROTOCOL,
        scene_id="G0",
        seed=args.seed,
    )
    for _ in range(round(0.25 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    public_trace: list[dict[str, object]] = []
    start = time.perf_counter()
    rgb, depth = capture_rgbd(model, data, camera="agentview")
    observation_files = save_rgbd(rgb, depth, args.output, "initial_agentview")
    rest_z = {name: float(spec["rest_z_m"]) for name, spec in protocol["objects"].items()}
    estimates = estimate_colored_objects(
        model,
        data,
        rgb,
        depth,
        rest_z_by_object=rest_z,
        camera="agentview",
    )
    perception_latency = time.perf_counter() - start
    if "red_cube" not in estimates:
        raise RuntimeError("RGB-D perception did not detect the red cube")
    public_trace.append(
        {
            "event": "scene_observation",
            "camera": "agentview",
            "files": observation_files,
            "object_estimates": {name: estimate.to_dict() for name, estimate in estimates.items()},
            "latency_s": perception_latency,
            "uses_privileged_simulator_state": False,
        }
    )

    hand_result = command_hand_synergy(model, data, "open", duration_s=0.25)
    estimate = np.asarray(estimates["red_cube"].center_xyz_m)
    pregrasp_target = np.asarray([estimate[0], estimate[1], 1.10])
    controller = RightArmController(model, data)
    video_renderer = mujoco.Renderer(model, height=480, width=640)
    video_frames: list[np.ndarray] = []

    def record_front_frame() -> None:
        video_renderer.update_scene(data, camera="frontview")
        video_frames.append(video_renderer.render().copy())

    record_front_frame()
    ik_start = time.perf_counter()
    # The source pose hangs below the tabletop. A direct joint interpolation
    # crosses the table edge, so route outside the right side of the fixture,
    # rise above the surface, and only then move inward.
    waypoint_targets = (
        np.asarray([-0.42, -0.50, 0.98]),
        np.asarray([-0.35, -0.50, 1.08]),
        np.asarray([-0.25, -0.45, 1.10]),
        np.asarray([estimate[0], -0.30, 1.10]),
        pregrasp_target,
    )
    solutions = []
    executions = []
    try:
        for waypoint_index, waypoint in enumerate(waypoint_targets):
            try:
                solution = controller.solve_position(
                    waypoint,
                    tolerance_m=0.025,
                    local_x_world=np.asarray([0.0, 0.0, 1.0]),
                    axis_tolerance=0.04,
                )
            except RuntimeError as error:
                raise RuntimeError(
                    f"safe-corridor waypoint {waypoint_index} {waypoint.tolist()} failed: {error}"
                ) from error
            solutions.append(solution)
            executions.append(
                controller.execute_joint_goal(
                    np.asarray(solution.joint_positions),
                    duration_s=0.8,
                    settle_s=0.1,
                    frame_callback=record_front_frame,
                    callback_interval_s=0.05,
                )
            )
        for _ in range(10):
            record_front_frame()
    finally:
        video_renderer.close()
    ik_latency = time.perf_counter() - ik_start
    video_path = args.output / "g0a_pregrasp_front.mp4"
    video_start = time.perf_counter()
    _write_mp4(video_path, video_frames, fps=20)
    video_encoding_latency = time.perf_counter() - video_start
    public_trace.append(
        {
            "event": "skill_execution",
            "skill": "move_to_pregrasp",
            "target_source": "agentview_rgbd",
            "target_xyz_m": pregrasp_target.round(6).tolist(),
            "path_policy": "declared_fixture_safe_right_corridor",
            "diagnostic_video": str(video_path),
            "waypoint_targets_xyz_m": [target.round(6).tolist() for target in waypoint_targets],
            "ik_solutions": [item.to_dict() for item in solutions],
            "executions": executions,
            "hand": hand_result,
            "uses_privileged_simulator_state": False,
        }
    )

    final_rgb, final_depth = capture_rgbd(model, data, camera="frontview")
    final_files = save_rgbd(final_rgb, final_depth, args.output, "final_frontview")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_hand_r")
    actual_site = data.site_xpos[site_id].copy()
    final_error = float(np.linalg.norm(actual_site - pregrasp_target))
    contacts = _contact_pairs(model, data)
    fragile_contacts = [
        pair
        for pair in contacts
        if any(name and "fragile_proxy" in name for name in pair)
        and "table_collision" not in pair
    ]
    perception_errors = {}
    for object_name, estimate_item in estimates.items():
        body_name = protocol["objects"][object_name]["body"]
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        estimated_xy = np.asarray(estimate_item.center_xyz_m[:2])
        perception_errors[object_name] = float(
            np.linalg.norm(estimated_xy - data.xpos[body_id, :2])
        )
    success = bool(
        perception_errors["red_cube"] <= 0.02
        and final_error <= 0.035
        and not fragile_contacts
        and np.all(np.isfinite(data.qpos))
    )

    _write_json(
        args.output / "scene_spec.json",
        {"role": "evaluator_initialization_only", **scene_spec},
    )
    with (args.output / "public_trace.jsonl").open("w", encoding="utf-8") as handle:
        for event in public_trace:
            handle.write(json.dumps(event) + "\n")
    _write_json(
        args.output / "private_evaluator.json",
        {
            "role": "private_evaluator_only",
            "admission_success": success,
            "pregrasp_target_xyz_m": pregrasp_target.tolist(),
            "actual_site_xyz_m": actual_site.tolist(),
            "position_error_m": final_error,
            "perception_center_xy_error_m": perception_errors,
            "contact_pairs": contacts,
            "fragile_contact_pairs": fragile_contacts,
        },
    )
    _write_json(
        args.output / "runtime_receipt.json",
        {
            "perception_latency_s": perception_latency,
            "ik_latency_s": ik_latency,
            "video_encoding_latency_s": video_encoding_latency,
            "simulation_time_s": float(data.time),
            "diagnostic_video": str(video_path),
            "final_observation_files": final_files,
        },
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "admission_success": success,
                "position_error_m": final_error,
                "ik_position_error_m": solutions[-1].position_error_m,
                "fragile_contacts": len(fragile_contacts),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
