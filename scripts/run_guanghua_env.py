#!/usr/bin/env python3
"""Load, simulate, render, or interactively inspect the Guanghua MuJoCo scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MJCF = (
    PROJECT_ROOT
    / "assets"
    / "guanghua_hand_env"
    / "mjcf"
    / "guanghua_hand_env.xml"
)
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "guanghua_experiment_protocol.json"


def initialize_position_targets(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Hold every position-controlled joint at its current configuration."""
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            continue
        qpos_address = int(model.jnt_qposadr[joint_id])
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            data.ctrl[actuator_id] = data.qpos[qpos_address]


def body_position(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> list[float]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return np.asarray(data.xpos[body_id]).round(6).tolist()


def configure_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    protocol_path: Path,
    scene_id: str,
    seed: int,
) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    try:
        scene = protocol["scene_families"][scene_id]
    except KeyError as exc:
        choices = ", ".join(sorted(protocol.get("scene_families", {})))
        raise ValueError(f"unknown scene {scene_id!r}; choose one of {choices}") from exc

    rng = np.random.default_rng(seed)
    xy_jitter = float(scene.get("xy_jitter_m", 0.0))
    yaw_jitter = np.deg2rad(float(scene.get("yaw_jitter_deg", 0.0)))
    realized: dict[str, object] = {}
    for object_name, base_xy in scene["layout_xy_m"].items():
        object_spec = protocol["objects"][object_name]
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, object_spec["free_joint"]
        )
        if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError(f"scene object {object_name} has no free joint")
        qpos_address = int(model.jnt_qposadr[joint_id])
        jitter = rng.uniform(-xy_jitter, xy_jitter, size=2)
        xy = np.asarray(base_xy, dtype=np.float64) + jitter
        yaw = float(rng.uniform(-yaw_jitter, yaw_jitter))
        data.qpos[qpos_address : qpos_address + 3] = [
            float(xy[0]),
            float(xy[1]),
            float(object_spec["rest_z_m"]),
        ]
        data.qpos[qpos_address + 3 : qpos_address + 7] = [
            np.cos(yaw / 2.0),
            0.0,
            0.0,
            np.sin(yaw / 2.0),
        ]
        realized[object_name] = {
            "xy_m": xy.round(6).tolist(),
            "yaw_rad": round(yaw, 6),
        }

    if "friction_scale_range" in scene:
        low, high = (float(value) for value in scene["friction_scale_range"])
        scale = float(rng.uniform(low, high))
        for geom_name in ("cube_col", "blue_cylinder_col", "fragile_proxy_col"):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            model.geom_friction[geom_id] *= scale
        realized["friction_scale"] = round(scale, 6)
    if "mass_scale_range" in scene:
        low, high = (float(value) for value in scene["mass_scale_range"])
        scale = float(rng.uniform(low, high))
        for body_name in ("cube", "blue_cylinder", "fragile_proxy"):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            model.body_mass[body_id] *= scale
            model.body_inertia[body_id] *= scale
        realized["mass_scale"] = round(scale, 6)

    mujoco.mj_forward(model, data)
    return {
        "scene_id": scene_id,
        "scene_name": scene["name"],
        "seed": seed,
        "objects": realized,
        "perturbation": scene.get("perturbation"),
    }


def apply_declared_perturbation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    protocol_path: Path,
    scene_id: str,
) -> dict[str, object] | None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    perturbation = protocol["scene_families"][scene_id].get("perturbation")
    if perturbation is None:
        return None
    object_spec = protocol["objects"][perturbation["object"]]
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, object_spec["free_joint"])
    qpos_address = int(model.jnt_qposadr[joint_id])
    delta = np.asarray(perturbation["delta_xy_m"], dtype=np.float64)
    data.qpos[qpos_address : qpos_address + 2] += delta
    mujoco.mj_forward(model, data)
    return {
        "role": "evaluator_only",
        "object": perturbation["object"],
        "delta_xy_m": delta.tolist(),
    }


def run_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    duration: float,
) -> None:
    steps = max(0, round(duration / model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)


def render_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    camera: str,
    output: Path,
    width: int,
    height: int,
) -> None:
    from PIL import Image

    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).save(output)
    finally:
        renderer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--scene", default="G0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--apply-perturbation", action="store_true")
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--camera", default="frontview")
    parser.add_argument("--render", type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--viewer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.mjcf.resolve()))
    data = mujoco.MjData(model)
    initialize_position_targets(model, data)
    scene_spec = configure_scene(
        model,
        data,
        protocol_path=args.protocol,
        scene_id=args.scene,
        seed=args.seed,
    )
    perturbation = None

    if args.viewer:
        from mujoco import viewer as mujoco_viewer

        with mujoco_viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                start = time.monotonic()
                mujoco.mj_step(model, data)
                viewer.sync()
                remaining = model.opt.timestep - (time.monotonic() - start)
                if remaining > 0:
                    time.sleep(remaining)
    else:
        if args.apply_perturbation:
            run_headless(model, data, args.duration / 2.0)
            perturbation = apply_declared_perturbation(
                model,
                data,
                protocol_path=args.protocol,
                scene_id=args.scene,
            )
            run_headless(model, data, args.duration / 2.0)
        else:
            run_headless(model, data, args.duration)

    if args.render is not None:
        render_frame(
            model,
            data,
            camera=args.camera,
            output=args.render,
            width=args.width,
            height=args.height,
        )

    summary = {
        "mjcf": str(args.mjcf.resolve()),
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "nbody": model.nbody,
        "ngeom": model.ngeom,
        "simulation_time_s": round(float(data.time), 6),
        "contacts": int(data.ncon),
        "scene_spec": scene_spec,
        "applied_perturbation": perturbation,
        "table_position": body_position(model, data, "table"),
        "cube_position": body_position(model, data, "cube"),
        "blue_cylinder_position": body_position(model, data, "blue_cylinder"),
        "fragile_proxy_position": body_position(model, data, "fragile_proxy"),
        "robot_root_position": body_position(model, data, "robot_root"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
