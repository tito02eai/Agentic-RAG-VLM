#!/usr/bin/env python3
"""Calibrate real MuJoCo hand contacts before admitting G0-B.

This is an evaluator-only engineering sweep.  It deliberately uses simulator
body/contact state to find a numerically stable low-level contact controller;
those values are never exposed to the Agentic planner.  A winning setting must
subsequently pass the public RGB-D G0-B admission runner.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import sys
from typing import Callable

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.grasp_planning import calibrate_grasp_frame, evaluate_top_down_candidate  # noqa: E402
from scripts.guanghua_control import (  # noqa: E402
    HAND_SYNERGIES,
    RIGHT_HAND_JOINTS,
    RightArmController,
)
from scripts.run_guanghua_env import configure_scene, initialize_position_targets  # noqa: E402


MJCF = PROJECT_ROOT / "assets" / "guanghua_hand_env" / "mjcf" / "guanghua_hand_env.xml"
PROTOCOL = PROJECT_ROOT / "configs" / "guanghua_experiment_protocol.json"


@dataclass
class TrialMonitor:
    model: mujoco.MjModel
    data: mujoco.MjData
    cube_body: int
    cube_geom: int
    object_geom_name: str
    initial_xyz: np.ndarray
    last_time: float = 0.0
    reset_detected: bool = False
    nonfinite_detected: bool = False
    bilateral_steps: int = 0
    longest_bilateral_steps: int = 0
    bilateral_lift_steps: int = 0
    max_lift_m: float = 0.0
    max_xy_shift_m: float = 0.0
    max_contact_force_n: float = 0.0
    max_qvel: float = 0.0
    hand_contact_geoms: set[str] | None = None
    protected_contact: bool = False

    def __post_init__(self) -> None:
        self.hand_contact_geoms = set()

    def sample(self, *, lift_phase: bool = False) -> None:
        if self.data.time + 1e-12 < self.last_time:
            self.reset_detected = True
        self.last_time = float(self.data.time)
        finite = bool(
            np.all(np.isfinite(self.data.qpos))
            and np.all(np.isfinite(self.data.qvel))
            and np.all(np.isfinite(self.data.qacc))
        )
        self.nonfinite_detected |= not finite
        self.max_qvel = max(self.max_qvel, float(np.max(np.abs(self.data.qvel))))
        xyz = self.data.xpos[self.cube_body].copy()
        self.max_lift_m = max(self.max_lift_m, float(xyz[2] - self.initial_xyz[2]))
        self.max_xy_shift_m = max(
            self.max_xy_shift_m, float(np.linalg.norm(xyz[:2] - self.initial_xyz[:2]))
        )
        thumb = False
        opposing = False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            names = [
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
            ]
            if any(name and "fragile_proxy" in name for name in names) and any(
                name and ("_col_r" in name or "tip_col" in name or name == "palm_col_r")
                for name in names
            ):
                self.protected_contact = True
            if self.cube_geom not in pair:
                continue
            hand_name = next(
                (name for name in names if name and name != self.object_geom_name), None
            )
            if hand_name is None:
                continue
            if "_col_r" not in hand_name and "tip_col" not in hand_name and hand_name != "palm_col_r":
                continue
            self.hand_contact_geoms.add(hand_name)
            thumb |= hand_name.startswith("th_")
            opposing |= hand_name.startswith(("if_", "mf_", "rf_", "lf_"))
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            self.max_contact_force_n = max(self.max_contact_force_n, float(np.linalg.norm(force[:3])))
        if thumb and opposing:
            self.bilateral_steps += 1
            self.longest_bilateral_steps = max(self.longest_bilateral_steps, self.bilateral_steps)
            if lift_phase:
                self.bilateral_lift_steps += 1
        else:
            self.bilateral_steps = 0


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def hand_contract(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in RIGHT_HAND_JOINTS]
    )
    addresses = model.jnt_qposadr[joint_ids]
    actuator_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{name}")
            for name in RIGHT_HAND_JOINTS
        ]
    )
    return joint_ids, addresses, actuator_ids


def step_duration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    duration_s: float,
    callback: Callable[[], None],
) -> None:
    for _ in range(max(1, round(duration_s / model.opt.timestep))):
        mujoco.mj_step(model, data)
        callback()


def command_hand_fraction(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_ids: np.ndarray,
    fraction: float,
    duration_s: float,
    callback: Callable[[], None],
) -> None:
    power = np.asarray([HAND_SYNERGIES["power"][name] for name in RIGHT_HAND_JOINTS])
    start = data.ctrl[actuator_ids].copy()
    goal = fraction * power
    steps = max(1, round(duration_s / model.opt.timestep))
    for step in range(steps):
        phase = (step + 1) / steps
        smooth = phase * phase * (3.0 - 2.0 * phase)
        data.ctrl[actuator_ids] = start + smooth * (goal - start)
        mujoco.mj_step(model, data)
        callback()
    data.ctrl[actuator_ids] = goal


def warning_counts(data: mujoco.MjData) -> list[int]:
    return [int(item.number) for item in data.warning]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "output" / "g0b_contact_sweep_v1"
    )
    parser.add_argument("--quick", action="store_true", help="Use a compact smoke-test grid.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    initialize_position_targets(model, data)
    controller = RightArmController(model, data)
    hand_joint_ids, hand_addresses, hand_actuator_ids = hand_contract(model)
    original_force_ranges = model.actuator_forcerange[hand_actuator_ids].copy()
    cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_col")

    def reset_scene() -> np.ndarray:
        mujoco.mj_resetData(model, data)
        initialize_position_targets(model, data)
        configure_scene(model, data, protocol_path=PROTOCOL, scene_id="G0", seed=args.seed)
        model.actuator_forcerange[hand_actuator_ids] = original_force_ranges
        for _ in range(round(0.3 / model.opt.timestep)):
            mujoco.mj_step(model, data)
        return data.xpos[cube_body].copy()

    object_center = reset_scene()
    frame = calibrate_grasp_frame(model, data, controller, "power")
    yaw_values = (90.0,) if args.quick else (60.0, 90.0, 120.0)
    candidates = {}
    for yaw in yaw_values:
        candidates[yaw] = evaluate_top_down_candidate(
            model, data, controller, object_center, frame, yaw
        )

    if args.quick:
        grid = itertools.product((90.0,), (0.45, 0.60), (0.90,), (1.0, 2.0), (0.0, 0.004))
    else:
        grid = itertools.product(
            yaw_values,
            (0.35, 0.50, 0.65),
            (0.85, 0.95, 1.00),
            (0.75, 1.5, 3.0),
            (-0.004, 0.0, 0.004),
        )

    rows: list[dict[str, object]] = []
    for trial_index, (yaw, preclose, final_close, force_limit, depth_offset) in enumerate(grid):
        initial_xyz = reset_scene()
        candidate = candidates[yaw]
        model.actuator_forcerange[hand_actuator_ids, 0] = -force_limit
        model.actuator_forcerange[hand_actuator_ids, 1] = force_limit
        power = np.asarray([HAND_SYNERGIES["power"][name] for name in RIGHT_HAND_JOINTS])
        data.qpos[hand_addresses] = preclose * power
        data.ctrl[hand_actuator_ids] = preclose * power

        down_target = np.asarray(candidate.wrist_target_xyz_m) + np.asarray([0.0, 0.0, depth_offset])
        ready_target = down_target + np.asarray([0.0, 0.0, 0.06])
        local_x = np.asarray(candidate.local_x_world)
        local_y = np.asarray(candidate.local_y_world)
        ready = controller.solve_position(
            ready_target,
            tolerance_m=0.012,
            local_x_world=local_x,
            local_y_world=local_y,
            axis_tolerance=0.04,
        )
        down = controller.solve_position(
            down_target,
            tolerance_m=0.012,
            local_x_world=local_x,
            local_y_world=local_y,
            axis_tolerance=0.04,
        )
        lift = controller.solve_position(
            down_target + np.asarray([0.0, 0.0, 0.05]),
            tolerance_m=0.012,
            local_x_world=local_x,
            local_y_world=local_y,
            axis_tolerance=0.04,
        )
        data.qpos[controller.qpos_addresses] = ready.joint_positions
        # This is a declared episode initialization, not a physical teleport
        # during a rollout.  Retaining velocities and the constraint warm-start
        # from the imported hanging pose injects a fictitious impact at the
        # first contact step and can trigger BADQACC on the free object.
        data.qvel[:] = 0.0
        data.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(model, data)
        monitor = TrialMonitor(model, data, cube_body, cube_geom, "cube_col", initial_xyz)
        warnings_before = warning_counts(data)

        controller.execute_joint_goal(
            np.asarray(ready.joint_positions),
            duration_s=0.02,
            settle_s=0.15,
            frame_callback=monitor.sample,
            callback_interval_s=model.opt.timestep,
        )
        controller.execute_joint_goal(
            np.asarray(down.joint_positions),
            duration_s=1.2,
            settle_s=0.15,
            frame_callback=monitor.sample,
            callback_interval_s=model.opt.timestep,
        )
        command_hand_fraction(
            model,
            data,
            hand_actuator_ids,
            final_close,
            1.5,
            monitor.sample,
        )
        step_duration(model, data, 0.35, monitor.sample)
        contact_before_lift = monitor.longest_bilateral_steps
        controller.execute_joint_goal(
            np.asarray(lift.joint_positions),
            duration_s=1.5,
            settle_s=0.25,
            frame_callback=lambda: monitor.sample(lift_phase=True),
            callback_interval_s=model.opt.timestep,
        )
        warnings_after = warning_counts(data)
        final_xyz = data.xpos[cube_body].copy()
        warning_delta = [after - before for before, after in zip(warnings_before, warnings_after)]
        stable = bool(
            not monitor.reset_detected
            and not monitor.nonfinite_detected
            and not any(value > 0 for value in warning_delta)
            and monitor.max_qvel < 50.0
        )
        contact_admitted = bool(
            contact_before_lift * model.opt.timestep >= 0.15
            and monitor.bilateral_lift_steps * model.opt.timestep >= 0.10
        )
        lift_admitted = bool(
            monitor.max_lift_m >= 0.04
            and final_xyz[2] - initial_xyz[2] >= 0.035
            and monitor.max_xy_shift_m <= 0.08
        )
        success = bool(stable and contact_admitted and lift_admitted and not monitor.protected_contact)
        row = {
            "trial": trial_index,
            "yaw_deg": yaw,
            "preclose_fraction": preclose,
            "final_close_fraction": final_close,
            "force_limit_n": force_limit,
            "depth_offset_m": depth_offset,
            "success": success,
            "stable": stable,
            "contact_admitted": contact_admitted,
            "lift_admitted": lift_admitted,
            "reset_detected": monitor.reset_detected,
            "nonfinite_detected": monitor.nonfinite_detected,
            "warning_delta": warning_delta,
            "contact_before_lift_s": contact_before_lift * model.opt.timestep,
            "bilateral_lift_contact_s": monitor.bilateral_lift_steps * model.opt.timestep,
            "hand_contact_geoms": sorted(monitor.hand_contact_geoms),
            "max_contact_force_n": monitor.max_contact_force_n,
            "max_qvel": monitor.max_qvel,
            "max_lift_m": monitor.max_lift_m,
            "final_lift_m": float(final_xyz[2] - initial_xyz[2]),
            "max_xy_shift_m": monitor.max_xy_shift_m,
            "protected_contact": monitor.protected_contact,
            "initial_xyz_m": initial_xyz.tolist(),
            "final_xyz_m": final_xyz.tolist(),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    ranked = sorted(
        rows,
        key=lambda item: (
            bool(item["success"]),
            bool(item["stable"]),
            float(item["final_lift_m"]),
            float(item["contact_before_lift_s"]),
            -float(item["max_contact_force_n"]),
        ),
        reverse=True,
    )
    payload = {
        "experiment": "G0-B evaluator-only contact calibration",
        "claim_boundary": "Private simulator state is used for low-level calibration only.",
        "seed": args.seed,
        "timestep_s": model.opt.timestep,
        "grasp_frame": frame.to_dict(),
        "trials": rows,
        "success_count": sum(bool(row["success"]) for row in rows),
        "stable_count": sum(bool(row["stable"]) for row in rows),
        "best": ranked[0] if ranked else None,
    }
    write_json(args.output / "results.json", payload)
    print(json.dumps({key: payload[key] for key in ("success_count", "stable_count", "best")}, indent=2))


if __name__ == "__main__":
    main()
