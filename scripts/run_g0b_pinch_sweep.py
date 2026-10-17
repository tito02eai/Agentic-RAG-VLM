#!/usr/bin/env python3
"""Audit a true thumb-index side pinch and 50 mm lift in MuJoCo.

The imported hand cannot safely top-down power-grasp the original 66 mm cube:
its collision aperture is smaller than the object and the open fingers reach
the tabletop.  This evaluator-only calibration recompiles the blue task object
as a slim 80 mm-tall rectangular prism with matched visual/collision geometry,
mass and inertia.  It evaluates contact stability only; a winning setting must
still pass a separate public-RGB-D admission before any physical-grasp claim.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.grasp_planning import RIGHT_TIP_GEOMS  # noqa: E402
from scripts.guanghua_control import (  # noqa: E402
    HAND_SYNERGIES,
    NATURAL_TABLETOP_POSTURE,
    RIGHT_HAND_JOINTS,
    RightArmController,
)
from scripts.run_g0b_contact_sweep import TrialMonitor, hand_contract, warning_counts  # noqa: E402
from scripts.run_guanghua_env import configure_scene, initialize_position_targets  # noqa: E402


MJCF = PROJECT_ROOT / "assets" / "guanghua_hand_env" / "mjcf" / "guanghua_hand_env.xml"
PROTOCOL = PROJECT_ROOT / "configs" / "guanghua_experiment_protocol.json"
PARTICIPATING_GEOMS = (
    "if_proximal_col_r",
    "if_distal_col_r",
    "if_tip_col",
    "th_proximal_col_r",
    "th_distal_col_r",
    "th_tip_col",
)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def side_grasp_rotation(yaw_degrees: float) -> np.ndarray:
    """Palm-side grasp: local x/y horizontal and local z points upward."""
    yaw = np.deg2rad(yaw_degrees)
    return np.asarray(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def build_calibration_model(radius: float) -> mujoco.MjModel:
    """Compile the candidate object/contact geometry before simulation.

    Editing ``MjModel.geom_size`` after compilation leaves the body's inferred
    mass and inertia at their old values.  G0-B instead compiles a true prism
    of the requested size and a deliberately compliant fingertip contact.  The
    resulting parameters can later be frozen into a dedicated public MJCF.
    """
    spec = mujoco.MjSpec.from_file(str(MJCF))
    collision = spec.geom("blue_cylinder_col")
    visual = spec.geom("blue_cylinder_visual")
    # A slim rectangular prism produces a well-defined two-face pinch and
    # avoids the redundant flat-cylinder/table contact manifold that is
    # numerically unstable in the imported hand model.
    collision.type = mujoco.mjtGeom.mjGEOM_BOX
    visual.type = mujoco.mjtGeom.mjGEOM_BOX
    collision.size = [radius, radius, 0.04]
    visual.size = [radius, radius, 0.04]
    collision.condim = 3
    # Give the calibrated object priority so its compliant normal parameters
    # are also used at the table interface; otherwise the table's 2 ms
    # time-constant dominates and destabilises this much lighter object.
    collision.priority = 1
    collision.density = 3000.0
    collision.margin = 0.0005
    collision.gap = 0.0
    collision.solref = [0.05, 1.0]
    collision.solimp = [0.8, 0.95, 0.005, 0.5, 2.0]
    for name in PARTICIPATING_GEOMS:
        geom = spec.geom(name)
        geom.condim = 3
        geom.margin = 0.0005
        geom.gap = 0.0
        geom.solref = [0.05, 1.0]
        geom.solimp = [0.8, 0.95, 0.005, 0.5, 2.0]
    free_joint = spec.joint("blue_cylinder_free")
    free_joint.armature = 0.05
    free_joint.damping = [2.0, 0.0, 0.0]
    return spec.compile()


def compensated_arm_control(controller: RightArmController, goal: np.ndarray) -> np.ndarray:
    data = controller.data
    model = controller.model
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    try:
        data.qpos[controller.qpos_addresses] = goal
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        dofs = model.jnt_dofadr[controller.joint_ids]
        torque = data.qfrc_bias[dofs] - data.qfrc_passive[dofs]
        gains = model.actuator_gainprm[controller.actuator_ids, 0]
        target = goal + torque / gains
        limited = model.actuator_ctrllimited[controller.actuator_ids].astype(bool)
        target[limited] = np.clip(
            target[limited],
            model.actuator_ctrlrange[controller.actuator_ids[limited], 0],
            model.actuator_ctrlrange[controller.actuator_ids[limited], 1],
        )
        return target
    finally:
        data.qpos[:] = saved_qpos
        data.qvel[:] = saved_qvel
        mujoco.mj_forward(model, data)


def local_tip_positions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: RightArmController,
) -> dict[str, np.ndarray]:
    origin = data.site_xpos[controller.site_id]
    rotation = data.site_xmat[controller.site_id].reshape(3, 3)
    return {
        name: rotation.T
        @ (
            data.geom_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)]
            - origin
        )
        for name in RIGHT_TIP_GEOMS
    }


def contact_snapshot(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, object]]:
    """Return the pre-step contact state used to diagnose G0-B admission.

    A physically valid rollout must not begin with millimetres of hidden
    interpenetration.  Keeping this in the result file also makes the contact
    calibration auditable rather than relying on a visually plausible frame.
    """
    rows: list[dict[str, object]] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        name1 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
        )
        name2 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
        )
        if not any(
            name
            and (
                name == "blue_cylinder_col"
                or name == "table_collision"
                or name == "palm_col_r"
                or "_col_r" in name
                or "tip_col" in name
            )
            for name in (name1, name2)
        ):
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        rows.append(
            {
                "geom1": name1,
                "geom2": name2,
                "distance_m": float(contact.dist),
                "force_n": float(np.linalg.norm(force[:3])),
            }
        )
    return rows


def command_pinch(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_ids: np.ndarray,
    final_fraction: float,
    duration_s: float,
    monitor: TrialMonitor,
) -> None:
    goal = data.ctrl[actuator_ids].copy()
    for index, name in enumerate(RIGHT_HAND_JOINTS):
        goal[index] = final_fraction * HAND_SYNERGIES["pinch"][name]
    start = data.ctrl[actuator_ids].copy()
    for step in range(max(1, round(duration_s / model.opt.timestep))):
        fraction = (step + 1) / max(1, round(duration_s / model.opt.timestep))
        smooth = fraction * fraction * (3.0 - 2.0 * fraction)
        data.ctrl[actuator_ids] = start + smooth * (goal - start)
        mujoco.mj_step(model, data)
        monitor.sample()
    data.ctrl[actuator_ids] = goal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--radius", type=float, default=0.010)
    parser.add_argument("--preshape-fraction", type=float, default=0.45)
    parser.add_argument("--pinch-fraction", type=float, default=0.60)
    parser.add_argument("--force-limit", type=float, default=0.75)
    parser.add_argument("--timestep", type=float, default=0.001)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "output" / "g0b_pinch_sweep_v1"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = build_calibration_model(args.radius)
    model.opt.timestep = args.timestep
    model.opt.solver = mujoco.mjtSolver.mjSOL_PGS
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.iterations = 100
    data = mujoco.MjData(model)
    initialize_position_targets(model, data)
    controller = RightArmController(model, data)
    # Contact phase uses a compliant joint impedance.  The imported free-space
    # position controller (kp=800, +/-100 Nm) is intentionally stiff and turns
    # sub-millimetre fingertip error into a rigid constraint loop.
    original_arm_damping = model.actuator_biasprm[controller.actuator_ids, 2].copy()
    model.actuator_gainprm[controller.actuator_ids, 0] = 200.0
    model.actuator_biasprm[controller.actuator_ids, 1] = -200.0
    model.actuator_biasprm[controller.actuator_ids, 2] = 0.5 * original_arm_damping
    model.actuator_forcerange[controller.actuator_ids, 0] = -30.0
    model.actuator_forcerange[controller.actuator_ids, 1] = 30.0
    _, hand_addresses, hand_actuator_ids = hand_contract(model)
    force_ranges = model.actuator_forcerange[hand_actuator_ids].copy()
    blue_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "blue_cylinder")
    blue_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "blue_cylinder_col")
    blue_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "blue_cylinder_free")
    blue_qpos = model.jnt_qposadr[blue_joint]
    table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_collision")
    yaw = 180.0
    rotation = side_grasp_rotation(yaw)

    def reset_scene(radius: float) -> np.ndarray:
        mujoco.mj_resetData(model, data)
        initialize_position_targets(model, data)
        configure_scene(model, data, protocol_path=PROTOCOL, scene_id="G2", seed=args.seed)
        # The public protocol z value belongs to the original 120 mm-tall
        # object.  This calibrated 80 mm prism must start on the tabletop, not
        # 20 mm above it with an artificial impact.
        table_top = float(data.geom_xpos[table_geom, 2] + model.geom_size[table_geom, 2])
        data.qpos[blue_qpos + 2] = table_top + 0.04
        data.qvel[:] = 0.0
        data.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(model, data)
        model.actuator_forcerange[hand_actuator_ids] = force_ranges
        if abs(radius - args.radius) > 1e-12:
            raise ValueError("object radius is fixed at compile time for valid mass/inertia")
        for _ in range(round(0.3 / model.opt.timestep)):
            mujoco.mj_step(model, data)
        return data.xpos[blue_body].copy()

    # Measure the intended closed thumb-index midpoint once in imported hand
    # coordinates.  Using the open midpoint was a kinematic-proxy shortcut and
    # cannot produce simultaneous opposing contact on a slim object.
    reset_scene(args.radius)
    data.qpos[hand_addresses] = [
        args.pinch_fraction * HAND_SYNERGIES["pinch"][name]
        for name in RIGHT_HAND_JOINTS
    ]
    mujoco.mj_forward(model, data)
    tips = local_tip_positions(model, data, controller)
    closed_midpoint = 0.5 * (tips["if_tip_col"] + tips["th_tip_col"])
    closed_separation = tips["if_tip_col"] - tips["th_tip_col"]

    if args.quick:
        # One deterministic diagnostic trial.  The full sweep is only useful
        # after this pose is shown to start without hidden penetration.
        grid = itertools.product(
            (args.radius,),
            (args.pinch_fraction,),
            (args.force_limit,),
            ((0.015, -0.021, 0.0),),
        )
    else:
        grid = itertools.product(
            (args.radius,),
            (0.12, 0.20, 0.30, 0.42),
            (0.20, 0.40, 0.75, 1.25),
            (
                (0.013, -0.021, 0.0),
                (0.015, -0.021, 0.0),
                (0.017, -0.021, 0.0),
            ),
        )

    rows: list[dict[str, object]] = []
    for trial_index, (radius, final_fraction, force_limit, shift_values) in enumerate(grid):
        initial_xyz = reset_scene(radius)
        world_separation = rotation @ closed_separation
        object_yaw = float(np.arctan2(world_separation[1], world_separation[0]))
        data.qpos[blue_qpos + 3 : blue_qpos + 7] = [
            np.cos(object_yaw / 2.0),
            0.0,
            0.0,
            np.sin(object_yaw / 2.0),
        ]
        data.qvel[:] = 0.0
        data.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(model, data)
        shift = np.asarray(shift_values)
        contact_center = closed_midpoint + shift
        grasp_point = initial_xyz + np.asarray([0.0, 0.0, 0.025])
        target = grasp_point - rotation @ contact_center
        lift_target = target + np.asarray([0.0, 0.0, 0.05])
        approach = controller.solve_position(
            target,
            tolerance_m=0.001,
            local_x_world=rotation[:, 0],
            local_y_world=rotation[:, 1],
            axis_tolerance=0.01,
            position_weight=30.0,
            continuity_weight=0.001,
            posture_reference=NATURAL_TABLETOP_POSTURE,
            posture_weight=0.005,
        )
        lift = controller.solve_position(
            lift_target,
            tolerance_m=0.001,
            local_x_world=rotation[:, 0],
            local_y_world=rotation[:, 1],
            axis_tolerance=0.01,
            position_weight=30.0,
            continuity_weight=0.001,
            posture_reference=NATURAL_TABLETOP_POSTURE,
            posture_weight=0.005,
        )
        data.qpos[controller.qpos_addresses] = approach.joint_positions
        preshape = np.asarray(
            [
                args.preshape_fraction * HAND_SYNERGIES["pinch"][name]
                for name in RIGHT_HAND_JOINTS
            ]
        )
        data.qpos[hand_addresses] = preshape
        # The arm pose is the declared episode initial condition.  Clear the
        # velocity and solver history from the previous imported pose before
        # the first physical contact step; otherwise a position teleport keeps
        # stale momentum and produces a non-physical contact impulse.
        data.qvel[:] = 0.0
        data.qacc_warmstart[:] = 0.0
        data.ctrl[hand_actuator_ids] = preshape
        model.actuator_forcerange[hand_actuator_ids, 0] = -force_limit
        model.actuator_forcerange[hand_actuator_ids, 1] = force_limit
        data.ctrl[controller.actuator_ids] = compensated_arm_control(
            controller, np.asarray(approach.joint_positions)
        )
        # compensated_arm_control runs an internal forward pass.  Do not let
        # its constraint warm-start leak into the declared episode state.
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        data.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(model, data)
        actual_rotation = data.site_xmat[controller.site_id].reshape(3, 3).copy()
        realized_center = data.site_xpos[controller.site_id] + actual_rotation @ contact_center
        alignment_error = float(np.linalg.norm(realized_center - grasp_point))
        initial_contacts = contact_snapshot(model, data)
        initial_min_distance = min(
            (float(item["distance_m"]) for item in initial_contacts), default=float("inf")
        )
        initial_max_force = max(
            (float(item["force_n"]) for item in initial_contacts), default=0.0
        )
        monitor = TrialMonitor(
            model,
            data,
            blue_body,
            blue_geom,
            "blue_cylinder_col",
            initial_xyz,
        )
        warnings_before = warning_counts(data)
        # Allow the initially near-touching pose to settle before squeezing.
        for _ in range(round(0.25 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            monitor.sample()
        command_pinch(
            model,
            data,
            hand_actuator_ids,
            final_fraction,
            1.5,
            monitor,
        )
        for _ in range(round(0.35 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            monitor.sample()
        contact_before_lift = monitor.longest_bilateral_steps
        controller.execute_joint_goal(
            np.asarray(lift.joint_positions),
            duration_s=1.5,
            settle_s=0.25,
            frame_callback=lambda: monitor.sample(lift_phase=True),
            callback_interval_s=model.opt.timestep,
        )
        final_xyz = data.xpos[blue_body].copy()
        warning_delta = [
            after - before for before, after in zip(warnings_before, warning_counts(data))
        ]
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
        success = bool(stable and contact_admitted and lift_admitted)
        row = {
            "trial": trial_index,
            "radius_m": radius,
            "yaw_deg": yaw,
            "object_yaw_deg": float(np.rad2deg(object_yaw)),
            "contact_center_local_m": contact_center.tolist(),
            "ik_position_error_m": approach.position_error_m,
            "ik_local_x_axis_error": approach.local_x_axis_error,
            "ik_local_y_axis_error": approach.local_y_axis_error,
            "realized_contact_center_m": realized_center.tolist(),
            "planned_grasp_point_m": grasp_point.tolist(),
            "initial_alignment_error_m": alignment_error,
            "initial_contact_min_distance_m": initial_min_distance,
            "initial_contact_max_force_n": initial_max_force,
            "initial_contacts": initial_contacts,
            "final_pinch_fraction": final_fraction,
            "force_limit_n": force_limit,
            "success": success,
            "stable": stable,
            "contact_admitted": contact_admitted,
            "lift_admitted": lift_admitted,
            "reset_detected": monitor.reset_detected,
            "warning_delta": warning_delta,
            "contact_before_lift_s": contact_before_lift * model.opt.timestep,
            "bilateral_lift_contact_s": monitor.bilateral_lift_steps * model.opt.timestep,
            "hand_contact_geoms": sorted(monitor.hand_contact_geoms),
            "max_contact_force_n": monitor.max_contact_force_n,
            "max_qvel": monitor.max_qvel,
            "max_lift_m": monitor.max_lift_m,
            "final_lift_m": float(final_xyz[2] - initial_xyz[2]),
            "max_xy_shift_m": monitor.max_xy_shift_m,
            "initial_xyz_m": initial_xyz.tolist(),
            "final_xyz_m": final_xyz.tolist(),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    ranked = sorted(
        rows,
        key=lambda row: (
            bool(row["success"]),
            bool(row["stable"]),
            float(row["final_lift_m"]),
            float(row["contact_before_lift_s"]),
            -float(row["max_contact_force_n"]),
        ),
        reverse=True,
    )
    result = {
        "experiment": "G0-B true thumb-index pinch calibration",
        "claim_boundary": "Evaluator-only geometry sweep; public RGB-D admission is separate.",
        "seed": args.seed,
        "visual_and_collision_radius_swept_together": True,
        "trials": rows,
        "success_count": sum(bool(row["success"]) for row in rows),
        "stable_count": sum(bool(row["stable"]) for row in rows),
        "best": ranked[0] if ranked else None,
    }
    write_json(args.output / "results.json", result)
    print(json.dumps({key: result[key] for key in ("success_count", "stable_count", "best")}, indent=2))


if __name__ == "__main__":
    main()
