"""Geometry-calibrated grasp frames and redundant-arm posture selection.

The planner deliberately separates three things that were conflated in the
first demo: the wrist site, the point between the closing digits, and the IK
branch.  All calibration below comes from the imported MuJoCo hand geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import atan2, cos, degrees, radians, sin
from typing import Iterable

import mujoco
import numpy as np

from scripts.guanghua_control import (
    HAND_SYNERGIES,
    NATURAL_TABLETOP_POSTURE,
    RIGHT_HAND_JOINTS,
    IKSolution,
    RightArmController,
)


RIGHT_TIP_GEOMS = ("if_tip_col", "mf_tip_col", "rf_tip_col", "lf_tip_col", "th_tip_col")
DEFAULT_TOP_DOWN_YAWS_DEG = (60.0, 90.0, 120.0, 150.0, 180.0, 210.0)
# The object centres in the hand site frame were fitted against the *visible*
# imported STL meshes while enforcing opposing-finger contact and tabletop
# clearance.  The earlier closed-tip midpoint placed the cube and cylinder
# 16--23 mm inside the displayed fingers.
MESH_CLEARANCE_GRASP_CENTERS: dict[str, tuple[float, float, float]] = {
    "power": (-0.1595343353200803, -0.06964525416932564, -0.07401752909615508),
    "pinch": (-0.1875472129823476, -0.06601958725113723, -0.034693152686235934),
}


@dataclass(frozen=True)
class GraspFrame:
    synergy: str
    center_local_m: tuple[float, float, float]
    calibration: str
    participating_tips: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GraspPoseCandidate:
    yaw_deg: float
    wrist_target_xyz_m: tuple[float, float, float]
    local_x_world: tuple[float, float, float]
    local_y_world: tuple[float, float, float]
    ik: IKSolution
    protected_clearance_m: float | None
    admitted: bool
    score: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["ik"] = self.ik.to_dict()
        return payload


def _hand_qpos_addresses(model: mujoco.MjModel) -> np.ndarray:
    return np.asarray(
        [
            model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in RIGHT_HAND_JOINTS
        ],
        dtype=int,
    )


def calibrate_grasp_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: RightArmController,
    synergy: str,
) -> GraspFrame:
    """Return a mesh-clearance-calibrated object centre in hand-site axes."""
    if synergy not in {"power", "pinch"}:
        raise ValueError("only power and pinch have calibrated grasp frames")
    center = MESH_CLEARANCE_GRASP_CENTERS[synergy]
    if synergy == "power":
        participating = RIGHT_TIP_GEOMS
        calibration = "visible-mesh thumb/finger contact with tabletop clearance"
    else:
        participating = ("if_tip_col", "th_tip_col")
        calibration = "visible-mesh thumb-index contact with tabletop clearance"
    return GraspFrame(
        synergy=synergy,
        center_local_m=tuple(float(value) for value in center),
        calibration=calibration,
        participating_tips=participating,
    )


def top_down_rotation(yaw_deg: float) -> np.ndarray:
    """Return a hand rotation whose local -X is the downward approach axis."""
    yaw = radians(yaw_deg)
    local_x = np.asarray([0.0, 0.0, 1.0])
    local_y = np.asarray([cos(yaw), sin(yaw), 0.0])
    return np.column_stack((local_x, local_y, np.cross(local_x, local_y)))


def _is_descendant(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    while body_id > 0:
        if body_id == ancestor_id:
            return True
        body_id = int(model.body_parentid[body_id])
    return False


def hand_collision_geoms(model: mujoco.MjModel) -> tuple[int, ...]:
    hand_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_root")
    return tuple(
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_contype[geom_id] != 0
        and _is_descendant(model, int(model.geom_bodyid[geom_id]), hand_root)
    )


def hand_visual_geoms(model: mujoco.MjModel) -> tuple[int, ...]:
    """Visible right-hand mesh geoms used by the no-penetration video gate."""
    hand_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_root")
    return tuple(
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
        and _is_descendant(model, int(model.geom_bodyid[geom_id]), hand_root)
    )


def minimum_geom_clearance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    source_geom_ids: Iterable[int],
    target_geom_ids: Iterable[int],
) -> float:
    distances = []
    endpoints = np.zeros(6)
    for source in source_geom_ids:
        for target in target_geom_ids:
            distance = float(mujoco.mj_geomDistance(model, data, source, target, 1.0, endpoints))
            if abs(distance) <= 1e-12:
                # MuJoCo's convex-distance query can very occasionally return
                # a transient exact zero for separated capsule/box pairs.  A
                # genuine contact repeats as zero; use the median of three
                # identical-state queries to reject a one-off solver sentinel.
                repeated = [
                    distance,
                    float(mujoco.mj_geomDistance(model, data, source, target, 1.0, endpoints)),
                    float(mujoco.mj_geomDistance(model, data, source, target, 1.0, endpoints)),
                ]
                nonzero = [value for value in repeated if abs(value) > 1e-12]
                if nonzero:
                    distance = float(np.median(nonzero))
                else:
                    pair = {int(source), int(target)}
                    contacting = any(
                        {int(data.contact[index].geom1), int(data.contact[index].geom2)} == pair
                        for index in range(data.ncon)
                    )
                    if not contacting:
                        # All-zero query with no corresponding broad-phase
                        # contact is a GJK sentinel, not a measured touch.
                        continue
            distances.append(distance)
    return min(distances) if distances else float("inf")


def evaluate_top_down_candidate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: RightArmController,
    object_center_xyz_m: np.ndarray,
    frame: GraspFrame,
    yaw_deg: float,
    *,
    protected_geom_names: Iterable[str] = (),
    minimum_joint_margin_rad: float = 0.14,
    minimum_protected_clearance_m: float = 0.0,
) -> GraspPoseCandidate:
    rotation = top_down_rotation(yaw_deg)
    center_local = np.asarray(frame.center_local_m)
    target = np.asarray(object_center_xyz_m, dtype=float) - rotation @ center_local
    local_x = rotation[:, 0]
    local_y = rotation[:, 1]
    solution = controller.solve_position(
        target,
        tolerance_m=0.012,
        local_x_world=local_x,
        local_y_world=local_y,
        axis_tolerance=0.04,
        position_weight=20.0,
        continuity_weight=0.005,
        posture_reference=NATURAL_TABLETOP_POSTURE,
        posture_weight=0.025,
        joint_center_weight=0.005,
    )
    saved_qpos = data.qpos.copy()
    try:
        data.qpos[controller.qpos_addresses] = solution.joint_positions
        hand_addresses = _hand_qpos_addresses(model)
        data.qpos[hand_addresses] = [HAND_SYNERGIES[frame.synergy][name] for name in RIGHT_HAND_JOINTS]
        mujoco.mj_forward(model, data)
        target_geoms = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in protected_geom_names
        ]
        clearance = (
            minimum_geom_clearance(model, data, hand_collision_geoms(model), target_geoms)
            if target_geoms
            else None
        )
    finally:
        data.qpos[:] = saved_qpos
        mujoco.mj_forward(model, data)
    admitted = bool(
        solution.minimum_joint_limit_margin_rad >= minimum_joint_margin_rad
        and (clearance is None or clearance >= minimum_protected_clearance_m)
    )
    # Clearance dominates when a protected object exists; otherwise the
    # neutral 90-degree wrist orientation and natural-posture prior dominate.
    wrapped_yaw_error = abs(degrees(atan2(sin(radians(yaw_deg - 90.0)), cos(radians(yaw_deg - 90.0)))))
    score = (
        (100.0 * min(clearance, 0.05) if clearance is not None else 0.0)
        + 0.20 * solution.minimum_joint_limit_margin_rad
        - 0.03 * solution.posture_deviation_l2
        - 0.001 * wrapped_yaw_error
    )
    return GraspPoseCandidate(
        yaw_deg=float(yaw_deg),
        wrist_target_xyz_m=tuple(float(value) for value in target),
        local_x_world=tuple(float(value) for value in local_x),
        local_y_world=tuple(float(value) for value in local_y),
        ik=solution,
        protected_clearance_m=clearance,
        admitted=admitted,
        score=float(score),
    )


def select_top_down_grasp(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: RightArmController,
    object_center_xyz_m: np.ndarray,
    synergy: str,
    *,
    protected_geom_names: Iterable[str] = (),
    yaw_candidates_deg: Iterable[float] = DEFAULT_TOP_DOWN_YAWS_DEG,
    minimum_joint_margin_rad: float = 0.14,
    minimum_protected_clearance_m: float = 0.0,
) -> tuple[GraspPoseCandidate, tuple[GraspPoseCandidate, ...], GraspFrame]:
    frame = calibrate_grasp_frame(model, data, controller, synergy)
    candidates = tuple(
        evaluate_top_down_candidate(
            model,
            data,
            controller,
            object_center_xyz_m,
            frame,
            yaw,
            protected_geom_names=protected_geom_names,
            minimum_joint_margin_rad=minimum_joint_margin_rad,
            minimum_protected_clearance_m=minimum_protected_clearance_m,
        )
        for yaw in yaw_candidates_deg
    )
    admitted = [candidate for candidate in candidates if candidate.admitted]
    if not admitted:
        raise RuntimeError("no top-down grasp candidate passed joint and protected-clearance gates")
    return max(admitted, key=lambda candidate: candidate.score), candidates, frame
