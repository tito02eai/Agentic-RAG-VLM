"""Right-arm IK, position control, and dexterous-hand synergies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import mujoco
import numpy as np
from scipy.optimize import least_squares


RIGHT_ARM_JOINTS = tuple(f"arm_r{index}_joint" for index in range(1, 8))
# A neutral tabletop posture for the imported Guanghua right arm.  It is not a
# commanded pose: it only resolves redundant IK branches so that two equally
# accurate Cartesian solutions do not arbitrarily flip the elbow or wrist.
NATURAL_TABLETOP_POSTURE = np.array([-0.42, 0.45, -0.50, -0.90, -0.70, 0.0, -1.00])
RIGHT_HAND_JOINTS = (
    "if_proximal_link",
    "if_distal_link",
    "mf_proximal_link",
    "mf_distal_link",
    "rf_proximal_link",
    "rf_distal_link",
    "lf_proximal_link",
    "lf_distal_link",
    "th_root_link",
    "th_proximal_link",
    "th_distal_link",
)

_POWER_FULL_CLOSE = {
    "if_proximal_link": -0.75,
    "if_distal_link": -0.70,
    "mf_proximal_link": -0.78,
    "mf_distal_link": -0.72,
    "rf_proximal_link": -0.76,
    "rf_distal_link": -0.70,
    "lf_proximal_link": -0.72,
    "lf_distal_link": -0.66,
    "th_root_link": 0.78,
    "th_proximal_link": -0.70,
    "th_distal_link": -0.62,
}
_PINCH_FULL_CLOSE = {
    "if_proximal_link": -0.82,
    "if_distal_link": -0.76,
    "mf_proximal_link": -0.78,
    "mf_distal_link": -0.72,
    "rf_proximal_link": -0.78,
    "rf_distal_link": -0.72,
    "lf_proximal_link": -0.76,
    "lf_distal_link": -0.70,
    "th_root_link": 0.88,
    "th_proximal_link": -0.82,
    "th_distal_link": -0.72,
}


def _scaled_synergy(values: dict[str, float], scale: float) -> dict[str, float]:
    return {name: scale * value for name, value in values.items()}


# These closure fractions were fitted against the visible imported STL meshes,
# not only the deliberately conservative capsule collision proxies.  A value of
# 1.0 drove several fingers 16--23 mm through the displayed task objects.
POWER_HOLD_FRACTION = 0.7087807734835753
PINCH_HOLD_FRACTION = 0.4681068437917144
POWER_PRESHAPE_FRACTION = 0.408 / POWER_HOLD_FRACTION
PINCH_PRESHAPE_FRACTION = 0.402 / PINCH_HOLD_FRACTION


HAND_SYNERGIES: dict[str, dict[str, float]] = {
    "open": {name: 0.0 for name in RIGHT_HAND_JOINTS},
    "power": _scaled_synergy(_POWER_FULL_CLOSE, POWER_HOLD_FRACTION),
    "pinch": {
        **_scaled_synergy(_PINCH_FULL_CLOSE, PINCH_HOLD_FRACTION),
        # Tuck the non-participating fingers.  Leaving them straight is both
        # visually implausible for a precision pinch and unsafe for a
        # top-down tabletop approach because they reach below the pinch pair.
    },
    "side": {
        "if_proximal_link": -0.48,
        "if_distal_link": -0.42,
        "mf_proximal_link": -0.52,
        "mf_distal_link": -0.45,
        "rf_proximal_link": -0.50,
        "rf_distal_link": -0.43,
        "lf_proximal_link": -0.45,
        "lf_distal_link": -0.38,
        "th_root_link": 0.58,
        "th_proximal_link": -0.48,
        "th_distal_link": -0.42,
    },
}

# Object-specific pre-shapes are close enough to keep the long imported finger
# meshes above the tabletop, yet retain a visible closing motion before touch.
HAND_SYNERGIES["power_preshape"] = _scaled_synergy(
    HAND_SYNERGIES["power"], POWER_PRESHAPE_FRACTION
)
HAND_SYNERGIES["pinch_preshape"] = _scaled_synergy(
    HAND_SYNERGIES["pinch"], PINCH_PRESHAPE_FRACTION
)


@dataclass(frozen=True)
class IKSolution:
    joint_positions: tuple[float, ...]
    target_xyz_m: tuple[float, float, float]
    achieved_xyz_m: tuple[float, float, float]
    position_error_m: float
    solver_cost: float
    evaluations: int
    local_x_axis_error: float | None = None
    local_y_axis_error: float | None = None
    joint_displacement_l2: float = 0.0
    posture_deviation_l2: float = 0.0
    minimum_joint_limit_margin_rad: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RightArmController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_hand_r")
        self.joint_ids = np.asarray(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in RIGHT_ARM_JOINTS],
            dtype=int,
        )
        self.qpos_addresses = np.asarray([model.jnt_qposadr[joint] for joint in self.joint_ids], dtype=int)
        self.actuator_ids = np.asarray(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{name}") for name in RIGHT_ARM_JOINTS],
            dtype=int,
        )
        if self.site_id < 0 or np.any(self.joint_ids < 0) or np.any(self.actuator_ids < 0):
            raise RuntimeError("right-arm model contract is incomplete")

    def arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_addresses].copy()

    def solve_position(
        self,
        target_xyz_m: np.ndarray,
        *,
        tolerance_m: float = 0.012,
        local_x_world: np.ndarray | None = None,
        local_y_world: np.ndarray | None = None,
        axis_tolerance: float = 0.03,
        position_weight: float = 10.0,
        continuity_weight: float = 0.02,
        posture_reference: np.ndarray | None = None,
        posture_weight: float = 0.0,
        joint_center_weight: float = 0.0,
    ) -> IKSolution:
        target = np.asarray(target_xyz_m, dtype=np.float64)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target_xyz_m must be a finite 3-vector")
        if position_weight <= 0.0:
            raise ValueError("position_weight must be positive")
        if continuity_weight < 0.0:
            raise ValueError("continuity_weight must be non-negative")
        if posture_weight < 0.0 or joint_center_weight < 0.0:
            raise ValueError("posture weights must be non-negative")
        def normalized_axis(value: np.ndarray | None, name: str) -> np.ndarray | None:
            if value is None:
                return None
            axis = np.asarray(value, dtype=np.float64)
            if axis.shape != (3,) or not np.all(np.isfinite(axis)):
                raise ValueError(f"{name} must be a finite 3-vector")
            norm = float(np.linalg.norm(axis))
            if norm <= 1e-8:
                raise ValueError(f"{name} must be non-zero")
            return axis / norm

        desired_axis = normalized_axis(local_x_world, "local_x_world")
        desired_y_axis = normalized_axis(local_y_world, "local_y_world")
        if desired_axis is not None and desired_y_axis is not None:
            if abs(float(np.dot(desired_axis, desired_y_axis))) > 1e-3:
                raise ValueError("local_x_world and local_y_world must be orthogonal")
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        current = self.arm_qpos()
        lower = self.model.jnt_range[self.joint_ids, 0] + 0.02
        upper = self.model.jnt_range[self.joint_ids, 1] - 0.02
        joint_center = 0.5 * (lower + upper)
        posture = NATURAL_TABLETOP_POSTURE if posture_reference is None else np.asarray(
            posture_reference, dtype=np.float64
        )
        if posture.shape != (7,) or not np.all(np.isfinite(posture)):
            raise ValueError("posture_reference must be a finite 7-vector")
        seeds = (
            current,
            posture,
            np.array([-0.8, 0.0, 0.0, -0.8, 0.0, 0.0, 0.0]),
            np.array([-0.6, -0.25, 0.25, -1.1, 0.0, 0.0, 0.0]),
            np.array([-1.0, 0.25, -0.25, -0.8, 0.0, 0.0, 0.0]),
            np.array([-0.6, -0.2, 0.2, -1.1, -0.2, -1.4, 0.2]),
        )
        best: IKSolution | None = None
        try:
            for seed in seeds:
                clipped_seed = np.clip(seed, lower, upper)

                def residual(joint_positions: np.ndarray) -> np.ndarray:
                    self.data.qpos[self.qpos_addresses] = joint_positions
                    mujoco.mj_forward(self.model, self.data)
                    position_error = position_weight * (self.data.site_xpos[self.site_id] - target)
                    # Resolve redundant-arm branches in favour of temporal
                    # continuity.  Regularising around each multi-start seed can
                    # otherwise make a nearby Cartesian target jump to a distant
                    # elbow/wrist configuration during execution.
                    regularization = continuity_weight * (joint_positions - current)
                    terms = [position_error]
                    if desired_axis is not None:
                        current_axis = self.data.site_xmat[self.site_id].reshape(3, 3)[:, 0]
                        terms.append(5.0 * (current_axis - desired_axis))
                    if desired_y_axis is not None:
                        current_y_axis = self.data.site_xmat[self.site_id].reshape(3, 3)[:, 1]
                        terms.append(5.0 * (current_y_axis - desired_y_axis))
                    terms.append(regularization)
                    if posture_weight > 0.0:
                        terms.append(posture_weight * (joint_positions - posture))
                    if joint_center_weight > 0.0:
                        # Normalise by half-range so this remains meaningful if
                        # a future URDF uses non-uniform joint limits.
                        half_range = np.maximum(0.5 * (upper - lower), 1e-6)
                        terms.append(joint_center_weight * (joint_positions - joint_center) / half_range)
                    return np.concatenate(terms)

                result = least_squares(
                    residual,
                    clipped_seed,
                    bounds=(lower, upper),
                    max_nfev=400,
                    ftol=1e-10,
                    xtol=1e-10,
                    gtol=1e-10,
                )
                self.data.qpos[self.qpos_addresses] = result.x
                mujoco.mj_forward(self.model, self.data)
                achieved = self.data.site_xpos[self.site_id].copy()
                error = float(np.linalg.norm(achieved - target))
                axis_error = None
                if desired_axis is not None:
                    current_axis = self.data.site_xmat[self.site_id].reshape(3, 3)[:, 0]
                    axis_error = float(np.linalg.norm(current_axis - desired_axis))
                y_axis_error = None
                if desired_y_axis is not None:
                    current_y_axis = self.data.site_xmat[self.site_id].reshape(3, 3)[:, 1]
                    y_axis_error = float(np.linalg.norm(current_y_axis - desired_y_axis))
                candidate = IKSolution(
                    joint_positions=tuple(float(value) for value in result.x),
                    target_xyz_m=tuple(float(value) for value in target),
                    achieved_xyz_m=tuple(float(value) for value in achieved),
                    position_error_m=error,
                    solver_cost=float(result.cost),
                    evaluations=int(result.nfev),
                    local_x_axis_error=axis_error,
                    local_y_axis_error=y_axis_error,
                    joint_displacement_l2=float(np.linalg.norm(result.x - current)),
                    posture_deviation_l2=float(np.linalg.norm(result.x - posture)),
                    minimum_joint_limit_margin_rad=float(
                        np.min(np.minimum(result.x - lower, upper - result.x)) + 0.02
                    ),
                )
                candidate_score = (
                    candidate.position_error_m
                    + (axis_error or 0.0)
                    + (y_axis_error or 0.0)
                    + 0.1 * continuity_weight * candidate.joint_displacement_l2
                    + 0.1 * posture_weight * candidate.posture_deviation_l2
                    - 0.02 * candidate.minimum_joint_limit_margin_rad
                )
                best_score = float("inf") if best is None else (
                    best.position_error_m
                    + (best.local_x_axis_error or 0.0)
                    + (best.local_y_axis_error or 0.0)
                    + 0.1 * continuity_weight * best.joint_displacement_l2
                    + 0.1 * posture_weight * best.posture_deviation_l2
                    - 0.02 * best.minimum_joint_limit_margin_rad
                )
                if best is None or candidate_score < best_score:
                    best = candidate
        finally:
            self.data.qpos[:] = saved_qpos
            self.data.qvel[:] = saved_qvel
            mujoco.mj_forward(self.model, self.data)
        if (
            best is None
            or best.position_error_m > tolerance_m
            or (best.local_x_axis_error is not None and best.local_x_axis_error > axis_tolerance)
            or (best.local_y_axis_error is not None and best.local_y_axis_error > axis_tolerance)
        ):
            error = float("inf") if best is None else best.position_error_m
            axis_error = None if best is None else best.local_x_axis_error
            y_axis_error = None if best is None else best.local_y_axis_error
            raise RuntimeError(
                f"right-arm IK target is unreachable; error={error:.4f} m, "
                f"axis_error={axis_error}, y_axis_error={y_axis_error}"
            )
        return best

    def execute_joint_goal(
        self,
        goal: np.ndarray,
        *,
        duration_s: float = 2.0,
        settle_s: float = 0.4,
        frame_callback: Callable[[], None] | None = None,
        callback_interval_s: float = 0.05,
    ) -> dict[str, object]:
        target = np.asarray(goal, dtype=np.float64)
        if target.shape != (7,):
            raise ValueError("right-arm joint goal must contain seven values")
        # MuJoCo position actuators express gravity load as a steady reference
        # error.  Compute a zero-velocity inverse-dynamics bias at the desired
        # pose and convert it to a position-reference offset (tau / kp).  This
        # is conventional gravity feed-forward, not privileged task state.
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        try:
            self.data.qpos[self.qpos_addresses] = target
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            dof_addresses = self.model.jnt_dofadr[self.joint_ids]
            required_torque = (
                self.data.qfrc_bias[dof_addresses] - self.data.qfrc_passive[dof_addresses]
            )
            position_gain = self.model.actuator_gainprm[self.actuator_ids, 0]
            control_target = target + required_torque / position_gain
            limited = self.model.actuator_ctrllimited[self.actuator_ids].astype(bool)
            control_target[limited] = np.clip(
                control_target[limited],
                self.model.actuator_ctrlrange[self.actuator_ids[limited], 0],
                self.model.actuator_ctrlrange[self.actuator_ids[limited], 1],
            )
        finally:
            self.data.qpos[:] = saved_qpos
            self.data.qvel[:] = saved_qvel
            mujoco.mj_forward(self.model, self.data)
        start = self.data.ctrl[self.actuator_ids].copy()
        steps = max(1, round(duration_s / self.model.opt.timestep))
        callback_steps = max(1, round(callback_interval_s / self.model.opt.timestep))
        for step in range(steps):
            fraction = (step + 1) / steps
            smooth = fraction * fraction * (3.0 - 2.0 * fraction)
            self.data.ctrl[self.actuator_ids] = start + smooth * (control_target - start)
            mujoco.mj_step(self.model, self.data)
            if frame_callback is not None and (step + 1) % callback_steps == 0:
                frame_callback()
        self.data.ctrl[self.actuator_ids] = control_target
        settle_steps = max(0, round(settle_s / self.model.opt.timestep))
        for step in range(settle_steps):
            mujoco.mj_step(self.model, self.data)
            if frame_callback is not None and (step + 1) % callback_steps == 0:
                frame_callback()
        actual = self.arm_qpos()
        return {
            "joint_target": target.round(6).tolist(),
            "gravity_compensated_control_target": control_target.round(6).tolist(),
            "joint_actual": actual.round(6).tolist(),
            "joint_error_l2": float(np.linalg.norm(actual - target)),
            "site_xyz_m": self.data.site_xpos[self.site_id].round(6).tolist(),
        }


def command_hand_synergy(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    synergy: str,
    *,
    duration_s: float = 0.6,
) -> dict[str, object]:
    if synergy not in HAND_SYNERGIES:
        raise ValueError(f"unknown hand synergy {synergy!r}")
    targets = HAND_SYNERGIES[synergy]
    actuator_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{name}") for name in RIGHT_HAND_JOINTS],
        dtype=int,
    )
    start = data.ctrl[actuator_ids].copy()
    goal = np.asarray([targets[name] for name in RIGHT_HAND_JOINTS], dtype=np.float64)
    steps = max(1, round(duration_s / model.opt.timestep))
    for step in range(steps):
        fraction = (step + 1) / steps
        smooth = fraction * fraction * (3.0 - 2.0 * fraction)
        data.ctrl[actuator_ids] = start + smooth * (goal - start)
        mujoco.mj_step(model, data)
    data.ctrl[actuator_ids] = goal
    return {"synergy": synergy, "joint_targets": dict(targets)}
