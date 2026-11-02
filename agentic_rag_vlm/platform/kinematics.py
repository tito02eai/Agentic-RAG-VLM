"""YAM 6-DOF forward/inverse kinematics (URDF-derived, numpy only).

The kinematic chain is transcribed from the official I2RT model
``i2rt/robot_models/arm/yam/v1/yam.urdf`` (flange convention v1, 2026-09-16).
Only the six revolute arm joints are modelled; the gripper mount frame
(``+X`` right, ``+Y`` down, ``+Z`` forward) is the end-effector frame, with an
optional configurable tool-centre-point (TCP) offset along ``+Z``.

Joint limits follow the URDF.  All public angles are in radians internally and
converted to/from degrees at the transport boundary.

Home-pose anchor (from the I2RT model README, used by the unit tests):
at ``q = 0`` the gripper mount sits at ``[0.110598, 0, 0.173501]`` in the base
frame with ``+Z`` pointing along base ``+X``, ``+X`` along base ``-Y`` and
``+Y`` along base ``-Z``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass(frozen=True)
class JointSpec:
    origin_xyz_m: tuple[float, float, float]
    origin_rpy_rad: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower_rad: float
    upper_rad: float


# Transcribed from yam.urdf (joint1..joint6, revolute, flange convention v1).
YAM_JOINT_CHAIN: tuple[JointSpec, ...] = (
    JointSpec((0.0, 0.0, 0.068), (0.0, math.pi / 2, math.pi), (-1.0, 0.0, 0.0), -2.61799, 3.14159),
    JointSpec((-0.0455, -0.0339, -0.02), (0.0, 0.0, -math.pi), (0.0, 1.0, 0.0), 0.0, 3.66519),
    JointSpec((0.0, -0.0688, 0.264), (0.0, 0.0, -math.pi), (0.0, 1.0, 0.0), 0.0, 3.14159),
    JointSpec((-0.0600003, -0.0688, -0.244999), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), -1.69297, 1.5708),
    JointSpec((-0.0405003, 0.0338995, -0.0739989), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), -1.5708, 1.5708),
    JointSpec((0.0404996, 0.0, -0.0356), (math.pi, 0.0, math.pi / 2), (0.0, 0.0, 1.0), -2.0944, 2.0944),
)

NUM_JOINTS = 6
_HOME_EE_XYZ_M = (0.110597512463, 0.0, 0.173501403591)
_GRIPPER_TRAVEL_M = 2 * 0.04695  # two fingers, 0.04695 m each (URDF joint7/8)


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis RPY: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) * c + s * cross + np.outer(axis, axis) * (1.0 - c)


def _rotation_log(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle)."""
    cosine = (np.trace(matrix) - 1.0) / 2.0
    cosine = max(-1.0, min(1.0, cosine))
    angle = math.acos(cosine)
    if angle < 1e-9:
        return np.zeros(3)
    if math.pi - angle < 1e-6:  # near pi: use the symmetric part
        aa = (matrix + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(aa), 0.0))
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        # recover the correct sign from the largest off-diagonal pair
        index = int(np.argmax(axis))
        if index == 0:
            if matrix[2, 1] - matrix[1, 2] < 0:
                axis = -axis
        elif index == 1:
            if matrix[0, 2] - matrix[2, 0] < 0:
                axis = -axis
        else:
            if matrix[1, 0] - matrix[0, 1] < 0:
                axis = -axis
        return axis * angle
    skew = (matrix - matrix.T) / 2.0
    return np.array([skew[2, 1], skew[0, 2], skew[1, 0]]) * (angle / math.sin(angle))


def minimal_rotation_to_axis(current_axis: np.ndarray, target_axis: np.ndarray) -> np.ndarray:
    """Rotation matrix that maps ``current_axis`` onto ``target_axis`` with the
    smallest possible rotation angle (Stuelpnagel / shortest-arc)."""
    current = current_axis / np.linalg.norm(current_axis)
    target = target_axis / np.linalg.norm(target_axis)
    cross = np.cross(current, target)
    dot = float(np.dot(current, target))
    if np.linalg.norm(cross) < 1e-12:
        if dot > 0.0:
            return np.eye(3)
        # 180 degrees: rotate about any axis orthogonal to `current`
        helper = np.array([1.0, 0.0, 0.0])
        if abs(current[0]) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        axis = np.cross(current, helper)
        axis = axis / np.linalg.norm(axis)
        return _axis_angle_matrix(axis, math.pi)
    return _axis_angle_matrix(cross, math.atan2(np.linalg.norm(cross), dot))


@dataclass(frozen=True)
class IkResult:
    success: bool
    joints_rad: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    iterations: int
    message: str = ""
    within_limits: bool = True


@dataclass
class YamKinematics:
    """Forward/inverse kinematics for one YAM 6-DOF arm.

    ``tcp_offset_m`` shifts the end-effector frame forward along the gripper
    mount ``+Z`` axis (fingertip centre).  ``base_yaw_deg`` optionally rotates
    the arm base frame, and ``joint_signs``/``joint_offsets_deg`` support
    station-specific mounting corrections for the first live calibration;
    defaults reproduce the unmodified URDF.
    """

    chain: tuple[JointSpec, ...] = YAM_JOINT_CHAIN
    tcp_offset_m: float = 0.06
    base_yaw_deg: float = 0.0
    joint_signs: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    joint_offsets_deg: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ik_position_tol_m: float = 5e-4
    ik_orientation_tol_rad: float = 2e-2
    ik_max_iterations: int = 200
    ik_damping: float = 1e-3

    def __post_init__(self) -> None:
        self._base_rotation = _axis_angle_matrix(
            np.array([0.0, 0.0, 1.0]), math.radians(self.base_yaw_deg)
        )
        self._limits = np.array(
            [[joint.lower_rad, joint.upper_rad] for joint in self.chain]
        )
        self._signs = np.array(self.joint_signs, dtype=float)
        self._offsets = np.radians(np.array(self.joint_offsets_deg, dtype=float))

    # ------------------------------------------------------------------ FK

    def _model_joints(self, joints_rad: np.ndarray) -> np.ndarray:
        """Commanded joints -> URDF model joints (signs/offsets applied)."""
        return self._signs * np.asarray(joints_rad, dtype=float) + self._offsets

    def fk(self, joints_rad: np.ndarray) -> np.ndarray:
        """4x4 transform base -> TCP frame for the given commanded joints."""
        model_joints = self._model_joints(joints_rad)
        transform = np.eye(4)
        transform[:3, :3] = self._base_rotation
        for spec, angle in zip(self.chain, model_joints):
            origin = np.eye(4)
            origin[:3, :3] = _rpy_to_matrix(*spec.origin_rpy_rad)
            origin[:3, 3] = spec.origin_xyz_m
            rotation = np.eye(4)
            rotation[:3, :3] = _axis_angle_matrix(np.array(spec.axis), float(angle))
            transform = transform @ origin @ rotation
        tcp = np.eye(4)
        tcp[2, 3] = self.tcp_offset_m
        return transform @ tcp

    def fk_mount(self, joints_rad: np.ndarray) -> np.ndarray:
        """4x4 transform base -> gripper mount frame (no TCP offset)."""
        saved = self.tcp_offset_m
        try:
            object.__setattr__(self, "tcp_offset_m", 0.0)
            self.__post_init__()
            return self.fk(joints_rad)
        finally:
            object.__setattr__(self, "tcp_offset_m", saved)
            self.__post_init__()

    # ------------------------------------------------------------------ IK

    def _jacobian(self, joints_rad: np.ndarray) -> np.ndarray:
        """Geometric Jacobian (6xN) in the base frame via central differences."""
        joints = np.asarray(joints_rad, dtype=float)
        epsilon = 1e-6
        jacobian = np.zeros((6, len(joints)))
        reference = self.fk(joints)
        for index in range(len(joints)):
            plus = joints.copy()
            minus = joints.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            t_plus = self.fk(plus)
            t_minus = self.fk(minus)
            jacobian[:3, index] = (t_plus[:3, 3] - t_minus[:3, 3]) / (2.0 * epsilon)
            jacobian[3:, index] = (
                _rotation_log(t_plus[:3, :3] @ t_minus[:3, :3].T) / (2.0 * epsilon)
            )
        return jacobian

    def clamp_to_limits(self, joints_rad: np.ndarray) -> np.ndarray:
        """Clamp *commanded* joints into the station model's limits.

        Signs/offsets map commanded -> model space; limits live in model space.
        """
        model = self._model_joints(joints_rad)
        model = np.clip(model, self._limits[:, 0], self._limits[:, 1])
        return (model - self._offsets) / self._signs

    def in_limits(self, joints_rad: np.ndarray) -> bool:
        model = self._model_joints(joints_rad)
        return bool(np.all(model >= self._limits[:, 0] - 1e-9)) and bool(
            np.all(model <= self._limits[:, 1] + 1e-9)
        )

    def orientation_for_approach(
        self, approach_dir_base: np.ndarray, *, reference: np.ndarray | None = None
    ) -> np.ndarray:
        """Desired end-effector rotation for a tool ``+Z`` approach direction.

        Builds the shortest-arc rotation from the home tool ``+Z`` axis to the
        requested approach direction and applies it to the home orientation,
        which keeps wrist twist close to the home pose.
        """
        if reference is None:
            home = self.fk(np.zeros(NUM_JOINTS))
            reference = home[:3, :3]
        home_z = reference[:, 2]
        rotation = minimal_rotation_to_axis(home_z, approach_dir_base)
        return rotation @ reference

    def ik(
        self,
        target_position_m: np.ndarray,
        target_rotation: np.ndarray,
        *,
        q_init_rad: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.3,
    ) -> IkResult:
        """Damped least-squares IK for position + full orientation.

        Returns the best solution found, converging to
        ``ik_position_tol_m`` / ``ik_orientation_tol_rad`` when reachable.
        """
        joints = np.asarray(q_init_rad, dtype=float).copy()
        joints = self.clamp_to_limits(joints)
        target_position = np.asarray(target_position_m, dtype=float)
        damping = self.ik_damping
        best = (math.inf, joints.copy())
        for iteration in range(1, self.ik_max_iterations + 1):
            current = self.fk(joints)
            position_error = target_position - current[:3, 3]
            orientation_error = _rotation_log(
                np.asarray(target_rotation) @ current[:3, :3].T
            )
            pos_norm = float(np.linalg.norm(position_error))
            ori_norm = float(np.linalg.norm(orientation_error))
            score = pos_norm + 0.5 * ori_norm
            if score < best[0]:
                best = (score, joints.copy())
            if (
                pos_norm <= self.ik_position_tol_m
                and ori_norm <= self.ik_orientation_tol_rad
            ):
                return IkResult(
                    True, joints, pos_norm, ori_norm, iteration, within_limits=self.in_limits(joints)
                )
            error = np.concatenate(
                [
                    position_weight * position_error,
                    orientation_weight * orientation_error,
                ]
            )
            jacobian = self._jacobian(joints)
            weighted = np.concatenate(
                [
                    position_weight * jacobian[:3, :],
                    orientation_weight * jacobian[3:, :],
                ]
            )
            # joint-limit repulsion keeps the solution inside the URDF range
            margin = 0.08
            model = self._model_joints(joints)
            repulsion = np.zeros(NUM_JOINTS)
            for index in range(NUM_JOINTS):
                low, high = self._limits[index]
                if model[index] < low + margin:
                    repulsion[index] = (low + margin - model[index]) / margin
                elif model[index] > high - margin:
                    repulsion[index] = (high - margin - model[index]) / margin
            weighted = np.vstack([weighted, 0.02 * np.eye(NUM_JOINTS)])
            error = np.concatenate([error, 0.02 * (repulsion - (model - joints))])
            jjt = weighted @ weighted.T + (damping**2) * np.eye(weighted.shape[0])
            step = weighted.T @ np.linalg.solve(jjt, error)
            step = np.clip(step, -0.25, 0.25)
            joints = self.clamp_to_limits(joints + step)
            if np.linalg.norm(step) < 1e-7:
                break
            if pos_norm > 0.5:  # lost: reset damping upward
                damping = min(damping * 1.5, 1.0)
            else:
                damping = max(damping * 0.9, 1e-4)
        joints = best[1]
        current = self.fk(joints)
        pos_norm = float(np.linalg.norm(target_position - current[:3, 3]))
        ori_norm = float(
            np.linalg.norm(_rotation_log(np.asarray(target_rotation) @ current[:3, :3].T))
        )
        success = (
            pos_norm <= self.ik_position_tol_m
            and ori_norm <= self.ik_orientation_tol_rad
        )
        return IkResult(
            success,
            joints,
            pos_norm,
            ori_norm,
            self.ik_max_iterations,
            message="" if success else "ik_did_not_converge",
            within_limits=self.in_limits(joints),
        )

    # --------------------------------------------------------------- gripper

    @staticmethod
    def gripper_command_for_aperture(aperture_m: float) -> float:
        """Map a desired gripper aperture (m) to the platform's 0..1 command.

        The linear gripper travels ``2 x 0.04695 m``; the platform command is
        0 (closed) to 1 (fully open).  Values are clamped to [0.02, 1.0] so a
        grasp always closes with margin instead of commanding a hard stop.
        """
        command = float(aperture_m) / _GRIPPER_TRAVEL_M
        return max(0.02, min(1.0, command))

    @staticmethod
    def aperture_for_gripper_command(command: float) -> float:
        return max(0.0, min(1.0, float(command))) * _GRIPPER_TRAVEL_M

    # ------------------------------------------------------------- utilities

    @property
    def limits_deg(self) -> np.ndarray:
        return np.degrees(self._limits)
