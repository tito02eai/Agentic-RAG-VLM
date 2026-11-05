"""BluPe published native-joint product-of-exponentials grasp-frame model."""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import numpy as np

from .geometry import GeometryError, finite_vector, rigid_transform
from .kinematics import YamKinematics, _axis_angle_matrix


class NativeYamKinematics(YamKinematics):
    def __init__(self, calibration: Mapping[str, Any], arm: str,
                 joint_limits_deg: Sequence[Sequence[float]], *, max_aperture_m: float = .085) -> None:
        super().__init__()
        if calibration["kinematics"]["joint_units"] != "degrees":
            raise GeometryError("native model requires degree API joints")
        model = calibration["kinematics"][arm]
        if len(model["joints"]) != 6:
            raise GeometryError("expected six native joint axes")
        self.axes = []
        self.origins = []
        for joint in model["joints"]:
            axis = finite_vector(joint["axis"], 3, "joint axis")
            if not np.isclose(np.linalg.norm(axis), 1, atol=1e-5):
                raise GeometryError("joint axis must be unit length")
            self.axes.append(axis / np.linalg.norm(axis))
            self.origins.append(finite_vector(joint["position_m"], 3, "joint origin"))
        self.home = rigid_transform(model["T_base_grasp_zero"])
        limits = np.asarray(joint_limits_deg, dtype=float)
        if limits.shape != (6, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
            raise GeometryError("native station joint limits must be measured/configured as six min/max pairs")
        self._limits = np.radians(limits)
        self.tcp_offset_m = 0.0  # published model already ends at the grasp frame
        if not np.isfinite(max_aperture_m) or not 0 < max_aperture_m <= .2:
            raise GeometryError("invalid measured gripper range")
        self.max_aperture_m = max_aperture_m

    def fk(self, joints_rad: np.ndarray) -> np.ndarray:
        joints = finite_vector(joints_rad, 6, "native joints")
        result = np.eye(4)
        for angle, axis, origin in zip(joints, self.axes, self.origins):
            rotation = _axis_angle_matrix(axis, float(angle))
            step = np.eye(4)
            step[:3, :3] = rotation
            step[:3, 3] = (np.eye(3) - rotation) @ origin
            result = result @ step
        return result @ self.home

    def fk_mount(self, joints_rad: np.ndarray) -> np.ndarray:
        raise GeometryError("native profile defines a grasp frame, not the URDF mount frame")

    def gripper_command_for_aperture(self, aperture_m: float) -> float:
        if not np.isfinite(aperture_m) or not 0 <= aperture_m <= self.max_aperture_m:
            raise GeometryError("requested aperture exceeds measured gripper range")
        return max(.02, min(1., aperture_m / self.max_aperture_m))
