"""Metric grounding from a measured surface and calibrated RGB pixels.

No language-model metric coordinates enter this path. The selected pixel is
an object contact/feature point on a measured plane, not necessarily its box
centre. Heights and support planes must come from operator measurements.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import numpy as np


class GeometryError(ValueError):
    pass


def finite_vector(value: Sequence[float], size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise GeometryError(f"{label} must contain {size} finite values")
    return result


def rigid_transform(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise GeometryError("expected a finite 4x4 transform")
    rotation = result[:3, :3]
    if not (np.allclose(result[3], [0, 0, 0, 1])
            and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            and np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
        raise GeometryError("invalid rigid transform")
    return result


class PlaneProjector:
    """Undistort a top-camera pixel and intersect a measured base-frame plane."""

    def __init__(self, calibration: Mapping[str, Any], arm: str, *,
                 plane_normal_base: Sequence[float], plane_offset_m: float) -> None:
        camera = calibration["cameras"]["top"]
        self.transform = rigid_transform(camera["T_base_camera"][arm])
        self.K = np.asarray(camera["K"], dtype=float)
        if (self.K.shape != (3, 3) or not np.isfinite(self.K).all()
                or self.K[0, 0] <= 0 or self.K[1, 1] <= 0
                or not np.allclose(self.K[2], [0, 0, 1])):
            raise GeometryError("invalid RGB intrinsics")
        self.image_size = finite_vector(camera["image_size"], 2, "image_size")
        if np.any(self.image_size <= 0):
            raise GeometryError("invalid image dimensions")
        model = camera.get("distortion_model", "none")
        if model not in {"none", "opencv_brown_conrady"}:
            raise GeometryError(f"unsupported RGB distortion model: {model}")
        self.distortion = finite_vector(camera.get("distortion", [0]*5), 5, "distortion")
        self.normal = finite_vector(plane_normal_base, 3, "plane normal")
        norm = np.linalg.norm(self.normal)
        if norm < 1e-9 or not np.isfinite(plane_offset_m):
            raise GeometryError("invalid measured plane")
        self.normal = self.normal / norm
        self.offset = float(plane_offset_m) / norm
        self.calibration_id = str(calibration["calibration_id"])

    def _distort(self, point: np.ndarray) -> np.ndarray:
        x, y = point
        k1, k2, p1, p2, k3 = self.distortion
        r2 = x*x + y*y
        radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
        return np.array([x*radial + 2*p1*x*y + p2*(r2+2*x*x),
                         y*radial + p1*(r2+2*y*y) + 2*p2*x*y])

    def project(self, pixel_xy: Sequence[float], *, height_m: float = 0.0) -> tuple[float, float, float]:
        pixel = finite_vector(pixel_xy, 2, "pixel")
        if np.any(pixel < 0) or np.any(pixel >= self.image_size):
            raise GeometryError("pixel outside calibrated image")
        if not np.isfinite(height_m) or not 0 <= height_m <= 0.5:
            raise GeometryError("invalid measured object feature height")
        observed = np.linalg.solve(self.K, np.r_[pixel, 1])[:2]
        point = observed.copy()
        for _ in range(30):
            point += observed - self._distort(point)
        if not np.isfinite(point).all() or np.linalg.norm(self._distort(point) - observed) > 1e-6:
            raise GeometryError("RGB undistortion did not converge")
        ray = np.r_[point, 1.0]
        origin = self.transform[:3, 3]
        direction = self.transform[:3, :3] @ ray
        denominator = float(self.normal @ direction)
        if abs(denominator) < 1e-8:
            raise GeometryError("camera ray parallel to measured plane")
        distance = -(float(self.normal @ origin) + self.offset - height_m) / denominator
        if not 0.05 <= distance <= 3.0:
            raise GeometryError("plane intersection behind camera or outside depth bounds")
        # Keep camera coordinates for the transport's existing frame contract.
        return tuple(float(v) for v in distance * ray)

    def project_base(self, pixel_xy: Sequence[float], *, height_m: float = 0.0) -> tuple[float, float, float]:
        camera = self.project(pixel_xy, height_m=height_m)
        return tuple(float(v) for v in (self.transform @ np.r_[camera, 1])[:3])
