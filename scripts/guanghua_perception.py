"""RGB-D-only color grounding for the frozen Guanghua experiment objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import mujoco
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class ObjectEstimate:
    name: str
    center_xyz_m: tuple[float, float, float]
    visible_surface_min_xyz_m: tuple[float, float, float]
    visible_surface_max_xyz_m: tuple[float, float, float]
    pixel_count: int
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def capture_rgbd(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    camera: str = "agentview",
    width: int = 640,
    height: int = 480,
) -> tuple[np.ndarray, np.ndarray]:
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=camera)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=camera)
        depth = renderer.render().copy()
    finally:
        renderer.close()
    return rgb, depth


def unproject_depth(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    depth: np.ndarray,
    *,
    camera: str,
) -> np.ndarray:
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if camera_id < 0:
        raise ValueError(f"unknown camera: {camera}")
    height, width = depth.shape
    focal = 0.5 * height / np.tan(np.deg2rad(model.cam_fovy[camera_id]) / 2.0)
    rows, columns = np.indices(depth.shape)
    camera_points = np.stack(
        [
            (columns - (width - 1) / 2.0) * depth / focal,
            -(rows - (height - 1) / 2.0) * depth / focal,
            -depth,
        ],
        axis=-1,
    )
    rotation = data.cam_xmat[camera_id].reshape(3, 3)
    translation = data.cam_xpos[camera_id]
    return camera_points @ rotation.T + translation


def _color_masks(rgb: np.ndarray) -> Mapping[str, np.ndarray]:
    image = rgb.astype(np.float32)
    red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    # The protected-glass asset has a deliberately orange rim.  A pure
    # red-dominance test aliases that rim with the manipulation target, so red
    # additionally requires a low green channel.  This remains an RGB-only
    # observation rule and does not use simulator segmentation labels.
    fragile = (
        (red > 100)
        & (green > 70)
        & (blue < 0.65 * np.minimum(red, green))
    )
    return {
        "red_cube": (
            (red > 80)
            & (green < 90)
            & (red > 2.0 * green)
            & (red > 1.5 * blue)
            & ~fragile
        ),
        "blue_cylinder": (blue > 80) & (blue > 1.25 * red) & (blue > 1.2 * green),
        "fragile_proxy": fragile,
    }


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Reject same-colour robot highlights without using simulator labels."""
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    return labels == int(np.argmax(component_sizes))


def _object_surface_mask(
    color_mask: np.ndarray,
    world_z: np.ndarray,
    *,
    name: str,
    support_z: float,
) -> np.ndarray:
    ceilings = {
        "red_cube": 0.055,
        "blue_cylinder": 0.075,
        "fragile_proxy": 0.090,
    }
    return (
        color_mask
        & (world_z > 0.815)
        & (world_z <= support_z + ceilings[name])
    )


def estimate_colored_objects(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    rest_z_by_object: Mapping[str, float],
    camera: str = "agentview",
    minimum_pixels: int = 100,
) -> dict[str, ObjectEstimate]:
    if rgb.shape[:2] != depth.shape:
        raise ValueError("RGB and depth shapes do not match")
    world_points = unproject_depth(model, data, depth, camera=camera)
    estimates: dict[str, ObjectEstimate] = {}
    # Public, class-level tabletop envelopes.  They reject robot highlights
    # that touch an object in the 2-D image but lie tens of centimetres above
    # its calibrated support height.  The envelopes are shared by all shape
    # variants and therefore do not reveal the evaluator's variant label.
    for name, color_mask in _color_masks(rgb).items():
        # Target-zone markers are at z≈0.802 m. Keep only object surfaces.
        support_z = float(rest_z_by_object[name])
        candidate_mask = _object_surface_mask(
            color_mask,
            world_points[..., 2],
            name=name,
            support_z=support_z,
        )
        mask = _largest_connected_component(candidate_mask)
        points = world_points[mask]
        if len(points) < minimum_pixels:
            continue
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        center_xy = 0.5 * (minimum[:2] + maximum[:2])
        confidence = min(1.0, float(len(points)) / 1500.0)
        estimate = ObjectEstimate(
            name=name,
            center_xyz_m=(
                float(center_xy[0]),
                float(center_xy[1]),
                float(rest_z_by_object[name]),
            ),
            visible_surface_min_xyz_m=tuple(float(value) for value in minimum),
            visible_surface_max_xyz_m=tuple(float(value) for value in maximum),
            pixel_count=int(len(points)),
            confidence=confidence,
        )
        estimates[name] = estimate
    return estimates


def save_rgbd(rgb: np.ndarray, depth: np.ndarray, directory: Path, stem: str) -> dict[str, str]:
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    rgb_path = directory / f"{stem}_rgb.png"
    depth_path = directory / f"{stem}_depth.npy"
    Image.fromarray(rgb).save(rgb_path)
    np.save(depth_path, depth)
    return {"rgb": str(rgb_path), "depth": str(depth_path)}
