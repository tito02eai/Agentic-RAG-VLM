"""Metric geometry from public depth and declared shape dimensions only."""
from __future__ import annotations

import numpy as np


def cylinder_cap_center(points, *, half_height_m, radius_m):
    """Fit a visible disk plane, then compensate its tilt to the body centre.

    A tilted cylinder's top-surface centroid is not its body centre. RANSAC
    rejects the curved side; fit quality/coverage gates reject partial disks.
    No temporal averaging can hide actual motion, and no object pose is read.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 40:
        return None
    if not np.isfinite(points).all() or not (0 < half_height_m < .2 and 0 < radius_m < .1):
        return None
    rng = np.random.default_rng(0)
    best = None
    for _ in range(128):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1]-sample[0], sample[2]-sample[0])
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        if normal[2] < 0:
            normal = -normal
        if normal[2] < .7:
            continue
        mask = np.abs((points-sample[0]) @ normal) < .0005
        if best is None or mask.sum() > best.sum():
            best = mask
    if best is None or best.sum() < max(40, .4*len(points)):
        return None
    cap = points[best]
    origin = cap.mean(axis=0)
    _, _, basis = np.linalg.svd(cap-origin, full_matrices=False)
    normal = basis[-1]
    if normal[2] < 0:
        normal = -normal
    if normal[2] < .7:
        return None
    uv = (cap-origin) @ basis[:2].T
    # A complete disk has balanced, nearly circular planar extent. Reject
    # an occluded crescent instead of confidently inventing its centre.
    lo, hi = np.quantile(uv, [.02, .98], axis=0)
    spans = hi-lo
    if np.any(spans < 1.55*radius_m) or np.any(spans > 2.15*radius_m):
        return None
    uv_center = (lo+hi)/2
    top_center = origin + uv_center @ basis[:2]
    radial = np.linalg.norm((cap-top_center) @ basis[:2].T, axis=1)
    if np.quantile(radial, .95) > radius_m*1.15:
        return None
    residual = float(np.sqrt(np.mean(((cap-top_center) @ normal)**2)))
    if residual > .0004:
        return None
    center = top_center - normal*half_height_m
    return {'center_xyz_m': center.tolist(), 'cap_center_xyz_m': top_center.tolist(),
            'axis_xyz': normal.tolist(), 'plane_rms_m': residual,
            'inlier_count': int(best.sum()), 'visible_points': len(points),
            'half_height_m': half_height_m, 'radius_m': radius_m,
            'source': 'public_rgbd_disk_plane_and_declared_cylinder_dimensions'}


def candidate_world_points(rgb, depth, bbox, camera):
    """Back-project foreground pixels in a candidate ROI; no class colour map."""
    x0, y0, x1, y1 = bbox
    region = rgb[y0:y1, x0:x1].astype(float)
    # Restricted scene foreground uses saturated colours. This determines
    # candidate pixels only; the VLM remains responsible for instance labels.
    mask = (region.max(axis=2)-region.min(axis=2) > 35)
    rows, cols = np.where(mask)
    rows += y0; cols += x0
    z = depth[rows, cols]
    valid = np.isfinite(z) & (z > .05) & (z < 2)
    rows, cols, z = rows[valid], cols[valid], z[valid]
    h, w = depth.shape
    f = h/(2*np.tan(np.deg2rad(camera['fovy_deg'])/2))
    points = np.column_stack(((cols+.5-w/2)*z/f, -(rows+.5-h/2)*z/f, -z))
    return points @ np.asarray(camera['rotation']).T + np.asarray(camera['position'])
