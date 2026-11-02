"""Offline unit tests for the YAM 6-DOF kinematics layer.

The forward-kinematics anchor at the zero pose is checked against the official
I2RT model README; everything else (IK roundtrips, approach orientation,
joint-limit helpers, gripper mapping) is verified without any hardware.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from agentic_rag_vlm.platform.kinematics import (  # noqa: E402
    NUM_JOINTS,
    YamKinematics,
    minimal_rotation_to_axis,
)

# I2RT model README anchor for q = 0 (gripper mount frame).
HOME_MOUNT_XYZ_M = np.array([0.110597512463, 0.0, 0.173501403591])
GRIPPER_TRAVEL_M = 2 * 0.04695


@pytest.fixture()
def kinematics() -> YamKinematics:
    return YamKinematics()


def test_fk_home_mount_anchor_matches_i2rt_readme(kinematics: YamKinematics) -> None:
    mount = kinematics.fk_mount(np.zeros(NUM_JOINTS))
    assert np.allclose(mount[:3, 3], HOME_MOUNT_XYZ_M, atol=1e-5)


def test_fk_home_mount_orientation_columns(kinematics: YamKinematics) -> None:
    mount = kinematics.fk_mount(np.zeros(NUM_JOINTS))
    rotation = mount[:3, :3]
    # mount +Z along base +X, +X along base -Y, +Y along base -Z
    assert np.allclose(rotation[:, 2], [1.0, 0.0, 0.0], atol=1e-5)
    assert np.allclose(rotation[:, 0], [0.0, -1.0, 0.0], atol=1e-5)
    assert np.allclose(rotation[:, 1], [0.0, 0.0, -1.0], atol=1e-5)
    # remains a valid rotation
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)


def test_fk_tcp_offset_extends_along_mount_z(kinematics: YamKinematics) -> None:
    mount = kinematics.fk_mount(np.zeros(NUM_JOINTS))
    tcp = kinematics.fk(np.zeros(NUM_JOINTS))
    expected = mount[:3, 3] + kinematics.tcp_offset_m * mount[:3, :3][:, 2]
    assert np.allclose(tcp[:3, 3], expected, atol=1e-9)


@pytest.mark.parametrize(
    "deltas_rad",
    [
        [0.2, 0.3, 0.2, 0.2, 0.3, 0.2],
        [-0.3, 0.4, 0.3, -0.2, 0.4, -0.3],
    ],
)
def test_ik_roundtrip_recovers_target_pose(kinematics: YamKinematics, deltas_rad: list[float]) -> None:
    truth = np.asarray(deltas_rad, dtype=float)
    target = kinematics.fk(truth)
    result = kinematics.ik(target[:3, 3], target[:3, :3], q_init_rad=np.zeros(NUM_JOINTS))
    assert result.success
    assert result.position_error_m <= kinematics.ik_position_tol_m
    assert result.orientation_error_rad <= kinematics.ik_orientation_tol_rad
    assert result.within_limits
    recovered = kinematics.fk(result.joints_rad)
    assert np.allclose(recovered[:3, 3], target[:3, 3], atol=1e-3)


def test_ik_top_grasp_converges_within_limits(kinematics: YamKinematics) -> None:
    target_position = np.array([0.25, 0.02, 0.20])
    rotation = kinematics.orientation_for_approach(np.array([0.0, 0.0, -1.0]))
    result = kinematics.ik(target_position, rotation, q_init_rad=np.zeros(NUM_JOINTS))
    assert result.success
    assert result.within_limits
    assert result.iterations < 50
    solved = kinematics.fk(result.joints_rad)
    assert np.allclose(solved[:3, 3], target_position, atol=2e-3)
    # tool +Z must point down (base -Z) for a top grasp
    assert np.allclose(solved[:3, :3][:, 2], [0.0, 0.0, -1.0], atol=2e-2)


@pytest.mark.parametrize("direction", [(0.0, 0.0, -1.0), (1.0, 0.0, 0.0)])
def test_orientation_for_approach_maps_tool_z(kinematics: YamKinematics, direction: tuple[float, float, float]) -> None:
    home = kinematics.fk(np.zeros(NUM_JOINTS))
    desired = kinematics.orientation_for_approach(np.array(direction))
    assert np.allclose(desired[:, 2], direction, atol=1e-9)
    # minimal rotation keeps the other axes close to home (small wrist twist)
    angle = math.acos(max(-1.0, min(1.0, (np.trace(desired @ home[:3, :3].T) - 1.0) / 2.0)))
    assert angle <= math.pi + 1e-9


def test_joint_limit_helpers(kinematics: YamKinematics) -> None:
    assert kinematics.in_limits(np.zeros(NUM_JOINTS))
    out_of_range = np.zeros(NUM_JOINTS)
    out_of_range[0] = -3.0  # joint1 lower limit is -2.61799 rad
    assert not kinematics.in_limits(out_of_range)
    clamped = kinematics.clamp_to_limits(out_of_range)
    assert kinematics.in_limits(clamped)
    assert clamped[0] == pytest.approx(-2.61799, abs=1e-6)


def test_gripper_aperture_mapping(kinematics: YamKinematics) -> None:
    assert YamKinematics.gripper_command_for_aperture(GRIPPER_TRAVEL_M) == pytest.approx(1.0)
    assert YamKinematics.gripper_command_for_aperture(0.0) == pytest.approx(0.02)  # clamped floor
    mid = YamKinematics.gripper_command_for_aperture(0.05)
    assert mid == pytest.approx(0.05 / GRIPPER_TRAVEL_M)
    assert YamKinematics.aperture_for_gripper_command(mid) == pytest.approx(0.05, abs=1e-9)


def test_minimal_rotation_identity_and_opposite() -> None:
    axis = np.array([0.0, 0.0, 1.0])
    assert np.allclose(minimal_rotation_to_axis(axis, axis), np.eye(3), atol=1e-12)
    opposite = minimal_rotation_to_axis(axis, -axis)
    assert np.allclose(opposite @ axis, -axis, atol=1e-9)
    assert np.allclose(opposite.T @ opposite, np.eye(3), atol=1e-9)
    tilted = minimal_rotation_to_axis(axis, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(tilted @ axis, [1.0, 0.0, 0.0], atol=1e-9)
