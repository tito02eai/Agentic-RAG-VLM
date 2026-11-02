"""Offline unit tests for the YAM motion planner.

Pick/place plans are checked for keyframe structure, geometric correctness
(verified through forward kinematics), interpolation bounds, and waypoint
budgets.  All targets are base-frame positions validated against the real
URDF-derived kinematics; no network or hardware is involved.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from agentic_rag_vlm.platform.kinematics import NUM_JOINTS, YamKinematics  # noqa: E402
from agentic_rag_vlm.platform.motion import (  # noqa: E402
    MotionPlanError,
    YamMotionPlanner,
)

HOME_JOINTS = (0.0,) * NUM_JOINTS
PICK_TARGET_M = (0.22, 0.00, 0.16)  # base frame, top-grasp reachable (validated)
PLACE_DEST_M = (0.20, -0.05, 0.15)
APERTURE_M = 0.07
OPEN_COMMAND = YamKinematics.gripper_command_for_aperture(APERTURE_M)


@pytest.fixture()
def planner() -> YamMotionPlanner:
    return YamMotionPlanner(YamKinematics())


def _plan_pick(planner: YamMotionPlanner):
    return planner.plan_pick(
        arm="right",
        joints_deg_current=HOME_JOINTS,
        gripper_current=0.5,
        target_xyz_m=PICK_TARGET_M,
        aperture_m=APERTURE_M,
        other_joints_deg=(1.0,) * NUM_JOINTS,
        other_gripper=0.4,
    )


def _plan_place(planner: YamMotionPlanner):
    return planner.plan_place(
        arm="right",
        joints_deg_current=HOME_JOINTS,
        gripper_current=0.02,
        destination_xyz_m=PLACE_DEST_M,
        aperture_m=APERTURE_M,
    )


def test_plan_pick_keyframe_structure(planner: YamMotionPlanner) -> None:
    plan = _plan_pick(planner)
    assert [frame.name for frame in plan.keyframes] == [
        "start",
        "approach_open",
        "descend",
        "close",
        "lift",
    ]
    grippers = [frame.gripper for frame in plan.keyframes]
    assert grippers[0] == pytest.approx(0.5)  # observed current gripper
    assert grippers[1] == pytest.approx(OPEN_COMMAND)
    assert grippers[2] == pytest.approx(OPEN_COMMAND)
    assert grippers[3] == pytest.approx(0.02)
    assert grippers[4] == pytest.approx(0.02)
    close = plan.keyframes[3]
    assert close.hold_steps == planner.hold_steps
    assert all(frame.hold_steps == 0 for frame in plan.keyframes[:3] + plan.keyframes[4:])
    assert plan.arm == "right"
    assert plan.other_joints_deg == (1.0,) * NUM_JOINTS
    assert plan.other_gripper == 0.4
    assert len(plan.notes) == 3  # one IK note per planned frame
    assert all(note.startswith("ik:right:top:") for note in plan.notes)


def test_plan_pick_keyframe_geometry_via_fk(planner: YamMotionPlanner) -> None:
    kinematics = planner.kinematics
    plan = _plan_pick(planner)
    frames = {frame.name: frame for frame in plan.keyframes}
    target = np.array(PICK_TARGET_M)
    expected = {
        "approach_open": target + np.array([0.0, 0.0, planner.approach_height_m]),
        "descend": target + np.array([0.0, 0.0, planner.grasp_offset_m]),
        "lift": target + np.array([0.0, 0.0, planner.lift_height_m]),
    }
    for name, position in expected.items():
        solved = kinematics.fk(np.radians(frames[name].joints_deg))
        assert np.allclose(solved[:3, 3], position, atol=2e-3), name
        # top grasp: tool +Z points down in the base frame
        assert np.allclose(solved[:3, :3][:, 2], [0.0, 0.0, -1.0], atol=2e-2), name


def test_plan_pick_interpolation_bounds_and_gripper(planner: YamMotionPlanner) -> None:
    plan = _plan_pick(planner)
    steps = plan.interpolated_steps(
        max_joint_delta_deg_per_step=planner.max_joint_delta_deg_per_step
    )
    assert 1 < len(steps) <= planner.max_plan_waypoints
    for (joints_a, _), (joints_b, _) in zip(steps, steps[1:]):
        delta = float(np.max(np.abs(np.asarray(joints_b) - np.asarray(joints_a))))
        assert delta <= planner.max_joint_delta_deg_per_step + 1e-9
    grippers = [gripper for _, gripper in steps]
    # starts at the observed gripper, opens toward the pre-grasp, closes at the grasp
    first_closed = grippers.index(0.02)
    assert grippers[0] == pytest.approx(0.5)
    floor = min(0.5, OPEN_COMMAND) - 1e-9
    assert all(value >= floor for value in grippers[:first_closed])
    assert all(value == pytest.approx(0.02) for value in grippers[first_closed:])
    dwell = [step for step, gripper in zip(steps, grippers) if gripper == pytest.approx(0.02)]
    assert len(dwell) >= planner.hold_steps


def test_plan_place_keyframe_structure_and_geometry(planner: YamMotionPlanner) -> None:
    kinematics = planner.kinematics
    plan = _plan_place(planner)
    assert [frame.name for frame in plan.keyframes] == [
        "start",
        "above_destination",
        "descend_to_destination",
        "release",
        "retract",
    ]
    frames = {frame.name: frame for frame in plan.keyframes}
    release = frames["release"]
    assert release.gripper == pytest.approx(OPEN_COMMAND)
    assert release.hold_steps == planner.hold_steps
    assert frames["above_destination"].gripper == pytest.approx(0.02)  # still holding
    destination = np.array(PLACE_DEST_M)
    expected = {
        "above_destination": destination + np.array([0.0, 0.0, planner.approach_height_m]),
        "descend_to_destination": destination + np.array([0.0, 0.0, planner.place_offset_m]),
        "retract": destination + np.array([0.0, 0.0, planner.approach_height_m]),
    }
    for name, position in expected.items():
        solved = kinematics.fk(np.radians(frames[name].joints_deg))
        assert np.allclose(solved[:3, 3], position, atol=2e-3), name
    steps = plan.interpolated_steps(
        max_joint_delta_deg_per_step=planner.max_joint_delta_deg_per_step
    )
    assert len(steps) <= planner.max_plan_waypoints


def test_plan_hold_is_a_frozen_noop(planner: YamMotionPlanner) -> None:
    plan = planner.plan_hold(
        arm="left",
        joints_deg_current=(5.0,) * NUM_JOINTS,
        gripper_current=0.3,
        other_joints_deg=(1.0,) * NUM_JOINTS,
        other_gripper=0.2,
    )
    assert len(plan.keyframes) == 1
    frame = plan.keyframes[0]
    assert frame.name == "hold"
    assert frame.joints_deg == (5.0,) * NUM_JOINTS
    assert frame.gripper == 0.3
    steps = plan.interpolated_steps(
        max_joint_delta_deg_per_step=planner.max_joint_delta_deg_per_step
    )
    assert steps == [((5.0,) * NUM_JOINTS, 0.3)]


def test_plan_pick_unreachable_target_raises(planner: YamMotionPlanner) -> None:
    with pytest.raises(MotionPlanError, match="IK failed"):
        planner.plan_pick(
            arm="right",
            joints_deg_current=HOME_JOINTS,
            gripper_current=0.5,
            target_xyz_m=(5.0, 0.0, 0.5),
            aperture_m=APERTURE_M,
        )


def test_plan_budget_is_enforced() -> None:
    tight = YamMotionPlanner(
        YamKinematics(),
        max_plan_waypoints=5,
    )
    with pytest.raises(MotionPlanError, match="budget"):
        tight.plan_pick(
            arm="right",
            joints_deg_current=HOME_JOINTS,
            gripper_current=0.5,
            target_xyz_m=PICK_TARGET_M,
            aperture_m=APERTURE_M,
        )
