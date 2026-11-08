"""GraspAction-to-joint-trajectory motion planning for YAM arms.

This module is the deployment's deterministic motion adapter: it converts the
pipeline's symbolic grasp output (target position, approach, aperture) into
platform waypoints through inverse kinematics and joint-space interpolation.
It contains no learned policy and emits no reactive control loop — it is a
geometric planner whose every keyframe is auditable, in the same spirit as the
IK/skill layer used in the VLABench experiments.

A pick has five stages (current pose, pre-grasp above the target,
descend, gripper close, lift) and a place as five more (current pose, above
destination, descend, gripper open, retract). Native deployment inserts
Cartesian IK waypoints into contact approach, lift and retract segments.
Including the observed current
pose as the first keyframe guarantees the executed trajectory starts with a
bounded step from where the arm actually is.  Between keyframes the commanded
joints interpolate at
a bounded per-step delta, and the resulting waypoints are packed at the
platform's 10 Hz cadence.  The non-operating arm repeats its initially
observed joints for the whole trajectory and is never moved by this planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from .kinematics import NUM_JOINTS, YamKinematics


class MotionPlanError(RuntimeError):
    """Raised when a motion plan cannot be produced safely."""


APPROACH_DIRECTIONS: dict[str, tuple[float, float, float]] = {
    # Tool +Z travel direction, expressed in the arm base frame
    # (x forward, y left, z up).
    "top": (0.0, 0.0, -1.0),
    "side": (1.0, 0.0, 0.0),
}


@dataclass(frozen=True)
class MotionKeyframe:
    name: str
    joints_deg: tuple[float, ...]
    gripper: float | None  # operating-arm gripper command (0 closed .. 1 open)
    hold_steps: int = 0  # extra dwell steps at this keyframe (gripper actuation)


@dataclass
class MotionPlan:
    arm: str
    keyframes: list[MotionKeyframe]
    other_joints_deg: tuple[float, ...]
    other_gripper: float | None
    cadence_hz: float = 10.0
    notes: list[str] = field(default_factory=list)
    velocity_scale: float = 1.0

    def interpolated_steps(
        self, *, max_joint_delta_deg_per_step: float
    ) -> list[tuple[tuple[float, ...], float | None]]:
        """Joint-space interpolation between keyframes.

        Returns ``(joints_deg, gripper)`` samples at the cadence rate; the
        gripper ramps linearly across the segment so the fingers move with the
        arm instead of snapping.
        """
        if not self.keyframes:
            raise MotionPlanError("plan has no keyframes")
        if not np.isfinite(self.velocity_scale) or not 0 < self.velocity_scale <= 1:
            raise MotionPlanError("invalid velocity scale")
        if not np.isfinite(max_joint_delta_deg_per_step) or max_joint_delta_deg_per_step <= 0:
            raise MotionPlanError("joint waypoint delta must be finite and positive")
        max_joint_delta_deg_per_step *= self.velocity_scale
        steps: list[tuple[tuple[float, ...], float | None]] = []
        previous = self.keyframes[0]
        steps.append((previous.joints_deg, previous.gripper))
        for current in self.keyframes[1:]:
            delta = np.asarray(current.joints_deg) - np.asarray(previous.joints_deg)
            max_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
            segments = max(
                1,
                math.ceil(max_delta / max_joint_delta_deg_per_step),
            )
            gripper_from = previous.gripper
            gripper_to = current.gripper
            for index in range(1, segments + 1):
                fraction = index / segments
                joints = (np.asarray(current.joints_deg) if index == segments else
                          np.asarray(previous.joints_deg) + fraction * delta)
                gripper = gripper_to
                if gripper_from is not None and gripper_to is not None and index < segments:
                    gripper = gripper_from + fraction * (gripper_to - gripper_from)
                steps.append((tuple(float(value) for value in joints), gripper))
            for _ in range(current.hold_steps):
                steps.append((current.joints_deg, current.gripper))
            previous = current
        return steps


@dataclass
class YamMotionPlanner:
    """Plans pick/place trajectories for one YAM arm via IK."""

    kinematics: YamKinematics
    approach_height_m: float = 0.10
    grasp_offset_m: float = 0.02  # TCP hover above the object centre at grasp
    lift_height_m: float = 0.08
    place_offset_m: float = 0.02  # hover above the destination before release
    max_joint_delta_deg_per_step: float = 3.0  # 30 deg/s at 10 Hz
    gripper_settle_s: float = 1.0
    cadence_hz: float = 10.0
    max_plan_waypoints: int = 1200  # keep headroom under the 3000 session cap
    observation_park_xyz_m: Sequence[float] | None = None
    cartesian_step_m: float | None = None

    def __post_init__(self) -> None:
        if self.cartesian_step_m is not None and (
            not np.isfinite(self.cartesian_step_m) or not .001 <= self.cartesian_step_m <= .01
        ):
            raise MotionPlanError("Cartesian waypoint spacing must be between 1 and 10 mm")
        self.hold_steps = max(1, round(self.gripper_settle_s * self.cadence_hz))

    @classmethod
    def from_deployment(cls, calibration, profile):
        """Use the same measured motion settings in the Runner and CLI."""
        from .native_kinematics import NativeYamKinematics
        return cls(
            NativeYamKinematics(calibration, profile['arm'], profile['joint_limits_deg'],
                                max_aperture_m=profile['gripper']['max_aperture_m']),
            grasp_offset_m=0., place_offset_m=0., cartesian_step_m=.01,
            max_joint_delta_deg_per_step=float(profile.get('max_joint_delta_deg_per_step', 3.)),
            lift_height_m=float(profile.get('lift_height_m', .08)),
            observation_park_xyz_m=profile.get('observation_park_xyz_m'),
        )

    # ------------------------------------------------------------- helpers

    def _ik_frame(
        self,
        *,
        arm: str,
        joints_deg_current: Sequence[float],
        position_m: np.ndarray,
        approach: str,
        notes: list[str],
    ) -> tuple[float, ...]:
        direction = np.array(APPROACH_DIRECTIONS.get(approach, APPROACH_DIRECTIONS["top"]))
        rotation = self.kinematics.orientation_for_approach(direction)
        result = self.kinematics.ik(
            position_m, rotation, q_init_rad=np.radians(np.asarray(joints_deg_current, dtype=float))
        )
        if not result.success:
            # The first seed can stop at a limit or singular configuration.
            # Try a bounded set without relaxing the solver tolerances.
            initial = np.radians(np.asarray(joints_deg_current, dtype=float))
            limits = np.radians(self.kinematics.limits_deg)
            middle = limits.mean(axis=1)
            candidates = []
            for seed in (middle, (middle + initial) / 2, np.zeros(NUM_JOINTS)):
                candidate = self.kinematics.ik(position_m, rotation, q_init_rad=seed)
                if candidate.success and candidate.within_limits:
                    candidates.append(candidate)
            if candidates:
                result = min(candidates, key=lambda item: np.linalg.norm(item.joints_rad - initial))
        if not result.success:
            raise MotionPlanError(
                f"{arm}-arm IK failed for approach '{approach}' at "
                f"{np.round(position_m, 3).tolist()}: {result.message} "
                f"(pos_err={result.position_error_m:.4f} m, ori_err={result.orientation_error_rad:.4f} rad)"
            )
        if not result.within_limits:
            raise MotionPlanError(
                f"{arm}-arm IK solution outside joint limits for {np.round(position_m, 3).tolist()}"
            )
        notes.append(
            f"ik:{arm}:{approach}:pos_err={result.position_error_m:.5f}:ori_err={result.orientation_error_rad:.5f}:iters={result.iterations}"
        )
        return tuple(float(value) for value in np.degrees(result.joints_rad))

    def _check_budget(self, plan: MotionPlan) -> MotionPlan:
        steps = plan.interpolated_steps(max_joint_delta_deg_per_step=self.max_joint_delta_deg_per_step)
        if len(steps) > self.max_plan_waypoints:
            raise MotionPlanError(
                f"planned {len(steps)} waypoints; budget is {self.max_plan_waypoints}"
            )
        return plan

    def _approach_frames(self, *, arm, current, goal, approach, name, gripper, notes):
        """Solve consecutive Cartesian targets before dispatching any motion."""
        start = self.kinematics.fk(np.radians(current))[:3, 3]
        count = (1 if self.cartesian_step_m is None else
                 max(1, math.ceil(np.linalg.norm(goal-start)/self.cartesian_step_m)))
        frames = []
        for index in range(1, count+1):
            position = goal if index == count else start+(goal-start)*index/count
            current = self._ik_frame(arm=arm, joints_deg_current=current,
                                    position_m=position, approach=approach, notes=notes)
            label = name if index == count else f"{name}_path_{index}"
            frames.append(MotionKeyframe(label, current, gripper))
        return frames

    # --------------------------------------------------------------- pick

    def plan_pick(
        self,
        *,
        arm: str,
        joints_deg_current: Sequence[float],
        gripper_current: float | None,
        target_xyz_m: Sequence[float],
        aperture_m: float,
        approach: str = "top",
        approach_height_delta_m: float = 0.0,
        approach_offset_xy_m: Sequence[float] = (0.0, 0.0),
        velocity_scale: float = 1.0,
        other_joints_deg: Sequence[float] = (),
        other_gripper: float | None = None,
    ) -> MotionPlan:
        """Pre-grasp -> descend -> close -> lift around the target object."""
        notes: list[str] = []
        if approach not in APPROACH_DIRECTIONS:
            raise MotionPlanError(f"unsupported approach: {approach}")
        if not np.isfinite(approach_height_delta_m) or not 0 <= approach_height_delta_m <= 0.15:
            raise MotionPlanError("invalid approach height correction")
        offset = np.asarray(approach_offset_xy_m, dtype=float)
        if offset.shape != (2,) or not np.isfinite(offset).all() or np.linalg.norm(offset) > 0.15:
            raise MotionPlanError("invalid approach offset")
        target = np.asarray(target_xyz_m, dtype=float)
        approach_dir = np.array(APPROACH_DIRECTIONS.get(approach, APPROACH_DIRECTIONS["top"]))
        open_command = self.kinematics.gripper_command_for_aperture(aperture_m)
        current = tuple(float(value) for value in joints_deg_current)

        pregrasp = self._ik_frame(
            arm=arm,
            joints_deg_current=current,
            position_m=target - approach_dir * (self.approach_height_m + approach_height_delta_m) + np.r_[offset, 0],
            approach=approach,
            notes=notes,
        )
        descent = self._approach_frames(
            arm=arm,
            current=pregrasp,
            goal=target - approach_dir * self.grasp_offset_m,
            approach=approach,
            name="descend",
            gripper=open_command,
            notes=notes,
        )
        grasp = descent[-1].joints_deg
        lift = self._approach_frames(
            arm=arm,
            current=grasp,
            goal=target - approach_dir * self.lift_height_m,
            approach=approach,
            name="lift",
            gripper=0.02,
            notes=notes,
        )
        start_gripper = gripper_current if gripper_current is not None else open_command
        keyframes = [
            MotionKeyframe("start", current, start_gripper),
            MotionKeyframe("approach_open", pregrasp, open_command),
            *descent,
            MotionKeyframe("close", grasp, 0.02, hold_steps=self.hold_steps),
            *lift,
        ]
        plan = MotionPlan(
            arm=arm,
            keyframes=keyframes,
            other_joints_deg=tuple(float(value) for value in other_joints_deg),
            other_gripper=other_gripper,
            cadence_hz=self.cadence_hz,
            notes=notes,
            velocity_scale=velocity_scale,
        )
        return self._check_budget(plan)

    # --------------------------------------------------------------- place

    def plan_place(
        self,
        *,
        arm: str,
        joints_deg_current: Sequence[float],
        gripper_current: float | None,
        destination_xyz_m: Sequence[float],
        aperture_m: float,
        approach: str = "top",
        velocity_scale: float = 1.0,
        other_joints_deg: Sequence[float] = (),
        other_gripper: float | None = None,
        hold_gripper_command: float | None = None,
    ) -> MotionPlan:
        """Above destination -> descend -> open -> retract."""
        notes: list[str] = []
        destination = np.asarray(destination_xyz_m, dtype=float)
        approach_dir = np.array(APPROACH_DIRECTIONS.get(approach, APPROACH_DIRECTIONS["top"]))
        open_command = self.kinematics.gripper_command_for_aperture(aperture_m)
        current = tuple(float(value) for value in joints_deg_current)

        # A blocked measured aperture must not replace the prior close command.
        hold_gripper = gripper_current if hold_gripper_command is None else hold_gripper_command
        if hold_gripper is None:
            hold_gripper = 0.02
        if not np.isfinite(hold_gripper) or not 0 <= hold_gripper <= 1:
            raise MotionPlanError('invalid gripper hold command')

        above = self._ik_frame(
            arm=arm,
            joints_deg_current=current,
            position_m=destination - approach_dir * self.approach_height_m,
            approach=approach,
            notes=notes,
        )
        descent = self._approach_frames(
            arm=arm,
            current=above,
            goal=destination - approach_dir * self.place_offset_m,
            approach=approach,
            name="descend_to_destination",
            gripper=hold_gripper,
            notes=notes,
        )
        down = descent[-1].joints_deg
        retract = self._approach_frames(
            arm=arm,
            current=down,
            goal=destination - approach_dir * self.approach_height_m,
            approach=approach,
            name="retract",
            gripper=open_command,
            notes=notes,
        )
        keyframes = [
            MotionKeyframe("start", current, hold_gripper),
            MotionKeyframe("above_destination", above, hold_gripper),
            *descent,
            MotionKeyframe("release", down, open_command, hold_steps=self.hold_steps),
            *retract,
        ]
        if self.observation_park_xyz_m is not None:
            park=np.asarray(self.observation_park_xyz_m,dtype=float)
            if park.shape!=(3,) or not np.isfinite(park).all():raise MotionPlanError('invalid observation park position')
            parked=self._ik_frame(arm=arm,joints_deg_current=retract[-1].joints_deg,position_m=park,approach=approach,notes=notes)
            keyframes.append(MotionKeyframe('clear_observation',parked,open_command))
        plan = MotionPlan(
            arm=arm,
            keyframes=keyframes,
            other_joints_deg=tuple(float(value) for value in other_joints_deg),
            other_gripper=other_gripper,
            cadence_hz=self.cadence_hz,
            notes=notes,
            velocity_scale=velocity_scale,
        )
        return self._check_budget(plan)

    # --------------------------------------------------------------- home

    def plan_hold(
        self,
        *,
        arm: str,
        joints_deg_current: Sequence[float],
        gripper_current: float | None,
        steps: int = 1,
        other_joints_deg: Sequence[float] = (),
        other_gripper: float | None = None,
    ) -> MotionPlan:
        """Hold the current pose (safety no-op with a receipt)."""
        keyframe = MotionKeyframe("hold", tuple(float(v) for v in joints_deg_current), gripper_current)
        plan = MotionPlan(
            arm=arm,
            keyframes=[keyframe],
            other_joints_deg=tuple(float(value) for value in other_joints_deg),
            other_gripper=other_gripper,
            cadence_hz=self.cadence_hz,
            notes=["hold"],
        )
        if steps > self.max_plan_waypoints:
            raise MotionPlanError("hold steps exceed budget")
        return plan
