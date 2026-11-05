"""Scene-change verification for platform sessions.

After the executor finishes a pick-and-place, the session re-observes the
workspace and this verifier compares public object positions before/after.
Success requires the target to have moved at least ``moved_threshold_m``,
every protected object to remain observed within ``protected_threshold_m``,
and, when supplied, the target to reach the destination in 3-D. The
evidence dictionary is passed into ``AgenticRuntime.verify`` and stored in
the session audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import dist, isfinite
from typing import Any, Mapping, Sequence

from scripts.agentic_framework import PublicObject, detect_scene_change


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    quality: float
    evidence: dict[str, Any] = field(default_factory=dict)


class SceneChangeVerifier:
    def __init__(
        self,
        *,
        moved_threshold_m: float = 0.05,
        protected_threshold_m: float = 0.015,
        destination_tolerance_m: float = 0.05,
    ) -> None:
        self.moved_threshold_m = float(moved_threshold_m)
        self.protected_threshold_m = float(protected_threshold_m)
        self.destination_tolerance_m = float(destination_tolerance_m)
        if not all(isfinite(value) and value > 0 for value in (
            self.moved_threshold_m, self.protected_threshold_m, self.destination_tolerance_m
        )):
            raise ValueError("verification thresholds must be finite and positive")

    def verify(
        self,
        target: str,
        protected_objects: Sequence[str],
        before: Mapping[str, PublicObject],
        after: Mapping[str, PublicObject],
        *,
        destination_xyz_m: Sequence[float] | None = None,
    ) -> VerificationResult:
        change = detect_scene_change(before, after, threshold_m=self.protected_threshold_m)
        displacements: Mapping[str, Any] = change["displacements"]
        problems: list[str] = []
        invalid_positions = sorted({
            name for name in (target, *protected_objects)
            for objects in (before, after)
            if name in objects and (
                len(objects[name].center_xyz_m) != 3
                or not all(isfinite(float(value)) for value in objects[name].center_xyz_m)
            )
        })
        if invalid_positions:
            problems.append("invalid_object_position")
        unobserved_protected = sorted({
            name for name in protected_objects if name not in before or name not in after
        })
        if unobserved_protected:
            problems.append("protected_object_not_observed")
        if target not in after:
            problems.append("target_not_observed_after_execution")
        elif target not in displacements:
            problems.append("target_displacement_unavailable")
        elif float(displacements[target]["distance_m"]) < self.moved_threshold_m:
            problems.append("target_did_not_move")
        protected_violations = [
            name
            for name in protected_objects
            if name in displacements and float(displacements[name]["distance_m"]) >= self.protected_threshold_m
        ]
        if protected_violations:
            problems.append("protected_object_displaced")
        # The shared scene-change monitor is planar. Safety verification also
        # catches vertical displacement (for example, a knocked-over neighbor).
        for name in protected_objects:
            if name in before and name in after and name not in invalid_positions:
                if dist(before[name].center_xyz_m, after[name].center_xyz_m) >= self.protected_threshold_m:
                    if name not in protected_violations:
                        protected_violations.append(name)
        if protected_violations and "protected_object_displaced" not in problems:
            problems.append("protected_object_displaced")
        destination_error_m = None
        if destination_xyz_m is not None:
            if len(destination_xyz_m) != 3 or not all(isfinite(float(v)) for v in destination_xyz_m):
                problems.append("invalid_destination_position")
            elif target in after and target not in invalid_positions:
                destination_error_m = dist(after[target].center_xyz_m, destination_xyz_m)
                if destination_error_m > self.destination_tolerance_m:
                    problems.append("target_outside_destination")
        success = not problems
        return VerificationResult(
            success=success,
            quality=1.0 if success else 0.0,
            evidence={
                "target": target,
                "problems": problems,
                "moved_threshold_m": self.moved_threshold_m,
                "protected_threshold_m": self.protected_threshold_m,
                "protected_violations": protected_violations,
                "unobserved_protected": unobserved_protected,
                "invalid_positions": invalid_positions,
                "destination_xyz_m": list(destination_xyz_m) if destination_xyz_m is not None else None,
                "destination_error_m": destination_error_m,
                "destination_tolerance_m": self.destination_tolerance_m,
                "verification_scope": "destination_and_displacement" if destination_xyz_m is not None else "displacement_only",
                "displacements": {
                    name: dict(item) for name, item in displacements.items()
                },
            },
        )
