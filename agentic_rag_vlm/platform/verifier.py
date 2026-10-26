"""Scene-change verification for platform sessions.

After the executor finishes a pick-and-place, the session re-observes the
workspace and this verifier compares public object positions before/after.
Success requires the target to have moved at least ``moved_threshold_m`` and
every protected object to have stayed within ``protected_threshold_m``.  The
evidence dictionary is passed into ``AgenticRuntime.verify`` and stored in
the session audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    ) -> None:
        self.moved_threshold_m = float(moved_threshold_m)
        self.protected_threshold_m = float(protected_threshold_m)

    def verify(
        self,
        target: str,
        protected_objects: Sequence[str],
        before: Mapping[str, PublicObject],
        after: Mapping[str, PublicObject],
    ) -> VerificationResult:
        change = detect_scene_change(before, after, threshold_m=self.protected_threshold_m)
        displacements: Mapping[str, Any] = change["displacements"]
        problems: list[str] = []
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
                "displacements": {
                    name: dict(item) for name, item in displacements.items()
                },
            },
        )
