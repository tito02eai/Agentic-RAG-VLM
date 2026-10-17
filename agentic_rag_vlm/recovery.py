"""Fourteen-type failure taxonomy and deterministic L1/L2/L3 recovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping


class FailureType(str, Enum):
    POSITION_ERROR = "position_error"
    UNREACHABLE_POSE = "unreachable_pose"
    APPROACH_ANGLE = "approach_angle"
    WIDTH_MISMATCH = "width_mismatch"
    COLLISION = "collision"
    ORIENTATION = "orientation"
    SLIP = "slip"
    DROP = "drop"
    FORCE_DAMAGE = "force_damage"
    DEFORMATION = "deformation"
    WRONG_OBJECT = "wrong_object"
    CONSTRAINT_VIOLATION = "constraint_violation"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RecoveryDecision:
    level: int
    failure: FailureType
    corrected_strategy: dict[str, Any]
    action: str
    safe_stop: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["failure"] = self.failure.value
        return payload


FACTOR_TO_FAILURE = {
    "position_accuracy": FailureType.POSITION_ERROR,
    "width_compatibility": FailureType.WIDTH_MISMATCH,
    "force_appropriateness": FailureType.FORCE_DAMAGE,
    "grasp_type_match": FailureType.ORIENTATION,
    "approach_clearance": FailureType.COLLISION,
    "object_difficulty": FailureType.UNREACHABLE_POSE,
    "grip_security": FailureType.SLIP,
}


def classify_failure(observation: Mapping[str, Any]) -> FailureType:
    explicit = observation.get("failure_type")
    if explicit in FailureType._value2member_map_:
        return FailureType(explicit)
    below = observation.get("below_threshold", [])
    return FACTOR_TO_FAILURE.get(below[0], FailureType.UNKNOWN) if below else FailureType.UNKNOWN


def _rotate_synergy(current: str) -> str:
    order = ("power", "pinch", "side")
    return order[(order.index(current) + 1) % len(order)] if current in order else "power"


def recover(strategy: Mapping[str, Any], failure: FailureType, attempt: int) -> RecoveryDecision:
    corrected = dict(strategy)
    if attempt <= 1:
        if failure in {FailureType.SLIP, FailureType.DROP}:
            corrected["force_n"] = 1.2 * float(corrected.get("force_n", 10.0))
        elif failure == FailureType.WIDTH_MISMATCH:
            corrected["aperture_m"] = float(corrected.get("aperture_m", 0.06)) + 0.01
        elif failure in {FailureType.COLLISION, FailureType.CONSTRAINT_VIOLATION}:
            corrected["approach_height_delta_m"] = float(corrected.get("approach_height_delta_m", 0.0)) + 0.03
        elif failure in {FailureType.FORCE_DAMAGE, FailureType.DEFORMATION}:
            corrected["force_n"] = 0.8 * float(corrected.get("force_n", 10.0))
        elif failure == FailureType.POSITION_ERROR:
            corrected["position_retry"] = True
        elif failure == FailureType.TIMEOUT:
            return RecoveryDecision(1, failure, corrected, "safe_stop", True)
        else:
            corrected["parameter_retry"] = True
        return RecoveryDecision(1, failure, corrected, "parameter_tuning")
    if attempt == 2:
        corrected["synergy"] = _rotate_synergy(str(corrected.get("synergy", "power")))
        corrected["force_n"] = 0.9 * float(corrected.get("force_n", 10.0))
        return RecoveryDecision(2, failure, corrected, "method_switch")
    corrected["full_replan"] = True
    corrected["deterministic_perturbation_m"] = [0.005, -0.005, 0.005]
    return RecoveryDecision(3, failure, corrected, "full_replan")
