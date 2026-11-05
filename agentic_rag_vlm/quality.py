"""Seven-factor analytical quality model from Eq. (2) of the paper."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping
import numpy as np


WEIGHTS = {
    "position_accuracy": 0.20,
    "width_compatibility": 0.15,
    "force_appropriateness": 0.10,
    "grasp_type_match": 0.15,
    "approach_clearance": 0.15,
    "object_difficulty": 0.10,
    "grip_security": 0.15,
}


@dataclass(frozen=True)
class QualityFactors:
    position_accuracy: float
    width_compatibility: float
    force_appropriateness: float
    grasp_type_match: float
    approach_clearance: float
    object_difficulty: float
    grip_security: float

    def clipped(self) -> "QualityFactors":
        if not all(math.isfinite(float(value)) for value in asdict(self).values()):
            raise ValueError("quality factors must be finite measured/analytical values")
        return QualityFactors(**{key: min(1.0, max(0.0, float(value))) for key, value in asdict(self).items()})


@dataclass(frozen=True)
class QualityResult:
    score: float
    factors: QualityFactors
    below_threshold: tuple[str, ...]
    success: bool


def evaluate_quality(
    factors: QualityFactors,
    *,
    factor_threshold: float | Mapping[str,float] = 0.4,
    success_threshold: float = 0.65,
    observed_success: bool | None = None,
) -> QualityResult:
    values = factors.clipped()
    if observed_success is not None and not isinstance(observed_success,(bool,np.bool_)):
        raise ValueError("physical outcome must be a boolean")
    payload = asdict(values)
    score = sum(WEIGHTS[name] * value for name, value in payload.items())
    thresholds = dict(factor_threshold) if isinstance(factor_threshold,Mapping) else {name:factor_threshold for name in WEIGHTS}
    if set(thresholds) != set(WEIGHTS) or not all(math.isfinite(v) and 0 <= v <= 1 for v in thresholds.values()):
        raise ValueError("one finite [0,1] threshold is required per quality factor")
    if not math.isfinite(success_threshold) or not 0 <= success_threshold <= 1:
        raise ValueError("invalid aggregate quality threshold")
    below = tuple(name for name, value in payload.items() if value < thresholds[name])
    return QualityResult(score=score, factors=values, below_threshold=below,
        success=score >= success_threshold and not below and (observed_success is None or bool(observed_success)))
