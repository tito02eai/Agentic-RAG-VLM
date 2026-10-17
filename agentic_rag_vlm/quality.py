"""Seven-factor analytical quality model from Eq. (2) of the paper."""

from __future__ import annotations

from dataclasses import asdict, dataclass


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
    factor_threshold: float = 0.4,
    success_threshold: float = 0.65,
) -> QualityResult:
    values = factors.clipped()
    payload = asdict(values)
    score = sum(WEIGHTS[name] * value for name, value in payload.items())
    below = tuple(name for name, value in payload.items() if value < factor_threshold)
    return QualityResult(score=score, factors=values, below_threshold=below, success=score >= success_threshold and not below)
