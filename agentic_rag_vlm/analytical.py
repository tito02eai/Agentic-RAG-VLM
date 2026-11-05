"""Explicit reconstructed factor formulas; paper publishes only factor names/weights.

Inputs are supplied by an evaluator/sensor adapter. No synthetic noise or task
identity determines success. These formulas are not claimed original formulas.
"""
from dataclasses import dataclass, asdict
import math
from .quality import QualityFactors,evaluate_quality


@dataclass(frozen=True)
class GraspMeasurements:
    position_error_m: float
    commanded_width_m: float
    object_width_m: float
    grip_force_n: float
    minimum_secure_force_n: float
    maximum_safe_force_n: float
    type_match: float
    clearance_m: float
    object_difficulty: float
    grip_security: float
    physical_success: bool


def analytical_factors(m: GraspMeasurements, *, position_tolerance_m=.02, width_tolerance_m=.02,
                       required_clearance_m=.03):
    if not all(math.isfinite(float(v)) for k,v in asdict(m).items() if k != "physical_success"):
        raise ValueError("nonfinite grasp measurement")
    if min(position_tolerance_m,width_tolerance_m,required_clearance_m) <= 0:
        raise ValueError("positive measurement scales required")
    if not 0 < m.minimum_secure_force_n <= m.maximum_safe_force_n or m.object_width_m <= 0:
        raise ValueError("measured force limits and object width required")
    clip = lambda x: min(1.,max(0.,x))
    force = (m.grip_force_n/m.minimum_secure_force_n if m.grip_force_n < m.minimum_secure_force_n
             else 1. if m.grip_force_n <= m.maximum_safe_force_n
             else m.maximum_safe_force_n/max(m.grip_force_n,1e-9))
    return QualityFactors(clip(1-m.position_error_m/position_tolerance_m),
        clip(1-abs(m.commanded_width_m-m.object_width_m)/width_tolerance_m),clip(force),
        clip(m.type_match),clip(m.clearance_m/required_clearance_m),
        clip(1-m.object_difficulty),clip(m.grip_security))


def evaluate_measurements(measurements, **kwargs):
    return evaluate_quality(analytical_factors(measurements),observed_success=measurements.physical_success,**kwargs)
