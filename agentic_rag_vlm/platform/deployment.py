"""Measured station profile admission and read-only deployment checks."""
from __future__ import annotations

from typing import Any, Mapping
import time
from datetime import datetime, timezone
import numpy as np
from urllib.parse import urlsplit

from .geometry import GeometryError, PlaneProjector
from .native_kinematics import NativeYamKinematics


def validate_profile_fields(profile: Mapping[str, Any], *, robot_id: str, arm: str) -> None:
    """Validate local execution configuration without network or calibration guesses."""
    if profile.get("robot_id") != robot_id or profile.get("arm") != arm:
        raise GeometryError("deployment profile does not match robot and operating arm")
    if arm not in {'left','right'} or not profile.get('calibration_id'):
        raise GeometryError('operating arm and calibration identity required')
    if profile.get("coordinate_frame") != "arm_base":
        raise GeometryError("first delivery requires arm_base task and observation coordinates")
    environment=profile.get('execution_environment','hardware')
    if environment not in {'hardware','simulation'}:raise GeometryError('unknown execution environment')
    validation=profile.get('simulation_validation' if environment=='simulation' else 'physical_validation',{})
    if validation.get('passed') is not True:
        raise GeometryError('simulation model validation is pending' if environment=='simulation' else 'physical calibration validation is pending')
    if not validation.get('record'):raise GeometryError('physical validation record/reference required' if environment=='hardware' else 'simulation validation record required')
    if environment=='simulation' and profile.get('physical_validation',{}).get('passed') is True:
        raise GeometryError('simulation must not claim physical validation')
    plane = profile["support_plane"]
    normal=np.asarray(plane['normal_base'],dtype=float)
    if normal.shape!=(3,) or not np.allclose(normal, [0, 0, 1]) or not np.isfinite(plane['offset_m']):
        raise GeometryError("first-delivery scene graph requires a horizontal measured support plane")
    limits=np.asarray(profile['joint_limits_deg'],dtype=float)
    if limits.shape!=(6,2) or not np.isfinite(limits).all() or np.any(limits[:,0]>=limits[:,1]):
        raise GeometryError('joint_limits_deg requires six finite min/max pairs')
    bounds = np.asarray(profile["workspace_bounds_m"], dtype=float)
    if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise GeometryError("workspace_bounds_m requires measured XYZ min/max bounds")
    heights = profile.get("feature_heights_m", {})
    if not heights or not all(np.isfinite(v) and 0 <= v <= 0.5 for v in heights.values()):
        raise GeometryError("measured feature heights required")
    offsets = profile.get("grasp_offsets_base_m", {})
    for name in heights:
        value = np.asarray(offsets.get(name), dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all() or np.linalg.norm(value) > .15:
            raise GeometryError(f"measured top-feature to grasp-frame offset required for {name}")
    aperture = profile.get("gripper", {}).get("max_aperture_m", 0)
    if not isinstance(aperture, (int, float)) or not np.isfinite(aperture) or not 0 < aperture <= .2:
        raise GeometryError("measured gripper max_aperture_m required")
    age = float(profile.get("max_observation_age_s", 2))
    if not np.isfinite(age) or not 0 < age <= 2:
        raise GeometryError("freshness bound must be within two seconds")
    if profile.get("approach") != "top":
        raise GeometryError("first delivery only admits a validated top approach")
    if 'max_joint_delta_deg_per_step' in profile:
        delta=profile['max_joint_delta_deg_per_step']
        if type(delta) not in (int,float) or not np.isfinite(delta) or not 0<delta<=3.:
            raise GeometryError('site waypoint delta must be positive and at most 3 degrees at 10 Hz')
    if 'lift_height_m' in profile and not .05<=float(profile['lift_height_m'])<=.25:
        raise GeometryError('lift height must be within configured 5–25 cm envelope')
    gate=profile.get('retention_observation')
    if gate and not (0<float(gate['minimum_blocked_aperture'])<float(gate['maximum_blocked_aperture'])<1
                    and .02<=float(gate['minimum_tcp_lift_m'])<=.15):
        raise GeometryError('invalid measured gripper retention proxy limits')


def validate_profile(profile: Mapping[str, Any], calibration: Mapping[str, Any],
                     *, robot_id: str, arm: str) -> None:
    validate_profile_fields(profile,robot_id=robot_id,arm=arm)
    if profile.get("calibration_id") != calibration.get("calibration_id"):
        raise GeometryError("deployment calibration identity mismatch")
    if calibration.get('quality',{}).get('failures'):
        raise GeometryError("published calibration has validation failures")
    environment=profile.get('execution_environment','hardware')
    if environment=='simulation' and calibration.get('execution_environment')!='simulation':
        raise GeometryError('simulation profile requires simulator calibration')
    if environment=='hardware' and calibration.get('execution_environment')=='simulation':
        raise GeometryError('simulator calibration cannot qualify hardware')
    plane=profile['support_plane']
    from .camera_contract import projection_calibration
    PlaneProjector(projection_calibration(calibration,profile),arm,plane_normal_base=plane['normal_base'],plane_offset_m=plane['offset_m'])
    NativeYamKinematics(calibration,arm,profile['joint_limits_deg'])


def validate_execution_binding(profile,endpoint,*,session_api=None):
    if profile.get('execution_environment','hardware')!='simulation':return
    parsed=urlsplit(endpoint or '')
    if parsed.scheme!='http' or parsed.hostname not in {'127.0.0.1','::1','localhost'}:
        raise GeometryError('simulation execution is restricted to an explicit loopback gateway')
    if session_api is not None and getattr(session_api,'local_simulation_only',False) is not True:
        raise GeometryError('simulation profile cannot bind a public robot SDK')
    if session_api is not None and getattr(session_api,'_base_url',None)!=(endpoint or '').rstrip('/'):
        raise GeometryError('simulation SDK origin differs from the configured loopback gateway')


def preflight(client: Any, *, robot_id: str, arm: str,
              profile: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """GET-only report. Never creates a session or submits a motion."""
    report: dict[str, Any] = {"robot_id": robot_id, "arm": arm, "read_only": True, "blockers": []}
    report["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    try:
        robots = client.list_robots()
        report["registered"] = any(robot.get("robot_id", robot.get("id")) == robot_id for robot in robots)
        if not report["registered"]:
            report["blockers"].append("robot_not_registered")
        report["queue"] = client.get_queue(robot_id)
        station = client.get_station_observation(robot_id)
        report["station"] = {key: station.get(key) for key in (
            "source", "mode", "homed", "settled", "safety", "observed_at")}
        if not station.get("safety", {}).get("ok") or station.get("safety", {}).get("estop_engaged"):
            report["blockers"].append("station_safety_not_ready")
        if not station.get("homed") or not station.get("settled"):
            report["blockers"].append("station_not_homed_and_settled")
        try:
            age = time.time() - float(station.get("observed_at"))
            report["station_age_s"] = age
            if not np.isfinite(age) or not -1 <= age <= 2:
                report["blockers"].append("stale_station_feedback")
        except (TypeError, ValueError):
            report["blockers"].append("station_timestamp_unavailable")
        if station.get("source") != "hardware":
            report["blockers"].append("hardware_feedback_required")
        calibration = client.get_calibration(robot_id)
        report["calibration_id"] = calibration.get("calibration_id")
        report["calibration_quality"] = calibration.get("quality")
        if profile is None:
            report["blockers"].append("measured_deployment_profile_missing")
        else:
            validate_profile(profile, calibration, robot_id=robot_id, arm=arm)
        report["note"] = "Read-only preflight does not establish collision-free motion or physical success."
    except Exception as error:
        report["blockers"].append(f"{type(error).__name__}: {error}")
    report["ready_for_session"] = not report["blockers"]
    return report
