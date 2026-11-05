"""Transport layer: normalized observations, command dispatch, and receipts.

``PlatformTransport`` normalizes whatever the remote platform provides into
``PlatformObservation`` (object metadata + camera image paths + receipt).  Two
implementations are provided:

- ``MockPlatformTransport``: a deterministic offline transport used for
  protocol qualification and dry runs without any network access.
- ``RemotePlatformTransport``: the BluPe Session API transport.  It downloads
  the station's camera JPEGs for perception, translates ``PlatformCommand``
  pick/place into joint trajectories through the deterministic YAM IK motion
  planner, submits them as 10 Hz waypoint packets, and follows the episode
  trace until each packet completes.  Perception coordinates are expressed in
  the top camera's optical frame and mapped into the selected arm's base
  frame with the station's published calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from datetime import datetime
import time
import numpy as np

from .blupe_client import (
    BluPeAPIError,
    BluPeSessionClient,
    SessionHandle,
    TrajectoryBuilder,
)
from .commands import PlatformCommand
from .config import PlatformConfig
from .kinematics import NUM_JOINTS, YamKinematics
from .motion import MotionPlan, MotionPlanError, YamMotionPlanner
from .geometry import finite_vector


class PlatformUnavailableError(RuntimeError):
    """Raised when the remote platform transport cannot proceed safely."""


@dataclass(frozen=True)
class ObjectMetadata:
    name: str
    center_xyz_m: tuple[float, float, float]
    category: str | None = None
    visual_aspect: float | None = None


@dataclass(frozen=True)
class PlatformObservation:
    observation_id: str
    objects: tuple[ObjectMetadata, ...]
    image_paths: tuple[str, ...] = ()
    receipt: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "objects": [
                {
                    "name": item.name,
                    "center_xyz_m": list(item.center_xyz_m),
                    "category": item.category,
                    "visual_aspect": item.visual_aspect,
                }
                for item in self.objects
            ],
            "image_paths": list(self.image_paths),
            "receipt": dict(self.receipt),
        }


class PlatformTransport(Protocol):
    def health_check(self) -> Mapping[str, Any]: ...

    def get_observation(self) -> PlatformObservation: ...

    def send_command(self, command: PlatformCommand) -> Mapping[str, Any]: ...


def _as_matrix(nested: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(value) for value in row] for row in nested]


class RemotePlatformTransport:
    """BluPe Session API transport for one remote YAM station.

    Lifecycle: construct (or ``connect``) -> ``health_check`` -> ``start_session``
    (joins the queue and waits for control) -> ``get_observation`` /
    ``send_command`` per the session loop -> ``stop_session``.  Every command
    returns a receipt with the planned keyframes, submitted packets, and the
    platform's trajectory results, preserving the audit discipline used
    elsewhere in this repository.
    """

    def __init__(
        self,
        config: PlatformConfig,
        *,
        client: BluPeSessionClient | None = None,
        motion_planner: YamMotionPlanner | None = None,
        destinations: Mapping[str, Sequence[float]] | None = None,
        image_dir: Path | str = "output/blupe/images",
        arm: str = "right",
        prompt: str = "Agentic RAG-VLM remote pick-and-place session",
        run_duration_s: int = 300,
        session_model: str | None = None,
        require_settled_first_command: bool = True,
        deployment_profile: Mapping[str, Any] | None = None,
    ) -> None:
        if config.platform != "blupe":
            raise ValueError(
                f"RemotePlatformTransport targets 'blupe', got platform {config.platform!r}"
            )
        if arm not in {"left", "right"}:
            raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
        self.config = config
        self.arm = arm
        self.prompt = prompt
        self.run_duration_s = int(run_duration_s)
        self.session_model = session_model
        self.require_settled_first_command = require_settled_first_command
        self.deployment_profile = dict(deployment_profile or {})
        self.image_dir = Path(image_dir)
        self.destinations: dict[str, tuple[float, float, float]] = {
            str(name): (float(xyz[0]), float(xyz[1]), float(xyz[2]))
            for name, xyz in (destinations or {}).items()
        }
        self.client = client or BluPeSessionClient(
            config.endpoint_base or "",
            request_timeout_s=config.request_timeout_s,
            max_retries=config.max_retries,
            auth_token_env=config.auth_token_env,
        )
        self.motion_planner = motion_planner or YamMotionPlanner(YamKinematics())
        self.handle: SessionHandle | None = None
        self._observation_counter = 0
        self._step_id = 0
        self._waypoints_submitted = 0
        self._trajectory_counter = 0
        self._calibration: dict[str, Any] | None = None
        self._last_station: dict[str, Any] = {}
        self._first_command_sent = False
        self._last_place: dict[str, Any] | None = None
        self._gripper_hold_command: float | None = None

    # ------------------------------------------------------------ helpers

    def _require_handle(self) -> SessionHandle:
        if self.handle is None:
            raise PlatformUnavailableError(
                "no active session; call start_session() before commanding the robot"
            )
        return self.handle

    def _station(self) -> dict[str, Any]:
        robot_id = self.config.robot_id
        if robot_id is None:
            raise PlatformUnavailableError("platform config is missing robot_id")
        try:
            self._last_station = self.client.get_station_observation(robot_id)
        except BluPeAPIError as error:
            raise PlatformUnavailableError(f"station observation failed: {error}") from error
        return self._last_station

    def calibration(self) -> dict[str, Any]:
        if self._calibration is None:
            robot_id = self.config.robot_id
            if robot_id is None:
                raise PlatformUnavailableError("platform config is missing robot_id")
            self._calibration = self.client.get_calibration(robot_id)
        return self._calibration

    def _camera_to_base(self, position_m: Sequence[float]) -> tuple[float, float, float]:
        """Map a top-camera optical frame point into the operating arm's base frame."""
        if self.deployment_profile.get("coordinate_frame") == "arm_base":
            return tuple(float(v) for v in finite_vector(position_m, 3, "base position"))
        calibration = self.calibration()
        top = calibration.get("cameras", {}).get("top", {})
        transforms = top.get("T_base_camera", {})
        matrix = transforms.get(self.arm) or transforms.get(
            "right" if self.arm == "right" else "left"
        )
        if matrix is None:
            raise PlatformUnavailableError(
                f"calibration has no top-camera T_base_camera for arm '{self.arm}'"
            )
        rows = _as_matrix(matrix)
        x, y, z = float(position_m[0]), float(position_m[1]), float(position_m[2])
        return (
            rows[0][0] * x + rows[0][1] * y + rows[0][2] * z + rows[0][3],
            rows[1][0] * x + rows[1][1] * y + rows[1][2] * z + rows[1][3],
            rows[2][0] * x + rows[2][1] * y + rows[2][2] * z + rows[2][3],
        )

    def _current_arm_state(self) -> tuple[tuple[float, ...], float | None, tuple[float, ...], float | None]:
        station = self._last_station or self._station()
        joints_key = f"{self.arm}_joints_deg"
        other_key = "left_joints_deg" if self.arm == "right" else "right_joints_deg"
        joints = tuple(float(value) for value in station.get(joints_key, ()))
        other = tuple(float(value) for value in station.get(other_key, ()))
        if len(joints) != NUM_JOINTS:
            raise PlatformUnavailableError(
                f"station observation has no {NUM_JOINTS}-joint state for the {self.arm} arm"
            )
        gripper = station.get(f"{self.arm}_gripper")
        other_gripper = station.get(("left" if self.arm == "right" else "right") + "_gripper")
        if self.deployment_profile:
            if len(other) != NUM_JOINTS or not np.isfinite(joints).all() or not np.isfinite(other).all():
                raise PlatformUnavailableError("finite measured state for both arms required")
            if any(value is None or not np.isfinite(value) or not 0 <= value <= 1 for value in (gripper, other_gripper)):
                raise PlatformUnavailableError("measured normalized gripper states required for both arms")
        return (
            joints,
            float(gripper) if gripper is not None else None,
            other if len(other) == NUM_JOINTS else (),
            float(other_gripper) if other_gripper is not None else None,
        )

    def _safety_gate(self) -> dict[str, Any]:
        station = self._station()
        if self.deployment_profile:
            self._check_station_freshness(station)
            if not station.get("homed") or not station.get("settled"):
                raise PlatformUnavailableError("fresh homed and settled state required before each command")
        safety = station.get("safety", {})
        if safety.get("estop_engaged"):
            raise PlatformUnavailableError("station e-stop is engaged; refusing to command")
        if not safety.get("ok", False):
            raise PlatformUnavailableError(
                f"station safety not ok: {safety.get('reason', 'unknown')}"
            )
        if self.require_settled_first_command and not self._first_command_sent:
            if not station.get("homed", False) or not station.get("settled", False):
                raise PlatformUnavailableError(
                    "station is not homed/settled; refusing the first motion command"
                )
        return station

    # ------------------------------------------------------- transport API

    def health_check(self) -> Mapping[str, Any]:
        robot_id = self.config.robot_id
        if robot_id is None:
            raise PlatformUnavailableError("platform config is missing robot_id")
        robots = self.client.list_robots()
        known = [item for item in robots if item.get("robot_id") == robot_id]
        queue = self.client.get_queue(robot_id)
        stations = queue.get("stations", [])
        station = stations[0] if stations else {}
        calibration = None
        calibrations = [item for item in robots if item.get("calibration_url")]
        healthy = {
            "platform": "blupe",
            "robot_registered": bool(known),
            "station_connected": bool(station.get("connected", False)),
            "station_available": bool(station.get("available", False)),
            "queue_entries": len(queue.get("entries", [])),
            "calibration_advertised": bool(calibrations),
            "endpoint": self.config.endpoint_base,
            "robot_id": robot_id,
        }
        if calibrations:
            try:
                calibration = self.calibration()
                healthy["calibration_id"] = calibration.get("calibration_id")
                healthy["top_camera_matrix_present"] = bool(
                    calibration.get("cameras", {}).get("top", {}).get("K")
                )
            except BluPeAPIError as error:
                healthy["calibration_error"] = str(error)
        return healthy

    def start_session(
        self, *, timeout_s: float = 600.0, wait: bool = True
    ) -> Mapping[str, Any]:
        """Create a session and (optionally) wait until it gains control."""
        if self.handle is not None:
            return {"already_started": True, **self.handle.to_public_dict()}
        robot_id = self.config.robot_id
        if robot_id is None:
            raise PlatformUnavailableError("platform config is missing robot_id")
        self.handle = self.client.create_session(
            robot_id,
            self.prompt,
            run_duration_s=self.run_duration_s,
            model=self.session_model,
        )
        self._station()
        if wait:
            self.client.wait_for_running(self.handle, timeout_s=timeout_s)
            self._station()
        return dict(self.handle.to_public_dict())

    def stop_session(self, reason: str = "policy_complete") -> Mapping[str, Any]:
        if self.handle is None:
            return {"stopped": False, "reason": "no_active_session"}
        response = self.client.stop_session(self.handle, reason)
        return {"stopped": True, "response": dict(response), **self.handle.to_public_dict()}

    def get_observation(self) -> PlatformObservation:
        station = self._station()
        self._observation_counter += 1
        observation_id = f"blupe_obs_{self._observation_counter:04d}"
        image_paths: dict[str, str] = {}
        try:
            if self.deployment_profile.get('top_rgb_source')=='paired_rgbd':
                from .camera_contract import fetch_paired_top
                image_paths=fetch_paired_top(self.client,self.config.robot_id,self.calibration(),self.image_dir/observation_id)
            else:
                image_paths = self.client.download_observation_images(station, self.image_dir / observation_id)
        except BluPeAPIError as error:
            raise PlatformUnavailableError(f"camera image download failed: {error}") from error
        receipt = {
            "transport": "blupe",
            "robot_id": self.config.robot_id,
            "observation_id": observation_id,
            "station": {
                "mode": station.get("mode"),
                "homed": station.get("homed"),
                "settled": station.get("settled"),
                "queue_ready": station.get("queue_ready"),
                "safety": dict(station.get("safety", {})),
                "observed_at": station.get("observed_at"),
            },
            "joints_deg": {
                "left": list(station.get("left_joints_deg", ())),
                "right": list(station.get("right_joints_deg", ())),
            },
            "session": self.handle.to_public_dict() if self.handle else None,
            "top_rgb_source":self.deployment_profile.get('top_rgb_source','jpeg'),
        }
        if self.deployment_profile:
            self._check_station_freshness(station)
            if not station.get("settled"):
                raise PlatformUnavailableError("geometric perception requires settled feedback")
            if "top" not in image_paths:
                raise PlatformUnavailableError("top camera missing")
            headers = getattr(self.client, "image_metadata", {}).get(image_paths["top"], {})
            headers = {key.lower(): value for key, value in headers.items()}
            captured = headers.get("x-captured-at")
            self._check_age(captured, "top capture")
            receipt.update({"camera_role": "top", "captured_at": captured,
                            "calibration_id": self.calibration()["calibration_id"]})
            if self._last_place is not None:
                joints, gripper, _, _ = self._current_arm_state()
                tcp = self.motion_planner.kinematics.fk(np.radians(joints))[:3, 3]
                distance = float(np.linalg.norm(tcp - self._last_place["position"]))
                receipt["release_state"] = {
                    "gripper_open": gripper is not None and gripper >= self._last_place["open_command"] - 0.05,
                    "retracted": distance >= 0.05,
                    "tcp_destination_distance_m": distance,
                    "measured_gripper": gripper,
                }
            image_paths = {"top": image_paths["top"]}
        return PlatformObservation(
            observation_id=observation_id,
            objects=(),
            image_paths=tuple(sorted(image_paths.values())),
            receipt=receipt,
        )

    def _check_age(self, value: Any, label: str) -> None:
        try:
            stamp = float(value)
        except (ValueError, TypeError):
            try:
                stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
            except (ValueError, TypeError) as error:
                raise PlatformUnavailableError(f"missing/invalid {label} timestamp") from error
        age = time.time() - stamp
        if not np.isfinite(age) or age < -1 or age > float(self.deployment_profile.get("max_observation_age_s", 2)):
            raise PlatformUnavailableError(f"stale or future {label}")

    def _check_station_freshness(self, station: Mapping[str, Any]) -> None:
        self._check_age(station.get("observed_at"), "station")
        environment=self.deployment_profile.get('execution_environment','hardware')
        if environment=='simulation':
            from .deployment import validate_execution_binding
            validate_execution_binding(self.deployment_profile,self.config.endpoint_base,
                session_api=getattr(self.client,'api',None))
            if station.get('source')!='simulation':raise PlatformUnavailableError('explicit simulation feedback required')
        elif station.get("source") != "hardware":
            raise PlatformUnavailableError("measured hardware feedback required")

    # -------------------------------------------------------- command path

    def _plan(self, command: PlatformCommand) -> MotionPlan:
        joints, gripper, other_joints, other_gripper = self._current_arm_state()
        parameters = dict(command.parameters)
        approach = str(parameters.get("approach", "top"))
        if any(float(parameters.get(key,0)) != 0 for key in ("approach_yaw_delta_rad","wrist_yaw_delta_rad")):
            raise MotionPlanError("yaw recovery is not calibrated for the hardware profile")
        if self.deployment_profile and approach != self.deployment_profile["approach"]:
            raise MotionPlanError("approach has not been validated for this station profile")
        aperture = float(parameters.get("aperture_m", 0.07))
        if command.command_type == "pick":
            position_xyz = parameters.get("position_xyz_m")
            if position_xyz is None:
                raise PlatformUnavailableError(
                    f"pick command {command.command_id} has no position_xyz_m"
                )
            position = self._camera_to_base(position_xyz)
            if self.deployment_profile:
                position = np.asarray(position) + np.asarray(self.deployment_profile["grasp_offsets_base_m"][command.target])
            return self.motion_planner.plan_pick(
                arm=self.arm,
                joints_deg_current=joints,
                gripper_current=gripper,
                target_xyz_m=position,
                aperture_m=aperture,
                approach=approach,
                approach_height_delta_m=float(parameters.get("approach_height_delta_m", 0)),
                approach_offset_xy_m=parameters.get("approach_offset_xy_m", (0, 0)),
                velocity_scale=float(parameters.get("velocity_scale", 1)),
                other_joints_deg=other_joints,
                other_gripper=other_gripper,
            )
        if command.command_type == "place":
            destination_xyz = parameters.get("destination_xyz_m")
            destination_name = str(parameters.get("destination", ""))
            if destination_xyz is None:
                if destination_name not in self.destinations:
                    raise PlatformUnavailableError(
                        f"place destination '{destination_name}' has no coordinate table entry"
                    )
                destination_xyz = self.destinations[destination_name]
            position = self._camera_to_base(destination_xyz)
            if self.deployment_profile:
                position = np.asarray(position) + np.asarray(self.deployment_profile["grasp_offsets_base_m"][command.target])
            return self.motion_planner.plan_place(
                arm=self.arm,
                joints_deg_current=joints,
                gripper_current=gripper,
                destination_xyz_m=position,
                aperture_m=aperture,
                approach=approach,
                velocity_scale=float(parameters.get("velocity_scale", 1)),
                other_joints_deg=other_joints,
                other_gripper=other_gripper,
                hold_gripper_command=self._gripper_hold_command,
            )
        if command.command_type == "home":
            return self.motion_planner.plan_hold(
                arm=self.arm,
                joints_deg_current=joints,
                gripper_current=gripper,
                other_joints_deg=other_joints,
                other_gripper=other_gripper,
            )
        raise PlatformUnavailableError(f"unsupported command type: {command.command_type}")

    def _execute_plan(self, command: PlatformCommand, plan: MotionPlan) -> dict[str, Any]:
        handle = self._require_handle()
        steps = plan.interpolated_steps(
            max_joint_delta_deg_per_step=self.motion_planner.max_joint_delta_deg_per_step
        )
        builder = TrajectoryBuilder(plan.arm, cadence_hz=plan.cadence_hz)
        for index, (joints, gripper) in enumerate(steps):
            builder.add(
                step_id=self._step_id + index,
                joints_deg=joints,
                gripper=gripper,
                other_joints_deg=plan.other_joints_deg,
                other_gripper=plan.other_gripper,
            )
        self._trajectory_counter += 1
        prefix = f"{command.command_id}_t{self._trajectory_counter:03d}"
        packets = builder.packets(
            episode_id=handle.episode_id or "", lease_id=handle.lease_id or "", prefix=prefix
        )
        packet_results: list[dict[str, Any]] = []
        try:
            for packet in packets:
                self._safety_gate()
                ack = self.client.submit_trajectory_packet(handle, packet)
                result = self.client.wait_for_trajectory(
                    handle,
                    str(packet["trajectory_id"]),
                    timeout_s=max(60.0, len(packet["waypoints"]) / plan.cadence_hz * 3.0 + 30.0),
                )
                packet_results.append(
                    {
                        "trajectory_id": packet["trajectory_id"],
                        "waypoints": len(packet["waypoints"]),
                        "ack": dict(ack),
                        "result": dict(result),
                    }
                )
                if result.get("status") != "completed":
                    break
        except BluPeAPIError as error:
            raise PlatformUnavailableError(f"trajectory execution failed: {error}") from error
        finally:
            self._step_id += len(steps)
            self._waypoints_submitted += len(steps)
            self._first_command_sent = True
            self._station()
        statuses = {str(item["result"].get("status")) for item in packet_results}
        accepted = bool(packet_results) and statuses == {"completed"}
        return {
            "transport": "blupe",
            "command_id": command.command_id,
            "command_type": command.command_type,
            "accepted": accepted,
            "arm": plan.arm,
            "keyframes": [
                {"name": frame.name, "joints_deg": list(frame.joints_deg), "gripper": frame.gripper}
                for frame in plan.keyframes
            ],
            "planner_notes": list(plan.notes),
            "waypoints": len(steps),
            "step_id_range": [builder.waypoints[0].step_id, builder.waypoints[-1].step_id],
            "packets": packet_results,
            "session_waypoints_used": self._waypoints_submitted,
        }

    def send_command(self, command: PlatformCommand) -> Mapping[str, Any]:
        self._safety_gate()
        try:
            plan = self._plan(command)
            if self.deployment_profile:
                self._validate_plan_geometry(plan)
        except MotionPlanError as error:
            return {
                "transport": "blupe",
                "command_id": command.command_id,
                "command_type": command.command_type,
                "accepted": False,
                "reason": "motion_plan_failed",
                "detail": str(error),
                "execution_started": False,
            }
        receipt = self._execute_plan(command, plan)
        if command.command_type=='pick':
            self._gripper_hold_command=plan.keyframes[-1].gripper if receipt.get('accepted') is True else None
            gate=self.deployment_profile.get('retention_observation')
            if receipt.get('accepted') is True and gate:
                joints,measured,_,_=self._current_arm_state()
                tcp=self.motion_planner.kinematics.fk(np.radians(joints))[:3,3]
                source=np.asarray(self._camera_to_base(command.parameters['position_xyz_m']))
                retained=(measured is not None and gate['minimum_blocked_aperture']<=measured<=gate['maximum_blocked_aperture']
                          and float(tcp[2]-source[2])>=gate['minimum_tcp_lift_m'])
                receipt['retention_observation']={'passed':bool(retained),'measured_gripper':measured,
                    'commanded_gripper':self._gripper_hold_command,'tcp_height_change_m':float(tcp[2]-source[2]),
                    'source':'measured_joint_and_gripper_proxy;not_calibrated_object_force'}
                if not retained:
                    receipt.update(accepted=False,execution_started=True,reason='retention_unconfirmed')
                    self._gripper_hold_command=None
        receipt["unsupported_control_fields"] = [
            key for key in ("force_n", "synergy") if key in command.parameters
        ]
        if command.command_type == "place" and receipt.get("accepted") is True:
            self._gripper_hold_command=None
            position = command.parameters.get("destination_xyz_m")
            if position is None:
                position = self.destinations[command.parameters["destination"]]
            self._last_place = {
                "position": np.asarray(self._camera_to_base(position)),
                "open_command": self.motion_planner.kinematics.gripper_command_for_aperture(float(command.parameters.get("aperture_m", 0.07))),
            }
        return receipt

    def _validate_plan_geometry(self, plan: MotionPlan) -> None:
        bounds = np.asarray(self.deployment_profile["workspace_bounds_m"], dtype=float)
        plane = self.deployment_profile["support_plane"]
        table_z = -float(plane["offset_m"])
        for joints, _ in plan.interpolated_steps(
            max_joint_delta_deg_per_step=self.motion_planner.max_joint_delta_deg_per_step
        ):
            q = np.radians(joints)
            if not np.isfinite(q).all() or not self.motion_planner.kinematics.in_limits(q):
                raise MotionPlanError("joint trajectory outside calibrated limits")
            tcp = self.motion_planner.kinematics.fk(q)[:3, 3]
            if np.any(tcp < bounds[:, 0]) or np.any(tcp > bounds[:, 1]):
                raise MotionPlanError("TCP trajectory leaves measured workspace")
            if tcp[2] < table_z + 0.005:
                raise MotionPlanError("TCP trajectory violates table clearance")
        plan.notes.append("validated:sampled_tcp_workspace_and_table_clearance;full_arm_collision_not_certified")


class MockPlatformTransport:
    """Deterministic offline transport for dry runs and regression tests.

    The mock keeps a table-top scene as named 3-D positions.  ``pick`` grips
    the target without moving it, ``place`` moves it to the resolved
    destination, and ``home`` is a no-op, so a successful pick-and-place is
    observable as target displacement while a rejected place leaves the
    target in place (matching the scene-change verifier's criterion).
    Every command returns a receipt so the session audit trail is exercised
    end-to-end without network access.
    """

    def __init__(
        self,
        scene: Mapping[str, Sequence[float]],
        *,
        destinations: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        self._positions: dict[str, tuple[float, float, float]] = {
            str(name): (float(xyz[0]), float(xyz[1]), float(xyz[2]))
            for name, xyz in scene.items()
        }
        self._destinations: dict[str, tuple[float, float, float]] = {
            str(name): (float(xyz[0]), float(xyz[1]), float(xyz[2]))
            for name, xyz in (destinations or {}).items()
        }
        self._observation_counter = 0
        self.command_log: list[dict[str, Any]] = []

    def health_check(self) -> Mapping[str, Any]:
        return {"ok": True, "transport": "mock"}

    def get_observation(self) -> PlatformObservation:
        self._observation_counter += 1
        objects = tuple(
            ObjectMetadata(name=name, center_xyz_m=xyz)
            for name, xyz in sorted(self._positions.items())
        )
        return PlatformObservation(
            observation_id=f"mock_obs_{self._observation_counter:04d}",
            objects=objects,
            receipt={"transport": "mock", "observation_id": f"mock_obs_{self._observation_counter:04d}"},
        )

    def send_command(self, command: PlatformCommand) -> Mapping[str, Any]:
        receipt: dict[str, Any] = {
            "transport": "mock",
            "command_id": command.command_id,
            "command_type": command.command_type,
            "accepted": True,
        }
        if command.command_type == "pick":
            if command.target not in self._positions:
                receipt["accepted"] = False
                receipt["reason"] = "target_not_in_scene"
            else:
                receipt["detail"] = "gripped_without_displacement"
        elif command.command_type == "place":
            if command.target not in self._positions:
                receipt["accepted"] = False
                receipt["reason"] = "target_not_in_scene"
            else:
                resolved = command.parameters.get("destination_xyz_m")
                destination_name = str(command.parameters.get("destination", ""))
                if resolved is None and destination_name in self._destinations:
                    resolved = list(self._destinations[destination_name])
                if resolved is None:
                    receipt["accepted"] = False
                    receipt["reason"] = "unresolved_destination"
                else:
                    self._positions[command.target] = (
                        float(resolved[0]),
                        float(resolved[1]),
                        float(resolved[2]),
                    )
                    receipt["target_placed_at_m"] = list(self._positions[command.target])
        elif command.command_type == "home":
            receipt["detail"] = "no-op"
        else:
            receipt["accepted"] = False
            receipt["reason"] = f"unknown_command_type:{command.command_type}"
        self.command_log.append({**command.to_dict(), "receipt": receipt})
        return receipt
