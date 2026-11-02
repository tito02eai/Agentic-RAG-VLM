"""HTTP client for the BluPe Session API v1 (standard library only).

Implements the public contract documented at ``https://blupe.io/docs/robot-api/``
(checked 2026-10-25): read-only station endpoints (robots, queue, observation,
calibration, camera poses, camera JPEGs), session lifecycle (create, snapshot,
stop), paginated session observations, the episode trace, and joint-trajectory
submission (10 Hz, 1-300 waypoints per packet, one active packet at a time).

Authentication: public monitoring and session creation need no BluPe account
credential.  ``POST /v1/sessions`` returns an opaque ``session_capability``
that authorizes that session's requests as a Bearer token.  The capability is
kept in memory only, never written to disk, URLs, or logs.  An optional
pre-shared token environment variable (``PlatformConfig.auth_token_env``) is
attached to public requests when present, for deployments that introduce one.

Waiting for execution: HTTP 202 ``accepted`` means dispatched, not executed.
This client follows the episode trace (and session snapshots) by polling,
which needs no WebSocket dependency; the event stream remains available for
future integrations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

DEFAULT_CADENCE_HZ = 10.0
MAX_WAYPOINTS_PER_PACKET = 300
MAX_SESSION_WAYPOINTS = 3000
WAYPOINT_STEP_ID_MAX = 12299


class BluPeAPIError(RuntimeError):
    """Raised when the BluPe Session API returns an error response."""


class BluPeConnectionError(BluPeAPIError):
    """Raised when the BluPe Session API cannot be reached."""


@dataclass
class SessionHandle:
    session_id: str
    capability: str
    robot_id: str
    status: str
    events_url: str
    episode_id: str | None = None
    lease_id: str | None = None
    position: int | None = None
    created_response: Mapping[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        """Audit-safe view: the capability never leaves the process."""
        return {
            "session_id": self.session_id,
            "robot_id": self.robot_id,
            "status": self.status,
            "episode_id": self.episode_id,
            "lease_id": self.lease_id,
            "position": self.position,
            "events_url": self.events_url,
        }


@dataclass(frozen=True)
class Waypoint:
    step_id: int
    left_joints_deg: tuple[float, ...]
    right_joints_deg: tuple[float, ...]
    left_gripper: float | None = None
    right_gripper: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "step_id": int(self.step_id),
            "left_joints_deg": [float(value) for value in self.left_joints_deg],
            "right_joints_deg": [float(value) for value in self.right_joints_deg],
        }
        if self.left_gripper is not None:
            payload["left_gripper"] = float(self.left_gripper)
        if self.right_gripper is not None:
            payload["right_gripper"] = float(self.right_gripper)
        return payload


class TrajectoryBuilder:
    """Accumulates waypoints and emits API-shaped packets (<=300 each)."""

    def __init__(self, arm: str, cadence_hz: float = DEFAULT_CADENCE_HZ) -> None:
        if arm not in {"left", "right"}:
            raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
        if cadence_hz != DEFAULT_CADENCE_HZ:
            raise ValueError(
                f"the live v1 contract locks cadence_hz to {DEFAULT_CADENCE_HZ}"
            )
        self.arm = arm
        self.cadence_hz = cadence_hz
        self.waypoints: list[Waypoint] = []

    @property
    def waypoint_count(self) -> int:
        return len(self.waypoints)

    def add(
        self,
        *,
        step_id: int,
        joints_deg: Sequence[float],
        gripper: float | None,
        other_joints_deg: Sequence[float],
        other_gripper: float | None,
    ) -> Waypoint:
        if step_id > WAYPOINT_STEP_ID_MAX:
            raise BluPeAPIError(f"step_id {step_id} exceeds the wire maximum")
        if len(self.waypoints) + 1 > MAX_SESSION_WAYPOINTS:
            raise BluPeAPIError(
                f"session waypoint budget of {MAX_SESSION_WAYPOINTS} exceeded"
            )
        if self.arm == "left":
            waypoint = Waypoint(step_id, tuple(joints_deg), tuple(other_joints_deg), gripper, other_gripper)
        else:
            waypoint = Waypoint(step_id, tuple(other_joints_deg), tuple(joints_deg), other_gripper, gripper)
        self.waypoints.append(waypoint)
        return waypoint

    def packets(self, *, episode_id: str, lease_id: str, prefix: str) -> list[dict[str, Any]]:
        """Split the accumulated waypoints into wire packets of <=300."""
        if not self.waypoints:
            raise BluPeAPIError("cannot build packets from an empty trajectory")
        packets: list[dict[str, Any]] = []
        for index in range(0, len(self.waypoints), MAX_WAYPOINTS_PER_PACKET):
            chunk = self.waypoints[index : index + MAX_WAYPOINTS_PER_PACKET]
            packets.append(
                {
                    "schema_version": 1,
                    "episode_id": episode_id,
                    "lease_id": lease_id,
                    "trajectory_id": f"{prefix}_{index // MAX_WAYPOINTS_PER_PACKET:03d}",
                    "cadence_hz": self.cadence_hz,
                    "waypoints": [waypoint.to_dict() for waypoint in chunk],
                }
            )
        return packets


class BluPeSessionClient:
    """Thin, dependency-free client for the BluPe Session API."""

    def __init__(
        self,
        endpoint_base: str,
        *,
        request_timeout_s: float = 30.0,
        max_retries: int = 2,
        auth_token_env: str | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.endpoint_base = endpoint_base.rstrip("/")
        self.request_timeout_s = float(request_timeout_s)
        self.max_retries = int(max_retries)
        self.auth_token_env = auth_token_env
        self._sleep = sleeper
        self._clock = clock

    # ------------------------------------------------------------ transport

    def _authorization(self, capability: str | None) -> str | None:
        if capability is not None:
            return f"Bearer {capability}"
        if self.auth_token_env and self.auth_token_env in os.environ:
            return f"Bearer {os.environ[self.auth_token_env]}"
        return None

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        capability: str | None = None,
        is_json: bool = True,
    ) -> Any:
        url = path if path.startswith("http") else f"{self.endpoint_base}{path}"
        headers = {"Accept": "application/json"}
        authorization = self._authorization(capability)
        if authorization is not None:
            headers["Authorization"] = authorization
        body = None
        if payload is not None:
            body = json.dumps(dict(payload)).encode("utf-8")
            headers["Content-Type"] = "application/json"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                request = urllib.request.Request(url, data=body, headers=headers, method=method)
                with urllib.request.urlopen(request, timeout=self.request_timeout_s) as response:
                    raw = response.read()
                    if not is_json:
                        return raw
                    if not raw:
                        return {}
                    return json.loads(raw.decode("utf-8"))
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", "replace")[:500]
                # Client errors are contract failures; do not retry blindly.
                if 400 <= error.code < 500 or attempt >= self.max_retries:
                    raise BluPeAPIError(
                        f"{method} {url} failed: HTTP {error.code}: {detail}"
                    ) from error
                last_error = error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                if attempt >= self.max_retries:
                    raise BluPeConnectionError(
                        f"{method} {url} unreachable after {self.max_retries + 1} attempts: {error}"
                    ) from error
                last_error = error
            self._sleep(0.5 * (attempt + 1))
        raise BluPeConnectionError(f"{method} {url} failed: {last_error}")

    # ---------------------------------------------------------- read-only

    def list_robots(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/v1/robots")["robots"])

    def get_queue(self, robot_id: str) -> dict[str, Any]:
        return dict(self._request("GET", f"/v1/robots/{robot_id}/queue"))

    def get_station_observation(self, robot_id: str) -> dict[str, Any]:
        return dict(self._request("GET", f"/v1/robots/{robot_id}/observation"))

    def get_calibration(self, robot_id: str) -> dict[str, Any]:
        return dict(self._request("GET", f"/v1/robots/{robot_id}/calibration"))

    def get_camera_poses(self, robot_id: str) -> dict[str, Any]:
        return dict(self._request("GET", f"/v1/robots/{robot_id}/camera-poses"))

    def fetch_image(self, image_url: str, destination: Path) -> Path:
        """Download a camera JPEG; relative URLs resolve against the API origin."""
        url = (
            image_url
            if image_url.startswith("http")
            else f"{self.endpoint_base}{image_url}"
        )
        raw = self._request("GET", url, is_json=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        return destination

    def download_observation_images(
        self, station_observation: Mapping[str, Any], directory: Path
    ) -> dict[str, str]:
        """Persist the observation's camera JPEGs; returns role -> local path."""
        paths: dict[str, str] = {}
        images = station_observation.get("images", {})
        for role in ("top", "left", "right"):
            entry = images.get(role)
            if not entry or not entry.get("url"):
                continue
            destination = directory / f"{role}.jpg"
            self.fetch_image(entry["url"], destination)
            paths[role] = str(destination)
        return paths

    # ------------------------------------------------------------- session

    def create_session(
        self,
        robot_id: str,
        prompt: str,
        *,
        run_duration_s: int = 300,
        cadence_hz: float = DEFAULT_CADENCE_HZ,
        model: str | None = None,
    ) -> SessionHandle:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "robot_id": robot_id,
            "prompt": prompt,
            "run_duration_s": int(run_duration_s),
        }
        if model is not None:
            payload["model"] = str(model)
        response = self._request("POST", "/v1/sessions", payload=payload)
        for key in ("session_id", "session_capability", "status", "events_url"):
            if key not in response:
                raise BluPeAPIError(f"session creation response missing '{key}': {sorted(response)}")
        return SessionHandle(
            session_id=str(response["session_id"]),
            capability=str(response["session_capability"]),
            robot_id=str(response.get("robot_id", robot_id)),
            status=str(response["status"]),
            events_url=str(response["events_url"]),
            episode_id=response.get("episode_id"),
            lease_id=response.get("lease_id"),
            position=response.get("position"),
            created_response=response,
        )

    def get_session(self, handle: SessionHandle) -> dict[str, Any]:
        return dict(
            self._request(
                "GET", f"/v1/sessions/{handle.session_id}", capability=handle.capability
            )
        )

    def refresh_handle(self, handle: SessionHandle) -> SessionHandle:
        """Update episode/lease/status fields from a fresh session snapshot."""
        snapshot = self.get_session(handle)
        handle.status = str(snapshot.get("status", handle.status))
        handle.episode_id = snapshot.get("episode_id", handle.episode_id)
        handle.lease_id = snapshot.get("lease_id", handle.lease_id)
        handle.position = snapshot.get("position", handle.position)
        return handle

    def wait_for_running(
        self,
        handle: SessionHandle,
        *,
        timeout_s: float,
        poll_interval_s: float = 2.0,
    ) -> dict[str, Any]:
        """Poll the session snapshot until running (or a terminal state)."""
        deadline = self._clock() + timeout_s
        last: dict[str, Any] = {}
        while True:
            last = self.get_session(handle)
            self.refresh_handle(handle)
            status = str(last.get("status", ""))
            if status == "running" and handle.episode_id and handle.lease_id:
                return last
            if status in {"complete", "resetting", "stopped", "timed_out", "disconnected", "safety_aborted"}:
                raise BluPeAPIError(
                    f"session {handle.session_id} reached terminal state '{status}' before running"
                )
            if self._clock() >= deadline:
                raise BluPeAPIError(
                    f"session {handle.session_id} still '{status}' after {timeout_s:.0f}s"
                )
            self._sleep(poll_interval_s)

    def submit_trajectory_packet(
        self, handle: SessionHandle, packet: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not handle.episode_id or not handle.lease_id:
            self.refresh_handle(handle)
        if not handle.episode_id or not handle.lease_id:
            raise BluPeAPIError(
                f"session {handle.session_id} has no episode/lease; cannot submit"
            )
        # Keep the freshest lease on the wire without mutating the caller's dict.
        wire_packet = dict(packet)
        wire_packet["episode_id"] = handle.episode_id
        wire_packet["lease_id"] = handle.lease_id
        response = self._request(
            "POST",
            f"/v1/sessions/{handle.session_id}/trajectories",
            payload=wire_packet,
            capability=handle.capability,
        )
        if not response.get("accepted"):
            raise BluPeAPIError(
                f"trajectory {packet.get('trajectory_id')} rejected: {response}"
            )
        return response

    def get_observation_page(
        self,
        handle: SessionHandle,
        *,
        after_step_id: int = -1,
        limit: int = 100,
    ) -> dict[str, Any]:
        return dict(
            self._request(
                "GET",
                f"/v1/sessions/{handle.session_id}/observations"
                f"?after_step_id={int(after_step_id)}&limit={int(limit)}",
                capability=handle.capability,
            )
        )

    def get_episode(self, handle: SessionHandle) -> dict[str, Any]:
        return dict(
            self._request(
                "GET",
                f"/v1/sessions/{handle.session_id}/episode",
                capability=handle.capability,
            )
        )

    def get_episode_trace(self, handle: SessionHandle) -> dict[str, Any]:
        return dict(
            self._request(
                "GET",
                f"/v1/sessions/{handle.session_id}/episode/trace",
                capability=handle.capability,
            )
        )

    def stop_session(self, handle: SessionHandle, reason: str = "policy_complete") -> dict[str, Any]:
        return dict(
            self._request(
                "POST",
                f"/v1/sessions/{handle.session_id}/stop",
                payload={"reason": reason},
                capability=handle.capability,
            )
        )

    def find_trajectory_result(
        self, handle: SessionHandle, trajectory_id: str
    ) -> dict[str, Any] | None:
        """Search the episode trace for a trajectory's terminal result."""
        trace = self.get_episode_trace(handle)
        for key in ("trajectory_results", "results", "events"):
            entries = trace.get(key)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if (
                    isinstance(entry, dict)
                    and entry.get("trajectory_id") == trajectory_id
                    and entry.get("status") in {"accepted", "rejected", "completed", "aborted"}
                ):
                    return dict(entry)
        return None

    def wait_for_trajectory(
        self,
        handle: SessionHandle,
        trajectory_id: str,
        *,
        timeout_s: float,
        poll_interval_s: float = 1.0,
    ) -> dict[str, Any]:
        """Poll the trace until the trajectory completes, aborts, or times out.

        ``accepted`` alone is not execution; keep polling until a terminal
        ``completed`` / ``aborted`` / ``rejected`` result (or timeout).
        """
        deadline = self._clock() + timeout_s
        while True:
            result = self.find_trajectory_result(handle, trajectory_id)
            if result is not None and result.get("status") in {"completed", "aborted", "rejected"}:
                return result
            snapshot = self.get_session(handle)
            status = str(snapshot.get("status", ""))
            if status in {"stopped", "timed_out", "disconnected", "safety_aborted"}:
                raise BluPeAPIError(
                    f"session ended in '{status}' while trajectory {trajectory_id} was active"
                )
            if self._clock() >= deadline:
                raise BluPeAPIError(
                    f"trajectory {trajectory_id} did not finish within {timeout_s:.0f}s"
                )
            self._sleep(poll_interval_s)
