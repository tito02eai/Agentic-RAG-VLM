"""End-to-end offline tests for the remote BluPe transport.

A fake BluPe Session API client (subclassing the real client, routing
``_request``) drives ``RemotePlatformTransport`` together with the *real*
YAM kinematics and motion planner: health check, session lifecycle,
observation download, pick/place execution with step-id continuity, the
safety gate, camera->base transform math, and failure receipts.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from agentic_rag_vlm.platform import (  # noqa: E402
    BluPeSessionClient,
    PlatformCommand,
    PlatformUnavailableError,
    RemotePlatformTransport,
    load_platform_config,
)

PLATFORM_EXAMPLE = PROJECT_ROOT / "configs" / "blupe_platform.example.json"
ROBOT_ID = "yam-1"
SESSION_ID = "sess_0001"
CAPABILITY = "cap_secret_token"
# Base-frame top-grasp positions validated against the real kinematics.
PICK_POSITION_M = [0.22, 0.00, 0.16]
PLACE_DESTINATION_M = [0.20, -0.05, 0.15]
UNREACHABLE_M = [5.0, 0.0, 0.5]


def station_observation(**overrides: Any) -> dict[str, Any]:
    station = {
        "observed_at": "2026-11-02T00:00:00Z",
        "mode": "autonomous",
        "homed": True,
        "settled": True,
        "queue_ready": True,
        "safety": {"ok": True, "estop_engaged": False, "reason": None},
        "left_joints_deg": [0.0] * 6,
        "right_joints_deg": [0.0] * 6,
        "left_gripper": 0.5,
        "right_gripper": 0.5,
        "images": {
            role: {"url": f"/images/{role}.jpg"} for role in ("top", "left", "right")
        },
    }
    station.update(overrides)
    return station


def calibration_payload(right_matrix: list[list[float]] | None = None) -> dict[str, Any]:
    if right_matrix is None:
        right_matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    return {
        "calibration_id": "cal_20261025",
        "cameras": {
            "top": {
                "K": [[500.0, 0.0, 320.0], [0.0, 500.0, 180.0], [0.0, 0.0, 1.0]],
                "T_base_camera": {"right": right_matrix, "left": right_matrix},
            }
        },
    }


class FakeBluPeClient(BluPeSessionClient):
    """Routes every request locally; records traffic for assertions."""

    def __init__(self, station: dict[str, Any] | None = None, calibration: dict[str, Any] | None = None):
        super().__init__("https://fake-blupe.example")
        self.station = station or station_observation()
        self.calibration = calibration or calibration_payload()
        self.session_snapshot = {
            "status": "running",
            "episode_id": "ep_1",
            "lease_id": "lease_1",
            "position": None,
        }
        self.trace: list[dict[str, Any]] = []
        self.submitted_packets: list[dict[str, Any]] = []
        self.created = False

    def _request(self, method, path, *, payload=None, capability=None, is_json=True):
        if method == "GET" and path == "/v1/robots":
            return {"robots": [{"robot_id": ROBOT_ID, "calibration_url": "/v1/robots/yam-1/calibration"}]}
        if method == "GET" and path == f"/v1/robots/{ROBOT_ID}/queue":
            return {"entries": [], "stations": [{"connected": True, "available": True}]}
        if method == "GET" and path == f"/v1/robots/{ROBOT_ID}/observation":
            return dict(self.station)
        if method == "GET" and path == f"/v1/robots/{ROBOT_ID}/calibration":
            return dict(self.calibration)
        if method == "GET" and path.endswith(".jpg"):
            return f"fake-jpeg-bytes:{Path(path).stem}".encode("utf-8")
        if method == "POST" and path == "/v1/sessions":
            self.created = True
            return {
                "session_id": SESSION_ID,
                "session_capability": CAPABILITY,
                "robot_id": ROBOT_ID,
                "status": "queued",
                "events_url": "https://fake-blupe.example/events",
                "episode_id": None,
                "lease_id": None,
                "position": 0,
            }
        if method == "GET" and path == f"/v1/sessions/{SESSION_ID}":
            return dict(self.session_snapshot)
        if method == "POST" and path == f"/v1/sessions/{SESSION_ID}/trajectories":
            self.submitted_packets.append(dict(payload))
            self.trace.append(
                {"trajectory_id": payload["trajectory_id"], "status": "completed"}
            )
            return {"accepted": True, "trajectory_id": payload["trajectory_id"]}
        if method == "GET" and path == f"/v1/sessions/{SESSION_ID}/episode/trace":
            return {"trajectory_results": [dict(entry) for entry in self.trace]}
        if method == "POST" and path == f"/v1/sessions/{SESSION_ID}/stop":
            return {"stopped": True, "reason": payload.get("reason")}
        raise AssertionError(f"unexpected request: {method} {path}")


def make_transport(
    *,
    client: FakeBluPeClient | None = None,
    image_dir: Path | str = "output/test_blupe_images",
    destinations: dict[str, list[float]] | None = None,
    station: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
) -> RemotePlatformTransport:
    client = client or FakeBluPeClient(station, calibration)
    return RemotePlatformTransport(
        load_platform_config(PLATFORM_EXAMPLE),
        client=client,
        destinations=destinations,
        image_dir=image_dir,
    )


def pick_command(position: list[float], command_id: str = "cmd_0001_pick") -> PlatformCommand:
    return PlatformCommand(
        command_id=command_id,
        command_type="pick",
        target="cup",
        origin="test",
        parameters={"position_xyz_m": position, "aperture_m": 0.07, "approach": "top"},
    )


def place_command(destination: str = "tray_zone", command_id: str = "cmd_0002_place") -> PlatformCommand:
    return PlatformCommand(
        command_id=command_id,
        command_type="place",
        target="cup",
        origin="test",
        parameters={"destination": destination, "aperture_m": 0.07, "approach": "top"},
    )


def started_transport(**kwargs: Any) -> tuple[RemotePlatformTransport, FakeBluPeClient]:
    client = kwargs.pop("client", None) or FakeBluPeClient(
        kwargs.pop("station", None), kwargs.pop("calibration", None)
    )
    transport = make_transport(client=client, **kwargs)
    transport.start_session(timeout_s=5.0)
    return transport, client


def test_aborted_packet_prevents_followup_packet_submission() -> None:
    from agentic_rag_vlm.platform.motion import MotionPlan, MotionKeyframe
    transport, client = started_transport()
    transport.motion_planner.max_joint_delta_deg_per_step = .1
    plan = MotionPlan("right", [MotionKeyframe("start", (0.,)*6, .5),
                               MotionKeyframe("end", (100.,)*6, .5)], (0.,)*6, .5)
    client.wait_for_trajectory = lambda *args, **kwargs: {"status": "aborted"}
    receipt = transport._execute_plan(pick_command(PICK_POSITION_M), plan)
    assert not receipt["accepted"] and len(client.submitted_packets) == 1


def test_health_check_reports_station_state() -> None:
    transport, client = started_transport()
    healthy = dict(transport.health_check())
    assert healthy["platform"] == "blupe"
    assert healthy["robot_registered"] is True
    assert healthy["station_connected"] is True
    assert healthy["queue_entries"] == 0
    assert healthy["calibration_id"] == "cal_20261025"
    assert healthy["top_camera_matrix_present"] is True


def test_session_lifecycle_start_and_stop() -> None:
    transport, client = started_transport()
    assert transport.handle is not None
    assert transport.handle.session_id == SESSION_ID
    assert transport.handle.status == "running"
    assert client.created is True
    again = dict(transport.start_session())
    assert again["already_started"] is True
    stopped = dict(transport.stop_session(reason="policy_complete"))
    assert stopped["stopped"] is True
    assert stopped["session_id"] == SESSION_ID
    idle = dict(RemotePlatformTransport(load_platform_config(PLATFORM_EXAMPLE), client=FakeBluPeClient()).stop_session())
    assert idle == {"stopped": False, "reason": "no_active_session"}


def test_get_observation_downloads_camera_images(tmp_path: Path) -> None:
    transport, _ = started_transport(image_dir=tmp_path / "cameras")
    observation = transport.get_observation()
    assert observation.observation_id == "blupe_obs_0001"
    assert observation.objects == ()
    assert len(observation.image_paths) == 3
    for path in observation.image_paths:
        assert Path(path).exists()
        assert Path(path).read_bytes().startswith(b"fake-jpeg-bytes")
    receipt = dict(observation.receipt)
    assert receipt["robot_id"] == ROBOT_ID
    assert receipt["joints_deg"]["right"] == [0.0] * 6
    assert receipt["station"]["safety"]["ok"] is True
    assert "capability" not in json.dumps(receipt)


def test_pick_command_executes_trajectory_packets() -> None:
    transport, client = started_transport()
    receipt = dict(transport.send_command(pick_command(PICK_POSITION_M)))
    assert receipt["accepted"] is True
    assert receipt["arm"] == "right"
    assert [frame["name"] for frame in receipt["keyframes"]] == [
        "start", "approach_open", "descend", "close", "lift",
    ]
    assert receipt["waypoints"] > 4
    assert receipt["step_id_range"] == [0, receipt["waypoints"] - 1]
    assert len(client.submitted_packets) >= 1
    packet = client.submitted_packets[0]
    assert packet["cadence_hz"] == 10.0
    assert packet["episode_id"] == "ep_1" and packet["lease_id"] == "lease_1"
    waypoint = packet["waypoints"][0]
    assert waypoint["right_joints_deg"] == [0.0] * 6  # starts from observed home
    assert waypoint["left_joints_deg"] == [0.0] * 6  # non-operating arm frozen
    assert all(item["result"]["status"] == "completed" for item in receipt["packets"])
    # audit discipline: the capability never leaks into receipts
    assert CAPABILITY not in json.dumps(receipt)


def test_step_ids_continue_across_commands() -> None:
    transport, _ = started_transport()
    first = dict(transport.send_command(pick_command(PICK_POSITION_M)))
    second = dict(transport.send_command(pick_command(PICK_POSITION_M, "cmd_0002_pick")))
    assert second["step_id_range"][0] == first["waypoints"]
    assert second["session_waypoints_used"] == first["waypoints"] + second["waypoints"]


def test_place_resolves_named_destination() -> None:
    transport, _ = started_transport(destinations={"tray_zone": PLACE_DESTINATION_M})
    receipt = dict(transport.send_command(place_command()))
    assert receipt["accepted"] is True
    assert [frame["name"] for frame in receipt["keyframes"]] == [
        "start", "above_destination", "descend_to_destination", "release", "retract",
    ]


def test_place_with_unknown_destination_fails_closed() -> None:
    transport, _ = started_transport(destinations={})
    with pytest.raises(PlatformUnavailableError, match="no coordinate table entry"):
        transport.send_command(place_command("unknown_zone"))


def test_pick_without_position_fails_closed() -> None:
    transport, _ = started_transport()
    command = PlatformCommand(
        command_id="cmd_bad", command_type="pick", target="cup", origin="test", parameters={}
    )
    with pytest.raises(PlatformUnavailableError, match="no position_xyz_m"):
        transport.send_command(command)


def test_unreachable_pick_returns_rejected_receipt() -> None:
    transport, _ = started_transport()
    receipt = dict(transport.send_command(pick_command(UNREACHABLE_M)))
    assert receipt["accepted"] is False
    assert receipt["reason"] == "motion_plan_failed"
    assert "IK failed" in receipt["detail"]
    assert len(transport.client.submitted_packets) == 0  # nothing reached the wire


def test_safety_gate_blocks_estop_and_unsettled_station() -> None:
    estop_transport, _ = started_transport(
        station=station_observation(safety={"ok": False, "estop_engaged": True, "reason": "estop"})
    )
    with pytest.raises(PlatformUnavailableError, match="e-stop"):
        estop_transport.send_command(pick_command(PICK_POSITION_M))

    unsettled_transport, _ = started_transport(
        station=station_observation(homed=False, settled=False)
    )
    with pytest.raises(PlatformUnavailableError, match="not homed/settled"):
        unsettled_transport.send_command(pick_command(PICK_POSITION_M))


def test_camera_to_base_applies_calibration_transform() -> None:
    # 180-degree rotation about X plus a translation
    matrix = [
        [1.0, 0.0, 0.0, 0.10],
        [0.0, -1.0, 0.0, 0.20],
        [0.0, 0.0, -1.0, 0.30],
    ]
    transport, _ = started_transport(calibration=calibration_payload(matrix))
    mapped = transport._camera_to_base((0.0, 0.0, 1.0))
    assert mapped == pytest.approx((0.10, 0.20, 0.30 - 1.0))
    mapped_point = transport._camera_to_base((0.5, 0.25, 0.75))
    expected = np.array(matrix) @ np.array([0.5, 0.25, 0.75, 1.0])
    assert mapped_point == pytest.approx(tuple(expected[:3]))


def test_calibration_missing_transform_fails_closed() -> None:
    broken = calibration_payload()
    broken["cameras"]["top"]["T_base_camera"] = {}
    transport, _ = started_transport(calibration=broken)
    with pytest.raises(PlatformUnavailableError, match="T_base_camera"):
        transport._camera_to_base((0.0, 0.0, 1.0))


def test_station_joint_state_missing_fails_closed() -> None:
    station = station_observation()
    station.pop("right_joints_deg")
    transport, _ = started_transport(station=station)
    with pytest.raises(PlatformUnavailableError, match="6-joint state"):
        transport.send_command(pick_command(PICK_POSITION_M))


def test_transport_rejects_non_blupe_platform(tmp_path: Path) -> None:
    config = load_platform_config(PLATFORM_EXAMPLE)
    broken = tmp_path / "broken.json"
    broken.write_text(
        json.dumps(
            {
                "platform": "mock",
                "auth_token_env": "X",
                "endpoint_base": "https://x",
                "robot_id": "yam-1",
                "camera_ids": ["top"],
                "request_timeout_s": 1.0,
                "max_retries": 0,
            }
        ),
        encoding="utf-8",
    )
    mock_config = load_platform_config(broken)
    with pytest.raises(ValueError, match="targets 'blupe'"):
        RemotePlatformTransport(mock_config, client=FakeBluPeClient())
    with pytest.raises(ValueError, match="arm must be"):
        RemotePlatformTransport(config, client=FakeBluPeClient(), arm="middle")
