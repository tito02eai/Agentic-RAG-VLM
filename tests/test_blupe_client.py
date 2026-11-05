"""Protocol-level unit tests for the BluPe Session API client.

A fake ``urlopen`` exercises the wire contract without network access: URL
construction, capability bearer headers, 4xx/5xx retry policy, trajectory
packet splitting, lease refresh on submit, and trace polling semantics.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import urllib.error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from agentic_rag_vlm.platform.blupe_client import (  # noqa: E402
    MAX_SESSION_WAYPOINTS,
    MAX_WAYPOINTS_PER_PACKET,
    WAYPOINT_STEP_ID_MAX,
    BluPeAPIError,
    BluPeConnectionError,
    BluPeSessionClient,
    SessionHandle,
    TrajectoryBuilder,
)

ENDPOINT = "https://fake-blupe.example"
SESSION_ID = "sess_0001"
CAPABILITY = "cap_secret_token"


class FakeResponse:
    def __init__(self, payload) -> None:
        self._raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakeHTTP:
    """Callable replacement for ``urllib.request.urlopen`` with routes."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.routes: list[tuple[object, object]] = []

    def route(self, predicate, responder) -> "FakeHTTP":
        self.routes.append((predicate, responder))
        return self

    def __call__(self, request, timeout=None):
        call = {
            "method": request.get_method(),
            "url": request.full_url,
            "authorization": request.headers.get("Authorization"),
            "body": json.loads(request.data.decode("utf-8")) if request.data else None,
        }
        self.calls.append(call)
        for predicate, responder in self.routes:
            if predicate(call["method"], call["url"]):
                payload = responder(call)
                if isinstance(payload, Exception):
                    raise payload
                return FakeResponse(payload)
        raise AssertionError(f"no fake route for {call['method']} {call['url']}")


def http_error(code: int, detail: str = '{"error": "boom"}') -> urllib.error.HTTPError:
    return urllib.error.HTTPError("url", code, "error", None, io.BytesIO(detail.encode("utf-8")))


def make_client(fake: FakeHTTP, **kwargs) -> BluPeSessionClient:
    defaults = {
        "request_timeout_s": 5.0,
        "max_retries": 0,
        "sleeper": lambda seconds: None,
        "clock": lambda: 0.0,
    }
    defaults.update(kwargs)
    return BluPeSessionClient(ENDPOINT, **defaults)


def make_handle(**overrides) -> SessionHandle:
    fields = {
        "session_id": SESSION_ID,
        "capability": CAPABILITY,
        "robot_id": "yam-1",
        "status": "running",
        "events_url": f"{ENDPOINT}/events",
        "episode_id": "ep_1",
        "lease_id": "lease_1",
    }
    fields.update(overrides)
    return SessionHandle(**fields)


def test_list_robots_builds_url_and_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeHTTP().route(
        lambda method, url: method == "GET" and url == f"{ENDPOINT}/v1/robots",
        lambda call: {"robots": [{"robot_id": "yam-1"}]},
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    robots = make_client(fake).list_robots()
    assert robots == [{"robot_id": "yam-1"}]


def test_session_requests_carry_capability_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeHTTP().route(
        lambda method, url: method == "GET" and url.endswith(f"/v1/sessions/{SESSION_ID}"),
        lambda call: {"status": "running", "episode_id": "ep_1", "lease_id": "lease_1"},
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    make_client(fake).get_session(make_handle())
    assert fake.calls[0]["authorization"] == f"Bearer {CAPABILITY}"


def test_public_requests_use_env_token_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeHTTP().route(
        lambda method, url: method == "GET" and url.endswith("/v1/robots"),
        lambda call: {"robots": []},
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    monkeypatch.setenv("BLUPE_TEST_TOKEN", "env_token")
    client = make_client(fake, auth_token_env="BLUPE_TEST_TOKEN")
    client.list_robots()
    assert fake.calls[0]["authorization"] == "Bearer env_token"


def test_create_session_parses_handle_and_hides_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeHTTP().route(
        lambda method, url: method == "POST" and url == f"{ENDPOINT}/v1/sessions",
        lambda call: {
            "session_id": SESSION_ID,
            "session_capability": CAPABILITY,
            "robot_id": "yam-1",
            "status": "queued",
            "events_url": f"{ENDPOINT}/events",
            "position": 2,
        },
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    handle = make_client(fake).create_session("yam-1", "test prompt", run_duration_s=120)
    assert handle.session_id == SESSION_ID
    assert handle.capability == CAPABILITY
    assert handle.status == "queued"
    assert handle.position == 2
    public = handle.to_public_dict()
    assert "capability" not in public
    assert CAPABILITY not in json.dumps(public)
    body = fake.calls[0]["body"]
    assert body["robot_id"] == "yam-1"
    assert body["prompt"] == "test prompt"
    assert body["run_duration_s"] == 120


def test_create_session_missing_field_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeHTTP().route(
        lambda method, url: method == "POST" and url.endswith("/v1/sessions"),
        lambda call: {"session_id": SESSION_ID, "status": "queued"},
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    with pytest.raises(BluPeAPIError, match="missing 'session_capability'"):
        make_client(fake).create_session("yam-1", "prompt")


def test_client_errors_are_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeHTTP().route(
        lambda method, url: url_endswith_robots(url),
        lambda call: http_error(404),
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    with pytest.raises(BluPeAPIError, match="HTTP 404"):
        make_client(fake, max_retries=3).list_robots()
    assert len(fake.calls) == 1


def url_endswith_robots(url: str) -> bool:
    return url.endswith("/v1/robots")


def test_server_errors_retry_then_succeed(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}

    def respond(call):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return http_error(503)
        return {"robots": []}

    fake = FakeHTTP().route(lambda method, url: url_endswith_robots(url), respond)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    sleeps: list[float] = []
    robots = make_client(fake, max_retries=2, sleeper=sleeps.append).list_robots()
    assert robots == []
    assert attempts["count"] == 2
    assert len(sleeps) == 1  # backoff between the two attempts


def test_connection_errors_raise_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeHTTP().route(
        lambda method, url: url_endswith_robots(url),
        lambda call: urllib.error.URLError(OSError("unreachable")),
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    with pytest.raises(BluPeConnectionError, match="unreachable"):
        make_client(fake, max_retries=1).list_robots()
    assert len(fake.calls) == 2


def test_session_creation_is_never_automatically_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeHTTP().route(lambda method, url: method == "POST" and url.endswith("/v1/sessions"),
                            lambda call: http_error(503))
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    with pytest.raises(BluPeAPIError):
        make_client(fake, max_retries=3).create_session("yam-1", "test")
    assert len(fake.calls) == 1


def test_fetch_image_downloads_bytes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    jpeg = b"\xff\xd8\xe0fakejpeg"
    fake = FakeHTTP().route(
        lambda method, url: method == "GET" and url.endswith("/images/top.jpg"),
        lambda call: jpeg,
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    destination = tmp_path / "nested" / "top.jpg"
    result = make_client(fake).fetch_image("/images/top.jpg", destination)
    assert result == destination
    assert destination.read_bytes() == jpeg


# --------------------------------------------------------------------- builder


def test_trajectory_builder_validates_arm_and_cadence() -> None:
    with pytest.raises(ValueError, match="arm must be"):
        TrajectoryBuilder("middle")
    with pytest.raises(ValueError, match="cadence_hz"):
        TrajectoryBuilder("right", cadence_hz=20.0)


def test_trajectory_builder_maps_operating_arm_to_right() -> None:
    builder = TrajectoryBuilder("right")
    waypoint = builder.add(
        step_id=0,
        joints_deg=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        gripper=0.5,
        other_joints_deg=[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        other_gripper=0.25,
    )
    assert waypoint.right_joints_deg == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert waypoint.left_joints_deg == (7.0, 8.0, 9.0, 10.0, 11.0, 12.0)
    payload = waypoint.to_dict()
    assert payload["right_gripper"] == 0.5
    assert payload["left_gripper"] == 0.25
    assert payload["step_id"] == 0


def test_trajectory_builder_packets_split_at_300() -> None:
    builder = TrajectoryBuilder("left")
    for step in range(MAX_WAYPOINTS_PER_PACKET + 1):
        builder.add(
            step_id=step,
            joints_deg=[0.0] * 6,
            gripper=0.5,
            other_joints_deg=[0.0] * 6,
            other_gripper=0.5,
        )
    packets = builder.packets(episode_id="ep_1", lease_id="lease_1", prefix="traj")
    assert [len(packet["waypoints"]) for packet in packets] == [MAX_WAYPOINTS_PER_PACKET, 1]
    assert [packet["trajectory_id"] for packet in packets] == ["traj_000", "traj_001"]
    assert all(packet["cadence_hz"] == 10.0 for packet in packets)
    assert all(packet["schema_version"] == 1 for packet in packets)
    assert all(packet["episode_id"] == "ep_1" and packet["lease_id"] == "lease_1" for packet in packets)


def test_trajectory_builder_enforces_wire_limits() -> None:
    builder = TrajectoryBuilder("right")
    with pytest.raises(BluPeAPIError, match="step_id"):
        builder.add(
            step_id=WAYPOINT_STEP_ID_MAX + 1,
            joints_deg=[0.0] * 6,
            gripper=None,
            other_joints_deg=[0.0] * 6,
            other_gripper=None,
        )
    budget = TrajectoryBuilder("right")
    for step in range(MAX_SESSION_WAYPOINTS):
        budget.add(
            step_id=step,
            joints_deg=[0.0] * 6,
            gripper=None,
            other_joints_deg=[0.0] * 6,
            other_gripper=None,
        )
    with pytest.raises(BluPeAPIError, match="budget"):
        budget.add(
            step_id=MAX_SESSION_WAYPOINTS,
            joints_deg=[0.0] * 6,
            gripper=None,
            other_joints_deg=[0.0] * 6,
            other_gripper=None,
        )
    with pytest.raises(BluPeAPIError, match="empty"):
        TrajectoryBuilder("right").packets(episode_id="ep", lease_id="lease", prefix="p")


# ------------------------------------------------------------------- session


def test_submit_refreshes_lease_onto_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeHTTP()
    fake.route(
        lambda method, url: method == "GET" and url.endswith(f"/v1/sessions/{SESSION_ID}"),
        lambda call: {"status": "running", "episode_id": "ep_9", "lease_id": "lease_9"},
    )
    fake.route(
        lambda method, url: method == "POST" and url.endswith(f"/v1/sessions/{SESSION_ID}/trajectories"),
        lambda call: {"accepted": True, "trajectory_id": call["body"]["trajectory_id"]},
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    handle = make_handle(episode_id=None, lease_id=None)
    packet = {
        "trajectory_id": "traj_000",
        "episode_id": "stale",
        "lease_id": "stale",
        "cadence_hz": 10.0,
        "waypoints": [],
    }
    response = make_client(fake).submit_trajectory_packet(handle, packet)
    assert response["accepted"] is True
    assert handle.episode_id == "ep_9" and handle.lease_id == "lease_9"
    submitted = fake.calls[-1]["body"]
    assert submitted["episode_id"] == "ep_9"
    assert submitted["lease_id"] == "lease_9"
    # the caller's packet dict is not mutated
    assert packet["episode_id"] == "stale"


def test_submit_raises_when_not_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeHTTP().route(
        lambda method, url: method == "POST" and url.endswith("/trajectories"),
        lambda call: {"accepted": False, "reason": "robot_busy"},
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    with pytest.raises(BluPeAPIError, match="rejected"):
        make_client(fake).submit_trajectory_packet(make_handle(), {"trajectory_id": "t"})


def test_wait_for_trajectory_polls_until_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    trace_calls = {"count": 0}

    def trace_response(call):
        trace_calls["count"] += 1
        if trace_calls["count"] < 3:
            return {"trajectory_results": []}
        return {"trajectory_results": [{"trajectory_id": "t_000", "status": "completed"}]}

    fake = FakeHTTP()
    fake.route(
        lambda method, url: method == "GET" and url.endswith("/episode/trace"),
        trace_response,
    )
    fake.route(
        lambda method, url: method == "GET" and url.endswith(f"/v1/sessions/{SESSION_ID}"),
        lambda call: {"status": "running", "episode_id": "ep_1", "lease_id": "lease_1"},
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    sleeps: list[float] = []
    result = make_client(fake, sleeper=sleeps.append).wait_for_trajectory(
        make_handle(), "t_000", timeout_s=30.0, poll_interval_s=1.0
    )
    assert result["status"] == "completed"
    assert trace_calls["count"] == 3
    assert len(sleeps) == 2


def test_wait_for_trajectory_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"now": 0.0}
    fake = FakeHTTP()
    fake.route(
        lambda method, url: method == "GET" and url.endswith("/episode/trace"),
        lambda call: {"trajectory_results": []},
    )
    fake.route(
        lambda method, url: method == "GET" and url.endswith(f"/v1/sessions/{SESSION_ID}"),
        lambda call: {"status": "running", "episode_id": "ep_1", "lease_id": "lease_1"},
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    client = make_client(
        fake,
        sleeper=lambda seconds: state.update(now=state["now"] + seconds),
        clock=lambda: state["now"],
    )
    with pytest.raises(BluPeAPIError, match="did not finish"):
        client.wait_for_trajectory(make_handle(), "t_000", timeout_s=3.0, poll_interval_s=1.0)


def test_wait_for_trajectory_raises_when_session_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeHTTP()
    fake.route(
        lambda method, url: method == "GET" and url.endswith("/episode/trace"),
        lambda call: {"trajectory_results": []},
    )
    fake.route(
        lambda method, url: method == "GET" and url.endswith(f"/v1/sessions/{SESSION_ID}"),
        lambda call: {"status": "safety_aborted", "episode_id": "ep_1", "lease_id": "lease_1"},
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    with pytest.raises(BluPeAPIError, match="safety_aborted"):
        make_client(fake).wait_for_trajectory(make_handle(), "t_000", timeout_s=5.0)


def test_wait_for_running_returns_on_running(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeHTTP().route(
        lambda method, url: method == "GET" and url.endswith(f"/v1/sessions/{SESSION_ID}"),
        lambda call: {"status": "running", "episode_id": "ep_1", "lease_id": "lease_1"},
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    handle = make_handle(status="queued", episode_id=None, lease_id=None)
    snapshot = make_client(fake).wait_for_running(handle, timeout_s=5.0)
    assert snapshot["status"] == "running"
    assert handle.episode_id == "ep_1" and handle.lease_id == "lease_1"


def test_wait_for_running_raises_on_terminal_state(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeHTTP().route(
        lambda method, url: method == "GET" and url.endswith(f"/v1/sessions/{SESSION_ID}"),
        lambda call: {"status": "stopped", "episode_id": None, "lease_id": None},
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    with pytest.raises(BluPeAPIError, match="terminal state 'stopped'"):
        make_client(fake).wait_for_running(make_handle(status="queued"), timeout_s=5.0)
