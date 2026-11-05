"""Offline regression tests for the platform deployment layer.

Everything here runs without network access, MuJoCo, or the school asset
bundle: the mock transport and a fake VLM client exercise the config,
registry, perception, mapper, verifier, and the full session loop.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from agentic_rag_vlm.pipeline import AgenticPipeline, GraspAction  # noqa: E402
from agentic_rag_vlm.platform import (  # noqa: E402
    BluPeSession,
    GraspActionMapper,
    MockPlatformTransport,
    ObjectMetadata,
    PerceptionError,
    PlatformObservation,
    PlatformUnavailableError,
    RegistryMetadataPerception,
    RemotePlatformTransport,
    SceneChangeVerifier,
    VLMObjectPerception,
    load_platform_config,
)
from agentic_rag_vlm.runtime import AgenticRuntime, RuntimeConfig, Subgoal, TaskSpec  # noqa: E402
from scripts.agentic_framework import (  # noqa: E402
    PublicObject,
    load_semantic_registry,
    semantic_adapter,
)


BLUPE_REGISTRY = PROJECT_ROOT / "knowledge_base" / "blupe_semantic_registry.json"
BLUPE_CARDS = json.loads(
    (PROJECT_ROOT / "knowledge_base" / "blupe_affordance_cards.json").read_text(encoding="utf-8")
)["cards"]
PLATFORM_EXAMPLE = PROJECT_ROOT / "configs" / "blupe_platform.example.json"


class Estimate:
    def __init__(self, center: tuple[float, float, float], extent: tuple[float, float, float]) -> None:
        self.center_xyz_m = center
        self.visible_surface_min_xyz_m = [center[i] - extent[i] / 2 for i in range(3)]
        self.visible_surface_max_xyz_m = [center[i] + extent[i] / 2 for i in range(3)]


def test_semantic_registry_loads_and_drives_adapter() -> None:
    registry = load_semantic_registry(BLUPE_REGISTRY)
    assert {"cup", "bottle", "banana", "cube", "vase"} <= set(registry)
    objects = semantic_adapter(
        {"cup": Estimate((0.3, 0.1, 0.82), (0.08, 0.08, 0.1)), "robot_base": Estimate((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))},
        registry=registry,
    )
    assert set(objects) == {"cup"}
    cup = objects["cup"]
    assert cup.category == "cup" and cup.material == "ceramic" and cup.fragility == 0.6


def test_semantic_registry_rejects_missing_fields(tmp_path: Path) -> None:
    bad = tmp_path / "bad_registry.json"
    bad.write_text(json.dumps({"objects": {"cup": {"category": "cup"}}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_semantic_registry(bad)


def test_platform_config_loads_example_and_validates(tmp_path: Path) -> None:
    config = load_platform_config(PLATFORM_EXAMPLE)
    assert config.platform == "blupe"
    assert config.auth_token_env == "BLUPE_API_TOKEN"
    assert config.camera_ids == ("top", "left", "right")
    assert config.endpoint_base is not None and config.endpoint_base.startswith("https://")
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"platform": "blupe"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_platform_config(broken)


def test_remote_transport_requires_session_before_commands() -> None:
    transport = RemotePlatformTransport(load_platform_config(PLATFORM_EXAMPLE))
    with pytest.raises(PlatformUnavailableError, match="no active session"):
        transport._require_handle()


def test_mapper_translates_grasp_action_into_commands() -> None:
    action = GraspAction(
        target="cup",
        position_xyz_m=(0.31, 0.1, 0.85),
        strategy={"synergy": "power", "force_n": 8.0, "aperture_m": 0.06, "approach": "top"},
        constraint={"active": False, "reason": "no_fragile_neighbor"},
        source="haa_rag",
    )
    mapper = GraspActionMapper(command_prefix="test")
    pick = mapper.pick_command(action)
    place = mapper.place_command(action, "tray_zone")
    assert pick.command_type == "pick" and pick.target == "cup" and pick.origin == "haa_rag"
    assert pick.parameters["synergy"] == "power" and pick.parameters["position_xyz_m"] == [0.31, 0.1, 0.85]
    assert pick.constraint["reason"] == "no_fragile_neighbor"
    assert place.command_type == "place" and place.parameters["destination"] == "tray_zone"
    assert "destination_xyz_m" not in place.parameters


def test_registry_perception_skips_unknown_and_uses_category_fallback() -> None:
    perception = RegistryMetadataPerception(load_semantic_registry(BLUPE_REGISTRY))
    observation = PlatformObservation(
        observation_id="obs1",
        objects=(
            ObjectMetadata("cup", (0.3, 0.1, 0.82)),
            ObjectMetadata("unknown_widget", (0.0, 0.0, 0.8)),
            ObjectMetadata("platform_object_7", (0.1, 0.2, 0.8), category="bottle"),
        ),
    )
    objects, receipt = perception.perceive(observation)
    assert set(objects) == {"cup", "platform_object_7"}
    assert objects["platform_object_7"].category == "bottle"
    assert receipt.mode == "registry_metadata"
    assert any("unknown_widget" in warning for warning in receipt.warnings)


class FakeVLM:
    def __init__(self, parsed: Any, raw_text: str = "") -> None:
        self._parsed = parsed
        self.raw_text = raw_text or json.dumps(parsed)

    def infer(self, images: list[Path], prompt: str) -> Any:
        class _Result:
            pass

        result = _Result()
        result.parsed = self._parsed
        result.raw_text = self.raw_text
        result.latency_ms = 1.0
        result.requested_model = "fake"
        result.response_model = "fake"
        result.usage = {"total_tokens": 1}
        result.parse_error = None
        return result


def test_vlm_perception_normalizes_semantics_from_registry() -> None:
    parsed = {
        "objects": [
            {
                "name": "detected_cup",
                "category": "cup",
                "superclass": "container",
                "affordance_type": "graspable_body",
                "material": "steel",
                "fragility": 0.9,
                "region": "body",
                "center_xyz_m": None,
                "visual_aspect": 0.8,
            },
            {
                "name": "mystery_mug",
                "category": "mug",
                "superclass": "container",
                "affordance_type": "graspable_body",
                "material": "ceramic",
                "fragility": 0.5,
                "region": "body",
                "center_xyz_m": [0.1, 0.2, 0.82],
                "visual_aspect": 0.9,
            },
        ]
    }
    observation = PlatformObservation(
        observation_id="obs2",
        objects=(ObjectMetadata("detected_cup", (0.3, 0.1, 0.82)),),
        image_paths=("frame_front.png",),
    )
    perception = VLMObjectPerception(FakeVLM(parsed), load_semantic_registry(BLUPE_REGISTRY))
    objects, receipt = perception.perceive(observation)
    assert set(objects) == {"detected_cup", "mystery_mug"}
    # registry-matched category: frozen registry semantics override VLM guesses
    assert objects["detected_cup"].category == "cup"
    assert objects["detected_cup"].material == "ceramic"
    assert objects["detected_cup"].fragility == 0.6
    assert objects["detected_cup"].center_xyz_m == (0.3, 0.1, 0.82)  # merged from metadata
    # unregistered category: VLM semantics retained, flagged in the receipt
    assert objects["mystery_mug"].category == "mug"
    assert any("unregistered_category_used_vlm_semantics:mystery_mug" in warning for warning in receipt.warnings)
    assert any("center_from_metadata:detected_cup" in warning for warning in receipt.warnings)
    assert receipt.vlm_result is not None and receipt.vlm_result["requested_model"] == "fake"


def test_vlm_perception_rejects_unparseable_output() -> None:
    class BrokenVLM:
        def infer(self, images: list[Path], prompt: str) -> Any:
            class _Result:
                parsed = None
                raw_text = "not json"
                parse_error = "no json object"

            return _Result()

    perception = VLMObjectPerception(BrokenVLM(), load_semantic_registry(BLUPE_REGISTRY))
    observation = PlatformObservation(
        observation_id="obs3",
        objects=(),
        image_paths=("frame_front.png",),
    )
    with pytest.raises(PerceptionError):
        perception.perceive(observation)


def blupe_object(name: str, xy: tuple[float, float]) -> PublicObject:
    registry = load_semantic_registry(BLUPE_REGISTRY)
    entry = registry[name]
    return PublicObject(
        name,
        (xy[0], xy[1], 0.82),
        entry["category"],
        entry["superclass"],
        entry["affordance_type"],
        entry["material"],
        entry["fragility"],
        entry["region"],
        1.0,
    )


def test_verifier_accepts_moved_target_with_protected_intact() -> None:
    verifier = SceneChangeVerifier()
    before = {"cup": blupe_object("cup", (0.3, 0.1)), "vase": blupe_object("vase", (0.42, 0.02))}
    after = {"cup": blupe_object("cup", (-0.25, 0.2)), "vase": blupe_object("vase", (0.42, 0.02))}
    result = verifier.verify("cup", ["vase"], before, after)
    assert result.success and result.quality == 1.0
    assert result.evidence["displacements"]["cup"]["distance_m"] > 0.5


def test_verifier_fails_when_target_did_not_move_or_protected_displaced() -> None:
    verifier = SceneChangeVerifier()
    before = {"cup": blupe_object("cup", (0.3, 0.1)), "vase": blupe_object("vase", (0.42, 0.02))}
    same = verifier.verify("cup", ["vase"], before, dict(before))
    assert not same.success and "target_did_not_move" in same.evidence["problems"]
    moved_vase = {"cup": blupe_object("cup", (-0.25, 0.2)), "vase": blupe_object("vase", (0.30, 0.02))}
    violated = verifier.verify("cup", ["vase"], before, moved_vase)
    assert not violated.success and "protected_object_displaced" in violated.evidence["problems"]


def build_session(scene: dict[str, tuple[float, float, float]], destinations: dict[str, tuple[float, float, float]]) -> tuple[BluPeSession, AgenticRuntime]:
    runtime = AgenticRuntime(AgenticPipeline(BLUPE_CARDS), RuntimeConfig(maximum_replans=1))
    session = BluPeSession(
        runtime,
        MockPlatformTransport(scene, destinations=destinations),
        RegistryMetadataPerception(load_semantic_registry(BLUPE_REGISTRY)),
        GraspActionMapper(command_prefix="test"),
        SceneChangeVerifier(),
        destinations=destinations,
    )
    return session, runtime


def test_session_completes_stage1_pick_and_place() -> None:
    session, runtime = build_session(
        {"cup": (0.30, 0.10, 0.80), "vase": (0.42, 0.02, 0.80)},
        {"tray_zone": (-0.25, 0.20, 0.80)},
    )
    task = TaskSpec("stage1", "pick the cup", (Subgoal("cup", "tray_zone"),), ())
    result = session.run(task)
    assert result["outcome"] == "complete"
    assert runtime.completed_targets == ["cup"]
    kinds = [receipt["kind"] for receipt in result["receipts"]]
    assert "observation" in kinds and "perception" in kinds and "command" in kinds and "verification" in kinds
    command_types = [
        item["payload"]["sent"]["command_type"]
        for item in result["receipts"]
        if item["kind"] == "command"
    ]
    assert command_types == ["pick", "place"]
    verification = next(item for item in result["receipts"] if item["kind"] == "verification")
    assert verification["payload"]["success"] is True


def test_contact_failure_evicts_memory_when_contact_recovery_is_disabled():
    session, runtime = build_session({'cube': (.3, .1, .82)}, {'tray': (-.25, .2, .82)})
    target = blupe_object('cube', (.3, .1))
    action, _ = runtime.pipeline.plan(target, {'cube': target})
    runtime.pipeline.remember('cube', action, 1., 'earlier_success')
    session.transport.send_command = lambda command: {'accepted': False, 'execution_started': True, 'reason': 'grasp_not_retained'}
    result = session.run(TaskSpec('failed_reuse', 'place cube', (Subgoal('cube', 'tray'),)))
    assert result['outcome'] == 'safe_stop'
    assert runtime.pipeline.memory.get('cube') is None
    assert any(r['kind'] == 'memory_invalidation' for r in result['receipts'])
    # Execution-time eviction must also reach runtime consumers (UI/event sink),
    # even though post-placement verification was never reached.
    events = [e for e in result['trace'] if e['event'] == 'memory_invalidation']
    assert len(events) == 1
    assert events[0]['payload']['reason'] == 'observed_retention_failure'


@pytest.mark.parametrize('force', [.96, 1.61, float('nan'), True])
def test_execution_force_feedback_is_bounded_and_only_verified_value_is_remembered(force):
    session, runtime = build_session({'cube': (.3, .1, .82)}, {'tray': (-.25, .2, .82)})
    target = blupe_object('cube', (.3, .1))
    action, _ = runtime.pipeline.plan(target, {'cube': target})
    runtime.pipeline.remember('cube', replace(action, strategy={**action.strategy, 'force_n': .8}), 1., 'earlier_success')
    session.mapper = GraspActionMapper(retention_force_adjustment=True)
    original = session.transport.send_command
    def send(command):
        receipt = dict(original(command))
        if command.command_type == 'pick':
            receipt['execution_feedback'] = {'force_n': force, 'source': 'public_tactile_low_height_force_adaptation'}
        return receipt
    session.transport.send_command = send
    result = session.run(TaskSpec('feedback', 'place cube', (Subgoal('cube', 'tray'),)))
    if force == .96 and type(force) is float:
        assert result['outcome'] == 'complete'
        assert runtime.pipeline.memory.get('cube').strategy['base_force_n'] == .96
    else:
        assert result['outcome'] == 'safe_stop'
        assert result['trace'][-1]['payload']['reason'] == 'invalid_execution_parameter_feedback'


def test_session_safe_stops_when_place_cannot_resolve_destination() -> None:
    session, runtime = build_session({"cup": (0.30, 0.10, 0.80)}, {})
    task = TaskSpec("stage1_bad", "pick the cup", (Subgoal("cup", "missing_zone"),), ())
    result = session.run(task)
    assert result["outcome"] == "safe_stop"
    assert runtime.completed_targets == []
    stop_events = [event for event in result["trace"] if event["event"] == "safe_stop"]
    assert stop_events and stop_events[0]["payload"]["reason"] == "destination_unavailable"
    assert session.transport.command_log == []


def test_session_stage2_protects_vase_and_completes() -> None:
    session, runtime = build_session(
        {"cup": (0.30, 0.10, 0.80), "vase": (0.34, 0.16, 0.80)},
        {"tray_zone": (-0.25, 0.20, 0.80)},
    )
    task = TaskSpec("stage2", "pick the cup, protect the vase", (Subgoal("cup", "tray_zone"),), ("vase",))
    result = session.run(task)
    assert result["outcome"] == "complete"
    assert runtime.completed_targets == ["cup"]
    verification = next(item for item in result["receipts"] if item["kind"] == "verification")
    assert verification["payload"]["result"]["protected_violations"] == []


def test_verifier_rejects_movement_to_wrong_destination() -> None:
    before = {"cup": blupe_object("cup", (0.3, 0.1))}
    after = {"cup": blupe_object("cup", (0.1, 0.1))}
    result = SceneChangeVerifier().verify(
        "cup", [], before, after, destination_xyz_m=(-0.25, 0.2, 0.82)
    )
    assert not result.success
    assert "target_outside_destination" in result.evidence["problems"]


@pytest.mark.parametrize("missing_from", ["before", "after"])
def test_verifier_requires_protected_object_in_both_observations(missing_from: str) -> None:
    before = {"cup": blupe_object("cup", (0.3, 0.1)), "vase": blupe_object("vase", (0.4, 0.1))}
    after = {**before, "cup": blupe_object("cup", (-0.25, 0.2))}
    (before if missing_from == "before" else after).pop("vase")
    result = SceneChangeVerifier().verify("cup", ["vase"], before, after)
    assert not result.success
    assert result.evidence["unobserved_protected"] == ["vase"]


@pytest.mark.parametrize("z", [float("nan"), float("inf")])
def test_verifier_rejects_nonfinite_positions(z: float) -> None:
    before = {"cup": blupe_object("cup", (0.3, 0.1))}
    after = {"cup": replace(before["cup"], center_xyz_m=(-0.25, 0.2, z))}
    result = SceneChangeVerifier().verify("cup", [], before, after)
    assert not result.success
    assert result.evidence["invalid_positions"] == ["cup"]


def test_verifier_rejects_vertical_protected_displacement() -> None:
    before = {"cup": blupe_object("cup", (0.3, 0.1)), "vase": blupe_object("vase", (0.4, 0.1))}
    after = {
        "cup": blupe_object("cup", (-0.25, 0.2)),
        "vase": replace(before["vase"], center_xyz_m=(0.4, 0.1, 0.9)),
    }
    result = SceneChangeVerifier().verify("cup", ["vase"], before, after)
    assert not result.success and result.evidence["protected_violations"] == ["vase"]


@pytest.mark.parametrize("rejected_type", ["pick", "place"])
def test_session_stops_on_rejected_command_without_learning(rejected_type: str) -> None:
    session, runtime = build_session({"cup": (0.3, 0.1, 0.8)}, {"tray": (-0.25, 0.2, 0.8)})
    original_send = session.transport.send_command
    sent = []

    def reject(command):
        sent.append(command.command_type)
        if command.command_type == rejected_type:
            return {"accepted": False, "reason": "test_executor_rejection"}
        return original_send(command)

    session.transport.send_command = reject
    result = session.run(TaskSpec("reject", "move cup", (Subgoal("cup", "tray"),)))
    assert result["outcome"] == "safe_stop"
    assert result["trace"][-1]["payload"]["reason"] == "command_rejected"
    assert sent == (["pick"] if rejected_type == "pick" else ["pick", "place"])
    assert runtime.pipeline.memory.to_dict() == {}
    assert runtime.completed_targets == []


def test_session_checks_protected_visibility_before_motion() -> None:
    session, _ = build_session({"cup": (0.3, 0.1, 0.8)}, {"tray": (-0.25, 0.2, 0.8)})
    result = session.run(TaskSpec("missing", "protect vase", (Subgoal("cup", "tray"),), ("vase",)))
    assert result["outcome"] == "safe_stop"
    assert session.transport.command_log == []


def test_session_does_not_learn_from_wrong_placement() -> None:
    session, runtime = build_session({"cup": (0.3, 0.1, 0.8)}, {"tray": (-0.25, 0.2, 0.8)})
    original_send = session.transport.send_command

    def misplace(command):
        if command.command_type == "place":
            command = replace(command, parameters={"destination_xyz_m": (0.1, 0.1, 0.8)})
        return original_send(command)

    session.transport.send_command = misplace
    result = session.run(TaskSpec("misplace", "move cup", (Subgoal("cup", "tray"),)))
    assert result["outcome"] == "safe_stop"
    assert runtime.pipeline.memory.to_dict() == {}
    verification = next(r for r in result["receipts"] if r["kind"] == "verification")
    assert "target_outside_destination" in verification["payload"]["result"]["problems"]


def test_session_refuses_to_move_a_protected_target() -> None:
    session, _ = build_session({"cup": (0.3, 0.1, 0.8)}, {"tray": (-0.25, 0.2, 0.8)})
    result = session.run(TaskSpec("conflict", "move cup", (Subgoal("cup", "tray"),), ("cup",)))
    assert result["outcome"] == "safe_stop"
    assert session.transport.command_log == []
    assert result["trace"][-1]["payload"]["reason"] == "target_is_protected"


def test_session_rejects_nonfinite_observation_before_motion() -> None:
    session, _ = build_session({"cup": (float("nan"), 0.1, 0.8)}, {"tray": (-0.25, 0.2, 0.8)})
    result = session.run(TaskSpec("invalid", "move cup", (Subgoal("cup", "tray"),)))
    assert result["outcome"] == "safe_stop"
    assert session.transport.command_log == []
    assert result["trace"][-1]["payload"]["reason"] == "invalid_object_position"


def test_session_recovers_only_from_known_precontact_ik_rejection() -> None:
    session, _ = build_session({"cup": (.3, .1, .8)}, {"tray": (-.25, .2, .8)})
    original_send = session.transport.send_command
    attempts = []
    def reject_first(command):
        if command.command_type == "pick":
            attempts.append(command)
            if len(attempts) == 1:
                return {"accepted": False, "execution_started": False,
                        "reason": "motion_plan_failed", "detail": "right-arm IK failed"}
        return original_send(command)
    session.transport.send_command = reject_first
    result = session.run(TaskSpec("retry", "move cup", (Subgoal("cup", "tray"),)))
    assert result["outcome"] == "complete" and len(attempts) == 2
    assert attempts[-1].parameters["approach_height_delta_m"] == pytest.approx(.03)
    assert any(e["event"] == "precontact_recovery" for e in result["trace"])


def test_session_recovery_budget_and_all_levels_are_audited() -> None:
    session, runtime = build_session({"cup": (.3, .1, .8)}, {"tray": (-.25, .2, .8)})
    sent = []
    def reject_all(command):
        sent.append(command)
        return {"accepted": False, "execution_started": False,
                "reason": "motion_plan_failed", "detail": "right-arm IK failed"}
    session.transport.send_command = reject_all
    result = session.run(TaskSpec("reject", "move cup", (Subgoal("cup", "tray"),)))
    assert result["outcome"] == "safe_stop" and len(sent) == 4
    levels = [e["payload"]["reflection"]["level"] for e in result["trace"] if e["event"] == "precontact_recovery"]
    assert levels == [1, 2, 3] and runtime.pipeline.memory.to_dict() == {}


def test_session_requires_release_feedback_when_requested() -> None:
    session, runtime = build_session({"cup": (.3, .1, .8)}, {"tray": (-.25, .2, .8)})
    session.require_release_confirmation = True
    result = session.run(TaskSpec("release", "move cup", (Subgoal("cup", "tray"),)))
    assert result["outcome"] == "safe_stop" and runtime.pipeline.memory.to_dict() == {}


def test_session_detects_final_target_drift() -> None:
    session, runtime = build_session({"cup": (.3, .1, .8)}, {"tray": (-.25, .2, .8)})
    original_observe = session.transport.get_observation
    count = 0
    def drift():
        nonlocal count
        count += 1
        observation = original_observe()
        if count == 3:
            objects = tuple(replace(obj, center_xyz_m=(-.21, .2, .8)) if obj.name == "cup" else obj for obj in observation.objects)
            return replace(observation, objects=objects)
        return observation
    session.transport.get_observation = drift
    result = session.run(TaskSpec("drift", "move cup", (Subgoal("cup", "tray"),)))
    assert result["outcome"] == "safe_stop" and runtime.pipeline.memory.to_dict() == {}
