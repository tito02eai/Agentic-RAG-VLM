"""Offline regression tests for the platform deployment layer.

Everything here runs without network access, MuJoCo, or the school asset
bundle: the mock transport and a fake VLM client exercise the config,
registry, perception, mapper, verifier, and the full session loop.
"""

from __future__ import annotations

import json
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
    assert config.camera_ids == ("front", "wrist_right")
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"platform": "blupe"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_platform_config(broken)


def test_remote_transport_fails_fast_before_api_contract() -> None:
    transport = RemotePlatformTransport(load_platform_config(PLATFORM_EXAMPLE))
    with pytest.raises(Exception, match="BluPe API contract"):
        transport.health_check()


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


def test_session_safe_stops_when_place_cannot_resolve_destination() -> None:
    session, runtime = build_session({"cup": (0.30, 0.10, 0.80)}, {})
    task = TaskSpec("stage1_bad", "pick the cup", (Subgoal("cup", "missing_zone"),), ())
    result = session.run(task)
    assert result["outcome"] == "safe_stop"
    assert runtime.completed_targets == []
    stop_events = [event for event in result["trace"] if event["event"] == "safe_stop"]
    assert stop_events and stop_events[0]["payload"]["reason"] == "subgoal_verification_failed"


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
