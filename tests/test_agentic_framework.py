from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agentic_rag_vlm.memory import EpisodicMemory  # noqa: E402
from agentic_rag_vlm.pipeline import AgenticPipeline, PipelineConfig  # noqa: E402
from agentic_rag_vlm.qwen_adapter import extract_json_object  # noqa: E402
from agentic_rag_vlm.quality import QualityFactors, WEIGHTS, evaluate_quality  # noqa: E402
from agentic_rag_vlm.recovery import FailureType, recover  # noqa: E402
from scripts.agentic_framework import (  # noqa: E402
    PublicObject,
    build_scene_graph,
    detect_scene_change,
    graph_constraint,
    load_cards,
    retrieve_strategy,
)
from scripts.validate_agentic_artifacts import PRIVATE_ONLY_KEYS, walk_keys  # noqa: E402
from scripts.run_qwen_vlm_pilot import evaluate as evaluate_qwen_decision  # noqa: E402


CARDS = PROJECT_ROOT / "knowledge_base" / "affordance_cards.json"


def obj(name: str, xy: tuple[float, float]) -> PublicObject:
    if name == "red_cube":
        return PublicObject(name, (*xy, 0.834), "cube", "rigid_body", "graspable_body", "rigid", 0.1, "body", 1.0)
    if name == "blue_cylinder":
        return PublicObject(name, (*xy, 0.861), "cylinder", "rigid_body", "pinchable", "rigid", 0.2, "body", 2.1)
    return PublicObject(name, (*xy, 0.876), "cylinder", "container", "fragile", "glass", 1.0, "body", 2.2)


def test_haa_rag_selects_affordance_specific_synergies() -> None:
    cards = load_cards(CARDS)
    assert retrieve_strategy(obj("red_cube", (0, 0)), cards, enabled=True).strategy["synergy"] == "power"
    assert retrieve_strategy(obj("blue_cylinder", (0, 0)), cards, enabled=True).strategy["synergy"] == "pinch"
    assert retrieve_strategy(obj("blue_cylinder", (0, 0)), cards, enabled=False).strategy["synergy"] == "power"


def test_scene_graph_generates_offset_away_from_fragile_neighbor() -> None:
    objects = {
        "red_cube": obj("red_cube", (-0.18, -0.20)),
        "fragile_proxy": obj("fragile_proxy", (-0.105, -0.20)),
    }
    graph = build_scene_graph(objects)
    constraint = graph_constraint(objects["red_cube"], objects, graph, enabled=True)
    offset = np.asarray(constraint["approach_offset_xy_m"])
    away = np.asarray([-0.18, -0.20]) - np.asarray([-0.105, -0.20])
    assert constraint["active"]
    assert np.dot(offset, away) > 0
    assert constraint["force_scale"] == 0.8


def test_public_monitor_detects_displacement_without_truth_metadata() -> None:
    before = {"blue_cylinder": obj("blue_cylinder", (-0.18, -0.05))}
    after = {"blue_cylinder": obj("blue_cylinder", (-0.155, -0.035))}
    result = detect_scene_change(before, after)
    assert result["change_detected"]
    assert result["stale_targets"] == ["blue_cylinder"]


def test_quality_memory_and_three_level_recovery_contract() -> None:
    assert np.isclose(sum(WEIGHTS.values()), 1.0)
    quality = evaluate_quality(QualityFactors(1, 1, 1, 1, 1, 1, 1))
    assert quality.success and np.isclose(quality.score, 1.0)
    assert len(FailureType) == 14
    strategy = {"synergy": "power", "force_n": 10.0, "aperture_m": 0.06}
    assert recover(strategy, FailureType.SLIP, 1).corrected_strategy["force_n"] == 12.0
    assert recover(strategy, FailureType.SLIP, 2).level == 2
    assert recover(strategy, FailureType.SLIP, 3).action == "full_replan"
    memory = EpisodicMemory()
    memory.put("cube", strategy, 0.8, "episode-1")
    memory.put("cube", {"synergy": "pinch"}, 0.7, "episode-2")
    assert memory.get("cube").strategy["synergy"] == "power"


def test_react_pipeline_stores_success_in_memory() -> None:
    pipeline = AgenticPipeline(load_cards(CARDS), PipelineConfig(max_retries=1))
    target = obj("red_cube", (-0.18, -0.23))

    def executor(_action):
        return {"quality_factors": {name: 1.0 for name in WEIGHTS}}

    result = pipeline.run("episode-1", target, {"red_cube": target}, executor)
    assert result["success"]
    assert pipeline.memory.get("cube") is not None


def test_public_trace_contract_rejects_private_evaluator_fields() -> None:
    public_event = {
        "event": "public_monitor",
        "uses_privileged_simulator_state": False,
        "observation": {"stale_targets": ["blue_cylinder"]},
    }
    assert not (walk_keys(public_event) & PRIVATE_ONLY_KEYS)
    leaked_event = json.loads(json.dumps(public_event))
    leaked_event["perturbation_receipt"] = {"object": "blue_cylinder"}
    assert "perturbation_receipt" in (walk_keys(leaked_event) & PRIVATE_ONLY_KEYS)


def test_qwen_json_parser_accepts_plain_and_fenced_outputs() -> None:
    expected = {"replan": True, "next_target": "blue_cylinder"}
    assert extract_json_object(json.dumps(expected)) == expected
    assert extract_json_object(f"```json\n{json.dumps(expected)}\n```") == expected


def test_qwen_graph_metric_requires_an_executable_away_offset() -> None:
    expected = {"approach_offset_xy_m": [-0.045, 0.02]}
    decision = {
        "hazard_present": True,
        "approach_direction": "away_from_fragile",
        "approach_offset_xy_m": [-0.045, 0.02],
        "height_delta_mm": 30,
        "force_scale": 0.8,
    }
    assert evaluate_qwen_decision("G2", decision, expected)["mechanism_success"] == 1.0
    decision["approach_offset_xy_m"] = [0.0, 0.0]
    assert evaluate_qwen_decision("G2", decision, expected)["mechanism_success"] == 0.0
