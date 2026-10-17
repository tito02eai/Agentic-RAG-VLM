from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_qwen_vlm_challenge import (  # noqa: E402
    apply_agent_tools,
    build_prompt,
    canonicalize_planner_payload,
    evaluate,
    public_consistency_errors,
    schema_errors,
)
from agentic_rag_vlm.qwen_adapter import VLMResult  # noqa: E402
from scripts.agentic_framework import PublicObject, load_cards  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "guanghua_challenge_v2.json"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def scene_spec() -> dict:
    return {
        "variants": {
            "red_cube": {"expected_grasp": "side", "expected_card": "cube_flat_edge_side"},
            "blue_cylinder": {"expected_grasp": "pinch", "expected_card": "cylinder_narrow_pinch"},
        }
    }


def executive_payload(*, changed: bool, replan: bool) -> dict:
    return {
        "requested_tools": ["role_aware_change_and_memory_verifier_v1"],
        "changed_objects": ["blue_cylinder"] if changed else [],
        "stale_pending_targets": ["blue_cylinder"] if changed else [],
        "replan": replan,
        "completed_memory": ["red_cube"],
        "next_target": "blue_cylinder",
    }


def test_v2_has_negative_controls_and_combined_scene() -> None:
    protocol = config()
    assert set(protocol["change_events"]) == {"target_move", "no_change", "irrelevant_fragile_move"}
    assert protocol["conditions"]["G4"] == ["C2_full", "C1_local", "B0_skill_only"]
    assert len(protocol["seeds"]["main"]) == 10


def test_g3_target_change_requires_detection_replan_and_memory() -> None:
    metrics = evaluate(
        "G3",
        "C2_full",
        executive_payload(changed=True, replan=True),
        scene_spec(),
        None,
        {"event": "target_move"},
        config(),
    )
    assert metrics["mechanism_success"] == 1.0
    assert metrics["false_replan"] == 0.0


def test_g3_negative_control_penalizes_false_replan() -> None:
    metrics = evaluate(
        "G3",
        "C2_full",
        executive_payload(changed=False, replan=True),
        scene_spec(),
        None,
        {"event": "irrelevant_fragile_move"},
        config(),
    )
    assert metrics["mechanism_success"] == 0.0
    assert metrics["false_replan"] == 1.0


def test_no_replan_condition_cannot_claim_executable_recovery() -> None:
    metrics = evaluate(
        "G3",
        "A_no_replan",
        executive_payload(changed=True, replan=True),
        scene_spec(),
        None,
        {"event": "target_move"},
        config(),
    )
    assert metrics["replan_decision_correct"] == 0.0
    assert metrics["mechanism_success"] == 0.0


def test_g2_accepts_computed_away_vector_and_rejects_direct_path() -> None:
    expected = {"hazard_expected": True, "away_unit_xy": [-1.0, 0.0]}
    safe = {
        "requested_tools": ["protected_relation_to_motion_constraint_v1"],
        "hazard_present": True,
        "approach_direction": "away_from_fragile",
        "approach_offset_xy_m": [-0.04, 0.0],
        "height_delta_mm": 35,
        "force_scale": 0.75,
    }
    unsafe = dict(safe, approach_offset_xy_m=[0.0, 0.0])
    assert evaluate("G2", "C2_full", safe, scene_spec(), expected, None, config())["mechanism_success"] == 1.0
    assert evaluate("G2", "C2_full", unsafe, scene_spec(), expected, None, config())["mechanism_success"] == 0.0


def test_g3_prompt_has_no_answer_revealing_directive() -> None:
    prompt = build_prompt(
        "G3",
        "C2_full",
        config(),
        {},
        {},
        [],
        {},
        {"measurement_threshold_m": 0.015, "per_object_displacement": {}},
        303,
    )
    lowered = prompt.lower()
    assert "must be blue_cylinder" not in lowered
    assert "replan must be true" not in lowered
    assert '"replan":null' in lowered


def test_schema_gate_rejects_null_next_target() -> None:
    payload = executive_payload(changed=True, replan=True)
    payload["next_target"] = None
    assert "next_target is missing or illegal" in schema_errors("G3", payload)


def test_public_gate_catches_ignored_pending_target_displacement() -> None:
    monitor = {
        "measurement_threshold_m": 0.015,
        "per_object_displacement": {
            "red_cube": {"distance_m": 0.16},
            "blue_cylinder": {"distance_m": 0.06},
            "fragile_proxy": {"distance_m": 0.0},
        },
    }
    payload = {
        "requested_tools": ["role_aware_change_and_memory_verifier_v1"],
        "changed_objects": ["red_cube"],
        "stale_pending_targets": [],
        "replan": False,
        "completed_memory": ["red_cube"],
        "next_target": "blue_cylinder",
    }
    errors = public_consistency_errors("G3", "C2_full", payload, monitor)
    assert any("changed_objects" in error for error in errors)
    assert any("stale_pending_targets" in error for error in errors)
    assert any("replan" in error for error in errors)


def test_full_agent_safety_tool_converts_relation_to_executable_offset() -> None:
    proposal = {
        "requested_tools": ["protected_relation_to_motion_constraint_v1"],
        "hazard_present": True,
        "approach_direction": "away_from_fragile",
        "approach_offset_xy_m": [0.0, 0.0],
        "height_delta_mm": 10,
        "force_scale": 1.0,
    }
    decision, receipts = apply_agent_tools(
        "G2",
        "C2_full",
        proposal,
        {"hazard_expected": True, "public_distance_m": 0.08, "away_unit_xy": [-1.0, 0.0]},
        None,
        config(),
    )
    assert decision is not None
    assert decision["approach_offset_xy_m"][0] <= -0.04
    assert decision["height_delta_mm"] >= 30
    assert decision["force_scale"] <= 0.8
    assert receipts[0]["uses_privileged_simulator_state"] is False


def test_full_agent_memory_tool_rejects_irrelevant_change_replan() -> None:
    monitor = {
        "measurement_threshold_m": 0.015,
        "per_object_displacement": {
            "red_cube": {"distance_m": 0.0},
            "blue_cylinder": {"distance_m": 0.0},
            "fragile_proxy": {"distance_m": 0.05},
        },
    }
    decision, receipts = apply_agent_tools(
        "G3",
        "C2_full",
        executive_payload(changed=True, replan=True),
        None,
        monitor,
        config(),
    )
    assert decision is not None
    assert decision["changed_objects"] == ["fragile_proxy"]
    assert decision["stale_pending_targets"] == []
    assert decision["replan"] is False
    assert receipts[0]["tool"] == "role_aware_change_and_memory_verifier_v1"


def test_ablation_does_not_receive_full_agent_tools() -> None:
    proposal = {"requested_tools": ["protected_relation_to_motion_constraint_v1"], "hazard_present": True}
    decision, receipts = apply_agent_tools(
        "G2",
        "A_no_graph",
        proposal,
        {"hazard_expected": True, "public_distance_m": 0.08, "away_unit_xy": [-1.0, 0.0]},
        None,
        config(),
    )
    assert decision == proposal
    assert receipts == []


def test_full_agent_routes_retrieved_card_to_executed_hand_skill() -> None:
    objects = {
        "red_cube": PublicObject(
            "red_cube", (0.0, 0.0, 0.0), "cube", "rigid_body", "graspable_body",
            "rigid", 0.1, "body", 0.4,
        ),
        "blue_cylinder": PublicObject(
            "blue_cylinder", (0.0, 0.0, 0.0), "cylinder", "rigid_body", "pinchable",
            "rigid", 0.2, "body", 2.1,
        ),
    }
    proposal = {
        "requested_tools": ["retrieval_to_dexterous_skill_router_v1"],
        "decisions": {
            "red_cube": {"grasp": "power", "card_id": "wrong"},
            "blue_cylinder": {"grasp": "power", "card_id": "wrong"},
        }
    }
    decision, receipts = apply_agent_tools(
        "G1", "C2_full", proposal, None, None, config(), objects,
        load_cards(PROJECT_ROOT / "knowledge_base" / "affordance_cards.json"),
    )
    assert decision is not None
    assert decision["decisions"]["red_cube"]["grasp"] == "side"
    assert decision["decisions"]["blue_cylinder"]["grasp"] == "pinch"
    assert receipts[0]["tool"] == "retrieval_to_dexterous_skill_router_v1"


def test_full_agent_does_not_route_from_scene_label_without_model_request() -> None:
    proposal = {
        "requested_tools": [],
        "hazard_present": True,
        "approach_direction": "direct",
        "approach_offset_xy_m": [0.0, 0.0],
        "height_delta_mm": 0,
        "force_scale": 1.0,
    }
    decision, receipts = apply_agent_tools(
        "G4",
        "C2_full",
        proposal,
        {"hazard_expected": True, "public_distance_m": 0.08, "away_unit_xy": [-1.0, 0.0]},
        {"measurement_threshold_m": 0.015, "per_object_displacement": {}},
        config(),
    )
    assert decision == proposal
    assert receipts == []


def test_unknown_tool_is_rejected_without_modifying_proposal() -> None:
    proposal = {"requested_tools": ["private_oracle_tool"], "reason": "proposal"}
    decision, receipts = apply_agent_tools("G1", "C2_full", proposal, None, None, config())
    assert decision == proposal
    assert receipts[0]["status"] == "rejected"
    assert receipts[0]["reason"] == "unknown_tool"
    assert receipts[0]["uses_privileged_simulator_state"] is False


def test_requested_tool_with_missing_public_input_is_rejected() -> None:
    proposal = {
        "requested_tools": ["protected_relation_to_motion_constraint_v1"],
        "hazard_present": False,
        "approach_direction": "direct",
        "approach_offset_xy_m": [0.0, 0.0],
        "height_delta_mm": 0,
        "force_scale": 1.0,
    }
    decision, receipts = apply_agent_tools("G2", "C2_full", proposal, None, None, config())
    assert decision == proposal
    assert receipts[0]["status"] == "rejected"
    assert receipts[0]["reason"] == "missing_public_protected_relation"


def test_tool_selection_is_an_explicit_full_agent_success_gate() -> None:
    safe = {
        "requested_tools": ["protected_relation_to_motion_constraint_v1"],
        "hazard_present": True,
        "approach_direction": "away_from_fragile",
        "approach_offset_xy_m": [-0.04, 0.0],
        "height_delta_mm": 40,
        "force_scale": 0.7,
    }
    expected = {"hazard_expected": True, "away_unit_xy": [-1.0, 0.0]}
    correct = evaluate(
        "G2", "C2_full", safe, scene_spec(), expected, None, config(),
        [{"tool": "protected_relation_to_motion_constraint_v1", "status": "executed"}],
    )
    missing = evaluate("G2", "C2_full", safe, scene_spec(), expected, None, config(), [])
    assert correct["tool_selection_correct"] == 1.0
    assert correct["mechanism_success"] == 1.0
    assert missing["tool_selection_correct"] == 0.0
    assert missing["mechanism_success"] == 0.0


def test_canonicalizer_wraps_root_target_decisions_without_changing_values() -> None:
    raw = {
        "red_cube": {"grasp": "side", "card_id": "cube_flat_edge_side"},
        "blue_cylinder": {"grasp": "power", "card_id": "cylinder_wide_power"},
        "reason": "kept",
    }
    result = VLMResult(raw, json.dumps(raw), 1.0, "model", "model", {}, None)
    normalized = canonicalize_planner_payload("G1", result)
    assert normalized.parsed == {
        "decisions": {
            "red_cube": raw["red_cube"],
            "blue_cylinder": raw["blue_cylinder"],
        },
        "reason": "kept",
    }
    assert normalized.raw_text == result.raw_text
