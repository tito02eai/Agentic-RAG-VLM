from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agentic_rag_vlm.pipeline import AgenticPipeline  # noqa: E402
from agentic_rag_vlm.runtime import AgenticRuntime, RuntimeConfig, RuntimePhase, Subgoal, TaskSpec  # noqa: E402
from scripts.agentic_framework import PublicObject, load_cards  # noqa: E402


CARDS = PROJECT_ROOT / "knowledge_base" / "affordance_cards.json"


def obj(name: str, xy: tuple[float, float]) -> PublicObject:
    if name == "red_cube":
        return PublicObject(name, (*xy, 0.834), "cube", "rigid_body", "graspable_body", "rigid", 0.1, "body", 1.0)
    if name == "blue_cylinder":
        return PublicObject(name, (*xy, 0.861), "cylinder", "rigid_body", "pinchable", "rigid", 0.2, "body", 2.1)
    return PublicObject(name, (*xy, 0.876), "cylinder", "container", "fragile", "glass", 1.0, "body", 2.2)


def task() -> TaskSpec:
    return TaskSpec(
        "demo",
        "place red then blue while protecting yellow",
        (Subgoal("red_cube", "red_zone"), Subgoal("blue_cylinder", "blue_zone")),
        ("fragile_proxy",),
    )


def objects() -> dict[str, PublicObject]:
    return {
        "red_cube": obj("red_cube", (-0.18, -0.2)),
        "blue_cylinder": obj("blue_cylinder", (-0.18, -0.03)),
        "fragile_proxy": obj("fragile_proxy", (-0.11, -0.2)),
    }


def test_runtime_completes_two_subgoals_with_memory_preserved() -> None:
    runtime = AgenticRuntime(AgenticPipeline(load_cards(CARDS)))
    scene = objects()
    runtime.start(task())
    runtime.observe(scene)
    runtime.plan(scene)
    runtime.begin_execution(skill="power_pick_place")
    runtime.verify(True, quality=1.0)
    assert runtime.phase == RuntimePhase.MONITOR
    runtime.monitor({"stale_targets": []})
    runtime.observe(scene)
    runtime.plan(scene)
    runtime.begin_execution(skill="pinch_pick_place")
    runtime.verify(True, quality=1.0)
    assert runtime.phase == RuntimePhase.COMPLETE
    assert runtime.completed_targets == ["red_cube", "blue_cylinder"]


def test_runtime_replans_only_pending_target_and_keeps_completed_work() -> None:
    runtime = AgenticRuntime(AgenticPipeline(load_cards(CARDS)), RuntimeConfig(maximum_replans=1))
    scene = objects()
    runtime.start(task())
    runtime.observe(scene)
    runtime.plan(scene)
    runtime.begin_execution(skill="power_pick_place")
    runtime.verify(True, quality=1.0)
    assert runtime.monitor({"stale_targets": ["red_cube", "blue_cylinder"]})
    assert runtime.phase == RuntimePhase.REPLAN
    assert runtime.completed_targets == ["red_cube"]
    runtime.observe(scene)
    runtime.plan(scene)
    assert runtime.current_action is not None and runtime.current_action.target == "blue_cylinder"


def test_runtime_safe_stops_when_replan_is_disabled() -> None:
    runtime = AgenticRuntime(AgenticPipeline(load_cards(CARDS)), RuntimeConfig(replanning_enabled=False))
    scene = objects()
    runtime.start(task())
    runtime.observe(scene)
    runtime.plan(scene)
    runtime.begin_execution(skill="power_pick_place")
    runtime.verify(True, quality=1.0)
    assert not runtime.monitor({"stale_targets": ["blue_cylinder"]})
    assert runtime.phase == RuntimePhase.SAFE_STOP
