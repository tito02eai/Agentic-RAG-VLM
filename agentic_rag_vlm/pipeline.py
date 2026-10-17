"""ReAct-style orchestration with injectable perception and execution adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Protocol

from scripts.agentic_framework import (
    PublicObject,
    build_scene_graph,
    graph_constraint,
    retrieve_strategy,
)

from .memory import EpisodicMemory
from .quality import QualityFactors, QualityResult, evaluate_quality
from .recovery import classify_failure, recover


class Executor(Protocol):
    def __call__(self, action: "GraspAction") -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class PipelineConfig:
    max_retries: int = 3
    rag_enabled: bool = True
    scene_graph_enabled: bool = True
    memory_enabled: bool = True
    recovery_enabled: bool = True


@dataclass(frozen=True)
class GraspAction:
    target: str
    position_xyz_m: tuple[float, float, float]
    strategy: dict[str, Any]
    constraint: dict[str, Any]
    source: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class AgenticPipeline:
    def __init__(self, cards: list[dict[str, object]], config: PipelineConfig | None = None) -> None:
        self.cards = cards
        self.config = config or PipelineConfig()
        self.memory = EpisodicMemory()

    def plan(self, target: PublicObject, objects: Mapping[str, PublicObject]) -> tuple[GraspAction, dict[str, object]]:
        retrieval = retrieve_strategy(target, self.cards, enabled=self.config.rag_enabled)
        graph = build_scene_graph(objects)
        constraint = graph_constraint(target, objects, graph, enabled=self.config.scene_graph_enabled)
        strategy = dict(retrieval.strategy)
        source = "haa_rag"
        memory = self.memory.get(target.category) if self.config.memory_enabled else None
        if memory is not None:
            strategy.update(memory.strategy)
            source = "episodic_memory"
        strategy["force_n"] = float(strategy.get("force_n", 10.0)) * float(constraint["force_scale"])
        strategy["approach_height_delta_m"] = constraint["approach_height_delta_m"]
        xyz = list(target.center_xyz_m)
        xyz[0] += float(constraint["approach_offset_xy_m"][0])
        xyz[1] += float(constraint["approach_offset_xy_m"][1])
        xyz[2] += float(constraint["approach_height_delta_m"])
        action = GraspAction(target.name, tuple(xyz), strategy, constraint, source)
        evidence = {"retrieval": retrieval.to_dict(), "scene_graph": graph, "constraint": constraint}
        return action, evidence

    def run(self, episode_id: str, target: PublicObject, objects: Mapping[str, PublicObject], executor: Executor) -> dict[str, object]:
        action, evidence = self.plan(target, objects)
        trace = []
        for attempt in range(self.config.max_retries + 1):
            observation = dict(executor(action))
            factors = QualityFactors(**observation["quality_factors"])
            quality: QualityResult = evaluate_quality(factors)
            event = {"attempt": attempt, "thought": {"source": action.source}, "action": action.to_dict(), "observation": observation, "quality": asdict(quality)}
            trace.append(event)
            if quality.success:
                if self.config.memory_enabled:
                    self.memory.put(target.category, action.strategy, quality.score, episode_id)
                return {"success": True, "attempts": attempt + 1, "trace": trace, "evidence": evidence}
            if not self.config.recovery_enabled or attempt >= self.config.max_retries:
                break
            failure = classify_failure({"failure_type": observation.get("failure_type"), "below_threshold": quality.below_threshold})
            decision = recover(action.strategy, failure, attempt + 1)
            event["reflection"] = decision.to_dict()
            if decision.safe_stop:
                break
            action = GraspAction(action.target, action.position_xyz_m, decision.corrected_strategy, action.constraint, f"recovery_l{decision.level}")
        return {"success": False, "attempts": len(trace), "trace": trace, "evidence": evidence}
