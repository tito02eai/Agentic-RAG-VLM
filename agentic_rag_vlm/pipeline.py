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
from .scene import build_scene_graph

from .memory import EpisodicMemory
from .quality import QualityFactors, QualityResult, evaluate_quality
from .recovery import classify_failure, recover
from .constraints import compile_constraints


class Executor(Protocol):
    def __call__(self, action: "GraspAction") -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class PipelineConfig:
    max_retries: int = 3
    rag_enabled: bool = True
    scene_graph_enabled: bool = True
    memory_enabled: bool = True
    recovery_enabled: bool = True

    def __post_init__(self):
        if not 0 <= self.max_retries <= 3:
            raise ValueError("paper retry budget is between zero and three")


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
    def __init__(self, cards: list[dict[str, object]], config: PipelineConfig | None = None,
                 *, retriever=None, reasoner=None, seed: int = 0, path_validator=None) -> None:
        self.cards = cards
        self.config = config or PipelineConfig()
        self.memory = EpisodicMemory()
        self.scene_context: Mapping[str, Any] = {}
        self.retriever = retriever
        self.reasoner = reasoner
        self.path_validator = path_validator
        self.visual_features: dict[str, Any] = {}
        self.embedding_space = None
        import numpy as np
        self.rng = np.random.default_rng(seed)

    def plan(self, target: PublicObject, objects: Mapping[str, PublicObject], *, use_memory=True) -> tuple[GraspAction, dict[str, object]]:
        if self.retriever is None:
            retrieval = retrieve_strategy(target, self.cards, enabled=self.config.rag_enabled)
        else:
            retrieval = self.retriever.retrieve(target,self.visual_features.get(target.name),
                embedding_space=self.embedding_space,enabled=self.config.rag_enabled)
        graph = build_scene_graph(objects,self.scene_context)
        if self.reasoner is not None:
            self.reasoner.negative_experiences = retrieval.to_dict().get("stages",{}).get("negative_examples",[])[:3]
        constraint = compile_constraints(target, objects, self.scene_context, enabled=self.config.scene_graph_enabled,
            path_validator=self.path_validator)
        if constraint["blocked"]:
            raise ValueError(f"scene constraint blocks execution: {constraint}")
        strategy = dict(retrieval.strategy)
        source = getattr(self.retriever,'action_source','haa_rag')
        memory = self.memory.get(target.category) if self.config.memory_enabled and use_memory else None
        domain = self.retriever.store.domain if self.retriever is not None else "legacy"
        same_semantics = memory is not None and memory.strategy.get("_semantic_signature") == self._signature(target)
        compatible = same_semantics and memory.strategy.get("_execution_domain","legacy") == domain
        if compatible:
            strategy.update(memory.strategy)
            source = "episodic_memory"
        # Cache intrinsic grasp parameters; apply transient scene constraints once.
        strategy["_semantic_signature"] = self._signature(target)
        strategy["_execution_domain"] = domain
        strategy["base_force_n"] = float(strategy.get("base_force_n", strategy.get("force_n", 10.0)))
        strategy["force_n"] = strategy["base_force_n"] * float(constraint["force_scale"])
        strategy["approach_height_delta_m"] = float(strategy.get("intrinsic_approach_height_m", 0)) + float(constraint["approach_height_delta_m"])
        if "required_approach" in constraint:
            strategy["approach"] = constraint["required_approach"]
        if strategy.get("synergy") == "do_not_grasp":
            raise ValueError("retrieved strategy prohibits grasping this target")
        strategy["velocity_scale"] = constraint.get("velocity_scale", 1.0)
        xyz = list(target.center_xyz_m)
        # A biased approach must still terminate at the object. Moving the
        # final grasp centre away from a neighbor produces an empty grasp.
        action = GraspAction(target.name, tuple(xyz), strategy, constraint, source)
        evidence = {"retrieval": retrieval.to_dict(), "scene_graph": graph, "constraint": constraint,
            "memory":{"enabled":self.config.memory_enabled,"hit":compatible,
                      "source_episode":memory.source_episode if compatible else None,
                      "rejected_reason":("semantic_mismatch" if not same_semantics else "execution_domain_mismatch") if memory is not None and not compatible else None}}
        return action, evidence

    @staticmethod
    def _signature(target: PublicObject) -> list[object]:
        return [target.category, target.affordance_type, target.material, target.fragility, target.region]

    def remember(self, category: str, action: GraspAction, quality: float, episode_id: str) -> None:
        strategy = dict(action.strategy)
        scale = float(action.constraint.get("force_scale", 1))
        strategy["base_force_n"] = float(strategy.get("force_n", 10)) / scale
        strategy["intrinsic_approach_height_m"] = (
            float(strategy.get("approach_height_delta_m", 0))
            - float(action.constraint.get("approach_height_delta_m", 0))
        )
        self.memory.put(category, strategy, quality, episode_id)

    def invalidate_failed_memory(self, action: GraspAction, *, reason: str):
        """Evict disproven reuse even when retry is disabled or exhausted."""
        if action.source != 'episodic_memory':
            return None
        signature = action.strategy.get('_semantic_signature', [])
        if not signature:
            return None
        old = self.memory.invalidate(signature[0])
        if old is None:
            return None
        return {'category': old.category, 'source_episode': old.source_episode,
                'failed_force_n': action.strategy.get('force_n'), 'reason': reason}

    def run(self, episode_id: str, target: PublicObject, objects: Mapping[str, PublicObject], executor: Executor,
            *, observer: Callable[[], Mapping[str, PublicObject]] | None = None) -> dict[str, object]:
        action, evidence = self.plan(target, objects)
        trace = []
        for attempt in range(self.config.max_retries + 1):
            thought = {"source":action.source}
            if self.reasoner is not None:
                thought = self.reasoner.think(action,evidence,trace,attempt)
            observation = dict(executor(action))
            factors = QualityFactors(**observation["quality_factors"])
            quality: QualityResult = evaluate_quality(factors,observed_success=observation.get("success"))
            event = {"attempt": attempt, "thought": thought, "action": action.to_dict(), "observation": observation, "quality": asdict(quality)}
            trace.append(event)
            if quality.success:
                if self.config.memory_enabled:
                    self.remember(target.category, action, quality.score, episode_id)
                return {"success": True, "attempts": attempt + 1, "trace": trace, "evidence": evidence}
            if action.source == "episodic_memory":
                event["memory_invalidation"] = self.invalidate_failed_memory(action, reason='observed_execution_failure')
                event["memory_invalidated"] = target.category
            if not self.config.recovery_enabled or attempt >= self.config.max_retries:
                break
            failure = classify_failure({**observation,"below_threshold":quality.below_threshold})
            decision = recover(action.strategy, failure, attempt + 1,rng=self.rng)
            event["reflection"] = decision.to_dict()
            if self.reasoner is not None:
                event["vlm_reflection"] = self.reasoner.reflect(action,decision.to_dict(),trace)
            if decision.safe_stop:
                break
            if decision.level == 3 or decision.corrected_strategy.get("position_retry"):
                if observer is None:
                    event["recovery_stop"] = "fresh_observation_required"
                    break
                fresh_observation = observer()
                from .observation import PlanningObservation
                if isinstance(fresh_observation,PlanningObservation):
                    if not fresh_observation.observation_id:
                        raise ValueError("fresh replan observation must have an identity")
                    objects = fresh_observation.objects
                    self.visual_features = dict(fresh_observation.visual_embeddings)
                    self.embedding_space = fresh_observation.embedding_space
                    self.scene_context = fresh_observation.scene_context
                    if self.reasoner is not None:
                        self.reasoner.images = list(fresh_observation.images)
                        self.reasoner.target_images = {}
                elif self.retriever is not None:
                    event["recovery_stop"] = "fresh_visual_observation_required"
                    break
                else:
                    objects = fresh_observation
                if target.name not in objects:
                    event["recovery_stop"] = "target_not_visible"
                    break
                target = objects[target.name]
                fresh, fresh_evidence = self.plan(target, objects,use_memory=decision.level != 3)
                event["replan_evidence"] = fresh_evidence
                # Reset to the newly retrieved parameters on L3. No unchecked
                # perturbation is injected into a physical command.
                strategy = fresh.strategy if decision.level == 3 else decision.corrected_strategy
                position = fresh.position_xyz_m
                if decision.level == 3:
                    import numpy as np
                    position = tuple(np.asarray(position)+decision.corrected_strategy["position_perturbation_m"])
                    strategy = {**strategy,"full_replan":True,"position_perturbation_m":decision.corrected_strategy["position_perturbation_m"]}
                action = GraspAction(target.name, position, strategy, fresh.constraint, f"recovery_l{decision.level}")
            else:
                action = GraspAction(action.target, action.position_xyz_m, decision.corrected_strategy, action.constraint, f"recovery_l{decision.level}")
        return {"success": False, "attempts": len(trace), "trace": trace, "evidence": evidence}
