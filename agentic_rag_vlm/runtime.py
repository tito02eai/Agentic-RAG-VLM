"""Multi-subgoal Agentic runtime shared by Guanghua simulation and future robots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping

from scripts.agentic_framework import PublicObject

from .pipeline import AgenticPipeline, GraspAction


class RuntimePhase(str, Enum):
    IDLE = "idle"
    OBSERVE = "observe"
    PLAN = "plan"
    EXECUTE = "execute"
    VERIFY = "verify"
    MONITOR = "monitor"
    REPLAN = "replan"
    COMPLETE = "complete"
    SAFE_STOP = "safe_stop"


@dataclass(frozen=True)
class Subgoal:
    target: str
    destination: str


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    instruction: str
    subgoals: tuple[Subgoal, ...]
    protected_objects: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeConfig:
    memory_enabled: bool = True
    replanning_enabled: bool = True
    maximum_replans: int = 1


@dataclass(frozen=True)
class RuntimeEvent:
    sequence: int
    event: str
    phase: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgenticRuntime:
    """Auditable lifecycle around the paper's single-target AgenticPipeline."""

    def __init__(self, pipeline: AgenticPipeline, config: RuntimeConfig | None = None) -> None:
        self.pipeline = pipeline
        self.config = config or RuntimeConfig()
        self.phase = RuntimePhase.IDLE
        self.task: TaskSpec | None = None
        self.completed_targets: list[str] = []
        self.current_action: GraspAction | None = None
        self._current_category: str | None = None
        self.replan_count = 0
        self._events: list[RuntimeEvent] = []

    @property
    def current_subgoal(self) -> Subgoal | None:
        if self.task is None:
            return None
        for subgoal in self.task.subgoals:
            if subgoal.target not in self.completed_targets:
                return subgoal
        return None

    @property
    def pending_targets(self) -> list[str]:
        if self.task is None:
            return []
        return [item.target for item in self.task.subgoals if item.target not in self.completed_targets]

    def _record(self, event: str, **payload: Any) -> RuntimeEvent:
        item = RuntimeEvent(len(self._events), event, self.phase.value, dict(payload))
        self._events.append(item)
        return item

    def start(self, task: TaskSpec) -> None:
        if self.phase not in {RuntimePhase.IDLE, RuntimePhase.COMPLETE, RuntimePhase.SAFE_STOP}:
            raise RuntimeError(f"cannot start from {self.phase.value}")
        self.task = task
        self.completed_targets = []
        self.current_action = None
        self._current_category = None
        self.replan_count = 0
        self.phase = RuntimePhase.OBSERVE
        self._events = []
        self._record("task_started", task=asdict(task), pending_targets=self.pending_targets)

    def observe(self, objects: Mapping[str, PublicObject], *, receipt: Mapping[str, Any] | None = None) -> None:
        if self.phase not in {RuntimePhase.OBSERVE, RuntimePhase.REPLAN, RuntimePhase.MONITOR}:
            raise RuntimeError(f"cannot observe from {self.phase.value}")
        visible = sorted(objects)
        missing = [target for target in self.pending_targets if target not in objects]
        self._record("public_observation", visible_objects=visible, missing_pending_targets=missing, receipt=dict(receipt or {}))
        if missing:
            self.safe_stop("pending_target_not_visible", missing_targets=missing)
            return
        self.phase = RuntimePhase.PLAN

    def plan(self, objects: Mapping[str, PublicObject], *, planner_receipt: Mapping[str, Any] | None = None) -> GraspAction:
        if self.phase != RuntimePhase.PLAN:
            raise RuntimeError(f"cannot plan from {self.phase.value}")
        subgoal = self.current_subgoal
        if subgoal is None:
            self.phase = RuntimePhase.COMPLETE
            self._record("task_completed", completed_targets=list(self.completed_targets))
            raise RuntimeError("task already complete")
        action, evidence = self.pipeline.plan(objects[subgoal.target], objects)
        self.current_action = action
        self._current_category = objects[subgoal.target].category
        self._record(
            "subgoal_planned",
            target=subgoal.target,
            destination=subgoal.destination,
            action=action.to_dict(),
            evidence=evidence,
            planner_receipt=dict(planner_receipt or {}),
        )
        self.phase = RuntimePhase.EXECUTE
        return action

    def begin_execution(self, *, skill: str, executor_receipt: Mapping[str, Any] | None = None) -> None:
        if self.phase != RuntimePhase.EXECUTE or self.current_action is None:
            raise RuntimeError(f"cannot execute from {self.phase.value}")
        self._record("skill_started", target=self.current_action.target, skill=skill, receipt=dict(executor_receipt or {}))
        self.phase = RuntimePhase.VERIFY

    def verify(self, success: bool, *, quality: float, evidence: Mapping[str, Any] | None = None) -> None:
        if self.phase != RuntimePhase.VERIFY or self.current_action is None:
            raise RuntimeError(f"cannot verify from {self.phase.value}")
        target = self.current_action.target
        self._record("subgoal_verified", target=target, success=bool(success), quality=float(quality), evidence=dict(evidence or {}))
        if not success:
            self.safe_stop("subgoal_verification_failed", target=target)
            return
        if target not in self.completed_targets:
            self.completed_targets.append(target)
        if self.config.memory_enabled:
            self.pipeline.memory.put(
                self._current_category or target,
                self.current_action.strategy,
                float(quality),
                self.task.task_id if self.task else "unknown",
            )
        self.current_action = None
        self._current_category = None
        if self.current_subgoal is None:
            self.phase = RuntimePhase.COMPLETE
            self._record("task_completed", completed_targets=list(self.completed_targets))
        else:
            self.phase = RuntimePhase.MONITOR

    def monitor(self, change: Mapping[str, Any]) -> bool:
        if self.phase != RuntimePhase.MONITOR:
            raise RuntimeError(f"cannot monitor from {self.phase.value}")
        stale = sorted(set(change.get("stale_targets", [])) & set(self.pending_targets))
        needs_replan = bool(stale)
        admitted = needs_replan and self.config.replanning_enabled and self.replan_count < self.config.maximum_replans
        self._record(
            "change_assessed",
            change=dict(change),
            pending_targets=self.pending_targets,
            stale_pending_targets=stale,
            replan_requested=needs_replan,
            replan_admitted=admitted,
        )
        if needs_replan and not admitted:
            self.safe_stop("replan_unavailable", stale_pending_targets=stale)
            return False
        if admitted:
            self.replan_count += 1
            self.phase = RuntimePhase.REPLAN
            self._record("l3_replan_started", count=self.replan_count, preserved_completed_targets=list(self.completed_targets))
            return True
        self.phase = RuntimePhase.OBSERVE
        return False

    def safe_stop(self, reason: str, **payload: Any) -> None:
        self.phase = RuntimePhase.SAFE_STOP
        self._record("safe_stop", reason=reason, **payload)

    def trace(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self._events]

    def summary(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "task_id": self.task.task_id if self.task else None,
            "completed_targets": list(self.completed_targets),
            "pending_targets": self.pending_targets,
            "replan_count": self.replan_count,
            "events": len(self._events),
        }
