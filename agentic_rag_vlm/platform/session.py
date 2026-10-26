"""End-to-end platform session driver around ``AgenticRuntime``.

``BluPeSession`` wires the platform-agnostic runtime state machine
(observe → plan → execute → verify → monitor → replan) to a transport,
a perception path, a command mapper, and a verifier.  Every transport,
perception, and verification outcome is appended to an audit receipt list,
mirroring the public/private evidence discipline of the frozen experiments.

Scope: the session drives platform-provided motion primitives through
``PlatformCommand`` translation.  It does not perform end-to-end VLA
control, and verification failures stop the session conservatively
(the pipeline's in-subgoal L1/L2 recovery loop is a separate,
single-target execution path).
"""

from __future__ import annotations

from typing import Any, Mapping

from scripts.agentic_framework import PublicObject, detect_scene_change

from ..runtime import AgenticRuntime, RuntimePhase, TaskSpec
from .commands import GraspActionMapper
from .perception import RegistryMetadataPerception, VLMObjectPerception
from .transport import PlatformTransport
from .verifier import SceneChangeVerifier


class BluPeSession:
    def __init__(
        self,
        runtime: AgenticRuntime,
        transport: PlatformTransport,
        perception: RegistryMetadataPerception | VLMObjectPerception,
        mapper: GraspActionMapper,
        verifier: SceneChangeVerifier,
    ) -> None:
        self.runtime = runtime
        self.transport = transport
        self.perception = perception
        self.mapper = mapper
        self.verifier = verifier
        self.receipts: list[dict[str, Any]] = []

    def _record(self, kind: str, payload: Mapping[str, Any]) -> None:
        self.receipts.append(
            {
                "sequence": len(self.receipts),
                "phase": self.runtime.phase.value,
                "kind": kind,
                "payload": dict(payload),
            }
        )

    def _perceive(self) -> dict[str, PublicObject]:
        observation = self.transport.get_observation()
        self._record("observation", observation.to_dict())
        objects, receipt = self.perception.perceive(observation)
        self._record("perception", receipt.to_dict())
        return objects

    def run(self, task: TaskSpec) -> dict[str, Any]:
        self.receipts = []
        self.runtime.start(task)
        objects: dict[str, PublicObject] = {}
        objects_before_execution: dict[str, PublicObject] = {}
        place_command_id: str | None = None
        try:
            while True:
                phase = self.runtime.phase
                if phase in {RuntimePhase.COMPLETE, RuntimePhase.SAFE_STOP, RuntimePhase.IDLE}:
                    break
                if phase in {RuntimePhase.OBSERVE, RuntimePhase.REPLAN}:
                    objects = self._perceive()
                    self.runtime.observe(objects, receipt={"perception_mode": self.perception.__class__.__name__})
                elif phase == RuntimePhase.PLAN:
                    action = self.runtime.plan(objects, planner_receipt={"pipeline": "agentic_rag_vlm"})
                    objects_before_execution = dict(objects)
                    place_command_id = None
                elif phase == RuntimePhase.EXECUTE:
                    action = self.runtime.current_action
                    subgoal = self.runtime.current_subgoal
                    if action is None or subgoal is None:
                        self.runtime.safe_stop("session_state_inconsistent")
                        break
                    pick = self.mapper.pick_command(action)
                    self._record("command", {"sent": pick.to_dict(), "receipt": dict(self.transport.send_command(pick))})
                    self.runtime.begin_execution(
                        skill=str(action.strategy.get("synergy", "power")),
                        executor_receipt={"pick_command_id": pick.command_id},
                    )
                    place = self.mapper.place_command(action, subgoal.destination)
                    place_command_id = place.command_id
                    self._record("command", {"sent": place.to_dict(), "receipt": dict(self.transport.send_command(place))})
                elif phase == RuntimePhase.VERIFY:
                    action = self.runtime.current_action
                    if action is None:
                        self.runtime.safe_stop("session_state_inconsistent")
                        break
                    objects_after = self._perceive()
                    result = self.verifier.verify(
                        action.target,
                        task.protected_objects,
                        objects_before_execution,
                        objects_after,
                    )
                    self._record("verification", {"result": result.evidence, "success": result.success})
                    self.runtime.verify(
                        result.success,
                        quality=result.quality,
                        evidence={**result.evidence, "place_command_id": place_command_id},
                    )
                    objects = dict(objects_after)
                elif phase == RuntimePhase.MONITOR:
                    change = detect_scene_change(objects_before_execution, objects)
                    self.runtime.monitor(change)
                else:
                    self.runtime.safe_stop(f"unhandled_phase:{phase.value}")
                    break
        except Exception as error:  # noqa: BLE001 - converted into an auditable safe stop
            self.runtime.safe_stop("platform_error", error_type=type(error).__name__, message=str(error))
            self._record("platform_error", {"error_type": type(error).__name__, "message": str(error)})
        return {
            "outcome": self.runtime.phase.value,
            "task_id": task.task_id,
            "summary": self.runtime.summary(),
            "trace": self.runtime.trace(),
            "receipts": list(self.receipts),
        }
