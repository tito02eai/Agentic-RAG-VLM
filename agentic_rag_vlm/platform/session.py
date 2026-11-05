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

from math import isfinite, dist
from dataclasses import replace
from typing import Any, Mapping, Sequence

from scripts.agentic_framework import PublicObject, detect_scene_change

from ..runtime import AgenticRuntime, RuntimePhase, TaskSpec
from ..recovery import FailureType, recover
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
        *,
        destinations: Mapping[str, Sequence[float]] | None = None,
        verification_samples: int = 2,
        require_release_confirmation: bool = False,
        max_precontact_retries: int = 3,
        prerequisite_destinations: Mapping[str,str] | None = None,
        contact_recovery=None,
    ) -> None:
        self.runtime = runtime
        self.transport = transport
        self.perception = perception
        self.mapper = mapper
        self.verifier = verifier
        self.destinations = {
            name: tuple(float(value) for value in xyz)
            for name, xyz in (destinations or {}).items()
        }
        self.receipts: list[dict[str, Any]] = []
        if verification_samples < 2:
            raise ValueError("at least two final observations required")
        self.verification_samples = verification_samples
        self.require_release_confirmation = require_release_confirmation
        self._last_observation = None
        if not 0 <= max_precontact_retries <= 3:
            raise ValueError("precontact retries must be bounded by 3")
        self.max_precontact_retries = max_precontact_retries
        self.prerequisite_destinations = dict(prerequisite_destinations or {})
        # Default remains disabled, including real hardware. A simulator may
        # supply a policy that proves release/stability before any retry.
        self.contact_recovery = contact_recovery

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
        self._last_observation = observation
        self._record("observation", observation.to_dict())
        objects, receipt = self.perception.perceive(observation)
        self.runtime.pipeline.scene_context = (receipt.vlm_result or {}).get("scene_context", {})
        self.runtime.pipeline.visual_features = (receipt.vlm_result or {}).get("visual_embeddings", {})
        self.runtime.pipeline.embedding_space = (receipt.vlm_result or {}).get("embedding_space")
        if self.runtime.pipeline.reasoner is not None:
            from pathlib import Path
            self.runtime.pipeline.reasoner.images = [Path(p) for p in observation.image_paths]
            crops = {c["candidate_id"]:Path(c["crop_path"]) for c in observation.receipt.get("rgbd_candidates",[])}
            self.runtime.pipeline.reasoner.target_images = {
                g["name"]:[crops[g["candidate_id"]]] for g in (receipt.vlm_result or {}).get("grounding",[])
                if g.get("candidate_id") in crops}
            for g in (receipt.vlm_result or {}).get('grounding',[]):
                if g.get('crop_path'):
                    self.runtime.pipeline.reasoner.target_images[g['name']]=[Path(g['crop_path'])]
        self._record("perception", receipt.to_dict())
        return objects

    def run(self, task: TaskSpec) -> dict[str, Any]:
        self.receipts = []
        self.runtime.start(task)
        objects: dict[str, PublicObject] = {}
        objects_before_execution: dict[str, PublicObject] = {}
        place_command_id: str | None = None
        retries = 0
        try:
            if set(task.protected_objects) & {goal.target for goal in task.subgoals}:
                self.runtime.safe_stop("target_is_protected")
            for subgoal in task.subgoals:
                xyz = self.destinations.get(subgoal.destination)
                if xyz is None or len(xyz) != 3 or not all(isfinite(v) for v in xyz):
                    self.runtime.safe_stop("destination_unavailable", destination=subgoal.destination)
                    break
            while True:
                phase = self.runtime.phase
                if phase in {RuntimePhase.COMPLETE, RuntimePhase.SAFE_STOP, RuntimePhase.IDLE}:
                    break
                if phase in {RuntimePhase.OBSERVE, RuntimePhase.REPLAN}:
                    objects = self._perceive()
                    if self.runtime.pipeline.config.scene_graph_enabled:
                        from ..scene import schedule_dependencies
                        scheduled = schedule_dependencies(task,self.runtime.pipeline.scene_context,
                            prerequisite_destinations=self.prerequisite_destinations)
                        if scheduled.subgoals != task.subgoals:
                            if any(g.destination not in self.destinations for g in scheduled.subgoals):
                                raise ValueError("support removal needs a configured destination")
                            task = scheduled
                            self.runtime.task = task
                            self._record("support_schedule",{"subgoals":[g.__dict__ for g in task.subgoals]})
                    invalid = sorted(
                        name for name, item in objects.items()
                        if len(item.center_xyz_m) != 3
                        or not all(isfinite(float(v)) for v in item.center_xyz_m)
                    )
                    if invalid:
                        self.runtime.safe_stop("invalid_object_position", invalid_objects=invalid)
                        break
                    missing = sorted(set(task.protected_objects) - set(objects))
                    if missing:
                        self.runtime.safe_stop("protected_object_not_visible", missing_objects=missing)
                        break
                    if (self.runtime.config.visible_goal_fallback
                            and set(self.runtime.pending_targets)-set(objects)
                            and not self.runtime.pipeline.scene_context.get('unresolved_supports')):
                        previous=dict(objects);previous_observation=self._last_observation
                        objects=self._perceive()
                        current=self._last_observation
                        old_time=previous_observation.receipt.get('captured_at')
                        new_time=current.receipt.get('captured_at')
                        fresh=(previous_observation.observation_id!=current.observation_id
                            and old_time is not None and new_time is not None and new_time>old_time)
                        stable=[name for name in set(previous)&set(objects)
                            if fresh and all(isfinite(v) for v in objects[name].center_xyz_m)
                            and dist(previous[name].center_xyz_m,objects[name].center_xyz_m)<=.004]
                        pair={'previous_observation_id':previous_observation.observation_id,
                            'current_observation_id':current.observation_id,
                            'previous_capture_time':old_time,'current_capture_time':new_time,
                            'stable_visible_targets':sorted(stable),'position_tolerance_m':.004}
                        self.runtime.pipeline.scene_context['visible_goal_stability']=pair
                        self._record('visibility_stability_check',pair)
                        if set(task.protected_objects)-set(objects):
                            self.runtime.safe_stop('protected_object_not_visible');break
                    self.runtime.observe(objects, receipt={"perception_mode": self.perception.__class__.__name__,
                        'observation_id':self._last_observation.observation_id})
                    task=self.runtime.task
                elif phase == RuntimePhase.PLAN:
                    action = self.runtime.plan(objects, planner_receipt={"pipeline": "agentic_rag_vlm"})
                    objects_before_execution = dict(objects)
                    place_command_id = None
                    retries = 0
                elif phase == RuntimePhase.EXECUTE:
                    action = self.runtime.current_action
                    subgoal = self.runtime.current_subgoal
                    if action is None or subgoal is None:
                        self.runtime.safe_stop("session_state_inconsistent")
                        break
                    pick = self.mapper.pick_command(action)
                    pick_receipt = dict(self.transport.send_command(pick))
                    if pick_receipt.get('accepted') is not True and pick_receipt.get('execution_started') is True:
                        invalidated = self.runtime.invalidate_failed_memory(action, reason='observed_retention_failure')
                        if invalidated:
                            pick_receipt['memory_invalidated'] = invalidated
                            self._record('memory_invalidation', invalidated)
                    self._record("command", {"sent": pick.to_dict(), "receipt": pick_receipt})
                    if pick_receipt.get("accepted") is not True:
                        if self.contact_recovery is not None and retries < self.max_precontact_retries:
                            recovered = self.contact_recovery(self,action,pick_receipt,objects_before_execution,task,retries+1)
                            if recovered is not None:
                                retries += 1
                                action,objects,reflection = recovered
                                self.runtime.replace_after_safe_release(action,evidence=reflection)
                                objects_before_execution = dict(objects)
                                continue
                        if (pick_receipt.get("execution_started") is False
                                and pick_receipt.get("reason") == "motion_plan_failed"
                                and "IK failed" in pick_receipt.get("detail", "")
                                and retries < self.max_precontact_retries):
                            retries += 1
                            decision = recover(action.strategy, FailureType.APPROACH_ANGLE, retries,rng=self.runtime.pipeline.rng)
                            strategy = dict(decision.corrected_strategy)
                            if retries == 1:
                                strategy["approach_height_delta_m"] = float(strategy.get("approach_height_delta_m", 0)) + .03
                                recovered = replace(action, strategy=strategy, source="recovery_l1")
                            elif retries == 2:
                                strategy["approach"] = "side" if strategy.get("approach", "top") == "top" else "top"
                                recovered = replace(action, strategy=strategy, source="recovery_l2")
                            else:
                                objects = self._perceive()
                                if set(task.protected_objects) - set(objects) or action.target not in objects:
                                    self.runtime.safe_stop("recovery_observation_incomplete")
                                    break
                                recovered, evidence = self.runtime.pipeline.plan(objects[action.target], objects,use_memory=False)
                                if self.runtime.pipeline.retriever is not None:
                                    perturbation = decision.corrected_strategy["position_perturbation_m"]
                                    recovered = replace(recovered,position_xyz_m=tuple(a+b for a,b in zip(recovered.position_xyz_m,perturbation)),
                                        strategy={**recovered.strategy,"position_perturbation_m":perturbation})
                                recovered = replace(recovered, source="recovery_l3")
                                self._record("replan", evidence)
                                objects_before_execution = dict(objects)
                            self.runtime.replace_precontact_action(recovered, reflection=decision.to_dict())
                            if self.runtime.pipeline.reasoner is not None:
                                self._record("vlm_reflection",self.runtime.pipeline.reasoner.reflect(action,decision.to_dict(),self.receipts))
                            continue
                        self.runtime.safe_stop("command_rejected", command_id=pick.command_id)
                        break
                    feedback = pick_receipt.get('execution_feedback')
                    if feedback:
                        policy = pick.parameters.get('retention_force_policy', {})
                        force = feedback.get('force_n')
                        if (feedback.get('source') != 'public_tactile_low_height_force_adaptation'
                                or type(force) not in (int, float) or not isfinite(force)
                                or not pick.parameters['force_n'] <= force <= policy.get('ceiling_n', 0)
                                or not isfinite(policy.get('observed_fragility', float('nan')))
                                or policy.get('observed_fragility', 1.) > policy.get('maximum_fragility', .25)):
                            self.runtime.safe_stop('invalid_execution_parameter_feedback')
                            break
                        action = replace(action, strategy={**action.strategy, 'force_n': force})
                        self.runtime.current_action = action
                        self._record('execution_parameter_feedback', {'force_n': force, 'source': feedback['source']})
                    self.runtime.begin_execution(
                        skill=str(action.strategy.get("synergy", "power")),
                        executor_receipt={"pick_command_id": pick.command_id},
                    )
                    place = self.mapper.place_command(
                        action, subgoal.destination,
                        destination_xyz_m=self.destinations[subgoal.destination],
                    )
                    place_command_id = place.command_id
                    place_receipt = dict(self.transport.send_command(place))
                    self._record("command", {"sent": place.to_dict(), "receipt": place_receipt})
                    if place_receipt.get("accepted") is not True:
                        invalidated = self.runtime.invalidate_failed_memory(action, reason='placement_command_failed')
                        if invalidated:
                            self._record('memory_invalidation', invalidated)
                        self.runtime.safe_stop("command_rejected", command_id=place.command_id)
                        break
                elif phase == RuntimePhase.VERIFY:
                    action = self.runtime.current_action
                    if action is None:
                        self.runtime.safe_stop("session_state_inconsistent")
                        break
                    sample_results = []
                    sample_evidence = []
                    previous_objects = None
                    release_ok = True
                    stable = True
                    previous_id = None
                    previous_capture = None
                    for _ in range(self.verification_samples):
                        objects_after = self._perceive()
                        if self._last_observation.observation_id == previous_id:
                            stable = False
                        previous_id = self._last_observation.observation_id
                        result = self.verifier.verify(
                            action.target, task.protected_objects, objects_before_execution, objects_after,
                            destination_xyz_m=self.destinations[self.runtime.current_subgoal.destination],
                        )
                        if previous_objects is not None:
                            stable = stable and action.target in previous_objects and action.target in objects_after
                            if stable:
                                stable = dist(previous_objects[action.target].center_xyz_m,
                                              objects_after[action.target].center_xyz_m) <= self.verifier.protected_threshold_m
                        if self.require_release_confirmation:
                            capture = self._last_observation.receipt.get("captured_at")
                            if capture is None or capture == previous_capture:
                                stable = False
                            previous_capture = capture
                            state = self._last_observation.receipt.get("release_state", {})
                            release_ok = release_ok and state.get("gripper_open") is True and state.get("retracted") is True
                        sample_results.append(result.success)
                        sample_evidence.append(result.evidence)
                        previous_objects = objects_after
                    if not stable:
                        result.evidence["problems"].append("target_not_stable_or_repeated_observation")
                    if not release_ok:
                        result.evidence["problems"].append("release_or_retraction_unconfirmed")
                    result = type(result)(all(sample_results) and stable and release_ok,
                        1.0 if all(sample_results) and stable and release_ok else 0.0,
                        {**result.evidence, "sample_successes": sample_results,
                         "samples": sample_evidence,
                         "stable": stable, "release_confirmed": release_ok if self.require_release_confirmation else None})
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
