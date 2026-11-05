# YAM contact-dynamics qualification — 2026-11-02

This is an executor / framework integration test for the BluPe delivery work.
It is separate from the original IROS experiments and is not a benchmark
superiority study. No hardware motion is performed by these commands.

## Reproduce

Use the source tree in an editable installation. Robot XML, meshes and the
upstream Apache-2.0 license are included in `assets/yam_mjlab/`.

```bash
python -m pip install -e '.[sim,dev]'
MUJOCO_GL=egl python scripts/run_yam_physics.py --neighbor --record \
  --output output/yam_single_run
MUJOCO_GL=egl python scripts/run_yam_physics.py --neighbor --vlm --record \
  --endpoint http://127.0.0.1:18070/v1/chat/completions \
  --model /path/to/the/served/model --output output/yam_vlm_run
MUJOCO_GL=egl python scripts/run_yam_qualification.py --vlm \
  --config configs/yam_physics_holdout.json \
  --endpoint http://127.0.0.1:18070/v1/chat/completions \
  --model /path/to/the/served/model --output output/yam_holdout
```

The first command uses deterministic diagnostic color identities; `--vlm`
requires actual model classifications and has no identity fallback. Configure a
running OpenAI-compatible vision endpoint. Weights are not included. The local
model name is an endpoint identifier, not a portable download requirement.

## What runs

`BluPeSession → AgenticRuntime → AgenticPipeline → PhysicsTransport` drives
observe, retrieve/plan, pick, place, verify, monitor, and completion. Registered
object dimensions and semantic priors are supplied explicitly. The protected
cylindrical vessel is a rigid fragile-object proxy, not a breakage simulator.

RGB-D color masks propose candidates. For model-enabled runs Qwen classifies
the cropped candidates as cube / vessel proxy / unknown. Candidate order changes
between observations; candidate IDs do not fix the identity. Depth and calibrated
camera geometry give visible top-feature positions; known half heights convert
them to centres. VLM metric-coordinate guesses are not accepted. This restricted
proposal mechanism does not handle arbitrary clutter, repeated colors or unknown
dimensions. Semantic material/fragility comes from the registry, not visual proof
that the rendered cylinder is glass.

HAA-RAG selects a grasp card and scene constraints alter approach offset/height
near the vessel. Joint-limit-constrained multi-start IK generates top-grasp
waypoints; position actuators execute smooth joint interpolation with `mj_step`.
The transport preflights waypoint reachability, including future placement,
before grasping. It does not yet perform full-arm collision-aware path search;
unexpected contacts during physical execution stop the run.

The free cube is grasped by two opposing fingers and gravity/friction/contact
dynamics. No object attachment, object-pose reset after initialization, disabled
robot collisions or scripted object motion is used. The scene uses ideal robot
link gravity compensation, approximate model collision shapes, a fixed cube
mass/friction, and bounded position servos. Requested grasp force is clipped to
the simulated actuator force cap; it is not calibrated fingertip force control.
Parallel-jaw actuation does not reproduce dexterous-hand synergies. The model
variant and native BluPe calibration are distinct; see asset provenance.

The blue rectangle is a non-colliding destination marker on the tabletop, not a
physical tray. Placement is followed by open-gripper retreat and parking away
from the destination **and** the protected vessel's camera view. Two distinct
final RGB-D captures verify placement, release and stability. Missing protected
objects are failures, never silently assumed safe.

## Acceptance and evidence

All criteria are in the frozen protocol JSON: successful framework outcome,
destination error below 4 cm, at least 0.5 s of bilateral finger contact while
object-centre height exceeds 10 cm, table-supported placement, no final finger
contact, protected displacement below 1.5 cm, no robot/protected contact, no arm
table or self penetration beyond 1 mm, target/robot penetration below 3 mm.
MuJoCo soft contacts may have small nonzero penetration; reporting zero
penetration would be incorrect. Contact monitoring runs every physics step.

Public receipts include model outputs, candidate crops/depth, retrieved card,
constraints, actuator parameters and settled TCP errors. Private evaluator
traces contain simulator object poses and contacts, and are not fed to the
identity/grasp planner. Internal physics contact monitoring also stops unsafe
rollouts; it is a simulation instrument, not a deployed real-robot sensor.

The suite freezes the source ZIP and config before launching cases. Each case
runs once from that snapshot, retains failures and writes `result.json`.
`summary.json` identifies the source/config hashes and every attempted seed.
Seed 10 is recorded for each scenario regardless of success; videos are raw
MuJoCo frames at simulation speed (25 fps), without outcome edits. Model-call
wall time occurs between simulation steps and is not part of this playback.

Development seeds 0–9 and earlier `dev_*` runs are not held-out results. The first
development batch completed 19/20 framework episodes: one protected-vessel
classification was unknown after placement because the parked hand partially
occluded it. The framework correctly rejected that episode despite successful
physical placement. Parking was moved farther away before declaring held-out
seeds 10–19. Earlier failures and their video/trace evidence remain preserved.

These two simple task families qualify integration; they do not independently
establish benefits over detection+IK. Paired ablations, multi-object sequences,
less restricted perception, validated recovery experiments and actual BluPe
hardware acceptance remain separate work.
