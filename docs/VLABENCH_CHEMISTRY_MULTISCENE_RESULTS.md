# VLABench chemistry multi-scene evidence

Date: 2026-09-02

## What is being evaluated

Each episode uses a frozen VLABench `take_chemistry_experiment` scene and an
independent RGB observation.  A real local Qwen3.5-4B multimodal call selects
the reagents and describes their manipulation attributes.  Reaction RAG checks
the reagent set and order, HAA-RAG selects the glass-vessel handling strategy,
the scene graph binds distractors and spatial constraints, and episodic memory
is updated only after trajectory-level verification.

The low-level joint trajectory is executed by registered VLABench skills plus
a Cartesian tube-mouth controller.  Therefore these runs test Agentic RAG-VLM
as a high-level manipulation agent; they are not evidence of an end-to-end VLA
policy generating joint actions.

The VLABench skill layer is IK based: Cartesian waypoints are converted to
Franka joint targets and then tracked by the simulator controller.  The local
executor now restricts IK to the seven arm joints, checks convergence and joint
limits, retries failed poses deterministically, interpolates large valid joint
changes, and blocks invalid targets before they can reach MuJoCo.

## Frozen-scene results

| Seed | Reaction | Qwen targets | Reaction RAG | Official | Strict Agentic | Successful-pour evidence | Video |
|---:|---|---|---|---:|---:|---|---|
| 2 | NaCl + AgNO3 -> AgCl | NaCl, AgNO3 | silver-chloride card, 0.95 | pass | pass | 4 + 1 simultaneous samples; 1.97/5.39 mm minimum mouth XY error | 47.2 s |
| 5 | HCl + NaOH -> NaCl + H2O | HCl, NaOH | acid-base card, 0.94 | pass | pass | 7 + 1 simultaneous samples; 0.88/1.39 mm minimum mouth XY error | 44.4 s |
| 7 | BaCl2 + K2CrO4 -> BaCrO4 | BaCl2, K2CrO4 | barium-chromate card, 0.95 | pass | pass | 8 + 1 simultaneous samples; 0.56/2.36 mm minimum mouth XY error | 41.3 s |
| 0 | Na2CO3 + HCl -> NaCl + CO2 | HCl, Na2CO3 | corrected order to Na2CO3, HCl | fail | fail | first pour passed; second tube did not leave its rack slot | failure retained |

The three successful videos total 132.9 seconds.  This is a small, deliberately
selected multi-scene demonstration, not a benchmark-wide success-rate claim.

## Post-gate chemistry regression

The frozen seed-5 HCl/NaOH episode was rerun after adding the IK and physics
execution gates.  The complete episode passed both the official benchmark and
the stricter Agentic pour verifier:

- the 14-stage registered skill sequence executed through benchmark success;
- 405 Cartesian waypoint IK calls, all accepted on the primary solve;
- 18 large-but-valid joint transitions split into bounded subtargets;
- zero rejected IK targets, joint-jump gates or physics-instability gates;
- both reagent pours verified while the tube mouth was aligned with the flask
  opening and the tube tilt exceeded 90 degrees.

The final seed-5 result also records one stage-boundary unexpected-contact
snapshot involving `HCl_tag`, with an anomalously large contact-force proxy.
This is under investigation as a non-physical label-geometry or contact-
classification issue.  Zero physics-instability gates must not be interpreted
as collision-free execution or glass-safety validation until it is resolved.

This regression result supersedes the older seed-5 video for control-quality
demonstration, while retaining the same frozen scene and planner receipt.

## Cross-task household generalization

The same Agentic interface was also evaluated on VLABench `cook_dishes` seed 7
with the instruction `Prepare the ingredients of broccoli_and_cheese_bake in
the plate.`  Recipe retrieval and the multimodal planner selected `broccoli`
and `cheese` from five candidate foods.  The executor ran the six-stage
registered pick/lift/place sequence, and post-action verification confirmed
both ingredients inside the plate.

- official benchmark: pass;
- strict Agentic result: pass;
- 304 Cartesian waypoint IK calls, all accepted on the primary solve;
- zero IK, joint-jump or physics-instability gates.

Chemistry remains the primary constrained-manipulation result; `cook_dishes`
is a cross-domain knowledge-retrieval and target-selection result rather than a
replacement for the chemistry task.

## Active container manipulation and natural RAG correction

The official VLABench `take_out_cool_drink` seed 0 was run with the indirect
instruction `I am so thirsty after sport, I want to drink something healthy
cool`.  A first online Qwen3-VL-2B RGB call observed the closed fridge and
routed the agent to open it.  After the robot opened the door and moved clear,
a second call observed `juice` and `monster` inside the fridge.

The raw second-stage model incorrectly proposed `monster` as healthy.  Public
intent RAG matched `healthy_post_sport_juice` and corrected the executable
target to `juice`; HAA-RAG then selected the narrow-cylinder pinch card.  The
robot extracted the juice, placed it on the table, withdrew, reacquired the
door handle and closed the fridge.

- official benchmark: pass;
- strict Agentic result: pass;
- the 14-stage registered VLABench sequence executed through final success;
- 457 Cartesian waypoint IK calls, all accepted on the primary solve;
- 7 large-but-valid joint transitions interpolated;
- zero IK, joint-jump or physics-instability gates;
- final verification: juice outside the fridge, fridge closed, memory complete.

The 50.8-second video is the strongest household Agentic demonstration in the
current suite because the model is called both before and after a physical
scene change, and the raw proposal, RAG correction and executed target are all
stored separately.

## Model and audit evidence

- Four independent Qwen3.5-4B calls were made for seeds 0, 2, 5 and 7.
- Total recorded usage is 5,625 tokens; all four responses parsed without an
  endpoint or JSON error.
- The planner inputs exclude expected targets, object poses, rewards and
  success flags.  Evaluator-only fields are stored separately.
- Seed 0 is a natural RAG correction case: the raw model emitted `HCl,
  Na2CO3`; the retrieved reaction card corrected this to the required
  `Na2CO3, HCl` order.
- Seed 5 initially failed closed because the reaction knowledge base lacked an
  HCl-NaOH card.  After adding the public chemistry card, the same frozen
  observation was replanned by Qwen and executed successfully.

## Verification definition

A pour enters episodic memory only when the physical tube `top_site` is within
15 mm in XY and 40 mm in Z of the flask opening while the measured tube-axis
tilt is greater than 90 degrees.  VLABench's asynchronous task flags are kept
as a separate official result and cannot by themselves produce
`agentic_success`.

The current controller treats the tube mouth as the controlled point.  During
rotation it translates the wrist so the mouth follows a continuous descending
trajectory from the upright clearance pose to the flask neck.  This replaced
the earlier fixed-wrist arc and removed the 80-110 degree sampling jump.

## Failure and recovery boundary

In seed 0 the monitor detected that HCl had moved only 4.6 mm from its rack
slot after the nominal transport, below the 120 mm transport gate.  It invoked
a bounded L3 regrasp policy.  Both the lower-body regrasp and a later diagnostic
using VLABench's upper-body keypoint failed to extract the tube from that
specific slot.  The run remains a failure and is useful as an honest recovery
boundary; it must not be presented as a successful correction.

The simulator has no glass-fracture model and the Franka gripper is
binary-position controlled.  Force and aperture values retrieved from the
knowledge card remain logged-only.  Stage-boundary tilt, drop and contact
proxies improve auditability but are not continuous-time safety guarantees.

## Canonical artifacts

- Seed 2 planner: `artifacts/vlabench/agentic_take_chemistry_20260901/seed_002/planner_receipt_v2.json`
- Seed 2 result: `artifacts/vlabench/agentic_take_chemistry_20260901/seed_002/execution_v8/result.json`
- Seed 5 planner: `artifacts/vlabench/chemistry_multiscene_20260902/scout/seed_005/planner_receipt.json`
- Seed 5 gated result: `artifacts/vlabench/chemistry_multiscene_20260902/scout/seed_005/execution_interpolated_ik_full_v7/result.json`
- Seed 5 gated video: `artifacts/vlabench/chemistry_multiscene_20260902/scout/seed_005/execution_interpolated_ik_full_v7/episode_success_True.mp4`
- Seed 7 planner: `artifacts/vlabench/chemistry_multiscene_20260902/scout/seed_007/planner_receipt.json`
- Seed 7 result: `artifacts/vlabench/chemistry_multiscene_20260902/scout/seed_007/execution_continuous_mouth_v4/result.json`
- Seed 0 full failure: `artifacts/vlabench/chemistry_multiscene_20260902/scout/seed_000/execution_replan_v3/result.json`
- Seed 0 keypoint-recovery diagnostic: `artifacts/vlabench/chemistry_multiscene_20260902/scout/seed_000/execution_recovery_diag_v4/result.json`
- Cook seed 7 planner: `artifacts/vlabench/agentic_cook_dishes_20260901/seed_007/planner_receipt_v2.json`
- Cook seed 7 gated result: `artifacts/vlabench/agentic_cook_dishes_20260901/seed_007/execution_ik_gate_v3/result.json`
- Cook seed 7 gated video: `artifacts/vlabench/agentic_cook_dishes_20260901/seed_007/execution_ik_gate_v3/episode_success_True.mp4`
- Cool-drink seed 0 result: `artifacts/vlabench/agentic_take_out_cool_drink_20260902/seed_000/execution_v2/result.json`
- Cool-drink seed 0 video: `artifacts/vlabench/agentic_take_out_cool_drink_20260902/seed_000/execution_v2/episode_success_True.mp4`
- Qwen3-VL-2B NF4 server receipt: `artifacts/vlabench/agentic_take_out_cool_drink_20260902/qwen3vl_2b_nf4_server_receipt.json`

## Rejected active-search candidate

`find_unseen_object` was evaluated but is not accepted as positive evidence.
Its official cabinet asset can eject the hidden target during drawer motion;
the benchmark `not_contain` condition then becomes true even though the robot
never found or grasped the object.  Seeds 0 and 20 retain this failure with
`agentic_success=false`.  The suite uses the stable fridge task above instead
of hiding or exploiting that benchmark-physics failure.
