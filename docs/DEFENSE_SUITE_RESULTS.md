# Guanghua Agentic RAG-VLM Defense Suite

## Deliverable

The primary defense/interview video is:

`output/defense_suite/showcase/guanghua_agentic_rag_vlm_showcase.mp4`

The recommended self-contained presentation reel is:

`output/defense_suite/agentic_rag_vlm_full_defense_reel_v9.mp4`

It is a 67.0-second 1920×1080/30 fps composition containing a Fudan-branded
opening, framework summary, the complete uncut Showcase, two paired failure/
correction experiments, Challenge v4 results, and the explicit claim boundary. The editable source and
official-logo provenance are under `video/guanghua_defense_reel/`.

The paired chapter makes the Agentic advantage visible under shared perception,
robot model, and IK execution: fixed wrist orientation retains only 1.5 mm
protected-object clearance and fails the declared 3 mm keep-out gate, while
relation-conditioned candidate selection retains +51.9 mm without penetration;
after a declared 80.0 mm target move, the stale waypoint is rejected at the
alignment gate while the Agentic runtime attributes the change and issues one
bounded L3 replan. The E3 comparison audits both methods at all 417 rendered
frames and admits only trajectories without visible table/target penetration.
The yellow object is explicitly identified as a protected fragile constraint, not
an unexplained target.

The directory also contains three controls with the same robot, table, targets,
camera, duration, and low-level executor:

| Scenario | Graph constraint | External change | L3 replans | Result |
|---|---:|---:|---:|---|
| Nominal | no | no | 0 | complete |
| Fragile-aware | yes | no | 0 | complete |
| Recovery | no | yes | 1 | complete |
| Showcase | yes | yes | 1 | complete |

Every rollout is one continuous 22.5 s, 675-frame, 1280×720/30 fps video. Each
places the red cube and blue cylinder into their matching zones, preserves the
fragile object at zero measured displacement, and terminates the reusable
Agentic runtime in `complete` with no pending target.

## What the suite validates

The experiment exercises a real multi-subgoal lifecycle rather than a fixed
animation label:

`OBSERVE → PLAN → EXECUTE → VERIFY → MONITOR → [REPLAN] → COMPLETE`

- public agent-view RGB-D produces semantic object estimates;
- HAA-RAG selects POWER for the cube and PINCH for the cylinder;
- the scene graph activates only when the fragile proxy is adjacent;
- verified red completion is stored before monitoring the remaining target;
- a declared 29 mm blue-object displacement invalidates only the pending blue
  plan and causes exactly one bounded L3 replan;
- no-change controls continue without a spurious replan;
- Qwen3.5-4B result receipts are retained alongside the deterministic,
  auditable geometry and execution layers.

The semantic graph proposes a 50 mm avoidance vector. With the calibrated grasp
frame, natural-posture IK prior, and full tool-orientation constraint, the
recorded Fragile/Showcase rollout executes the full vector (projection scale
1.0) while retaining at least 0.3411 rad joint-limit margin. Both planned and
executed offsets are recorded, and the red transport then proceeds away from
the protected object.

## Motion and evidence gates

Across the four videos:

- minimum joint-limit margin: at least 0.3411 rad;
- minimum fingertip/table clearance: 2.035 mm;
- maximum terminal IK error: 0.0272 mm;
- target-zone errors: 0 mm under the declared skill proxy;
- fragile-object displacement: 0 mm;
- runtime/video/artifact audit: PASS for all four scenarios;
- unit test suite: 41 passed.

Each run keeps `public_trace.jsonl`, `agentic_runtime_trace.jsonl`, RGB-D
observations, `private_evaluator.json`, `timeline.json`, `runtime_receipt.json`,
and a captioned MP4. `suite_manifest.json` records hashes and metrics;
`validation_report.json` records the automated audit.

## Claim boundary

The suite validates Agentic orchestration, public-observation replanning,
constraint-aware IK, memory, and calibrated manipulation-skill execution. Object
transport is an explicitly logged kinematic grasp-skill proxy. Therefore the
videos are suitable evidence for the framework and Guanghua simulation path,
but they are not evidence of contact-dynamics grasp robustness or sim-to-real
transfer. Those remain the next experimental stage.

## Reproduce

```bash
cd Agentic-RAG-VLM
MUJOCO_GL=egl python scripts/render_defense_suite.py --render-rollouts
python scripts/validate_defense_suite.py
pytest -q tests
```
