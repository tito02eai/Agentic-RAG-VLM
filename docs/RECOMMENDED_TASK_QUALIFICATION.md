# Recommended BluPe task: protected-neighbor rigid-object pick/place

Follow-up: [0.3.4 local fault, memory and Stop qualification](BLUPE_ROBUSTNESS_20261104.md).
This preserves the normal-task batches below and adds ten separate functional
checks; intentional stops are not counted as successful grasp tasks.

## Completed qualification — version 0.3.3

The production provider, official RunnerController and real HTTP/WebSocket SDK
completed this task against a local actuator-driven protocol simulator. The
official browser Run entry also completed it. Source, models and knowledge were
frozen for each batch; no transport, native IK or model replies were replaced.

| Cohort | Task results | Interpretation |
|---|---:|---|
| First frozen batch, 1010–1014 twice | 8/10 | Two pre-motion duplicate-candidate stops, retained |
| After RGB-component repair, same layouts twice | 10/10 | Known-layout regression, not new generalization |
| New layouts 1020–1024 | 5/5 | Small separate new-layout acceptance cohort |

The repaired cohorts submitted 1,499 actual wire waypoints in 30 trajectory
packets. Final placement errors were approximately 1.5–3.0 mm in this simulator;
all independent lift/release/support/protected-object/contact checks passed.
The two frozen batches retain 94 actual Qwen replies in total (34 then 60).
These are engineering trials, not original-paper results or hardware success
rates. They do not establish broad scene coverage or module superiority.

A separate operator-Stop test cancelled an active production-provider trajectory:
the gateway returned `aborted`, the agent stopped, no second packet was submitted,
and the cancelled episode was excluded from success cohorts. Full regression:
334 passed, 12 skipped (10 historical asset tests and two optional older
recorded-image model tests). All qualification integrity audits passed.

Current task UI: http://127.0.0.1:18793/

Video: http://127.0.0.1:18793/recommended/video

The fresh local episode has already completed; its result/video remain viewable.
The local gateway intentionally uses a fresh instance for each new trial. The
production provider itself follows the host's session lifecycle.

The public BluPe robot **has not performed this task**. The read-only station
snapshot is disabled/uninitialized and published calibration needs fresh site
validation. The delivered software is ready for partner-side integration review;
public availability/registration and hardware acceptance remain separate work.

Workspace evidence: `仿真实验/blupe_recommended_task_20261104/qualification_v1/`
and `qualification_v2/`. Their raw replies, images, wire packets, contact traces,
videos, memory, frozen source and all failures are preserved. Seed-1000
development/UI demonstrations and cancellation tests are excluded from scores.

The original pre-run protocol follows below.

Declared 2026-11-04 before the new acceptance batch. The first partner task is:
move a registered red rigid cube into a blue destination while leaving a green
vessel untouched. The vessel is a protected rigid proxy, not simulated breakable
glass. This exercises HAA-RAG, compiled neighbor constraints, VLM review, native
IK, SDK dispatch, completion feedback, stable release verification and memory.

## Execution chain and boundary

The actual `AgenticRAGVLMProvider.run_session` runs inside the unmodified official
RunnerController. The official HttpSessionAPI uses real loopback HTTP and
WebSocket connections to a protocol gateway. Its waypoints drive actual MuJoCo
actuator targets and `mj_step`; images and joint/gripper feedback return through
the same gateway. No transport, planner or model doubles are substituted.

Simulation profiles are explicitly labelled and restricted to a matching
loopback SDK origin with `local_simulation_only`. They cannot claim physical
validation or qualify public hardware. The idle second-arm channel is virtual
and held fixed; this single-active-arm MuJoCo robot is not a calibrated BluPe
bimanual twin. No public queue entry or hardware command is authorized by these
local tests.

The public station's read-only snapshot on 2026-11-04 reports DISABLED,
hardware_not_initialized, homed=false, settled=false and queue_ready=false.
Its calibration is experimental/provisional and calls for fresh physical
validation. Those external conditions prevent claiming a public hardware run.

## Predeclared acceptance

- Five new layout seeds: 1010, 1011, 1012, 1013, 1014.
- Two fresh service/session trials per layout: 10 tasks, every result retained.
- Seed 1000 development and UI demonstrations are excluded from that count.
- Freeze source and the 14-record KB before running. Do not edit knowledge,
  model, geometry or acceptance thresholds between cases or silently retry a
  failed case. Preserve development failures separately.
- Require framework completion AND independent physical destination/support/
  release/bilateral-lift/protected-object/contact/penetration checks.
- Every accepted trajectory must have terminal completed feedback. Acceptance
  alone does not establish execution or a grasp. Camera identities/timestamps,
  measured-state release/retraction and the configured aperture/lift retention
  proxy must pass. The proxy is not calibrated object-force sensing.
- This is a bounded software/task qualification, not a benchmark comparison,
  statistical superiority claim or estimate of public hardware success rate.

The object layouts vary within the original registered table setup. RGB region
proposals exclude the known blue painted destination and carry no identity or
metric object truth. Actual VLM classification and calibrated camera rays assign
registered identities and metric feature positions. Known object heights and
grasp offsets are explicit simulator geometry. Force/synergy recommendations
remain unsupported priors on the joint/gripper API; the simulated gripper uses
its declared fixed 1.6-N actuator cap.

## Reproduce

Use the pinned simulation dependencies in `configs/extended_sim_dependencies.txt`
and the runner HTTP/WebSocket dependencies. Start actual Qwen/CLIP services.

```bash
python scripts/run_blupe_protocol_task.py \
  --runner-checkout /path/to/blupe-remote-bots \
  --config configs/blupe_provider.example.json \
  --output /new/episode --seed 1010
```

For the official browser interface use `scripts/serve_blupe_recommended.py` with
the same arguments and `--port 18793`. One fresh local episode is permitted per
launch. Run/Stop and the task hook are official; the provider is the production
implementation, not the older subprocess simulation showcase adapter.

## Production defects repaired during development

- Accepted events no longer mask later completed events, including wrapped
  episode-trace entries.
- The other arm's normalized gripper field is read without a spurious `_deg`
  suffix.
- Carry preserves the previous close command. A measured aperture blocked by an
  object must not become an opening command that relaxes the grasp.
- A configured post-release observation pose clears the camera before final
  verification; lift clearance and the retention proxy are explicit profiles.
- VLM vetoes and raw review receipts are retained even when no plan is approved.
- Tool-review semantics distinguish an active protective constraint from a
  blocked action. VLM vetoes and deterministic motion checks remain enforced.

The frozen acceptance results are recorded separately when every case finishes.

## First frozen batch and independent repair validation

`qualification_v1` completed all ten declared tasks: eight succeeded; seed 1010
stopped before motion in both trials because JPEG edge pixels split a red
object's top and side faces into two candidate components. Duplicate VLM
identities were correctly rejected. These two failures remain in the original
batch. No thresholds, object poses or VLM decisions were overwritten.

The repair merges one-pixel gaps/eight-connected faces in the RGB proposal tool.
It supplies no identity or simulator object position. For `qualification_v2`,
declare ten repeats of the five known layouts as regression cases and five new
layout seeds 1020–1024 once each as a separate new-layout cohort (15 tasks total).
Freeze source and KB again before execution. Do not pool the known regressions,
new layouts, development seed or previous batch into one claimed success rate.
