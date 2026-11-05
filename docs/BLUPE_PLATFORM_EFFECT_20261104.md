# Local BluPe task-effect protocol — 2026-11-04

## Completed results

All three declared runs completed through the official local Run route. Each
performed three contact-driven grasps and passed all independent physical
checks. Original commands, raw replies, contact traces, videos and the frozen
pre-run protocol are retained in
`仿真实验/blupe_local_effect_20261104/campaign_v1/` at the workspace root.

| Task / approach | Completed objects | Failed grasps | Memory hits | Motor-cap updates | L3 replans | Actual model replies |
|---|---:|---:|---:|---:|---:|---:|
| Three-object sorting / Full Agent | 3 | 0 | 1 | 1 | 0 | 12 |
| Unstack + sort / Full Agent | 3 | 0 | 1 | 0 | 1 | 14 |
| Same unstack scene / geometric | 3 | 0 | 0 | 6 | 0 | 11 |

The stacked pair has **byte-identical scene XML** (SHA256
`b5b8c348e7fbe7137afeb937acb2317696185c502e37d1afc1e3993ecc9737d3`).
The full Agent planned yellow upper block → red base → magenta cylinder and
reused verified experience for the next compatible object. Its L3 event follows
the changed observation after unstacking. The 37 actual model replies include
perception, identity review and action review; the baseline has no action
reviewer. The KB was not updated between cases.

The full stacked case needed fewer motor-force-cap adjustments than this
baseline case (0 versus 6). These were **preventive low-height feedback updates**,
not failed grasps, calibrated hardware gripping forces or proof of a general
performance gain. Both approaches succeeded with three grasps. The batch did
not trigger post-failure recovery or the visible-goal fallback, so those abilities
are not established by this demonstration.

`audit.json` independently rechecks final body positions/table contacts against
the saved evaluator, source/knowledge hashes, real model identities, every case
and its video; no audit errors remain. Full regression: 322 passed, 12 skipped
(10 historical-asset tests plus two optional previous recorded-image preview
checks). A later UI-only change is covered by six focused UI/cancellation tests
and a real browser check, including MP4 byte-range playback. Frozen simulation
source and past benchmark figures remain unchanged.

Local task page: http://127.0.0.1:18791/

Results and all three videos: http://127.0.0.1:18791/experiment/report

Stable raw videos are also copied to the campaign's `videos/` directory.
The experiment verifies a complete **local simulation integration**, not a
public BluPe lease, native robot wire dispatch or physical hardware performance.

The original protocol below was frozen before these runs.

This new engineering batch starts complete manipulation tasks through the
official local Playground `/api/run` route and RunnerController task hook.
Its child executor is the existing MuJoCo contact-dynamics/Agentic pipeline,
with actual Qwen3.5-4B perception/review and CLIP/HAA-RAG. It is not a public
BluPe hardware trial or a calibrated digital twin of the remote station.

## Declared cases, before execution

1. `agentic-flat`, seed 980: sort three separate colored objects to individual
   destination zones while protecting the green vessel. Full framework.
2. `agentic-stack`, seed 981: remove the upper target before its base and sort
   all three objects; mixed shape/contact, 17-mm upper half-width, guarded
   controller, visibility replanning and bounded visible-goal fallback.
3. `geometric-stack`, seed 981: identical stacked scene and contact controller,
   same real VLM identity perception; fixed-skill retrieval, height-based
   ordering, no scene-graph task reasoning, no episodic memory, no VLM action
   reviewer. This is the existing strong geometric comparison.

All three use the 14-record `yam_verified_diverse_r8.json` store and the existing
bounded force-feedback controller. The full conditions use actual HAA-RAG and
verified memory reuse. Models, knowledge, scene material choices, cameras and
core executor are not changed after seeing case outcomes. This is a small
integration/showcase batch, not a benchmark, significance test or proof of
success-rate superiority. Record all outcomes, errors and operator cancellations;
do not relabel a fresh rerun as the original case.

Independent simulation evaluation uses the existing checks: each object near
its destination, table support after release, bilateral contact during lift,
protected-object/contact/displacement limits, penetration limits, and support
order for stacked tasks. Model claims of completion are insufficient. Any
missing final result is a failed/incomplete run. No object attachments or
post-initialization object-pose resets are introduced.

## Visibility and execution evidence

The official controller admits only a marked local mock session. The task hook
launches a frozen-source simulation child; the actual model and contact executor
run there. Native robot queue/trajectory writes are not used by this adapter.
It must never be described as proof of public BluPe motion or native wire
control. Stop terminates the local simulation child and retains a cancellation
receipt; it does not turn partial motion into a successful task.

The browser receives a current MuJoCo overview frame and a read-only copy of
runtime events. Observer callbacks neither supply observations to the model
nor update physical state. RGB-D proposals are color based, identities are
classified by the VLM, and physical body-state traces are reserved for the
independent evaluator. The video is written by the actual simulation renderer.

Launch with `scripts/serve_blupe_simulation.py`; the generated UI adds local
progress, result, video and all-run-receipt links while retaining official
model selection, Run, CSRF and Stop. Requests use fixed declared scenarios and
cannot inject arbitrary script arguments, coordinates or server paths.

Per-run evidence: original command, source/knowledge/model identities,
`process.log`, observations/raw model replies, runtime events, simulation
result/contacts, memory and `rollout.mp4`. Frozen prior r8/r9 studies and the
0.3.1 interface acceptance are unchanged.
