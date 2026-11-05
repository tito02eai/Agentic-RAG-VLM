# Agentic RAG-VLM — BluPe software evaluation handoff

**2026-11-04 / 0.3.5:** [first hardware trial preparation](BLUPE_HARDWARE_TRIAL.md)
adds an explicitly read-only official UI, pending site configuration, configurable
RGB proposals and factory/paired-RGB contract checks. Actual saved camera and
RGB-D packet checks passed, as did a local grasp smoke. Regression: 354 passed,
12 skipped. No public hardware session, motion, publication or email was performed.

**2026-11-04 / 0.3.4:** [ten local robustness checks](BLUPE_ROBUSTNESS_20261104.md)
passed: four task completions and six expected stops, not ten successful grasps.
Moved-target memory reuse, eviction after failed reuse, stable-ID retry, missing
feedback, WebSocket reconnect and active Stop are covered. Full regression:
335 passed, 12 skipped. Public hardware qualification is still pending.

**2026-11-04 / 0.3.3:** the [recommended protected-neighbor task](RECOMMENDED_TASK_QUALIFICATION.md)
now runs through the production provider and official HTTP/WebSocket SDK, with
actual local actuator physics. Known-layout regression 10/10 and separate new
layouts 5/5 follow an initial retained 8/10 batch. Operator Stop and raw-receipt
audits passed. This is a partner integration candidate, not a completed public
BluPe hardware deployment. The read-only public station is uninitialized and
its calibration explicitly requires fresh physical validation.

Updated 2026-11-03. This is a simulation-qualified software candidate. No hardware command has been issued by this revision. The current task is to make the framework and complete grasp demonstrations reviewable before scheduling hardware work.

**0.3.0 platform update:** use [PLATFORM_DEPLOYMENT.md](PLATFORM_DEPLOYMENT.md) for the deployable backend and official runner adapter. Software installation/registration no longer depends on having a live robot. The static-card integration gap identified in 0.2.1 has been resolved: the remote CLI defaults to PaperAgent, and the registered runner provider uses the same full chain. Physical qualification remains a later activity.

## What is implemented

| Component | Implementation | Evidence |
|---|---|---|
| Task runtime | Observe–Plan–Execute–Verify–Monitor–Replan; explicit task/destination/protected-object contracts | `agentic_rag_vlm/runtime.py`, `platform/session.py`, trace and receipts |
| HAA-RAG | Category and four-dimensional affordance compatibility, followed by actual CLIP ViT-L/14 visual ranking | `retrieval.py`, pinned embedding namespace, per-action shortlist |
| Knowledge | 14 command-bound observed records: 12 completed grasp/place actions and 2 actual retention rejections | `knowledge_base/yam_verified_diverse_r8.json` and its bundled source images/results |
| Scene reasoning | Observed relations/states compiled to constraints and dependency ordering; unobserved identity stays unknown | `scene.py`, `constraints.py`, original perception receipts |
| VLM review | Qwen3.5-4B NF4 reviews candidate actions and bounded tools, retaining a stop decision | `reasoning.py`, raw prompts and responses |
| Memory | Verified successful execution parameters only; fresh localization on reuse; failed reuse invalidated | Session trace, actual execution-feedback force, memory snapshots |
| Physical execution | Actuator-driven YAM parallel-jaw skills, bounded IK refinement, proof lift, tactile retention and supported placement | `simulation/robust.py`, `guarded.py`, video and raw contact/evaluator traces |
| Platform adapter | Observation/command/verification interfaces, native-joint kinematics and measured-calibration admission | `platform/`, `DELIVERY_README.md`; offline contracts tested, hardware validation pending |

The planner does not output a trained VLA joint policy. It orchestrates checked skills. The gripper force recorded here is a MuJoCo motor limit; contact normal force is logged separately. Model inference latency is excluded from the simulation-time video, but wall-clock run times and actual request counts are retained.

## Latest repair and evaluation

The completed r8 study used 12 seeds and four paired conditions (48 rollouts). Full Agent completed 8/12; the strong geometric policy completed 12/12. All old failures are retained. The geometric comparison uses the same Qwen identity perception, CLIP-compatible interface, feedback controller and bounded contact retries, while omitting task-level Agent review, retrieved experience selection, scene constraints and memory.

The primary gap was premature stopping under partial visibility. r9 introduces an opt-in visible-goal progress tool: two fresh observations must establish a known elevated target and all protected objects as stable (within 4 mm); existing task permissions, constraints, destination, replan budget and physical guards remain active. The VLM must approve the bounded action. No identity or support relationship is invented for the missing target. That target remains pending and is re-observed after the known target is moved.

The r9 protocol declares known regression seeds 928–931 separately from new seeds 960–963 (four paired full/geometric runs, eight cases). Use the final `DELIVERY_STATUS.json`, `evidence/r9/qualification_v1/REPORT_CN.md` and the independent audits for completed scores. Do not combine known regressions with new-seed scores or infer statistical superiority from four pairs. The delivery entry plays a complete new-seed task and a same-scene before/after known regression.

Completed r9 scores: all four known regressions passed; both conditions completed 4/4 new seeds. Full/geometric used 12/16 grasp attempts, 0/4 failed attempts and 50/48 real model replies. This trades fewer grasps for slightly more inference in this small cohort. New-seed audit passed without warnings; frozen-source tests passed 270 with 10 historical-asset skips.

## Reproduce and inspect

1. Unpack the source ZIP; use Python 3.10 (tested: 3.10.21). Install `pip install -r configs/extended_sim_dependencies.txt` and then `pip install -e . --no-deps`. Run `python -m pytest -q`. Broad extras may select a newer MuJoCo than the measured 3.2.2; use the pinned file for result reproduction.
2. Start the separately provisioned Qwen3.5-4B and CLIP ViT-L/14 services as described in `docs/ROBUST_SIM_PROTOCOL.md`. Model weights are not bundled. Default localhost ports: 18074 (Qwen), 18072 (CLIP).
3. Follow `docs/CURRENT_DELIVERY.md` for one recorded grasp run or the frozen study runner. Always use a new output directory. The runner freezes code/config/knowledge and checks model identity before each case.
4. Audit a delivered batch without model services: `python scripts/audit_extended_benchmark.py /path/to/evidence/r9/qualification_v1`. The same audit accepts the r8 batch and the separately labelled known-regression wrapper.
5. Open `index.html` or an evidence `viewer.html` locally. Videos, frame/crop evidence and JSON are bundled. Absolute paths inside historical prompts preserve provenance; files are also available in the case-relative directories.

The final source includes SHA256 manifests. Executed runtime and inference scripts are compared byte-for-byte with the frozen study source. A post-run analysis wording correction is separately recorded; original raw rollouts and the frozen report generator are retained.

## Limits and first hardware step

The physical testbed uses known categories/sizes, color-based RGB-D proposals, a calibrated horizontal table, two rigid blocks, a rigid cylinder, a protected vessel and a parallel-jaw gripper. It does not validate open-world segmentation, breakable objects, hand dexterity, long-term cross-session memory, real-hardware force control or the complete VLABench suite. Semantic fragility constraints are implemented but damage physics is not. The 14 new records are not the missing original paper-era 116-entry store; these new engineering results are not substitutes for paper-era measurements.

For a first BluPe session, use one known rigid object and one measured destination, with fresh independently validated calibration, initialized/homed hardware, native joint/workspace limits, measured grasp offsets, camera intrinsics and on-site oversight. Do not transfer MuJoCo motor limits as calibrated hardware fingertip forces. Multi-object clearing and protected-neighbor tasks should follow the single-object hardware acceptance. This document does not authorize submitting commands to the live station.

## Previous integration gap, resolved in 0.3.0

The full profile is now available through `run_extended_sim.py`, `run_platform_session.py --policy-profile paper`, and `AgenticRAGVLMProvider.run_session`. The platform perception adapter supplies actual CLIP crops/embeddings and observed scene context to PaperAgent; candidate identities and quantitative tools remain separate. The official runner retains queue, capability and Stop ownership. Executor-domain binding separates hardware-session memory from the simulation-origin knowledge. The 14 records remain retrieved priors, and unsupported force/synergy fields are labelled unapplied. Measured site geometry and supervised physical acceptance are required when later enabling actual motion, not to deploy or prepare the software.

For portable simulation runs, `--endpoint`, `--model` and `--clip-endpoint` are accepted by the campaign runner and forwarded to each frozen case. Environment alternatives are `AGENTIC_VLM_ENDPOINT`, `AGENTIC_VLM_MODEL` and `AGENTIC_CLIP_ENDPOINT`. An omitted model is resolved from the actual local service health response rather than a developer's filesystem path. `check_sim_readiness.py` checks the pinned dependencies, source-bearing knowledge and both services without creating a hardware session.
