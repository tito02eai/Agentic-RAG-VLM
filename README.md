# Agentic RAG-VLM

Reconstructed framework, auditable mechanism evaluation, and selected complete
VLABench manipulation tasks for the paper "Agentic RAG-VLM: Affordance-Aware
Retrieval-Augmented Generation with Self-Reflective Planning for Robotic
Grasping" (IROS 2026). The earlier Guanghua environment is retained as legacy
engineering evidence; current presentation work uses VLABench.

> **Provenance.** This standalone repository is derived from the public
> `Agentic-RAG-VLM` subtree and `paper/` sources of the
> [Agentic-VLA monorepo](https://github.com/tito02eai/Agentic-VLA)
> (subtree commit `09a1b01`). The reconstruction boundary is documented in
> [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

> **Asset note.** The school-provided Guanghua robot meshes/URDF bundle
> (`assets/`), model weights, original paper-era experience data, and raw
> `output/` run directories are **not** distributed in this repository. Some
> commands below require these locally obtained assets. See
> [docs/REPRODUCTION.md](docs/REPRODUCTION.md) before claiming a rerun.

## Project at a glance

**Problem.** Turn multimodal task candidates into auditable, constrained, and
failure-aware robot decisions without retraining a VLA policy.

**System.** Public RGB-D → Qwen3.5-VL tool router + planner → HAA-RAG /
deterministic Agent tools → scene-graph-constrained IK/hand skills →
verification, memory, and bounded replan.

| Evidence | Verified result |
|---|---:|
| Frozen paired evaluation | 100 method rollouts / 10 seeds |
| Real local multimodal inference | 246 Qwen3.5-4B calls / 0 endpoint errors |
| Full Agent | G1–G4: 10/10 each |
| Audit coverage | 260 public events / 60 Agent-tool calls / PASS |
| Regression suite | 41 tests passed |
| Current VLABench evidence | 3 final task outcomes / 1,166 accepted IK calls; chemistry cleanup pending |

**Claim boundary.** The project demonstrates real multimodal high-level
planning, public-observation tool execution, and completion of three selected
VLABench tasks through fixed skills and IK. It does not claim benchmark-wide
success, real-robot/contact-safety certification, end-to-end VLA control, or
sim-to-real. The earlier Guanghua G0-B contact-dynamics grasp remains
`NOT_ADMITTED`.

Key documentation:

- [Current experiment progress and defense narrative](docs/CURRENT_EXPERIMENT_PROGRESS.md)
- [BluPe remote-platform deployment plan and pending API questions](docs/BLUPE_DEPLOYMENT.md)
- [Reproduction boundary](docs/REPRODUCTION.md)
- [Challenge v4 (Agent-tool routing) results](docs/AGENT_TOOLS_V4_RESULTS.md)
- [Chinese paper/method introduction (engineering-status sections are historical)](docs/PROJECT_INTRODUCTION_CN.md)

## Layout

- `agentic_rag_vlm/`: reusable ReAct pipeline, quality model, recovery,
  memory, and adapter interfaces.
- `agentic_rag_vlm/runtime.py`: reusable multi-subgoal
  observe/plan/execute/verify/monitor/replan state machine.
- `agentic_rag_vlm/platform/`: remote-platform deployment layer (config,
  transport, perception, command translation, verification, session driver)
  for the planned BluPe deployment; see
  [docs/BLUPE_DEPLOYMENT.md](docs/BLUPE_DEPLOYMENT.md).
- `scripts/`: experiment entry points, renderers, validators, and summarizers
  (see headings below).
- `knowledge_base/affordance_cards.json`: compact, auditable Guanghua
  affordance-card library.
- `knowledge_base/blupe_semantic_registry.json` and
  `knowledge_base/blupe_affordance_cards.json`: frozen semantic registry and
  affordance cards for the BluPe stage plan (single objects, fragile
  neighbor, multi-object cleanup).
- `knowledge_base/vlabench_*.json`: affordance-card libraries for the VLABench
  tasks (chemistry, condiment, drink intent, food, reaction, recipe).
- `configs/`: frozen scene, condition, seed, metric, and evidence contracts.
- `tests/`: asset, framework, challenge, and runtime regression tests.
- `docs/`: experiment plans, frozen results, and claim boundaries.
- `paper/`: LaTeX sources, figures, `references.bib`, and compiled pre/final
  PDFs of the IROS 2026 submission.

Result documents:

- [docs/GUANGHUA_EXPERIMENT_PLAN.md](docs/GUANGHUA_EXPERIMENT_PLAN.md): midterm-oriented experiment design.
- [docs/PILOT_RESULTS.md](docs/PILOT_RESULTS.md): frozen five-seed mechanism results.
- [docs/QWEN_VLM_RESULTS.md](docs/QWEN_VLM_RESULTS.md): primary real-Qwen 20-seed main results.
- [docs/CHALLENGE_V2_RESULTS.md](docs/CHALLENGE_V2_RESULTS.md): upgraded 100-rollout result and interpretation boundary.
- [docs/AGENT_TOOLS_V3_RESULTS.md](docs/AGENT_TOOLS_V3_RESULTS.md): final 100-rollout Agent-tool results, statistics, and audit boundary.
- [docs/COMPLETE_TASK_DEMO.md](docs/COMPLETE_TASK_DEMO.md): complete success-video evidence, reproduction, and claim boundary.
- [docs/DEFENSE_SUITE_RESULTS.md](docs/DEFENSE_SUITE_RESULTS.md): four-scenario Guanghua defense suite and presentation guidance.
- [docs/TRADITIONAL_VS_AGENTIC_RESULTS.md](docs/TRADITIONAL_VS_AGENTIC_RESULTS.md): fair baseline definitions and E1–E4 results.
- [docs/RECOVERY_HIERARCHY_RESULTS.md](docs/RECOVERY_HIERARCHY_RESULTS.md): failure-correction experiment and evidence boundary.
- [docs/EXPERT_ASSESSMENT_20260830.md](docs/EXPERT_ASSESSMENT_20260830.md): defense readiness, limitations, and likely questions.
- [docs/CURRENT_EXPERIMENT_PROGRESS.md](docs/CURRENT_EXPERIMENT_PROGRESS.md): current quantitative evidence, complete VLABench tasks, claim boundary, and defense structure.

## Environment contents

- fixed Guanghua upper-body robot with two five-finger hands;
- 0.8 m × 0.8 m table, 0.8 m surface height, and collision-enabled legs;
- floor and visual room walls;
- a red cube, blue cylinder, translucent yellow protected-glass vessel, two
  target zones, and a staging zone;
- `frontview`, `birdview`, `agentview`, and moving `right_wrist` cameras;
- position actuators for the head, waist, both 7-DoF arms, and 22 hand joints;
- simplified collision proxies for the palms, finger segments, and fingertips.

The collision proxies are deliberately simple. Arm-link collision is currently
disabled because uncalibrated capsules incorrectly blocked reachable tabletop
motions; it must be added from measured link geometry before safety claims. The
cylindrical task objects use stable box-shaped collision proxies while retaining
cylindrical visual geometry. They are appropriate for Agentic mechanism
qualification, but should be calibrated or replaced before making fine
contact-dynamics claims.

The arm executor uses multi-start IK with Cartesian-axis and continuity
constraints plus model-based gravity feed-forward. Hand actuators use compliant
position control (`kp=30`, force range ±8) and padded contact margins. These
choices are generated by `scripts/prepare_assets.py` and appear in the public
execution trace rather than being hidden simulator interventions.

## Setup

```bash
# Full simulation experiment stack
pip install -r requirements.txt

# Core decision framework only (no MuJoCo; platform deployment machines)
pip install -e .
```

Python 3.10+ is required. The verified dependency combination is recorded in
`requirements.txt`; the core/sim/dev dependency split is declared in
`pyproject.toml`.

## Tests in a clean environment

`pytest -q tests` runs the pure framework, runtime, and evaluation-contract
regression tests on any machine. Tests that load the Guanghua MJCF/URDF or
import MuJoCo-dependent modules skip with an explanatory reason when the
school asset bundle is absent (it is not distributed, see the asset note
above) or when no MuJoCo runtime can initialize (for example on some headless
Windows setups). A skip is an environment limitation, not a protocol change;
on a machine with the assets and a working MuJoCo runtime the full suite runs
as before.

## Prepare and verify (requires locally obtained school bundle assets)

`scripts/prepare_assets.py` regenerates the canonical MJCF and URDF from the
school-provided `guanghua_hand_env.bundle.xml` source files, which are not
distributed here. With those assets in place under `assets/guanghua_hand_env/`:

```bash
python scripts/prepare_assets.py
pytest -q tests/test_guanghua_assets.py
```

## Run

Challenge v4 contains 100 method rollouts in a paired frozen-seed design and
246 real multimodal calls. It varies object geometry, fragile-object
distance/bearing, instruction wording, online event type, mass, friction, pose,
and yaw. It includes no-change and irrelevant-change controls, does not provide
G2's answer offset, and records all bounded schema/consistency repairs as cost.
An independent Qwen routing turn selects the required tools before planning;
scene labels do not enter the runtime dispatcher. The Full Agent executes three
public-observation-only contracts for RAG-to-skill routing,
protected-relation motion constraints, and role-aware change/memory
verification. It passes G1–G4 at 10/10 each with 100% tool-selection accuracy;
the audit passes all 100 run directories, 260 public events and 60 tool calls
with zero endpoint errors. See [docs/AGENT_TOOLS_V4_RESULTS.md](docs/AGENT_TOOLS_V4_RESULTS.md).
Challenge v3 is retained as the pre-explicit-router comparison.

The canonical Challenge v4 result must not be overwritten. For a new run, use a
new port, a truthful BF16/NF4 deployment label, a new server receipt, and a
dated output directory.

The earlier real-Qwen pilot remains reproducible as a mechanism diagnostic:

```bash
MUJOCO_GL=egl python scripts/run_qwen_vlm_pilot.py \
  --output output/qwen35_nf4_main --seed-split main \
  --endpoint http://127.0.0.1:18110/v1/chat/completions \
  --model /home/admin1/g2_multimodal_agent/models/Qwen3.5-4B \
  --deployment-label qwen3.5-4b-bnb-nf4-local
python scripts/summarize_qwen_vlm_pilot.py output/qwen35_nf4_main
python scripts/validate_agentic_artifacts.py output/qwen35_nf4_main
```

The v1 study made 140 real multimodal calls with 100% valid JSON and no
endpoint failures. Full passed 20/20 in G1, G2, and G3; each isolated ablation
passed 0/20. Every paired McNemar exact test gives `p=1.90735e-6`. See
[docs/QWEN_VLM_RESULTS.md](docs/QWEN_VLM_RESULTS.md). Its 20/20 versus 0/20
result should not replace the harder Challenge v4 in the main defense table.
Both are still high-level mechanism experiments, not physical pick success.

The deterministic adapter remains as a fast offline diagnostic:

```bash
MUJOCO_GL=egl python scripts/run_agentic_pilot.py --output output/agentic_pilot
python scripts/summarize_agentic_pilot.py output/agentic_pilot
python scripts/validate_agentic_artifacts.py output/agentic_pilot
```

Its output in [docs/PILOT_RESULTS.md](docs/PILOT_RESULTS.md) must not be
substituted for the real-Qwen result in a paper or presentation.

## Guanghua environment commands (require local assets)

Headless one-second physics check:

```bash
python scripts/run_guanghua_env.py --duration 1
```

Render the front camera:

```bash
MUJOCO_GL=egl python scripts/run_guanghua_env.py \
  --scene G2 --seed 13 --duration 1 \
  --camera agentview --render output/g2_agent.png
```

Open the interactive viewer on a desktop session:

```bash
python scripts/run_guanghua_env.py --viewer
```

Run the first physical admission gate:

```bash
MUJOCO_GL=egl python scripts/run_g0_admission.py \
  --seed 7 --output output/g0_admission_seed7
```

The admission writes `public_trace.jsonl` separately from
`private_evaluator.json`. The controller target is estimated from rendered
RGB-D; MuJoCo object poses are used only by the private evaluator. Current
frozen result for seed 7: G0-A (RGB-D → safe physical pregrasp) passes with
sub-millimetre end-effector error and no non-support contact with the fragile
proxy. G0-B (contact closure → lift) is deliberately not marked as passed.
See [docs/GUANGHUA_CALIBRATION_STATUS.md](docs/GUANGHUA_CALIBRATION_STATUS.md).
Full task success rates must not be reported until G0-B passes without a
MuJoCo reset.

## Traditional vision + IK paired comparison

The comparison includes both a fixed centroid+IK baseline and a stronger
reactive geometry+IK baseline. The latter is allowed a primitive-specific
grasp heuristic, a globally tuned fixed wrist yaw, and visual replanning on any
scene change. This avoids attributing gains to an intentionally weak
comparator.

```bash
python scripts/run_traditional_vs_agentic.py
python scripts/render_paired_comparisons.py
```

The frozen E1–E4 results, the exact role of the translucent yellow protected
glass, negative controls, and claim boundary are documented in
[docs/TRADITIONAL_VS_AGENTIC_RESULTS.md](docs/TRADITIONAL_VS_AGENTIC_RESULTS.md).

## Failure-correction hierarchy

The recovery experiment compares no recovery, one unchanged retry, and the
Agentic L1 parameter retry / L2 method switch / L3 full-replan hierarchy on
three recoverable failures plus no-failure and timeout controls:

```bash
python scripts/run_recovery_hierarchy_experiment.py
```

Frozen recoverable-failure results are T0 `0/3`, T1 `1/3`, and Agentic `3/3`.
L1/L2 are explicitly disclosed as controlled verifier fault injection; the L3
segment uses a paired MuJoCo target-displacement video. This experiment tests
recovery-state selection, not contact-dynamics grasp success. See
[docs/RECOVERY_HIERARCHY_RESULTS.md](docs/RECOVERY_HIERARCHY_RESULTS.md).

## Presentation artifacts

Rendered rollouts, defense reels, and captioned videos (Guanghua defense
suite, complete-task demo, VLABench 84-second reel, recovery evidence reel)
are not distributed in this repository. Their evidence contracts, reproduction
commands, and claim boundaries are documented in
[docs/COMPLETE_TASK_DEMO.md](docs/COMPLETE_TASK_DEMO.md),
[docs/DEFENSE_SUITE_RESULTS.md](docs/DEFENSE_SUITE_RESULTS.md), and
[docs/CURRENT_EXPERIMENT_PROGRESS.md](docs/CURRENT_EXPERIMENT_PROGRESS.md).
The chemistry source reaches the benchmark goal after the second forward pour,
but the executor stops after 14 of 20 planned stages; this remains an open
cleanup item and must be resolved before the reel is described as a fully
cleaned-up safety sequence.

## Modeling boundary

This environment is intended for high-level Agentic RAG-VLM evaluation with an
IK / skill-based low-level executor. It does not contain or require a trained
VLA policy. MuJoCo private body poses and contacts should remain evaluator-only;
the agent should consume rendered RGB-D, proprioception, gripper state, and
structured experience memory.

## Citation

Paper: "Agentic RAG-VLM: Affordance-Aware Retrieval-Augmented Generation with
Self-Reflective Planning for Robotic Grasping", IROS 2026. A formal BibTeX
entry will be added with the camera-ready proceedings information.
