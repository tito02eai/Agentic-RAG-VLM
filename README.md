# Agentic RAG-VLM

**Affordance-Aware Retrieval-Augmented Generation with Self-Reflective Planning for Robotic Grasping**

IROS 2026 accepted paper · First author Tao Chen · Fudan University

[中文说明](README.zh-CN.md) · [Paper PDF](paper/main_final.pdf) · [Paper source](paper/main_final.tex) · [Quick start](docs/GETTING_STARTED.md) · [Demos](examples/blupe/README.md)

Agentic RAG-VLM connects task understanding to robot execution through experience retrieval, spatial reasoning and feedback. The agent retrieves compatible grasp experience, checks scene constraints, reviews the action with a vision-language model, executes a skill, verifies the outcome and updates memory.

```text
Observe -> HAA-RAG + scene constraints -> Plan -> VLM review -> Execute
   ^                                                         |
   +------------ Monitor <- Verify <- execution feedback -----+
                         |
                  Memory / bounded replan
```

## What is available

| Component | Implementation |
| --- | --- |
| HAA-RAG | Category matching, four-dimensional affordance compatibility and CLIP visual reranking |
| Scene reasoning | Spatial relations and object states compiled into grasp constraints and task dependencies |
| Agent runtime | Observe, plan, execute, verify, monitor and bounded replan |
| Memory | Reuse verified strategies, observe fresh positions and invalidate failed reuse |
| Robot execution | Native YAM IK and joint/gripper trajectories through the official BluPe Runner and SDK |
| Local validation | Actuator-driven MuJoCo grasp/place, protected neighbors, moved-target memory reuse and fault handling |

The current engineering knowledge store contains 14 provenance-labelled records and source images. The engineering experiments have their own [validation record](docs/VALIDATION.md), separate from the paper's historical experiments.

## Quick start

The tested simulator environment uses Linux, Python 3.10, MuJoCo 3.2.2 and EGL. Model inference runs in a separate environment.

```bash
git clone https://github.com/tito02eai/Agentic-RAG-VLM.git
cd Agentic-RAG-VLM
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r configs/extended_sim_dependencies.txt
python -m pip install -e '.[runner,dev]'

# Pinned official Runner used by the integration tests.
git clone https://github.com/andlyu/blupe-remote-bots.git third_party/blupe-runner
git -C third_party/blupe-runner checkout e6040394c5ed62efbcc7d627acc0dfc8ccc100a6
export BLUPE_RUNNER_CHECKOUT="$PWD/third_party/blupe-runner"
export MUJOCO_GL=egl
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m pytest -q tests
```

Install system EGL/OpenGL libraries and `ffmpeg` as described in the [installation and model-service guide](docs/GETTING_STARTED.md). The offline tests can run before starting the model services.

### Run a complete grasp task

Start the Qwen and CLIP services using the [model setup instructions](docs/GETTING_STARTED.md#model-services), then run:

```bash
python scripts/check_platform_provider.py \
  --config configs/blupe_provider.example.json \
  --runner-source "$BLUPE_RUNNER_CHECKOUT/codex-runner/src/remote_yam"

python scripts/run_blupe_protocol_task.py \
  --runner-checkout "$BLUPE_RUNNER_CHECKOUT" \
  --config configs/blupe_provider.example.json \
  --candidate-config configs/blupe_local_candidates.example.json \
  --output output/first_grasp --seed 1040
```

The production provider and official HTTP/WebSocket SDK drive our local MuJoCo backend. Outputs include `receipt.json`, raw model replies, command packets, physical checks and `physics/rollout.mp4`. Each run requires a new output directory.

### Watch the current demonstrations

| Demonstration | Video | Execution receipt |
| --- | --- | --- |
| Grasp, lift, place and retreat with a protected neighbor | [16.08 s](examples/blupe/local_pick_place.mp4) | [Receipt](examples/blupe/local_pick_place_receipt.json) |
| Two objects, moved-target replanning and memory reuse | [29.80 s](examples/blupe/local_memory_replan.mp4) | [Receipt](examples/blupe/local_memory_replan_receipt.json) |

Videos follow simulation time and omit model waiting. The second demonstration includes a declared external-force disturbance to the pending object. See the [demo guide](examples/blupe/README.md) for the sequence and provenance.

## Current validation

| Check | Recorded outcome |
| --- | --- |
| Software regression on local GPU, Mesa and GitHub CI | 379 passed, 12 skipped at validation commit `440a40e` |
| Local task qualification | 3 layouts × 2 renderers, all 6 grasp/place tasks completed |
| Partial tabletop-ROI observation | Rejected before motion |
| Ten memory, communication and Stop criteria | 4 tasks completed and 6 expected stops |
| Real Qwen replies retained in the qualified cohorts | 60 |
| Public BluPe hardware trial | Pending measured site setup and a coordinated first trial |

The [selected receipt](examples/blupe/pre_hardware_20261108.json) fixes the original validation scope and runtime hashes. Repository reorganization has a separate regression check. The front end currently uses registered semantics, controlled RGB candidates, known feature heights and a measured support plane. Hardware calibration, physical gripper settings and full-arm/bimanual collision qualification require site validation.

The separate [2026-11-09 integration review](examples/blupe/integration_review_20261109.json)
records 384 passed and 12 skipped local regression checks. It covers concurrent
result/video downloads, per-target lift/release verification, a complete browser
task and the ten memory/fault criteria. Earlier qualification remains recorded
at its original revision.

## Project layout

```text
agentic_rag_vlm/   Core retrieval, reasoning, memory, runtime and adapters
  platform/       BluPe SDK, native motion, calibration and verification
  simulation/     MuJoCo physics, RGB-D geometry and protocol gateway
configs/          Runnable local examples and pending hardware templates
knowledge_base/   Provenance-labelled engineering experience and images
scripts/          Launchers, services, qualification and audit commands
tests/            Unit, interface and physical regression checks
examples/blupe/   Two current demonstrations and selected receipts
paper/            Accepted PDF, LaTeX source, bibliography and figures
docs/             Installation, architecture, simulation and BluPe guides
```

Private site files, collaboration records and generated outputs use the ignored `local/` directory. The official Runner is installed separately under ignored `third_party/`. Historical deliveries, earlier videos, drafts and large raw experiment archives are retained locally. The repository's Git history preserves earlier public revisions.

## Documentation

- [Getting started](docs/GETTING_STARTED.md) — installation, model services and first run.
- [Architecture](docs/ARCHITECTURE.md) — paper mechanisms and their code locations.
- [Simulation](docs/SIMULATION.md) — CLI, browser UI, batch checks and output interpretation.
- [BluPe integration](docs/BLUPE.md) — read-only monitoring, measured configuration and first hardware setup.
- [Validation](docs/VALIDATION.md) — exact evidence, task outcomes and limits.
- [Reproduction and assets](docs/REPRODUCTION.md) — versions, knowledge provenance and optional historical assets.
- [Paper and citation](paper/README.md) — manuscript files and BibTeX.

Code is distributed under [Apache-2.0](LICENSE). Robot assets retain their [upstream license](assets/yam_mjlab/LICENSE). Publication rights and model-weight licenses are described in the relevant guides.

### Asset note

The included YAM geometry supports the current local tasks. Historical humanoid
asset bundles are stored locally; their dependent regression cases skip in a
public checkout. See [asset and skip details](docs/REPRODUCTION.md#asset-note).
