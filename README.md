# Agentic RAG-VLM

**Affordance-Aware Retrieval-Augmented Generation with Self-Reflective Planning for Robotic Grasping** · IROS 2026

Agentic RAG-VLM combines affordance-aware experience retrieval, scene constraints,
VLM action review, execution feedback and episodic memory in an
**Observe → Plan → Execute → Verify → Monitor → Replan** loop.

This repository contains the reconstructed framework and its new engineering
validation. The accepted paper's historical measurements are separate from these
experiments. **Version 0.3.5 prepares a BluPe hardware integration; no public
BluPe hardware task has been executed by this implementation.**

[Paper](paper/main_final.pdf) · [Method reference](paper/main_final.tex) ·
[BluPe first-trial guide](docs/BLUPE_HARDWARE_TRIAL.md) ·
[Local demos and verification](examples/blupe/README.md)

## Framework

- **HAA-RAG:** category matching and compatibility of affordance type, material,
  fragility and grasp region, followed by CLIP visual ranking.
- **Scene reasoning:** spatial relations and observed states become execution
  constraints and task dependencies.
- **Agent runtime:** VLM review, bounded replanning, fresh observations and
  conservative stopping after uncertain contact failures.
- **Memory:** reuse verified grasp strategies between compatible objects,
  reobserve current positions, and invalidate disproven experience reuse.
- **Execution:** YAM/MuJoCo contact physics and BluPe native joint/gripper
  trajectories. The framework orchestrates skills and IK without VLA training.

The 14-entry engineering knowledge store includes source images and execution
provenance. It is simulation-origin knowledge, not the paper-era store or
calibrated hardware force control. Model weights are not bundled.

## Install and check

Use Linux and Python 3.10. Keep the source checkout: knowledge, configuration and
robot assets accompany the editable installation.

```bash
git clone https://github.com/tito02eai/Agentic-RAG-VLM.git
cd Agentic-RAG-VLM
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r configs/extended_sim_dependencies.txt
python -m pip install -e '.[runner]'
export MUJOCO_GL=egl
python -m pytest -q
```

MuJoCo 3.2.2 is pinned for contact-experiment reproduction. The offline suite
needs no model credentials or robot session. Historical asset and opt-in checks
may skip; skips are not executed validation. Start Qwen3.5-4B NF4 and CLIP ViT-L/14
separately using the [model service guide](docs/CMU_EVALUATION.md).

## Start here for BluPe

Our process can run locally and connect through the official Session API. Clone
the tested official Runner separately:

```bash
git clone https://github.com/andlyu/blupe-remote-bots.git ../blupe-remote-bots
git -C ../blupe-remote-bots checkout e6040394c5ed62efbcc7d627acc0dfc8ccc100a6
export BLUPE_RUNNER_CHECKOUT="$(cd ../blupe-remote-bots && pwd)"
python -m pytest -q

# Optional public metadata/camera GETs only; no queue or motion.
python scripts/prepare_blupe_hardware.py \
  --platform-config configs/blupe_platform.example.json \
  --output output/site_preparation --capture

# Requires separately running Qwen and CLIP services.
python scripts/check_platform_provider.py \
  --config output/site_preparation/site/provider.monitor.local.json \
  --runner-source "$BLUPE_RUNNER_CHECKOUT/codex-runner/src/remote_yam" \
  --output output/provider_readiness.json

python scripts/serve_blupe_hardware.py \
  --runner-checkout "$BLUPE_RUNNER_CHECKOUT" \
  --config output/site_preparation/site/provider.hardware.local.json \
  --output output/hardware_monitor --port 18797
```

Open `http://127.0.0.1:18797/`. The default monitor blocks **Run** before queue
admission and works while measurements are pending. Choose the actual rig in the
platform config before generating its site files.

The [first-trial guide](docs/BLUPE_HARDWARE_TRIAL.md) covers measured geometry,
camera-stream pairing, native joints, workspace limits, gripper feedback and the
later explicitly enabled hardware Run. Simulator coordinates do not qualify the
pending templates. Hardware execution and public provider registration remain
separate from this software handoff.

## Complete local grasp demonstration

After starting the model services:

```bash
python scripts/run_blupe_protocol_task.py \
  --runner-checkout "$BLUPE_RUNNER_CHECKOUT" \
  --config configs/blupe_provider.example.json \
  --output output/protected_neighbor --seed 1010
```

The production provider and official HTTP/WebSocket SDK drive a local MuJoCo
Session API gateway. Joint actuators drive the physics; no object is attached or
teleported. Outputs include raw images/model replies, trajectories, independent
contact/placement checks, memory and `physics/rollout.mp4`.
**This is our local simulator, not BluPe's public hardware or a calibrated twin.**

## Validation and limits

| Check | Recorded result | Scope |
|---|---|---|
| 0.3.5 full regression | 354 passed, 12 skipped | Local environment with pinned official runner |
| Additional current-runner subset | 24 passed, 2 skipped | Separate recorded revision |
| 0.3.4 robustness | 10 criteria passed: 4 task completions, 6 expected stops | Functional checks, not a grasp success rate |
| Saved real camera image/packet | Identity and stream/calibration contracts checked | No hardware metric-accuracy or grasp validation |

[Verification receipt](examples/blupe/verification_0.3.5.json) ·
[Fault and memory report](docs/BLUPE_ROBUSTNESS_20261104.md) ·
[Recommended-task protocol](docs/RECOMMENDED_TASK_QUALIFICATION.md)

Selected examples do not replace the complete frozen raw-case archives. Preserve
failures and declare new cohorts before changing methods or retrying cases.

The current front end uses registered semantics, controlled RGB proposals, known
dimensions and a measured horizontal support plane. Full-arm/bimanual collision
safety, physical calibration accuracy, damage prevention and automatic
post-contact regrasp have not been validated on hardware. The paired RealSense
adapter admits the observed zero-distortion factory profile; other factory
models are rejected. Depth is retained in paired packets, while current metric
grounding uses the measured plane. Video timing excludes model inference waits.

## Repository map

| Path | Purpose |
|---|---|
| `agentic_rag_vlm/` | Retrieval, reasoning, memory, runtime and adapters |
| `configs/` | Examples and intentionally incomplete site templates |
| `knowledge_base/` | Engineering knowledge with source provenance |
| `assets/yam_mjlab/` | Licensed YAM simulation model and meshes |
| `scripts/` | Model services, experiments, audits and hardware preparation |
| `tests/` | Offline contracts and opt-in integration checks |
| `examples/blupe/` | Local videos and selected verification receipts |
| `docs/` | Setup, experiment protocols and historical reports |
| `paper/` | Final paper, its figures and method reference |

Code is Apache-2.0; third-party assets retain their
[license and provenance](assets/yam_mjlab/PROVENANCE.md). Model weights are obtained
separately under their own licenses.
