# Agentic RAG-VLM: software evaluation quick start

Release 0.3.0, 2026-11-03. For platform deployment now, start with [PLATFORM_DEPLOYMENT.md](PLATFORM_DEPLOYMENT.md): it uses a lightweight backend and official runner provider, requiring no active robot for installation/registration/preview. The sections below also explain simulation reproduction. No live robot session has been created by this work.

## 1. Start in the extracted source directory

Use Linux, Python 3.10 (measured version 3.10.21), a working EGL implementation for headless rendering, and `ffmpeg` on PATH. Keep the extracted source tree: the supported setup is an editable source installation, with the bundled knowledge/configuration/robot assets alongside the code.

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r configs/extended_sim_dependencies.txt
python -m pip install -e . --no-deps
export MUJOCO_GL=egl
python -m pytest -q
python scripts/run_delivery_acceptance.py --output output/offline_acceptance
```

The pinned file selects MuJoCo **3.2.2**. Installing only broad extras can select 3.17; this is not the solver version used in the measured experiments. Historical Guanghua asset tests are expected to skip. Offline acceptance uses synthetic observations and validates workflow contracts; its expected stops are not physical grasp successes.

## 2. Provision the two model services separately

Weights are not bundled. The measured setup uses Qwen3.5-4B NF4 and CLIP ViT-L/14. Keep GPU-model dependencies in a separate environment from the pinned simulator. The Qwen helper assumes CUDA device 0 and a compatible NVIDIA/PyTorch/BF16 setup; it is not a CPU server. Tested server versions are torch 2.11.0+cu129, transformers 5.13.1 and bitsandbytes 0.49.2. The source `serve` extra also declares Accelerate, required by model device mapping.

```bash
# In a separately provisioned model environment:
python -m pip install -e '.[vision,serve]'
python scripts/serve_local_vlm.py --model-path /your/models/Qwen3.5-4B \
  --port 18074 --quantization nf4

# In another terminal, with vision dependencies:
python scripts/download_clip.py --output /your/models/clip-vit-large-patch14
python scripts/serve_clip.py --model-path /your/models/clip-vit-large-patch14 \
  --port 18072 --device cpu
```

The CLIP downloader pins its public revision. Its weight fingerprint must match the supplied knowledge embeddings; a mismatched encoder is rejected. Do not rename a different model or relabel existing vectors to pass checks. Measured Qwen and CLIP identities are retained in the evidence receipts.

## 3. Check services before starting an experiment

```bash
export AGENTIC_VLM_ENDPOINT=http://127.0.0.1:18074/v1/chat/completions
export AGENTIC_CLIP_ENDPOINT=http://127.0.0.1:18072
# Optional: export AGENTIC_VLM_MODEL=/your/models/Qwen3.5-4B
python scripts/check_sim_readiness.py --output output/sim_readiness.json
```

If the model variable/`--model` is absent, the helper reads the actual checkpoint identity from the inference service's `/health`. It does not use the developer's home directory. The campaign runner accepts and forwards `--endpoint`, `--model` and `--clip-endpoint`. These checks make no hardware requests.

## 4. Run a complete recorded grasp task

```bash
python scripts/run_extended_sim.py --output output/first_complete_task \
  --store knowledge_base/yam_verified_diverse_r8.json --seed 960 \
  --mode stacked_sort --controller-profile guarded \
  --scene-variation mixed_contact --mixed-shapes --recovery-policy force_ladder \
  --grounding-repair --multiccd --visibility-replanning \
  --visible-goal-fallback --upper-half-width .017 --record
```

Outputs include unedited `rollout.mp4`, `result.json`, public observations and model responses, tactile/command logs, independent physical evaluation and memory. Use a new output directory for each run. Declared failures remain failures; do not rerun until success and replace a study record. The study command/configuration is in `docs/CURRENT_DELIVERY.md`.

The latest small frozen cohort completed 4/4 with both full Agent and the strong geometric comparison. Full/geometric used 12/16 grasp attempts and 50/48 real model replies. Known repaired seeds are separate. These numbers are engineering evidence, not original-paper measurements or proof of broad superiority. Videos show simulation time, excluding inference waits.

## 5. Platform integration and later motion configuration

- `run_extended_sim.py` uses the full `PaperAgent`: HAA-RAG, actual CLIP, VLM action review, constraints and verified memory.
- `run_platform_session.py` now defaults to full PaperAgent remotely; `--policy-profile cards` is a legacy option and the metadata mock remains cards-only. The official runner adapter hosts the complete task loop inside the existing session. Software deployment does not join a queue.
- The 14-record store retains `mjlab_yam_retention_v1` provenance. Executor-bound retrieval marks it as a prior, uses separate robot/arm/calibration memory identity and labels unsupported force/synergy. It does not turn simulation motor limits into physical force commands.
- The native hardware adapter needs measured arm-base destinations, independently validated calibration, feature heights, grasp offsets, limits and an initialized station. The mock example is marked mock-only. The measured templates intentionally contain no invented coordinates.
- A supervised single rigid-object pick/place/release/retreat acceptance is still required before public hardware trials. Collision checks and observed displacement alone do not certify full-arm collision safety or physical damage prevention.

See `docs/PLATFORM_DEPLOYMENT.md` for software hosting/registration and `docs/BLUPE_HANDOFF.md` for the component map. The package is prepared for platform integration; activation on BluPe's server is a host-operator step. A later physical acceptance is not a prerequisite for delivering these interfaces.

## 6. What is included

The compact evaluation package contains current source/knowledge/assets, this English quick start, selected complete videos and frozen result/audit receipts. The larger r9 evidence archive retains every r8/r9 case, original failures, frozen experiment source and raw RGB-D/contact/evaluator data. Selected receipts alone are not a replacement for the full raw archive when independently rerunning all audits. Release portability fixes do not rewrite the frozen experiment results.
