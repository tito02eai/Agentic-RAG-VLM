# Getting started

## Simulator environment

The tested setup uses Linux and Python 3.10. The contact-physics dependencies are pinned in `configs/extended_sim_dependencies.txt`.

```bash
sudo apt-get update
sudo apt-get install -y libegl1 libgl1 ffmpeg
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r configs/extended_sim_dependencies.txt
python -m pip install -e '.[runner,dev]'
export MUJOCO_GL=egl
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

Install the separately licensed official Runner for integration tests and browser interfaces:

```bash
git clone https://github.com/andlyu/blupe-remote-bots.git third_party/blupe-runner
git -C third_party/blupe-runner checkout e6040394c5ed62efbcc7d627acc0dfc8ccc100a6
export BLUPE_RUNNER_CHECKOUT="$PWD/third_party/blupe-runner"
python -m pytest -q tests
```

Keep the source checkout alongside the editable installation. Knowledge images, configuration and licensed robot geometry are loaded from that tree. The ordinary offline tests use recorded/synthetic contract inputs and local physics. Twelve historical or opt-in cases were skipped in the recorded 379-test qualification, as detailed in [reproduction](REPRODUCTION.md).

## Model services

Run model inference separately from the pinned simulator. The measured setup used Qwen3.5-4B with NF4 quantization and CLIP ViT-L/14. Qwen requires compatible NVIDIA CUDA hardware. Recorded server versions include torch `2.11.0+cu129`, transformers `5.13.1` and bitsandbytes `0.49.2`.

Create a separate model environment and install the service dependencies:

```bash
python3.10 -m venv .venv-models
source .venv-models/bin/activate
python -m pip install -e '.[vision,serve]'
```

Set `QWEN_MODEL_PATH` to a local Qwen3.5-4B checkpoint directory containing its configuration, tokenizer/processor and safetensors files. Install the PyTorch build appropriate for the machine's CUDA runtime before launching the GPU service.

Terminal A starts the VLM:

```bash
export QWEN_MODEL_PATH=/absolute/path/to/Qwen3.5-4B
python scripts/serve_local_vlm.py \
  --model-path "$QWEN_MODEL_PATH" --port 18074 --quantization nf4
```

Terminal B starts CLIP in the model environment:

```bash
python scripts/download_clip.py --output models/clip-vit-large-patch14
python scripts/serve_clip.py \
  --model-path models/clip-vit-large-patch14 --port 18072 --device cpu
```

The CLIP downloader pins the expected public revision. The supplied experience embeddings are tied to that encoder fingerprint. Services expose `/health`, including model identity and weight fingerprints. Their weights and generated caches remain outside Git.

## First complete task

Return to the simulator environment in terminal C:

```bash
source .venv/bin/activate
export BLUPE_RUNNER_CHECKOUT="$PWD/third_party/blupe-runner"
export MUJOCO_GL=egl
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python scripts/check_platform_provider.py \
  --config configs/blupe_provider.example.json \
  --runner-source "$BLUPE_RUNNER_CHECKOUT/codex-runner/src/remote_yam" \
  --output output/provider_readiness.json

python scripts/run_blupe_protocol_task.py \
  --runner-checkout "$BLUPE_RUNNER_CHECKOUT" \
  --config configs/blupe_provider.example.json \
  --candidate-config configs/blupe_local_candidates.example.json \
  --output output/first_grasp --seed 1040
```

Inspect `output/first_grasp/receipt.json` and `output/first_grasp/physics/rollout.mp4`. A successful task requires both framework completion and independent physical checks. Raw observations, model replies, trajectories and memory remain in the same run directory.

The default endpoints are `http://127.0.0.1:18074/v1/chat/completions` and `http://127.0.0.1:18072`. Configure the provider JSON when using other endpoints. A null `vlm_model` resolves the actual model from the service health response.

## Common startup issues

| Symptom | Check |
| --- | --- |
| EGL initialization fails | Confirm system EGL libraries and driver availability, and set `MUJOCO_GL=egl` before Python starts |
| Model service cannot be reached | Check ports 18074/18072 and their `/health` responses |
| Encoder/store mismatch | Use the CLIP revision and weight fingerprint expected by the supplied knowledge store |
| Output directory already exists | Choose a new directory to retain the earlier outcome |
| Official Runner cannot be imported | Export `BLUPE_RUNNER_CHECKOUT` to the pinned checkout |
| Hardware configuration is pending | Follow the measured site setup in [BluPe](BLUPE.md) |

For software rendering, select an EGL device that supports Mesa llvmpipe on the current host. EGL device indices are host-specific. Record the renderer and selected device when comparing runs.
