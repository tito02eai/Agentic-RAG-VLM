# Reproduction and assets

## Versions

The simulator uses Python 3.10 and the exact dependency pins in `configs/extended_sim_dependencies.txt`. MuJoCo is pinned to 3.2.2. `ffmpeg` is required for recorded video. The official Runner pin is `e6040394c5ed62efbcc7d627acc0dfc8ccc100a6`.

Model inference is provisioned separately. The recorded configuration uses Qwen3.5-4B NF4 and CLIP ViT-L/14. Weight fingerprints are saved by the services and checked in the evidence audit. See [Getting started](GETTING_STARTED.md).

## Robot assets

The YAM model and source geometry required for local contact execution are included in `assets/yam_mjlab/` with their upstream license. The original collision geometry remains active in the local physics scene. Link gravity compensation is part of the declared simulator setup.

### Asset note

The historical school-provided humanoid URDF/mesh bundle is retained in the local archive and is excluded from the public source distribution. Ten asset-dependent regression cases skip when that bundle is absent. Some geometry/perception compatibility tests remain executable without it. Two recorded-image checks require explicitly configured archived previews and are opt-in.

The maintained test suite keeps these skip reasons distinct from executed validation. Legacy helpers retained under `scripts/` support these tests and older data readers; use the [script index](../scripts/README.md) for the current launchers.

## Knowledge provenance

`knowledge_base/yam_verified_diverse_r8.json` contains 14 engineering records. Referenced images and prior source records remain available for runtime or regression readers. The store's CLIP embedding identity must match the encoder service. New hardware experience should use its own execution-domain and calibration identity.

## Re-run and archive

Use fresh output directories and preserve every outcome. Before a qualification batch, freeze source, knowledge, model identity, renderer, task layouts and acceptance rules. Report software-test counts, task completion and expected stopping criteria separately.

Build a source-only package with:

```bash
python scripts/build_delivery_bundle.py --output output/Agentic_RAG_VLM_source.zip
```

The package includes a per-file SHA256 manifest, current code/configuration, knowledge, robot assets, paper files and selected demonstrations. Site-local files, credentials, model weights and large raw experiment archives stay in their own local storage.
