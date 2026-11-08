# Script entry points

## Current workflows

| Purpose | Command |
| --- | --- |
| Check models and official Runner | `check_platform_provider.py` |
| Start Qwen / CLIP | `serve_local_vlm.py`, `serve_clip.py` |
| Download the pinned CLIP weights | `download_clip.py` |
| Run a complete official-SDK local grasp | `run_blupe_protocol_task.py` |
| Open the official local simulation UI | `serve_blupe_recommended.py` |
| Run ten memory/fault criteria | `run_blupe_robustness_batch.py` |
| Run one memory/fault case | `run_blupe_robustness_case.py` |
| Audit that qualification | `audit_blupe_robustness.py` |
| Generate pending site files / public GET capture | `prepare_blupe_hardware.py` |
| Open the default read-only station monitor | `serve_blupe_hardware.py` |
| Standalone task CLI | `run_platform_session.py` |
| Non-actuating HTTP policy/preview server | `serve_platform_policy.py` |
| Source-only archive with SHA256 manifest | `build_delivery_bundle.py` |

Commands run from the repository root. See [Getting started](../docs/GETTING_STARTED.md), [Simulation](../docs/SIMULATION.md) and [BluPe](../docs/BLUPE.md) for arguments and outputs.

## Diagnostics and compatibility

The retained YAM/extended-physics diagnostics, paper recovery contracts, audit readers and regression helpers support reproducibility and tests. `agentic_framework.py` provides shared representations still imported by the core package. Some humanoid-era geometry and perception helpers remain dependencies of compatibility tests. Their original assets are kept in local archival storage.

Earlier rendering scripts, obsolete preview servers and one-off experiment/report generators have been archived locally. The source Git history preserves their former public locations.
