# Configuration index

## Current startup files

| File | Role |
| --- | --- |
| `blupe_provider.example.json` | Model endpoints, knowledge store and registry for the production provider |
| `blupe_platform.example.json` | Public platform/API identity used to generate site templates |
| `blupe_local_candidates.example.json` | Runnable RGB candidate settings for the local MuJoCo camera |
| `extended_sim_dependencies.txt` | Exact simulator/regression dependency versions |
| `platform_policy.openapi.json` | Non-actuating HTTP policy/preview API schema |

## Hardware templates

`blupe_deployment.template.json`, `blupe_measured_task.template.json` and
`blupe_candidates.template.json` hold pending site measurements. Generate private
working copies using `scripts/prepare_blupe_hardware.py`; see [BluPe setup](../docs/BLUPE.md).
`blupe_stage1_task.example.json` is the metadata-mock task used by contract checks.

## Compatibility resources

The retained `guanghua_*`, `complete_task_demo_v2.json`,
`extended_sim_qualification.json` and `yam_physics_qualification.json` files
support preserved regression readers and diagnostics. Earlier one-off campaign
configurations are retained in the local historical archive.
