# BluPe integration and first hardware trial

## Current state

The official Runner and Session API integration has completed local software and MuJoCo qualification. The provider can run on our machine while connecting to BluPe over the network. Public hardware execution remains at the coordinated first-trial stage.

The official Runner owns queue admission, session leases and Stop. Agentic RAG-VLM implements `run_session`, perception, retrieval, planning, native IK, command submission and outcome verification. Simulator force/hand-shape parameters retain their simulation provenance. The station's native joint/gripper interface uses measured site settings.

## Software preparation

Start the model services and install the pinned Runner using [Getting started](GETTING_STARTED.md). Generate private site templates:

```bash
python scripts/prepare_blupe_hardware.py \
  --platform-config configs/blupe_platform.example.json \
  --output local/blupe
```

Choose a fresh output directory. Optional `--capture` performs public metadata and image GET requests. The generated files contain pending measurements. `local/` is ignored by Git.

Check software/model compatibility:

```bash
python scripts/check_platform_provider.py \
  --config local/blupe/site/provider.monitor.local.json \
  --runner-source "$BLUPE_RUNNER_CHECKOUT/codex-runner/src/remote_yam" \
  --output local/output/provider_readiness.json
```

## Read-only station monitor

```bash
python scripts/serve_blupe_hardware.py \
  --runner-checkout "$BLUPE_RUNNER_CHECKOUT" \
  --config local/blupe/site/provider.hardware.local.json \
  --output local/output/hardware_monitor --port 18797
```

Open `http://127.0.0.1:18797/`. The default monitor supports pending site files and blocks Run before queue admission. It is suitable for inspecting the chosen station during a setup call.

## Site fields to confirm

| Field group | Required information |
| --- | --- |
| Station and arm | Selected `robot_id`, active arm, operator contact and held pose of the other arm |
| Calibration | Current calibration ID, camera intrinsics/extrinsics, native FK, tool frame and independent physical validation record |
| RGB source | Matching JPEG calibration or paired factory RGB-D profile, original resolution, alignment and timestamps |
| Table and objects | Support plane, observed feature heights, grasp offsets and object registry |
| Destinations | Named feature coordinates in the selected arm base, in metres |
| Motion | Measured joint limits, approved workspace, joint delta at 10 Hz, lift height and camera-clear observation pose |
| Gripper | Measured maximum aperture and blocked-aperture/lift retention thresholds |
| Image proposals | Site-checked ROI and HSV bands, visibility after placement and handling of unknown regions |
| Operation | Readiness, initialization/Home, Stop and first-trial observation workflow |

The generated provider paths resolve relative to their configuration files. When moving a site directory, regenerate its templates or verify every referenced file. A successful software precheck and the public calibration package have different roles from independent physical calibration.

## First trial

1. Confirm the selected station, one active arm, operator readiness and initialization workflow.
2. Check fresh images and site geometry. Use one registered rigid object and a nearby measured destination.
3. Preview the plan and inspect the grasp frame, units, workspace and approach/retreat.
4. After site review, launch the same monitor command with `--enable-hardware-run`. Clicking Run requests the actual session through the official Runner.
5. Wait for fresh homed/settled feedback. Verify grasp, lift, transfer, release and retreat with terminal execution feedback and operator observation.
6. Retain all outcomes. Add a protected neighbor and a second compatible target after the first task is accepted.

The joint/gripper retention proxy needs site measurements. Full-arm/bimanual collision qualification, automatic post-contact regrasp and physical damage-prevention claims require further hardware validation. Uncertain contact outcomes stop by default.

## Other interfaces

- `scripts/run_platform_session.py` provides a standalone task CLI. Its native planner uses the same measured settings as the official Runner path.
- `scripts/serve_platform_policy.py` exposes the non-actuating HTTP policy/preview interface described by [the OpenAPI schema](../configs/platform_policy.openapi.json).
- [The official Session API documentation](https://blupe.io/docs/robot-api/) and [shared Runner](https://github.com/andlyu/blupe-remote-bots) define the external transport contract. The tested checkout is pinned in the startup guide.
