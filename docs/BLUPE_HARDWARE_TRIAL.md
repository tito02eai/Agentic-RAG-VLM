# First BluPe hardware integration trial — 0.3.5 preparation

Prepared 2026-11-04. This is a **software and setup handoff**, not a completed
hardware deployment. No public robot session or motion was requested.
The preceding 0.3.4 ten-case qualification remains a frozen local MuJoCo study:
four task completions and six expected stops, not a hardware success rate.

## Local acceptance update, 2026-11-08

The native Runner and CLI now share measured motion settings, with Cartesian
waypoints for contact approach, lift and retract. RGB candidates touching the
tabletop ROI boundary stop before metric grounding. Local GPU and Mesa software
regressions each passed 379 tests with 12 historical/optional skips.

The final-source task qualification completed all six runs of three layouts
across two renderers. The partial-ROI rejection passed with zero motion packets.
All ten separate fault/memory criteria passed, including moved-target memory
reuse, failed-memory invalidation, lost acknowledgements, feedback outage,
WebSocket reconnection and active Stop. Four fault-cohort tasks completed and
six stopped as required. The cohorts retained 60 actual Qwen replies, original
wire packets, contact logs, videos and frozen source manifests.

See the [selected receipt](../examples/blupe/pre_hardware_20261108.json).
The local workspace archive is `交付/20261108/pre_hardware_sim/`. Its preliminary
and final-source cohorts remain separate. The software/model/Runner precheck
passed and public metadata was refreshed using GET requests only.

For the first hardware call, fill and validate the pending site files, confirm
the chosen arm and operator workflow, then use the existing Run entry below.
The local candidate example is specific to the simulator's 640×480 view.
Use independently checked site image/ROI/HSV measurements on hardware.

Reproduce the local production-path check with running Qwen/CLIP services:

```bash
python scripts/run_blupe_protocol_task.py \
  --runner-checkout /path/to/blupe-remote-bots \
  --config configs/blupe_provider.example.json \
  --candidate-config configs/blupe_local_candidates.example.json \
  --output /new/native_task --seed 1040

python scripts/run_blupe_robustness_batch.py \
  --runner-checkout /path/to/blupe-remote-bots \
  --candidate-config configs/blupe_local_candidates.example.json \
  --output /new/native_faults
```

## What is ready on our side

- Full Agentic RAG-VLM provider, HAA-RAG/CLIP, scene constraints, VLM review,
  verification, memory and native joint planning through the official Runner.
- A real-station monitor using the official UI. By default it blocks Run before
  queue admission and wraps the SDK to prevent session creation/trajectory writes.
- A GET-only metadata/camera capture tool and intentionally pending site files.
- Configurable RGB region proposals, loaded by the production provider. These
  expose image regions, not identities or metric positions. VLM classifies them.
- Robo-house factory intrinsics are bound to the RGB inside the paired RGB-D
  packet, not its independently rectified public JPEG. The adapter verifies
  robot/camera/calibration ID, factory intrinsics, dimensions, sensor identity,
  alignment and timestamps. It supports the observed zero-distortion factory
  profile; other factory distortions remain rejected. The current controller
  uses measured-plane grounding, **not depth-guided grasping**.
- Site-configurable joint waypoint delta (at most the existing 3-degree bound
  at 10 Hz), lift height, observation park and measured gripper retention proxy.
- A saved Robo-house top-camera image was actually checked with Qwen3.5-4B:
  the blue block was identified and three other proposed regions were unknown.
  **This is one recorded-image identity check, not 3-D localization or a grasp.**
- The new RGB proposer completed a real-model local production-path pick/place
  smoke run with independent contact/release/destination checks. This is local
  MuJoCo, not the Robo-house hardware scene or a calibrated hardware twin.
- Full regression for the final preparation revision: **354 passed, 12 skipped**
  (10 missing historical school assets, two optional old recorded-image checks).

The model services can run on our machine during a joint trial. No partner
server installation, SSH credential, VLA training or weight upload is required
for this connection path. A persistent/public Playground entry is a separate
deployment decision to coordinate after the first hardware trial.

## Public read-only inspection

On 2026-11-04 both YAM endpoints returned `DISABLED`, `homed=false` and
`hardware_not_initialized`. Robo-house additionally returned `queue_ready=true`;
`yam-1` returned false. This can describe standby before session preparation,
so it is **not proof that neither station can be used**. Ask which rig and
Home/initialization workflow the operator recommends.

Robo-house top/left/right RGB requests succeeded. The `yam-1` top-camera read
returned HTTP 503 (no fresh image) in this snapshot; wrist RGB reads succeeded.
These are time-stamped observations, not permanent availability claims.
Both calibration packages were marked `experimental_requires_fresh_validation`.
We have not established their independent physical or depth accuracy.

## Suggested trial sequence

1. Confirm the rig, operating arm, operator contact, initialization/Home and Stop
   behavior. Use a known rigid block on a clear horizontal surface and a nearby
   marked destination. Existing objects are fine; no specific colors are required.
2. Confirm the site profile below. Reinspect current images and pixel proposals
   without joining a robot queue. Check the source AND destination remain inside
   the configured camera ROI; do not copy the saved-image ROI into an unreviewed
   live setup. Review unknown proposals, occlusion and same-color backgrounds.
3. With independently checked geometry, run non-actuating plan preview. Confirm
   frame/units, grasp feature, grasp offset, joint limits, table/workspace bounds,
   approach and retreat. Preview approval is not proof of collision-free hardware.
4. After operator readiness, explicitly enable hardware Run, then use the normal
   official Run/queue/preparing flow. Wait for fresh, homed/settled measured
   feedback before the provider submits motion. Do not require pre-queue Home
   if the station normally performs it during preparation.
5. Complete one grasp → lift → transfer → release → retreat. Accept only after
   terminal execution feedback, measured aperture/lift proxy, fresh stable visual
   placement and release/retreat verification. Have the operator confirm actual
   retention and placement for the first session. The proxy is not force sensing.
6. Repeat the same task in a few declared layouts, retaining every failure.
   Then add a protected rigid neighbor and a second same-category target to
   demonstrate constraints, task memory and scene-change replanning.

Contact failure or uncertain completion stops by default. Automatic post-contact
regrasp is not enabled. Simulator force/synergy priors are not calibrated hardware
force commands. TCP workspace checks do not certify full-arm/bimanual collision
safety; the first trial uses an operator-approved single-arm workspace.

## Site information to confirm together

| Site field | Source / meaning |
|---|---|
| `robot_id`, `arm` | Chosen rig and active arm; keep the other arm in its measured held pose |
| `calibration_id`, `physical_validation.record` | Current published package plus independent site validation evidence |
| Camera intrinsics/extrinsics, native FK | Already fetched from API; verify the selected package physically, use original image resolution, no extra joint shifts |
| `top_rgb_source` | `jpeg` for its matching advertised camera profile, or `paired_rgbd` for Robo-house factory RGB; do not mix rectified JPEG with raw factory intrinsics |
| `joint_limits_deg`, `workspace_bounds_m` | Operator-approved selected-arm limits and workspace |
| `max_joint_delta_deg_per_step` | Site-approved waypoint delta at 10 Hz; configured bound cannot exceed 3 degrees |
| `support_plane` | Horizontal support plane in the selected arm base; include any plate/support height |
| `feature_heights_m` | Height of the consistently observed object feature above that support plane |
| `grasp_offsets_base_m` | Observed feature → grasp frame offset, in the same arm base |
| `task.destinations` | Final observable object FEATURE coordinates in metres, not an unlabelled TCP point |
| `gripper.max_aperture_m` | Measured full aperture; API normalized open/closed conventions and feedback semantics |
| `retention_observation` | Measured blocked-aperture/lift thresholds suitable for the chosen object |
| `lift_height_m`, `observation_park_xyz_m` | Approved lift/clearance and post-release camera-clear pose |
| Registry / candidate configuration | Actual material/geometry and visual description; original-image ROI and reviewed HSV bands |

The generated files deliberately contain `null`/pending values. Public calibration
ID alone is not a physical validation certificate; do not fill the validation flag
from simulation or merely to satisfy a parser. The saved-image candidate example
is separate from the pending hardware configuration.

If independent validation is not already available, agree on a few surveyed,
non-collinear visible reference points on the actual support surface. Compare
their measured base-frame coordinates with pixel projections, retain residuals
and agree on tolerance based on the object's/gripper's clearance. Separately
confirm tool offsets, measured aperture/feedback and the approved workspace.
A camera fit residual alone does not validate the whole hardware execution setup.
These checks can be done jointly during the setup call with the operator.

## Commands

Use the existing Python 3.10 simulation/runner environment for local qualification.
For a hardware-only runner install `.[deploy,runner]`; Qwen and CLIP services are
separate processes. Clone the official runner at the tested revision
`e6040394c5ed62efbcc7d627acc0dfc8ccc100a6` and install its launcher dependencies.

An additional checkout of current HEAD `7ba2427ba960ad30af99c8fedd05c5196685295a`
was checked on 2026-11-04: controller, Session SDK, providers, Playground Python
and hosted HTML are byte-identical to the tested pin. Its separate official
runner/monitor subset passed 24 tests with two optional recorded-image skips.
The 354-test full regression above used the pinned checkout. Keep the pin for
reproduction; recheck a later HEAD rather than assuming future compatibility.

Generate a NEW preparation directory; optional capture performs public GET only:

```bash
python scripts/prepare_blupe_hardware.py \
  --platform-config configs/blupe_platform.example.json \
  --output /new/preparation --capture
```

Prepare/check models and runner without a robot session:

```bash
python scripts/check_platform_provider.py \
  --config /new/preparation/site/provider.monitor.local.json \
  --runner-source /path/to/blupe-remote-bots/codex-runner/src/remote_yam \
  --output /new/preparation/software_readiness.json
```

Open the read-only hardware UI even with pending site measurements:

```bash
python scripts/serve_blupe_hardware.py \
  --runner-checkout /path/to/blupe-remote-bots \
  --config /new/preparation/site/provider.hardware.local.json \
  --output /new/preparation/monitor --port 18797
```

Opening the page performs monitoring; Run is rejected. After site files are
measured, reviewed and completed, the same command with `--enable-hardware-run`
admits the configured fixed task. Starting that process still sends no motion;
**clicking Run creates a real queue entry**. The official Runner retains Stop.
This flag was not enabled during preparation. Models/geometry are checked before
queue admission; actual hardware readiness is checked when adopting the lease.

The older `run_platform_session.py` standalone CLI has a stricter pre-queue
homed-state check. For a rig that Homes during `preparing`, use the official
Runner entry above. No reset/operator-control endpoint is assumed.

For current raw station status, `check_blupe_readiness.py` remains GET-only.
Its `ready_for_session` is a conservative snapshot of motion conditions, not
a definitive admission decision for a station that initializes after queueing.

Model text sharing is disabled for the first trial. Station video/task recording
can still be public under platform settings; text-sharing settings do not make
the whole hardware episode private.

## Evidence and handoff

Local workspace: `BluPe对接/硬件联调准备_20261104/` (project root).
It contains both station snapshots, saved-image classification, the candidate
physics smoke run, official monitor Run-block receipt, software readiness and
regression output. A separate compact partner kit contains source, templates,
instructions, the short local demos and selected non-secret evidence.
The original ten-case qualification remains in its unchanged 0.3.4 package.

Official references: [Session API](https://blupe.io/docs/robot-api/) and
[shared Runner](https://github.com/andlyu/blupe-remote-bots). API availability
and repository HEAD may change; verify contract/profile before a live trial.
