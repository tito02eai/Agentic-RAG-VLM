# Current software status — 0.3.5

Updated 2026-11-08 after native deployment-path and perception checks.

The framework and BluPe integration preparation are available in the repository.
Use [the repository README](../README.md) and
[the hardware trial guide](BLUPE_HARDWARE_TRIAL.md).

- Full paper-profile retrieval, VLM review, scene constraints, verification and memory are connected.
- The production provider uses the official Runner and native joint/gripper trajectories.
- The hardware monitor is read-only by default; site measurements remain pending.
- Latest local GPU and software-rendered validation each recorded 379 passed and 12 skipped tests with the pinned official Runner.
- Ten local robustness criteria passed: four task completions and six expected stops.
- Saved real-camera image and paired-packet checks do not establish metric accuracy or a hardware grasp.
- Public BluPe hardware execution and public provider registration remain unperformed.

## Native deployment-path acceptance, 2026-11-08

The official Runner/SDK path now uses Cartesian IK waypoints for contact
approach, lift and retract. A saved-calibration FK audit measured lift lateral
excursion falling from 6.57 mm to 0.15 mm. The Runner and standalone CLI share
the same measured joint-step, lift-height and camera-clearance configuration.
The RGB candidate provider rejects regions cut by the configured tabletop ROI.
Fractional joint/gripper trajectories preserve exact endpoints and were checked
through JSON serialization and the official SDK validator for both arms.

Final-source acceptance repeated three layouts under GPU and Mesa rendering.
All six grasp/place tasks completed, with independent destination errors of
0.56–0.67 mm in this simulator. One partial-ROI case stopped before motion.
Ten separate fault/memory criteria passed with four task completions and six
expected stops. These runs retained 60 actual Qwen replies and passed a read-only
audit of source hashes, model identity, wire packets and physical evidence.
The final-source repeats are known-layout regressions. Preliminary runs are
retained separately and are not pooled into a larger success rate.

[Selected acceptance receipt](../examples/blupe/pre_hardware_20261108.json)
records the exact scope. The software is prepared for a coordinated first
single-active-arm rigid-object trial. Site calibration, measured object/tool
offsets, destinations, workspace and operator readiness still require physical
confirmation. The current read-only station snapshot is not a completed trial.

[Demo videos and selected receipts](../examples/blupe/README.md) are included.
Larger frozen experiment archives and original failures remain separate; selected
summaries are not a replacement for independently auditing all raw cases.

## Mixed-shape regression repair, 2026-11-08

The seed-20 heterogeneous-contact diagnostic reproduced the earlier CI failure
under Mesa software rendering. A single joint-space lift produced up to 8.2 mm
of lateral TCP motion and stopped at the 3 mm table-penetration limit. The robust
controller now uses vertical Cartesian waypoints spaced at most 10 mm apart.
The repaired software-rendered case completed all three placements and all 12
physical checks, with a maximum lateral lift excursion of 0.43 mm and maximum
target/table penetration of 1.54 mm. Failed memory reuse and bounded force
recovery still occurred and were logged.

This is a color-labelled RGB-D controller diagnostic. It makes no model calls.
Its six grasp attempts are part of one task rollout and must be reported with
that scope. The earlier failed rollout and the intermediate geometry-only
experiment remain in the local investigation archive.

The older date-labelled reports in this directory retain their original scope.
Original-paper results and new engineering measurements must not be pooled.
