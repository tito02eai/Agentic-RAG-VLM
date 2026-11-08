# Current software status — 0.3.5

Updated 2026-11-08 after the mixed-shape controller regression repair.

The framework and BluPe integration preparation are available in the repository.
Use [the repository README](../README.md) and
[the hardware trial guide](BLUPE_HARDWARE_TRIAL.md).

- Full paper-profile retrieval, VLM review, scene constraints, verification and memory are connected.
- The production provider uses the official Runner and native joint/gripper trajectories.
- The hardware monitor is read-only by default; site measurements remain pending.
- Latest local software-rendered validation recorded 360 passed and 12 skipped tests with the pinned official Runner.
- Ten local robustness criteria passed: four task completions and six expected stops.
- Saved real-camera image and paired-packet checks do not establish metric accuracy or a hardware grasp.
- Public BluPe hardware execution and public provider registration remain unperformed.

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
