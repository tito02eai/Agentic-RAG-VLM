# Current software status — 0.3.5

Updated 2026-11-05 for the GitHub handoff.

The framework and BluPe integration preparation are available in the repository.
Use [the repository README](../README.md) and
[the hardware trial guide](BLUPE_HARDWARE_TRIAL.md).

- Full paper-profile retrieval, VLM review, scene constraints, verification and memory are connected.
- The production provider uses the official Runner and native joint/gripper trajectories.
- The hardware monitor is read-only by default; site measurements remain pending.
- Local validation recorded 354 passed and 12 skipped tests, with separate current-runner checks.
- Ten local robustness criteria passed: four task completions and six expected stops.
- Saved real-camera image and paired-packet checks do not establish metric accuracy or a hardware grasp.
- Public BluPe hardware execution and public provider registration remain unperformed.

[Demo videos and selected receipts](../examples/blupe/README.md) are included.
Larger frozen experiment archives and original failures remain separate; selected
summaries are not a replacement for independently auditing all raw cases.

The older date-labelled reports in this directory retain their original scope.
Original-paper results and new engineering measurements must not be pooled.
