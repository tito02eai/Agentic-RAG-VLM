# Validation and provenance

## Integration review, 2026-11-09

The local gateway now reuses its finalized result for concurrent or repeated
result/video requests. Observation refreshes queued before finalization keep
the last image after the renderer closes. The independent evaluator uses the
first target's own bilateral-lift duration and explicitly requires the second
target to be released from robot contact.

The [review receipt](../examples/blupe/integration_review_20261109.json) records
the regression, ten fault/memory criteria and one browser task. The browser
check uses the official HostedRunner HTTP routes to launch a complete task,
then downloads the result and video concurrently and repeatedly. These runs
are a separate engineering cohort from the qualification below.

Local regression recorded 384 passed and 12 skipped tests, including five new
lifecycle and false-positive counterexamples. The frozen ten-case run passed
all criteria with four completed tasks, six expected stops and 36 saved real
Qwen replies. The browser task completed its independent physical checks.
All 126 Python source files matched the frozen cohort during the review.

## Qualified engineering revision

Validation commit `440a40e8fd9f0cb1d7c3155d8a9fc165e8cffe9a` qualified the native deployment path on 2026-11-08. The selected [machine-readable receipt](../examples/blupe/pre_hardware_20261108.json) retains model identity, source/runtime hashes and case outcomes.

| Group | Result | Scope |
| --- | --- | --- |
| Local GPU, local Mesa and GitHub CI | 379 passed, 12 skipped in each | Software and physical regression tests |
| Normal task qualification | 6 complete tasks | Three layouts, each tested with two renderers |
| Partial tabletop ROI | Rejected before motion | One declared rejection case |
| Memory/protocol/Stop criteria | All 10 criteria passed | Four completed tasks and six expected stops |
| Saved real Qwen replies | 60 | Perception and planning responses across these cohorts |
| Post-packaging reproduction | One complete task at seed 1043 | A separate source-package smoke check |

The final-source layout runs repeat the preliminary layouts and are labelled as known-layout regression. Preliminary and final cohorts remain separate. The six normal runs had independent simulator destination errors of approximately 0.56–0.67 mm. The standalone package smoke check measured approximately 0.85 mm. These measurements describe this local testbed.

The independent audit verified frozen source files, model identity, raw replies, JSON trajectory validation through the official SDK, contiguous execution steps, terminal feedback, placement/support checks and the expected stopping behavior. Original failures and earlier qualification batches are retained in the local archive.

[Qualified revision CI](https://github.com/tito02eai/Agentic-RAG-VLM/actions/runs/42476397836)

## Current demonstrations

The two videos in [examples/blupe](../examples/blupe/README.md) come from the final-source qualification. The one-object video is 16.08 seconds. The moved-target memory video is 29.80 seconds. Their receipts identify the local MuJoCo environment and production provider/SDK execution.

## Paper and engineering results

The [accepted manuscript](../paper/main_final.pdf) documents the original research. The reconstructed software, the current 14-entry engineering store and later YAM/BluPe checks have separate provenance. Historical paper measurements remain in the manuscript; the current task receipts describe the new engineering runs.

## Boundaries

- The local gateway has one active YAM arm and a virtual held second-arm channel.
- Perception uses registered semantics, controlled RGB proposals, known feature heights and a configured support plane.
- Simulation retention and hardware blocked-aperture feedback have separately declared meanings.
- The public BluPe hardware trial, site calibration and operator setup remain pending joint validation.
- Earlier dated reports, raw images, failures and source snapshots are preserved in the local historical archive. Git history retains the earlier public revisions.

See [BluPe integration](BLUPE.md) for the first hardware setup and [Reproduction](REPRODUCTION.md) for dependency and asset details.
