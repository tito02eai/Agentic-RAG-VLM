# BluPe official-runner local validation — 2026-11-04

The Agentic RAG-VLM 0.3.1 integration has been exercised inside the **unmodified
official RunnerController, HTTP SDK and Playground application**, rather than
only against adapters that we wrote. This is local software integration
validation. No physical session, public queue entry or partner deployment was
created.

Official repository: https://github.com/andlyu/blupe-remote-bots

Checked revision: `e6040394c5ed62efbcc7d627acc0dfc8ccc100a6`.

## What was actually tested

The opt-in integration suite contains 18 checks, all passing. The full regression
run with both actual-model previews enabled passed **319 tests**, with **10
historical-asset tests skipped**. These are software checks, not robot-task trial
counts.

1. The actual official controller invokes the production provider's preparation,
   admission and complete-task hooks. The real PaperAgent then completes two
   sequential targets, using HAA-RAG followed by verified episodic-memory reuse.
   **Perception/model and motion are explicit deterministic test doubles in this
   lifecycle case.** This is not a measured grasp-success experiment.
2. The official HTTP SDK, connected only to a loopback gateway fixture, carries
   the session capability internally, refreshes the lease, submits a valid wire
   trajectory and reads terminal feedback. Changed episodes, incompatible
   session/packet cadence, non-running sessions, an already-active trajectory
   and cancellation produce no trajectory POST.
3. The official Playground's `/api/session`, `/api/run` and `/api/status` routes
   register and invoke our approach. The controller owns the single local
   session admission and Stop; the provider creates neither a second session nor
   its own stop request. User Stop cancels the running task hook.
4. Actual Qwen3.5-4B and CLIP services process a previously recorded simulation
   image through the official controller's task hook. One request produces a
   grounded cube proposal with HAA-RAG and an actual VLM action review. A second
   request with an unconfirmed protected object stops before proposing/reviewing
   an action. Both perform **zero execution**. These are interface smoke checks,
   not new independent benchmark trials.

The upstream `MockSessionAPI` is used for controller events and lifecycle. It
does not expose all deployed gateway fields in its snapshot/episode trace;
therefore a separately labelled loopback HTTP fixture supplies lease/terminal
results for the SDK test. We do not assume that accepting a packet establishes
execution or that the in-memory mock is a physics simulator.

## Fixes made after exercising the actual runner

- Added `prepare_before_session`: missing/invalid execution configuration and
  unavailable models now fail **before admission**, instead of occupying a
  public queue and failing inside `run_session`.
- Added explicit trajectory-capability admission. This matters because the
  official bare `HttpSessionAPI` defaults `supports_trajectories` to false.
- Tightened dispatch to running sessions, 10-Hz session/packet cadence and no
  in-flight trajectory; episode/lease and cancellation remain checked.
- Split local profile-field validation from live calibration validation, so
  preparation can validate configuration without inventing calibration data or
  contacting a robot. The full live calibration check still runs on adoption.

Software-only provider preparation and `/v1/preview` continue to work with no
deployment profile. Execution admission is a separate step.

## Reproduce the integration suite

From the `code` directory, with Python 3.10+:

```bash
python -m pip install -e '.[runner,dev]'
git clone https://github.com/andlyu/blupe-remote-bots.git /path/to/blupe-remote-bots
git -C /path/to/blupe-remote-bots checkout e6040394c5ed62efbcc7d627acc0dfc8ccc100a6
python scripts/validate_blupe_runner.py \
  --runner-checkout /path/to/blupe-remote-bots \
  --recorded-request /path/to/recorded_single_request.json \
  --protected-request /path/to/recorded_protected_request.json \
  --output output/blupe_runner_validation
```

The delivered `examples/` directory contains both recorded requests. Without
these two optional arguments the actual-model checks are explicitly skipped;
the deterministic controller/SDK/Playground contracts still run. The script
saves JUnit, source fingerprints, upstream revision and the raw preview receipts.
It never takes a BluPe public endpoint argument.

The Agent process remains lightweight with `[deploy]`. The separate `[runner]`
extra includes MuJoCo because the official runner imports its simulation IK
module even during local interface tests; it does not imply that our lightweight
HTTP backend needs MuJoCo.

## Open the official UI locally

```bash
python scripts/serve_blupe_local_preview.py \
  --runner-checkout /path/to/blupe-remote-bots \
  --config configs/blupe_provider.example.json \
  --recorded-request /path/to/recorded_single_request.json \
  --output output/blupe_local_preview --port 18791
```

Open http://127.0.0.1:18791/. The configured model is **Agentic RAG-VLM — local
preview**. Its fixed task uses the displayed recorded top-camera input. Run
performs real classification, CLIP retrieval, constraints and action review.
The result link shows the complete receipt. The adapter admits only a marked
local mock API and has no trajectory/action submission path. It is intentionally
separate from the production `AgenticRAGVLMProvider.run_session` adapter.

The UI preserves the official cookie/CSRF, task lifecycle and Stop. A local
stylesheet/input-image overlay makes the recorded input visible instead of
leaving the official live-stream viewer black. Status notices and the last
preview result are labelled; “Stopped” at completion means the local session
has ended, not that a physical grasp failed. Models and knowledge are warmed
before starting this local fixture. No subscription login or API key is needed.

## Remaining boundary

This proves local software compatibility at the recorded upstream revision. It
does not demonstrate an actual public BluPe lease, hardware grasp, deployment
on their server or publication of our method in their shared model catalogue.
The current 4B model still benefits from detector-provided, identity-free object
candidates; unrestricted whole-image detection is not a validated capability.
Original r8/r9 simulation records and success rates are unchanged.
