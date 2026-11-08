# Local simulation

The current integration executes the production Agentic RAG-VLM provider through the official BluPe Runner and HTTP/WebSocket SDK. A local Session API gateway supplies images and measured joint/gripper state from MuJoCo. Its second-arm channel is virtual and held fixed. The physics scene has one active YAM arm.

## Complete grasp/place

Start the services and export the Runner path using [Getting started](GETTING_STARTED.md).

```bash
python scripts/run_blupe_protocol_task.py \
  --runner-checkout "$BLUPE_RUNNER_CHECKOUT" \
  --config configs/blupe_provider.example.json \
  --candidate-config configs/blupe_local_candidates.example.json \
  --output output/grasp_seed_1040 --seed 1040
```

The registered task moves a rigid object to a marked area while preserving a neighboring object. `blupe_local_candidates.example.json` is specific to the local 640×480 camera view. It supplies pixel regions; the VLM supplies object identities.

## Browser interface

```bash
python scripts/serve_blupe_recommended.py \
  --runner-checkout "$BLUPE_RUNNER_CHECKOUT" \
  --config configs/blupe_provider.example.json \
  --output output/browser_grasp --seed 1040 --port 18793
```

Open `http://127.0.0.1:18793/`, select the configured task and run it. The official UI displays the local simulation. One episode is supported per launch. The result and raw video become available after the task finishes. Start a fresh process and output directory for another episode. This UI uses its registered local RGB proposal function; the CLI above exercises the configurable candidate provider used for site preparation.

## Memory and fault qualification

```bash
python scripts/run_blupe_robustness_batch.py \
  --runner-checkout "$BLUPE_RUNNER_CHECKOUT" \
  --candidate-config configs/blupe_local_candidates.example.json \
  --output output/fault_qualification

python scripts/audit_blupe_robustness.py output/fault_qualification \
  --output output/fault_qualification/independent_audit.json
```

The ten declared criteria cover normal control, empty grasp, slip after physical lift, moved-target memory reuse, failed-memory invalidation, one lost acknowledgement, persistent acknowledgement loss, feedback outage, WebSocket reconnection and active Stop. Faults are explicit physical or loopback-protocol test conditions. Four criteria expect task completion and six expect stopping.

For one case:

```bash
python scripts/run_blupe_robustness_case.py \
  --runner-checkout "$BLUPE_RUNNER_CHECKOUT" \
  --config configs/blupe_provider.example.json \
  --candidate-config configs/blupe_local_candidates.example.json \
  --output output/memory_moved --case memory_moved --seed 1010
```

## Independent controller diagnostic

This diagnostic uses colour-labelled RGB-D geometry and requires no VLM service:

```bash
python scripts/run_extended_sim.py --diagnostic --controller-profile robust \
  --mode stacked_sort --scene-variation mixed_contact --mixed-shapes \
  --recovery-policy force_ladder --seed 20 --record \
  --output output/mixed_shape_diagnostic
```

It checks contact retention, force recovery, support ordering and memory invalidation under fixed diagnostic semantics. Its results have a different scope from the model-driven protocol tasks.

## Outputs

| File or directory | Meaning |
| --- | --- |
| `receipt.json` | Task result and declared execution provenance |
| `agent_runs/` | Agent events, perceptions, raw VLM replies and memory |
| `trajectory_packets.json` | Joint/gripper data actually dispatched through the SDK |
| `trajectory_events.json` | Acceptance and terminal execution feedback |
| `physics/private_evaluator.jsonl` | Independent physical evaluation evidence |
| `physics/rollout.mp4` | Unedited simulation-time video |
| `physical_result.json` | Independent lift, placement, release and contact checks |

Keep every outcome and use a fresh output directory for each run. Declare layouts, model identity, code and acceptance rules before a qualification batch. Report completed tasks and expected stopping criteria separately. See [Validation](VALIDATION.md) for the selected frozen results.
