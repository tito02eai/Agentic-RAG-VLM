# BluPe platform integration — Agentic RAG-VLM 0.3.5

For a real station use the [first hardware trial guide](BLUPE_HARDWARE_TRIAL.md)
and `scripts/serve_blupe_hardware.py`. Its default is a read-only monitor, with
Run blocked before queue admission. Site geometry and actual hardware acceptance
are still pending; the local simulator launcher is a separate entry.

Latest: [bounded local robustness qualification](BLUPE_ROBUSTNESS_20261104.md).
Ten functional checks passed (four completions, six expected stops).
Site geometry and hardware acceptance remain pending.

Start with the [recommended-task qualification](RECOMMENDED_TASK_QUALIFICATION.md).
It exercises the actual production provider, official HTTP/WebSocket SDK and
native joint planner against local protocol-driven contact physics. The older
subprocess showcase adapter is not the production path. Public hardware remains
uninitialized; local simulator profiles explicitly cannot qualify hardware.

The [2026-11-04 local official-runner validation](BLUPE_LOCAL_VALIDATION.md) exercises
the real controller, HTTP SDK and Playground. It also provides a recorded-image
local UI. Runner execution can be hosted on our own machine; installation on
BluPe's servers is not required to connect through their public API.

This release delivers a complete Agent provider integration and a deployable model backend. Installing, registering, preparing and previewing the policy do **not** require an active robot, a queue entry, or physical qualification. Site geometry is required when enabling motion, not when deploying the software.

## What the email asks us to integrate

The saved Andrew email asks about bringing Agentic RAG-VLM to BluPe, making it available to users, and choosing a starting task/setup. The relevant platform is [BluPe](https://blupe.io/), its [Playground](https://playground.blupe.io/), and the official [shared runner](https://github.com/andlyu/blupe-remote-bots). The older `blupe-remote-yam` URL redirects to this repository. The documented public API and actual runner source were inspected; source hashes are included in the validation receipt.

Our package runs the actual `PaperAgent`: registered object perception → real CLIP embeddings → HAA-RAG → scene constraints → VLM action review → platform commands → fresh verification → memory/monitor/replan. The remote CLI now defaults to this profile. Static cards remain an explicit legacy/mock option.

## Install the lightweight platform process

Unpack `source.zip`, enter that source directory, and use Python 3.10+:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[deploy]'
python scripts/check_platform_provider.py \
  --config configs/blupe_provider.example.json \
  --output output/provider_readiness.json
```

The Agent process uses NumPy, Pillow and the standard library. It does not require MuJoCo, Torch or a locally initialized robot. Qwen/CLIP inference services run separately. Their setup is described in `docs/CMU_EVALUATION.md`; the measured checkpoint identity and CLIP embedding fingerprint must match the configured knowledge. No training or VLA checkpoint is required.

The example provider configuration has no deployment profile and no real destinations. It can be loaded, registered and prepared in that state. It must not fabricate measured coordinates to make a motion test pass.

## Deploy the HTTP model backend now

```bash
python scripts/serve_platform_policy.py \
  --config configs/blupe_provider.example.json --port 18076
curl http://127.0.0.1:18076/health
curl http://127.0.0.1:18076/v1/capabilities
```

Available routes are `GET /health`, `GET /v1/capabilities`, and `POST /v1/preview`. See `configs/platform_policy.openapi.json`. A preview returns `decision: propose|stop`, the candidate action, grounded perception, retrieval/constraint evidence and raw model-review receipts. `execution_performed` is always false. There is no HTTP robot-write route.

For a non-loopback binding, set `AGENTIC_POLICY_TOKEN` and use `--host 0.0.0.0`; callers supply its Bearer token. HTTPS/reverse-proxy routing belongs to the host deployment. Tokens are not stored in result files.

**This preview URL is not a generic Responses API endpoint to paste into the Playground's arbitrary-model field.** The complete task policy integrates through the runner provider below. This avoids reducing a multi-stage task into one guessed joint command.

## Register the provider with BluPe's existing runner

The inspected `RunnerController` has a trusted `run_session(prompt, observation, session_api, context, cancelled)` hook. Our `AgenticRAGVLMProvider` implements it. The runner continues to own queue admission, capabilities, lifecycle and Stop; the provider adopts the existing session and uses the supplied SDK for trajectory submission and terminal-result reads.

Add a provider factory in the host application's model/provider registry:

```python
from agentic_rag_vlm.platform.runner_provider import load_provider

def make_agentic_rag_vlm():
    provider = load_provider('/config/agentic-rag-vlm.json')
    provider.prepare()                 # Models/knowledge only; no robot request.
    return provider

# Registry key / displayed approach: agentic-rag-vlm / Agentic RAG-VLM.
# In the host's existing user-initiated Run handler:
# controller.join(make_agentic_rag_vlm(), prompt)
# Keep its existing Stop and queue controls.
```

The last line creates a real queue entry when invoked; it is a deployment recipe, not a command executed by our installation or tests. There is no second queue admission inside the provider, and no session capability is serialized. Underlying model configuration is distinct from the displayed research-method name.

Check a host runner checkout without importing or executing it:

```bash
python scripts/check_platform_provider.py \
  --config /config/agentic-rag-vlm.json \
  --runner-source /path/to/blupe-remote-bots/codex-runner/src/remote_yam \
  --output output/platform_compatibility.json
```

This checks required SDK methods and the task-policy hook, and records exact source fingerprints. Actual semantics are covered by local adapter contract tests. The motion adapter currently supports 10-Hz YAM sessions; it rejects a different cadence, changed episode, wrong session, expired/ended execution context, or cancellation before dispatch. Fresh lease IDs are taken from the existing session, not cached forever.

## Observation and perception interface

The platform supplies a top RGB image, its calibration identity/intrinsics, and arm-base geometry. The model supplies identities/pixels, never executable metric coordinates. `PlaneProjector` computes feature positions from the calibrated image and declared feature heights. Pixel coordinates remain in original image resolution even though inference crops are enlarged.

Optional `object_candidates` contain only opaque IDs, `bbox_xyxy` and `pixel_xy`. An external detector/tool may supply them via HTTP or a Python `candidate_provider(observation)` callback. Candidate identities are still classified by the VLM; unknown/duplicate identities are rejected or kept unknown. Only registered objects with geometry are offered in the model's scene catalogue. Objects without usable geometry cannot become executable targets.

For small objects, the pipeline adds eight original-image pixels of context and prepares 192×192 inference crops, retaining original-image crop bounds and hashes. This fixes tiny-crop input degradation. The registered-scene candidate path is preferred with the current 4B model; unrestricted whole-frame localization has shown unreliable output and is not a validated open-world detector. A missing protected object blocks even a preview before action review.

`POST /v1/preview` accepts an inline image rather than arbitrary server file paths:

```json
{
  "task": {
    "task_id": "example", "instruction": "Move the registered cube to the tray",
    "coordinate_frame": "arm_base",
    "subgoals": [{"target": "cube", "destination": "tray"}],
    "protected_objects": [], "destinations": {"tray": [0.2, 0.1, 0.04]}
  },
  "image": {"data_base64": "BASE64_OF_ORIGINAL_RGB"},
  "calibration": {"calibration_id": "SITE_CALIBRATION", "cameras": {}},
  "geometry": {
    "arm": "right", "coordinate_frame": "arm_base",
    "support_plane": {"normal_base": [0, 0, 1], "offset_m": 0},
    "feature_heights_m": {"cube": 0.04}
  },
  "object_candidates": [{"candidate_id": "candidate_0", "bbox_xyxy": [10, 10, 80, 80], "pixel_xy": [45, 45]}]
}
```

Values above illustrate schema only; camera matrices and site coordinates must come from the caller's setup. The included recorded-image requests are labelled simulation-camera software tests, not hardware measurements.

## When the operator later enables task execution

Set `deployment_profile`, a fixed arm-base task or named destinations for bounded VLM task planning, and the site's registry. Existing measured-profile checks apply when `run_session` is invoked. They do not block provider deployment/registration or HTTP preview.

The 14-record store remains simulation-origin knowledge. `ExecutorBoundRetriever` preserves that provenance and binds session memory to robot/arm/calibration. Unsupported force/synergy recommendations remain labelled unapplied priors; the current wire API uses joints and normalized gripper commands. No hardware experience is invented, and no calibrated force-control claim is made.

After each pick/place, the existing full session requires fresh stable observations, destination checks, protected-object checks and release/retraction confirmation before recording success or memory. Pre-contact failures can trigger bounded reflection; unsupported recovery motions and uncertain post-contact failures stop. Stop/cancellation is supplied by the runner and interrupts further requests.

## Delivery scope

Completed here: software composition, model backend, provider/SDK adapters, OpenAPI/configuration, local official-runner runtime tests, true-model interface smoke checks and source packaging. A local runner may connect directly to BluPe; hosted catalogue publication is a separate option. No physical robot, public queue admission, public run or outbound email was used by these tests. Physical-task performance is a separate later acceptance, not a prerequisite for this software handoff.
