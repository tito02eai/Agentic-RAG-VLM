# Paper-aligned framework: implementation and reproducibility boundary

Authority: `paper/main_final.tex`, Sections III–IV, Algorithm 1. This document
supersedes earlier engineering-status descriptions for the new **paper profile**.
Frozen historical proxy scripts retain their original purpose and outputs.

## Run the strict profile

Use two environments if the robot process should not load torch. The simulator
needs `.[sim,dev]`; the local feature service needs `.[vision]`. An existing
OpenAI-compatible Qwen vision endpoint supplies model decisions and identity
classification. Qwen3.5-4B is used locally; the paper's Qwen3-VL-8B remains a
swappable deployment, not the model used for the new results.

```bash
python -m pip install -e '.[sim,dev,vision]'
python scripts/download_clip.py --output /path/to/clip-vit-large-patch14
python scripts/serve_clip.py --model-path /path/to/clip-vit-large-patch14 --device cpu
# Keep the feature service running; execute the following in another shell.
MUJOCO_GL=egl python scripts/run_paper_agent.py --neighbor --record \
  --endpoint http://127.0.0.1:18070/v1/chat/completions \
  --model /path/to/the/served/qwen-model \
  --output output/paper_continuous --memory-out output/paper_continuous/memory.json
```

The default knowledge store is `knowledge_base/yam_observed_clip.json`. It
contains **10 newly observed successful contact episodes**, with portable source
images, results, actuator receipts and SHA256 provenance. It is not the missing
116-entry original knowledge base. There are no invented successful labels,
duplicated entries added to reach 116, or fake CLIP embeddings.

Knowledge-source images/results are validated before the paper example starts
moving. Encoder weights are fingerprinted. Robot domain must match. Feature
vectors are 768-dimensional CLIP ViT-L/14 image projections, normalized then
compared by cosine. The encoder uses the
[official Transformers CLIP implementation](https://huggingface.co/docs/transformers/model_doc/clip)
and pinned public `openai/clip-vit-large-patch14` revision
`32bd64288804d66eefd0ccbe215aa642df71cc41`.

## Implemented mechanisms

1. **Experience schema and HAA-RAG.** Store category, superclass, four affordance
   fields, complete strategy, binary measured outcome, CLIP vector and provenance.
   Exact category scores 1; same superclass scores 0.5; unrelated categories are
   discarded. Keep at most 30, then reject affordance score below 0.3. Affordance
   weights are 0.5/0.2/0.15/0.15; final category/affordance/visual weights are
   0.2/0.4/0.4. Failed-experience score is multiplied by 0.7; retain its record for
   reflection. Return top 3. Missing compatible experiences/embeddings stop
   explicitly. There is no aspect-ratio fallback in this profile.
2. **Session episodic memory.** Store only verified successful parameters by
   category, with source task and quality. Reuse current observed position;
   strip transient scene force/height adjustments before applying current scene
   constraints. Reject incompatible semantics and robot domains. Failed reuse
   invalidates the session entry. Optional atomic save/load restores a session;
   long-term autonomous learning is not a paper requirement.
3. **Observed scene graph.** Directed relations, empty/filled/stacked states and
   0.15 m relation analysis. Filled containers require vertical slow approach;
   fragile neighbors within 0.10 m cause ×0.8 force, +30 mm height and away bias.
   Support edges insert removal prerequisites and reject cycles/protected
   prerequisites/missing destinations. Occlusion proposes a lateral route,
   accepted only by a supplied physical path validator. The supplied swept-box
   validator checks known object/gripper envelopes and workspace/table bounds;
   it does not certify an entire arm. Conflicting content-preservation and
   lateral-route constraints stop instead of silently overriding each other.
4. **VLM ReAct and quantitative recovery.** Actual model decision summaries
   occur before motion; preserve prompt, images/hashes, raw/parsed responses,
   usage and latency. Reflection receives diagnosed failures, retrieved negative
   examples and quantitative corrections. Model output cannot invent metric
   coordinates or override collision limits. Initial execution plus at most
   3 retries; 14 failure classes; L1 targeted force/width/pose adjustments,
   L2 power→pinch→side and ×0.9 force, L3 fresh observation/retrieval, reset
   parameters and seeded U(-0.01,0.01)^3 position perturbation. Strict generic
   replanning requires `PlanningObservation` with fresh visual features.
5. **Seven-factor evaluator.** Preserve published weights
   0.20/0.15/0.10/0.15/0.15/0.10/0.15. Accept validated sensor/evaluator factors,
   reject nonfinite inputs, allow per-factor thresholds, and never allow a
   declared physical failure to pass on aggregate score alone. The measurement
   adapter provides explicit reconstructed formulas; original per-factor
   formulas and most thresholds were not published/recovered.

The common implementation is `agentic_rag_vlm.paper.PaperAgent`, composed with
`BluPeSession` for physical task execution. Generic `PaperAgent.run` also accepts
an executor returning factors, outcome and diagnostics for the full analytical
recovery loop. Its observer returns a fresh `PlanningObservation` on L3.

For free-form instructions, `agentic_rag_vlm.task_planner.VLMTaskPlanner` takes
the current grounded objects, named destinations and protected IDs and returns
a validated `TaskSpec` plus the full model receipt; pass that task to the session.
It cannot output executable metric coordinates. The physical paired protocol
uses fixed task subgoals for experimental control; language-plan acceptance is
recorded separately in the results document.

Acceptance and evidence: [PAPER_ALIGNMENT_RESULTS_20261102.md](PAPER_ALIGNMENT_RESULTS_20261102.md).

## Verification scope

The continuous demonstration moves a red cube and a yellow cube sequentially
to separate zones while protecting a green vessel proxy. Both targets have the
same semantic category. The first grasp has a fragile-neighbor constraint; the
second reuses successful intrinsic parameters and computes its own constraints.
All objects remain free/contact-driven. Current YAM skills implement validated
top approaches; unsupported lateral skills stop. Hardware yaw corrections also
stop until calibrated. These are execution capability limits, not claims that
all four constraint families have been physically demonstrated on YAM.

The paired protocol in `configs/paper_memory_qualification.json` enables or
disables memory on identical seeds and scene/controller parameters. It tests
actual reuse and its bypass; high success for both conditions does not establish
a success-rate improvement. Early batches and failures remain separate.

```bash
MUJOCO_GL=egl python scripts/run_yam_qualification.py --vlm \
  --paper-store knowledge_base/yam_observed_clip.json \
  --config configs/paper_memory_qualification.json --output output/paper_pairs
python scripts/audit_yam_qualification.py output/paper_pairs
python -m pytest -q tests
```

`scripts/run_paper_recovery_contract.py` exercises real model reflection and
L1→successful-memory-reuse with an explicitly synthetic analytical fixture.
Its result states `physical_robot_trials: 0`. It is not mixed with physical
success rates or treated as recovery after a measured robot failure.

## Irrecoverable original details and documented reconstruction choices

- Original 116 experience records, including negative examples and CLIP vectors,
  original VLM prompts and original 12-task raw logs/seeds are absent.
- Six relation names are published; the other ten registry names are explicitly
  reconstruction choices in `scene.py`.
- The paper gives example L1 corrections for slip/width; other L1 numerical
  rules, safe-stop policies, quality formulas/thresholds and tie-breaking choices
  are implementation choices, documented in code. They are not recovered code.
- Current local VLM is Qwen3.5-4B; original Qwen3-VL-8B results are not reproduced.
- The new physical tests are YAM MuJoCo tests with restricted RGB-D color
  proposals and known dimensions, approximate collision shapes and ideal link
  gravity compensation. Native BluPe model/calibration and hardware remain
  separate. Autonomous KB expansion and learning fragility from vision are not
  claimed.

Accordingly this is a working reconstruction of the published mechanisms, not
bit-for-bit recovery of the lost project or numerical reproduction of 78.3%.
