# Paper-aligned reconstruction acceptance — 2026-11-02

Method mapping and installation: [PAPER_ALIGNED_FRAMEWORK.md](PAPER_ALIGNED_FRAMEWORK.md).

## Final continuous physical qualification

Run: `仿真实验/paper_alignment_20261102/memory_paired_final_r2/` in the parent
workspace. Five declared seeds (40–44), identical geometry/controller/knowledge,
with memory on/off. Each case runs once from frozen source and protocol. Each
task grasps two free objects sequentially while protecting a third object.

| Condition | Complete two-object tasks | Complete grasps | Memory hits | Mean first / second placement error |
|---|---:|---:|---:|---|
| Memory enabled | 5/5 | 10/10 | 5 | 1.667 / 1.479 mm |
| Memory disabled | 5/5 | 10/10 | 0 | 1.667 / 1.479 mm |

The batch includes **60 actual Qwen3.5-4B identity calls plus 20 actual
pre-execution VLM decision calls**. Actual CLIP ViT-L/14 encodes observed crops.
The first cube has a fragile-neighbor constraint: 12 N intrinsic motor-force
cap becomes 9.6 N with +30 mm approach height. The second uses verified category
memory at its **new** RGB-D location, removes that transient constraint and
restores 12 N / +0 mm. Enabled tasks log `episodic_memory`; disabled tasks keep
empty memory. These are simulated actuator caps, not calibrated fingertip force.

Both conditions have identical outcomes because retrieved parameters already
suffice. The experiment verifies use/bypass, not a success-rate benefit or less
exploration in challenging tasks. Maximum protected displacement is 0.221 mm;
no robot/protected contact or >1 mm arm/table/self penetration was recorded.
Small compliant target/finger contact penetration remains modeled. No object
attachment or post-initialization object reset is used.

The two raw MP4s are the predeclared seed 40 (one per condition), 41.48 s,
H.264, 640×480, 25 fps. Time is simulation time; inference waits are not included.
Videos show both complete grasp/place/release sequences and retreat.

Provenance:

- Protocol SHA256: `ce03bbaaf93ffed29567bcf8d57970bd9a84188847e331641f54310c8a2a7b7b`.
- Frozen source ZIP SHA256: `d796ffd44d85d5243db123d008fd32bf1ab8fe45639817ed7dd7fc8efe1a0f19`.
- 197 frozen source files pass the manifest audit. Existing core files match
  current source byte for byte; later additions are the standalone language
  task planner, tests and docs.
- CLIP weights SHA256: `a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb`.
- Artifact audit passes for declared cases, hashes, raw first/second physical
  evidence, saved model responses and memory use/bypass.
- Physical environment: Python 3.10, MuJoCo 3.2.2, NumPy 1.26.4. CPU feature
  service: Python 3.12, torch 2.11.0, Transformers 5.13.1.

## Separate mechanism checks

**180 tests pass, 10 skip** for unavailable historical Guanghua assets. Coverage
includes exact retrieval gates/weights/negative penalty; provenance; memory
persistence, semantic/domain rejection and corrected-parameter reuse; fresh
visual L3 reset/jitter; all 14 failure types; support-removal session ordering;
protected/cyclic dependencies; swept-envelope path validation; unsafe-contact
stop; invalid quality rejection; and VLM decision-before-execution ordering.

`recovery_contract_final.json` is an explicitly **synthetic analytical fixture**
with 4 real VLM reasoning/reflection calls. Initial 12 N fails a declared 14 N
fixture requirement; L1 reaches 14.4 N, then a same-category task reuses it in
one attempt. It records zero physical trials and is not part of robot rates.

`language_task_plan.json` contains one actual model call translating instruction
into `cube→tray`, `cube2→tray2` while protecting the vase. This is planner
acceptance, not another grasp. The generic `VLMTaskPlanner` validates object IDs,
destination names, protected objects, finite goal count and action schema.

The bundled KB has **10 observed successful episodes**, each with source image,
result, actuator receipt and hashes. Failed-record retention/penalty are tested
with fixtures; no fabricated physical failures replace the lost original data.

## Preserved failures and limits

Development runs include an actual protected-object collision stop, a hidden
target stop and a VLM veto under an ambiguous overview. Fixes clear the camera
view and provide the correctly grounded target crop. Earlier evidence is kept.
An earlier source batch (`memory_paired_v1`, seeds 30–34) completed 10/10 tasks.

The startup batch `memory_paired_final` produced 10 connection-refused process
errors after the tool-hosted CLIP server terminated; none reached task-agent or
grasp execution. Logs are retained. Restarting the local service independently
and adding startup dependency checks preceded the valid final batch above.
Startup errors are not counted as successful robot episodes.

Original 116 experience records, original prompts/trial logs/seeds, ten relation
definitions and exact individual quality formulas remain unavailable. This is
a working reconstruction of published software mechanisms; original data and
the reported 78.3% are not reproduced. All constraint/recovery types have not
been physically demonstrated on YAM, and native BluPe hardware acceptance is
separate. Details and reconstruction choices are explicit in the method guide.
