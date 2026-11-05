# YAM integration qualification results — 2026-11-02

Run: `仿真实验/yam_20261102/qualification_holdout_v2/` in the parent workspace.
The protocol, scene assumptions and limitations are documented in
[YAM_PHYSICS_QUALIFICATION.md](YAM_PHYSICS_QUALIFICATION.md).

| Scenario | Declared seeds | Completed physical + framework episodes | Mean / max destination error |
|---|---|---|---|
| Single cube into marked zone | 10–19 | 10/10 | 1.569 / 1.797 mm |
| Same task with protected vessel proxy | 10–19 | 10/10 | 1.582 / 1.769 mm |

There were 60 real Qwen3.5-4B classification calls in these 20 episodes (one
initial plus two final observations per episode). The actual response model was
`/home/admin1/g2_multimodal_agent/models/Qwen3.5-4B@main`; this is the server's
reported model identifier, not a cryptographic weights fingerprint. Weights were
not trained or changed by this work.

Across all cases the object was physically lifted to approximately 13.4 cm,
with at least 3.754 s of bilateral finger contact above 10 cm. Placement ended
with table support and no finger contact. No robot/protected-object contact,
arm/table penetration beyond 1 mm or self penetration beyond 1 mm was recorded.
Maximum target/robot soft-contact penetration was **1.332 mm**, not zero.
Maximum protected-object displacement was **0.090 mm**. These are simulation
measurements with the documented collision geometry and parameters.

The configuration was frozen before running all declared cases exactly once.
No case was rerun or replaced in this held-out batch. The two recorded episodes
are the declared first seed (10), one for each scenario. Each raw MP4 is
20.4 seconds, 640×480, H.264, 25 fps, at simulation time. Inference pauses are
not represented as waiting frames.

Provenance:

- Protocol SHA256: `55fb392754f29f20044ede6d2b14a74d97f8d5e15c4cb902f865b63cafb7b397`.
- Frozen source ZIP SHA256: `f6175a8d5768b952fd28a9d46fbd49b2326f663545ea1715621e57edc0626ca8`.
- 143 frozen source files verified against the ZIP's per-file manifest.
- The final active physics executor, simulation implementation and perception
  implementation match their frozen evaluated copies byte for byte. Subsequent
  changes add an evidence auditor, result documentation and a safety test.
- Full active test suite: **149 passed, 10 skipped**. All skipped tests require
  the historical Guanghua assets, which are not distributed in this source.
  Tests include actual MuJoCo single/protected grasps, unreachable preflight
  rejection and table-collision interruption.
- Runtime: Python 3.10, MuJoCo 3.2.2, NumPy 1.26.4, mediapy 1.2.0,
  Pillow 12.3.0; pytest 8.4.2. EGL rendering; 2 ms physics step.

Audit the saved run with:

```bash
python scripts/audit_yam_qualification.py /path/to/qualification_holdout_v2
```

The audit checks declared cases, source/config hashes, result/trace consistency,
sampled raw lift/support/release evidence and saved model-response receipts. It
does not replace physics/model review or certify a real robot.

The earlier development batch is retained separately (19/20; one model
classification became unknown under partial occlusion). A parking/view fix was
made before selecting seeds 10–19 for this batch. Do not combine development
and held-out counts or describe these results as the original IROS trials.

These two narrow scenarios qualify the reconstructed framework and executor.
They do not establish retrieval/scene-graph/recovery superiority over a fixed
detection+IK baseline. Multi-object memory reuse, challenging paired ablations,
broader perception, recovery after genuine physical failures and calibrated
BluPe hardware operation have not been accepted by this experiment.
