# Extended contact-dynamics protocol (2026-11-02)

This revision targets three gaps: dependent multi-object manipulation, recovery
from measured contact failure, and reuse of a recovered strategy. It is a new
controlled stress test, not the original IROS dataset or a benchmark-wide claim.

## Tasks and controls

`configs/extended_sim_qualification.json` freezes five seeds (100–104) for each:

1. Complex nominal: individually sort three blocks with a smaller yellow block
   resting on red, a separate magenta block, three destinations and a protected
   green vessel. Nominal object friction is 1.0.
2. The same dependent task after a controlled friction shift to 0.7, full agent.
3. The identical shifted task with session memory disabled.
4. The identical shifted task with contact recovery disabled.
5. A shifted single-object task isolating the recovery loop.

The initial goal queue is red/yellow/magenta; observed support relations insert
the necessary yellow-before-red order. Relations are computed from public RGB-D
surface heights/overlap and known dimensions, not MuJoCo object poses. A visible
but unclassified supporting surface may confirm physical support for retry;
its semantic identity is not invented. Protected/completed objects must remain
identified and stationary. Color masks provide candidate crops on a neutral
background; Qwen identifies them and provides actual execution/reflection
decisions. Actual CLIP ViT-L/14 embeddings feed HAA-RAG.

## Prior and genuine contact failure

The new `yam_retention_observed.json` store contains three measured nominal
single-object successes, each using a 0.8 N simulated gripper motor-force cap.
Those initial parameters are imported from saved results and actuator receipts;
an evaluated run cannot override them with a condition-specific force.

Objects weigh 0.06 kg (0.5886 N gravity). Finger friction is 0.7. MuJoCo combines
the nominal object/finger parameters; with the shifted object the stable 0.8 N
total normal load has insufficient frictional capacity (about 0.56 N), whereas
the paper's L1 +20% motor cap provides about 0.672 N capacity. Actual finger
forces and retention are measured. These are simulated actuator/contact values,
not a real YAM force-controller calibration.

There is no failure flag, attempt-index outcome rule, object attachment or pose
reset. A pick is accepted only after a 0.4 s bilateral contact window while the
arm is lifted. Initial touch followed by lost contact is diagnosed as slip.
After failure, open, retreat and settle. Retry requires two distinct stable
RGB-D captures, released/retracted gripper, a visible supporting surface and
unchanged protected/completed objects. L1 modifies force from 0.8 to 0.96 N and
uses a freshly measured target location. Only verified successful placement is
remembered. Unsupported later grasp methods reject rather than pretending a
parallel jaw performed a different dexterous skill.

## Controller and numerical settings

Same controller for all conditions: known fingertip geometry sets the lowest
pad about 2.2 mm above the observed supporting surface; IK tolerances and
contact limits are retained. Translation uses a 4 s transfer. Placement lowers
incrementally, then senses reduced downward fingertip load before release.
This avoids pressing an object into the tabletop after it has slipped slightly
within the pads. No object/table contact truth is used by this placement rule.

Extended scenes uniformly use MuJoCo 3.2.2, 2 ms steps, elliptic friction cone,
10 NoSlip iterations and 1e-10 solver tolerance. These settings are fixed from
reset, never toggled by attempt or outcome. MuJoCo documents soft-contact drift
and the NoSlip postprocessor in its
[modeling guidance](https://mujoco.readthedocs.io/en/3.2.0/modeling.html).
Development cases under earlier solver/geometry settings remain preserved;
they are not pooled with the final evaluation. Ideal robot gravity compensation
and approximate collision geometry remain explicit assumptions.

## Acceptance and reporting

Success requires all goal objects within 25 mm, released and table-supported,
at least 0.5 s bilateral lift above 10 cm for each, proper support-removal order,
no protected contact, no >1 mm arm/table/self penetration, target contact
penetration under 3 mm, protected and untargeted motion under 15 mm, and framework
completion. Private object poses are used only by the evaluator/side-effect
audit, never supplied to the planner/recovery policy. Source/config and training
attachments are hashed before launching every declared case once.

Report every outcome, including safe stops, attempts, recovery events, memory
hits, model calls, physical time and collateral motion. Comparisons pair the
same seeds. A memory benefit means fewer failed physical attempts/model calls
at comparable completion, not merely a cache flag. No claim of generalization
outside these selected task/parameter ranges follows from this small study.
The first declared seed in every condition is recorded irrespective of outcome.
Videos are raw simulation-time frames, without model waiting pauses or edits.

```bash
MUJOCO_GL=egl python scripts/run_extended_benchmark.py --output output/extended
python scripts/audit_extended_benchmark.py output/extended
python -m pytest -q tests
```

No native hardware API is called by these scripts.
