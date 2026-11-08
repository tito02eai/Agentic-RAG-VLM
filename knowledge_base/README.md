# Engineering knowledge store

The production provider loads `yam_verified_diverse_r8.json`, containing 14
provenance-labelled records. Entries include semantic/affordance attributes,
grasp strategy, visual embedding and references to observed source evidence.

The four compatibility dimensions are affordance type, material, fragility and
graspable region. CLIP reranks compatible records. The expected visual space is
the pinned CLIP ViT-L/14 encoder exposed by `scripts/serve_clip.py`.

Referenced source images and earlier stores remain available for current
records, evidence readers and regression tests. The `blupe_*` semantic registry
and affordance cards also support explicitly labelled contract/mock workflows.

The records originate from engineering simulation execution. Their motor-force
and hand-shape recommendations retain that execution-domain identity. Hardware
experience should use independently verified outcomes and its own calibration
and execution-domain binding. See [Architecture](../docs/ARCHITECTURE.md) and
[Reproduction](../docs/REPRODUCTION.md).
