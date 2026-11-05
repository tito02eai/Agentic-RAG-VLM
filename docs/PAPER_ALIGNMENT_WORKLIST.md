# Paper-aligned reconstruction — active worklist

Authority: `paper/main_final.tex`, Sections III–IV and Algorithm 1. The original
116-entry experience store, original model prompts, CLIP vectors, trial logs,
and complete seven-factor formulas are unavailable. They must not be fabricated.

Required implementation and acceptance:

- [x] Typed, provenance-bearing experience store (category, 4D affordance,
  strategy, binary outcome, CLIP vector); retain failed experiences.
- [x] HAA-RAG: category 1/0.5 gate, at most 30 candidates, affordance threshold
  0.3, actual CLIP ViT-L/14 cosine, 0.2/0.4/0.4 fusion, failed score ×0.7, top 3.
- [x] Session memory: verified success only, category indexing, intrinsic
  parameters reused and current scene constraints reapplied; continuous-task test.
- [x] Typed scene graph, explicit observed states/relations, 0.15 m analysis,
  filled-content/fragile-neighbor/support-prerequisite/occlusion adjustments.
  Ten of the paper's 16 relation names are unspecified; label additional names
  as reconstruction choices. Physical path validation precedes lateral motion.
- [x] Actual VLM Thought/Action/Observation/Reflection receipts, bounded by
  initial attempt plus three retries; 14 failure types; L1 quantitative rules,
  L2 method rotation and ×0.9 force, L3 fresh plan plus seeded U(-0.01,0.01)^3.
- [x] Seven-factor weighted evaluator with declared measurement adapters and
  configurable thresholds. Undocumented individual formulas/defaults are
  reconstruction choices, not recovered original code.
- [x] Executable integration with complete contact-dynamics tasks; test genuine
  within-session reuse, rejection of incompatible memory, and recovery contracts.
- [x] Documentation and source archive; fresh-extraction verification is recorded
  in the r3 delivery manifest after packaging.

Historical scripts and frozen experiment evidence remain unchanged. New strict
paper-aligned entry points must fail explicitly when CLIP or experience evidence
is unavailable; aspect-ratio proxy results cannot be relabelled as CLIP results.

Implementation acceptance: 180 tests pass; 10 historical-asset tests skip.
Final paired physical qualification: `memory_paired_final_r2`, seeds 40–44,
5/5 tasks per memory condition, 20 completed grasps. See
`PAPER_ALIGNMENT_RESULTS_20261102.md`. Check marks mean the software mechanism
and its stated acceptance are complete, not that missing original datasets or
all physical constraint/recovery demonstrations were recovered.
