# Local BluPe-interface demonstrations

These are **our local actuator-driven MuJoCo runs**, using the production
Agentic RAG-VLM provider and official BluPe HTTP/WebSocket SDK. They are not
public hardware videos or an official BluPe cloud simulator. Videos use
simulation time and omit model waiting; they do not show wall-clock speed.

- [Complete protected-neighbor pick/place](local_pick_place.mp4): configurable
  RGB proposals, native IK/joint execution and verified placement.
  [Original run receipt](local_pick_place_receipt.json).
- [Moved-target memory reuse and replanning](local_memory_replan.mp4): after the
  first verified placement, an explicitly injected external force moves the
  second object about 5.5 cm. The agent reobserves, replans and reuses grasp
  experience with the fresh position. Both placements pass independent checks.
- [0.3.4 robustness audit](robustness_audit_0.3.4.json): ten local criteria pass,
  with four completed tasks and six expected stops. Stops are not successful
  grasps. Full raw-case archives and original failures are retained separately.
- [0.3.5 verification](verification_0.3.5.json): recorded software tests and
  compatibility results. No public robot session or hardware motion occurred.
- [2026-11-08 native-path acceptance](pre_hardware_20261108.json): six local
  grasp/place completions, one partial-ROI rejection and ten separate fault/
  memory criteria, with 60 actual Qwen replies. Native approach/lift/retract
  paths and shared CLI/Runner motion configuration were checked.

Run fresh cases with the repository README and a new output directory each time.
Paper-era measurements are separate from these engineering results.
