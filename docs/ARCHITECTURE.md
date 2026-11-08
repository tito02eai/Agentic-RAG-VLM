# Architecture

## Agent workflow

1. Obtain camera observations and measured robot state.
2. Ground registered semantic identities to observable image regions and configured geometry.
3. Retrieve experience using HAA-RAG and compile scene constraints.
4. Review the candidate action with the VLM and map it to an execution command.
5. Execute native robot motion and wait for terminal feedback.
6. Verify the outcome, update memory and monitor the next subgoal.
7. Reobserve and replan within the configured retry budget, or stop when completion is uncertain.

The framework and low-level controller communicate through explicit command/receipt contracts. The local protocol gateway uses actuator-driven MuJoCo physics. The BluPe integration uses the official Runner's session lifecycle and SDK transport.

## Paper mechanisms and code

| Mechanism | Main implementation |
| --- | --- |
| Category and affordance retrieval, CLIP reranking | `agentic_rag_vlm/retrieval.py`, `experience.py` |
| Four-dimensional descriptors and paper agent | `agentic_rag_vlm/paper.py`, `scripts/agentic_framework.py` |
| Spatial graph and constraints | `agentic_rag_vlm/scene.py`, `constraints.py` |
| VLM action review and planning | `agentic_rag_vlm/reasoning.py`, `task_planner.py` |
| Closed-loop task execution | `agentic_rag_vlm/runtime.py`, `platform/session.py` |
| Memory compatibility and invalidation | `agentic_rag_vlm/memory.py` |
| Recovery taxonomy and bounded decisions | `agentic_rag_vlm/recovery.py` |
| Observation and execution verification | `agentic_rag_vlm/platform/perception.py`, `verifier.py` |

HAA-RAG first matches categories, then checks affordance type, material, fragility and graspable region. Compatible records are reranked with CLIP. The current 14-entry engineering store includes source-image and execution provenance.

## Robot execution

`platform/runner_provider.py` implements the official `run_session` hook. The official Runner retains queue admission, leases and Stop. Our provider adopts its session and executes the agent loop.

`platform/native_kinematics.py` uses the published native joint model. `platform/motion.py` converts task targets to audited joint/gripper trajectories. Native deployment uses short Cartesian waypoints for approach, lift and retract. The Runner and standalone CLI share the measured motion configuration.

`platform/transport.py` validates and serializes commands, follows terminal execution feedback and checks the measured retention proxy. `platform/camera_contract.py` checks image/calibration pairing. `platform/geometry.py` projects camera rays using the measured support plane and registered feature heights.

## Knowledge and memory

The engineering store `knowledge_base/yam_verified_diverse_r8.json` contains 14 records. Earlier source images remain where current records or regression tests reference them. Simulation-origin force and hand-shape recommendations are labelled as priors. Hardware joint/gripper commands use the station's measured configuration.

Memory stores verified strategies with an execution-domain identity. Current positions are observed again. Failed reuse invalidates the corresponding memory and emits an audit event.

## Simulation paths

- `simulation/protocol_gateway.py` connects the official HTTP/WebSocket SDK to local actuator physics. This is the primary integration path.
- `simulation/extended.py`, `robust.py` and `guarded.py` support dedicated contact, support, visibility and recovery diagnostics.
- `platform/simulation_runner.py` and selected historical helpers remain for compatibility with preserved tests and scripted diagnostics. The current start commands are indexed in [scripts](../scripts/README.md).

The original manuscript is available under [paper/](../paper/README.md). Original-paper measurements and subsequent engineering validation have separate provenance.
