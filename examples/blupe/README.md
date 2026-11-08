# Current demonstrations

Both videos were recorded by the final-source 2026-11-08 qualification using the production Agentic RAG-VLM provider, official BluPe HTTP/WebSocket SDK and local actuator-driven MuJoCo backend.

| Demonstration | Video | Receipt |
| --- | --- | --- |
| Protected-neighbor grasp/place, seed 1040 | [16.08 seconds](local_pick_place.mp4) | [Execution receipt](local_pick_place_receipt.json) |
| Two-object memory reuse and moved-target replanning | [29.80 seconds](local_memory_replan.mp4) | [Execution receipt](local_memory_replan_receipt.json) |

## Grasp/place sequence

The agent identifies a rigid object, retrieves compatible experience, checks the neighboring protected object, reviews the action with the VLM and executes grasp, lift, transfer, release and retreat. Placement, support, release, retention and contact conditions are checked independently.

## Memory/replanning sequence

The first placement is verified. A declared physical external-force disturbance moves the pending second object. The agent observes its fresh position, replans and reuses the compatible strategy. Both placements are checked. This disturbance is an explicit test condition.

Videos use simulation time and omit model waiting. The [selected qualification receipt](pre_hardware_20261108.json) gives source hashes, actual model identities and the scope of the normal-task and fault cohorts. Public hardware execution is covered by the separate [first-trial setup](../../docs/BLUPE.md).

Earlier video versions and large raw evidence are retained in the local archive and earlier Git revisions.
