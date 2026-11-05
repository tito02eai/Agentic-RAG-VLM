# Standalone BluPe client consolidation

The framework and BluPe integration are maintained together at
[tito02eai/Agentic-RAG-VLM](https://github.com/tito02eai/Agentic-RAG-VLM).
The former `tito02eai/agentic-rag-vlm-blupe` repository is superseded. New users
need only this repository and the separately licensed official BluPe Runner.

## What was retained

The consolidation was checked on 2026-11-05 against the standalone client's
final `main` commit, `767c7e309bbe732ab7cfb954f7da254c62b5f9d4` (version 0.1.0),
and this repository's `5ab89a2be4de95a4a653041925553fbd6db1c27a` (version 0.3.5).
All 36 tracked source paths already had counterparts here: 13 were identical
and 23 had newer implementations, configuration, tests or documentation.
No legacy Python class or function name was missing from its corresponding
module. This inventory check is not a claim that all old behavior is unchanged.

| Standalone content | Location in this repository |
|---|---|
| Retrieval, memory, runtime and recovery | `agentic_rag_vlm/` |
| Session API client and transports | `agentic_rag_vlm/platform/blupe_client.py`, `transport.py` |
| IK and joint/gripper trajectories | `agentic_rag_vlm/platform/kinematics.py`, `motion.py`; measured hardware integration also uses `native_kinematics.py` |
| Perception, command mapping and verification | `agentic_rag_vlm/platform/perception.py`, `commands.py`, `session.py`, `verifier.py` |
| Semantic registry and legacy affordance cards | `knowledge_base/blupe_semantic_registry.json`, `blupe_affordance_cards.json` |
| Platform and mock-task examples | `configs/blupe_platform.example.json`, `blupe_stage1_task.example.json` |
| Session CLI and shared object adapter | `scripts/run_platform_session.py`, `scripts/agentic_framework.py` |
| Deployment instructions | [Current deployment entry](BLUPE_DEPLOYMENT.md), [first-trial guide](BLUPE_HARDWARE_TRIAL.md) |
| Five legacy test modules | Same names under `tests/`, with additional contract checks |
| License and packaging | `LICENSE` (Apache-2.0), `pyproject.toml`, `requirements.txt` |

The standalone repository had one branch, no tags, releases, issues or pull
requests at inspection. Its complete Git history was backed up locally before
retirement. Old source and obsolete instructions are not copied into a second
active package: that would create two conflicting implementations.

## Migrating an installation

Use a fresh Python 3.10 environment and clone the canonical repository. The
distribution name is now `agentic-rag-vlm`; the import namespace is still
`agentic_rag_vlm`. Keep the source checkout because configuration, knowledge and
robot assets accompany the editable installation. Follow the root
[installation instructions](../README.md#install-and-check) for the complete
simulation and Runner dependencies.

The lightweight core can still be installed separately for offline API checks:

```bash
python -m pip install -e '.[dev]'
python scripts/run_platform_session.py \
  --task configs/blupe_stage1_task.example.json \
  --cards knowledge_base/blupe_affordance_cards.json \
  --registry knowledge_base/blupe_semantic_registry.json \
  --mock --output output/blupe_mock_migration
```

Use a new output directory for each run. This mock command checks the software
contract with synthetic metadata; it does not demonstrate visual recognition or
physical grasping. The [local grasp demonstration](../README.md#complete-local-grasp-demonstration)
uses the production provider, model services, official SDK and local MuJoCo
contact physics.

## Intentional behavior changes

- **Destinations are checked before motion.** Direct callers of `BluPeSession`
  must supply its `destinations=` mapping as well as the transport's mapping.
  The session uses it for independent placement verification. A missing or
  invalid destination now returns `destination_unavailable` without sending a
  command. The session CLI supplies this mapping from the task file.
- **Mock coordinates do not qualify a hardware task.** The stage-1 example is
  explicitly mock-only. Hardware needs measured `arm_base` destinations, a
  validated station profile, current camera pairing and operator setup. VLM
  metric-coordinate guesses from old examples are not used as calibration.
- **The official Runner manages queue admission and Stop.** The recommended
  entry is `scripts/serve_blupe_hardware.py`, initially in its default read-only
  mode. Opening this monitor does not admit a robot session. See the
  [first-trial guide](BLUPE_HARDWARE_TRIAL.md) before enabling hardware Run.
- **Verification and memory require execution evidence.** Destination checks,
  protected-object visibility, repeated observations and configured release
  confirmation supplement motion receipts. Rejected or unverified actions do
  not become successful memory entries.

## Compatibility checks

All 63 named legacy test functions remain in the five corresponding current
test files. Parameterization produces 65 cases in the original test suite.
Running that unmodified suite against the current implementation gave
**62 passed and 3 failed**. The three failures were the two old session fixtures
that omitted `destinations=`, and the old missing-destination assertion that
expected a later verification failure.

After updating only the session fixture and the expected early-stop assertion,
the same 65 cases passed. The original result, migrated fixture, test logs and
JUnit receipts were preserved in the local consolidation record. The current
suite additionally exercises the stricter behavior; no execution or placement
checks were weakened to make the legacy suite pass.

These are offline software checks. They do not establish public BluPe hardware
performance. The current repository still requires measured site configuration
and a coordinated first hardware trial.
