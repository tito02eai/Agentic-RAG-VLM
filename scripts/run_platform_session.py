"""Run an Agentic RAG-VLM session against a robot platform.

Offline dry run (no network, deterministic mock transport):

    python scripts/run_platform_session.py \
      --task configs/blupe_stage1_task.example.json \
      --cards knowledge_base/blupe_affordance_cards.json \
      --registry knowledge_base/blupe_semantic_registry.json \
      --mock --output output/blupe_session_dryrun

Remote mode against the BluPe Session API (requires --vlm-endpoint/--vlm-model
for image-based perception; joins the robot queue, then commands the station
through deterministic YAM IK trajectories; see docs/BLUPE_DEPLOYMENT.md):

    export BLUPE_API_TOKEN=...  # optional, only if the deployment requires one
    python scripts/run_platform_session.py \
      --task configs/blupe_stage1_task.example.json \
      --cards knowledge_base/blupe_affordance_cards.json \
      --registry knowledge_base/blupe_semantic_registry.json \
      --platform-config configs/blupe_platform.example.json \
      --vlm-endpoint http://localhost:8000/v1 --vlm-model qwen2.5-vl-7b \
      --output output/blupe_session_run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agentic_rag_vlm.pipeline import AgenticPipeline  # noqa: E402
from agentic_rag_vlm.platform import (  # noqa: E402
    BluPeSession,
    GraspActionMapper,
    MockPlatformTransport,
    PlatformUnavailableError,
    RegistryMetadataPerception,
    REMOTE_VLM_PERCEPTION_PROMPT,
    RemotePlatformTransport,
    SceneChangeVerifier,
    VLMObjectPerception,
    load_platform_config,
)
from agentic_rag_vlm.qwen_adapter import OpenAICompatibleQwenVL  # noqa: E402
from agentic_rag_vlm.runtime import AgenticRuntime, RuntimeConfig, Subgoal, TaskSpec  # noqa: E402
from scripts.agentic_framework import load_semantic_registry  # noqa: E402


DEFAULT_MOCK_SCENE: dict[str, dict[str, list[float]]] = {
    "objects": {
        "cup": [0.30, 0.10, 0.80],
        "vase": [0.42, 0.02, 0.80],
    },
    "destinations": {
        "tray_zone": [-0.25, 0.20, 0.80],
    },
}


def load_task(path: Path) -> tuple[TaskSpec, dict[str, tuple[float, float, float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("task_id", "instruction", "subgoals"):
        if key not in payload:
            raise ValueError(f"task file {path} is missing '{key}'")
    subgoals = tuple(
        Subgoal(str(item["target"]), str(item["destination"])) for item in payload["subgoals"]
    )
    if not subgoals:
        raise ValueError(f"task file {path} must contain at least one subgoal")
    destinations = {
        str(name): (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        for name, xyz in payload.get("destinations", {}).items()
    }
    task = TaskSpec(
        task_id=str(payload["task_id"]),
        instruction=str(payload["instruction"]),
        subgoals=subgoals,
        protected_objects=tuple(str(item) for item in payload.get("protected_objects", [])),
    )
    return task, destinations


def load_mock_scene(path: Path | None) -> dict:
    if path is None:
        return DEFAULT_MOCK_SCENE
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "objects" not in payload or not isinstance(payload["objects"], dict):
        raise ValueError(f"mock scene {path} must contain an 'objects' mapping")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=Path, required=True, help="task spec JSON (target/destination subgoals)")
    parser.add_argument("--cards", type=Path, required=True, help="affordance cards JSON")
    parser.add_argument("--registry", type=Path, required=True, help="semantic registry JSON")
    parser.add_argument("--output", type=Path, required=True, help="output directory for the session result")
    parser.add_argument("--mock", action="store_true", help="offline dry run with the deterministic mock transport")
    parser.add_argument("--scene", type=Path, default=None, help="mock scene JSON (default: built-in stage-1 scene)")
    parser.add_argument("--platform-config", type=Path, default=None, help="platform config JSON for remote mode")
    parser.add_argument("--vlm-endpoint", default=None, help="OpenAI-compatible VLM endpoint for image-based perception")
    parser.add_argument("--vlm-model", default=None, help="VLM model name for image-based perception")
    parser.add_argument("--max-replans", type=int, default=1, help="maximum replans per session (default: 1)")
    parser.add_argument("--arm", choices=("left", "right"), default="right", help="operating arm for remote mode")
    parser.add_argument("--run-duration-s", type=int, default=300, help="remote session duration (60-600 s)")
    parser.add_argument("--queue-timeout-s", type=float, default=600.0, help="remote queue wait timeout")
    parser.add_argument("--no-wait", action="store_true", help="create the remote session without waiting for control")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task, task_destinations = load_task(args.task)
    registry = load_semantic_registry(args.registry)
    cards = json.loads(args.cards.read_text(encoding="utf-8"))["cards"]
    runtime = AgenticRuntime(
        AgenticPipeline(cards),
        RuntimeConfig(maximum_replans=args.max_replans),
    )
    mode = "unknown"
    remote_transport: RemotePlatformTransport | None = None

    if args.mock:
        mode = "mock"
        scene = load_mock_scene(args.scene)
        transport = MockPlatformTransport(
            scene["objects"],
            destinations=scene.get("destinations", {}),
        )
    else:
        if args.platform_config is None:
            print("error: remote mode requires --platform-config (or use --mock for an offline dry run)", file=sys.stderr)
            return 2
        if not (args.vlm_endpoint and args.vlm_model):
            print(
                "error: remote mode requires --vlm-endpoint and --vlm-model "
                "(the station provides images, not object metadata)",
                file=sys.stderr,
            )
            return 2
        mode = "remote"
        config = load_platform_config(args.platform_config)
        remote_transport = RemotePlatformTransport(
            config,
            destinations=task_destinations,
            arm=args.arm,
            run_duration_s=args.run_duration_s,
            prompt=task.instruction,
        )
        transport = remote_transport
        try:
            health = dict(transport.health_check())
            print("health_check:", json.dumps(health, indent=2))
            if not health.get("robot_registered") or not health.get("station_connected"):
                print("error: station not reachable for a session; aborting before queueing", file=sys.stderr)
                return 2
            session_info = dict(
                transport.start_session(timeout_s=args.queue_timeout_s, wait=not args.no_wait)
            )
            print("session:", json.dumps(session_info, indent=2))
        except PlatformUnavailableError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2

    if args.vlm_endpoint and args.vlm_model:
        client = OpenAICompatibleQwenVL(args.vlm_endpoint, args.vlm_model)
        prompt = REMOTE_VLM_PERCEPTION_PROMPT if mode == "remote" else None
        if prompt is not None:
            perception = VLMObjectPerception(client, registry, prompt=prompt)
        else:
            perception = VLMObjectPerception(client, registry)
    else:
        perception = RegistryMetadataPerception(registry)

    session = BluPeSession(
        runtime,
        transport,
        perception,
        GraspActionMapper(command_prefix=f"{task.task_id}"),
        SceneChangeVerifier(),
    )
    try:
        result = session.run(task)
    finally:
        if remote_transport is not None:
            try:
                stop_info = dict(remote_transport.stop_session())
                print("session stopped:", json.dumps(stop_info, indent=2))
            except PlatformUnavailableError as error:
                print(f"warning: session stop failed: {error}", file=sys.stderr)
    result["mode"] = mode
    result["inputs"] = {
        "task": str(args.task),
        "cards": str(args.cards),
        "registry": str(args.registry),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "session_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"outcome={result['outcome']} completed={result['summary']['completed_targets']} receipts={len(result['receipts'])}")
    print(f"result written to {result_path}")
    return 0 if result["outcome"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
