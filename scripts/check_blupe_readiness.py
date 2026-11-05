"""GET-only readiness report; no queue entry and no robot movement."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentic_rag_vlm.platform.blupe_client import BluPeSessionClient
from agentic_rag_vlm.platform.config import load_platform_config
from agentic_rag_vlm.platform.deployment import preflight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform-config", type=Path, required=True)
    parser.add_argument("--deployment-profile", type=Path)
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_platform_config(args.platform_config)
    profile = json.loads(args.deployment_profile.read_text()) if args.deployment_profile else None
    client = BluPeSessionClient(config.endpoint_base, request_timeout_s=config.request_timeout_s,
                               max_retries=config.max_retries, auth_token_env=config.auth_token_env)
    result = preflight(client, robot_id=config.robot_id, arm=args.arm, profile=profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["ready_for_session"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
