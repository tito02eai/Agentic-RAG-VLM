"""Platform deployment configuration for remote-robot sessions.

The framework itself stays platform-agnostic; this module carries the
deployment-specific settings (endpoint, auth token environment variable,
robot and camera identifiers) loaded from a JSON file.  Secrets are never
stored in the repository: the configuration only names the environment
variable that a transport reads at call time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PlatformConfig:
    platform: str
    auth_token_env: str
    endpoint_base: str | None = None
    robot_id: str | None = None
    camera_ids: tuple[str, ...] = ()
    request_timeout_s: float = 30.0
    max_retries: int = 2
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["camera_ids"] = list(self.camera_ids)
        payload["extra"] = dict(self.extra)
        return payload


_REQUIRED_FIELDS = ("platform", "auth_token_env")


def load_platform_config(path: Path) -> PlatformConfig:
    """Load and validate a platform configuration JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [key for key in _REQUIRED_FIELDS if key not in payload]
    if missing:
        raise ValueError(f"platform config {path} is missing fields: {missing}")
    endpoint = payload.get("endpoint_base")
    camera_ids = tuple(str(item) for item in payload.get("camera_ids", []))
    retries = int(payload.get("max_retries", 2))
    timeout = float(payload.get("request_timeout_s", 30.0))
    if retries < 0:
        raise ValueError(f"platform config {path}: max_retries must be >= 0")
    if timeout <= 0.0:
        raise ValueError(f"platform config {path}: request_timeout_s must be > 0")
    return PlatformConfig(
        platform=str(payload["platform"]),
        auth_token_env=str(payload["auth_token_env"]),
        endpoint_base=str(endpoint) if endpoint is not None else None,
        robot_id=str(payload["robot_id"]) if payload.get("robot_id") is not None else None,
        camera_ids=camera_ids,
        request_timeout_s=timeout,
        max_retries=retries,
        extra=dict(payload.get("extra", {})),
    )
