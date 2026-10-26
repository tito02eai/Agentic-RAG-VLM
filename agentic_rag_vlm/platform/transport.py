"""Transport layer: normalized observations, command dispatch, and receipts.

``PlatformTransport`` normalizes whatever the remote platform provides into
``PlatformObservation`` (object metadata + camera image paths + receipt).  Two
implementations are provided:

- ``MockPlatformTransport``: a deterministic offline transport used for
  protocol qualification and dry runs without any network access.
- ``RemotePlatformTransport``: a placeholder that fails fast until the BluPe
  API contract (observation format, control primitive level, safety limits,
  quota) is received and implemented.  The four open API questions are listed
  in ``docs/BLUPE_DEPLOYMENT.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .commands import PlatformCommand
from .config import PlatformConfig


class PlatformUnavailableError(RuntimeError):
    """Raised when a remote platform transport is used before its API contract is implemented."""


@dataclass(frozen=True)
class ObjectMetadata:
    name: str
    center_xyz_m: tuple[float, float, float]
    category: str | None = None
    visual_aspect: float | None = None


@dataclass(frozen=True)
class PlatformObservation:
    observation_id: str
    objects: tuple[ObjectMetadata, ...]
    image_paths: tuple[str, ...] = ()
    receipt: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "objects": [
                {
                    "name": item.name,
                    "center_xyz_m": list(item.center_xyz_m),
                    "category": item.category,
                    "visual_aspect": item.visual_aspect,
                }
                for item in self.objects
            ],
            "image_paths": list(self.image_paths),
            "receipt": dict(self.receipt),
        }


class PlatformTransport(Protocol):
    def health_check(self) -> Mapping[str, Any]: ...

    def get_observation(self) -> PlatformObservation: ...

    def send_command(self, command: PlatformCommand) -> Mapping[str, Any]: ...


class RemotePlatformTransport:
    """Fails fast; implement against the BluPe API once its contract arrives.

    Pending API questions (tracked in docs/BLUPE_DEPLOYMENT.md):
    1. observation format: RGB-D or RGB-only, camera views, object metadata;
    2. control primitive level (named skills vs. Cartesian/joint targets);
    3. safety limits and interruption semantics;
    4. quota, rate limits, and open-source requirements.
    """

    def __init__(self, config: PlatformConfig) -> None:
        self.config = config

    def _fail(self, operation: str) -> None:
        raise PlatformUnavailableError(
            f"remote platform '{self.config.platform}' operation '{operation}' is not implemented: "
            "the BluPe API contract has not been received yet; see docs/BLUPE_DEPLOYMENT.md"
        )

    def health_check(self) -> Mapping[str, Any]:
        self._fail("health_check")

    def get_observation(self) -> PlatformObservation:
        self._fail("get_observation")

    def send_command(self, command: PlatformCommand) -> Mapping[str, Any]:
        self._fail("send_command")


class MockPlatformTransport:
    """Deterministic offline transport for dry runs and regression tests.

    The mock keeps a table-top scene as named 3-D positions.  ``pick`` grips
    the target without moving it, ``place`` moves it to the resolved
    destination, and ``home`` is a no-op, so a successful pick-and-place is
    observable as target displacement while a rejected place leaves the
    target in place (matching the scene-change verifier's criterion).
    Every command returns a receipt so the session audit trail is exercised
    end-to-end without network access.
    """

    def __init__(
        self,
        scene: Mapping[str, Sequence[float]],
        *,
        destinations: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        self._positions: dict[str, tuple[float, float, float]] = {
            str(name): (float(xyz[0]), float(xyz[1]), float(xyz[2]))
            for name, xyz in scene.items()
        }
        self._destinations: dict[str, tuple[float, float, float]] = {
            str(name): (float(xyz[0]), float(xyz[1]), float(xyz[2]))
            for name, xyz in (destinations or {}).items()
        }
        self._observation_counter = 0
        self.command_log: list[dict[str, Any]] = []

    def health_check(self) -> Mapping[str, Any]:
        return {"ok": True, "transport": "mock"}

    def get_observation(self) -> PlatformObservation:
        self._observation_counter += 1
        objects = tuple(
            ObjectMetadata(name=name, center_xyz_m=xyz)
            for name, xyz in sorted(self._positions.items())
        )
        return PlatformObservation(
            observation_id=f"mock_obs_{self._observation_counter:04d}",
            objects=objects,
            receipt={"transport": "mock", "observation_id": f"mock_obs_{self._observation_counter:04d}"},
        )

    def send_command(self, command: PlatformCommand) -> Mapping[str, Any]:
        receipt: dict[str, Any] = {
            "transport": "mock",
            "command_id": command.command_id,
            "command_type": command.command_type,
            "accepted": True,
        }
        if command.command_type == "pick":
            if command.target not in self._positions:
                receipt["accepted"] = False
                receipt["reason"] = "target_not_in_scene"
            else:
                receipt["detail"] = "gripped_without_displacement"
        elif command.command_type == "place":
            if command.target not in self._positions:
                receipt["accepted"] = False
                receipt["reason"] = "target_not_in_scene"
            else:
                resolved = command.parameters.get("destination_xyz_m")
                destination_name = str(command.parameters.get("destination", ""))
                if resolved is None and destination_name in self._destinations:
                    resolved = list(self._destinations[destination_name])
                if resolved is None:
                    receipt["accepted"] = False
                    receipt["reason"] = "unresolved_destination"
                else:
                    self._positions[command.target] = (
                        float(resolved[0]),
                        float(resolved[1]),
                        float(resolved[2]),
                    )
                    receipt["target_placed_at_m"] = list(self._positions[command.target])
        elif command.command_type == "home":
            receipt["detail"] = "no-op"
        else:
            receipt["accepted"] = False
            receipt["reason"] = f"unknown_command_type:{command.command_type}"
        self.command_log.append({**command.to_dict(), "receipt": receipt})
        return receipt
