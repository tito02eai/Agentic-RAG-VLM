"""Platform command intermediate representation and GraspAction translation.

``PlatformCommand`` is the framework's own action schema (target selection,
grasp parameters, scene-graph constraint, and optional placement destination).
A platform-specific transport translates it into the remote API's primitive
calls.  Keeping this layer explicit means that when the BluPe API contract
arrives, only the transport translation changes — the pipeline, mapper, and
session code stay frozen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..pipeline import GraspAction


COMMAND_TYPES = ("pick", "place", "home")


@dataclass(frozen=True)
class PlatformCommand:
    command_id: str
    command_type: str
    target: str
    origin: str
    parameters: dict[str, Any] = field(default_factory=dict)
    constraint: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GraspActionMapper:
    """Translate pipeline ``GraspAction`` outputs into platform commands."""

    def __init__(self, *, command_prefix: str = "cmd", retention_force_adjustment: bool = False) -> None:
        self.command_prefix = command_prefix
        self._counter = 0
        self.retention_force_adjustment = retention_force_adjustment

    def _next_id(self, command_type: str) -> str:
        self._counter += 1
        return f"{self.command_prefix}_{self._counter:04d}_{command_type}"

    def pick_command(self, action: GraspAction) -> PlatformCommand:
        strategy = dict(action.strategy)
        parameters = {
            "position_xyz_m": list(action.position_xyz_m),
            "synergy": strategy.get("synergy", "power"),
            "force_n": float(strategy.get("force_n", 10.0)),
            "aperture_m": float(strategy.get("aperture_m", 0.07)),
            "approach": strategy.get("approach", "top"),
            "velocity_scale": float(strategy.get("velocity_scale", 1)),
            "approach_height_delta_m": float(strategy.get("approach_height_delta_m", 0)),
            "approach_offset_xy_m": action.constraint.get("approach_offset_xy_m", [0.0, 0.0]),
            "approach_yaw_delta_rad":float(strategy.get("approach_yaw_delta_rad",0)),
            "wrist_yaw_delta_rad":float(strategy.get("wrist_yaw_delta_rad",0)),
        }
        if self.retention_force_adjustment:
            signature = strategy.get('_semantic_signature', [])
            parameters['retention_force_policy'] = {'maximum_adjustments': 3, 'scale': 1.2,
                'ceiling_n': 1.6, 'maximum_fragility': .25,
                'observed_fragility': signature[3] if len(signature) > 3 else 1.}
        return PlatformCommand(
            command_id=self._next_id("pick"),
            command_type="pick",
            target=action.target,
            origin=action.source,
            parameters=parameters,
            constraint=dict(action.constraint),
        )

    def place_command(
        self,
        action: GraspAction,
        destination: str,
        *,
        destination_xyz_m: tuple[float, float, float] | None = None,
    ) -> PlatformCommand:
        parameters: dict[str, Any] = {
            "destination": destination,
            "aperture_m": float(action.strategy.get("aperture_m", 0.07)),
            "approach": action.strategy.get("approach", "top"),
            "velocity_scale": float(action.strategy.get("velocity_scale", 1)),
        }
        if destination_xyz_m is not None:
            parameters["destination_xyz_m"] = list(destination_xyz_m)
        return PlatformCommand(
            command_id=self._next_id("place"),
            command_type="place",
            target=action.target,
            origin=action.source,
            parameters=parameters,
            constraint=dict(action.constraint),
        )

    def home_command(self) -> PlatformCommand:
        return PlatformCommand(
            command_id=self._next_id("home"),
            command_type="home",
            target="",
            origin="session",
            parameters={},
            constraint={},
        )
