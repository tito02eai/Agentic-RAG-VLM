"""Platform deployment layer for remote-robot sessions (BluPe and mocks).

The framework's decision core (pipeline, runtime, memory, recovery) stays
platform-agnostic.  This subpackage adds the deployment-specific pieces:
configuration, the BluPe Session API client, YAM kinematics and motion
planning, transport, perception, command translation, verification, and the
session driver.  See ``docs/BLUPE_DEPLOYMENT.md`` for the deployment plan and
the scope boundary.
"""

from __future__ import annotations

from .blupe_client import (
    BluPeAPIError,
    BluPeConnectionError,
    BluPeSessionClient,
    SessionHandle,
    TrajectoryBuilder,
    Waypoint,
)
from .commands import GraspActionMapper, PlatformCommand
from .config import PlatformConfig, load_platform_config
from .kinematics import IkResult, YamKinematics
from .motion import MotionKeyframe, MotionPlan, MotionPlanError, YamMotionPlanner
from .perception import (
    PerceptionError,
    PerceptionReceipt,
    RegistryMetadataPerception,
    REMOTE_VLM_PERCEPTION_PROMPT,
    VLMObjectPerception,
    VLM_PERCEPTION_PROMPT,
)
from .session import BluPeSession
from .transport import (
    MockPlatformTransport,
    ObjectMetadata,
    PlatformObservation,
    PlatformTransport,
    PlatformUnavailableError,
    RemotePlatformTransport,
)
from .verifier import SceneChangeVerifier, VerificationResult

__all__ = [
    "BluPeAPIError",
    "BluPeConnectionError",
    "BluPeSession",
    "BluPeSessionClient",
    "GraspActionMapper",
    "IkResult",
    "MockPlatformTransport",
    "MotionKeyframe",
    "MotionPlan",
    "MotionPlanError",
    "ObjectMetadata",
    "PerceptionError",
    "PerceptionReceipt",
    "PlatformCommand",
    "PlatformConfig",
    "PlatformObservation",
    "PlatformTransport",
    "PlatformUnavailableError",
    "RegistryMetadataPerception",
    "REMOTE_VLM_PERCEPTION_PROMPT",
    "RemotePlatformTransport",
    "SceneChangeVerifier",
    "SessionHandle",
    "TrajectoryBuilder",
    "VerificationResult",
    "VLMObjectPerception",
    "VLM_PERCEPTION_PROMPT",
    "Waypoint",
    "YamKinematics",
    "YamMotionPlanner",
    "load_platform_config",
]
