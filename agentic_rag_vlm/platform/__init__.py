"""Platform deployment layer for remote-robot sessions (BluPe and mocks).

The framework's decision core (pipeline, runtime, memory, recovery) stays
platform-agnostic.  This subpackage adds the deployment-specific pieces:
configuration, transport, perception, command translation, verification, and
the session driver.  See ``docs/BLUPE_DEPLOYMENT.md`` for the deployment
plan, scope boundary, and the pending BluPe API questions.
"""

from __future__ import annotations

from .commands import GraspActionMapper, PlatformCommand
from .config import PlatformConfig, load_platform_config
from .perception import (
    PerceptionError,
    PerceptionReceipt,
    RegistryMetadataPerception,
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
    "BluPeSession",
    "GraspActionMapper",
    "MockPlatformTransport",
    "ObjectMetadata",
    "PerceptionError",
    "PerceptionReceipt",
    "PlatformCommand",
    "PlatformConfig",
    "PlatformObservation",
    "PlatformTransport",
    "PlatformUnavailableError",
    "RegistryMetadataPerception",
    "RemotePlatformTransport",
    "SceneChangeVerifier",
    "VerificationResult",
    "VLMObjectPerception",
    "VLM_PERCEPTION_PROMPT",
    "load_platform_config",
]
