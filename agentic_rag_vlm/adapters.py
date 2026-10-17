"""Interfaces for replacing the local adapter with a learned VLM or real robot."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from scripts.agentic_framework import PublicObject
from .pipeline import GraspAction


class PerceptionAdapter(Protocol):
    def observe(self) -> Mapping[str, PublicObject]: ...


class RobotExecutor(Protocol):
    def execute(self, action: GraspAction) -> Mapping[str, Any]: ...


class PublicVerifier(Protocol):
    def verify(self, before: Mapping[str, PublicObject], after: Mapping[str, PublicObject]) -> Mapping[str, Any]: ...


class PrivateEvaluator(Protocol):
    def evaluate(self, episode_id: str) -> Mapping[str, Any]: ...
