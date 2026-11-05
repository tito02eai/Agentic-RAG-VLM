"""Session-level episodic memory described in Sec. IV-C of the paper."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any
from copy import deepcopy
import math
from .experience import atomic_json


@dataclass(frozen=True)
class MemoryEntry:
    category: str
    strategy: dict[str, Any]
    quality: float
    source_episode: str


class EpisodicMemory:
    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}

    def put(self, category: str, strategy: dict[str, Any], quality: float, source_episode: str) -> None:
        if not category or not source_episode or not math.isfinite(quality) or not 0 <= quality <= 1:
            raise ValueError("memory requires category, episode and finite quality in [0,1]")
        candidate = MemoryEntry(category, deepcopy(strategy), float(quality), source_episode)
        previous = self._entries.get(category)
        if previous is None or candidate.quality >= previous.quality:
            self._entries[category] = candidate

    def get(self, category: str) -> MemoryEntry | None:
        return deepcopy(self._entries.get(category))

    def invalidate(self, category: str) -> MemoryEntry | None:
        """Drop stale session parameters; the source experience stays in the KB."""
        return self._entries.pop(category,None)

    def to_dict(self) -> dict[str, object]:
        return {key: asdict(value) for key, value in sorted(self._entries.items())}

    def save(self, path: Path) -> None:
        atomic_json(path,self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "EpisodicMemory":
        memory = cls()
        if not path.exists():
            return memory
        for category, item in json.loads(path.read_text(encoding="utf-8")).items():
            memory.put(category, item["strategy"], item["quality"], item["source_episode"])
        return memory
