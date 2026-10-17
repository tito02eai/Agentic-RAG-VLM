"""Session-level episodic memory described in Sec. IV-C of the paper."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any


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
        candidate = MemoryEntry(category, dict(strategy), float(quality), source_episode)
        previous = self._entries.get(category)
        if previous is None or candidate.quality >= previous.quality:
            self._entries[category] = candidate

    def get(self, category: str) -> MemoryEntry | None:
        return self._entries.get(category)

    def to_dict(self) -> dict[str, object]:
        return {key: asdict(value) for key, value in sorted(self._entries.items())}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EpisodicMemory":
        memory = cls()
        if not path.exists():
            return memory
        for category, item in json.loads(path.read_text(encoding="utf-8")).items():
            memory.put(category, item["strategy"], item["quality"], item["source_episode"])
        return memory
