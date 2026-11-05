"""Fresh public perception packet for full replanning (no simulator truth)."""
from dataclasses import dataclass,field
from pathlib import Path
from typing import Mapping,Any
from scripts.agentic_framework import PublicObject


@dataclass(frozen=True)
class PlanningObservation:
    objects: Mapping[str,PublicObject]
    visual_embeddings: Mapping[str,list[float]]
    embedding_space: str
    observation_id: str
    scene_context: Mapping[str,Any]=field(default_factory=dict)
    images: tuple[Path,...]=()
