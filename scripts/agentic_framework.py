"""Auditable Agentic RAG-VLM mechanism layer for the Guanghua pilot.

This module intentionally separates the current deterministic semantic adapter
from a future learned VLM adapter.  All planning inputs are derived from public
RGB-D estimates; MuJoCo truth is reserved for the pilot evaluator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import exp
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


SEMANTIC_OBJECTS = {
    "red_cube": {
        "category": "cube",
        "superclass": "rigid_body",
        "affordance_type": "graspable_body",
        "material": "rigid",
        "fragility": 0.1,
        "region": "body",
        "expected_synergy": "power",
    },
    "blue_cylinder": {
        "category": "cylinder",
        "superclass": "rigid_body",
        "affordance_type": "pinchable",
        "material": "rigid",
        "fragility": 0.2,
        "region": "body",
        "expected_synergy": "pinch",
    },
    "fragile_proxy": {
        "category": "cylinder",
        "superclass": "container",
        "affordance_type": "fragile",
        "material": "glass",
        "fragility": 1.0,
        "region": "body",
        "expected_synergy": "do_not_grasp",
    },
}


@dataclass(frozen=True)
class PublicObject:
    name: str
    center_xyz_m: tuple[float, float, float]
    category: str
    superclass: str
    affordance_type: str
    material: str
    fragility: float
    region: str
    visual_aspect: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalResult:
    target: str
    selected_card_id: str
    strategy: dict[str, object]
    ranked_cards: tuple[dict[str, object], ...]
    retrieval_enabled: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def semantic_adapter(estimates: Mapping[str, object]) -> dict[str, PublicObject]:
    """Convert public color/depth estimates into frozen semantic descriptors."""
    objects: dict[str, PublicObject] = {}
    for name, estimate in estimates.items():
        if name not in SEMANTIC_OBJECTS:
            continue
        semantic = SEMANTIC_OBJECTS[name]
        minimum = np.asarray(estimate.visible_surface_min_xyz_m, dtype=float)
        maximum = np.asarray(estimate.visible_surface_max_xyz_m, dtype=float)
        extent = np.maximum(maximum - minimum, 1e-4)
        planar_diameter = max(float(extent[0]), float(extent[1]), 1e-4)
        aspect = float(np.clip(extent[2] / planar_diameter, 0.1, 5.0))
        objects[name] = PublicObject(
            name=name,
            center_xyz_m=tuple(float(v) for v in estimate.center_xyz_m),
            visual_aspect=aspect,
            **{key: semantic[key] for key in (
                "category", "superclass", "affordance_type", "material", "fragility", "region"
            )},
        )
    return objects


def load_cards(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["cards"])


def _category_score(query: PublicObject, card: Mapping[str, object]) -> float:
    if query.category == card["category"]:
        return 1.0
    if query.superclass == card["superclass"]:
        return 0.5
    return 0.0


def _affordance_score(query: PublicObject, card: Mapping[str, object]) -> float:
    return (
        0.5 * float(query.affordance_type == card["affordance_type"])
        + 0.2 * float(query.material == card["material"])
        + 0.15 * (1.0 - min(1.0, abs(query.fragility - float(card["fragility"]))))
        + 0.15 * float(query.region == card["region"])
    )


def _visual_score(query: PublicObject, card: Mapping[str, object]) -> float:
    return float(exp(-abs(query.visual_aspect - float(card["visual_aspect"]))))


def retrieve_strategy(
    query: PublicObject,
    cards: Iterable[dict[str, object]],
    *,
    enabled: bool,
) -> RetrievalResult:
    cards = list(cards)
    if not enabled:
        return RetrievalResult(
            target=query.name,
            selected_card_id="generic_power_default",
            strategy={"synergy": "power", "force_n": 15.0, "aperture_m": 0.07, "approach": "top"},
            ranked_cards=(),
            retrieval_enabled=False,
        )
    # HAA-RAG is hierarchical: category matching is a gate, not merely a
    # low-weight feature.  Fall back to superclass only when the store has no
    # exact-category experience.
    exact_category = [card for card in cards if query.category == card["category"]]
    eligible_cards = exact_category or [
        card for card in cards if query.superclass == card["superclass"]
    ] or cards
    ranking = []
    for card in eligible_cards:
        category = _category_score(query, card)
        affordance = _affordance_score(query, card)
        visual = _visual_score(query, card)
        final = 0.2 * category + 0.4 * affordance + 0.4 * visual
        ranking.append(
            {
                "card_id": card["id"],
                "category_score": category,
                "affordance_score": affordance,
                "visual_score": visual,
                "final_score": final,
                "strategy": card["strategy"],
            }
        )
    ranking.sort(key=lambda item: (-float(item["final_score"]), str(item["card_id"])))
    selected = ranking[0]
    return RetrievalResult(
        target=query.name,
        selected_card_id=str(selected["card_id"]),
        strategy=dict(selected["strategy"]),
        ranked_cards=tuple(ranking[:3]),
        retrieval_enabled=True,
    )


def build_scene_graph(objects: Mapping[str, PublicObject], *, adjacency_m: float = 0.10) -> dict[str, object]:
    edges = []
    names = sorted(objects)
    for index, source in enumerate(names):
        for target in names[index + 1 :]:
            source_xy = np.asarray(objects[source].center_xyz_m[:2])
            target_xy = np.asarray(objects[target].center_xyz_m[:2])
            distance = float(np.linalg.norm(source_xy - target_xy))
            if distance <= adjacency_m:
                edges.append({"source": source, "target": target, "relation": "adjacent_to", "distance_m": distance})
    return {"nodes": [objects[name].to_dict() for name in names], "edges": edges, "adjacency_threshold_m": adjacency_m}


def graph_constraint(
    target: PublicObject,
    objects: Mapping[str, PublicObject],
    graph: Mapping[str, object],
    *,
    enabled: bool,
) -> dict[str, object]:
    neutral = {
        "active": False,
        "reason": "scene_graph_disabled" if not enabled else "no_fragile_neighbor",
        "approach_offset_xy_m": [0.0, 0.0],
        "approach_height_delta_m": 0.0,
        "force_scale": 1.0,
    }
    if not enabled:
        return neutral
    adjacent = any(
        edge["relation"] == "adjacent_to" and target.name in {edge["source"], edge["target"]}
        and "fragile_proxy" in {edge["source"], edge["target"]}
        for edge in graph["edges"]
    )
    if not adjacent or "fragile_proxy" not in objects:
        return neutral
    target_xy = np.asarray(target.center_xyz_m[:2], dtype=float)
    fragile_xy = np.asarray(objects["fragile_proxy"].center_xyz_m[:2], dtype=float)
    away = target_xy - fragile_xy
    norm = float(np.linalg.norm(away))
    if norm <= 1e-6:
        away = np.asarray([-1.0, 0.0])
    else:
        away /= norm
    return {
        "active": True,
        "reason": "fragile_neighbor_within_0.10_m",
        "approach_offset_xy_m": (0.05 * away).round(6).tolist(),
        "approach_height_delta_m": 0.03,
        "force_scale": 0.8,
    }


def detect_scene_change(
    before: Mapping[str, PublicObject],
    after: Mapping[str, PublicObject],
    *,
    threshold_m: float = 0.015,
) -> dict[str, object]:
    displacements = {}
    for name in sorted(set(before) & set(after)):
        delta = np.asarray(after[name].center_xyz_m[:2]) - np.asarray(before[name].center_xyz_m[:2])
        distance = float(np.linalg.norm(delta))
        displacements[name] = {"delta_xy_m": delta.round(6).tolist(), "distance_m": distance}
    stale = [name for name, item in displacements.items() if item["distance_m"] >= threshold_m]
    return {"threshold_m": threshold_m, "displacements": displacements, "stale_targets": stale, "change_detected": bool(stale)}


def condition_flags(condition: str) -> dict[str, bool]:
    return {
        "rag": condition != "A_no_rag",
        "graph": condition != "A_no_graph",
        "memory": condition != "A_no_memory",
        "replan": condition != "A_no_replan",
    }
