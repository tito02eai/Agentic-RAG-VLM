"""Observation-to-PublicObject perception paths for platform deployments.

Two auditable paths are provided:

- ``RegistryMetadataPerception``: deterministic.  The platform transport
  reports object metadata (names, positions, optional categories) and the
  frozen semantic registry supplies category/affordance/material/fragility.
- ``VLMObjectPerception``: the platform provides camera images; a VLM lists
  visible task objects as JSON, and the registry normalizes semantic fields
  for known categories (the VLM supplies identities and poses, the registry
  keeps semantics auditable and frozen).

Both paths return the perceived ``PublicObject`` mapping plus a receipt that
records every decision (lookups, overrides, skips) for the session audit
trail.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from scripts.agentic_framework import PublicObject

from .transport import PlatformObservation


class PerceptionError(RuntimeError):
    """Raised when an observation cannot be turned into public objects."""


SEMANTIC_FIELDS = ("category", "superclass", "affordance_type", "material", "fragility", "region")


@dataclass(frozen=True)
class PerceptionReceipt:
    mode: str
    object_count: int
    warnings: tuple[str, ...]
    vlm_result: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "object_count": self.object_count,
            "warnings": list(self.warnings),
            "vlm_result": dict(self.vlm_result) if self.vlm_result is not None else None,
        }


def _registry_indexes(
    registry: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    """Index a semantic registry by entry name and by category."""
    by_name: dict[str, Mapping[str, Any]] = {str(name): entry for name, entry in registry.items()}
    by_category: dict[str, Mapping[str, Any]] = {}
    for entry in registry.values():
        by_category.setdefault(str(entry["category"]), entry)
    return by_name, by_category


class RegistryMetadataPerception:
    """Deterministic metadata + registry perception (no VLM, no network)."""

    def __init__(
        self,
        registry: Mapping[str, Mapping[str, Any]],
        *,
        default_visual_aspect: float = 1.0,
    ) -> None:
        self.by_name, self.by_category = _registry_indexes(registry)
        self.default_visual_aspect = float(default_visual_aspect)

    def perceive(self, observation: PlatformObservation) -> tuple[dict[str, PublicObject], PerceptionReceipt]:
        objects: dict[str, PublicObject] = {}
        warnings: list[str] = []
        for item in observation.objects:
            entry = self.by_name.get(item.name)
            if entry is None and item.category is not None:
                entry = self.by_category.get(item.category)
            if entry is None:
                warnings.append(f"unknown_object_skipped:{item.name}")
                continue
            aspect = item.visual_aspect if item.visual_aspect is not None else self.default_visual_aspect
            objects[item.name] = PublicObject(
                name=item.name,
                center_xyz_m=tuple(float(value) for value in item.center_xyz_m),
                visual_aspect=float(aspect),
                **{key: entry[key] for key in SEMANTIC_FIELDS},
            )
        receipt = PerceptionReceipt(
            mode="registry_metadata",
            object_count=len(objects),
            warnings=tuple(warnings),
        )
        return objects, receipt


class VLMLike(Protocol):
    """Minimal client interface; ``OpenAICompatibleQwenVL`` satisfies this."""

    def infer(self, images: list[Path], prompt: str) -> Any: ...


VLM_PERCEPTION_PROMPT = """You are the perception module of a robot manipulation system.
List every task-relevant object visible in the provided image(s).
Answer with exactly one JSON object, no extra text, in this shape:
{"objects": [{"name": "stable_snake_case_id", "category": "e.g. cup/bottle/banana/cube/vase", "superclass": "e.g. container/rigid_body", "affordance_type": "graspable_body|pinchable|fragile|graspable_edge|tool_handle|wrappable", "material": "e.g. ceramic/plastic/glass/rigid/organic", "fragility": 0.0, "region": "body|edge|handle", "center_xyz_m": [x, y, z], "visual_aspect": 1.0}]}
Rules:
- center_xyz_m is the object center in meters; use the image and any provided workspace bounds to estimate it.
- visual_aspect is height over planar diameter (tall thin objects > 1, flat objects < 1).
- Omit objects that are part of the robot or the fixed environment."""


REMOTE_VLM_PERCEPTION_PROMPT = """You are the perception module of a remote bimanual robot station.
The image(s) come from the station's top camera. Estimate every task-relevant
object's position in the TOP CAMERA OPTICAL FRAME: origin at the camera, x to
the right in the image, y downward in the image, z forward away from the
camera (the distance to the object), all in meters. Typical workspace depths
are 0.3 to 1.2 m.
Answer with exactly one JSON object, no extra text, in this shape:
{"objects": [{"name": "stable_snake_case_id", "category": "e.g. cup/bottle/banana/cube/vase", "superclass": "e.g. container/rigid_body", "affordance_type": "graspable_body|pinchable|fragile|graspable_edge|tool_handle|wrappable", "material": "e.g. ceramic/plastic/glass/rigid/organic", "fragility": 0.0, "region": "body|edge|handle", "center_xyz_m": [x, y, z], "visual_aspect": 1.0}]}
Rules:
- center_xyz_m must be in the top camera optical frame as defined above.
- visual_aspect is height over planar diameter (tall thin objects > 1, flat objects < 1).
- Omit objects that are part of the robot arms, grippers, or the fixed environment."""


class VLMObjectPerception:
    """Image-based perception via a VLM, with registry-normalized semantics."""

    def __init__(
        self,
        client: VLMLike,
        registry: Mapping[str, Mapping[str, Any]],
        *,
        prompt: str = VLM_PERCEPTION_PROMPT,
    ) -> None:
        self.client = client
        self.by_name, self.by_category = _registry_indexes(registry)
        self.prompt = prompt

    def _normalize(
        self,
        parsed_object: Mapping[str, Any],
        observation: PlatformObservation,
        warnings: list[str],
    ) -> PublicObject | None:
        name = str(parsed_object.get("name", "")).strip()
        if not name:
            warnings.append("vlm_object_without_name_skipped")
            return None
        center = parsed_object.get("center_xyz_m")
        if center is None:
            metadata = {item.name: item for item in observation.objects}
            if name in metadata:
                center = list(metadata[name].center_xyz_m)
                warnings.append(f"center_from_metadata:{name}")
            else:
                warnings.append(f"vlm_object_without_position_skipped:{name}")
                return None
        category = str(parsed_object.get("category", ""))
        entry = self.by_name.get(name) or self.by_category.get(category)
        if entry is not None:
            semantic = {key: entry[key] for key in SEMANTIC_FIELDS}
            if category and category != str(entry["category"]):
                warnings.append(f"category_override:{name}:{category}->{entry['category']}")
        else:
            semantic = {key: parsed_object.get(key) for key in SEMANTIC_FIELDS}
            missing = [key for key in SEMANTIC_FIELDS if semantic[key] is None]
            if missing:
                warnings.append(f"vlm_object_missing_semantics_skipped:{name}:{missing}")
                return None
            semantic = {key: str(value) for key, value in semantic.items()}
            warnings.append(f"unregistered_category_used_vlm_semantics:{name}")
        aspect = parsed_object.get("visual_aspect")
        if aspect is None:
            aspect = 1.0
            warnings.append(f"default_visual_aspect:{name}")
        return PublicObject(
            name=name,
            center_xyz_m=(float(center[0]), float(center[1]), float(center[2])),
            visual_aspect=float(aspect),
            **semantic,
        )

    def perceive(self, observation: PlatformObservation) -> tuple[dict[str, PublicObject], PerceptionReceipt]:
        if not observation.image_paths:
            raise PerceptionError("VLM perception requires image paths in the observation")
        result = self.client.infer([Path(path) for path in observation.image_paths], self.prompt)
        parsed = getattr(result, "parsed", None)
        if parsed is None:
            raise PerceptionError(
                f"VLM perception returned unparseable output: {getattr(result, 'parse_error', 'unknown')}"
            )
        listed = parsed.get("objects")
        if not isinstance(listed, list) or not listed:
            raise PerceptionError("VLM perception JSON must contain a non-empty 'objects' list")
        warnings: list[str] = []
        objects: dict[str, PublicObject] = {}
        for item in listed:
            if not isinstance(item, dict):
                warnings.append("vlm_non_object_entry_skipped")
                continue
            public_object = self._normalize(item, observation, warnings)
            if public_object is not None:
                objects[public_object.name] = public_object
        vlm_receipt = {
            "raw_text": getattr(result, "raw_text", None),
            "latency_ms": getattr(result, "latency_ms", None),
            "requested_model": getattr(result, "requested_model", None),
            "response_model": getattr(result, "response_model", None),
            "usage": dict(getattr(result, "usage", {}) or {}),
            "parse_error": getattr(result, "parse_error", None),
        }
        receipt = PerceptionReceipt(
            mode="vlm",
            object_count=len(objects),
            warnings=tuple(warnings),
            vlm_result=vlm_receipt,
        )
        return objects, receipt
