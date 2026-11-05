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
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Protocol
import numpy as np

from scripts.agentic_framework import PublicObject

from .transport import PlatformObservation
from .geometry import PlaneProjector, GeometryError


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
            semantic = {key: float(value) if key == "fragility" else str(value) for key, value in semantic.items()}
            warnings.append(f"unregistered_category_used_vlm_semantics:{name}")
        aspect = parsed_object.get("visual_aspect")
        if aspect is None:
            aspect = 1.0
            warnings.append(f"default_visual_aspect:{name}")
        if len(center) != 3 or not all(isfinite(float(v)) for v in center):
            raise PerceptionError(f"invalid metric position: {name}")
        if not isfinite(float(aspect)) or float(aspect) <= 0:
            raise PerceptionError(f"invalid visual aspect: {name}")
        if not isfinite(float(semantic["fragility"])) or not 0 <= float(semantic["fragility"]) <= 1:
            raise PerceptionError(f"invalid fragility: {name}")
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


class GroundedVLMPerception(VLMObjectPerception):
    """VLM identity/pixel predictions with measured-plane metric grounding."""

    def __init__(self, client: VLMLike, registry: Mapping[str, Mapping[str, Any]],
                 projector: PlaneProjector, feature_heights_m: Mapping[str, float], *, visual_encoder=None, candidate_provider=None) -> None:
        super().__init__(client, registry)
        self.projector = projector
        self.heights = dict(feature_heights_m)
        self.allowed_ids=sorted(set(self.by_name)&set(self.heights))
        if not self.allowed_ids:raise ValueError('at least one registered object with measured feature height is required')
        self.visual_encoder=visual_encoder
        self.candidate_provider=candidate_provider
        self.prompt = (
            "Detect visible objects in the TOP camera image. Return JSON only: "
            '{"objects":[{"name":"registered_id","pixel_xy":[u,v],"visual_aspect":1.0}]}. '
            "Use exact IDs from this measured scene catalogue: " + ", ".join(self.allowed_ids) + ". "
            "pixel_xy must identify the centre of the visible top feature of the object, "
            "in ORIGINAL image pixels (x right, y down). Do not invent hidden objects "
            "or output metric positions. If an identity is ambiguous, omit it. "
            'Also return optional "states":{"id":"filled|empty|stacked|unknown"} and '
            '"relations":[{"source":"id","target":"id","relation":"supports|contains|occludes|adjacent_to|on_top_of|behind"}] '
            "only for visible evidence; omit uncertain relations. "
            f"Original resolution: {projector.image_size.tolist()}."
        )
        if visual_encoder is not None:
            self.prompt+=' Include optional bbox_xyxy=[left,top,right,bottom] in ORIGINAL pixels for each visible object, containing its pixel_xy.'

    def perceive(self, observation: PlatformObservation) -> tuple[dict[str, PublicObject], PerceptionReceipt]:
        if observation.receipt.get("calibration_id") != self.projector.calibration_id:
            raise PerceptionError("calibration changed or observation has no calibration identity")
        if observation.receipt.get("camera_role") != "top" or len(observation.image_paths) != 1:
            raise PerceptionError("grounded perception requires one identified top-camera image")
        from PIL import Image
        with Image.open(observation.image_paths[0]) as image:
            if tuple(image.size) != tuple(self.projector.image_size):
                raise PerceptionError("capture resolution does not match calibrated intrinsics")
        candidates=observation.receipt.get('object_candidates')
        if candidates is None and self.candidate_provider is not None:
            candidates=self.candidate_provider(observation)
        proposal_objects=None;prompt=self.prompt;images=[Path(observation.image_paths[0])]
        if candidates is not None:
            if not isinstance(candidates,list) or not 1<=len(candidates)<=8:
                raise PerceptionError('one to eight public object proposals required')
            ids=[];images=[];proposal_dir=Path(observation.image_paths[0]).parent/'public_proposals'
            proposal_dir.mkdir(exist_ok=True)
            with Image.open(observation.image_paths[0]) as image:
                width,height=image.size
                for i,candidate in enumerate(candidates):
                    name=candidate.get('candidate_id') if isinstance(candidate,dict) else None
                    box=candidate.get('bbox_xyxy',[]) if isinstance(candidate,dict) else []
                    pixel=candidate.get('pixel_xy',[]) if isinstance(candidate,dict) else []
                    if (not isinstance(name,str) or not name or name in ids or len(box)!=4 or len(pixel)!=2
                            or not all(type(v) in (int,float) and isfinite(v) for v in box+pixel)
                            or not 0<=box[0]<box[2]<=width or not 0<=box[1]<box[3]<=height
                            or box[2]-box[0]<1 or box[3]-box[1]<1
                            or not box[0]<=pixel[0]<=box[2] or not box[1]<=pixel[1]<=box[3]):
                        raise PerceptionError('invalid public object proposal')
                    ids.append(name);path=proposal_dir/f'proposal_{i}.png'
                    from PIL import ImageOps
                    padded=[max(0,box[0]-8),max(0,box[1]-8),min(width,box[2]+8),min(height,box[3]+8)]
                    ImageOps.pad(image.crop(tuple(int(v) for v in padded)).convert('RGB'),(192,192),color=(128,128,128)).save(path)
                    images.append(path)
            catalogue=', '.join(label+' ('+str(self.by_name[label].get('visual_description',self.by_name[label]['category']))+')'
                for label in self.allowed_ids)
            prompt=('These are object candidate crops in order '+', '.join(ids)+'. '
                'Classify using exact registered/measured scene IDs: '+catalogue+', or unknown. '
                'Return exactly ONE flat JSON object mapping every candidate ID to one label. '
                'No coordinates, arrays or explanations. A candidate has no supplied semantic identity. '
                'Example: {"candidate_0":"cube","candidate_1":"vase"}.')
        result = self.client.infer(images, prompt)
        from ..experience import atomic_json
        raw_receipt={key:getattr(result,key,None) for key in ('parsed','raw_text','requested_model','response_model','latency_ms','parse_error')}
        atomic_json(Path(observation.image_paths[0]).parent/'grounded_vlm_response.json',
            {'prompt':prompt,'observation_id':observation.observation_id,**raw_receipt})
        parsed = getattr(result, "parsed", None)
        if candidates is not None:
            if not isinstance(parsed,dict) or set(parsed)!=set(ids):
                raise PerceptionError('model must classify exactly the public proposal IDs')
            proposal_objects=[];assigned=set()
            for candidate in candidates:
                label=parsed[candidate['candidate_id']]
                if label=='unknown':continue
                if not isinstance(label,str) or label not in self.by_name or label in assigned:
                    raise PerceptionError('ambiguous or unsupported proposal identity')
                assigned.add(label);proposal_objects.append({'name':label,'pixel_xy':candidate['pixel_xy'],
                    'bbox_xyxy':candidate['bbox_xyxy']})
            context=observation.receipt.get('scene_context',{})
            if not isinstance(context,dict):raise PerceptionError('invalid proposal scene context')
            parsed={'objects':proposal_objects,'states':context.get('states',{}),'relations':context.get('relations',[])}
        if not isinstance(parsed, dict) or not isinstance(parsed.get("objects"), list):
            raise PerceptionError("expected VLM objects list")
        objects: dict[str, PublicObject] = {}
        warnings: list[str] = []
        grounding = []
        for raw in parsed["objects"]:
            if not isinstance(raw, dict):
                raise PerceptionError("invalid object detection")
            name = raw.get("name")
            if name not in self.by_name or name not in self.heights:
                raise PerceptionError(f"unregistered or unmeasured detection: {name}")
            if name in objects:
                raise PerceptionError(f"ambiguous duplicate identity: {name}")
            try:
                position = self.projector.project_base(raw["pixel_xy"], height_m=self.heights[name])
                item = self._normalize({**raw, "center_xyz_m": position}, observation, warnings)
            except (KeyError, ValueError, TypeError, GeometryError) as error:
                raise PerceptionError(f"invalid geometric detection for {name}: {error}") from error
            if item is None:
                raise PerceptionError(f"failed to normalize detection: {name}")
            objects[name] = item
            grounding.append({"name": name, "pixel_xy": raw["pixel_xy"],
                              "measured_feature_height_m": self.heights[name],
                              "arm_base_xyz_m": list(position), "coordinate_frame": "arm_base"})
        states = parsed.get("states", {})
        relations = parsed.get("relations", [])
        if not isinstance(states, dict) or not isinstance(relations, list):
            raise PerceptionError("invalid scene context")
        if any(name not in objects or state not in {"filled", "empty", "stacked", "unknown"} for name, state in states.items()):
            raise PerceptionError("scene state references unknown object or state")
        from ..scene import RELATIONS
        for edge in relations:
            if (not isinstance(edge, dict) or edge.get("source") not in objects or edge.get("target") not in objects
                    or edge.get("relation") not in RELATIONS):
                raise PerceptionError("invalid scene relation")
        features={};embedding_space=self.visual_encoder.embedding_space if self.visual_encoder is not None else None
        if self.visual_encoder is not None and objects:
            import hashlib
            crops=[];records=[]
            folder=Path(observation.image_paths[0]).parent/'agent_crops'/hashlib.sha256(observation.observation_id.encode()).hexdigest()[:16]
            folder.mkdir(parents=True,exist_ok=True)
            with Image.open(observation.image_paths[0]) as image:
                width,height=image.size
                raw_by_name={r['name']:r for r in parsed['objects']}
                for i,g in enumerate(grounding):
                    raw=raw_by_name[g['name']];u,v=raw['pixel_xy']
                    box=raw.get('bbox_xyxy',[max(0,u-24),max(0,v-24),min(width,u+24),min(height,v+24)])
                    if (not isinstance(box,list) or len(box)!=4
                            or any(type(x) not in (int,float) or not np.isfinite(x) for x in box)
                            or not 0<=box[0]<box[2]<=width or not 0<=box[1]<box[3]<=height
                            or not box[0]<=u<=box[2] or not box[1]<=v<=box[3]):
                        raise PerceptionError('invalid object crop bounds')
                    path=folder/f'candidate_{i}.png'
                    from PIL import ImageOps
                    padded=[max(0,box[0]-8),max(0,box[1]-8),min(width,box[2]+8),min(height,box[3]+8)]
                    ImageOps.pad(image.crop(tuple(int(v) for v in padded)).convert('RGB'),(192,192),color=(128,128,128)).save(path)
                    g.update(candidate_id=f'candidate_{i}',crop_path=str(path),crop_bbox_xyxy=padded,
                        object_bbox_xyxy=box,crop_size_px=[192,192],crop_context_px=8,
                        crop_source='vlm_bbox' if 'bbox_xyxy' in raw else 'declared_48px_context_at_measured_feature',
                        image_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    crops.append(path);records.append(g['name'])
            vectors=self.visual_encoder.encode(crops)
            if np.asarray(vectors).shape!=(len(crops),768) or not np.isfinite(vectors).all():
                raise PerceptionError('invalid CLIP embedding batch')
            features={name:np.asarray(vec).tolist() for name,vec in zip(records,vectors)}
            embedding_space=self.visual_encoder.embedding_space
        return objects, PerceptionReceipt(
            "vlm_measured_plane", len(objects), tuple(warnings), {
                "raw_text": getattr(result, "raw_text", None),
                "requested_model": getattr(result, "requested_model", None),
                "latency_ms": getattr(result, "latency_ms", None),
                "calibration_id": self.projector.calibration_id,
                "grounding": grounding,
                "scene_context": {"states": states, "relations": relations},
                "visual_embeddings":features,"embedding_space":embedding_space,
                "response_model":getattr(result,'response_model',None),
                "usage":dict(getattr(result,'usage',{}) or {}),
                "identity_source":'vlm_candidate_classification' if candidates is not None else 'vlm_full_frame_detection',
                "effective_objects":parsed['objects'],
            },
        )
