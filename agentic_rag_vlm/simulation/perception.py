"""VLM identity classification with restricted RGB-D candidate proposals.

Object centres use rendered depth, calibrated camera geometry and known object
dimensions. The model never supplies metric positions. Missing/ambiguous model
labels stop the session; diagnostic color identities are never a fallback.
"""
from pathlib import Path
from ..platform.perception import VLMObjectPerception, PerceptionReceipt, PerceptionError


class CandidateRGBDVLMPerception(VLMObjectPerception):
    def __init__(self, client, registry, half_heights, *, visual_encoder=None, catalogue=None, identity_reviewer=None, cylinder_models=None):
        super().__init__(client, registry)
        self.half_heights = dict(half_heights)
        self.visual_encoder = visual_encoder
        self.catalogue = catalogue or "cube, vase (a cylindrical vessel proxy), or unknown"
        self.identity_reviewer=identity_reviewer
        self.cylinder_models = dict(cylinder_models or {})

    def perceive(self, observation):
        candidates = observation.receipt.get("rgbd_candidates", [])
        if not candidates:
            raise PerceptionError("no visible RGB-D candidates")
        ids = [item["candidate_id"] for item in candidates]
        prompt = (
            "These are object candidate crops in order " + ", ".join(ids) + ". "
            "Classify each using these exact IDs: " + self.catalogue + ". "
            "Reply with exactly one flat JSON object mapping candidate IDs to labels. "
            'Example format: {"candidate_0":"cube","candidate_1":"vase"}. '
            "Only include IDs that were provided. No array, no coordinates, no explanation."
        )
        if observation.receipt.get("candidate_preprocessing"):
            prompt += (" The images are visible FOREGROUND masks on a uniform gray background. "
                "Gray is excluded background, not an object. Classify the colored foreground; "
                "a partially occluded but identifiable visible rim is still a valid candidate. "
                "Use unknown only when the visible foreground identity is ambiguous.")
        result = self.client.infer([Path(item["crop_path"]) for item in candidates], prompt)
        # Persist even malformed output so a failed model call is reviewable.
        import json
        Path(candidates[0]["crop_path"]).parent.joinpath("vlm_response.json").write_text(
            json.dumps({"prompt":prompt, **result.to_dict()}, indent=2)+"\n")
        labels = result.parsed
        review={}
        if self.identity_reviewer is not None:
            labels,review=self.identity_reviewer.review(candidates,result)
        if not isinstance(labels, dict) or set(labels) != set(ids):
            raise PerceptionError("VLM must classify exactly the observed candidate IDs")
        objects, warnings, grounding = {}, [], []
        for item in candidates:
            name = labels[item["candidate_id"]]
            if name == "unknown":
                warnings.append(f"unknown_candidate:{item['candidate_id']}")
                continue
            if name not in self.half_heights or name in objects:
                raise PerceptionError("unsupported or duplicate VLM identity")
            xyz = list(item["top_xyz_m"])
            xyz[2] -= self.half_heights[name]
            geometry = None
            if name in self.cylinder_models:
                import numpy as np
                from PIL import Image
                from .geometry import candidate_world_points, cylinder_cap_center
                calibration = observation.receipt.get('camera_calibration')
                if calibration is None:
                    raise PerceptionError('cylinder geometry requires saved public camera calibration')
                folder = Path(item['crop_path']).parent
                points = candidate_world_points(np.asarray(Image.open(folder/'top.png')),
                    np.load(folder/'depth_m.npy'), item['bbox_xyxy'], calibration)
                geometry = cylinder_cap_center(points, **self.cylinder_models[name])
                if geometry is None:
                    raise PerceptionError('cylinder cap geometry is incomplete or inconsistent')
                xyz = geometry['center_xyz_m']
            objects[name] = self._normalize({"name":name,"center_xyz_m":xyz},observation,warnings)
            grounding.append({"candidate_id":item["candidate_id"],"name":name,
                "center_xyz_m":xyz,"known_half_height_m":self.half_heights[name],
                **({'geometry_fit':geometry} if geometry is not None else {})})
        features = {}
        if self.visual_encoder is not None and grounding:
            crops = {item["candidate_id"]:Path(item["crop_path"]) for item in candidates}
            vectors = self.visual_encoder.encode([crops[item["candidate_id"]] for item in grounding])
            features = {item["name"]:vector.tolist() for item,vector in zip(grounding,vectors)}
        return objects, PerceptionReceipt("vlm_classification_color_rgbd_proposals", len(objects), tuple(warnings),
            {**result.to_dict(),"grounding":grounding,"scene_context":{},"visual_embeddings":features,
             "embedding_space":self.visual_encoder.embedding_space if self.visual_encoder else None,
             **({'effective_identity_labels':labels,'identity_review':review} if review else {})})
