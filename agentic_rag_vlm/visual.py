"""Actual CLIP ViT-L/14 image encoder; lazy optional model dependency."""
from pathlib import Path
import numpy as np


class CLIPVisualEncoder:
    model_id = "openai/clip-vit-large-patch14"

    def __init__(self, model_path, *, device="cpu", revision=None):
        import torch
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        self.torch = torch
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_path,revision=revision).eval().to(device)
        config = self.model.config
        if config.patch_size != 14 or config.hidden_size != 1024 or config.projection_dim != 768:
            raise ValueError("paper requires CLIP ViT-L/14, not a substitute encoder")
        self.processor = CLIPImageProcessor.from_pretrained(model_path,revision=revision)
        self.device = device
        # Pin local weights by file digest rather than an ambiguous model alias.
        import hashlib
        files = sorted(Path(model_path).glob("*.safetensors")) or sorted(Path(model_path).glob("pytorch_model*.bin"))
        if files:
            digest = hashlib.sha256()
            for path in files:
                with path.open("rb") as stream:
                    for chunk in iter(lambda:stream.read(1024*1024),b""):
                        digest.update(chunk)
            self.embedding_space = f"{self.model_id}@sha256:{digest.hexdigest()}"
        elif revision:
            self.embedding_space = f"{self.model_id}@{revision}"
        else:
            raise ValueError("pin model revision or supply hashable local weights")

    def encode(self, paths):
        from PIL import Image
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        if not images:
            return np.empty((0,768))
        inputs = self.processor(images=images,return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            features = self.model(**inputs).image_embeds.float()
            features = features/features.norm(dim=-1,keepdim=True)
        return features.cpu().numpy()
