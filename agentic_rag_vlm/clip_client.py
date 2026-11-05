"""Local CLIP feature RPC keeps torch optional for robot/simulator processes."""
import base64
import json
from urllib import request
import numpy as np


class CLIPFeatureClient:
    def __init__(self, endpoint="http://127.0.0.1:18072", *, timeout=120):
        self.endpoint, self.timeout = endpoint.rstrip('/'),timeout
        with request.urlopen(self.endpoint+"/health",timeout=timeout) as response:
            info = json.load(response)
        self.embedding_space = info["embedding_space"]
        if not self.embedding_space.startswith("openai/clip-vit-large-patch14@"):
            raise ValueError("CLIP service must serve the paper encoder")

    def encode(self, paths):
        payload = json.dumps({"images":[base64.b64encode(p.read_bytes()).decode() for p in paths]}).encode()
        call = request.Request(self.endpoint+"/encode",data=payload,headers={"Content-Type":"application/json"})
        with request.urlopen(call,timeout=self.timeout) as response:
            data = json.load(response)
        vectors = np.asarray(data["embeddings"],dtype=float)
        if data["embedding_space"] != self.embedding_space or vectors.shape != (len(paths),768) or not np.isfinite(vectors).all():
            raise ValueError("invalid or changed CLIP encoder response")
        return vectors
