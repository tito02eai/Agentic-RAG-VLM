"""Public composition of paper-aligned retrieval, memory and VLM reflection."""
from pathlib import Path
from .experience import ExperienceStore
from .retrieval import HAARetriever
from .reasoning import VLMReasoner
from .pipeline import AgenticPipeline,PipelineConfig


class PaperAgent(AgenticPipeline):
    def __init__(self, knowledge_path: Path, vlm_client, *, seed=0,
                 config: PipelineConfig | None=None, verify_sources=True, path_validator=None):
        store=ExperienceStore.load(knowledge_path)
        if not store.embedding_space.startswith("openai/clip-vit-large-patch14@") or store.dimension!=768:
            raise ValueError("paper profile requires pinned CLIP ViT-L/14 embeddings")
        if verify_sources:
            store.validate_sources(knowledge_path.parent)
        super().__init__([],config,retriever=HAARetriever(store),reasoner=VLMReasoner(vlm_client),
                         seed=seed,path_validator=path_validator)

    def bind_observation(self, *, visual_embeddings, embedding_space, scene_context=None, images=None):
        if embedding_space!=self.retriever.store.embedding_space:
            raise ValueError("query encoder weights differ from knowledge encoder")
        self.visual_features=dict(visual_embeddings)
        self.embedding_space=embedding_space
        self.scene_context=dict(scene_context or {})
        self.reasoner.images=list(images or [])
