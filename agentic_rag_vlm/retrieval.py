"""Exact three-stage HAA-RAG equations from paper Sec. III-A.

No appearance proxy fallback. Cosine similarity requires matching real visual
features. The caller retains historical proxy retrieval as a separate API.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
from .experience import ExperienceStore


class NoCompatibleExperience(ValueError):
    pass


class FixedSkillRetriever:
    """Calibrated nominal-skill control; never use the paper's 15-N generic
    fallback as a tabletop ablation. The same 0.8-N nominal skill is fixed for
    all targets, and the shared low-level feedback controller remains active.
    """
    action_source='fixed_skill'

    def __init__(self,store):
        self.store=store
        self.strategy={'synergy':'power','force_n':.8,'aperture_m':.075,'approach':'top',
                       'base_force_n':.8,'approach_height_delta_m':0.,'velocity_scale':1.}

    def retrieve(self,target,visual_embedding,*,embedding_space,enabled=True):
        if self.store.domain!='mjlab_yam_retention_v1':
            raise ValueError('nominal 0.8-N baseline is calibrated only for the YAM retention domain')
        return PaperRetrieval(target.name,'fixed_nominal_0p8n',dict(self.strategy),(),False,
            {'baseline':'fixed_nominal_calibrated_skill','does_not_select_from_knowledge':True},'unused')


@dataclass(frozen=True)
class PaperRetrieval:
    target: str
    selected_card_id: str
    strategy: dict
    ranked_cards: tuple[dict,...]
    retrieval_enabled: bool
    stages: dict
    visual_feature_space: str

    def to_dict(self):
        return asdict(self)


class HAARetriever:
    def __init__(self, store: ExperienceStore, *, k1=30, k=3, affordance_threshold=.3):
        if k1 != 30 or k != 3 or affordance_threshold != .3:
            raise ValueError("paper profile fixes k1=30, k=3, affordance threshold=0.3")
        self.store = store

    def retrieve(self, target, visual_embedding, *, embedding_space: str, enabled=True):
        if not enabled:
            return PaperRetrieval(target.name,"generic_power_default",
                {"synergy":"power","force_n":15.,"aperture_m":.07,"approach":"top"},(),False,{},"disabled")
        if embedding_space != self.store.embedding_space:
            raise ValueError("query and store embedding spaces differ")
        vector = np.asarray(visual_embedding,dtype=float)
        if vector.shape != (self.store.dimension,) or not np.isfinite(vector).all() or np.linalg.norm(vector)<1e-8:
            raise ValueError("valid CLIP image embedding required")
        records = self.store.records
        candidates = []
        for item in records:
            category = 1. if item["category"] == target.category else .5 if item["superclass"] == target.superclass else 0.
            if category:
                candidates.append((category,item))
        candidates.sort(key=lambda entry:(-entry[0],entry[1]["id"]))
        gated = candidates[:30]
        ranked, rejected = [], []
        for category,item in gated:
            affordance = (.5*(target.affordance_type == item["affordance_type"])
                + .2*(target.material == item["material"])
                + .15*(1-abs(target.fragility-item["fragility"]))
                + .15*(target.region == item["region"]))
            if affordance < .3:
                rejected.append(item["id"])
                continue
            stored = np.asarray(item["visual_embedding"])
            visual = float(np.clip(vector @ stored / (np.linalg.norm(vector)*np.linalg.norm(stored)),-1,1))
            raw = .2*category + .4*affordance + .4*visual
            # Paper specifies score reduction, not deletion, for failed records.
            score = raw*(.7 if item["outcome"] == 0 else 1.)
            ranked.append({"card_id":item["id"],"category_score":category,"affordance_score":affordance,
                "visual_score":visual,"unpenalized_score":raw,"final_score":score,"outcome":item["outcome"],
                "negative_example":item["outcome"] == 0,"strategy":item["strategy"],"provenance":item["provenance"]})
        ranked.sort(key=lambda entry:(-entry["final_score"],entry["card_id"]))
        if not ranked:
            raise NoCompatibleExperience("no category/superclass candidate passed affordance threshold")
        first = ranked[0]
        return PaperRetrieval(target.name,first["card_id"],dict(first["strategy"]),tuple(ranked[:3]),True,
            {"store":len(records),"category_matches":len(candidates),"category_top30":len(gated),
             "affordance_rejected":rejected,"visual_ranked":len(ranked),
             "negative_examples":[entry for entry in ranked if entry["negative_example"]]},self.store.embedding_space)
