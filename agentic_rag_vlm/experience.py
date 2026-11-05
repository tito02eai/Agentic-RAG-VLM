"""Provenance-bearing manipulation experiences (paper Sec. III-A).

The store contains actual outcome-labelled records, never treats an unvalidated
recommendation as a measured success, and refuses mixed embedding spaces.
"""
from __future__ import annotations
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any
import numpy as np


def atomic_json(path: Path, value: Any) -> None:
    import os
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w",dir=path.parent,delete=False,encoding="utf-8") as stream:
            temporary = Path(stream.name)
            json.dump(value,stream,indent=2,allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class ExperienceStore:
    schema = "agentic_rag_vlm.experiences.v1"

    def __init__(self, *, embedding_space: str, dimension: int = 768, domain: str):
        if not embedding_space or not domain or dimension <= 0:
            raise ValueError("embedding space, positive dimension and robot domain required")
        self.embedding_space, self.dimension, self.domain = embedding_space, dimension, domain
        self._records: dict[str,dict] = {}

    def add(self, record: dict) -> None:
        item = deepcopy(record)
        fields = {"id","category","superclass","affordance_type","material","fragility","region",
                  "strategy","outcome","visual_embedding","provenance"}
        if fields-set(item):
            raise ValueError(f"missing experience fields: {sorted(fields-set(item))}")
        for name in ("id","category","superclass","affordance_type","material","region"):
            if not isinstance(item[name],str) or not item[name].strip():
                raise ValueError(f"invalid experience field: {name}")
        if type(item["outcome"]) is not int or item["outcome"] not in (0,1):
            raise ValueError("outcome must be an explicitly measured binary label")
        if not math.isfinite(item["fragility"]) or not 0 <= item["fragility"] <= 1:
            raise ValueError("invalid fragility")
        vector = np.asarray(item["visual_embedding"],dtype=float)
        if vector.shape != (self.dimension,) or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-8:
            raise ValueError("invalid visual embedding")
        origin = item["provenance"]
        if not isinstance(origin,dict) or origin.get("domain") != self.domain or not origin.get("source_episode"):
            raise ValueError("experience needs source episode and matching robot domain")
        if not origin.get("evidence_kind") or not origin.get("image_sha256"):
            raise ValueError("experience needs outcome evidence and observation hash")
        strategy = item["strategy"]
        if not isinstance(strategy,dict) or not {"force_n","aperture_m","approach","synergy"} <= strategy.keys():
            raise ValueError("complete grasp strategy required")
        for key in ("force_n","aperture_m"):
            if not math.isfinite(float(strategy[key])) or float(strategy[key]) < 0:
                raise ValueError(f"invalid grasp parameter: {key}")
        if item["id"] in self._records and item != self._records[item["id"]]:
            raise ValueError("experience IDs are immutable; use a new ID for a new observation")
        self._records[item["id"]] = item

    @property
    def records(self) -> list[dict]:
        return deepcopy(list(self._records.values()))

    def save(self, path: Path) -> None:
        atomic_json(path,{"schema_version":self.schema,"embedding_space":self.embedding_space,
            "dimension":self.dimension,"domain":self.domain,"records":self.records})

    def validate_sources(self, base: Path) -> None:
        """Validate portable source attachments for the observed-run importer."""
        import hashlib
        base = base.resolve()
        for item in self._records.values():
            origin = item["provenance"]
            for path_field,hash_field in (("observation_image","image_sha256"),("source_result_path","source_result_sha256")):
                if path_field not in origin or hash_field not in origin:
                    raise ValueError("observed store needs portable image/result attachments")
                path = (base/origin[path_field]).resolve()
                if not path.is_relative_to(base) or not path.is_file():
                    raise ValueError("experience attachment missing or outside knowledge directory")
                if hashlib.sha256(path.read_bytes()).hexdigest() != origin[hash_field]:
                    raise ValueError("experience source hash mismatch")
            result = json.loads((base/origin["source_result_path"]).read_text())
            scope=origin.get('label_scope')
            if scope=='observed_retention_rejection':
                if item['outcome']!=0 or 'source_command_id' not in origin:
                    raise ValueError('negative retention labels need an actual rejected command')
            elif scope!='verified_complete_grasp_place' or item['outcome']!=int(result['physical_success']):
                raise ValueError("experience outcome does not match attached episode evidence")
            if 'source_command_id' in origin:
                validate_command_experience(item,result)

    @classmethod
    def load(cls, path: Path):
        payload = json.loads(path.read_text())
        if payload.get("schema_version") != cls.schema:
            raise ValueError("unsupported experience schema")
        store = cls(embedding_space=payload["embedding_space"],dimension=payload["dimension"],domain=payload["domain"])
        for item in payload["records"]:
            store.add(item)
        return store


def validate_command_experience(item, result):
    """Validate the specific attempt, appearance binding and ACTUAL force.

    A successful episode is not evidence that its first failed command worked.
    Failure labels describe measured retention rejection, not object damage.
    """
    origin=item['provenance'];last_observation=None;last_perception=None;matched=None
    for receipt in result.get('receipts',[]):
        if receipt['kind']=='observation':
            last_observation=receipt['payload'];last_perception=None
        elif receipt['kind']=='perception':last_perception=receipt['payload']['vlm_result']
        elif receipt['kind']=='command' and receipt['payload']['sent']['command_id']==origin['source_command_id']:
            if matched is not None:raise ValueError('source command ID is not unique')
            matched=(receipt['payload'],last_observation,last_perception)
    if matched is None or not result.get('uses_vlm'):
        raise ValueError('command experience requires an actual VLM-grounded command')
    command,observation,perception=matched
    target=origin['source_target'];sent=command['sent'];receipt=command['receipt']
    if sent['command_type']!='pick' or sent['target']!=target or not observation or not perception:
        raise ValueError('experience target/observation does not match command')
    if observation['observation_id']!=origin['source_observation_id']:
        raise ValueError('experience refers to a different observation')
    ground=[g for g in perception['grounding'] if g['name']==target and g['candidate_id']==origin['source_candidate_id']]
    if len(ground)!=1:raise ValueError('source target has no unique model-grounded crop')
    vector=perception.get('visual_embeddings',{}).get(target)
    if vector is None or not np.array_equal(np.asarray(vector),np.asarray(item['visual_embedding'])):
        raise ValueError('experience embedding differs from the recorded source crop embedding')
    cited=any(image.get('sha256')==origin['image_sha256']
              and Path(image.get('path','')).name==origin['source_candidate_id']+'.png'
              for call in result.get('reasoner_calls',[]) for image in call.get('images',[]))
    if not cited:
        raise ValueError('source crop hash was not supplied to a recorded VLM action review')
    retained=receipt.get('retention',{})
    if not retained or retained.get('command_id')!=sent['command_id']:
        raise ValueError('source command lacks measured retention evidence')
    scale=float(sent.get('constraint',{}).get('force_scale',1.))
    if not math.isfinite(scale) or scale<=0:
        raise ValueError('invalid source force scale')
    actual=float(retained['motor_force_cap_n'])
    if not math.isclose(float(item['strategy']['force_n'])*scale,actual,abs_tol=1e-8):
        raise ValueError('experience force does not match actual execution feedback')
    if item['outcome']==1:
        verified=any(e['event']=='subgoal_verified' and e['payload']['target']==target
                     and e['payload']['success'] is True for e in result['trace'])
        if (not result.get('physical_success') or not result.get('checks') or not all(result['checks'].values())
                or receipt.get('accepted') is not True or retained.get('retained') is not True or not verified):
            raise ValueError('positive experience lacks complete verified grasp/place evidence')
    elif (receipt.get('accepted') is not False or receipt.get('execution_started') is not True
          or receipt.get('reason')!='grasp_not_retained' or retained.get('retained') is not False):
        raise ValueError('negative experience is not a measured retention rejection')
