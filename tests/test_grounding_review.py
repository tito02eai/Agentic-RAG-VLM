"""Software contracts: reinspection must preserve raw output and cannot invent IDs."""
import json
from pathlib import Path
import pytest
from agentic_rag_vlm.grounding import IdentityConflictReviewer
from agentic_rag_vlm.qwen_adapter import VLMResult
from agentic_rag_vlm.platform.perception import PerceptionError
from agentic_rag_vlm.platform.transport import PlatformObservation
from agentic_rag_vlm.simulation.perception import CandidateRGBDVLMPerception
from scripts.agentic_framework import load_semantic_registry

ROOT=Path(__file__).resolve().parents[1]


class Client:
    def __init__(self,outputs):self.outputs=iter(outputs);self.calls=0
    def infer(self,images,prompt):
        self.calls+=1;value=next(self.outputs)
        return VLMResult(value,json.dumps(value),1,'test','test',{})


def setup_perception(tmp_path,client,max_calls=6):
    items=[]
    for i,top in enumerate(([.3,-.12,.036],[.375,-.12,.08])):
        path=tmp_path/f'candidate_{i}.png';path.write_bytes(f'test image {i}'.encode())
        items.append({'candidate_id':f'candidate_{i}','top_xyz_m':top,'crop_path':str(path)})
    reviewer=IdentityConflictReviewer(client,{'cube','vase'},'cube = red block; vase = green vessel',max_calls=max_calls)
    registry=load_semantic_registry(ROOT/'knowledge_base/blupe_semantic_registry.json')
    perception=CandidateRGBDVLMPerception(client,registry,{'cube':.018,'vase':.04},identity_reviewer=reviewer)
    return perception,PlatformObservation('test',(),receipt={'rgbd_candidates':items})


def test_conflict_reinspection_preserves_raw_model_proposal(tmp_path):
    initial={'candidate_0':'vase','candidate_1':'vase'}
    client=Client([initial,{'label':'cube'},{'label':'vase'}])
    perception,observation=setup_perception(tmp_path,client)
    objects,receipt=perception.perceive(observation)
    assert objects['cube'].center_xyz_m==pytest.approx([.3,-.12,.018])
    assert objects['vase'].center_xyz_m==pytest.approx([.375,-.12,.04])
    assert json.loads((tmp_path/'vlm_response.json').read_text())['parsed']==initial
    assert json.loads(receipt.vlm_result['raw_text'])==initial
    assert receipt.vlm_result['effective_identity_labels']=={'candidate_0':'cube','candidate_1':'vase'}
    assert receipt.vlm_result['identity_review']['review_calls']==2 and client.calls==3


def test_unresolved_duplicates_stop_after_bounded_calls(tmp_path):
    initial={'candidate_0':'vase','candidate_1':'vase'}
    client=Client([initial,*[{'label':'vase'}]*3])
    perception,observation=setup_perception(tmp_path,client,max_calls=3)
    with pytest.raises(PerceptionError,match='duplicate'):
        perception.perceive(observation)
    decision=json.loads((tmp_path/'grounding_decision.json').read_text())
    assert decision['review_calls']==3 and client.calls==4
    assert decision['unresolved_candidates']==['candidate_0','candidate_1']
    assert 'cube' not in decision['effective_labels'].values()  # no elimination fallback


def test_unsupported_reinspection_labels_remain_unknown(tmp_path):
    client=Client([{'candidate_0':'unknown','candidate_1':'vase'},{'label':'invented'},{'label':'invented'}])
    perception,observation=setup_perception(tmp_path,client)
    objects,receipt=perception.perceive(observation)
    assert set(objects)=={'vase'} and receipt.vlm_result['effective_identity_labels']['candidate_0']=='unknown'
    assert client.calls==3
