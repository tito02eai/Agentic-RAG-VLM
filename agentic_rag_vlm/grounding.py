"""Bounded VLM reinspection of conflicting instance identities; no identity oracle."""
from collections import Counter
import hashlib
from pathlib import Path
from .experience import atomic_json


class IdentityConflictReviewer:
    def __init__(self,client,allowed_ids,catalogue,*,max_rounds=2,max_calls=6):
        if not 1<=max_rounds<=2 or not 1<=max_calls<=6:
            raise ValueError('identity review must remain bounded')
        self.client=client;self.allowed=set(allowed_ids)|{'unknown'};self.catalogue=catalogue
        self.max_rounds=max_rounds;self.max_calls=max_calls

    def review(self,candidates,initial_result):
        ids=[c['candidate_id'] for c in candidates];folder=Path(candidates[0]['crop_path']).parent
        initial=initial_result.parsed
        exact=isinstance(initial,dict) and set(initial)==set(ids)
        labels={key:initial.get(key,'unknown') if isinstance(initial,dict) else 'unknown' for key in ids}
        labels={key:value if isinstance(value,str) else 'unknown' for key,value in labels.items()}
        calls=[]
        def conflicts():
            counts=Counter(value for value in labels.values() if value!='unknown')
            return [key for key,value in labels.items()
                    if value not in self.allowed or value=='unknown' or counts[value]>1]
        for round_index in range(self.max_rounds):
            bad=conflicts() if exact or round_index else ids
            if not bad:break
            for key in bad:
                if len(calls)>=self.max_calls:break
                candidate=next(c for c in candidates if c['candidate_id']==key)
                image=Path(candidate['crop_path'])
                prompt=('Classify this single foreground candidate. Grey is background. '
                    'Catalogue IDs: '+self.catalogue+'. '
                    'A visible rim may belong to a partly occluded object. '
                    'If the visible foreground is ambiguous, use unknown. '
                    'Return JSON only: {"label":"one exact ID"}.')
                response=self.client.infer([image],prompt)
                name=f'identity_review_{len(calls)+1:02d}.json'
                record={'candidate_id':key,'round':round_index+1,'prompt':prompt,
                    'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),**response.to_dict()}
                atomic_json(folder/name,record);calls.append(name)
                value=response.parsed.get('label') if isinstance(response.parsed,dict) else None
                labels[key]=value if isinstance(value,str) and value in self.allowed else 'unknown'
        receipt={'schema':'agentic_rag_vlm.identity_review.v1','initial_labels':initial,'effective_labels':labels,
            'review_calls':len(calls),'call_files':calls,'unresolved_candidates':conflicts(),
            'initial_response_sha256':hashlib.sha256((folder/'vlm_response.json').read_bytes()).hexdigest(),
            'identity_source':'actual_single_crop_vlm_reinspection; no geometry/color identity override',
            'max_rounds':self.max_rounds,'max_calls':self.max_calls}
        atomic_json(folder/'grounding_decision.json',receipt)
        return labels,receipt
