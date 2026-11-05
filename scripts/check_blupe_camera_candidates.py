"""Classify RGB proposals on a saved hardware image; no geometry or robot calls."""
import argparse
from pathlib import Path
import sys
import json
import hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agentic_rag_vlm.platform.candidates import load_candidate_provider
from agentic_rag_vlm.platform.transport import PlatformObservation
from agentic_rag_vlm.qwen_adapter import OpenAICompatibleQwenVL
from agentic_rag_vlm.services import resolve_model
from agentic_rag_vlm.experience import atomic_json


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--catalogue',type=Path,required=True,help='JSON object ID -> visual description')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--vlm-endpoint',default='http://127.0.0.1:18074/v1/chat/completions')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    provider=load_candidate_provider(args.candidates)
    proposals=provider(PlatformObservation('saved_hardware_image',(),(str(args.image),)))
    catalogue=json.loads(args.catalogue.read_text())
    from PIL import Image,ImageOps
    crops=[]
    with Image.open(args.image) as image:
        for region in proposals:
            x0,y0,x1,y1=region['bbox_xyxy']
            box=(max(0,x0-8),max(0,y0-8),min(image.width,x1+8),min(image.height,y1+8))
            crop=args.output/(region['candidate_id']+'.png')
            ImageOps.pad(image.crop(box).convert('RGB'),(192,192),color=(128,128,128)).save(crop)
            crops.append(crop)
    if not crops:raise ValueError('no candidate regions; inspect saved image and thresholds')
    prompt=('These are object crops in order '+', '.join(r['candidate_id'] for r in proposals)+'. '
        'Classify each crop using these registered IDs and descriptions: '+json.dumps(catalogue)+
        ', or unknown. Return one flat JSON object mapping every region ID to a label; no coordinates or explanations.')
    model=resolve_model(args.vlm_endpoint,None)
    result=OpenAICompatibleQwenVL(args.vlm_endpoint,model).infer(crops,prompt)
    labels=result.parsed or {}
    assigned=[label for label in labels.values() if label!='unknown']
    valid=(set(labels)=={r['candidate_id'] for r in proposals} and len(set(assigned))==len(assigned)
           and all(label in catalogue for label in assigned))
    receipt={'scope':'saved real camera RGB proposal/identity check only; no metric localization or hardware task',
        'valid_identity_output':valid,'image_sha256':hashlib.sha256(args.image.read_bytes()).hexdigest(),
        'proposals':proposals,'prompt':prompt,'parsed':result.parsed,'raw_text':result.raw_text,
        'requested_model':result.requested_model,'response_model':result.response_model,
        'latency_ms':result.latency_ms,'execution_performed':False,'robot_session_created':False}
    atomic_json(args.output/'result.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
    raise SystemExit(0 if valid else 1)
