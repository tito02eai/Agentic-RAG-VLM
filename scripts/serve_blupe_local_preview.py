"""Start the official BluPe Playground in recorded-image-only validation mode."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from agentic_rag_vlm.platform.local_runner import make_local_playground
from agentic_rag_vlm.platform.runner_provider import load_provider


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runner-checkout',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--recorded-request',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('output/blupe_local_preview'))
    p.add_argument('--port',type=int,default=18791)
    args=p.parse_args()
    payload=json.loads(args.recorded_request.read_text());provider=load_provider(args.config)
    args.output.mkdir(parents=True,exist_ok=True)
    app=make_local_playground(args.runner_checkout,provider,payload,args.output,port=args.port)
    import uvicorn
    print(f'Official BluPe local preview: http://127.0.0.1:{args.port}; no physical/public session',flush=True)
    uvicorn.run(app,host='127.0.0.1',port=args.port)


if __name__=='__main__':main()
