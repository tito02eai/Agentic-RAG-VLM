"""Official BluPe local Playground with actual MuJoCo/Agent closed-loop tasks."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agentic_rag_vlm.platform.runner_provider import load_provider
from agentic_rag_vlm.platform.simulation_runner import make_simulation_playground


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runner-checkout',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=18791)
    args=p.parse_args();provider=load_provider(args.config)
    app,_=make_simulation_playground(args.runner_checkout,provider,args.source_root.resolve(),args.output,port=args.port)
    import uvicorn
    print(f'Local MuJoCo/Agent task runner at http://127.0.0.1:{args.port}; no BluPe hardware connection',flush=True)
    uvicorn.run(app,host='127.0.0.1',port=args.port)


if __name__=='__main__':main()
