"""Local official Runner for a real BluPe station. Default: GET-only monitor."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agentic_rag_vlm.platform.hardware_runner import make_hardware_playground


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runner-checkout',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=18797)
    p.add_argument('--enable-hardware-run',action='store_true',help='allow user Run after measured site configuration; creates a real queue entry only on Run')
    args=p.parse_args()
    app=make_hardware_playground(args.runner_checkout,args.config,args.output,port=args.port,enable_run=args.enable_hardware_run)
    print(f'http://127.0.0.1:{args.port}/ — '+('real hardware Run enabled' if args.enable_hardware_run else 'read-only monitor; no robot session'),flush=True)
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=args.port)
