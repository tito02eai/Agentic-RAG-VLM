"""Deploy the full Agent model backend without joining a robot session."""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agentic_rag_vlm.platform.runner_provider import load_provider
from agentic_rag_vlm.platform.http_policy import make_server


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=18076);p.add_argument('--output',type=Path,default=Path('output/platform_previews'))
    args=p.parse_args();provider=load_provider(args.config);provider.prepare()
    server=make_server(provider,args.output,host=args.host,port=args.port,token=os.environ.get('AGENTIC_POLICY_TOKEN'))
    print(f'Agentic RAG-VLM model backend ready at {args.host}:{args.port}; no hardware session created',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
