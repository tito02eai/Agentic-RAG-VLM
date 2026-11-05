"""Official Playground with the production provider and local protocol physics."""
import argparse,asyncio,json,os,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agentic_rag_vlm.simulation.protocol_gateway import PhysicsWorker,make_gateway,recommended_provider
from agentic_rag_vlm.platform.local_runner import import_official_runner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runner-checkout',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=1000)
    p.add_argument('--port',type=int,default=18793);args=p.parse_args()
    os.environ.setdefault('MUJOCO_GL','egl');worker=PhysicsWorker(args.output,seed=args.seed)
    origin=f'http://127.0.0.1:{args.port}'
    gateway,backend,requests=make_gateway(args.runner_checkout,worker,origin)
    _,sessions=import_official_runner(args.runner_checkout)
    from playground import HostedRunner,RequestError
    provider=recommended_provider(args.config,worker,origin)
    def sdk():
        value=sessions.HttpSessionAPI(origin,supports_trajectories=True,robot_id='yam-local-1')
        value.local_simulation_only=True;return value
    class Playground(HostedRunner):
        def page(self):
            page=super().page().replace(b'</head>',b'<link rel="stylesheet" href="/recommended/style.css">'
                b'<script src="/recommended/status.js" defer></script></head>')
            banner=(b'<div id="recommendedBanner">PRODUCTION PROVIDER / OFFICIAL HTTP + WEBSOCKET / LOCAL MUJOCO<br>'
                b'Rigid-object grasp with protected neighbor. No public robot connection. '
                b'<a href="/recommended/result">Result</a> | <a href="/recommended/video">Video</a>'
                b'<div id="recommendedStatus">Ready. One episode per launch.</div></div>')
            page=page.replace(b'<body class="playgroundConsole">',b'<body class="playgroundConsole">'+banner,1)
            return re.sub(b'(<section id="liveViewer"[^>]*>)',rb'\1<img id="recommendedFrame" src="/recommended/frame" alt="Live protocol-driven simulation">',page,count=1)
        def validate_launch(self,visitor,payload,paid):
            if backend._sessions:raise RequestError(409,'This local episode is already used; start a fresh instance for a new trial')
            if payload.get('provider')!='agentic-recommended' or payload.get('prompt')!=provider.task.instruction:
                raise RequestError(400,'Select the configured protected-neighbor task')
            if payload.get('share_conversation') is not False:raise RequestError(400,'Local-only validation')
        async def session_extension(self,cookies,visitor):
            return {'local_runner':True,'default_provider':'agentic-recommended','configured_providers':{
                'agentic-recommended':{'label':'Agentic RAG-VLM — protected neighbor','model':provider.model,
                'task':provider.task.instruction,'help':'Production provider, real native IK and SDK; local simulation only.',
                'default_run_duration_s':600}}},[]
        async def http(self,scope,receive,send):
            path=scope['path']
            if path.startswith('/recommended/'):
                headers=dict(scope.get('headers',[]))
                if scope['method']!='GET' or headers.get(b'host',b'').decode()!=self.authority:raise RequestError(403,'Local GET only')
                if path=='/recommended/frame':
                    file=worker.output/'live/frame.jpg';return await self.content(send,file.read_bytes(),b'image/jpeg')
                if path=='/recommended/style.css':return await self.content(send,
                    b'#recommendedBanner{background:#dceee3;color:#123;padding:12px;text-align:center}'
                    b'#recommendedFrame{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#151719;z-index:20}'
                    b'#liveViewer .liveStopwatch,#liveViewer #viewerCue{z-index:25}',b'text/css')
                if path=='/recommended/status.js':return await self.content(send,
                    b'async function refresh(){try{const s=await(await fetch("/recommended/status")).json();'
                    b'document.getElementById("recommendedStatus").textContent="Agent: "+s.stage+" | packets: "+s.packets+" | simulation: "+s.simulation;'
                    b'document.getElementById("recommendedFrame").src="/recommended/frame?t="+Date.now();}catch(e){}}refresh();setInterval(refresh,750);',b'application/javascript')
                if path=='/recommended/status':
                    file=worker.output/'physical_result.json';physical=json.loads(file.read_text()) if file.exists() else {}
                    return await self.json(send,200,{'stage':provider._runtime_event['event'] if provider._runtime_event else 'ready',
                        'packets':len(backend.trajectory_log),'simulation':physical.get('simulation_success','pending')})
                if path in {'/recommended/result','/recommended/video'}:
                    if not worker.finalized:
                        if provider._active or not list(provider.output.glob('*/result.json')):raise RequestError(409,'Task has not finished')
                        await asyncio.to_thread(worker.rpc,worker.finalize)
                    if path.endswith('/result'):
                        return await self.json(send,200,json.loads((worker.output/'physical_result.json').read_text()))
                    return await self.content(send,(worker.output/'physics/rollout.mp4').read_bytes(),b'video/mp4')
                raise RequestError(404,'Unknown recommended-task route')
            return await super().http(scope,receive,send)
        async def content(self,send,data,mime):
            await send({'type':'http.response.start','status':200,'headers':[(b'content-type',mime),
                (b'content-length',str(len(data)).encode()),(b'cache-control',b'no-store')]})
            await send({'type':'http.response.body','body':data})
    app=Playground(public_origin=origin,session_api=origin,camera_origin=origin,development=True,hardware_control=False,
        robot_id='yam-local-1',api_factory=sdk,share_conversation=False,camera_names=('top',),
        recording_root=worker.output/'runner_recordings',extra_providers={'agentic-recommended':lambda model,directory:provider},
        robots=[{'id':'yam-local-1','name':'YAM — PROTOCOL SIMULATION','url':origin+'/'}])
    app.simulation=True
    async def combined(scope,receive,send):
        if scope.get('path','').startswith('/v1/') or scope['type']=='websocket':return await gateway(scope,receive,send)
        return await app(scope,receive,send)
    import uvicorn
    print(f'Production-provider Playground at {origin}; local-only protocol simulation',flush=True)
    uvicorn.run(combined,host='127.0.0.1',port=args.port)


if __name__=='__main__':main()
