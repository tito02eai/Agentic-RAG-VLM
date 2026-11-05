"""Non-actuating model backend: health/capabilities and grounded plan preview.

Actual task execution uses the runner provider's run_session interface, retaining
the runner's existing queue/Stop/feedback ownership. This HTTP API has no robot
write route and never accepts arbitrary server filesystem image paths.
"""
import base64
import json
from pathlib import Path
import threading
import uuid
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

from .transport import PlatformObservation


def preview_request(provider,payload,output):
    folder=Path(output)/uuid.uuid4().hex;folder.mkdir(parents=True,exist_ok=False)
    image=payload['image'];raw=base64.b64decode(image['data_base64'],validate=True)
    if not 0<len(raw)<=8*1024*1024:raise ValueError('image size must be 1 byte to 8 MiB')
    from PIL import Image
    import io
    with Image.open(io.BytesIO(raw)) as decoded:
        if decoded.width*decoded.height>4096*4096:raise ValueError('image resolution limit')
        decoded.convert('RGB').save(folder/'top.png')
    task_payload=payload['task']
    if task_payload.get('coordinate_frame')!='arm_base':raise ValueError('preview task must use arm_base coordinates')
    (folder/'task.json').write_text(json.dumps(task_payload,allow_nan=False))
    from scripts.run_platform_session import load_task
    task,_=load_task(folder/'task.json')
    calibration=payload['calibration'];geometry=payload['geometry']
    observation=PlatformObservation('preview_'+folder.name,(),(str(folder/'top.png'),),
        {'calibration_id':calibration['calibration_id'],'camera_role':'top',
         'captured_at':payload.get('captured_at'),'source_observation_id':payload.get('observation_id'),
         **({'object_candidates':payload['object_candidates']} if 'object_candidates' in payload else {}),
         **({'scene_context':payload['scene_context']} if 'scene_context' in payload else {})})
    result=provider.preview(task,observation,calibration,geometry)
    from ..experience import atomic_json
    atomic_json(folder/'result.json',result)
    return result


def make_server(provider,output,*,host='127.0.0.1',port=18076,token=None):
    if host not in ('localhost','127.0.0.1','::1') and not token:
        raise ValueError('set AGENTIC_POLICY_TOKEN when binding a non-loopback interface')
    lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def respond(self,status,payload):
            data=json.dumps(payload,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def authorized(self):
            import hmac
            return not token or hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token)
        def do_GET(self):
            if not self.authorized():return self.respond(401,{'error':'unauthorized'})
            if self.path=='/health':
                return self.respond(200,{'status':'ready' if provider._prepared else 'starting',
                    'provider':provider.public_config(),'hardware_session_created':False})
            if self.path=='/v1/capabilities':
                return self.respond(200,{'schema_version':'agentic_rag_vlm.platform_policy.v1',
                    'http':['GET /health','GET /v1/capabilities','POST /v1/preview'],
                    'runner':['public_config','prepare','preview','run_session'],
                    'queue_and_stop_owner':'RunnerController','http_execution_routes':[]})
            self.respond(404,{'error':'route_not_found'})
        def do_POST(self):
            if not self.authorized():return self.respond(401,{'error':'unauthorized'})
            if self.path!='/v1/preview':return self.respond(404,{'error':'route_not_found'})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=12*1024*1024:raise ValueError('request size limit')
                payload=json.loads(self.rfile.read(size))
                with lock:result=preview_request(provider,payload,output)
                self.respond(200,result)
            except (ValueError,KeyError,TypeError,OSError) as error:
                self.respond(400,{'error':type(error).__name__,'message':str(error),'execution_performed':False})
            except Exception:
                import logging
                logging.exception('platform preview backend failure')
                self.respond(503,{'error':'inference_unavailable','execution_performed':False})
    return ThreadingHTTPServer((host,port),Handler)
