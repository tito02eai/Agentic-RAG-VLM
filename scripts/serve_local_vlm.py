"""Local-only, serialized multimodal inference with explicit checkpoint identity."""
import argparse
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path',type=Path,required=True)
    parser.add_argument('--port',type=int,default=18074)
    parser.add_argument('--quantization',choices=['nf4','bf16'],default='nf4')
    args=parser.parse_args()
    import torch,transformers
    from transformers import AutoModelForImageTextToText,AutoProcessor,BitsAndBytesConfig
    from PIL import Image
    torch.set_num_threads(2)
    model_path=args.model_path.resolve()
    weights={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(model_path.glob('*.safetensors'))}
    if not weights:raise ValueError('local safetensors checkpoint required')
    fingerprint=hashlib.sha256(json.dumps(weights,sort_keys=True).encode()).hexdigest()
    config={'dtype':torch.bfloat16,'device_map':{'':'cuda:0'},'attn_implementation':'sdpa','local_files_only':True}
    if args.quantization=='nf4':
        config['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(model_path,**config).eval()
    identity=f'{model_path.name}@sha256:{fingerprint[:16]}:{args.quantization}'
    health={'model':identity,'requested_model':str(model_path),'weights_sha256':fingerprint,
            'weight_files':weights,'quantization':args.quantization,'torch':torch.__version__,
            'transformers':transformers.__version__,'enable_thinking':False,'server_host':'127.0.0.1',
            'server_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    class Handler(BaseHTTPRequestHandler):
        def respond(self,status,value):
            body=json.dumps(value).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)

        def do_GET(self):
            if self.path=='/health':return self.respond(200,health)
            if self.path=='/v1/models':return self.respond(200,{'data':[{'id':str(model_path),'object':'model'}]})
            return self.respond(404,{'error':'not found'})

        def do_POST(self):
            if self.path!='/v1/chat/completions':return self.respond(404,{'error':'not found'})
            try:
                size=int(self.headers.get('Content-Length',0))
                if not 0<size<=20*1024*1024:raise ValueError('request size limit exceeded')
                request=json.loads(self.rfile.read(size))
                if request.get('model') not in {str(model_path),model_path.name,identity}:
                    raise ValueError('requested model does not match the loaded checkpoint')
                images=[];messages=[]
                for message in request['messages']:
                    content=message['content']
                    if isinstance(content,str):content=[{'type':'text','text':content}]
                    parts=[]
                    for part in content:
                        if part['type']=='image_url':
                            url=part['image_url']['url']
                            if not url.startswith('data:image/') or ';base64,' not in url:
                                raise ValueError('only inline image data is supported')
                            img=Image.open(io.BytesIO(base64.b64decode(url.split(',',1)[1],validate=True))).convert('RGB')
                            images.append(img);parts.append({'type':'image','image':img})
                        elif part['type']=='text':parts.append({'type':'text','text':part['text']})
                        else:raise ValueError('unsupported content type')
                    messages.append({'role':message['role'],'content':parts})
                if len(images)>8:raise ValueError('at most eight images per request')
                prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                inputs=processor(text=[prompt],images=images or None,return_tensors='pt').to(model.device)
                count=int(inputs['input_ids'].shape[-1]);started=time.perf_counter()
                temperature=float(request.get('temperature',0));options={'do_sample':temperature>0}
                if temperature>0:options['temperature']=temperature
                with torch.inference_mode():
                    output=model.generate(**inputs,max_new_tokens=min(512,int(request.get('max_tokens',300))),**options)
                tokens=output[:,count:];text=processor.batch_decode(tokens,skip_special_tokens=True)[0]
                self.respond(200,{'object':'chat.completion','model':identity,
                    'choices':[{'index':0,'message':{'role':'assistant','content':text},'finish_reason':'stop'}],
                    'usage':{'prompt_tokens':count,'completion_tokens':tokens.shape[-1],'total_tokens':count+tokens.shape[-1]},
                    'inference_seconds':time.perf_counter()-started})
            except (ValueError,KeyError,OSError,RuntimeError) as error:
                self.respond(500,{'error':type(error).__name__+': '+str(error)})
    print(json.dumps(health),flush=True)
    HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__=='__main__':main()
