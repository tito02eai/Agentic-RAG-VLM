"""Serve pinned CLIP ViT-L/14 image features on localhost only."""
import argparse
import base64
from http.server import HTTPServer,BaseHTTPRequestHandler
import io
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agentic_rag_vlm.visual import CLIPVisualEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path",required=True)
    parser.add_argument("--port",type=int,default=18072)
    parser.add_argument("--device",default="cpu")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    encoder = CLIPVisualEncoder(args.model_path,device=args.device)
    class Handler(BaseHTTPRequestHandler):
        def respond(self,status,payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            if self.path != "/health":
                return self.respond(404,{"error":"not found"})
            self.respond(200,{"embedding_space":encoder.embedding_space,"dimension":768})
        def do_POST(self):
            if self.path != "/encode":
                return self.respond(404,{"error":"not found"})
            try:
                size = int(self.headers.get("Content-Length",0))
                if not 0 < size <= 20*1024*1024:
                    raise ValueError("request size limit exceeded")
                payload = json.loads(self.rfile.read(size))
                images = payload["images"]
                if not isinstance(images,list) or not 1 <= len(images) <= 32:
                    raise ValueError("supply 1–32 images")
                from PIL import Image
                with tempfile.TemporaryDirectory(prefix="rag-clip-") as folder:
                    paths = []
                    for index,data in enumerate(images):
                        image = Image.open(io.BytesIO(base64.b64decode(data,validate=True))).convert("RGB")
                        path = Path(folder)/f"{index}.png"
                        image.save(path)
                        paths.append(path)
                    vectors = encoder.encode(paths)
                self.respond(200,{"embedding_space":encoder.embedding_space,"embeddings":vectors.tolist()})
            except (ValueError,KeyError,OSError) as error:
                self.respond(400,{"error":str(error)})
    print(json.dumps({"port":args.port,"embedding_space":encoder.embedding_space}),flush=True)
    HTTPServer(("127.0.0.1",args.port),Handler).serve_forever()


if __name__ == "__main__":
    main()
