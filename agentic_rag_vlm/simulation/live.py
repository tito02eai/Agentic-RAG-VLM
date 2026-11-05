"""Read-only progress output for a local simulation hosted by BluPe Runner."""
import json
from pathlib import Path
import time
from PIL import Image
from ..experience import atomic_json


class LiveSimulationObserver:
    def __init__(self,output):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.last_frame_at=0.;self.stage=None

    def frame(self,rgb,stage,simulated_time_s):
        now=time.monotonic()
        if stage==self.stage and now-self.last_frame_at<.2:return
        self.stage=stage;self.last_frame_at=now
        temporary=self.output/'frame.tmp.jpg'
        Image.fromarray(rgb).save(temporary,format='JPEG',quality=85)
        temporary.replace(self.output/'frame.jpg')
        atomic_json(self.output/'motion.json',{'stage':stage,'simulated_time_s':simulated_time_s,
            'updated_at':time.time(),'evidence_kind':'mujoco_contact_dynamics'})

    def event(self,event):
        with (self.output/'runtime_events.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
        atomic_json(self.output/'runtime.json',event)
