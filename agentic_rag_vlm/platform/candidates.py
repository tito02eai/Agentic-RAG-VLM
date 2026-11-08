"""Configurable RGB region proposals for a registered, controlled tabletop.

Regions supply pixels only. Object identities remain VLM decisions and metric
geometry still requires site measurements. No simulator or cached object pose.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageFilter


class ColorRegionCandidates:
    def __init__(self, config):
        self.config = dict(config)
        if not isinstance(config.get('image_size_px'),list) or not isinstance(config.get('roi_xyxy'),list):
            raise ValueError('site image dimensions and tabletop ROI are pending')
        self.size = tuple(config['image_size_px'])
        self.roi = tuple(config['roi_xyxy'])
        if len(self.size)!=2 or any(type(x) is not int or x<=0 for x in self.size):
            raise ValueError('candidate image_size_px must match original camera dimensions')
        if (len(self.roi)!=4 or any(type(x) is not int for x in self.roi)
                or not 0<=self.roi[0]<self.roi[2]<=self.size[0]
                or not 0<=self.roi[1]<self.roi[3]<=self.size[1]):
            raise ValueError('candidate ROI must be inside the original image')
        self.bands = config['hsv_bands']
        if not isinstance(self.bands,list) or not 1<=len(self.bands)<=8:
            raise ValueError('configure 1–8 HSV proposal bands; no object labels')
        for band in self.bands:
            if set(band)!={'hue_min','hue_max','saturation_min','value_min'} or any(type(v) is not int or not 0<=v<=255 for v in band.values()):
                raise ValueError('HSV proposal bands use integer 0–255 image values only')
        self.minimum = config.get('minimum_area_px',15)
        self.maximum = config.get('maximum_area_px',20000)
        if type(self.minimum) is not int or type(self.maximum) is not int or not 1<=self.minimum<=self.maximum:
            raise ValueError('invalid candidate area bounds')

    def __call__(self, observation):
        with Image.open(observation.image_paths[0]) as image:
            if image.size!=self.size:
                raise ValueError('camera resolution changed; revalidate ROI and calibration')
            hsv=np.asarray(image.convert('HSV'))
        x0,y0,x1,y1=self.roi
        crop=hsv[y0:y1,x0:x1];h,s,v=[crop[:,:,i] for i in range(3)]
        output=[]
        for band in self.bands:
            low,high=band['hue_min'],band['hue_max']
            hue=(h>=low)&(h<=high) if low<=high else (h>=low)|(h<=high)
            mask=hue&(s>=band['saturation_min'])&(v>=band['value_min'])
            connected=np.array(Image.fromarray(mask.astype('uint8')*255).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3)))>0
            # Flood-fill only thresholded regions; works without SciPy/OpenCV.
            remaining=connected.copy();height,width=remaining.shape
            for row,col in zip(*np.where(connected)):
                if not remaining[row,col]:continue
                stack=[(row,col)];remaining[row,col]=False;points=[]
                while stack:
                    yy,xx=stack.pop()
                    if mask[yy,xx]:points.append((yy,xx))
                    for ny in range(max(0,yy-1),min(height,yy+2)):
                        for nx in range(max(0,xx-1),min(width,xx+2)):
                            if remaining[ny,nx]:remaining[ny,nx]=False;stack.append((ny,nx))
                if not self.minimum<=len(points)<=self.maximum:continue
                if any(yy in (0,height-1) or xx in (0,width-1) for yy,xx in points):
                    raise ValueError('candidate intersects the tabletop ROI boundary; clear the view or revalidate the ROI')
                pixels=np.asarray(points);ys,xs=pixels[:,0]+y0,pixels[:,1]+x0
                output.append({'bbox_xyxy':[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1],
                    'pixel_xy':[float(np.median(xs)),float(np.median(ys))]})
        output.sort(key=lambda r:(r['bbox_xyxy'][1],r['bbox_xyxy'][0]))
        # Overlapping hue bands must not silently duplicate one image region.
        boxes=[tuple(r['bbox_xyxy']) for r in output]
        if len(set(boxes))!=len(boxes) or len(output)>8:
            raise ValueError('ambiguous or excessive color proposals; adjust site thresholds')
        return [dict(candidate_id=f'region_{i}',**r) for i,r in enumerate(output)]


def load_candidate_provider(path):
    data=json.loads(Path(path).read_text())
    if data.get('kind')!='color_regions':raise ValueError('unsupported candidate provider')
    return ColorRegionCandidates(data)
