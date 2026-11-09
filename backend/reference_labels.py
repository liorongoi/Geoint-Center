"""Pre-inference image labeling: display copies and exact source-grid provenance."""
import hashlib
import json
import math
import re
from pathlib import Path
import numpy as np
from PIL import Image
from rasterio import Affine
from rasterio.warp import transform_geom
from raster_pipeline import read_stack


def fingerprint(record):
    return hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()


def preview_path(root, token):
    if not re.fullmatch('[a-f0-9]{64}',token):raise ValueError('Invalid reference preview')
    return Path(root)/'reference-previews'/token


def prepare_preview(root, mission_id, job_id, image, record):
    identity={'mission_id':mission_id,'job_id':job_id,'image_id':image['id'],'access_sha256':fingerprint(record)}
    token=fingerprint(identity);folder=preview_path(root,token);folder.mkdir(parents=True,exist_ok=True)
    metadata=folder/'grid.json'
    if metadata.is_file() and (folder/'source.png').is_file():return json.loads(metadata.read_text())
    stack=read_stack(record,image,max_size=1024)
    if not all(b in stack.bands for b in ('RED','GREEN','BLUE')):raise ValueError('Connect named RED, GREEN and BLUE display bands in Source data first')
    rgba=np.zeros((*stack.valid.shape,4),dtype='uint8')
    for c,band in enumerate(('RED','GREEN','BLUE')):
        values=stack.pixels[stack.bands.index(band)];valid=stack.valid&np.isfinite(values)
        if valid.any():
            low,high=np.percentile(values[valid],[2,98]);high=max(high,low+1e-6)
            rgba[:,:,c]=np.where(valid,255*np.clip((values-low)/(high-low),0,1),0).astype('uint8')
    rgba[:,:,3]=stack.valid.astype('uint8')*255
    from uuid import uuid4
    temporary=folder/(uuid4().hex+'.png');Image.fromarray(rgba).save(temporary);temporary.replace(folder/'source.png')
    result={**identity,'id':token,'width':rgba.shape[1],'height':rgba.shape[0],
            'transform':list(stack.profile['transform'])[:6],'crs':str(stack.profile['crs']),
            'source_record':record,'bands':['RED','GREEN','BLUE'],
            'source':f'/api/reference-images/{token}/source.png','display_only':True}
    temp=folder/(uuid4().hex+'.json');temp.write_text(json.dumps(result));temp.replace(metadata)
    return result


def label_geometry(preview, vertices, kind):
    if kind not in ('rectangle','polygon','point'):raise ValueError('Choose rectangle, polygon or point')
    if not isinstance(vertices,list) or len(vertices)>2000:raise ValueError('At most 2,000 vertices per region')
    if len(vertices)<(1 if kind=='point' else 3):raise ValueError('Draw a complete region first')
    for p in vertices:
        if not isinstance(p,list) or len(p)!=2 or any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or not 0<=v<=1 for v in p):
            raise ValueError('Every vertex must be inside the source image')
    if kind=='point' and len(vertices)!=1:raise ValueError('A point label needs one location')
    if kind!='point':
        area=sum(p[0]*q[1]-q[0]*p[1] for p,q in zip(vertices,vertices[1:]+vertices[:1]))
        if abs(area)<1e-10:raise ValueError('Draw a region with a non-zero area')
        # Reject self-crossing annotations rather than rasterizing an ambiguous class.
        def orientation(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        edges=list(zip(vertices,vertices[1:]+vertices[:1]));n=len(edges)
        for i,(a,b) in enumerate(edges):
            for j in range(i+2,n):
                if i==0 and j==n-1:continue
                c,d=edges[j]
                if orientation(a,b,c)*orientation(a,b,d)<0 and orientation(c,d,a)*orientation(c,d,b)<0:raise ValueError('Polygon edges cross; redraw this region')
    affine=Affine(*preview['transform']);points=[list(affine*(x*preview['width'],y*preview['height'])) for x,y in vertices]
    geometry={'type':'Point','coordinates':points[0]} if kind=='point' else {'type':'Polygon','coordinates':[points+[points[0]]]}
    return transform_geom(preview['crs'],'EPSG:4326',geometry)
