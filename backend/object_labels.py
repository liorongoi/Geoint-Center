"""Human-approved oriented object instances, isolated from semantic pixel references."""
import json,math
from uuid import uuid4
from typing import Literal
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel,Field
from reference_labels import preview_path,fingerprint,label_geometry

class ObjectLabelRequest(BaseModel):
 label:str=Field(min_length=1,max_length=80)
 vertices:list[list[float]]=Field(min_length=4,max_length=4)
 role:Literal['train','validation']='train'
 confidence:int=Field(default=3,ge=1,le=5)
 notes:str=Field(default='',max_length=2000)

def register(backend):
 app=backend.app
 def init():
  backend._init_database()
  with backend._database() as db:db.execute('CREATE TABLE IF NOT EXISTS object_labels(id TEXT PRIMARY KEY,mission_id TEXT,job_id TEXT,image_id TEXT,payload TEXT,created_at TEXT)')
 @app.get('/api/object-labels')
 async def listing(mission_id:str,job_id:str,image_id:str|None=None):
  backend.workflow_store.mission(mission_id,job_id);init()
  with backend._database() as db:rows=[json.loads(r['payload']) for r in db.execute('SELECT payload FROM object_labels WHERE mission_id=? AND job_id=? ORDER BY created_at',(mission_id,job_id))]
  return {'labels':[r for r in rows if not image_id or r['image_id']==image_id],'scope':'Object instances only; never semantic LULC training pixels'}
 @app.post('/api/reference-images/{preview_id}/object-labels')
 async def save(preview_id:str,request:ObjectLabelRequest):
  init()
  try:
   preview=json.loads((preview_path(backend.DATA_DIR,preview_id)/'grid.json').read_text());_,record=backend._reference_scope(preview['mission_id'],preview['job_id'],preview['image_id'])
   if fingerprint(record)!=preview['access_sha256']:raise ValueError('Source changed. Reopen the image before labeling.')
   xy=np.asarray(request.vertices,dtype='float64')
   if xy.shape!=(4,2) or not np.isfinite(xy).all() or (xy<0).any() or (xy>1).any():raise ValueError('Place all four corners inside the source image')
   pixels=xy*np.array([preview['width'],preview['height']]);edges=np.roll(pixels,-1,axis=0)-pixels;lengths=np.linalg.norm(edges,axis=1)
   if (lengths<1).any():raise ValueError('Draw an object at least one display pixel wide and high')
   if not np.allclose(edges[0],-edges[2],atol=.01) or not np.allclose(edges[1],-edges[3],atol=.01) or abs(np.dot(edges[0],edges[1]))>1e-4*lengths[0]*lengths[1]:raise ValueError('Use Rotated box to draw a valid rectangular object instance')
   geometry=label_geometry(preview,request.vertices,'polygon');angle=math.degrees(math.atan2(edges[0][1],edges[0][0]));created=backend._now()
   item={'id':str(uuid4()),'mission_id':preview['mission_id'],'job_id':preview['job_id'],'image_id':preview['image_id'],**request.model_dump(),'label':request.label.strip(),'geometry':geometry,'annotation_type':'oriented_box','angle_degrees_image':angle,'created_at':created,'source':{'type':'manual_object_reference','preview_id':preview_id,'kind':'rotated_box','vertices':request.vertices,'access_sha256':preview['access_sha256'],'raster_access_snapshot':preview['source_record'],'grid':{k:preview[k] for k in ('crs','transform','width','height')}}}
   if not item['label']:raise ValueError('Enter an object class')
   with backend._database() as db:db.execute('INSERT INTO object_labels VALUES(?,?,?,?,?,?)',(item['id'],item['mission_id'],item['job_id'],item['image_id'],json.dumps(item),created))
   backend.workflow_store.event('object.label_saved','Human-reviewed rotated object instance saved separately from LULC pixels',mission_id=item['mission_id'],image_id=item['image_id'],role=item['role'],label=item['label'],audit=True)
   return item
  except (ValueError,OSError,KeyError) as exc:raise HTTPException(422,str(exc)) from exc
