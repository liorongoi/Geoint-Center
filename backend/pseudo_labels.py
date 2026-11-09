"""Reviewed Dynamic World bootstrapping. Teacher predictions never become validation."""
import hashlib,json,re,shutil
from datetime import datetime,timezone
from pathlib import Path
from uuid import uuid4
import numpy as np
import rasterio
from rasterio.features import rasterize,bounds,is_valid_geom
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import transform_geom
from fastapi import HTTPException
from pydantic import BaseModel,Field
from starlette.concurrency import run_in_threadpool
from offline_policy import local_raster

STUDENTS=('random_forest','svm','xgboost','unet','prithvi','ensemble')

def now():return datetime.now(timezone.utc).isoformat()
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def signature(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def projected_mask(profile,geometry):
 return rasterize([(transform_geom('EPSG:4326',profile['crs'],geometry),1)],out_shape=(profile['height'],profile['width']),transform=profile['transform'],fill=0,dtype='uint8').astype(bool)

def validate_geometry(geometry):
 if not geometry or geometry.get('type') not in ('Polygon','MultiPolygon') or not is_valid_geom(geometry):raise ValueError('Select a polygon or rectangle for the training region')
 box=bounds(geometry)
 if not np.isfinite(box).all() or box[0]<-180 or box[2]>180 or box[1]<-90 or box[3]>90:raise ValueError('Training regions must use valid longitude/latitude coordinates')


def corrected_values(values,valid,profile,classes,edits,exclusions):
 values=values.copy()
 for edit in edits:
  if edit.get('role','train')!='train' or edit.get('confidence',3)<3:continue
  if edit['label'] not in classes:continue
  selected=projected_mask(profile,edit['geometry'])&valid
  values[selected]=classes.index(edit['label'])+1
 for geometry in exclusions:valid&=~projected_mask(profile,geometry)
 values[~valid]=0
 return values,valid


def read_reference(reference,stack):
 """Verify immutable teacher bytes before GDAL; warp categorical codes only."""
 path=Path(local_raster(reference['path']))
 if path.is_symlink() or not path.is_file() or digest(path)!=reference['sha256']:raise ValueError('Approved pseudo-label snapshot is missing or changed')
 p=stack.profile
 with rasterio.open(path,driver='GTiff') as src:
  with WarpedVRT(src,crs=p['crs'],transform=p['transform'],width=p['width'],height=p['height'],resampling=Resampling.nearest,nodata=0) as vrt:
   values=vrt.read(1);valid=(values>0)&(vrt.read_masks(1)>0)&stack.valid
 return corrected_values(values,valid,p,reference['classes'],reference['edits'],reference['validation_exclusions'])[0]


class StartLabels(BaseModel):
 pass
class ApproveLabels(BaseModel):
 review_signature: str = Field(min_length=64,max_length=64)
 scope: str = Field(pattern='^(image|region)$')
 geometry: dict | None = None
 reviewed: bool = False
 notes: str = Field(default='',max_length=500)


def register(app,database,data,get_run,result_folder,initialize,auto_training):
 root=data/'pseudo-labels'
 def init():
  initialize()
  with database() as db:db.execute('CREATE TABLE IF NOT EXISTS pseudo_label_sets (id TEXT PRIMARY KEY,mission_id TEXT NOT NULL,job_id TEXT NOT NULL,payload TEXT NOT NULL,created_at TEXT NOT NULL)')
 def put(item):
  with database() as db:db.execute('INSERT INTO pseudo_label_sets VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload',(item['id'],item['mission_id'],item['job_id'],json.dumps(item),item['created_at']))
 def get(identifier):
  init()
  if not re.fullmatch('[a-f0-9-]{36}',identifier):raise HTTPException(422,'Invalid pseudo-label set')
  with database() as db:row=db.execute('SELECT payload FROM pseudo_label_sets WHERE id=?',(identifier,)).fetchone()
  if not row:raise HTTPException(404,'Pseudo-label draft not found')
  return json.loads(row['payload'])
 def context(item):
  with database() as db:
   workspace=json.loads(db.execute('SELECT payload FROM workspace WHERE id=1').fetchone()['payload'])
   mission=next((m for m in workspace['missions'] if m['id']==item['mission_id']),{})
   job=next((j for j in mission.get('jobs',[]) if j['id']==item['job_id']),{})
   image=next((i for i in job.get('images',[]) if i['id']==item['image_id']),None)
   labels=[json.loads(r['payload']) for r in db.execute('SELECT payload FROM ground_truth WHERE mission_id=? AND job_id=? ORDER BY created_at,rowid',(item['mission_id'],item['job_id']))]
   reviews=[json.loads(r['payload']) for r in db.execute('SELECT payload FROM manual_reviews WHERE run_id=? AND image_index=? ORDER BY created_at,rowid',(item['source_run'],item['image_index']))]
  if not image or mission.get('control') in ('delete','deleted'):raise HTTPException(409,'The source image is no longer in an active mission')
  if image.get('benchmark_split') in ('validation','test'):raise HTTPException(409,'Held-out benchmark imagery cannot supply training labels')
  protected=[l for l in labels if (not l.get('image_id') or l['image_id']==item['image_id']) and (l.get('role')=='validation' or l.get('benchmark_split') in ('validation','test'))]
  edits=[r for r in reviews if r.get('role','train')=='train' and r.get('confidence',3)>=3]
  exclusions=[l['geometry'] for l in protected if l.get('geometry')]
  token=signature({'reviews':reviews,'heldout':protected,'image':image})
  return edits,exclusions,token
 def public(item):
  edits,exclusions,token=context(item)
  return {k:v for k,v in item.items() if k!='teacher_path'}|{'review_signature':token,'corrections':len(edits),'validation_regions':len(exclusions),'supported_students':list(STUDENTS)}

 @app.post('/api/results/{run_id}/{image_index}/pseudo-labels')
 async def start(run_id: str,image_index: int):
  init();report=await get_run(run_id)
  from pretrained_lulc import PRETRAINED
  if report.get('classes')!=PRETRAINED['dynamicworld']['classes']:raise HTTPException(422,'Dynamic World taxonomy does not match its native class head')
  if report.get('requested_models')!=['dynamicworld'] or not any(m.get('id')=='dynamicworld' and m.get('status')=='complete' for m in report.get('models',[])):
   raise HTTPException(422,'Choose a completed Dynamic World result with its native nine-class head')
  if not 0<=image_index<len(report.get('outputs',[])):raise HTTPException(404,'Image output not found')
  item={'id':str(uuid4()),'mission_id':report['mission_id'],'job_id':report['job_id'],'image_id':report['outputs'][image_index]['image_id'],'source_run':run_id,'image_index':image_index,'classes':report['classes'],'status':'draft','created_at':now(),'kind':'reviewed_pseudo_labels','source_access_snapshot':report.get('raster_access_snapshot',{}).get(str(report['outputs'][image_index]['image_id'])),'label_role':'training_only'}
  if report.get('benchmark'):raise HTTPException(409,'Benchmark results cannot be reused as training labels through this workflow')
  context(item)
  with database() as db:
   existing=[json.loads(r['payload']) for r in db.execute('SELECT payload FROM pseudo_label_sets WHERE mission_id=? AND job_id=? ORDER BY created_at DESC',(item['mission_id'],item['job_id']))]
  prior=next((d for d in existing if d['source_run']==run_id and d['image_index']==image_index and d['status']=='draft'),None)
  if prior:return public(prior)
  folder=root/item['id'];folder.mkdir(parents=True);source=result_folder(run_id)/f'{image_index}-consensus.tif';target=folder/'teacher.tif'
  def copy():
   before=digest(source);shutil.copyfile(source,target)
   if before!=digest(target) or before!=digest(source):raise ValueError('Teacher raster changed during snapshot creation')
   with rasterio.open(target) as src:
    item.update(teacher_sha256=before,teacher_path=str(target.resolve()),grid={'width':src.width,'height':src.height,'crs':str(src.crs),'transform':list(src.transform)[:6]})
  try:await run_in_threadpool(copy)
  except Exception as exc:shutil.rmtree(folder,ignore_errors=True);raise HTTPException(422,str(exc)) from exc
  put(item);return public(item)

 @app.get('/api/pseudo-labels')
 async def listing(mission_id: str,job_id: str):
  init()
  with database() as db:items=[json.loads(r['payload']) for r in db.execute('SELECT payload FROM pseudo_label_sets WHERE mission_id=? AND job_id=? ORDER BY created_at DESC',(mission_id,job_id))]
  return {'sets':[public(i) for i in items]}

 @app.get('/api/pseudo-labels/{identifier}')
 async def status(identifier: str):return public(get(identifier))

 @app.post('/api/pseudo-labels/{identifier}/approve')
 async def approve(identifier: str,request: ApproveLabels):
  item=get(identifier)
  if item['status']!='draft':raise HTTPException(409,'This draft was already approved; create a new draft for additional regions')
  if not request.reviewed:raise HTTPException(422,'Confirm that you reviewed this training region and understand these remain model-derived labels')
  edits,exclusions,token=context(item)
  if token!=request.review_signature:raise HTTPException(409,'Corrections or validation references changed. Refresh the draft and review its updated summary before approving.')
  profile=item['grid']
  if request.scope=='image':
   from rasterio.transform import array_bounds
   w,s,e,n=array_bounds(profile['height'],profile['width'],rasterio.Affine(*profile['transform']))
   geometry=transform_geom(profile['crs'],'EPSG:4326',{'type':'Polygon','coordinates':[[[w,s],[e,s],[e,n],[w,n],[w,s]]]})
  else:geometry=request.geometry
  try:validate_geometry(geometry)
  except ValueError as exc:raise HTTPException(422,str(exc)) from exc
  reference={'path':item['teacher_path'],'sha256':item['teacher_sha256'],'classes':item['classes'],'edits':edits,'validation_exclusions':exclusions}
  def count():
   if digest(Path(local_raster(item['teacher_path'])))!=item['teacher_sha256']:raise ValueError('Teacher snapshot changed; refusing approval')
   counts=np.zeros(len(item['classes'])+1,dtype='int64')
   with rasterio.open(item['teacher_path'],driver='GTiff') as src:
    for _,window in src.block_windows(1):
     p={'width':int(window.width),'height':int(window.height),'crs':src.crs,'transform':src.window_transform(window)}
     region=projected_mask(p,geometry)
     if not region.any():continue
     values=src.read(1,window=window);valid=(values>0)&(src.read_masks(1,window=window)>0)&region
     values,valid=corrected_values(values,valid,p,item['classes'],edits,exclusions)
     if valid.any():counts+=np.bincount(values[valid],minlength=len(counts))[:len(counts)]
   return counts
  try:counts=await run_in_threadpool(count)
  except Exception as exc:raise HTTPException(422,str(exc)) from exc
  if not counts[1:].sum():raise HTTPException(422,'No eligible training pixels remain in this region after excluding nodata and validation')
  # Re-read before committing: a long scan must not bypass a newly saved validation region.
  if context(item)[2]!=token:raise HTTPException(409,'References changed during approval. Refresh and review the draft again.')
  created=now();records=[]
  for value,label in enumerate(item['classes'],1):
   if not counts[value]:continue
   records.append({'id':str(uuid4()),'mission_id':item['mission_id'],'job_id':item['job_id'],'image_id':item['image_id'],'label':label,'geometry':geometry,'role':'train','confidence':3,'origin':'reviewed_pseudo_label','created_at':created,'pixels':int(counts[value]),'notes':request.notes,'raster_reference':{**reference,'class_value':value},'source':{'type':'dynamicworld_pseudo_label','set_id':identifier,'source_run':item['source_run'],'teacher_sha256':item['teacher_sha256'],'review_signature':token,'human_reviewed':True,'correction_ids':[r['id'] for r in edits],'independent_ground_truth':False}})
  original=json.dumps(item);item.update(status='approved',approved_at=created,approval_geometry=geometry,review_signature=token,scope=request.scope,training_pixels=int(counts[1:].sum()),class_counts={label:int(counts[v]) for v,label in enumerate(item['classes'],1) if counts[v]},label_ids=[r['id'] for r in records],notes=request.notes)
  with database() as db:
   db.execute('BEGIN IMMEDIATE')
   if context(item)[2]!=token:raise HTTPException(409,'References changed before approval committed. Refresh and review again.')
   updated=db.execute('UPDATE pseudo_label_sets SET payload=? WHERE id=? AND payload=?',(json.dumps(item),identifier,original)).rowcount
   if updated!=1:raise HTTPException(409,'Draft already approved; no duplicate labels were created')
   db.executemany('INSERT INTO ground_truth VALUES(?,?,?,?,?)',[(r['id'],r['mission_id'],r['job_id'],json.dumps(r),r['created_at']) for r in records])
  return {'set':public(item),'labels':records,'message':'Reviewed pseudo-labels saved for training only. Choose a supported student model and train a separate candidate version. Independent validation is unchanged.'}
