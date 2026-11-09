"""Persistent multi-AOI DAG dispatch; processing always uses approved local workers."""
from __future__ import annotations
import asyncio,json,hashlib,shutil,zipfile
from pathlib import Path
from uuid import uuid4
from datetime import date,datetime,timezone,timedelta
import numpy as np,rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.features import bounds,is_valid_geom
from rasterio.warp import transform_geom
from fastapi import HTTPException
from pydantic import BaseModel,Field
from starlette.concurrency import run_in_threadpool
from mission_workspace import now,sha,ACTIVE
from workflow_graph import validate_graph,geometries_intersect,footprint,TASK_TYPES
from workflow_api import OperationalRequest

class PipelineRequest(BaseModel):
 mission_id:str
 name:str=Field(min_length=1,max_length=120)
 geometry:dict
 sensors:list[str]=Field(min_length=1,max_length=10)
 task_types:list[str]=Field(default_factory=lambda:['lulc'],min_length=1,max_length=10)
 source:str=Field(default='local',pattern='^(local|sentinel-2|sentinel-2-l1c|landsat|eo1-hyperion)$')
 watch_id:str|None=None
 nodes:list[dict]=Field(min_length=2,max_length=50)
 links:list[dict]=Field(max_length=150)
 max_size:int=Field(default=384,ge=32,le=1024)
 cloud_max:float=Field(default=100,ge=0,le=100)
 backfill:bool=False
class TaskTypeRequest(BaseModel):
 id:str=Field(pattern='^[a-z][a-z0-9_-]{2,50}$')
 name:str=Field(min_length=1,max_length=100)
 description:str=Field(max_length=500)
 adapter_tasks:list[str]=Field(min_length=1,max_length=3)


def register(backend,store):
 app=backend.app;tasks={};scans={};scan_slots=asyncio.Semaphore(2);loop_task=None;last_poll={}
 def types():return {**TASK_TYPES,**{t['id']:t for t in store.all('workflow_tasks')}}
 def public_validation(p,qualified=False):
  try:
   graph={**p,'task_types':list({x for t in p['task_types'] for x in types().get(t,{}).get('adapter_tasks',[t])})}
   v=validate_graph(graph,store,qualified);return {k:value for k,value in v.items() if k!='models'}
  except (ValueError,KeyError,HTTPException) as exc:raise HTTPException(422,str(getattr(exc,'detail',exc))) from exc
 def valid_geometry(g):
  if g.get('type') not in ('Polygon','MultiPolygon','Point') or not is_valid_geom(g):raise HTTPException(422,'Draw or import a valid AOI')
  box=bounds(g)
  if not np.isfinite(box).all() or box[0]<-180 or box[2]>180 or box[1]<-90 or box[3]>90 or box[2]-box[0]>180:raise HTTPException(422,'Use longitude/latitude AOIs; split antimeridian-crossing areas')
 def validate_request(request):
  store.mission(request.mission_id);valid_geometry(request.geometry)
  for t in request.task_types:
   if t not in types():raise HTTPException(422,'Register this mission type first')
   if not types()[t].get('available'):raise HTTPException(422,types()[t].get('reason','No supported adapter installed'))
  validated=public_validation(request.model_dump())
  if request.source!='local' and 'enhanced' in validated['kinds'].values():
   raise HTTPException(422,'DSen2 automatic pipelines require a local Sentinel-2 SAFE catalog source')
 @app.get('/api/mission-workspace/task-types')
 async def task_types():return {'types':list(types().values())}
 @app.post('/api/mission-workspace/task-types')
 async def add_task_type(request: TaskTypeRequest):
  if request.id in types():raise HTTPException(409,'Mission type already exists')
  if any(t not in TASK_TYPES for t in request.adapter_tasks):raise HTTPException(422,'Extensions may select installed compatible adapter families. A new detector/worker must be installed and registered in source before use; generated code is never executed.')
  item={**request.model_dump(),'mission_id':'','created_at':now(),'available':True};store.put('workflow_tasks',item);store.event('task_type.created','Mission type registered with installed adapters',audit=True,task_type=item['id']);return item
 @app.post('/api/mission-workspace/check-pipeline')
 async def check_pipeline(request:PipelineRequest):
  validate_request(request);return public_validation(request.model_dump())
 @app.get('/api/mission-workspace/pipelines')
 async def pipelines(mission_id:str):
  return {'pipelines':[p for p in store.all('workflow_pipelines',mission_id) if p['state']!='deleted'],'executions':store.all('workflow_executions',mission_id)[:200]}
 @app.post('/api/mission-workspace/pipelines')
 async def create(request: PipelineRequest):
  validate_request(request);p={'id':str(uuid4()),**request.model_dump(),'created_at':now(),'state':'draft','version':1,'activation_count':0};p['validation']=public_validation(p);store.put('workflow_pipelines',p);store.event('pipeline.created','Automatic pipeline saved as a draft',mission_id=p['mission_id'],pipeline_id=p['id'],audit=True);return p
 @app.put('/api/mission-workspace/pipelines/{identifier}')
 async def update(identifier:str,request:PipelineRequest):
  p=store.get('workflow_pipelines',identifier)
  if p['mission_id']!=request.mission_id:raise HTTPException(422,'A pipeline cannot move between missions')
  if p['state']=='active' or any(e['status'] in ACTIVE for e in store.all('workflow_executions',p['mission_id']) if e['pipeline_id']==identifier):raise HTTPException(409,'Stop active runs before changing the pipeline version')
  validate_request(request);p.update(**request.model_dump(),version=p['version']+1,state='draft',activation_count=0);p['validation']=public_validation(p);store.put('workflow_pipelines',p);return p
 @app.post('/api/mission-workspace/pipelines/{identifier}/validate')
 async def validate(identifier:str):return public_validation(store.get('workflow_pipelines',identifier))
 @app.post('/api/mission-workspace/pipelines/{identifier}/{action}')
 async def control(identifier:str,action:str):
  if action not in ('activate','pause','resume','stop','delete','scan'):raise HTTPException(422,'Unknown pipeline control')
  p=store.get('workflow_pipelines',identifier)
  if p['state']=='deleted':raise HTTPException(409,'Pipeline deleted')
  if action in ('activate','resume'):
   if not backend.mission_control.allowed(store.database,p['mission_id']):raise HTTPException(409,'Resume this mission before activating its pipeline')
   p['validation']=public_validation(p,True)
   releases={n['profile_id']:store.model(n['profile_id'],True)['release']['id'] for n in p['nodes'] if n['type']=='algorithm'}
   from dsen2 import MODELS as ENHANCEMENT_MODELS, checkpoint
   for profile_id in releases:
    algorithm=store.model(profile_id)['algorithm']
    if algorithm in ENHANCEMENT_MODELS and checkpoint(algorithm,store.root()) is None:
     raise HTTPException(409,'Install the local DSen2 model pack and TensorFlow runtime before activation')
   if action=='resume' and p.get('model_releases')!=releases:raise HTTPException(409,'Model versions changed. Stop existing runs, review the pipeline and activate it with the new versions.')
   if not p.get('activation_count') and not p['backfill']:
    for image in await discover(p):remember(p,image,'baseline')
   p.update(state='active',activation_count=p.get('activation_count',0)+1,model_releases=releases,activated_at=now())
  elif action=='scan':
   if p['state']!='active':raise HTTPException(409,'Activate the pipeline before scanning')
   await tick(p);return store.get('workflow_pipelines',identifier)
  else:p['state']={'pause':'paused','stop':'stopped','delete':'deleted'}[action]
  store.put('workflow_pipelines',p);store.event('pipeline.'+action,'Pipeline '+action,mission_id=p['mission_id'],pipeline_id=identifier,audit=True)
  for e in store.all('workflow_executions',p['mission_id']):
   if e['pipeline_id']!=identifier or e['status'] not in ACTIVE:continue
   e['control']='resume' if action in ('activate','resume') else 'pause' if action=='pause' else 'stop';store.put('workflow_executions',e,control_change=True)
   for node in e.get('node_runs',{}).values():
    for run in node.get('runs',[]):
     try:await backend.workflow_run_control(run['id'],'resume' if e['control']=='resume' else 'pause' if e['control']=='pause' else 'stop')
     except HTTPException:pass
  return p
 def source_key(p,image):
  value=[p['id'],p['version'],image['id'],image.get('sha256'),image.get('review_revision'),[(s.get('key'),s.get('sha256')) for s in image.get('sources',[])]]
  return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
 def remember(p,image,state,execution_id=None):
  key=source_key(p,image)
  with store.database() as db:
   try:db.execute('INSERT INTO workflow_discoveries VALUES(?,?,?)',(key,p['id'],json.dumps({'state':state,'execution_id':execution_id,'at':now()})))
   except Exception as exc:
    import sqlite3
    if isinstance(exc,sqlite3.IntegrityError):return False
    raise
  return True
 async def discover(p):
  if p['source']=='local':
   catalog=backend.search_local_imagery.catalog
   if p.get('watch_id'):
    watch=catalog.watch(p['watch_id'])
    if not watch['enabled']:return []
    if watch['status'] in ('error','scanning','settling'):return []
   images=await run_in_threadpool(catalog.images)
  else:
   import offline_policy
   from public_raster_transport import public_http_client
   from data_policy import public_search_geometry
   if offline_policy.enabled():raise ValueError('Online catalog unavailable in offline-only mode; local monitoring remains available')
   endpoint=backend.CATALOG_ENDPOINTS[p['source']]
   begin=date(1972,1,1) if p['backfill'] else datetime.fromisoformat(p.get('activated_at',p['created_at'])).date()-timedelta(days=2)
   payload={'collections':[backend.COLLECTIONS[p['source']]],'intersects':public_search_geometry(p['geometry']),'datetime':f'{begin}T00:00:00Z/{date.today()}T23:59:59Z','limit':100}
   images=[]
   async with public_http_client([endpoint],timeout=45,allow_search_post=True) as client:
    data=await backend._catalog_request(client,endpoint,payload)
    pages=0
    while True:
     pages+=1
     images.extend(backend._normalize_item(i,p['source']) for i in data.get('features',[]))
     link=backend._next_link(data,endpoint)
     if not link:break
     if pages>=100:raise ValueError('Catalog has more than 100 pages; narrow the AOI before monitoring. No incomplete discovery batch was published.')
     data=await backend._catalog_request(client,endpoint,payload,link)
   images=[i for i in images if i.get('cloud_cover') is None or i['cloud_cover']<=p['cloud_max']]
  sensor_alias={'sentinel-2':'sentinel-2','sentinel-2-l1c':'sentinel-2','landsat':'landsat','eo1-hyperion':'hyperion'}
  return [i for i in images if i.get('status','ready')=='ready' and (not p.get('watch_id') or i.get('watch_id')==p['watch_id']) and any(s=='*' or s.lower() in (str(i.get('sensor','')).lower(),str(i.get('catalog','')).lower(),sensor_alias.get(i.get('catalog'),'')) for s in p['sensors']) and geometries_intersect(p['geometry'],footprint(i))]
 async def attach(p,image,execution):
  # Persist a separate job with immutable band snapshots before queuing processing.
  with store.database() as db:row=db.execute('SELECT payload FROM workspace WHERE id=1').fetchone()
  workspace=json.loads(row['payload']);mission=next(m for m in workspace['missions'] if m['id']==p['mission_id']);job={'id':str(uuid4()),'name':p['name']+' · '+(image.get('name') or image['id']),'images':[image],'aoi':p['geometry'],'pipeline_id':p['id'],'createdAt':now()}
  def persist_job():
   with store.database() as db:
    if image.get('catalog')=='local':backend.attach_local_imagery(db,[{'id':mission['id'],'jobs':[job]}],now())
    if not db.in_transaction:db.execute('BEGIN IMMEDIATE')
    current=json.loads(db.execute('SELECT payload FROM workspace WHERE id=1').fetchone()['payload']);m=next(m for m in current['missions'] if m['id']==p['mission_id']);m.setdefault('jobs',[]).append(job);db.execute('UPDATE workspace SET payload=?,updated_at=? WHERE id=1',(json.dumps(current),now()))
    execution['job_id']=job['id'];db.execute('UPDATE workflow_executions SET payload=? WHERE id=?',(json.dumps(execution),execution['id']))
  await run_in_threadpool(persist_job)
  if image.get('catalog')!='local':await backend.configure_raster_access(backend.RasterAccessRequest(mission_id=p['mission_id'],job_id=job['id'],image_ids=[image['id']],mode='stream'))
  return job
 def record_scan(p,error):
  with store.database() as db:
   db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT payload FROM workflow_pipelines WHERE id=?',(p['id'],)).fetchone()
   if not row:return
   current=json.loads(row['payload'])
   if current['version']!=p['version']:return
   current.update(last_scan=now(),last_error=error);db.execute('UPDATE workflow_pipelines SET payload=? WHERE id=?',(json.dumps(current),p['id']))
 async def tick(p):
  if p['state']!='active' or not backend.mission_control.allowed(store.database,p['mission_id']):return
  try:
   public_validation(p,True)
   for identifier,release in p.get('model_releases',{}).items():
    if store.model(identifier,True)['release']['id']!=release:raise ValueError('A model release changed; review and reactivate the pipeline')
   images=await discover(p)
   current=store.get('workflow_pipelines',p['id'])
   if current['state']!='active' or current['version']!=p['version']:return
   for image in images:
    identifier=str(uuid4())
    e={'id':identifier,'mission_id':p['mission_id'],'pipeline_id':p['id'],'pipeline_version':p['version'],'definition':p,'image':image,'created_at':now(),'status':'queued','control':'resume','node_runs':{}}
    with store.database() as db:
     db.execute('BEGIN IMMEDIATE')
     current=json.loads(db.execute('SELECT payload FROM workflow_pipelines WHERE id=?',(p['id'],)).fetchone()['payload'])
     if current['state']!='active' or current['version']!=p['version']:break
     added=db.execute('INSERT OR IGNORE INTO workflow_discoveries VALUES(?,?,?)',(source_key(p,image),p['id'],json.dumps({'state':'reserved','execution_id':identifier,'at':now()}))).rowcount
     if not added:continue
     db.execute('INSERT INTO workflow_executions VALUES(?,?,?,?)',(identifier,p['mission_id'],json.dumps(e),e['created_at']))
    store.event('pipeline.image_queued','Complete image intersects the AOI and was queued once',mission_id=p['mission_id'],pipeline_id=p['id'],run_id=identifier,image_id=image['id'],audit=True)
   record_scan(p,None)
  except Exception as exc:
   text=str(getattr(exc,'detail',exc))
   if p.get('last_error')!=text:store.event('pipeline.waiting',text,level='warning',mission_id=p['mission_id'],pipeline_id=p['id'])
   record_scan(p,text)
 async def state(run):
  for table in ('inference_runs','segmentation_runs'):
   with store.database() as db:row=db.execute(f'SELECT payload FROM {table} WHERE id=?',(run['id'],)).fetchone()
   if row:return json.loads(row['payload'])
  raise ValueError('Node execution disappeared')
 async def compute(function,*args):
  from functools import partial
  from resource_admission import run_reserved
  return await asyncio.get_running_loop().run_in_executor(backend.model_executor,partial(run_reserved,'Pipeline raster operation',2,function,*args))
 async def execute(e):
  p=e['definition']
  try:
   if not e.get('job_id'):
    # Native connections must be created and consumed in the same thread.
    job=await attach(p,e['image'],e);e['job_id']=job['id'];store.put('workflow_executions',e)
   validation=public_validation(p);order=validation['order'];incoming=validation['incoming'];nodes={n['id']:n for n in p['nodes']}
   while True:
    latest=store.get('workflow_executions',e['id']);e['control']=latest.get('control','resume')
    if e['control']=='stop':
     children=[await state(r) for node in e['node_runs'].values() for r in node.get('runs',[])]
     if any(r.get('status') in ACTIVE for r in children):
      e.update(status='stopping',stage='Waiting for native processing to stop at a safe checkpoint');store.put('workflow_executions',e);await asyncio.sleep(1);continue
     e.update(status='cancelled',stage='Stopped; completed outputs retained');break
    if e['control']=='pause' or not backend.mission_control.allowed(store.database,e['mission_id']):
     e.update(status='paused',stage='Paused at orchestration checkpoint');store.put('workflow_executions',e);return
    ready_models=True
    for identifier,release in p.get('model_releases',{}).items():
     model=store.model(identifier)
     if model['state']!='qualified':ready_models=False
     elif model.get('release',{}).get('id')!=release:raise ValueError('Pinned model release changed. Stop this execution and activate a reviewed pipeline version.')
    if not ready_models:e.update(status='blocked',stage='Waiting for model qualification');store.put('workflow_executions',e);return
    progress=False;e.update(status='running')
    for identifier in order:
     if store.get('workflow_executions',e['id']).get('control')!='resume':break
     node=nodes[identifier];saved=e['node_runs'].get(identifier)
     if saved and saved['status']=='complete':continue
     if saved and saved.get('runs'):
      runs=[await state(r) for r in saved['runs']]
      if any(r['status'] in ('failed','needs_input','partial') for r in runs):raise ValueError('Algorithm node failed: '+str(next((r.get('error') or r.get('stage') for r in runs if r['status']!='complete'),'Unknown error')))
      if any(r['status']=='cancelled' for r in runs):raise ValueError('Algorithm node was stopped; start a new pipeline execution')
      if any(r['status']=='interrupted' for r in runs):
       saved.setdefault('attempts',[]).extend(saved.pop('runs'));saved['status']='pending'
      elif all(r['status']=='complete' for r in runs):saved.update(status='complete',runs=runs);progress=True;continue
      else:continue
     if any(e['node_runs'].get(parent,{}).get('status')!='complete' for parent in incoming[identifier]):continue
     kind=node['type'];e['stage']='Executing '+str(node.get('name') or kind);store.put('workflow_executions',e)
     if kind in ('input','prepare'):result={'status':'complete','kind':'science','image_id':e['image']['id']}
     elif kind=='algorithm':
      generated=next((e['node_runs'][parent].get('box_geometry') for parent in incoming[identifier] if e['node_runs'][parent].get('kind')=='prompt'),None)
      runs=await backend.queue_operational(OperationalRequest(mission_id=e['mission_id'],job_id=e['job_id'],profile_ids=[node['profile_id']],image_ids=[e['image']['id']],max_size=p['max_size'],geometry=p['geometry'],points=node.get('points',[]),box_geometry=generated or node.get('box_geometry')),p['id'])
      result={'status':'running','runs':runs,'profile_id':node['profile_id']}
     elif kind=='class_prompt':result=await compute(class_prompt,e,identifier,incoming[identifier],node['class_value'])
     elif kind=='compare':result=await compute(compare,e,identifier,incoming[identifier])
     elif kind=='export':result=await compute(export,e,identifier,incoming[identifier],node['formats'])
     else:raise ValueError('Unsupported node')
     if saved and saved.get('attempts'):result['attempts']=saved['attempts']
     e['node_runs'][identifier]=result;progress=True;store.put('workflow_executions',e)
    done=sum(n.get('status')=='complete' for n in e['node_runs'].values());e['progress']=int(100*done/len(nodes))
    if done==len(nodes):e.update(status='complete',stage='Pipeline outputs ready');break
    store.put('workflow_executions',e);await asyncio.sleep(.5 if progress else 1)
  except asyncio.CancelledError:
   e.update(status='interrupted',stage='Service stopped; completed blocks retained. Unfinished native stages require restarting.')
   raise
  except Exception as exc:
   stopped=store.get('workflow_executions',e['id']).get('control')=='stop'
   e.update(status='cancelled' if stopped else 'failed',error=str(getattr(exc,'detail',exc)),stage='Stopped; completed artifacts retained' if stopped else 'Pipeline needs attention')
  finally:
   store.put('workflow_executions',e);store.event('pipeline.execution_waiting' if e['status'] in ('paused','blocked') else 'pipeline.execution_finished',e.get('error',e.get('stage','Execution ended')),level='error' if e['status']=='failed' else 'info',mission_id=e['mission_id'],pipeline_id=p['id'],run_id=e['id'],status=e['status'],audit=True);tasks.pop(e['id'],None)
 def artifacts(e,parents):
  files=[]
  for parent in parents:
   result=e['node_runs'][parent]
   if result.get('artifact'):files.append(result['artifact']);continue
   for run in result.get('runs',[]):
    if run.get('family')=='super_resolution':
     files.append({'path':str(store.root()/'segmentations'/run['id']/'enhanced.tif'),'bands':run['bands'],'run_id':run['id'],'kind':'enhanced','provenance':run['provenance']});continue
    if run.get('family')=='detection':
     files.append({'path':str(store.root()/'segmentations'/run['id']/'detections.geojson'),'classes':run['classes'],'run_id':run['id'],'kind':'detections'});continue
    if run.get('model')=='sam3_tracker':
     for c in run.get('candidates',[]):files.append({'path':str(store.root()/'segmentations'/run['id']/Path(c['mask']).name),'classes':['Proposed object'],'class_values':[1],'run_id':run['id'],'kind':'mask'})
    else:
     for i,output in enumerate(run.get('outputs',[])):files.append({'path':str(store.root()/'results'/run['id']/f'{i}-consensus.tif'),'classes':run['classes'],'class_values':list(range(1,len(run['classes'])+1)),'run_id':run['id'],'image_id':output['image_id'],'kind':'classification'})
  return files
 def execution_checkpoint(e,stage):
  import time
  while True:
   current=store.get('workflow_executions',e['id'])
   if current.get('control')=='stop':raise ValueError('Stopped at a safe processing checkpoint')
   mission_state=backend.mission_control.action(store.database,e['mission_id'])
   if mission_state in ('stop','delete'):raise ValueError('Mission stopped')
   if current.get('control')!='pause' and mission_state!='pause':return
   current.update(status='paused',stage='Paused before '+stage);store.put('workflow_executions',current);time.sleep(.25)
 def class_prompt(e,identifier,parents,value):
  files=artifacts(e,parents)
  if len(files)!=1:raise ValueError('Class prompt needs one class raster')
  box=None
  with rasterio.open(files[0]['path']) as src:
   for _,window in src.block_windows(1):
    execution_checkpoint(e,'class prompt generation')
    a=src.read(1,window=window,masked=True);ys,xs=np.where((a.data==value)&~np.ma.getmaskarray(a))
    if not len(xs):continue
    t=src.window_transform(window);points=[t*(int(xs.min()),int(ys.min())),t*(int(xs.max())+1,int(ys.max())+1)]
    current=[min(p[0] for p in points),min(p[1] for p in points),max(p[0] for p in points),max(p[1] for p in points)]
    box=current if box is None else [min(box[0],current[0]),min(box[1],current[1]),max(box[2],current[2]),max(box[3],current[3])]
   if box is None:raise ValueError('Selected class has no pixels; no SAM prompt was fabricated')
   from workflow_graph import polygon
   geometry=transform_geom(src.crs,'EPSG:4326',polygon(box))
  return {'status':'complete','kind':'prompt','box_geometry':geometry,'class_value':value,'method':'One enclosing box of the selected class; not a human annotation or semantic SAM vote'}
 def compare(e,identifier,parents):
  files=artifacts(e,parents)
  if len(files)!=2 or files[0]['classes']!=files[1]['classes']:raise ValueError('Comparison requires two compatible class rasters')
  folder=store.root()/'workflow-exports'/e['id'];folder.mkdir(parents=True,exist_ok=True);path=folder/(identifier+'-'+uuid4().hex[:8]+'-comparison.tif');temporary=path.with_suffix('.partial.tif')
  with rasterio.open(files[0]['path']) as a,rasterio.open(files[1]['path']) as b:
   profile=a.profile.copy();profile.update(dtype='uint8',count=1,nodata=255,compress='deflate')
   with WarpedVRT(b,crs=a.crs,transform=a.transform,width=a.width,height=a.height,resampling=Resampling.nearest) as aligned,rasterio.open(temporary,'w',**profile) as dst:
    for _,window in a.block_windows(1):
     execution_checkpoint(e,'classification comparison')
     x=a.read(1,window=window,masked=True);y=aligned.read(1,window=window,masked=True);valid=~np.ma.getmaskarray(x)&~np.ma.getmaskarray(y);dst.write(np.where(valid,x.data!=y.data,255).astype('uint8'),1,window=window)
  temporary.replace(path)
  return {'status':'complete','artifact':{'path':str(path),'classes':['Same class','Different class'],'class_values':[0,1],'kind':'binary','source_artifacts':files}}
 def export(e,identifier,parents,formats):
  from result_vectors import polygon_export
  from vector_exports import shapefile_zip
  source_files=artifacts(e,parents);folder=store.root()/'workflow-exports'/e['id'];folder.mkdir(parents=True,exist_ok=True);outputs=[]
  for i,source in enumerate(source_files):
   execution_checkpoint(e,'output export')
   src=Path(source['path']);stem=identifier+'-'+str(i)+'-'+uuid4().hex[:8];metadata={'mission_id':e['mission_id'],'job_id':e['job_id'],'pipeline_id':e['pipeline_id'],'pipeline_version':e['pipeline_version'],'execution_id':e['id'],'image_id':e['image']['id'],'source':source,'source_sha256':sha(src),'prediction_not_ground_truth':True}
   if source.get('kind')=='detections':
    if 'geotiff' in formats:raise ValueError('Object boxes require vector export, not semantic GeoTIFF')
    dst=folder/(stem+'.geojson');shutil.copyfile(src,dst)
    if 'geojson' in formats:outputs.append({'filename':dst.name,'sha256':sha(dst)})
    if 'shapefile' in formats:
     path=folder/(stem+'-shapefile.zip');shapefile_zip(dst,path);outputs.append({'filename':path.name,'sha256':sha(path)})
    meta=folder/(stem+'.json');meta.write_text(json.dumps(metadata,indent=2));outputs.append({'filename':meta.name,'sha256':sha(meta)});continue
   if source.get('kind')=='enhanced':
    if set(formats)!={'geotiff'}:raise ValueError('Super-resolution exports require GeoTIFF')
    if source['provenance']['output_sha256']!=metadata['source_sha256']:raise ValueError('Enhanced raster changed after the run')
   with rasterio.open(src) as r:metadata['grid']={'crs':str(r.crs),'transform':list(r.transform),'width':r.width,'height':r.height}
   if 'geotiff' in formats:
    dst=folder/(stem+'.tif');shutil.copyfile(src,dst)
    if sha(dst)!=metadata['source_sha256']:raise ValueError('Exported raster checksum mismatch')
    outputs.append({'filename':dst.name,'sha256':sha(dst)})
   if any(f in formats for f in ('geojson','shapefile')):
    dst=folder/(stem+'.geojson');polygon_export(src,dst,source['classes'],metadata,progress=lambda *args:execution_checkpoint(e,'polygon export'),class_values=source['class_values'])
    if 'geojson' in formats:outputs.append({'filename':dst.name,'sha256':sha(dst)})
    if 'shapefile' in formats:
     zip_path=folder/(stem+'-shapefile.zip');shapefile_zip(dst,zip_path);outputs.append({'filename':zip_path.name,'sha256':sha(zip_path)})
   meta=folder/(stem+'.json');meta.write_text(json.dumps(metadata,indent=2));outputs.append({'filename':meta.name,'sha256':sha(meta)})
  return {'status':'complete','files':[{**f,'download_url':f'/api/mission-workspace/exports/{e["id"]}/{f["filename"]}'} for f in outputs]}
 async def scheduler():
  from mac_web_gate import wait_for_commit
  await wait_for_commit()
  store.init()
  for e in store.all('workflow_executions'):
   if e['status']=='interrupted' and e.get('control','resume')=='resume':
    p=store.get('workflow_pipelines',e['pipeline_id'])
    if p['state']=='active':e.update(status='queued',stage='Recovering completed blocks; unfinished native stages restart as new attempts');store.put('workflow_executions',e)
  store.event('service.started','Mission workflow scheduler running; saved monitor settings preserved')
  try:
   while True:
    try:
     for p in store.all('workflow_pipelines'):
      if p['state']=='active' and p['id'] not in scans and (asyncio.get_running_loop().time()-last_poll.get(p['id'],0)>30):
       last_poll[p['id']]=asyncio.get_running_loop().time()
       async def poll(item):
        try:
         async with scan_slots:await tick(item)
        finally:scans.pop(item['id'],None)
       scans[p['id']]=asyncio.create_task(poll(p))
     # Source polling cannot block dispatch of ready local executions.
     for e in reversed(store.all('workflow_executions')):
      if len(tasks)>=4:break
      if e['status'] in ACTIVE and e['id'] not in tasks and (e.get('control','resume')=='stop' or e.get('control','resume')=='resume' and backend.mission_control.allowed(store.database,e['mission_id'])):
       if e['status']=='blocked' and any(store.model(i)['state']!='qualified' for i in e['definition'].get('model_releases',{})):continue
       tasks[e['id']]=asyncio.create_task(execute(e))
    except Exception as exc:store.event('scheduler.error',str(exc),level='error')
    await asyncio.sleep(2)
  except asyncio.CancelledError:
   pending=[*tasks.values(),*scans.values()]
   for t in pending:t.cancel()
   if pending:await asyncio.gather(*pending,return_exceptions=True)
   raise
 @app.on_event('startup')
 async def start():
  nonlocal loop_task
  loop_task=asyncio.create_task(scheduler())
 @app.on_event('shutdown')
 async def stop():
  if loop_task:loop_task.cancel();await asyncio.gather(loop_task,return_exceptions=True)
 @app.post('/api/mission-workspace/executions/{identifier}/{action}')
 async def execution_control(identifier:str,action:str):
  if action not in ('pause','resume','stop','delete','retry'):raise HTTPException(422,'Unknown execution control')
  e=store.get('workflow_executions',identifier)
  if action=='retry':
   if e['status'] not in ('failed','interrupted','cancelled'):raise HTTPException(409,'Only failed/interrupted/stopped executions can retry')
   for value in e['node_runs'].values():
    if value['status']!='complete':value.update(status='pending',attempts=[*value.get('attempts',[]),*value.pop('runs',[])])
   e.update(status='queued',control='resume',error=None)
  elif action=='resume':
   if e['status'] not in ('paused','blocked','pausing'):raise HTTPException(409,'Only paused/blocked work can resume')
   e.update(control='resume',status='queued')
  else:
   e['control']='pause' if action=='pause' else 'stop'
   if action=='delete':e['archived']=True
   if e['status'] not in ACTIVE and action!='delete':raise HTTPException(409,'This execution has already ended')
  store.put('workflow_executions',e,control_change=True)
  for node in e['node_runs'].values():
   for run in node.get('runs',[]):
    try:await backend.workflow_run_control(run['id'],'resume' if action=='resume' else 'pause' if action=='pause' else 'stop')
    except HTTPException:pass
  store.event('execution.'+action,'Pipeline execution '+action,mission_id=e['mission_id'],pipeline_id=e['pipeline_id'],run_id=e['id'],audit=True)
  return e
 @app.get('/api/mission-workspace/exports/{execution_id}/{filename}')
 async def download_export(execution_id:str,filename:str):
  from fastapi.responses import FileResponse
  e=store.get('workflow_executions',execution_id)
  filenames={f['filename'] for n in e['node_runs'].values() for f in n.get('files',[])}
  if filename not in filenames:raise HTTPException(404,'Export not found')
  p=(store.root()/'workflow-exports'/execution_id/filename).resolve();root=(store.root()/'workflow-exports').resolve()
  if not p.is_relative_to(root) or not p.is_file():raise HTTPException(404,'Export unavailable')
  return FileResponse(p,filename=p.name)
 backend.workflow_tick=tick;backend.workflow_discover=discover;backend.workflow_execute=execute
 return {'tick':tick,'execute':execute,'control':control}
