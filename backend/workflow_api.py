"""Application integration for qualified models and the three mission workspaces."""
import asyncio,json,sys
from pathlib import Path
from uuid import uuid4
from fastapi import HTTPException
from fastapi.responses import Response,JSONResponse
from pydantic import BaseModel,Field
from starlette.concurrency import run_in_threadpool
from resource_admission import run_reserved
from mission_workspace import Workspace,now,ACTIVE
from algorithm_registry import MODEL_DETAILS,SEGMENTATION_DETAILS
from object_detection import DETAILS as DETECTION_DETAILS
from dsen2 import DETAILS as SUPER_RESOLUTION_DETAILS
from detection_experiments import MODEL_ID as AIRCRAFT_ID, NAME as AIRCRAFT_NAME, RUN_KIND as AIRCRAFT_RUN
from portable_lulc import MODEL as PORTABLE, DETAILS as PORTABLE_DETAILS

class Goals(BaseModel):
 precision: float=Field(default=.8,ge=0,le=1)
 recall: float=Field(default=.8,ge=0,le=1)
 f1: float=Field(default=.8,ge=0,le=1)
 minimum_pixels: int=Field(default=50,ge=1)
 per_class: dict[str,dict[str,float]]=Field(default_factory=dict,max_length=100)
class ModelRequest(BaseModel):
 mission_id: str
 job_id: str
 name: str=Field(min_length=1,max_length=120)
 algorithm: str
 classes: list[str]=Field(default_factory=list,max_length=256)
 qualification_mode: str=Field(default='goals',pattern='^(goals|verified)$')
 goals: Goals=Field(default_factory=Goals)
 candidate_run: str|None=None
class ReleaseRequest(BaseModel):
 acknowledged: bool=False
 note: str=Field(default='',max_length=1000)
class CandidateRequest(BaseModel):
 run_id: str
class ChangeRequest(BaseModel):
 reason: str=Field(min_length=1,max_length=1000)
 classes: list[str]|None=None
class PromptContext(BaseModel):
 mission_id:str
 job_id:str
 image_id:str
 preview_id:str|None=None

class OperationalRequest(BaseModel):
 mission_id: str
 job_id: str
 profile_ids: list[str]=Field(min_length=1,max_length=12)
 image_ids: list[str]=Field(min_length=1,max_length=100)
 max_size: int=Field(default=384,ge=32,le=1024)
 geometry: dict|None=None
 points: list[dict]=Field(default_factory=list,max_length=32)
 box_geometry: dict|None=None
 prompt_context:PromptContext|None=None
class DetectionExperimentRequest(BaseModel):
 mission_id: str
 job_id: str
 image_id: str
 examples_per_class: int|None=Field(default=None,ge=3,le=10)
 candidate_run_id: str|None=Field(default=None,pattern=r'^[a-f0-9-]{36}$')

class DetectionAuditRequest(BaseModel):
 all_targets_labeled: bool=False

class SegmentationVerificationRequest(BaseModel):
 image_id: str=Field(min_length=1)
 max_size: int=Field(default=384,ge=32,le=1024)
 points: list[dict]=Field(default_factory=list,max_length=32)
 box_geometry: dict|None=None

class TrainRequest(BaseModel):
 max_size: int=Field(default=384,ge=32,le=1024)
 epochs: int=Field(default=20,ge=1,le=200)
 label_ids: list[str]=Field(default_factory=list)


def register(backend):
 app=backend.app;store=Workspace(backend._database,lambda:backend.DATA_DIR,backend._init_database)
 backend.workflow_store=store
 def describe(item):
  item={**item}
  if item['algorithm'] in DETECTION_DETAILS:
   from object_detection import component
   ready=component() is not None;item['availability']={'ready':ready,'reason':'' if ready else 'Install the Oriented R-CNN offline component (D1)'}
  elif item['algorithm'] in SUPER_RESOLUTION_DETAILS:
   from dsen2 import checkpoint
   ready=checkpoint(item['algorithm'],store.root()) is not None;item['availability']={'ready':ready,'reason':'' if ready else 'Install the local DSen2 weights and TensorFlow runtime'}
  elif item['algorithm'] in SEGMENTATION_DETAILS:
   from sam3_runtime import runtime_available,weights_available
   ready=runtime_available() and weights_available(store.root());item['availability']={'ready':ready,'reason':'' if ready else 'Install the SAM 3 offline component (S3)'}
  try:item={**item,'evaluation':store.evaluate(item)}
  except HTTPException as exc:item={**item,'evaluation':{'passed':False,'reasons':[str(exc.detail)]}}
  if item.get('candidate_run'):
   table='segmentation_runs' if item['algorithm'] in SEGMENTATION_DETAILS else 'inference_runs'
   with store.database() as db:r=db.execute(f'SELECT payload FROM {table} WHERE id=?',(item['candidate_run'],)).fetchone()
   if r:
    run=json.loads(r['payload']);item['activity']={k:run.get(k) for k in ('status','progress','stage','models','validation_protocol')}
  return item
 @app.get('/api/mission-workspace/model-catalog')
 async def model_catalog():
  from model_catalog import CATALOG
  return {'models':CATALOG,'notice':'Published results are author-reported evidence, not local validation or directly comparable rankings. Adapter/weight availability is shown independently.'}
 @app.get('/api/mission-workspace/models')
 async def models(mission_id: str):
  store.mission(mission_id);await run_in_threadpool(store.ensure_pretrained,mission_id)
  from detection_experiments import availability
  reason=availability(store.root())
  return {'detection_experiments':{'name':AIRCRAFT_NAME,'ready':not reason,'reason':reason},'models':[describe(m) for m in store.all('workflow_models',mission_id) if m['state']!='deleted'],'algorithms':{**MODEL_DETAILS,**SEGMENTATION_DETAILS,**DETECTION_DETAILS,**SUPER_RESOLUTION_DETAILS,PORTABLE:PORTABLE_DETAILS}}
 @app.get('/api/mission-workspace/candidates')
 async def candidates(mission_id:str,job_id:str,algorithm:str):
  store.mission(mission_id,job_id);store.init();items=[]
  table='segmentation_runs' if algorithm in SEGMENTATION_DETAILS else 'inference_runs'
  with store.database() as db:rows=db.execute(f'SELECT payload FROM {table} WHERE mission_id=? AND job_id=? ORDER BY created_at DESC',(mission_id,job_id)).fetchall() if table=='inference_runs' else db.execute('SELECT payload FROM segmentation_runs ORDER BY created_at DESC').fetchall()
  for row in rows:
   run=json.loads(row['payload'])
   if run.get('mission_id')!=mission_id or run.get('job_id')!=job_id:continue
   if run.get('status') not in ('complete','partial'):continue
   if table=='segmentation_runs':valid=run.get('model')==algorithm
   else:valid=any(m['id']==algorithm and m.get('status')=='complete' for m in run.get('models',[]))
   if valid:items.append({'id':run['id'],'created_at':run.get('created_at'),'classes':run.get('classes',[]),'metrics':next((m.get('metrics') for m in run.get('models',[]) if m['id']==algorithm),None)})
  return {'candidates':items}
 @app.post('/api/mission-workspace/models/{identifier}/verify')
 async def verify(identifier:str,request:TrainRequest):
  from pretrained_lulc import PRETRAINED
  item=store.model(identifier)
  if describe(item).get('activity',{}).get('status') in ACTIVE:raise HTTPException(409,'Verification or training is already active for this model')
  if item['state']=='qualified':raise HTTPException(409,'Model already qualified')
  portable=item['algorithm']==PORTABLE
  if not portable and item['algorithm'] not in PRETRAINED:raise HTTPException(422,'Train this classifier from labels, or select an existing completed candidate. SAM is verified from an actual prompted run.')
  _,job=store.mission(item['mission_id'],item['job_id'])
  metadata={'run_kind':'verification','workflow_verification_model':identifier}
  if portable:
   metadata.update(run_kind='portable_validation',portable_model=item['portable_model'],qualified_classes=item['classes'])
  run=await backend._queue_inference_run(backend.InferenceRequest(mission_id=item['mission_id'],job_id=item['job_id'],algorithms=[item['algorithm']],image_ids=[i['id'] for i in job['images']],max_size=request.max_size,geometry=job.get('aoi'),band_alignment=False),metadata)
  item.update(candidate_run=run['id'],state='training');store.put('workflow_models',item);store.event('model.verification_started','Running fixed model weights without local fitting',mission_id=item['mission_id'],model_id=identifier,run_id=run['id']);return run
 @app.post('/api/mission-workspace/models')
 async def create_model(request: ModelRequest):
  store.mission(request.mission_id,request.job_id)
  if request.algorithm not in {**MODEL_DETAILS,**SEGMENTATION_DETAILS}:raise HTTPException(422,'No supported worker for this algorithm')
  if any(not c.strip() or len(c)>80 for c in request.classes) or len(set(request.classes))!=len(request.classes):raise HTTPException(422,'Class names must be nonempty and unique')
  for name,targets in request.goals.per_class.items():
   if name not in request.classes or any(k not in ('precision','recall','f1') or not 0<=v<=1 for k,v in targets.items()):raise HTTPException(422,'Per-class goals must name a configured class and valid 0–1 precision/recall/F1 values')
  from pretrained_lulc import PRETRAINED
  if request.algorithm not in PRETRAINED and request.algorithm not in SEGMENTATION_DETAILS and len(request.classes)<2:raise HTTPException(422,'Define at least two LULC classes before creating this training configuration')
  if request.algorithm in SEGMENTATION_DETAILS and request.qualification_mode!='verified':raise HTTPException(422,'SAM currently supports operator verification of real proposals. Mask-accuracy qualification needs a supervised validation adapter; estimated quality is not accuracy.')
  if request.algorithm in PRETRAINED:
   native=PRETRAINED[request.algorithm]['classes']
   if request.classes and request.classes!=native:raise HTTPException(422,'This pretrained head has fixed native classes. Use a locally trainable classifier for a different taxonomy.')
   request.classes=native
  item={'id':str(uuid4()),**request.model_dump(),'state':'training','task':'segmentation' if request.algorithm in SEGMENTATION_DETAILS else 'lulc','created_at':now(),'history':[]}
  if item.get('candidate_run'):
   run,_=store.candidate(item,item['candidate_run']);item['classes']=run.get('classes',item['classes'])
  store.put('workflow_models',item);store.event('model.created','Configured model added to Training & qualification',mission_id=item['mission_id'],model_id=item['id'],audit=True,algorithm=item['algorithm'])
  return describe(item)
 @app.post('/api/mission-workspace/models/{identifier}/fine-tune-copy')
 async def fine_tune_copy(identifier:str,request:dict):
  source=store.model(identifier)
  if source['algorithm']==PORTABLE or source['algorithm'] in {**SEGMENTATION_DETAILS,**DETECTION_DETAILS,**SUPER_RESOLUTION_DETAILS}:raise HTTPException(422,'This adapter does not support supervised fine-tuning yet')
  job_id=request.get('job_id') or source['job_id'];store.mission(source['mission_id'],job_id)
  item={**source,'id':str(uuid4()),'job_id':job_id,'name':source['name']+' · local fine-tuning','builtin':False,'state':'training','qualification_mode':'goals','goals':Goals().model_dump(),'created_at':now(),'history':[]}
  item.pop('release',None);item.pop('candidate_run',None);store.put('workflow_models',item);store.event('model.fine_tune_copy','Separate training candidate created; original pretrained weights remain available',mission_id=item['mission_id'],model_id=item['id'],audit=True);return describe(item)
 @app.post('/api/mission-workspace/models/{identifier}/verify-segmentation')
 async def verify_segmentation(identifier:str,request:SegmentationVerificationRequest):
  item=store.model(identifier)
  if item['algorithm'] not in SEGMENTATION_DETAILS:raise HTTPException(422,'Choose a segmentation configuration')
  if item['state']=='qualified':raise HTTPException(409,'Model already qualified')
  if describe(item).get('activity',{}).get('status') in ACTIVE:raise HTTPException(409,'A verification run is already active')
  operation=OperationalRequest(mission_id=item['mission_id'],job_id=item['job_id'],profile_ids=[identifier],image_ids=[request.image_id],max_size=request.max_size,points=request.points,box_geometry=request.box_geometry)
  task=await queue_segment(operation,item,{'run_kind':'verification','workflow_verification_model':identifier})
  item.update(candidate_run=task['id'],state='training');store.put('workflow_models',item);store.event('model.verification_started','SAM verification queued from source-image prompts; not an operational release',mission_id=item['mission_id'],model_id=identifier,run_id=task['id']);return task
 @app.post('/api/mission-workspace/models/{identifier}/candidate')
 async def candidate(identifier: str,request: CandidateRequest):
  item=store.model(identifier)
  if item['state']=='qualified':raise HTTPException(409,'Return this model to training before changing its candidate')
  if item['algorithm']==PORTABLE:raise HTTPException(422,'Run independent validation for this exact portable model version')
  run,_=store.candidate(item,request.run_id);item.update(candidate_run=request.run_id,classes=run.get('classes',item['classes']))
  store.put('workflow_models',item);store.event('model.candidate','Completed candidate selected',mission_id=item['mission_id'],model_id=identifier,run_id=request.run_id,audit=True)
  return describe(item)
 @app.post('/api/mission-workspace/models/{identifier}/release')
 async def release(identifier: str,request: ReleaseRequest):return await run_in_threadpool(store.qualify,identifier,request.acknowledged,request.note)
 @app.post('/api/mission-workspace/models/{identifier}/return-to-training')
 async def retract(identifier: str,request: ChangeRequest):
  from pretrained_lulc import PRETRAINED
  before=store.model(identifier)
  if before.get('builtin'):raise HTTPException(422,'Create a fine-tuning copy; the original pretrained model stays available for inference')
  if before['algorithm']==PORTABLE and request.classes is not None:raise HTTPException(422,'Portable model classes are fixed. Stage a new reviewed package for different classes.')
  if request.classes is not None and (len(set(request.classes))!=len(request.classes) or any(not c.strip() or len(c)>80 for c in request.classes)):raise HTTPException(422,'Classes must be unique and nonempty')
  if before['algorithm'] in PRETRAINED and request.classes is not None and request.classes!=PRETRAINED[before['algorithm']]['classes']:raise HTTPException(422,'This released head has fixed classes. Create a local student classifier with the new taxonomy and reviewed starting labels instead.')
  item=store.retract(identifier,request.reason)
  if request.classes is not None:
   item.update(classes=request.classes,candidate_run=None)
   item['goals']['per_class']={k:v for k,v in item['goals'].get('per_class',{}).items() if k in request.classes}
   store.put('workflow_models',item)
  return item
 @app.delete('/api/mission-workspace/models/{identifier}')
 async def delete_model(identifier: str):
  item=store.model(identifier);dependencies=[p['name'] for p in store.all('workflow_pipelines',item['mission_id']) if p.get('state')!='deleted' and any(n.get('profile_id')==identifier for n in p.get('nodes',[]))]
  if dependencies:raise HTTPException(409,'Remove this model from these pipelines before deleting it: '+', '.join(dependencies))
  item=store.retract(identifier,'Removed by operator');item['state']='deleted';store.put('workflow_models',item);return {'archived':True,'data_preserved':True}
 @app.get('/api/mission-workspace/models/{identifier}/sampling')
 async def sampling(identifier:str,max_size:int=384):
  if not 32<=max_size<=1024:raise HTTPException(422,'Use a sampling grid from 32 to 1024 pixels')
  from training_sampling import measure
  return await run_in_threadpool(measure,backend,store.model(identifier),max_size)
 @app.post('/api/mission-workspace/models/{identifier}/train')
 async def train(identifier: str,request: TrainRequest):
  item=store.model(identifier)
  if describe(item).get('activity',{}).get('status') in ACTIVE:raise HTTPException(409,'A candidate is already active for this configuration. Pause or stop it before starting another.')
  if item['algorithm'] in SEGMENTATION_DETAILS:raise HTTPException(422,'SAM mask training is unsupported. Verify an actual segmentation run instead.')
  if item['algorithm']==PORTABLE:raise HTTPException(422,'Portable weights are fixed. Run independent validation instead.')
  if item['state']=='qualified':raise HTTPException(409,'Return this model to training first')
  with store.database() as db:
   refs=[json.loads(r['payload']) for r in db.execute('SELECT payload FROM ground_truth WHERE mission_id=? AND job_id=?',(item['mission_id'],item['job_id']))]
  from workflow_inputs import training_references
  refs=training_references(refs,item['classes'],request.label_ids or None)
  actual=sorted({r['label'] for r in refs})
  if item.get('classes') and set(actual)!=set(item['classes']):raise HTTPException(422,'Training references must cover the configured classes. Add missing classes or explicitly return the model with a revised class schema.')
  run=await backend._create_training_version(backend.FineTuneRequest(mission_id=item['mission_id'],job_id=item['job_id'],model=item['algorithm'],max_size=request.max_size,epochs=request.epochs,label_ids=[r['id'] for r in refs]),workflow_profile_id=identifier)
  item.update(state='training',candidate_run=run['id']);store.put('workflow_models',item);store.event('training.started','Training a separate candidate version',mission_id=item['mission_id'],model_id=identifier,run_id=run['id'],audit=True)
  return run

 async def queue_operational(request,pipeline_id=None):
  _,job=store.mission(request.mission_id,request.job_id)
  if len(set(request.image_ids))!=len(request.image_ids) or any(i not in {x['id'] for x in job['images']} for i in request.image_ids):raise HTTPException(422,'Choose distinct source images from this job')
  if request.prompt_context and (request.prompt_context.mission_id!=request.mission_id or request.prompt_context.job_id!=request.job_id or request.image_ids!=[request.prompt_context.image_id]):raise HTTPException(422,'SAM prompts belong to another source image. Draw them again on the selected image.')
  await run_in_threadpool(store.guarded,request.profile_ids)
  profiles=[store.model(i,True) for i in request.profile_ids]
  if len(set(request.profile_ids))!=len(request.profile_ids):raise HTTPException(422,'Choose each model once')
  if any(p['algorithm'] in SEGMENTATION_DETAILS for p in profiles) and (len(request.image_ids)!=1 or not request.points and not request.box_geometry):raise HTTPException(422,'SAM requires one source image and drawn points or a box before any selected models are queued')
  if any(p['mission_id']!=request.mission_id for p in profiles):raise HTTPException(422,'Choose configured models in this mission')
  if any(p['algorithm'] in DETECTION_DETAILS for p in profiles):
   from object_detection import component
   if component() is None:raise HTTPException(409,'Install the separate offline Oriented R-CNN component before queuing any selected models')
  from workflow_inputs import preflight
  def model_geometry(profile):
   return request.geometry or (job.get('aoi') if profile['algorithm'] in SUPER_RESOLUTION_DETAILS or profile['algorithm']=='prithvi_crop' else None)
  for image_id in request.image_ids:
   image=next(i for i in job['images'] if i['id']==image_id);_,record=backend._reference_scope(request.mission_id,request.job_id,image_id)
   for profile in profiles:
    geometry=model_geometry(profile)
    try:await run_in_threadpool(preflight,profile,record,image,geometry,store.root())
    except (ValueError,OSError) as exc:raise HTTPException(422,str(exc)) from exc
  output=[]
  for profile in profiles:
   release=profile['release'];metadata={'run_kind':'operational_inference','workflow_model_ids':[profile['id']],'qualified_releases':{profile['id']:release['id']},'qualified_classes':profile['classes'],'qualified_checkpoints':{profile['algorithm']:release.get('checkpoint')},'pipeline_id':pipeline_id,'qualification':{profile['id']:release}}
   if profile['algorithm']==PORTABLE:metadata['portable_model']=profile['portable_model']
   try:
    if profile['algorithm'] in {**DETECTION_DETAILS,**SUPER_RESOLUTION_DETAILS}:
     for image_id in request.image_ids:
      geometry=model_geometry(profile)
      single=request.model_copy(update={'image_ids':[image_id],'geometry':geometry});run=await queue_segment(single,profile,metadata);output.append(run)
      store.event('inference.queued','Released model queued without local fitting',mission_id=request.mission_id,model_id=profile['id'],run_id=run['id'],pipeline_id=pipeline_id,audit=True,release_id=release['id'])
     continue
    if profile['algorithm'] in SEGMENTATION_DETAILS:run=await queue_segment(request,profile,metadata)
    else:run=await backend._queue_inference_run(backend.InferenceRequest(mission_id=request.mission_id,job_id=request.job_id,image_ids=request.image_ids,algorithms=[profile['algorithm']],max_size=request.max_size,geometry=model_geometry(profile),band_alignment=False,combination='separate'),metadata)
    output.append(run);store.event('inference.queued','Released model queued without local fitting',mission_id=request.mission_id,model_id=profile['id'],run_id=run['id'],pipeline_id=pipeline_id,audit=True,release_id=release['id'])
   except Exception as exc:
    if not output:raise
    # Preserve visibility of already queued work if a later enqueue fails.
    failed={'id':str(uuid4()),'mission_id':request.mission_id,'job_id':request.job_id,'created_at':now(),'status':'failed','progress':0,'stage':'Could not queue this model; earlier requested runs remain visible','queue_error':str(getattr(exc,'detail',exc)),'error':str(getattr(exc,'detail',exc)),'requested_models':[profile['algorithm']],'models':[{'id':profile['algorithm'],'status':'failed'}],'outputs':[],**metadata}
    with store.database() as db:db.execute('INSERT INTO inference_runs VALUES(?,?,?,?,?)',(failed['id'],request.mission_id,request.job_id,json.dumps(failed),failed['created_at']))
    output.append(failed);store.event('inference.queue_failed',failed['error'],level='error',mission_id=request.mission_id,model_id=profile['id'],run_id=failed['id'],audit=True)
  return output
 async def queue_segment(request,profile,metadata,support=None,candidate=None):
  backend._require_active_mission(request.mission_id)
  import numpy as np,rasterio
  from segmentation import execute
  research=profile['algorithm']==AIRCRAFT_ID
  is_detection=research or profile['algorithm'] in DETECTION_DETAILS
  is_enhancement=profile['algorithm'] in SUPER_RESOLUTION_DETAILS
  if is_enhancement:
   from dsen2 import execute
  if research:
   from object_detection import execute
  elif is_detection:
   from object_detection import component
   from native_gpu_bridge import execute
   if component() is None:raise HTTPException(409,'Install the separate offline Oriented R-CNN component before running this model')
  if len(request.image_ids)!=1:raise HTTPException(422,'This adapter uses one image per run')
  if not (is_detection or is_enhancement) and not request.points and not request.box_geometry:raise HTTPException(422,'SAM requires include points or a box. Draw prompts on the map or configure them in the pipeline.')
  _,job=store.mission(request.mission_id,request.job_id);image=next((i for i in job['images'] if i['id']==request.image_ids[0]),None)
  if not image:raise HTTPException(404,'Source image not found')
  _,record=backend._reference_scope(request.mission_id,request.job_id,image['id']);identifier=str(uuid4());folder=store.root()/'segmentations'/identifier;folder.mkdir(parents=True)
  def grid():
   stack=backend.read_stack(record,image,max_size=request.max_size,geometry=request.geometry);p=folder/'reference-grid.tif'
   with rasterio.open(p,'w',driver='GTiff',**{**stack.profile,'count':1,'dtype':'uint8','nodata':0}) as dst:dst.write(stack.valid.astype('uint8'),1)
   return p
  reference=None if is_enhancement or is_detection else await run_in_threadpool(run_reserved,'Segmentation reference grid',2,grid)
  task={'id':identifier,'model':profile['algorithm'],'mission_id':request.mission_id,'job_id':request.job_id,'image_id':image['id'],'run_id':identifier,'image_index':0,'created_at':now(),'status':'queued','progress':0,'points':request.points,'box_geometry':request.box_geometry,'rgb_bands':['RED','GREEN','BLUE'],'family':'super_resolution' if is_enhancement else 'detection' if is_detection else 'segmentation','geometry':request.geometry,'max_size':request.max_size,**metadata}
  with store.database() as db:db.execute('INSERT INTO segmentation_runs VALUES(?,?,?,?,?)',(identifier,identifier,0,json.dumps(task),task['created_at']))
  worker_options={'support':support,'candidate':candidate} if support or candidate else {}
  backend.model_executor.submit(backend.mission_control.execute,store.database,'segmentation_runs',dict(task),execute,
                                backend._persist_segmentation,store.root(),reference,image,record,**worker_options)
  return task
 backend.queue_operational=queue_operational
 @app.post('/api/mission-workspace/source-prompts/{preview_id}')
 async def source_prompts(preview_id:str,request:dict):
  from reference_labels import preview_path,label_geometry,fingerprint
  try:
   path=preview_path(store.root(),preview_id)/'grid.json'
   if not path.is_file():raise ValueError('Open a source preview first')
   preview=json.loads(path.read_text());_,record=backend._reference_scope(preview['mission_id'],preview['job_id'],preview['image_id'])
   if fingerprint(record)!=preview['access_sha256']:raise ValueError('Source changed; reopen its preview before drawing prompts')
   points=request.get('points',[])
   if not isinstance(points,list) or len(points)>32:raise ValueError('Use a list of at most 32 points')
   output=[]
   for point in points:
    if not isinstance(point,dict) or point.get('label') not in (0,1):raise ValueError('Point labels must be include (1) or exclude (0)')
    geometry=label_geometry(preview,[point['xy']],'point');output.append({'longitude':geometry['coordinates'][0],'latitude':geometry['coordinates'][1],'label':point['label']})
   box=label_geometry(preview,request['box'],'rectangle') if request.get('box') else None
   return {'points':output,'box_geometry':box,'image_id':preview['image_id'],'prompt_context':{k:preview[k] for k in ('mission_id','job_id','image_id')}|{'preview_id':preview_id}}
  except (ValueError,KeyError,TypeError) as exc:raise HTTPException(422,str(exc)) from exc
 @app.get('/api/mission-workspace/operational-runs')
 async def operational_runs(mission_id:str):
  store.mission(mission_id);store.init();items=[]
  with store.database() as db:
   for table,family in [('inference_runs','lulc'),('segmentation_runs','segmentation')]:
    rows=db.execute(f'SELECT payload FROM {table} ORDER BY created_at DESC').fetchall()
    for row in rows:
     run=json.loads(row['payload'])
     if run.get('mission_id')==mission_id and run.get('run_kind') in ('operational_inference',AIRCRAFT_RUN) and not run.get('archived'):items.append({**run,'family':run.get('family',family)})
  return {'runs':sorted(items,key=lambda r:r['created_at'],reverse=True)[:200]}
 @app.post('/api/mission-workspace/segments/{task_id}/{candidate_id}/approve')
 async def approve_segment(task_id:str,candidate_id:int,request:backend.ManualReviewRequest):
  task=await backend.get_segmentation(task_id)
  if task.get('status')!='complete':raise HTTPException(409,'Segmentation has not completed')
  candidate=next((c for c in task.get('candidates',[]) if c['id']==candidate_id),None)
  if not candidate:raise HTTPException(404,'Mask proposal not found')
  source={'type':'human_approved_segmentation','task_id':task_id,'source_run':task.get('run_id'),'model':task['model'],'provenance':task.get('provenance'),'prompts':task.get('points'),'box_geometry':task.get('box_geometry'),'candidate':candidate_id,'predicted_mask_quality':candidate.get('predicted_mask_quality'),'human_decision':True}
  return await backend._save_ground_truth(backend.GroundTruthRequest(mission_id=task['mission_id'],job_id=task['job_id'],image_id=task['image_id'],label=request.label,geometry=candidate['geometry'],role=request.role,confidence=request.confidence,notes=request.notes),source=source)
 def object_references(mission_id,job_id):
  with store.database() as db:
   db.execute('CREATE TABLE IF NOT EXISTS object_labels(id TEXT PRIMARY KEY,mission_id TEXT,job_id TEXT,image_id TEXT,payload TEXT,created_at TEXT)')
   rows=db.execute('SELECT payload FROM object_labels WHERE mission_id=? AND job_id=? ORDER BY created_at,id LIMIT 2001',(mission_id,job_id)).fetchall()
  if len(rows)>2000:raise HTTPException(422,'Use a smaller, dedicated job for this object experiment.')
  return [json.loads(row['payload']) for row in rows]

 @app.post('/api/mission-workspace/detection-experiments')
 async def detection_experiment(request:DetectionExperimentRequest):
  from detection_experiments import availability
  from reference_labels import fingerprint
  from workflow_inputs import preflight
  store.mission(request.mission_id,request.job_id)
  reason=availability(store.root())
  if reason:raise HTTPException(409,reason)
  image,record=backend._reference_scope(request.mission_id,request.job_id,request.image_id)
  if record.get('mode') not in ('local_path','download'):
   raise HTTPException(422,'Connect an offline local image or a previously downloaded copy.')
  profile={'algorithm':AIRCRAFT_ID,'name':AIRCRAFT_NAME,'task':'detection'}
  try:await run_in_threadpool(preflight,profile,record,image,None,store.root())
  except (ValueError,OSError) as exc:raise HTTPException(422,str(exc)) from exc
  operation=OperationalRequest(mission_id=request.mission_id,job_id=request.job_id,
                               image_ids=[request.image_id],profile_ids=[AIRCRAFT_ID])
  support=candidate=None
  if request.candidate_run_id:
   if request.examples_per_class is not None:raise HTTPException(422,'Choose a saved candidate or build a new one.')
   with store.database() as db:
    row=db.execute('SELECT payload FROM segmentation_runs WHERE id=?',(request.candidate_run_id,)).fetchone()
   existing=json.loads(row['payload']) if row else {}
   if (existing.get('run_kind')!=AIRCRAFT_RUN or existing.get('mission_id')!=request.mission_id
       or existing.get('status')!='complete' or not existing.get('candidate_fit') or existing.get('candidate_run_id')
       or existing.get('archived')):
    raise HTTPException(409,'Select a completed training experiment from this mission.')
   if request.image_id==existing['image_id'] and request.job_id==existing['job_id']:
    raise HTTPException(422,'Select a different evaluation image for the saved candidate.')
   classes={r['label'] for r in object_references(request.mission_id,request.job_id)
            if r['image_id']==request.image_id and r.get('role')=='validation' and r.get('confidence',0)>=3
            and r.get('source',{}).get('access_sha256')==fingerprint(record)}
   if set(existing['classes'])-classes:raise HTTPException(422,'Add independent validation boxes for every candidate class on this image.')
   bank=store.root()/'segmentations'/request.candidate_run_id/'candidate-bank.json'
   candidate=(bank,existing['provenance']['fewshot']['exemplar_sha256'])
  elif request.examples_per_class is not None:
   from detection_experiments import select_support
   refs=object_references(request.mission_id,request.job_id)
   try:chosen=select_support(refs,request.image_id,fingerprint(record),request.examples_per_class)
   except ValueError as exc:raise HTTPException(422,str(exc)) from exc
   connected={}
   for ref in chosen:
    if ref['image_id'] not in connected:
     connected[ref['image_id']]=backend._reference_scope(request.mission_id,request.job_id,ref['image_id'])
    support_image,access=connected[ref['image_id']]
    if access.get('mode') not in ('local_path','download') or fingerprint(access)!=ref['source']['access_sha256']:
     raise HTTPException(422,'Reconnect and label the exact local training snapshot before running.')
   support=[(ref,*connected[ref['image_id']]) for ref in chosen]
  task=await queue_segment(operation,profile,{'run_kind':AIRCRAFT_RUN,'qualification_status':'research only',
                                              'source_access_sha256':fingerprint(record),'candidate_fit':bool(support or candidate),
                                              'candidate_run_id':request.candidate_run_id,
                                              'model_name':'Rotated object improvement experiment' if support or candidate else AIRCRAFT_NAME},support,candidate)
  store.event('detection.experiment_queued','Queued a local rotated detection experiment',
              mission_id=request.mission_id,run_id=task['id'],audit=True)
  return task
 @app.post('/api/mission-workspace/detection-experiments/{run_id}/audit')
 async def detection_audit(run_id:str,request:DetectionAuditRequest):
  from detection_experiments import audit
  store.init()
  with store.database() as db:
   row=db.execute('SELECT payload FROM segmentation_runs WHERE id=?',(run_id,)).fetchone()
   task=json.loads(row['payload']) if row else {}
   if task.get('run_kind')!=AIRCRAFT_RUN or task.get('archived'):raise HTTPException(404,'Detection experiment not found')
   if task['status']!='complete':raise HTTPException(409,'Wait for the detection experiment to complete')
  labels=[ref for ref in object_references(task['mission_id'],task['job_id']) if ref['image_id']==task['image_id']]
  store.mission(task['mission_id'],task['job_id'])
  try:result=await run_in_threadpool(audit,task,labels,request.all_targets_labeled)
  except (ValueError,OSError) as exc:raise HTTPException(422,str(exc)) from exc
  name=f'aircraft-audit-{uuid4().hex}.json'
  result.update(run_id=run_id,created_at=now(),report=f'/api/segmentations/{run_id}/{name}')
  with (store.root()/'segmentations'/run_id/name).open('x') as handle:json.dump(result,handle,allow_nan=False)
  store.event('detection.audit_saved','Independent object-box review saved',mission_id=task['mission_id'],run_id=run_id,audit=True,report=name)
  return result
 @app.post('/api/mission-workspace/infer')
 async def infer(request: OperationalRequest):return {'runs':await queue_operational(request)}
 @app.get('/api/mission-workspace/logs')
 async def logs(mission_id: str|None=None,level: str|None=None,after:int=0,pipeline_id:str|None=None,model_id:str|None=None,run_id:str|None=None,since:str|None=None):
  return {'events':store.events(mission_id,level,after,pipeline_id=pipeline_id,model_id=model_id,run_id=run_id,since=since),'directory':str(store.root()/'logs'),'retention':'Operational events: 30 days. Rotating JSONL: 2 MiB × 6 files. Qualification/control audit records retained in SQLite.'}
 @app.get('/api/mission-workspace/logs/download')
 async def download(mission_id: str|None=None,level: str|None=None,format:str='text',pipeline_id:str|None=None,model_id:str|None=None,run_id:str|None=None,since:str|None=None):
  events=store.events(mission_id,level,limit=10000,pipeline_id=pipeline_id,model_id=model_id,run_id=run_id,since=since)
  text='\n'.join(json.dumps(e,ensure_ascii=False) if format=='jsonl' else f'{e["created_at"]} [{e["level"].upper()}] {e["code"]} · {e["message"]} · {json.dumps(e["detail"],ensure_ascii=False)}' for e in reversed(events))
  return Response(text,media_type='application/x-ndjson' if format=='jsonl' else 'text/plain',headers={'Content-Disposition':f'attachment; filename="SpectraScope-activity.{"jsonl" if format=="jsonl" else "log"}"'})
 @app.post('/api/mission-workspace/runs/{identifier}/{action}')
 async def run_control(identifier: str,action: str):
  if action not in ('pause','resume','stop','delete'):raise HTTPException(422,'Unknown control')
  store.init();found=None
  with store.database() as db:
   db.execute('BEGIN IMMEDIATE')
   for table in ('inference_runs','segmentation_runs','anomaly_runs'):
    row=db.execute(f'SELECT payload FROM {table} WHERE id=?',(identifier,)).fetchone()
    if not row:continue
    run=json.loads(row['payload']);found=run
    if action=='resume' and run.get('status') not in ('paused','pausing','blocked'):raise HTTPException(409,'Only paused work can resume. Stopped/interrupted work needs a new run.')
    if action=='delete' and run.get('status') not in ACTIVE:run['archived']=True
    elif action in ('stop','delete') and run.get('status')=='queued':
     run.update(status='cancelled',control_stop=True,control_action='stop',stage='Stopped before native execution; data preserved')
     if action=='delete':run['archived']=True
    else:
     run['control_action']='resume' if action=='resume' else 'pause' if action=='pause' else 'stop'
     if action=='delete':run['archive_after_stop']=True
     run.update(status='pausing' if action=='pause' else 'stopping' if action in ('stop','delete') else 'queued',stage='Control requested; waiting for a safe processing checkpoint')
    db.execute(f'UPDATE {table} SET payload=? WHERE id=?',(json.dumps(run),identifier));break
  if not found:raise HTTPException(404,'Run not found')
  store.event('run.'+action,'Run control requested',mission_id=found['mission_id'],run_id=identifier,audit=True)
  return found
 backend.workflow_run_control=run_control
 @app.exception_handler(HTTPException)
 async def explain_http_error(request,exc):
  context={'path':request.url.path,'status':exc.status_code}
  parts=request.url.path.split('/')
  if 'models' in parts:
   try:
    identifier=parts[parts.index('models')+1];model=store.model(identifier);context.update(model_id=identifier,mission_id=model['mission_id'])
   except (HTTPException,IndexError):pass
  store.event('request.rejected',str(exc.detail),level='warning',**context)
  return JSONResponse({'detail':exc.detail},status_code=exc.status_code,headers=exc.headers)
 @app.middleware('http')
 async def log_request_error(request,call_next):
  try:response=await call_next(request)
  except Exception as exc:
   store.event('request.failed',str(exc),level='error',path=request.url.path);raise
  if response.status_code>=500:
   store.event('request.server_error','Server request failed; inspect the associated run and error details',level='error',path=request.url.path,status=response.status_code)
  return response
 return store
