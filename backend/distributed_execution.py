"""Real job dispatch through immutable bundles and enrolled internal workers.

One task executes one registered classifier run. Supplier rasters remain read-only;
workers receive private verified copies, never database credentials or code.
"""
import asyncio,hashlib,importlib.metadata,json,os,shutil,tempfile,time
import threading
from pathlib import Path,PurePosixPath
from uuid import uuid4
from agent_services import file_hash
from distributed_queue import Queue,ACTIVE,public,re_hash

SUPPORTED={'random_forest','svm','xgboost','unet','dynamicworld'}
class CoordinatorRestart(BaseException):pass
_MERGE_SLOT=threading.Semaphore(1)
CODE=('job_runner.py','model_workers.py','native_lulc.py','raster_pipeline.py','pretrained_lulc.py','sensor_metadata.py','execution_policy.py','runtime_profiles.py','resource_admission.py','native_network_boundary.py','algorithm_registry.py','model_packs.py','offline_policy.py')

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def signature(adapter):
 root=Path(__file__).resolve().parent;packages=['numpy','scipy','rasterio','scikit-learn','joblib']+(['xgboost'] if adapter=='xgboost' else ['torch'] if adapter=='unet' else ['tensorflow'] if adapter=='dynamicworld' else [])
 code=CODE+('neural_training_sync.py','unet_pool.py') if adapter=='unet' else CODE
 if adapter=='unet':packages+=['safetensors']
 versions={name:importlib.metadata.version(name) for name in packages}
 # CPU/CUDA builds share the pinned PyTorch release and numeric contract.
 if adapter=='unet':versions['torch']=versions['torch'].split('+',1)[0]
 return {'code':{name:file_hash(root/name) for name in code},'packages':versions}
def root(data):return Path(data)/'pool'
def setting(database):
 with database() as db:row=db.execute("SELECT payload FROM settings WHERE key='worker_pool'").fetchone()
 return json.loads(row['payload']) if row else {'enabled':False,'forest_shards':1,'wait_seconds':120,'maximum_seconds':3600}
def suitable(database,adapter):
 return [w for w in Queue(database).snapshot()['workers'] if w['state']=='enabled' and adapter in w['capabilities'] and w.get('last_seen') and time.time()-w['last_seen']<90]
def route(database,run,access):
 config=setting(database)
 if not config.get('enabled'):return False
 if len(run['requested_models'])!=1 or run['requested_models'][0] not in SUPPORTED:
  run['execution_route']='local · this adapter combination has no distributed worker contract';return False
 if any(a.get('mode') not in ('local_path','download') for a in access.values()):raise ValueError('Worker execution needs downloaded/local science rasters. Download streaming assets first; no source URL is sent to a worker.')
 if run.get('run_kind')=='training' and run['requested_models']==['unet'] and config.get('unet_workers',1)>1:
  from unet_pool import select_workers
  run['unet_workers']=select_workers(suitable(database,'unet'),config['unet_workers'])
 if not suitable(database,run['requested_models'][0]):raise ValueError('No recently connected compatible worker. Start an enrolled worker or turn off worker-pool routing; no hidden local duplicate will be started.')
 run['execution_route']='enrolled worker pool';run['worker_pool_configuration']=dict(config);return True

def replace_paths(value,bindings):
 if isinstance(value,str):return bindings.get(value,value)
 if isinstance(value,list):return [replace_paths(v,bindings) for v in value]
 if isinstance(value,dict):return {bindings.get(k,k):replace_paths(v,bindings) for k,v in value.items()}
 return value

def bundle(data,run,images,access,labels,options,extra=()):
 """Copy only explicit science/calibration/checkpoint files to content-addressed storage."""
 from raster_pipeline import source_assets
 store=root(data);objects=store/'objects';objects.mkdir(parents=True,exist_ok=True);files={};bindings={}
 def add(path):
  p=Path(path)
  if not p.is_absolute() or p.is_symlink() or not p.is_file():raise ValueError('Worker input is missing, relative or a symbolic link')
  if p.suffix.lower() not in ('.tif','.tiff','.jp2','.xml','.txt','.json','.joblib','.safetensors'):raise ValueError('Unsupported worker artifact type; executable sources are never bundled')
  fd,temp=tempfile.mkstemp(prefix='.stage-',dir=objects);h=hashlib.sha256();size=0
  try:
   with p.open('rb') as source,os.fdopen(fd,'wb') as out:
    for chunk in iter(lambda:source.read(1024*1024),b''):
     size+=len(chunk)
     if size>8*1024**3:raise ValueError('One worker input exceeds the 8 GiB transfer limit; use a bounded AOI or smaller source product')
     h.update(chunk);out.write(chunk)
    out.flush();os.fsync(out.fileno())
   oid=h.hexdigest()+p.suffix.lower();target=objects/oid
   if target.exists():
    if target.stat().st_size!=size or file_hash(target)!=h.hexdigest():raise ValueError('Immutable worker object was changed')
   else:
    try:os.link(temp,target)
    except FileExistsError:
     if file_hash(target)!=h.hexdigest():raise ValueError('Concurrent immutable input publication differs')
   files[oid]={'sha256':h.hexdigest(),'bytes':size};bindings[str(p)]=oid
  finally:Path(temp).unlink(missing_ok=True)
 for path in extra:add(path)
 for record in access.values():
  if record.get('mode') not in ('local_path','download'):raise ValueError('Only local science snapshots can be dispatched')
  for _,path in source_assets(record):add(path)
  for item in record.get('local_sources',[]):
   if (item.get('radiometry') or {}).get('metadata_path'):add(item['radiometry']['metadata_path'])
 checkpoints={**options.get('initial_checkpoints',{}),**run.get('qualified_checkpoints',{})}
 for path in checkpoints.values():
  if path:add(path)
 for release in run.get('qualification',{}).values():
  for path,expected in release.get('files',{}).items():
   add(path)
   if files[bindings[path]]['sha256']!=expected:raise ValueError('Qualified model changed before worker dispatch')
 adapter=run['requested_models'][0]
 record={'protocol':1,'signature':signature(adapter),'files':files,'bindings':bindings,'request':{'run':run,'images':images,'access':access,'labels':labels,'options':options}}
 encoded=canonical(record);identifier=hashlib.sha256(encoded).hexdigest();folder=store/'bundles';folder.mkdir(exist_ok=True);target=folder/(identifier+'.json')
 if target.exists() and target.read_bytes()!=encoded:raise ValueError('Immutable worker bundle differs')
 if not target.exists():
  fd,name=tempfile.mkstemp(prefix='.bundle-',dir=folder)
  try:
   with os.fdopen(fd,'wb') as stream:stream.write(encoded);stream.flush();os.fsync(stream.fileno())
   try:os.link(name,target)
   except FileExistsError:
    if target.read_bytes()!=encoded:raise ValueError('Concurrent worker bundle differs')
  finally:Path(name).unlink(missing_ok=True)
 return {'id':identifier,'sha256':hashlib.sha256(encoded).hexdigest()}

def get_task(database,task_id):
 with database() as db:row=db.execute('SELECT payload FROM pool_tasks WHERE id=?',(task_id,)).fetchone()
 if not row:raise ValueError('Worker task not found')
 return json.loads(row['payload'])
def lease(database,worker_id,key,task_id,token,allow_stopping=False):
 pool=Queue(database);pool.worker(worker_id,key);pool.snapshot();task=get_task(database,task_id)
 import secrets
 from distributed_queue import digest
 if task.get('worker_id')!=worker_id or task['status'] not in ACTIVE or not secrets.compare_digest(task.get('lease_token_hash',''),digest(token or '')) or (not allow_stopping and task['control']=='stop'):raise PermissionError('Worker lease is no longer authorized')
 return task

def enqueue(database,data,run,images,access,labels,options,dedup=None):
 manifest=bundle(data,run,images,access,labels,options);training=run.get('run_kind')=='training'
 val=[r for r in labels if r.get('role')=='validation' or r.get('benchmark_split') in ('validation','test')]
 payload={'run_id':run['id'],'mission_id':run['mission_id'],'job_id':run['job_id'],'input_manifest':manifest,'model_release':hashlib.sha256(canonical(run.get('qualified_releases',{}))).hexdigest() if not training else None}
 if training:payload.update(training_version=run['id'],validation_manifest={'sha256':hashlib.sha256(canonical(val)).hexdigest()})
 return Queue(database).enqueue(run['requested_models'][0],'training' if training else 'inference',payload,dedup or run['id'],ram_gib=2 if run['requested_models'][0] in ('random_forest','svm','xgboost') else 5,device='cpu' if run['requested_models'][0]=='dynamicworld' else 'auto')

def desired(database,run):
 import mission_control
 action=mission_control.action(database,run['mission_id'])
 with database() as db:
  saved=db.execute('SELECT payload FROM inference_runs WHERE id=?',(run['id'],)).fetchone();saved=json.loads(saved['payload']) if saved else {}
  if saved.get('control_stop'):return 'stop'
  if action=='resume':action=saved.get('control_action','resume')
  for identifier,release in run.get('qualified_releases',{}).items():
   profile=db.execute('SELECT payload FROM workflow_models WHERE id=?',(identifier,)).fetchone();profile=json.loads(profile['payload']) if profile else {}
   if profile.get('state')!='qualified':return 'pause'
   if profile.get('release',{}).get('id')!=release:return 'stop'
 return 'stop' if action=='delete' else action

def collect(database,data,task,parent):
 """Verify worker output, preserve its original, import a new managed result copy."""
 pool=Queue(database)
 if task['status']=='awaiting_verification':task=pool.verify_result(task['id'],root(data)/'received',root(data)/'verified')
 if task['status']!='complete':raise ValueError('Worker result was not verified')
 source=Path(task['verification']['directory']);report=json.loads((source/'report.json').read_text())
 if report.get('id')!=task['payload']['run_id'] or report.get('requested_models')!=parent['requested_models'] or report.get('mission_id')!=parent['mission_id'] or report.get('job_id')!=parent['job_id']:raise ValueError('Worker report does not match the requested run')
 if report.get('status')!='complete':raise ValueError('Worker model did not complete successfully: '+str(report.get('stage','unknown failure')))
 destination=Path(data)/'results'/parent['id'];destination.parent.mkdir(exist_ok=True,parents=True)
 if destination.exists():
  previous=json.loads((destination/'report.json').read_text())
  if previous.get('id')!=parent['id'] or previous.get('distributed',{}).get('task_id')!=task['id']:raise ValueError('Managed result already exists with different lineage; it will not be overwritten')
  for item in json.loads((source/'verification.json').read_text())['files']:
   if item['path']=='report.json' or (parent['requested_models']==['dynamicworld'] and item['path']=='dynamicworld.joblib'):continue
   if not (destination/item['path']).is_file() or file_hash(destination/item['path'])!=item['sha256']:raise ValueError('Previously imported scientific output changed; recovery stopped')
  return previous
 stage=Path(tempfile.mkdtemp(prefix='.import-',dir=destination.parent))
 try:
  for item in json.loads((source/'verification.json').read_text())['files']:
   p=source/item['path'];target=stage/item['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
   if file_hash(target)!=item['sha256']:raise ValueError('Verified output changed during import')
  origin=report.pop('_worker_result_root',None)
  if not origin:raise ValueError('Worker did not declare its result namespace')
  def rebase(value):
   if isinstance(value,str):return (str(destination)+value[len(origin):] if value==origin or value.startswith(origin+'/') or value.startswith(origin+'\\') else value).replace(task['payload']['run_id'],parent['id'])
   if isinstance(value,list):return [rebase(v) for v in value]
   if isinstance(value,dict):return {rebase(k):rebase(v) for k,v in value.items()}
   return value
  report=rebase(report)
  # Native pretrained cache paths belong to the worker. Rebind the portable
  # checkpoint only after comparing every official released model file hash.
  if parent['requested_models']==['dynamicworld']:
   from pretrained_lulc import prepare_dynamicworld
   import joblib
   model_path,info=prepare_dynamicworld(Path(data)/'models')
   checkpoint=stage/'dynamicworld.joblib';record=joblib.load(checkpoint)
   if record.get('files')!=info['files']:raise ValueError('Worker Dynamic World model differs from the local pinned release')
   record['model_path']=str(model_path);joblib.dump(record,checkpoint)
  report.update(id=parent['id'],execution_route='enrolled worker pool',distributed={'task_id':task['id'],'worker_id':task['worker_id'],'device':task.get('device'),'attempt':task['attempt'],'verified_original':str(source),'input_manifest':task['payload']['input_manifest']})
  for model in report.get('models',[]):
   checkpoint=stage/(model['id']+'.joblib')
   if checkpoint.exists():model.setdefault('training',{})['path']=str(destination/checkpoint.name)
  # Preserve coordinator-side control/version/provenance fields as authoritative.
  for key in ('candidate_version','correction_evidence','training_policy','trigger','qualified_releases','qualification','workflow_model_ids','pipeline_id','run_kind'):
   if key in parent:report[key]=parent[key]
  (stage/'report.json').write_text(json.dumps(report,indent=2));stage.rename(destination)
 finally:
  if stage.exists():shutil.rmtree(stage)
 return report

def wait_tasks(database,data,run,persist,identifiers,config,base=0,span=98,shutdown=None):
 pool=Queue(database);last=time.monotonic();queued={i:0. for i in identifiers}
 while True:
  if shutdown and shutdown.is_set():raise CoordinatorRestart()
  pool.snapshot();tasks=[get_task(database,i) for i in identifiers];command=desired(database,run);now=time.monotonic();delta=now-last;last=now
  for task in tasks:
   if command in ('pause','stop') and task['status'] not in ('complete','cancelled','failed','awaiting_verification') and task['control']!=command:pool.control(task['id'],command)
   elif command=='resume' and task['control']=='pause' and task['status'] not in ('complete','failed','cancelled','awaiting_verification'):pool.control(task['id'],'resume')
  if command=='stop':
   for task in tasks:
    if task['status']=='awaiting_verification':pool.control(task['id'],'stop')
   raise InterruptedError('Stopped; worker artifacts preserved')
  if command=='pause':run.update(status='paused' if all(t['status'] in ('paused','awaiting_verification','complete') for t in tasks) else 'pausing',stage='Waiting for worker safe checkpoints; no result will be published')
  else:
   run['dispatch_active_seconds']=run.get('dispatch_active_seconds',0)+delta
   failed=next((t for t in tasks if t['status'] in ('failed','cancelled')),None)
   if failed:raise ValueError(failed.get('stage') or failed.get('error') or 'Worker task failed')
   if all(t['status'] in ('awaiting_verification','complete') for t in tasks):return tasks
   progress=sum(t.get('progress',0) for t in tasks)/len(tasks)
   run.update(status='queued' if all(t['status']=='queued' for t in tasks) else 'running',stage=' · '.join(dict.fromkeys(t.get('stage','Waiting for enrolled worker') for t in tasks))[:500],progress=base+min(span,int(span*progress/100)),distributed_workers=list(dict.fromkeys(t.get('worker_id') for t in tasks if t.get('worker_id'))))
   for task in tasks:
    if task['status']=='queued':queued[task['id']]+=delta
    if queued[task['id']]>config.get('wait_seconds',120):raise TimeoutError('An eligible worker did not accept a task before its queue deadline')
   if run['dispatch_active_seconds']>config.get('maximum_seconds',3600):raise TimeoutError('Worker computation exceeded its bounded active run time')
  persist(run);time.sleep(.25)

def merge_forest(database,data,run,tasks):
 with _MERGE_SLOT:return _merge_forest(database,data,run,tasks)
def _merge_forest(database,data,run,tasks):
 import copy,joblib
 from sklearn.ensemble import RandomForestClassifier
 from execution_policy import available_ram_bytes
 records=[];members=[];pool=Queue(database);checkpoints=[]
 for task in tasks:
  if task['status']=='awaiting_verification':task=pool.verify_result(task['id'],root(data)/'received',root(data)/'verified')
  source=Path(task['verification']['directory']);checkpoints.append((task,source,source/'random_forest.joblib'))
 budget=3*sum(p.stat().st_size for _,_,p in checkpoints)+256*1024**2;available=available_ram_bytes()
 if available is None or available<budget:raise ValueError('Insufficient coordinator RAM to combine forest shards safely; immutable shard outputs are preserved. Use a smaller training grid or a coordinator with more available RAM.')
 for task,source,checkpoint in checkpoints:
  report=json.loads((source/'report.json').read_text())
  if report.get('status')!='complete' or report.get('requested_models')!=['random_forest']:raise ValueError('A forest shard failed or returned the wrong algorithm')
  record=joblib.load(checkpoint)
  if not isinstance(record.get('estimator'),RandomForestClassifier) or not record.get('distributed_forest') or record['model_id']!='random_forest':raise ValueError('Worker output is not a typed fitted Random Forest shard')
  if records and any(record.get(k)!=records[0].get(k) for k in ('bands','classes','training_data_sha256','training_pixels','sampling_seed')):raise ValueError('Forest shards used different bands, taxonomy or training pixels')
  spec=record['distributed_forest']
  if record['estimator'].random_state!=spec['seed'] or len(record['estimator'].estimators_)!=spec['trees']:raise ValueError('Forest shard parameters differ from the requested tree group')
  records.append(record);members.append({'task_id':task['id'],'worker_id':task['worker_id'],'attempt':task['attempt'],'checkpoint':str(checkpoint),'sha256':file_hash(checkpoint),'trees':len(record['estimator'].estimators_),'training_seconds':report['models'][0]['training']['training_seconds']})
 specifications=[r['distributed_forest'] for r in records];count=len(records)
 if {s['shard'] for s in specifications}!=set(range(count)) or any(s['total_shards']!=count for s in specifications):raise ValueError('Forest has missing or duplicate tree shards')
 result=copy.deepcopy(records[0]);result['estimator'].estimators_=[tree for r in records for tree in r['estimator'].estimators_];result['estimator'].n_estimators=len(result['estimator'].estimators_)
 if result['estimator'].n_estimators!=128:raise ValueError('Distributed forest must contain exactly the declared 128 trees')
 result.pop('distributed_forest',None);result['implementation']='Random Forest · 128 trees grown across enrolled workers on identical approved training samples'
 result['distributed_training']={'method':'independent tree groups, identical approved training pixels; equal weight per tree','members':members,'validation_fitted':False,'distinct_workers':len({m['worker_id'] for m in members}),'not_generic_neural_multi_node_training':True}
 folder=root(data)/'trained'/run['id'];folder.mkdir(parents=True,exist_ok=True);path=folder/'random_forest.joblib'
 if path.exists():
  previous=joblib.load(path)
  if previous.get('distributed_training')!=result['distributed_training']:raise ValueError('Existing candidate differs; immutable model will not be overwritten')
 else:
  temp=path.with_suffix('.part');joblib.dump(result,temp);temp.replace(path)
 return path,result

def execute(database,run,persist,images=None,access=None,labels=None,data=None,options=None,shutdown=None):
 """Real remote jobs, plus validated distributed tree training for Random Forest."""
 import copy
 pool=Queue(database);config=run.get('worker_pool_configuration') or setting(database);plan=run.get('distributed_plan');all_ids=list(run.get('distributed_task_ids',[]))
 if run.get('unet_workers'):
  from unet_pool import execute as execute_unet
  return execute_unet(database,run,persist,images,access,labels,data,options,shutdown)
 try:
  if not plan:
   if run.get('_recovery_bundle'):
    recovered=json.loads((root(data)/'bundles'/(run.pop('_recovery_bundle')+'.json')).read_text())['request'];images,access,labels,options=[recovered[k] for k in ('images','access','labels','options')];count=recovered['run'].get('distributed_forest',{}).get('total_shards',1)
   else:count=min(config.get('forest_shards',1),len(suitable(database,'random_forest')),8) if run.get('run_kind')=='training' and run['requested_models']==['random_forest'] else 1
   count=max(1,count);ids=[]
   for i in range(count):
    request=copy.deepcopy(run)
    if count>1:
     request['id']=str(uuid4());request['distributed_forest']={'trees':128//count+(i<128%count),'seed':42+i*1009,'shard':i,'total_shards':count};request['distributed_forest']['trees']=int(request['distributed_forest']['trees'])
    task=enqueue(database,data,request,images,access,labels,options,dedup=run['id']+':'+str(i));ids.append(task['id'])
   plan={'phase':'shards' if count>1 else 'single','tasks':ids,'original_bundle':get_task(database,ids[0])['payload']['input_manifest']};all_ids=ids
   run.update(distributed_plan=plan,distributed_task_ids=all_ids,execution_route='enrolled worker pool',status='queued',stage='Waiting for enrolled workers',progress=0);persist(run)
  if plan['phase']=='shards':
   tasks=wait_tasks(database,data,run,persist,plan['tasks'],config,0,70,shutdown);path,record=merge_forest(database,data,run,tasks)
   original=json.loads((root(data)/'bundles'/(plan['original_bundle']['id']+'.json')).read_text())['request'];evaluation=copy.deepcopy(run);evaluation.update(id=str(uuid4()),run_kind='operational_inference',qualified_classes=record['classes'],qualified_checkpoints={'random_forest':str(path)},qualification={},qualified_releases={});evaluation.pop('distributed_forest',None)
   opts={**original['options'],'initial_checkpoints':{}};task=enqueue(database,data,evaluation,original['images'],original['access'],original['labels'],opts,dedup=run['id']+':validation')
   plan.update(phase='validation',tasks=[task['id']],checkpoint=str(path),checkpoint_sha256=file_hash(path),members=record['distributed_training']);all_ids.append(task['id']);run.update(distributed_plan=plan,distributed_task_ids=all_ids,stage='Evaluating the combined forest on held-out validation; no fitting',progress=70);persist(run)
  tasks=wait_tasks(database,data,run,persist,plan['tasks'],config,70 if plan['phase']=='validation' else 0,28 if plan['phase']=='validation' else 98,shutdown)
  if desired(database,run)!='resume':raise InterruptedError('Model/control changed before publication')
  run.update(status='running',stage='Verifying and importing immutable worker outputs',progress=99);persist(run);result=collect(database,data,tasks[0],run)
  if plan['phase']=='validation':
   if file_hash(Path(plan['checkpoint']))!=plan['checkpoint_sha256']:raise ValueError('Combined candidate changed before publication')
   result['distributed_training']=plan['members'];result['models'][0]['evaluation_fitting_seconds']=result['models'][0]['training']['training_seconds'];result['models'][0]['training'].update(initialization='Combined independently grown tree shards; no held-out labels used for fitting',implementation='Random Forest · distributed tree training',training_mode='distributed_tree_groups',members=plan['members']['members'],training_seconds=sum(m['training_seconds'] for m in plan['members']['members']),timing_scope='Sum of shard fitting times; not wall-clock speedup')
   result['validation_protocol']='User-designated validation regions excluded from every tree shard. Combined forest evaluated on held-out references without refitting; field accuracy not implied.'
  result.update(distributed_plan=plan,distributed_task_ids=all_ids,dispatch_active_seconds=run.get('dispatch_active_seconds',0))
  if desired(database,run)!='resume':raise InterruptedError('Control/model changed during import; copied artifacts preserved without publication')
  (Path(data)/'results'/run['id']/'report.json').write_text(json.dumps(result,indent=2));run.clear();run.update(result);persist(run);return run
 except CoordinatorRestart:
  run.update(status='queued',stage='Coordinator restarting; saved worker leases/results will be recovered');persist(run);return run
 except Exception as exc:
  for identifier in all_ids:
   try:pool.control(identifier,'stop')
   except ValueError:pass
  cancelled=isinstance(exc,InterruptedError) or desired(database,run)=='stop';run.update(status='cancelled' if cancelled else 'failed',stage='Stopped; artifacts preserved' if cancelled else 'Worker execution needs attention',error=str(exc));persist(run);return run

def register(backend):
 from fastapi import Header,HTTPException,Request
 from fastapi.responses import FileResponse
 from pydantic import BaseModel,Field
 from starlette.concurrency import run_in_threadpool
 from agent_services import access_key
 import secrets
 app=backend.app
 import threading
 from concurrent.futures import ThreadPoolExecutor
 backend.pool_executor=ThreadPoolExecutor(max_workers=8,thread_name_prefix='pool-coordinator');backend.pool_shutdown=threading.Event()
 async def recover_worker_runs():
  backend.pool_shutdown.clear()
  Queue(backend._database).init()
  with backend._database() as db:
   rows=db.execute('SELECT payload FROM inference_runs').fetchall();tasks=[(r['dedup'],json.loads(r['payload'])) for r in db.execute('SELECT dedup,payload FROM pool_tasks')]
  for row in rows:
   run=json.loads(row['payload'])
   if run.get('execution_route')!='enrolled worker pool' or run.get('status') not in ('queued','running','pausing','paused','stopping','blocked'):continue
   if not run.get('distributed_plan'):
    previous=next((t for key,t in tasks if key.startswith(run['id']+':')),None)
    if previous:run['_recovery_bundle']=previous['payload']['input_manifest']['id']
    else:
     run.update(status='interrupted',stage='Worker dispatch was not committed; submit a new run. No duplicate work was started.');backend._persist_run(run);continue
   backend.pool_executor.submit(execute,backend._database,run,backend._persist_run,data=backend.DATA_DIR,shutdown=backend.pool_shutdown)
 backend.recover_pool_runs=recover_worker_runs
 recovery=None
 @app.on_event('startup')
 async def start_recovery():
  from mac_web_gate import pending,wait_for_commit
  nonlocal recovery
  if not pending():
   await recover_worker_runs()
   return
  async def deferred():
   await wait_for_commit()
   await recover_worker_runs()
  recovery=asyncio.create_task(deferred())
 @app.on_event('shutdown')
 async def suspend_coordinators():
  backend.pool_shutdown.set()
  if recovery and not recovery.done():
   recovery.cancel()
   await asyncio.gather(recovery,return_exceptions=True)
 def admin(key):
  if not key or not secrets.compare_digest(key,access_key(backend.DATA_DIR)):raise HTTPException(401,'Local services key required')
 def authorized(worker,task,key,token):
  try:return lease(backend._database,worker,key,task,token)
  except (ValueError,PermissionError) as exc:raise HTTPException(401,'Active authenticated worker lease required') from exc
 class Configuration(BaseModel):
  enabled:bool=False
  forest_shards:int=Field(default=1,ge=1,le=8)
  unet_workers:int=Field(default=1,ge=1,le=4)
  wait_seconds:int=Field(default=120,ge=5,le=600)
  maximum_seconds:int=Field(default=3600,ge=10,le=86400)
  lease_seconds:int=Field(default=60,ge=10,le=300)
 class Invitation(BaseModel):
  capabilities:list[str]=Field(default_factory=lambda:sorted(SUPPORTED),max_length=20)
  maximum_ram_gib:float=Field(default=128,ge=1,le=65536)
  seconds:int=Field(default=600,ge=60,le=3600)
 class WorkerEnrollment(BaseModel):
  name:str=Field(min_length=1,max_length=120)
  capabilities:list[str]=Field(min_length=1,max_length=20)
  ram_gib:float=Field(ge=1,le=65536)
  devices:list[dict]=Field(min_length=1,max_length=16)
 class EndpointCheck(BaseModel):
  endpoint:str=Field(min_length=1,max_length=2048)
 @app.post('/api/services/pool/check-endpoint')
 async def check_endpoint(body:EndpointCheck,x_geo_services_key:str|None=Header(default=None)):
  admin(x_geo_services_key)
  from internal_http import InternalHTTP
  try:
   client=await run_in_threadpool(InternalHTTP,body.endpoint)
   return {'permitted':True,'address':client.address,'transport':client.scheme,'connection_tested':False,'note':'Address/transport policy checked; worker still verifies the actual TLS connection and certificate.'}
  except (ValueError,OSError) as exc:raise HTTPException(422,str(exc)) from exc
 @app.post('/api/services/pool/invitations')
 async def invitation(body:Invitation,x_geo_services_key:str|None=Header(default=None)):
  admin(x_geo_services_key);return await run_in_threadpool(Queue(backend._database).invitation,body.capabilities,body.maximum_ram_gib,body.seconds)
 @app.post('/api/services/pool/enroll')
 async def enroll(body:WorkerEnrollment,x_geo_worker_invitation:str|None=Header(default=None)):
  try:return await run_in_threadpool(Queue(backend._database).redeem,x_geo_worker_invitation,body.name,body.capabilities,body.ram_gib,body.devices)
  except PermissionError as exc:raise HTTPException(401,str(exc)) from exc
  except ValueError as exc:raise HTTPException(422,str(exc)) from exc
 @app.post('/api/services/pool/configuration')
 async def configure(body:Configuration,x_geo_services_key:str|None=Header(default=None)):
  admin(x_geo_services_key)
  with backend._database() as db:db.execute('INSERT INTO settings VALUES(?,?) ON CONFLICT(key) DO UPDATE SET payload=excluded.payload',('worker_pool',json.dumps(body.model_dump())))
  return body.model_dump()
 @app.get('/api/services/pool/workers/{worker}/tasks/{task}/inputs/{oid}')
 async def input_file(worker:str,task:str,oid:str,x_geo_worker_key:str|None=Header(default=None),x_geo_lease:str|None=Header(default=None)):
  t=authorized(worker,task,x_geo_worker_key,x_geo_lease);manifest=t['payload']['input_manifest'];path=root(backend.DATA_DIR)/'bundles'/(manifest['id']+'.json')
  if oid=='bundle':return FileResponse(path,media_type='application/json')
  data=json.loads(path.read_text())
  if oid not in data['files']:raise HTTPException(404,'File is not part of this task')
  return FileResponse(root(backend.DATA_DIR)/'objects'/oid,media_type='application/octet-stream')
 @app.put('/api/services/pool/workers/{worker}/tasks/{task}/blobs/{sha}')
 async def upload(worker:str,task:str,sha:str,request:Request,x_geo_worker_key:str|None=Header(default=None),x_geo_lease:str|None=Header(default=None)):
  t=authorized(worker,task,x_geo_worker_key,x_geo_lease)
  if not re_hash(sha):raise HTTPException(422,'Expected SHA256 blob identity')
  folder=root(backend.DATA_DIR)/'uploads'/task/str(t['attempt']);folder.mkdir(parents=True,exist_ok=True);target=folder/sha;temp=folder/(sha+'.'+str(uuid4())+'.part');h=hashlib.sha256();size=0
  try:
   with temp.open('xb') as output:
    async for chunk in request.stream():
     size+=len(chunk)
     if size>8*1024**3:raise HTTPException(413,'Worker artifact exceeds 8 GiB')
     h.update(chunk);output.write(chunk)
    output.flush();os.fsync(output.fileno())
   authorized(worker,task,x_geo_worker_key,x_geo_lease)
   if h.hexdigest()!=sha:raise HTTPException(422,'Artifact checksum mismatch')
   if target.exists():
    if file_hash(target)!=sha:raise HTTPException(409,'Immutable worker blob differs')
   else:temp.rename(target)
  finally:temp.unlink(missing_ok=True)
  return {'sha256':sha,'bytes':size}
 @app.post('/api/services/pool/workers/{worker}/tasks/{task}/finish')
 async def finish(worker:str,task:str,request:Request,x_geo_worker_key:str|None=Header(default=None),x_geo_lease:str|None=Header(default=None)):
  authorized(worker,task,x_geo_worker_key,x_geo_lease);raw=bytearray()
  async for chunk in request.stream():
   raw.extend(chunk)
   if len(raw)>2*1024**2:raise HTTPException(413,'Result inventory exceeds 2 MiB')
  return await run_in_threadpool(assemble_result,worker,task,x_geo_worker_key,x_geo_lease,raw)
 def assemble_result(worker,task,x_geo_worker_key,x_geo_lease,raw):
  t=authorized(worker,task,x_geo_worker_key,x_geo_lease)
  try:
   manifest=json.loads(raw);files=manifest['files']
   if not isinstance(files,list) or not 1<=len(files)<=10000 or manifest.get('task_id')!=task or manifest.get('attempt')!=t['attempt']:raise ValueError('Invalid result identity')
   destination=root(backend.DATA_DIR)/'received'/task/str(t['attempt']);destination.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.receive-',dir=destination.parent));seen=set()
   try:
    for item in files:
     relative=PurePosixPath(item['path'])
     if relative.is_absolute() or '..' in relative.parts or '\\' in str(relative) or ':' in str(relative) or str(relative) in seen or not re_hash(item['sha256']):raise ValueError('Invalid artifact path or hash')
     seen.add(str(relative));source=root(backend.DATA_DIR)/'uploads'/task/str(t['attempt'])/item['sha256'];target=stage/str(relative)
     if not source.is_file() or source.stat().st_size!=item['bytes'] or file_hash(source)!=item['sha256']:raise ValueError('Uploaded artifact did not verify')
     target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    encoded=canonical(manifest);(stage/'result.json').write_bytes(encoded)
    authorized(worker,task,x_geo_worker_key,x_geo_lease)
    if destination.exists():
     # A coordinator restart can occur after the directory commit but before
     # the queue update. Resume only the exact, still-verified same inventory.
     if (destination/'result.json').read_bytes()!=encoded:raise ValueError('Committed result inventory differs')
     for item in files:
      committed=destination/item['path']
      if committed.is_symlink() or not committed.is_file() or committed.stat().st_size!=item['bytes'] or file_hash(committed)!=item['sha256']:raise ValueError('Committed artifact changed')
    else:stage.rename(destination)
   finally:
    if stage.exists():shutil.rmtree(stage)
   return Queue(backend._database).update(worker,x_geo_worker_key,task,x_geo_lease,100,'Worker computation complete; awaiting coordinator verification','complete',{'manifest_sha256':hashlib.sha256(encoded).hexdigest()})
  except (ValueError,KeyError,TypeError) as exc:raise HTTPException(422,'Worker result inventory is invalid: '+str(exc)) from exc
