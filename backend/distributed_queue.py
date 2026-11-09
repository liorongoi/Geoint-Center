"""Durable authenticated worker-pool protocol. No arbitrary commands or code.

Workers lease typed tasks. The execution layer owns algorithm-specific fitting
and merging; the queue never partitions a training dataset.
"""
import hashlib,json,math,secrets,time
from uuid import uuid4

ADAPTERS={'random_forest':{'inference','training'},'svm':{'inference','training'},'xgboost':{'inference','training'},'unet':{'inference','training'},'prithvi':{'inference','training'},'dynamicworld':{'inference'},'oriented_rcnn_dota':{'inference'}}
CPU_ONLY={'random_forest','svm','xgboost'}
ACTIVE={'leased','running','pausing','paused','stopping'}

def digest(token):return hashlib.sha256(token.encode()).hexdigest()
def number(value,minimum=0):
 if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<minimum:raise ValueError('Resource values must be finite nonnegative numbers')
 return value

class Queue:
 def __init__(self,database,clock=time.time):self.database,self.clock=database,clock
 def init(self):
  with self.database() as db:
   db.execute('CREATE TABLE IF NOT EXISTS pool_workers(id TEXT PRIMARY KEY,secret_hash TEXT NOT NULL,payload TEXT NOT NULL)')
   db.execute('CREATE TABLE IF NOT EXISTS pool_tasks(id TEXT PRIMARY KEY,dedup TEXT NOT NULL UNIQUE,payload TEXT NOT NULL,created_at REAL NOT NULL)')
   db.execute('CREATE TABLE IF NOT EXISTS pool_invites(secret_hash TEXT PRIMARY KEY,payload TEXT NOT NULL)')
   db.execute("CREATE INDEX IF NOT EXISTS pool_task_leases ON pool_tasks(json_extract(payload,'$.status'),json_extract(payload,'$.lease_until'))")
   db.execute("CREATE INDEX IF NOT EXISTS pool_task_owner ON pool_tasks(json_extract(payload,'$.worker_id'),json_extract(payload,'$.status'))")
 def enroll(self,name,capabilities,ram_gib,devices,connection=None):
  if connection is None:self.init()
  if not name.strip() or len(name)>120:raise ValueError('Name this trusted internal worker')
  if not capabilities or set(capabilities)-set(ADAPTERS):raise ValueError('Workers may declare only installed typed adapters')
  ram_gib=number(ram_gib,1)
  if not devices or len(devices)>16:raise ValueError('Declare the actual CPU/GPU devices')
  normalized=[]
  for d in devices:
   if set(d)-{'id','kind','vram_gib'} or d.get('kind') not in ('cpu','cuda','mps') or not isinstance(d.get('id'),str) or not d['id'] or len(d['id'])>64:raise ValueError('Use explicit cpu/cuda/mps device identifiers')
   normalized.append({'id':d['id'],'kind':d['kind'],'vram_gib':number(d.get('vram_gib',0))})
  if len({d['id'] for d in normalized})!=len(normalized):raise ValueError('Device identifiers must be unique')
  key=secrets.token_hex(32);worker={'id':str(uuid4()),'name':name.strip(),'capabilities':sorted(set(capabilities)),'ram_gib':ram_gib,'devices':normalized,'state':'enabled','last_seen':None,'created_at':self.clock(),'trust':'Explicitly enrolled internal worker; hardware declaration must be checked on the target'}
  if connection is not None:connection.execute('INSERT INTO pool_workers VALUES(?,?,?)',(worker['id'],digest(key),json.dumps(worker)))
  else:
   with self.database() as db:db.execute('INSERT INTO pool_workers VALUES(?,?,?)',(worker['id'],digest(key),json.dumps(worker)))
  return {**worker,'access_key':key,'key_notice':'Save once on the trusted worker; the coordinator stores only its hash'}
 def invitation(self,capabilities,maximum_ram_gib=128,seconds=600):
  self.init()
  if not capabilities or set(capabilities)-set(ADAPTERS):raise ValueError('Choose installed worker adapters')
  if not 60<=seconds<=3600:raise ValueError('Worker invitations last 1–60 minutes')
  token=secrets.token_hex(32);record={'capabilities':capabilities,'maximum_ram_gib':number(maximum_ram_gib,1),'expires_at':self.clock()+seconds,'used_by':None}
  with self.database() as db:db.execute('INSERT INTO pool_invites VALUES(?,?)',(digest(token),json.dumps(record)))
  return {**record,'invitation':token,'scope':'One enrollment only; not a services/admin key'}
 def redeem(self,token,name,capabilities,ram_gib,devices):
  self.init()
  with self.database() as db:
   db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT payload FROM pool_invites WHERE secret_hash=?',(digest(token or ''),)).fetchone()
   if not row:raise PermissionError('Invalid, consumed or expired enrollment invitation')
   record=json.loads(row['payload'])
   if record['used_by'] or record['expires_at']<=self.clock():raise PermissionError('Invalid, consumed or expired enrollment invitation')
   if set(capabilities)-set(record['capabilities']) or number(ram_gib)>record['maximum_ram_gib']:raise PermissionError('Worker capabilities/resources exceed the approved invitation')
   worker=self.enroll(name,capabilities,ram_gib,devices,connection=db);record['used_by']=worker['id'];db.execute('UPDATE pool_invites SET payload=? WHERE secret_hash=?',(json.dumps(record),digest(token)));return worker
 def worker(self,identifier,key,connection=None):
  if connection is None:
   self.init()
   with self.database() as db:return self.worker(identifier,key,db)
  row=connection.execute('SELECT secret_hash,payload FROM pool_workers WHERE id=?',(identifier,)).fetchone()
  if not row or not secrets.compare_digest(row['secret_hash'],digest(key or '')):raise PermissionError('Worker authentication failed')
  worker=json.loads(row['payload'])
  if worker['state']!='enabled':raise PermissionError('Worker enrollment was revoked')
  return worker
 def enqueue(self,adapter,kind,payload,dedup,ram_gib=1,device='auto',vram_gib=0,max_attempts=3,assigned_worker=None):
  self.init()
  if adapter not in ADAPTERS or kind not in ADAPTERS[adapter]:raise ValueError('No registered typed adapter for this task and training mode')
  if device not in ('auto','cpu','cuda','mps'):raise ValueError('Choose auto, cpu, cuda or mps')
  if adapter in CPU_ONLY and device not in ('auto','cpu'):raise ValueError('This classical adapter executes on CPU')
  if not isinstance(dedup,str) or not 1<=len(dedup)<=256:raise ValueError('Provide a stable bounded idempotency key')
  if not isinstance(payload,dict) or set(payload)-{'run_id','mission_id','job_id','input_manifest','output_namespace','model_release','training_version','validation_manifest','checkpoint_manifest'}:raise ValueError('Use a typed artifact-manifest task; arbitrary commands, source code and URLs are not accepted')
  if not isinstance(payload.get('input_manifest'),dict) or not re_hash(payload['input_manifest'].get('sha256')):raise ValueError('Tasks require an immutable hashed input manifest')
  if kind=='training' and (not payload.get('training_version') or not isinstance(payload.get('validation_manifest'),dict)):raise ValueError('Training tasks require a separate candidate version and explicit independent validation manifest')
  if kind=='inference' and not payload.get('model_release'):raise ValueError('Inference requires a pinned model release')
  if not isinstance(max_attempts,int) or isinstance(max_attempts,bool) or not 1<=max_attempts<=10:raise ValueError('Use 1–10 bounded attempts')
  if assigned_worker is not None and (adapter!='unet' or kind!='training' or not isinstance(assigned_worker,str) or len(assigned_worker)!=36):raise ValueError('Only typed U-Net training may reserve an enrolled worker')
  task={'id':str(uuid4()),'adapter':adapter,'kind':kind,'payload':payload,'status':'queued','control':'resume','progress':0,'created_at':self.clock(),'attempt':0,'max_attempts':max_attempts,'requirements':{'ram_gib':number(ram_gib,0.1),'device':'cpu' if adapter in CPU_ONLY else device,'vram_gib':number(vram_gib)},'history':[]}
  if assigned_worker is not None:task['assigned_worker']=assigned_worker
  with self.database() as db:
   db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT payload FROM pool_tasks WHERE dedup=?',(dedup,)).fetchone()
   if row:return json.loads(row['payload'])
   db.execute('INSERT INTO pool_tasks VALUES(?,?,?,?)',(task['id'],dedup,json.dumps(task),task['created_at']))
  return task
 def _expire(self,db):
  for row in db.execute("SELECT payload FROM pool_tasks WHERE json_extract(payload,'$.status') IN ('leased','running','pausing','paused','stopping') AND json_extract(payload,'$.lease_token_hash') IS NOT NULL AND json_extract(payload,'$.lease_until')<=?",(self.clock(),)):
   task=json.loads(row['payload'])
   if task['status'] not in ACTIVE or not task.get('lease_token_hash') or task.get('lease_until',0)>self.clock():continue
   task['history'].append({'attempt':task['attempt'],'worker_id':task.get('worker_id'),'event':'lease_expired','at':self.clock()})
   task.update(status='cancelled' if task['control']=='stop' else 'paused' if task['control']=='pause' else 'queued' if task['attempt']<task['max_attempts'] else 'failed',error='Worker lease expired; stale results are rejected')
   task.pop('lease_token_hash',None);task.pop('worker_id',None);task.pop('lease_until',None)
   db.execute('UPDATE pool_tasks SET payload=? WHERE id=?',(json.dumps(task),task['id']))
 def claim(self,identifier,key,lease_seconds=60,available_ram_gib=None):
  self.init()
  if not 10<=lease_seconds<=300:raise ValueError('Use a lease duration from 10 to 300 seconds')
  with self.database() as db:
   db.execute('BEGIN IMMEDIATE');worker=self.worker(identifier,key,db);self._expire(db)
   budget=worker['ram_gib']
   if available_ram_gib is not None:
    budget=min(budget,number(available_ram_gib))
    worker['available_ram_gib']=budget
   else:worker.pop('available_ram_gib',None)
   worker['last_seen']=self.clock();db.execute('UPDATE pool_workers SET payload=? WHERE id=?',(json.dumps(worker),identifier))
   if db.execute("SELECT 1 FROM pool_tasks WHERE json_extract(payload,'$.worker_id')=? AND json_extract(payload,'$.status') IN ('leased','running','pausing','paused','stopping') LIMIT 1",(identifier,)).fetchone():return None
   tasks=[json.loads(r['payload']) for r in db.execute("SELECT payload FROM pool_tasks WHERE json_extract(payload,'$.status')='queued' ORDER BY created_at")]
   # GPU-capable workers take compatible neural work before CPU-only tasks.
   has_gpu=any(d['kind'] in ('cuda','mps') for d in worker['devices'])
   tasks.sort(key=lambda t:(has_gpu and t['adapter'] in CPU_ONLY,t['created_at']))
   for task in tasks:
    if task['status']!='queued' or task['control']!='resume' or task['adapter'] not in worker['capabilities']:continue
    if task.get('assigned_worker',identifier)!=identifier:continue
    req=task['requirements']
    if req['ram_gib']>budget:continue
    devices=[d for d in worker['devices'] if (req['device']=='auto' or d['kind']==req['device']) and (d['kind']=='cpu' or d['vram_gib']>=req['vram_gib'])]
    if not devices:continue
    devices.sort(key=lambda d:d['kind']=='cpu');device=devices[0];lease=secrets.token_hex(32)
    task.update(status='leased',worker_id=identifier,device=device['id'],device_kind=device['kind'],lease_token_hash=digest(lease),lease_until=self.clock()+lease_seconds,lease_seconds=lease_seconds,attempt=task['attempt']+1)
    db.execute('UPDATE pool_tasks SET payload=? WHERE id=?',(json.dumps(task),task['id']))
    worker['last_seen']=self.clock();db.execute('UPDATE pool_workers SET payload=? WHERE id=?',(json.dumps(worker),identifier))
    return {**public(task),'lease_token':lease}
  return None
 def update(self,identifier,key,task_id,lease_token,progress,stage,terminal=None,result=None,paused=False):
  self.init();number(progress)
  if not 0<=progress<=100 or len(stage)>500:raise ValueError('Use bounded progress and stage text')
  if terminal not in (None,'complete','failed','cancelled'):raise ValueError('Unknown task completion state')
  with self.database() as db:
   db.execute('BEGIN IMMEDIATE');self.worker(identifier,key,db);self._expire(db);row=db.execute('SELECT payload FROM pool_tasks WHERE id=?',(task_id,)).fetchone()
   if not row:raise ValueError('Task not found')
   task=json.loads(row['payload'])
   if task.get('worker_id')!=identifier or not secrets.compare_digest(task.get('lease_token_hash',''),digest(lease_token)) or task['status'] not in ACTIVE:raise PermissionError('Lease is expired, revoked or belongs to another worker; stale output rejected')
   if terminal=='complete' and task['control']!='resume':raise ValueError('Paused/stopped work cannot commit a successful result')
   if terminal=='complete' and (not isinstance(result,dict) or not re_hash(result.get('manifest_sha256'))):raise ValueError('Completion requires a hashed result manifest; artifact verification is a separate mandatory coordinator step')
   task.update(progress=max(task['progress'],progress),stage=stage,lease_until=self.clock()+task['lease_seconds'],status=('awaiting_verification' if terminal=='complete' else terminal) or ('pausing' if task['control']=='pause' else 'stopping' if task['control']=='stop' else 'running'))
   if paused and not terminal and task['control']=='pause':task['status']='paused'
   if terminal:
    task.update(finished_at=self.clock(),result=result);task.pop('lease_token_hash',None)
    if task.get('archive_after_stop') and terminal in ('failed','cancelled'):task['archived']=True
   db.execute('UPDATE pool_tasks SET payload=? WHERE id=?',(json.dumps(task),task_id))
   return public(task)
 def verify_result(self,task_id,artifact_root,verified_root):
  """Coordinator-only integrity check. Does not qualify scientific accuracy."""
  from pathlib import Path,PurePosixPath
  import os,shutil,tempfile
  from agent_services import file_hash
  root=Path(artifact_root).resolve()
  with self.database() as db:
   row=db.execute('SELECT payload FROM pool_tasks WHERE id=?',(task_id,)).fetchone()
  if not row:raise ValueError('Task not found')
  task=json.loads(row['payload'])
  if task['status']!='awaiting_verification':raise ValueError('No completed worker result awaiting verification')
  folder=root/task_id/str(task['attempt']);manifest=folder/'result.json'
  if any(p.is_symlink() for p in [manifest,*manifest.parents]) or not manifest.is_file() or manifest.stat().st_size>2*1024**2 or file_hash(manifest)!=task['result']['manifest_sha256']:raise ValueError('Worker result manifest is missing, changed or outside its namespace')
  data=json.loads(manifest.read_text())
  if data.get('task_id')!=task_id or data.get('attempt')!=task['attempt'] or data.get('input_sha256')!=task['payload']['input_manifest']['sha256'] or data.get('model_release')!=task['payload'].get('model_release') or data.get('training_version')!=task['payload'].get('training_version'):raise ValueError('Worker result lineage or model/candidate version does not match its task')
  files=data.get('files')
  if not isinstance(files,list) or not 1<=len(files)<=10000:raise ValueError('Expected a bounded artifact inventory')
  seen=set()
  for item in files:
   relative=PurePosixPath(item['path'])
   if relative.is_absolute() or '..' in relative.parts or '\\' in str(relative) or ':' in str(relative) or str(relative) in seen:raise ValueError('Invalid or duplicate worker artifact path')
   seen.add(str(relative));path=folder/str(relative)
   if any(p.is_symlink() for p in [path,*path.parents]) or not path.resolve().is_relative_to(folder.resolve()) or not path.is_file() or path.stat().st_size!=item['bytes'] or file_hash(path)!=item['sha256']:raise ValueError('Worker artifact failed path/size/hash verification')
  # Copy into coordinator-owned storage before publication. Workers retain
  # their scratch outputs, but cannot change the published scientific copy.
  owned=Path(verified_root).resolve()
  if owned==root or owned.is_relative_to(root) or root.is_relative_to(owned):raise ValueError('Verified results require separate coordinator-owned storage outside worker scratch')
  owned.mkdir(parents=True,exist_ok=True);destination=owned/task_id/str(task['attempt']);destination.parent.mkdir(parents=True,exist_ok=True)
  stage=Path(tempfile.mkdtemp(prefix='.verify-',dir=destination.parent))
  try:
   for item in files:
    target=stage/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
    with (folder/item['path']).open('rb') as source,target.open('xb') as output:
     shutil.copyfileobj(source,output,1024*1024);output.flush();os.fsync(output.fileno())
    if target.stat().st_size!=item['bytes'] or file_hash(target)!=item['sha256']:raise ValueError('Worker artifact changed while copying; no result committed')
   receipt={**data,'worker_manifest_sha256':task['result']['manifest_sha256'],'scientific_accuracy':'not implied'}
   (stage/'verification.json').write_text(json.dumps(receipt,sort_keys=True))
   if destination.exists():
    # A prior interruption may have finished the atomic copy before DB commit.
    previous=json.loads((destination/'verification.json').read_text())
    if previous!=receipt or any(file_hash(destination/f['path'])!=f['sha256'] for f in files):raise ValueError('Existing verified result differs; immutable outputs will not be overwritten')
   else:stage.rename(destination)
  finally:
   if stage.exists():shutil.rmtree(stage)
  # CAS prevents controls/revocation racing a slow checksum/copy pass.
  with self.database() as db:
   db.execute('BEGIN IMMEDIATE');current=json.loads(db.execute('SELECT payload FROM pool_tasks WHERE id=?',(task_id,)).fetchone()['payload'])
   if current['status']!='awaiting_verification' or current['attempt']!=task['attempt'] or current['control']!='resume':raise ValueError('Task changed while verifying artifacts; no result committed')
   current.update(status='complete',progress=100,verification={'files':len(files),'integrity':'hashes and coordinator-owned copies verified','directory':str(destination),'scientific_accuracy':'not implied','at':self.clock()});db.execute('UPDATE pool_tasks SET payload=? WHERE id=?',(json.dumps(current),task_id));return public(current)
 def control(self,task_id,action):
  self.init()
  if action not in ('pause','resume','stop','delete'):raise ValueError('Unknown task action')
  with self.database() as db:
   db.execute('BEGIN IMMEDIATE');self._expire(db);row=db.execute('SELECT payload FROM pool_tasks WHERE id=?',(task_id,)).fetchone()
   if not row:raise ValueError('Task not found')
   task=json.loads(row['payload'])
   if task['status']=='awaiting_verification':
    if action not in ('stop','delete'):raise ValueError('Worker finished; verify its artifacts or stop the pending result')
    task.update(control='stop',status='cancelled')
    if action=='delete':task['archived']=True
   elif task['status'] in ('complete','cancelled','failed'):
    if action!='delete':raise ValueError('Terminal tasks are immutable; create a new task to retry')
    task['archived']=True
   elif action=='resume':
    if task['control']!='pause':raise ValueError('Only paused work can resume')
    task.update(control='resume',status='running' if task.get('lease_token_hash') else 'queued')
   else:
    task['control']='pause' if action=='pause' else 'stop'
    task['status']=('pausing' if action=='pause' else 'stopping') if task.get('lease_token_hash') else ('paused' if action=='pause' else 'cancelled')
    if action=='delete':task['archive_after_stop']=True
   db.execute('UPDATE pool_tasks SET payload=? WHERE id=?',(json.dumps(task),task_id));return public(task)
 def revoke(self,identifier):
  self.init()
  with self.database() as db:
   db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT payload FROM pool_workers WHERE id=?',(identifier,)).fetchone()
   if not row:raise ValueError('Worker not found')
   worker=json.loads(row['payload']);worker['state']='revoked';db.execute('UPDATE pool_workers SET payload=? WHERE id=?',(json.dumps(worker),identifier))
   for row in db.execute("SELECT payload FROM pool_tasks WHERE json_extract(payload,'$.worker_id')=?",(identifier,)):
    task=json.loads(row['payload'])
    if task.get('worker_id')==identifier and task['status'] in ACTIVE:
     task['lease_until']=0;db.execute('UPDATE pool_tasks SET payload=? WHERE id=?',(json.dumps(task),task['id']))
    elif task.get('worker_id')==identifier and task['status']=='awaiting_verification':
     task.update(control='stop',status='cancelled',error='Worker revoked before artifact verification');db.execute('UPDATE pool_tasks SET payload=? WHERE id=?',(json.dumps(task),task['id']))
   self._expire(db)
 def snapshot(self):
  self.init()
  with self.database() as db:
   db.execute('BEGIN IMMEDIATE');self._expire(db)
   workers=[json.loads(r['payload']) for r in db.execute('SELECT payload FROM pool_workers')]
   exists=db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='settings'").fetchone();row=db.execute("SELECT payload FROM settings WHERE key='worker_pool'").fetchone() if exists else None
   config=json.loads(row['payload']) if row else {'enabled':False,'forest_shards':1,'wait_seconds':120,'maximum_seconds':3600,'lease_seconds':60}
   return {'workers':workers,'tasks':[public(json.loads(r['payload'])) for r in db.execute('SELECT payload FROM pool_tasks ORDER BY created_at DESC LIMIT 200')],'configuration':config,'implementation_ready':True,'execution_ready':bool(config.get('enabled') and any(w['state']=='enabled' and w.get('last_seen') and self.clock()-w['last_seen']<90 for w in workers)),'adapters':{'inference':['random_forest','svm','xgboost','unet','dynamicworld'],'independent_training':['random_forest','svm','xgboost','unet'],'single_model_distributed_training':['random_forest','unet'],'neural_multi_node_training':True},'limits':'U-Net synchronization uses distinct enrolled workers and coordinator-owned checkpoints. Physical multi-host Windows/CUDA and deployment-specific TLS remain separately unverified.'}

def re_hash(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def public(task):return {k:v for k,v in task.items() if k!='lease_token_hash'}

def register(backend):
 from fastapi import Header,HTTPException,Request
 from pydantic import BaseModel,Field
 from starlette.concurrency import run_in_threadpool
 from agent_services import access_key
 import ipaddress
 app=backend.app
 def pool():return Queue(backend._database)
 def admin(value):
  if not value or not secrets.compare_digest(value,access_key(backend.DATA_DIR)):raise HTTPException(401,'Local services key required')
 def transport(request):
  from agent_services import private_transport
  if not private_transport(request):raise HTTPException(403,'Remote worker credentials require verified internal HTTPS or an explicitly configured same-machine proxy')
 async def invoke(fn,*args):
  try:return await run_in_threadpool(fn,*args)
  except PermissionError as exc:raise HTTPException(401,str(exc)) from exc
  except (ValueError,KeyError) as exc:raise HTTPException(422,str(exc)) from exc
 class Enrollment(BaseModel):
  name:str=Field(min_length=1,max_length=120)
  capabilities:list[str]=Field(min_length=1,max_length=30)
  ram_gib:float=Field(gt=0,le=65536)
  devices:list[dict]=Field(min_length=1,max_length=16)
 class Progress(BaseModel):
  lease_token:str
  progress:float=Field(ge=0,le=100)
  stage:str=Field(max_length=500)
  terminal:str|None=None
  result:dict|None=None
  paused:bool=False
 class Capacity(BaseModel):
  available_ram_gib:float|None=Field(default=None,ge=0,le=65536)
 @app.get('/api/services/pool')
 async def state(x_geo_services_key:str|None=Header(default=None)):
  admin(x_geo_services_key);return await invoke(pool().snapshot)
 @app.post('/api/services/pool/workers')
 async def enroll(body:Enrollment,x_geo_services_key:str|None=Header(default=None)):
  admin(x_geo_services_key);return await invoke(pool().enroll,body.name,body.capabilities,body.ram_gib,body.devices)
 @app.post('/api/services/pool/workers/{identifier}/revoke')
 async def revoke(identifier:str,x_geo_services_key:str|None=Header(default=None)):
  admin(x_geo_services_key);await invoke(pool().revoke,identifier);return {'revoked':True}
 @app.post('/api/services/pool/workers/{identifier}/claim')
 async def claim(identifier:str,request:Request,body:Capacity|None=None,x_geo_worker_key:str|None=Header(default=None)):
  from distributed_execution import setting
  transport(request);return {'task':await invoke(pool().claim,identifier,x_geo_worker_key,setting(backend._database).get('lease_seconds',60),body.available_ram_gib if body else None)}
 @app.post('/api/services/pool/workers/{identifier}/tasks/{task_id}')
 async def progress(identifier:str,task_id:str,body:Progress,request:Request,x_geo_worker_key:str|None=Header(default=None)):
  transport(request);return await invoke(pool().update,identifier,x_geo_worker_key,task_id,body.lease_token,body.progress,body.stage,body.terminal,body.result,body.paused)
 @app.post('/api/services/pool/tasks/{task_id}/{action}')
 async def control(task_id:str,action:str,x_geo_services_key:str|None=Header(default=None)):
  admin(x_geo_services_key);return await invoke(pool().control,task_id,action)
