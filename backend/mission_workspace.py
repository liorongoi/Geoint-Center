"""Three mission workspaces: durable qualification, controls and operational evidence."""
from __future__ import annotations
import asyncio,hashlib,json,logging,logging.handlers,re,threading
from datetime import datetime,timezone,timedelta
from pathlib import Path
from uuid import uuid4
from fastapi import HTTPException
from algorithm_registry import MODEL_DETAILS,SEGMENTATION_DETAILS,ANOMALY_DETAILS
from object_detection import DETAILS as DETECTION_DETAILS
from dsen2 import DETAILS as SUPER_RESOLUTION_DETAILS
from portable_lulc import MODEL as PORTABLE

ACTIVE={'queued','running','pausing','paused','blocked','stopping'}
TERMINAL={'complete','partial','failed','cancelled','interrupted','needs_input'}
def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def clean(value):
 if isinstance(value,dict):return {k:('[redacted]' if re.search(r'password|api.?key|authorization|secret|access.?token',k,re.I) else clean(v)) for k,v in value.items()}
 if isinstance(value,list):return [clean(v) for v in value]
 if isinstance(value,str):
  value=re.sub(r'(?i)(bearer\s+)[^\s,;]+',r'\1[redacted]',value)
  value=re.sub(r'(?i)((?:api[_-]?key|token|password|secret)=)[^&\s]+',r'\1[redacted]',value)
  value=re.sub(r'(?i)([a-z][a-z0-9+.-]*://)[^/@\s]+:[^/@\s]+@',r'\1[redacted]@',value)
  return value[:6000]
 return value

class Workspace:
 def __init__(self,database,data,initialize):
  self.database,self.data,self.initialize=database,data,initialize;self.lock=threading.RLock();self.handlers={};self.started=set()
 def root(self):return Path(self.data())
 def init(self):
  self.initialize()
  with self.database() as db:
   for table in ('workflow_models','workflow_pipelines','workflow_executions','workflow_tasks'):
    db.execute(f'CREATE TABLE IF NOT EXISTS {table}(id TEXT PRIMARY KEY,mission_id TEXT NOT NULL,payload TEXT NOT NULL,created_at TEXT NOT NULL)')
   db.execute('CREATE TABLE IF NOT EXISTS workflow_events(seq INTEGER PRIMARY KEY AUTOINCREMENT,created_at TEXT NOT NULL,level TEXT NOT NULL,mission_id TEXT,pipeline_id TEXT,model_id TEXT,run_id TEXT,code TEXT NOT NULL,message TEXT NOT NULL,payload TEXT NOT NULL,audit INTEGER NOT NULL DEFAULT 0)')
   db.execute('CREATE INDEX IF NOT EXISTS workflow_events_mission ON workflow_events(mission_id,seq)')
   db.execute('CREATE TABLE IF NOT EXISTS workflow_discoveries(key TEXT PRIMARY KEY,pipeline_id TEXT NOT NULL,payload TEXT NOT NULL)')
 def get(self,table,identifier):
  self.init()
  with self.database() as db:row=db.execute(f'SELECT payload FROM {table} WHERE id=?',(identifier,)).fetchone()
  if not row:raise HTTPException(404,'Mission workspace item not found')
  return json.loads(row['payload'])
 def all(self,table,mission_id=None):
  self.init()
  with self.database() as db:rows=db.execute(f'SELECT payload FROM {table}'+(' WHERE mission_id=?' if mission_id else '')+' ORDER BY created_at DESC', (mission_id,) if mission_id else ()).fetchall()
  return [json.loads(r['payload']) for r in rows]
 def put(self,table,item,control_change=False):
  self.init()
  with self.database() as db:
   db.execute('BEGIN IMMEDIATE')
   if table=='workflow_executions' and not control_change:
    old=db.execute('SELECT payload FROM workflow_executions WHERE id=?',(item['id'],)).fetchone()
    if old:item['control']=json.loads(old['payload']).get('control','resume')
   db.execute(f'INSERT INTO {table} VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload',(item['id'],item.get('mission_id',''),json.dumps(item),item['created_at']))
  return item
 def event(self,code,message,level='info',audit=False,**context):
  self.init();context=clean(context);message=clean(message);at=now()
  with self.lock:
   with self.database() as db:
    cursor=db.execute('INSERT INTO workflow_events(created_at,level,mission_id,pipeline_id,model_id,run_id,code,message,payload,audit) VALUES(?,?,?,?,?,?,?,?,?,?)',(at,level,*[context.get(k) for k in ('mission_id','pipeline_id','model_id','run_id')],code,message,json.dumps(context),int(audit)))
    seq=cursor.lastrowid
    db.execute('DELETE FROM workflow_events WHERE audit=0 AND created_at<?',((datetime.now(timezone.utc)-timedelta(days=30)).isoformat(),))
   root=self.root()/'logs';root.mkdir(parents=True,exist_ok=True)
   key=str(root.resolve())
   if key not in self.handlers:
    handler=logging.handlers.RotatingFileHandler(root/'operations.jsonl',maxBytes=2*1024*1024,backupCount=5,encoding='utf-8');handler.setFormatter(logging.Formatter('%(message)s'));self.handlers[key]=handler
   record={'seq':seq,'time':at,'level':level,'code':code,'message':message,**context}
   log=logging.LogRecord('spectrascope.operations',logging.INFO,'',0,json.dumps(record,ensure_ascii=False),(),None);self.handlers[key].emit(log)
  return record
 def events(self,mission_id=None,level=None,after=0,limit=200,pipeline_id=None,model_id=None,run_id=None,since=None):
  self.init();where=['seq>?'];args=[after]
  for column,value in [('mission_id',mission_id),('level',level),('pipeline_id',pipeline_id),('model_id',model_id),('run_id',run_id)]:
   if value:where.append(column+'=?');args.append(value)
  if since:
   try:stamp=datetime.fromisoformat(since.replace('Z','+00:00')).astimezone(timezone.utc).isoformat()
   except ValueError:raise HTTPException(422,'Use a valid ISO timestamp for the log time filter')
   where.append('created_at>=?');args.append(stamp)
  with self.database() as db:rows=db.execute('SELECT * FROM workflow_events WHERE '+' AND '.join(where)+' ORDER BY seq DESC LIMIT ?',(*args,min(10000,max(1,limit)))).fetchall()
  return [{**dict(r),'detail':json.loads(r['payload'])} for r in rows]
 def mission(self,mission_id,job_id=None):
  with self.database() as db:row=db.execute('SELECT payload FROM workspace WHERE id=1').fetchone()
  missions=json.loads(row['payload']).get('missions',[]) if row else []
  mission=next((m for m in missions if m['id']==mission_id and m.get('control')!='delete'),None)
  if not mission:raise HTTPException(404,'Active mission not found')
  if job_id:
   job=next((j for j in mission.get('jobs',[]) if j['id']==job_id),None)
   if not job:raise HTTPException(404,'Job not found in this mission')
   return mission,job
  return mission
 def model(self,identifier,qualified=False):
  item=self.get('workflow_models',identifier)
  if item.get('state')=='deleted':raise HTTPException(409,'This configured model was removed')
  if qualified and item['state']!='qualified':raise HTTPException(409,'Model is in Training & qualification and cannot run operationally')
  return item
 def candidate(self,model,run_id):
  self.init();family='segmentation_runs' if model['algorithm'] in SEGMENTATION_DETAILS else 'inference_runs'
  with self.database() as db:row=db.execute(f'SELECT payload FROM {family} WHERE id=?',(run_id,)).fetchone()
  if not row:raise HTTPException(404,'Candidate run not found')
  run=json.loads(row['payload'])
  if run['mission_id']!=model['mission_id'] or run['job_id']!=model['job_id']:raise HTTPException(422,'Choose a run belonging to this configured model’s mission and training job')
  if run.get('status') not in ('complete','partial'):raise HTTPException(409,'Candidate has not completed successfully')
  if model['algorithm']==PORTABLE and (
      run.get('run_kind')!='portable_validation' or run.get('workflow_verification_model')!=model['id']
      or run.get('portable_model')!=model['portable_model']):
   raise HTTPException(422,'Qualification requires a validation run bound to this portable package')
  if family=='segmentation_runs':
   if run.get('model')!=model['algorithm']:raise HTTPException(422,'Segmentation adapter mismatch')
   if not run.get('candidates'):raise HTTPException(422,'No real mask proposals in this candidate')
   return run,{'id':model['algorithm'],'status':'complete','metrics':{'status':'unmeasured'},'training':{},'provenance':run.get('provenance')}
  detail=next((m for m in run.get('models',[]) if m['id']==model['algorithm'] and m.get('status')=='complete'),None)
  if not detail:raise HTTPException(422,'That algorithm did not complete this run')
  if model.get('classes') and set(run.get('classes',[]))!=set(model['classes']):raise HTTPException(422,'Candidate class schema differs from this model configuration')
  return run,detail
 def ensure_pretrained(self,mission_id):
  """Native released heads need no local fitting or user qualification."""
  from pretrained_lulc import PRETRAINED
  mission=self.mission(mission_id);job_id=next((j['id'] for j in mission.get('jobs',[])), '')
  self.init()
  for algorithm,spec in {**PRETRAINED,**SEGMENTATION_DETAILS,**DETECTION_DETAILS,**SUPER_RESOLUTION_DETAILS}.items():
   identifier='pretrained-'+hashlib.sha256((mission_id+':'+algorithm).encode()).hexdigest()[:24]
   revision=str(spec.get('revision','bundled-release'))
   release={'id':identifier+':'+revision,'released_at':now(),'source':'bundled pretrained model','checkpoint':None,'files':{},'note':'Native pretrained model; no local training or qualification required. Local accuracy is unmeasured.','evaluation':{'passed':True,'metrics':{'status':'unmeasured'},'scope':'Published pretrained weights; local suitability still depends on the source sensor and required bands.'}}
   item={'id':identifier,'mission_id':mission_id,'job_id':job_id,'name':spec.get('name',algorithm),'algorithm':algorithm,'classes':spec.get('classes',[]),'task':'segmentation' if algorithm in SEGMENTATION_DETAILS else spec.get('task','lulc'),'builtin':True,'state':'qualified','qualification_mode':'pretrained','goals':{},'created_at':now(),'release':release,'history':[]}
   with self.database() as db:db.execute('INSERT OR IGNORE INTO workflow_models VALUES(?,?,?,?)',(identifier,mission_id,json.dumps(item),item['created_at']))

 def evaluate(self,item):
  if item.get('builtin') and item['state']=='qualified':return item['release']['evaluation']
  if not item.get('candidate_run'):return {'passed':False,'reasons':['Choose a completed candidate run'],'metrics':None}
  run,detail=self.candidate(item,item['candidate_run']);metrics=detail.get('metrics',{})
  if item['qualification_mode']=='verified':return {'passed':True,'reasons':[],'metrics':metrics,'scope':'Operator verification of completed inference; local accuracy is unmeasured unless independent metrics are present'}
  reasons=[];goals=item['goals']
  if metrics.get('status')!='measured':reasons.append('Independent validation metrics are required')
  if run.get('validation_protocol','').startswith('Internal spatial holdout'):reasons.append('Use independent human validation; internal training holdout cannot qualify this model')
  if metrics.get('pixels',0)<goals.get('minimum_pixels',50):reasons.append('Insufficient independent validation pixels')
  if item['algorithm']==PORTABLE:
   for label in item['classes']:
    if metrics.get('per_class',{}).get(label,{}).get('support',0)<=0:reasons.append(f'{label}: no independent validation pixels')
  for name,key in [('precision','macro_precision'),('recall','macro_recall'),('f1','macro_f1')]:
   value=metrics.get(key)
   if value is None:
    value=metrics.get('per_class',{}).get('macro avg',{}).get({'precision':'precision','recall':'recall','f1':'f1-score'}[name])
   if value is None or value<goals[name]:reasons.append(f'{name} below goal {goals[name]}')
  for label,thresholds in goals.get('per_class',{}).items():
   record=metrics.get('per_class',{}).get(label,{})
   for name,target in thresholds.items():
    key='f1-score' if name=='f1' else name
    if record.get('support',0)<=0 or record.get(key,-1)<target:reasons.append(f'{label}: {name} goal not met')
  return {'passed':not reasons,'reasons':reasons,'metrics':metrics,'scope':run.get('validation_protocol')}
 def checkpoint_manifest(self,item):
  run,detail=self.candidate(item,item['candidate_run']);training=detail.get('training',{});path=training.get('path')
  if item['algorithm'] in SEGMENTATION_DETAILS:return {'checkpoint':None,'files':{},'provenance':detail.get('provenance')}
  from pretrained_lulc import PRETRAINED
  if not path:
   if item['algorithm'] in PRETRAINED:return {'checkpoint':None,'files':{},'provenance':training}
   raise HTTPException(422,'No fitted checkpoint in this candidate')
  path=Path(path).resolve();root=(self.root()/'results').resolve()
  if not path.is_relative_to(root) or not path.is_file():raise HTTPException(422,'Candidate checkpoint is missing or outside managed results')
  # The original bundle remains immutable; include ensemble member dependencies.
  files={str(path):sha(path)}
  if item['algorithm']==PORTABLE:
   if files[str(path)]!=item['portable_model']['sha256']:
    raise HTTPException(422,'Evaluated portable checkpoint changed. Run independent validation again.')
   return {'checkpoint':str(path),'files':files,'provenance':training}
  import joblib
  record=joblib.load(path)
  linked=record.get('model_path')
  if linked:
   dependency=Path(linked).resolve()
   if not dependency.is_relative_to(self.root().resolve()) or not dependency.exists():raise HTTPException(422,'Released model weights are missing or outside managed local model storage')
   for f in ([dependency] if dependency.is_file() else dependency.rglob('*')):
    if f.is_file():files[str(f)]=sha(f)
  if item['algorithm']=='ensemble':
   import joblib
   record=joblib.load(path)
   for member in record.get('members',[]):
    p=(path.parent/member['path']).resolve()
    if not p.is_relative_to(root) or not p.is_file():raise HTTPException(422,'Ensemble member checkpoint missing')
    files[str(p)]=sha(p)
  return {'checkpoint':str(path),'files':files,'provenance':training}
 def qualify(self,identifier,acknowledged,note):
  item=self.model(identifier)
  if item['state']=='qualified':raise HTTPException(409,'This version is already released')
  if not acknowledged or not note.strip():raise HTTPException(422,'Confirm release and record the verification / qualification note')
  result=self.evaluate(item)
  if not result['passed']:raise HTTPException(422,{'message':'Qualification goals not met','reasons':result['reasons']})
  snapshot=self.checkpoint_manifest(item);run,_=self.candidate(item,item['candidate_run'])
  release={'id':str(uuid4()),'released_at':now(),'source_run':item['candidate_run'],'evaluation':result,'note':note,**snapshot}
  item.setdefault('history',[]).append({'action':'released','release':release});item.update(state='qualified',release=release,classes=run.get('classes',item.get('classes',[])))
  if item['algorithm']==PORTABLE:
   with self.database() as db:
    db.execute('BEGIN IMMEDIATE')
    current=json.loads(db.execute('SELECT payload FROM workflow_models WHERE id=?',(identifier,)).fetchone()['payload'])
    if current['state']!='training' or current.get('candidate_run')!=release['source_run']:
     raise HTTPException(409,'Portable qualification changed during release. Refresh and review the current candidate.')
    db.execute('UPDATE workflow_models SET payload=? WHERE id=?',(json.dumps(item),identifier))
  else:self.put('workflow_models',item)
  self.event('model.qualified','Configured model released for inference',model_id=item['id'],mission_id=item['mission_id'],audit=True,release_id=release['id'],evaluation=result)
  return item
 def retract(self,identifier,reason):
  item=self.model(identifier);item.setdefault('history',[]).append({'action':'returned_to_training','at':now(),'release_id':item.get('release',{}).get('id'),'reason':reason});item.update(state='training',return_reason=reason)
  if item['algorithm']==PORTABLE:item.pop('candidate_run',None)
  self.put('workflow_models',item);self.event('model.retracted','Model returned to training; affected operational work is blocked',model_id=item['id'],mission_id=item['mission_id'],audit=True,reason=reason)
  return item
 def guarded(self,profiles):
  releases={}
  for identifier in profiles:
   item=self.model(identifier,True);release=item['release']
   for path,expected in release.get('files',{}).items():
    if not Path(path).is_file() or sha(path)!=expected:raise HTTPException(409,'Qualified checkpoint changed or is missing; return this model to training')
   releases[identifier]=release['id']
  return releases
 def run_changed(self,run):
  prior=run.get('_workflow_logged')
  state=(run.get('status'),run.get('stage'),run.get('progress'))
  if prior==state:return
  run['_workflow_logged']=state
  self.event('run.progress',run.get('stage') or run.get('status','Run updated'),level='error' if run.get('status')=='failed' else 'info',mission_id=run.get('mission_id'),run_id=run['id'],pipeline_id=run.get('pipeline_id'),status=run.get('status'),progress=run.get('progress'),error=run.get('error'))
