"""Current-user internal worker; isolated scientific processes, no runtime downloads."""
import argparse,copy,http.client,json,os,re,shutil,ssl,subprocess,sys,threading,time
from pathlib import Path
from internal_http import InternalHTTP

class Stopped(BaseException):pass

def child(request_path):
 # The model process has no networking at all, even when its parent uses LAN.
 os.environ.update(SPECTRASCOPE_OFFLINE='1',SPECTRASCOPE_MODELS_LOCAL_ONLY='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PROJ_NETWORK='OFF',HF_HUB_DISABLE_TELEMETRY='1',WANDB_DISABLED='true')
 import offline_policy;offline_policy.install()
 from distributed_execution import signature,replace_paths
 from job_runner import execute_job
 info=json.loads(Path(request_path).read_text());bundle=info['bundle'];request=bundle['request'];adapter=request['run']['requested_models'][0]
 if signature(adapter)!=bundle['signature']:raise ValueError('Worker code or native package versions differ from the coordinator; update matching offline application/runtime components')
 context=Path(request_path).parent;data=context/'data';data.mkdir(exist_ok=True);progress_file=context/'progress.json';control=context/'control.json'
 expected_parent=info['parent_pid']
 if os.getppid()!=expected_parent:raise ValueError('Worker supervisor exited before native computation started')
 def supervisor_watch():
  while True:
   if os.getppid()!=expected_parent or (control.exists() and time.time()-control.stat().st_mtime>120):os._exit(86)
   time.sleep(1)
 threading.Thread(target=supervisor_watch,daemon=True).start()
 bindings={original:str(Path(info['objects'])/oid) for original,oid in bundle['bindings'].items()};request=replace_paths(request,bindings)
 run=request['run']
 if not re.fullmatch(r'[a-f0-9-]{36}',run['id']):raise ValueError('Worker run must use a generated UUID')
 out=data/'results'/run['id'];out.mkdir(parents=True)
 if adapter=='dynamicworld':request['options']['initial_checkpoints']={}
 for model,path in list(run.get('qualified_checkpoints',{}).items()):
  if path:
   target=out/(model+'.joblib');shutil.copyfile(path,target);run['qualified_checkpoints'][model]=str(target)
   for release in run.get('qualification',{}).values():
    if path in release.get('files',{}):release['files'][str(target)]=release['files'].pop(path)
 def persist(record):
  while True:
   command=json.loads(control.read_text()).get('action','resume') if control.exists() else 'resume'
   if command=='stop':raise Stopped()
   snapshot={**record,'status':'paused' if command=='pause' else record.get('status')};temp=progress_file.with_suffix('.part');temp.write_text(json.dumps(snapshot,default=str));temp.replace(progress_file)
   if command!='pause':return
   time.sleep(.1)
 try:
  from resource_admission import reserve
  def waiting():
   persist({'status':'running','stage':'Waiting for shared host resources','progress':0})
  with reserve('Enrolled worker', info['ram_gib'], gpu=info['device_kind']!='cpu', checkpoint=waiting):
   if 'unet_gradient' in request['options']:
    if adapter!='unet' or request['images'] or request['labels'] or request['access']:
     raise ValueError('Invalid synchronized U-Net partition request')
    from unet_pool import work_partition
    result=work_partition(run,request['options']['unet_gradient'],out,persist,info['device_kind'])
   else:
    result=execute_job(run,request['images'],request['access'],request['labels'],data,persist,**request['options'])
  if result['status']!='complete':raise ValueError(str(result.get('error') or result.get('stage')))
  result['_worker_result_root']=str(out);(out/'report.json').write_text(json.dumps(result,default=str));(context/'completed.json').write_text(json.dumps({'status':'complete','directory':str(out)}))
 except Stopped:(context/'completed.json').write_text(json.dumps({'status':'cancelled'}))

class Worker:
 def __init__(self,url,identifier,key_file,cache,ca_file=None):
  self.id=identifier;self.cache=Path(cache).resolve()
  from data_policy import protect_output
  protect_output(self.cache,json.loads(os.getenv('GEO_SERVICE_SOURCE_ROOTS','[]')))
  marker=self.cache/'worker-owner.json'
  if self.cache.exists() and any(self.cache.iterdir()) and not marker.exists():raise ValueError('Use a dedicated empty private worker cache; supplier/source folders are never used as scratch storage')
  self.cache.mkdir(parents=True,exist_ok=True)
  if marker.exists() and json.loads(marker.read_text()).get('worker')!=identifier:raise ValueError('Cache belongs to another worker identity')
  if not marker.exists():marker.write_text(json.dumps({'worker':identifier,'private_worker_storage':True}))
  key=Path(key_file).read_text().strip()
  if not re.fullmatch(r'[a-f0-9]{64}',key):raise ValueError('Use the enrolled worker’s private key file')
  self.client=InternalHTTP(url,{'X-Geo-Worker-Key':key},ca_file);self.base='/api/services/pool/workers/'+self.id
 def once(self):
  from agent_services import file_hash
  from execution_policy import available_ram_bytes
  available=available_ram_bytes()
  # Keep capacity for the host and transfer buffers before reserving a lease.
  task=self.client.json('POST',self.base+'/claim',{'available_ram_gib':max(0,(available or 0)/1024**3-1)})['task']
  if task is None:return False
  identifier=task.get('id');attempt=task.get('attempt')
  if (not isinstance(identifier,str) or not re.fullmatch(r'[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}',identifier)
      or type(attempt) is not int or not 1<=attempt<=10):
   raise ValueError('Coordinator returned an invalid worker task identity or attempt')
  prefix=self.base+'/tasks/'+identifier;client=copy.copy(self.client);client.headers={**client.headers,'X-Geo-Lease':task['lease_token']};context=self.cache/'work'/identifier/str(attempt);context.mkdir(parents=True,exist_ok=False)
  objects=self.cache/'objects';objects.mkdir(exist_ok=True);state={'command':'resume','progress':0,'stage':'Downloading verified input bundle','paused':False,'error':None};done=threading.Event();lock=threading.RLock();process=None;log=None
  heartbeat_client=copy.copy(client);heartbeat_client.timeout=min(5,task['lease_seconds']/2)
  def report(terminal=None):
   with lock:body={'lease_token':task['lease_token'],'progress':state['progress'],'stage':state['stage'][:500],'paused':state['paused'],'terminal':terminal}
   result=heartbeat_client.json('POST',prefix,body)
   with lock:state['command']=result['control']
  def heartbeat():
   deadline=time.monotonic()+task['lease_seconds']
   while not done.wait(.5):
    try:
     report()
     deadline=time.monotonic()+task['lease_seconds']
    except Exception as exc:
     temporary=isinstance(exc,(OSError,http.client.RemoteDisconnected)) and not isinstance(exc,ssl.SSLError)
     if temporary and time.monotonic()<deadline:
      continue
     with lock:state['error']=str(exc)
     return
  thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
  def checkpoint():
   while True:
    with lock:command=state['command'];error=state['error']
    if error:raise ValueError('Worker lost its authenticated lease; isolated computation stopped')
    if command=='stop':raise Stopped()
    if command!='pause':return
    with lock:state['paused']=True;state['stage']='Paused while transferring task inputs'
    time.sleep(.1)
  try:
   report();manifest=task['payload']['input_manifest'];bundle_path=context/'bundle.json';client.download(prefix+'/inputs/bundle',bundle_path,manifest['sha256'],2*1024**2);bundle=json.loads(bundle_path.read_text())
   from distributed_execution import signature
   if signature(task['adapter'])!=bundle['signature']:raise ValueError('Worker code/native packages differ from this task; no model was started')
   for oid,info in bundle['files'].items():
    checkpoint()
    if not re.fullmatch(r'[a-f0-9]{64}\.[a-z0-9]+',oid):raise ValueError('Invalid worker input object identity')
    target=objects/oid
    if not target.is_file() or file_hash(target)!=info['sha256']:client.download(prefix+'/inputs/'+oid,target,info['sha256'],info['bytes'],checkpoint)
   available=available_ram_bytes()
   if available is None or available<(task['requirements']['ram_gib']+1)*1024**3:raise ValueError('Worker lost required RAM headroom after claiming this task; no scientific bands were dropped')
   request=context/'request.json';request.write_text(json.dumps({'bundle':bundle,'objects':str(objects),'parent_pid':os.getpid(),'ram_gib':task['requirements']['ram_gib'],'device_kind':task['device_kind']}));control=context/'control.json';control.write_text('{"action":"resume"}')
   env={**os.environ,'SPECTRASCOPE_DEVICE':'cpu' if task['device_kind']=='cpu' else 'auto','PYTHONUNBUFFERED':'1'}
   if task['device_kind']=='cuda':env['CUDA_VISIBLE_DEVICES']=task['device'].removeprefix('cuda:')
   log=(context/'native.log').open('wb');process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--child',str(request)],env=env,stdout=log,stderr=subprocess.STDOUT)
   while process.poll() is None:
    with lock:
     if state['error']:raise ValueError('Worker lease communication failed')
     command=state['command']
    if command=='stop':raise Stopped()
    temp=control.with_suffix('.part');temp.write_text(json.dumps({'action':command}));temp.replace(control)
    p=context/'progress.json'
    if p.exists():
     record=json.loads(p.read_text())
     with lock:state.update(progress=min(98,record.get('progress',0)),stage=record.get('stage','Processing'),paused=record.get('status')=='paused')
    time.sleep(.1)
   log.close()
   completed=context/'completed.json'
   if process.returncode or not completed.exists():raise ValueError('Native worker process failed; inspect its private native.log. No model/package download was attempted.')
   result=json.loads(completed.read_text())
   if result['status']=='cancelled':raise Stopped()
   checkpoint();out=Path(result['directory']);files=[]
   for p in sorted(out.rglob('*')):
    if not p.is_file():continue
    if p.is_symlink():raise ValueError('Worker output contains a symbolic link')
    checkpoint();sha=file_hash(p);client.json('PUT',prefix+'/blobs/'+sha,file=p);files.append({'path':p.relative_to(out).as_posix(),'sha256':sha,'bytes':p.stat().st_size})
   checkpoint();inventory={'task_id':identifier,'attempt':task['attempt'],'input_sha256':manifest['sha256'],'model_release':task['payload'].get('model_release'),'training_version':task['payload'].get('training_version'),'files':files}
   # Keep the lease alive while the coordinator checks and copies large outputs.
   # Completion is still fenced by the coordinator's lease/control transaction.
   with lock:state.update(progress=99,stage='Verifying and publishing received artifacts')
   final_client=copy.copy(client);final_client.timeout=180
   final_client.json('POST',prefix+'/finish',inventory)
   print(json.dumps({'task':identifier,'status':'uploaded for coordinator verification','device':task['device']}),flush=True);return True
  except Stopped:
   try:
    with lock:state['stage']='Stopped; private partial artifacts retained'
    report('cancelled')
   except Exception:pass
   return True
  except Exception as exc:
   (context/'failure.json').write_text(json.dumps({'error':str(exc),'task_id':identifier}))
   try:
    with lock:state['stage']='Worker failed; inspect private worker log and task diagnostics'
    report('failed')
   except Exception:pass
   raise RuntimeError('Worker task failed; inspect its private failure.json/native.log') from exc
  finally:
   done.set()
   if process and process.poll() is None:
    process.terminate()
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
   if log is not None:log.close()
   thread.join(timeout=3)

def main():
 from resource_admission import configure_native_host
 configure_native_host()
 p=argparse.ArgumentParser();p.add_argument('--child',type=Path);p.add_argument('--coordinator');p.add_argument('--worker');p.add_argument('--key-file',type=Path);p.add_argument('--cache',type=Path);p.add_argument('--ca-file',type=Path);p.add_argument('--once',action='store_true');p.add_argument('--enroll',action='store_true');p.add_argument('--enrollment-file',type=Path);p.add_argument('--name',default='Analysis worker');p.add_argument('--config',type=Path);p.add_argument('--models',type=Path);args=p.parse_args()
 if args.child:
  from native_network_boundary import enforce_mac_loopback
  enforce_mac_loopback()
  return child(args.child)
 if args.config:
  configuration=json.loads(args.config.read_text())
  if set(configuration)-{'coordinator','worker','key_file','cache','ca_file','models'}:raise SystemExit('Worker configuration contains unsupported fields; commands/code are never accepted')
  for key,value in configuration.items():setattr(args,key,Path(value) if value and key in ('key_file','cache','ca_file','models') else value)
 if args.models:
  if not args.models.is_dir():raise SystemExit('Local model-pack folder is unavailable; no download will be attempted')
  os.environ['SPECTRASCOPE_BUNDLED_MODELS']=str(args.models.resolve())
 if args.enroll:
  if not all((args.enrollment_file,args.cache)):p.error('Enrollment needs the one-time enrollment file and an empty private cache')
  invitation=json.loads(args.enrollment_file.read_text())
  if set(invitation)-{'coordinator','invitation','expires_at','name','ca_file'}:raise SystemExit('Unsupported enrollment-file fields')
  args.coordinator=invitation['coordinator'];args.name=invitation.get('name',args.name)
  if invitation.get('ca_file'):args.ca_file=Path(invitation['ca_file'])
  if not os.getenv('SPECTRASCOPE_BUNDLED_MODELS'):
   for base in (Path(sys.executable).resolve().parent.parent,Path(sys.executable).resolve().parent.parent.parent):
    if (base/'models').is_dir():os.environ['SPECTRASCOPE_BUNDLED_MODELS']=str(base/'models');break
  from runtime_profiles import native_hardware,docker_memory_plan
  from execution_policy import torch_device
  import importlib.util
  hardware=native_hardware(detect_gpu=False);ram=hardware.get('ram_gb')
  if not ram or ram<12:raise SystemExit('Detect at least 12 GiB before enrolling a worker; no unsafe RAM default is assumed')
  budget=docker_memory_plan(ram)['app_gb'];devices=[{'id':'cpu','kind':'cpu'}];capabilities=['random_forest','svm']
  if importlib.util.find_spec('xgboost'):capabilities.append('xgboost')
  if importlib.util.find_spec('torch'):
   capabilities.append('unet');device=torch_device()
   if device.type=='cuda':
    import torch
    devices.append({'id':str(device),'kind':'cuda','vram_gib':round(torch.cuda.get_device_properties(device).total_memory/1024**3,2)})
   elif device.type=='mps':devices.append({'id':'mps','kind':'mps','vram_gib':budget})
  if importlib.util.find_spec('tensorflow') and os.getenv('SPECTRASCOPE_BUNDLED_MODELS') and (Path(os.environ['SPECTRASCOPE_BUNDLED_MODELS'])/'dynamicworld').is_dir():capabilities.append('dynamicworld')
  from data_policy import protect_output
  protect_output(args.cache,json.loads(os.getenv('GEO_SERVICE_SOURCE_ROOTS','[]')))
  if args.cache.exists() and any(args.cache.iterdir()):raise SystemExit('Choose an empty private worker cache. Existing files will not be overwritten.')
  http=InternalHTTP(args.coordinator,{'X-Geo-Worker-Invitation':invitation['invitation']},args.ca_file);record=http.json('POST','/api/services/pool/enroll',{'name':args.name,'ram_gib':budget,'capabilities':capabilities,'devices':devices})
  args.cache.mkdir(parents=True,exist_ok=True);key_path=args.cache/'worker.key'
  with key_path.open('x') as stream:stream.write(record.pop('access_key'))
  key_path.chmod(0o600);configuration={'coordinator':args.coordinator,'worker':record['id'],'key_file':str(key_path.resolve()),'cache':str(args.cache.resolve()),'ca_file':str(args.ca_file.resolve()) if args.ca_file else None}
  (args.cache/'worker-owner.json').write_text(json.dumps({'worker':record['id'],'private_worker_storage':True}))
  configuration['models']=os.getenv('SPECTRASCOPE_BUNDLED_MODELS')
  (args.cache/'worker.json').write_text(json.dumps(configuration,indent=2));print(json.dumps({'enrolled':record['id'],'ram_budget_gib':budget,'devices':devices,'configuration':str(args.cache/'worker.json'),'note':'One-time invitation consumed; no services/admin key was used'}));return
 if not all((args.coordinator,args.worker,args.key_file,args.cache)):p.error('Provide coordinator, worker, key-file and private cache directory')
 worker=Worker(args.coordinator,args.worker,args.key_file,args.cache,args.ca_file)
 while True:
  try:worker.once()
  except Exception as exc:
   if args.once:raise
   print(json.dumps({'worker':args.worker,'status':'Task/connection needs attention; private diagnostics retained','error_type':type(exc).__name__}),flush=True)
   if isinstance(exc,ValueError) and 'HTTP 401' in str(exc):raise SystemExit('Worker enrollment rejected or revoked')
   time.sleep(2)
  if args.once:return
  time.sleep(1)
if __name__=='__main__':main()
