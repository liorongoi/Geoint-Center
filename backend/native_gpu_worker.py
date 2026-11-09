"""Native local GPU worker. Typed tasks only; no arbitrary commands or generated code."""
import argparse,copy,json,os,re,secrets,threading,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal
from fastapi import FastAPI,Header,HTTPException
from pydantic import BaseModel

class Request(BaseModel):
 kind:Literal['oriented_detection']
 task:dict
 image:dict
 access:dict

class Cancelled(BaseException):pass

def create_app(data,token,imports=None):
 data=Path(data).resolve();imports=Path(imports).resolve() if imports else None;app=FastAPI();jobs={};pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='native-gpu');lock=threading.RLock()
 def auth(value):
  if not value or not secrets.compare_digest(value,token):raise HTTPException(401,'Local worker authentication required')
 def translate(v):
  if isinstance(v,dict):return {k:translate(x) for k,x in v.items()}
  if isinstance(v,list):return [translate(x) for x in v]
  if isinstance(v,str):
   for before,after in [('/app/data/',data),('/app/imports/',imports)]:
    if after and v.startswith(before):return str(after/v[len(before):])
  return v
 @app.get('/health')
 async def health(x_spectrascope_token:str|None=Header(default=None)):
  auth(x_spectrascope_token);import platform
  return {'ok':True,'protocol':1,'platform':platform.system(),'architecture':platform.machine(),'supported':['oriented_detection'],'active':sum(not j['done'] for j in jobs.values())}
 @app.post('/work')
 async def work(request:Request,x_spectrascope_token:str|None=Header(default=None)):
  auth(x_spectrascope_token);identifier=request.task.get('id','')
  if not re.fullmatch(r'[a-f0-9-]{36}',identifier):raise HTTPException(422,'Use a generated run UUID')
  if request.task.get('model')!='oriented_rcnn_dota':raise HTTPException(422,'Unsupported native model')
  access=translate(request.access)
  from raster_pipeline import source_assets
  import offline_policy
  if access.get('mode') not in ('local_path','download'):raise HTTPException(422,'Native worker requires local science rasters; download streaming assets first')
  try:
   sources=source_assets(access)
   if not sources:raise ValueError('No local science rasters supplied')
   for _,source in sources:
    p=Path(source).resolve()
    if not p.is_relative_to(data) and not (imports and p.is_relative_to(imports)):raise HTTPException(422,'Raster lies outside the shared local imagery roots')
    offline_policy.raster_driver(str(p))
  except (ValueError,OSError,KeyError) as exc:raise HTTPException(422,'Local raster validation failed: '+str(exc)) from exc
  with lock:
   finished=[key for key,value in jobs.items() if value['done']]
   for key in finished[:-50]:jobs.pop(key,None)
   if identifier in jobs:return {'id':identifier,'existing':True}
   if sum(not j['done'] for j in jobs.values())>=4:raise HTTPException(409,'Native GPU worker queue is full; retry after active jobs finish')
   state={'id':identifier,'record':copy.deepcopy(request.task),'sequence':0,'acknowledged':0,'done':False,'cancel':False,'heartbeat':time.monotonic(),'condition':threading.Condition()};jobs[identifier]=state
  def execute():
   def progress(record):
    with state['condition']:
     state['record']=copy.deepcopy(record);state['sequence']+=1;state['condition'].notify_all()
     while state['acknowledged']<state['sequence']:
      if state['cancel']:raise Cancelled()
      if time.monotonic()-state['heartbeat']>120:raise TimeoutError('Coordinator did not acknowledge the safe checkpoint and its heartbeat stopped for two minutes')
      state['condition'].wait(.5)
   try:
    from object_detection import execute as detect
    from resource_admission import accelerator, analysis_gib, reserve
    def waiting():
     with state['condition']:
      if state['cancel']:raise Cancelled()
      if time.monotonic()-state['heartbeat']>120:raise TimeoutError('Coordinator stopped while waiting for host resources')
      state['record'].update(stage='Waiting for shared host resources')
    with reserve('Native GPU worker',analysis_gib(),gpu=accelerator(),checkpoint=waiting):
     result=detect(copy.deepcopy(request.task),data,None,request.image,access,progress)
    with state['condition']:state['record']=result
   except Cancelled:
    with state['condition']:state['record'].update(status='cancelled',stage='Stopped at native worker checkpoint; completed artifacts preserved')
   except BaseException as exc:
    with state['condition']:state['record'].update(status='failed',error=str(exc),stage='Native worker failed')
   finally:
    with state['condition']:state['done']=True;state['condition'].notify_all()
  pool.submit(execute);return {'id':identifier}
 @app.get('/work/{identifier}')
 async def status(identifier:str,x_spectrascope_token:str|None=Header(default=None)):
  auth(x_spectrascope_token)
  if identifier not in jobs:raise HTTPException(404,'Native worker no longer has this job; start a new run')
  state=jobs[identifier]
  with state['condition']:return {'record':state['record'],'sequence':state['sequence'],'done':state['done']}
 @app.post('/work/{identifier}/{action}')
 async def control(identifier:str,action:str,body:dict,x_spectrascope_token:str|None=Header(default=None)):
  auth(x_spectrascope_token)
  if identifier not in jobs:raise HTTPException(404,'Native job not found')
  state=jobs[identifier]
  with state['condition']:
   if action=='ack':state['acknowledged']=max(state['acknowledged'],min(int(body.get('sequence',0)),state['sequence']))
   elif action=='heartbeat':state['heartbeat']=time.monotonic()
   elif action=='cancel':state['cancel']=True
   else:raise HTTPException(422,'Unknown worker control')
   state['condition'].notify_all()
  return {'ok':True}
 return app

def main():
 from resource_admission import configure_native_host
 configure_native_host()
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--imports',type=Path);p.add_argument('--port',type=int,default=18767);args=p.parse_args();folder=args.data.resolve()/'native-gpu';folder.mkdir(parents=True,exist_ok=True);token_file=folder/'worker.token'
 if not token_file.exists():token_file.write_text(secrets.token_hex(32));token_file.chmod(0o600)
 import data_policy
 os.environ.update(SPECTRASCOPE_OFFLINE='1',SPECTRASCOPE_MODELS_LOCAL_ONLY='1',SPECTRASCOPE_DATA_DIR=str(args.data.resolve()),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PROJ_NETWORK='OFF')
 from native_network_boundary import enforce_mac_loopback
 enforce_mac_loopback()
 import offline_policy;offline_policy.install()
 import uvicorn
 print('Native local GPU worker, loopback only. GPU compatibility is checked by each model. No package/model downloads.',flush=True)
 uvicorn.run(create_app(args.data,token_file.read_text().strip(),args.imports),host='127.0.0.1',port=args.port)
if __name__=='__main__':main()
