"""Docker coordinator to authenticated native GPU worker; scientific data stay local."""
import json,time,threading
from pathlib import Path
from urllib.parse import urlparse
from internal_http import InternalHTTP

def _local_base(url):
 if not isinstance(url,str) or any(c.isspace() for c in url) or '\\' in url:raise ValueError('Native worker must be a configured local service')
 parsed=urlparse(url)
 if (parsed.scheme!='http' or parsed.hostname not in ('localhost','127.0.0.1','::1','host.docker.internal')
     or parsed.username is not None or parsed.password is not None or '?' in url or '#' in url):
  raise ValueError('Native worker must use a local HTTP endpoint without credentials, query or fragment')
 try:port=parsed.port
 except ValueError as exc:raise ValueError('Invalid native worker port') from exc
 if port is not None and not 1<=port<=65535:raise ValueError('Invalid native worker port')
 return url.rstrip('/')

def connection(data):
 root=Path(data)/'native-gpu';path=root/'connection.json'
 if not path.is_file():return None
 cfg=json.loads(path.read_text());url=_local_base(cfg.get('url',''))
 return url,(root/'worker.token').read_text().strip()

def transport(connection):
 url,token=connection
 return InternalHTTP(_local_base(url),{'X-SpectraScope-Token':token},timeout=8,local_model_bridge=True)

def request(conn,path,body=None):
 if not isinstance(path,str) or not path.startswith('/') or path.startswith('//') or any(c.isspace() for c in path) or any(c in path for c in ('\\','?','#')):raise ValueError('Use a local native-worker operation path')
 return conn.json('POST' if body is not None else 'GET',path,body,maximum=16*1024*1024)

def execute(task,data,reference,image,access,persist):
 from object_detection import execute as local
 identifier=task['id'];conn=connection(data)
 if not conn or access.get('mode') not in ('local_path','download'):return local(task,data,reference,image,access,persist)
 try:
  conn=transport(conn)
  health=request(conn,'/health')
  if 'oriented_detection' not in health.get('supported',[]):raise ValueError('Native worker has no compatible detector')
 except Exception as exc:
  # No work has been accepted. Record the local CPU fallback explicitly.
  task['native_worker_fallback']=str(exc);return local(task,data,reference,image,access,persist)
 try:request(conn,'/work',{'kind':'oriented_detection','task':task,'image':image,'access':access})
 except Exception:
  # A timed-out submission might already be accepted. Never duplicate it on CPU.
  try:request(conn,'/work/'+identifier)
  except Exception:raise ValueError('Native worker could not accept this job. No duplicate CPU run was started; check worker diagnostics and retry.')
 heartbeat_stop=threading.Event()
 def heartbeat():
  while not heartbeat_stop.wait(5):
   try:request(conn,'/work/'+identifier+'/heartbeat',{})
   except Exception:pass
 threading.Thread(target=heartbeat,daemon=True).start()
 acknowledged=-1;started=time.monotonic();last=time.monotonic();failures=0
 try:
  while True:
   try:value=request(conn,'/work/'+identifier);failures=0
   except Exception:
    failures+=1
    if failures>=3:raise ValueError('Native worker lost contact; this run stopped instead of waiting indefinitely. Original inputs and partial artifacts are preserved.')
    time.sleep(.5);continue
   record=value['record'];record['execution_route']='authenticated native local worker';record['native_worker_elapsed_seconds']=round(time.monotonic()-started,2)
   if value['sequence']!=acknowledged:
    # Worker waits here until the shared mission/run control permits continuation.
    persist(record);task.clear();task.update(record);request(conn,'/work/'+identifier+'/ack',{'sequence':value['sequence']});acknowledged=value['sequence'];last=time.monotonic()
   if value['done']:return record
   if time.monotonic()-last>300:raise ValueError('Native model made no progress for five minutes. Inspect its worker log and retry; no success was reported.')
   time.sleep(.2)
 finally:
  heartbeat_stop.set()
  try:request(conn,'/work/'+identifier+'/cancel',{})
  except Exception:pass
