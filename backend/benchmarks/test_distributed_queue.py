"""Durable pool protocol: no live LAN or multi-node-training claim."""
import hashlib,json,sqlite3,tempfile,unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from distributed_queue import Queue

class PoolTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.now=1000
  @contextmanager
  def db():
   conn=sqlite3.connect(self.root/'pool.sqlite',timeout=30);conn.row_factory=sqlite3.Row
   try:
    with conn:yield conn
   finally:conn.close()
  self.db=db;self.pool=Queue(db,lambda:self.now)
  self.cpu=self.pool.enroll('CPU',list(('random_forest','unet')),12,[{'id':'cpu','kind':'cpu'}]);self.gpu=self.pool.enroll('GPU',['unet'],40,[{'id':'cpu','kind':'cpu'},{'id':'cuda:0','kind':'cuda','vram_gib':16}])
 def tearDown(self):self.tmp.cleanup()
 def task(self,key='first',adapter='random_forest',kind='inference',**kwargs):
  payload={'run_id':'run','input_manifest':{'sha256':'a'*64},'model_release':'release-1'}
  if kind=='training':payload.update(training_version='candidate-1',validation_manifest={'sha256':'b'*64})
  return self.pool.enqueue(adapter,kind,payload,key,**kwargs)
 def claim(self,worker):return self.pool.claim(worker['id'],worker['access_key'])
 def test_live_capacity_delays_lease_and_cannot_expand_enrollment(self):
  heavy=self.task('heavy',ram_gib=16)
  light=self.task('light',ram_gib=3)
  def claim(ram):
   return self.pool.claim(self.cpu['id'],self.cpu['access_key'],available_ram_gib=ram)
  self.assertIsNone(claim(2))
  self.assertEqual(claim(32)['id'],light['id'])
  self.assertIsNone(claim(32))
  task=next(t for t in self.pool.snapshot()['tasks'] if t['id']==heavy['id'])
  self.assertEqual((task['status'],task['attempt']),('queued',0))
 def update(self,worker,task,**kwargs):return self.pool.update(worker['id'],worker['access_key'],task['id'],task['lease_token'],50,'Working',**kwargs)
 def test_exactly_one_lease_under_concurrent_claims_and_idempotency(self):
  task=self.task();self.assertEqual(self.task()['id'],task['id'])
  second=self.pool.enroll('CPU2',['random_forest'],16,[{'id':'cpu','kind':'cpu'}])
  with ThreadPoolExecutor(2) as executor:results=list(executor.map(self.claim,[self.cpu,second]))
  self.assertEqual(sum(r is not None for r in results),1)
  self.assertNotIn('lease_token_hash',next(r for r in results if r))
 def test_device_ram_matching_and_cpu_gpu_jobs(self):
  self.task('neural','unet',device='cuda',vram_gib=12,ram_gib=20)
  self.assertIsNone(self.claim(self.cpu));neural=self.claim(self.gpu);self.assertEqual(neural['device'],'cuda:0')
  cpu=self.task('classical');leased=self.claim(self.cpu);self.assertEqual(leased['id'],cpu['id']);self.assertEqual(leased['device_kind'],'cpu')
  with self.assertRaisesRegex(ValueError,'CPU'):self.task('bad','random_forest',device='cuda')
 def test_independent_training_requires_candidate_and_validation(self):
  train=self.task('training','random_forest','training');leased=self.claim(self.cpu);self.assertEqual(leased['kind'],'training')
  with self.assertRaises(ValueError):self.pool.enqueue('random_forest','training',{'input_manifest':{'sha256':'a'*64}},'missing-split')
  with self.assertRaises(ValueError):self.task('unsupported','dynamicworld','training')
 def test_expired_lease_requeues_and_rejects_old_owner(self):
  self.task();old=self.claim(self.cpu);self.now+=61
  new=self.claim(self.cpu);self.assertEqual(new['id'],old['id']);self.assertEqual(new['attempt'],2);self.assertNotEqual(new['lease_token'],old['lease_token'])
  with self.assertRaises(PermissionError):self.update(self.cpu,old,terminal='complete',result={'manifest_sha256':'b'*64})
 def test_pause_resume_stop_and_terminal_states(self):
  task=self.task();self.pool.control(task['id'],'pause');self.assertIsNone(self.claim(self.cpu));self.pool.control(task['id'],'resume');leased=self.claim(self.cpu)
  self.pool.control(task['id'],'pause');progress=self.update(self.cpu,leased);self.assertEqual(progress['control'],'pause')
  with self.assertRaises(ValueError):self.update(self.cpu,leased,terminal='complete',result={'manifest_sha256':'b'*64})
  self.pool.control(task['id'],'resume');self.pool.control(task['id'],'stop');stopped=self.update(self.cpu,leased,terminal='cancelled');self.assertEqual(stopped['status'],'cancelled')
  with self.assertRaises(ValueError):self.pool.control(task['id'],'resume')
 def test_revocation_and_restart_preserve_queue(self):
  task=self.task();leased=self.claim(self.cpu);self.pool.revoke(self.cpu['id'])
  with self.assertRaises(PermissionError):self.update(self.cpu,leased)
  restarted=Queue(self.db,lambda:self.now);snapshot=restarted.snapshot();self.assertEqual(snapshot['tasks'][0]['status'],'queued');self.assertEqual(snapshot['tasks'][0]['id'],task['id']);self.assertNotIn('secret_hash',json.dumps(snapshot))
 def test_result_manifest_is_verified_before_completion(self):
  self.task();leased=self.claim(self.cpu);folder=self.root/'results'/leased['id']/str(leased['attempt']);folder.mkdir(parents=True);artifact=folder/'result.tif';artifact.write_bytes(b'hashed result fixture, not a real raster')
  sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
  manifest=folder/'result.json';manifest.write_text(json.dumps({'task_id':leased['id'],'attempt':leased['attempt'],'input_sha256':'a'*64,'model_release':'release-1','files':[{'path':artifact.name,'bytes':artifact.stat().st_size,'sha256':sha(artifact)}]}))
  finished=self.update(self.cpu,leased,terminal='complete',result={'manifest_sha256':sha(manifest)});self.assertEqual(finished['status'],'awaiting_verification')
  artifact.write_bytes(b'changed')
  with self.assertRaisesRegex(ValueError,'verification'):self.pool.verify_result(leased['id'],self.root/'results',self.root/'verified')
  artifact.write_bytes(b'hashed result fixture, not a real raster');verified=self.pool.verify_result(leased['id'],self.root/'results',self.root/'verified');self.assertEqual(verified['status'],'complete');self.assertEqual(verified['verification']['scientific_accuracy'],'not implied')
  artifact.write_bytes(b'worker scratch changed after publication');self.assertEqual((Path(verified['verification']['directory'])/'result.tif').read_bytes(),b'hashed result fixture, not a real raster')
 def test_arbitrary_commands_and_missing_auth_are_rejected(self):
  with self.assertRaises(PermissionError):self.pool.claim(self.cpu['id'],'wrong')
  with self.assertRaises(ValueError):self.pool.enqueue('random_forest','inference',{'input_manifest':{'sha256':'a'*64},'command':'rm -rf /'},'unsafe')
 def test_invitation_is_scoped_single_use_and_expires(self):
  invite=self.pool.invitation(['random_forest'],16,60)
  with self.assertRaises(PermissionError):self.pool.redeem(invite['invitation'],'Wrong scope',['unet'],8,[{'id':'cpu','kind':'cpu'}])
  worker=self.pool.redeem(invite['invitation'],'Approved',['random_forest'],8,[{'id':'cpu','kind':'cpu'}]);self.assertIn('access_key',worker)
  with self.assertRaises(PermissionError):self.pool.redeem(invite['invitation'],'Reused',['random_forest'],8,[{'id':'cpu','kind':'cpu'}])
  expired=self.pool.invitation(['random_forest'],16,60);self.now+=61
  with self.assertRaises(PermissionError):self.pool.redeem(expired['invitation'],'Expired',['random_forest'],8,[{'id':'cpu','kind':'cpu'}])
  with self.db() as db:stored=''.join(row['payload'] for row in db.execute('SELECT payload FROM pool_invites'))
  self.assertNotIn(invite['invitation'],stored)
 def test_paused_expired_task_is_recovered_once_without_duplicate_history(self):
  task=self.task();leased=self.claim(self.cpu);self.pool.control(task['id'],'pause');self.now+=61
  one=self.pool.snapshot()['tasks'][0];two=self.pool.snapshot()['tasks'][0]
  self.assertEqual(one['status'],'paused');self.assertEqual(len(one['history']),1);self.assertEqual(two['history'],one['history'])
  self.pool.control(task['id'],'resume');self.assertEqual(self.claim(self.cpu)['attempt'],2)
 def test_real_pool_api_authentication_and_private_transport(self):
  import app as backend
  from agent_services import access_key
  from fastapi.testclient import TestClient
  from unittest.mock import patch
  with patch.object(backend,'DATA_DIR',self.root),patch.object(backend,'DATABASE_PATH',self.root/'api.sqlite'):
   backend._init_database();admin=TestClient(backend.app,client=('127.0.0.1',50000));headers={'X-Geo-Services-Key':access_key(self.root)}
   self.assertEqual(admin.get('/api/services/pool').status_code,401)
   response=admin.post('/api/services/pool/workers',headers=headers,json={'name':'API CPU','ram_gib':12,'capabilities':['random_forest'],'devices':[{'id':'cpu','kind':'cpu'}]});self.assertEqual(response.status_code,200,response.text);worker=response.json()
   pool=Queue(backend._database);task=pool.enqueue('random_forest','inference',{'input_manifest':{'sha256':'a'*64},'model_release':'release'},'api-task')
   path='/api/services/pool/workers/'+worker['id']+'/claim';self.assertEqual(admin.post(path,headers={'X-Geo-Worker-Key':'wrong'}).status_code,401)
   leased=admin.post(path,headers={'X-Geo-Worker-Key':worker['access_key']});self.assertEqual(leased.status_code,200,leased.text);self.assertEqual(leased.json()['task']['id'],task['id'])
   self.assertNotIn(worker['access_key'],admin.get('/api/services/pool',headers=headers).text)
   remote=TestClient(backend.app,client=('10.0.0.5',50000));self.assertEqual(remote.post(path,headers={'X-Geo-Worker-Key':worker['access_key']}).status_code,403)
   self.assertEqual(admin.post('/api/services/pool/workers/'+worker['id']+'/revoke',headers=headers).status_code,200)
   self.assertEqual(admin.post(path,headers={'X-Geo-Worker-Key':worker['access_key']}).status_code,401)
 def test_finish_resumes_exact_directory_after_interrupted_commit(self):
  import app as backend
  from fastapi.testclient import TestClient
  from unittest.mock import patch
  data=b'fixture output';identity=hashlib.sha256(data).hexdigest()
  with patch.object(backend,'DATA_DIR',self.root),patch.object(backend,'DATABASE_PATH',self.root/'recovery.sqlite'):
   backend._init_database();pool=Queue(backend._database);worker=pool.enroll('Recovery worker',['random_forest'],12,[{'id':'cpu','kind':'cpu'}])
   task=pool.enqueue('random_forest','inference',{'input_manifest':{'sha256':'a'*64},'model_release':'release-1'},'recover-api');lease=pool.claim(worker['id'],worker['access_key'])
   prefix='/api/services/pool/workers/'+worker['id']+'/tasks/'+task['id'];headers={'X-Geo-Worker-Key':worker['access_key'],'X-Geo-Lease':lease['lease_token']};client=TestClient(backend.app,client=('127.0.0.1',50000),raise_server_exceptions=False)
   uploaded=client.put(prefix+'/blobs/'+identity,headers=headers,content=data);self.assertEqual(uploaded.status_code,200,uploaded.text)
   manifest={'task_id':task['id'],'attempt':lease['attempt'],'input_sha256':'a'*64,'model_release':'release-1','files':[{'path':'result.tif','bytes':len(data),'sha256':identity}]}
   with patch.object(Queue,'update',side_effect=OSError('Simulated database failure after file commit')):
    self.assertEqual(client.post(prefix+'/finish',headers=headers,json=manifest).status_code,500)
   self.assertEqual(client.post(prefix+'/finish',headers=headers,json={**manifest,'model_release':'different'}).status_code,422)
   finished=client.post(prefix+'/finish',headers=headers,json=manifest);self.assertEqual(finished.status_code,200,finished.text)
   committed=pool.verify_result(task['id'],self.root/'pool'/'received',self.root/'pool'/'verified')
   self.assertEqual((Path(committed['verification']['directory'])/'result.tif').read_bytes(),data)

if __name__=='__main__':unittest.main()
