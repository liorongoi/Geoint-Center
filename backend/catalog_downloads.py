"""Explicit background downloads of complete catalog science assets to local folders."""
import asyncio,hashlib,json,os,re,tempfile,time
from ipaddress import ip_address
import httpx
from pathlib import Path
from uuid import uuid4
from urllib.parse import urlparse
from public_raster_transport import bounded_download, complete_download
from fastapi import HTTPException, Request
from pydantic import BaseModel,Field
from starlette.concurrency import run_in_threadpool
import offline_policy
import earthdata_hls
import copernicus_download as cdse

class DownloadRequest(BaseModel):
 image:dict
 directory:str=Field(default='',max_length=2000)


def register(backend):
 app=backend.app;tasks={};safe_slots=asyncio.Semaphore(2)
 def init():
  backend._init_database()
  with backend._database() as db:db.execute('CREATE TABLE IF NOT EXISTS catalog_downloads(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
 def save(item,control_change=False):
  init()
  with backend._database() as db:
   db.execute('BEGIN IMMEDIATE')
   prior=db.execute('SELECT payload FROM catalog_downloads WHERE id=?',(item['id'],)).fetchone()
   if prior and not control_change:item['control']=json.loads(prior['payload']).get('control','resume')
   db.execute('INSERT OR REPLACE INTO catalog_downloads VALUES(?,?)',(item['id'],json.dumps(item)))
 def load(identifier):
  init()
  with backend._database() as db:row=db.execute('SELECT payload FROM catalog_downloads WHERE id=?',(identifier,)).fetchone()
  if not row:raise HTTPException(404,'Download not found')
  return json.loads(row['payload'])
 def sha(path):
  h=hashlib.sha256()
  with path.open('rb') as f:
   for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
  return h.hexdigest()
 def writable_output(path):
  from data_policy import protect_output
  roots=[w['path'] for w in backend.search_local_imagery.catalog.watches()]
  roots.extend(json.loads(os.getenv('GEO_SERVICE_SOURCE_ROOTS','[]')))
  with backend._database() as db:
   configured=db.execute("SELECT payload FROM settings WHERE key='service_source_roots'").fetchone()
  if configured:roots.extend(json.loads(configured['payload']))
  roots.extend([backend.LOCAL_RASTER_ROOT,Path(backend.DATA_DIR)/'local-imagery'])
  return protect_output(path,roots)
 def destination(value):
  default=Path(backend.DATA_DIR)/'downloads'
  p=Path(value).expanduser() if value.strip() else default
  if not p.is_absolute() or (value.strip() and not p.is_dir()) or p.is_symlink():raise ValueError('Choose an existing absolute folder on the computer running SpectraScope')
  if str(p).startswith(('//','\\\\')):raise ValueError('Choose a local disk folder')
  p=writable_output(p)
  for protected in [Path(backend.ROOT),Path(backend.DATA_DIR)/'models',Path(backend.LOCAL_RASTER_ROOT)/'snapshots']:
   protected=protected.resolve()
   if p.is_relative_to(protected) and not p.is_relative_to(default.resolve()):raise ValueError('Choose a download folder outside application code, model packs and immutable snapshots')
  if not value.strip():p.mkdir(parents=True,exist_ok=True)
  return p
 def write_metadata(path,value):
  writable_output(path)
  temporary=path.with_name(path.name+'.'+uuid4().hex+'.part')
  with temporary.open('x',encoding='utf-8') as output:
   try:
    json.dump(value,output,indent=2)
   except BaseException:
    output.close();temporary.unlink(missing_ok=True);raise
  try:temporary.replace(path)
  finally:temporary.unlink(missing_ok=True)
 def require_local_sign_in(request):
  try:local_client=request.client is not None and ip_address(request.client.host).is_loopback
  except ValueError:local_client=False
  if (not local_client or request.url.hostname not in ('localhost','127.0.0.1','::1')
      or request.headers.get('Sec-Fetch-Site','same-origin') not in ('same-origin','none')
      or request.headers.get('Origin',str(request.base_url).rstrip('/'))!=str(request.base_url).rstrip('/')):
   raise ValueError('Open the app on this computer at localhost to enter provider credentials.')
 def earthdata_token(request):
  require_local_sign_in(request)
  token=request.headers.get('X-Earthdata-Token','')
  if not re.fullmatch(r'[A-Za-z0-9._~-]{16,2048}',token):
   raise ValueError(earthdata_hls.TOKEN_REQUIRED)
  return token
 async def execute(item,token=None):
  try:
   offline_policy.require_online('Catalog download')
   folder=Path(item['folder'])
   if writable_output(folder)!=folder:raise ValueError('Download folder has moved or become a link. Start a new download.')
   folder.mkdir(exist_ok=True)
   item.update(status='running',stage='Downloading science bands');save(item)
   for asset in item['files']:
    from data_policy import public_raster_url
    if token is None:public_raster_url(item['image'].get('catalog'),asset['url'])
    path=folder/asset['name']
    if writable_output(path)!=path:raise ValueError('Download destination is a link')
    if asset.get('sha256') and path.is_file() and await run_in_threadpool(sha,path)==asset['sha256']:continue
    temp=path.with_name(f'.{uuid4().hex}.part{path.suffix}');written=0;last=0;created=False
    try:
     options={'earthdata_token':token} if token is not None else {}
     async with complete_download(item['image'].get('catalog'),asset['url'],backend.MAX_RASTER_DOWNLOAD_BYTES,**options) as (expected,chunks):
      asset['expected_bytes']=expected
      writable_output(temp)
      with temp.open('xb') as f:
       created=True
       async for chunk in chunks:
        if load(item['id']).get('control')=='stop':raise InterruptedError('Stopped; completed bands retained. Resume restarts only unfinished bands.')
        written+=len(chunk)
        f.write(chunk);asset['bytes']=written;item['stage']='Downloading '+asset['key'];item['bytes']=sum(a.get('bytes',0) for a in item['files'])
        if time.monotonic()-last>1:save(item);last=time.monotonic()
     await run_in_threadpool(offline_policy.raster_driver,str(temp));writable_output(path);temp.replace(path);asset['sha256']=await run_in_threadpool(sha,path);item['completed_files']=sum(bool(a.get('sha256')) for a in item['files']);save(item)
    finally:
     if created:temp.unlink(missing_ok=True)
   # Preserve raw provider metadata and an explicit multiband product manifest.
   image=item['image'];bands=[]
   from imagery_products import inspect,products
   indexed=[]
   for a in item['files']:
    p=folder/a['name'];record=await run_in_threadpool(inspect,p,p.name);indexed.append(record)
    metadata=image.get('asset_metadata',{}).get(a['key'],{});eo=metadata.get('eo:bands',[]);raster=metadata.get('raster:bands',[])
    names=[b.get('name') or b.get('common_name') for b in eo]
    if len(names)!=record['count'] or not all(names):names=[a['key']] if record['count']==1 else [a['key']+'_'+str(i+1) for i in range(record['count'])]
    entry={'name':a['key'],'path':a['name'],'band_names':names}
    if len(raster)==1:
     for k in ('scale','offset'):
      if k in raster[0]:entry[k]=raster[0][k]
    bands.append(entry)
   sensor=image.get('sensor') or ('sentinel-2' if image.get('catalog','').startswith('sentinel-2') else 'landsat' if image.get('catalog')=='landsat' else '')
   manifest={'name':image.get('name') or image['id'],'sensor':sensor,'processing_level':image.get('processing_level','unknown'),'datetime':image.get('datetime'),'bands':bands,'source_catalog':image.get('catalog'),'source_item_id':image['id'],'completeness_scope':'All science raster assets supplied by this catalog item; not a claim of a complete supplier SAFE package.'}
   manifests=[]
   if image.get('catalog') not in earthdata_hls.COLLECTIONS:
    mp=folder/'spectrascope-image.json';write_metadata(mp,manifest);manifests.append((mp,manifest))
   write_metadata(folder/'catalog-source.json',image)
   found,issues=await run_in_threadpool(products,indexed,folder,manifests)
   if image.get('catalog') in earthdata_hls.COLLECTIONS:
    if len(found)!=1 or found[0]['status']!='ready' or found[0]['name']!=image['id']:
     raise ValueError('The NASA HLS files did not pass local product verification.')
   await run_in_threadpool(backend.search_local_imagery.catalog.save_products,found)
   item.update(status='complete',stage='Saved locally and indexed in Local imagery',progress=100,issues=issues,local_images=[p['id'] for p in found]);save(item)
  except asyncio.CancelledError:
   item.update(status='interrupted',stage='Service stopped. Resume to continue completed-band checkpoints.');save(item);raise
  except Exception as exc:
   message=str(exc)
   if token is not None and not isinstance(exc,(ValueError,InterruptedError)):
    message='NASA HLS transfer or local file verification failed. Check the connection and choose Resume.'
   item.update(status='stopped' if isinstance(exc,InterruptedError) else 'failed',stage=message,error=message);save(item)
  finally:tasks.pop(item['id'],None)
 async def execute_safe(item, session):
  acquired=False
  try:
   await safe_slots.acquire()
   acquired=True
   offline_policy.require_online('Copernicus download')
   folder=Path(item['folder'])
   if writable_output(folder)!=folder:raise ValueError('Download folder moved or became a link.')
   folder.mkdir(exist_ok=True)
   asset=item['files'][0]
   archive=folder/asset['name']
   if writable_output(archive)!=archive:raise ValueError('SAFE archive destination is a link.')
   def check_stop():
    if load(item['id']).get('control')=='stop':raise InterruptedError('Stopped. Resume retains a verified completed ZIP.')
   item.update(status='running',stage='Downloading original SAFE ZIP');save(item)
   if not (asset.get('sha256') and archive.is_file() and await run_in_threadpool(sha,archive)==asset['sha256']):
    access=await session.token()
    last=time.monotonic();written=0
    with tempfile.NamedTemporaryFile(dir=folder,suffix='.part',delete=False) as output:
     temp=Path(output.name)
     try:
      async with bounded_download(lambda: cdse.archive_response(asset['url'],access),cdse.MAX_ARCHIVE,retries=2) as (expected,chunks):
       asset['expected_bytes']=expected
       async for chunk in chunks:
        output.write(chunk);written+=len(chunk)
        if time.monotonic()-last>=1:
         check_stop()
         item['bytes']=asset['bytes']=written;save(item);last=time.monotonic()
      check_stop();output.close()
      asset['sha256']=await run_in_threadpool(sha,temp)
      if writable_output(archive)!=archive:raise ValueError('SAFE archive destination changed during transfer.')
      temp.replace(archive);item['bytes']=asset['bytes']=written;item['completed_files']=1;save(item)
     finally:
      output.close();temp.unlink(missing_ok=True)
   check_stop()
   name=item['image']['id']+'.SAFE';safe=folder/name
   if writable_output(safe)!=safe:raise ValueError('SAFE folder destination is a link.')
   item['stage']='Verifying and importing SAFE bands';save(item)
   if not safe.exists():
    with tempfile.TemporaryDirectory(prefix='.extract-',dir=folder) as staging:
     extracted=await run_in_threadpool(cdse.extract_safe,archive,Path(staging),name,item['image']['processing_level'],check_stop)
     check_stop()
     if writable_output(safe)!=safe:raise ValueError('SAFE destination changed during extraction.')
     extracted.rename(safe)
   found,issues=await run_in_threadpool(cdse.index_safe,safe,check_stop)
   check_stop()
   write_metadata(folder/'catalog-source.json',item['image'])
   await run_in_threadpool(backend.search_local_imagery.catalog.save_products,found)
   item.update(status='complete',stage='Original SAFE verified and indexed in Local imagery',issues=issues,local_images=[p['id'] for p in found]);save(item)
  except asyncio.CancelledError:
   item.update(status='interrupted',stage='Service stopped. Sign in to resume.');save(item);raise
  except (ValueError,InterruptedError) as exc:
   item.update(status='stopped' if isinstance(exc,InterruptedError) else 'failed',stage=str(exc),error=str(exc));save(item)
  except Exception:
   message='Copernicus transfer or SAFE import failed. Check the connection and disk space, then sign in to resume.'
   item.update(status='failed',stage=message,error=message);save(item)
  finally:
   if acquired:safe_slots.release()
   tasks.pop(item['id'],None)

 @app.post('/api/catalog-downloads/copernicus')
 async def copernicus(http_request:Request):
  try:
   offline_policy.require_online('Copernicus download')
   require_local_sign_in(http_request)
   raw=bytearray()
   async for chunk in http_request.stream():
    raw.extend(chunk)
    if len(raw)>16384:raise ValueError('Copernicus sign-in request is too large.')
   try:payload=json.loads(raw)
   except ValueError:raise ValueError('Invalid Copernicus sign-in request.') from None
   if not isinstance(payload,dict):raise ValueError('Invalid Copernicus sign-in request.')
   choices=payload.get('products')
   resume=payload.get('resume_id')
   if resume:
    item=load(resume)
    if item['image'].get('catalog') not in cdse.CATALOGS or item['status'] not in ('stopped','failed','interrupted'):
     raise ValueError('This Copernicus download cannot be resumed.')
    items=[item]
   else:
    if not isinstance(choices,list) or not 1<=len(choices)<=8 or any(not isinstance(choice,dict) for choice in choices):
     raise ValueError('Select one to eight Sentinel-2 products per sign-in.')
    keys=[(choice.get('catalog'),choice.get('id')) for choice in choices]
    if any(not isinstance(part,str) for key in keys for part in key) or len(set(keys))!=len(keys):
     raise ValueError('Select distinct Sentinel-2 products.')
    directory=payload.get('directory','')
    if not isinstance(directory,str) or len(directory)>2000:raise ValueError('Invalid download folder.')
    root=destination(directory)
    items=[]
    for catalog,ident in keys:
     image,url=await cdse.exact_product(catalog,ident)
     identifier=uuid4().hex
     items.append({'id':identifier,'image':image,'folder':str(root/(ident+'-'+identifier[:8])),
                   'files':[{'key':'SAFE archive','url':url,'name':ident+'.zip','bytes':0}],
                   'bytes':0,'completed_files':0,'created_at':backend._now()})
   session=await cdse.sign_in(payload.get('username'),payload.get('password'),payload.get('totp',''))
  except httpx.HTTPError as exc:raise HTTPException(502,'Could not reach Copernicus.') from exc
  except ValueError as exc:raise HTTPException(422,str(exc)) from exc
  if resume:
   current=load(item['id'])
   if current['status'] not in ('stopped','failed','interrupted') or item['id'] in tasks:
    raise HTTPException(409,'This download is already running.')
  for item in items:
   item.update(control='resume',status='queued',stage='Waiting for Copernicus SAFE download',error=None)
   save(item,True)
   tasks[item['id']]=asyncio.create_task(execute_safe(item,session))
  return {'items':items}

 def launch(item,token=None):tasks[item['id']]=asyncio.create_task(execute(item,token))
 @app.get('/api/catalog-downloads')
 async def listing():
  init()
  with backend._database() as db:items=[json.loads(r['payload']) for r in db.execute('SELECT payload FROM catalog_downloads ORDER BY rowid DESC LIMIT 100')]
  return {'items':items,'default_directory':str(Path(backend.DATA_DIR)/'downloads'),'path_scope':'Folder on the computer running SpectraScope. Docker installations use a writable mounted folder.'}
 @app.post('/api/catalog-downloads')
 async def create(request:DownloadRequest, http_request:Request):
  image=request.image
  nasa=image.get('catalog') in earthdata_hls.COLLECTIONS
  token=None
  try:
   offline_policy.require_online('Catalog download')
   if image.get('catalog') in ('sentinel-2','sentinel-2-l1c'):
    raise ValueError('Use Copernicus sign-in in the download dialog to download and import the original SAFE package.')
   root=destination(request.directory)
   if nasa:
    token=earthdata_token(http_request)
    image=await earthdata_hls.exact_product(image.get('catalog'),image.get('id'))
    assets=list(image['assets'].items())
   else:
    assets=backend._science_assets(image)
   if not image.get('id') or not assets:raise ValueError('This online catalog item has no downloadable science raster assets')
  except httpx.HTTPError as exc:raise HTTPException(502,'Could not reach NASA CMR for this acquisition.') from exc
  except ValueError as exc:raise HTTPException(422,str(exc)) from exc
  identifier=uuid4().hex;name=re.sub(r'[^A-Za-z0-9_-]','_',str(image['id']))[:55]+'-'+identifier[:8];folder=root/name
  files=[]
  for key,url in assets:
   suffix=Path(urlparse(url).path).suffix.lower()
   filename=(f"{image['id']}.{key}.tif" if nasa else
             re.sub(r'[^A-Za-z0-9_-]','_',key)[:45]+'-'+hashlib.sha256(key.encode()).hexdigest()[:6]+(suffix if suffix in ('.jp2','.tif','.tiff') else '.tif'))
   files.append({'key':key,'url':url,'name':filename,'bytes':0})
  item={'id':identifier,'image':image,'folder':str(folder),'files':files,'status':'queued','stage':'Waiting to download','bytes':0,'completed_files':0,'created_at':backend._now()};save(item);launch(item,token);return item
 @app.get('/api/catalog-downloads/{identifier}')
 async def get(identifier:str):return load(identifier)
 @app.post('/api/catalog-downloads/{identifier}/{action}')
 async def control(identifier:str,action:str,http_request:Request):
  item=load(identifier)
  if action=='stop' and item['status'] in ('queued','running'):item['control']='stop';save(item,True)
  elif action=='resume' and item['status'] in ('failed','interrupted','stopped'):
   try:
    offline_policy.require_online('Resume download')
    if item['image'].get('catalog') in cdse.CATALOGS:raise ValueError('Sign in to Copernicus in the download window to resume.')
    token=earthdata_token(http_request) if item['image'].get('catalog') in earthdata_hls.COLLECTIONS else None
   except ValueError as exc:raise HTTPException(422,str(exc)) from exc
   item.update(control='resume',status='queued',error=None);save(item,True);launch(item,token)
  else:raise HTTPException(409,'This download cannot '+action+' in its current state')
  return item
 @app.on_event('shutdown')
 async def shutdown():
  pending=list(tasks.values())
  for task in pending:task.cancel()
  if pending:await asyncio.gather(*pending,return_exceptions=True)
 @app.on_event('startup')
 async def recover():
  init()
  with backend._database() as db:items=[json.loads(r['payload']) for r in db.execute('SELECT payload FROM catalog_downloads')]
  for item in items:
   if item['status'] in ('running','queued'):item.update(status='interrupted',stage='Previous service interrupted. Choose Resume.');save(item)
 backend.execute_catalog_download=execute
