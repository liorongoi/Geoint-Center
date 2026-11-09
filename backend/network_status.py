"""Runtime connectivity for catalogs/maps. Installation and model packs stay local."""
from __future__ import annotations
import asyncio,json,os,time
from pathlib import Path
import httpx
from public_raster_transport import public_http_client
PROBES={
 'osm':('https://tile.openstreetmap.org/0/0/0.png','image'),
 'carto_dark':('https://basemaps.cartocdn.com/dark_all/0/0/0.png','image'),
 'satellite':('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/0/0/0','image'),
 'sentinel-2':('https://stac.dataspace.copernicus.eu/v1/','json'),
 'landsat':('https://landsatlook.usgs.gov/stac-server/','json'),
 'hls':('https://cmr.earthdata.nasa.gov/stac/LPCLOUD','json'),
 'cbers-4a-wpm':('https://data.inpe.br/bdc/stac/v1/collections/CB4A-WPM-L4-DN-1','json'),
}
MAP_PROBES=('osm','satellite','carto_dark')
CATALOG_PROBES=tuple(key for key in PROBES if key not in MAP_PROBES)
_lock=asyncio.Lock()
_monitor=None
def automatic():return os.getenv('SPECTRASCOPE_NETWORK_MODE')=='auto'
def strict():return os.getenv('SPECTRASCOPE_OFFLINE','').lower() in ('1','true','yes')
def state_file():return Path(os.getenv('SPECTRASCOPE_CONNECTIVITY_FILE',str(Path(os.getenv('SPECTRASCOPE_DATA_DIR','data'))/'connectivity.json')))
def snapshot():
 if strict():return {'mode':'offline','online':False,'checked_at':None,'maps':dict.fromkeys(MAP_PROBES,False),'catalogs':dict.fromkeys(CATALOG_PROBES,False),'reason':'Offline-only mode'}
 if not automatic():return {'mode':'connected','online':True,'checked_at':None,'maps':dict.fromkeys(MAP_PROBES,True),'catalogs':dict.fromkeys(CATALOG_PROBES,True)}
 try:
  result=json.loads(state_file().read_text())
  if 0<=time.time()-result['checked_at']<120:return result
 except (OSError,ValueError,KeyError,TypeError):pass
 return {'mode':'auto','online':False,'checked_at':None,'maps':dict.fromkeys(MAP_PROBES,False),'catalogs':dict.fromkeys(CATALOG_PROBES,False),'reason':'Checking access to map and catalog services'}
async def _probe(client,key,url,kind):
 try:
  async with client.stream('GET',url,headers={'User-Agent':'SpectraScope/0.2.8 (connectivity check)','Accept':'image/*' if kind=='image' else 'application/json'},follow_redirects=False) as response:
   if response.url.scheme!='https' or response.url.host!=httpx.URL(url).host:return key,False,'Unexpected redirect destination'
   if response.status_code!=200:return key,False,f'HTTP {response.status_code}'
   raw=bytearray()
   async for part in response.aiter_bytes():
    raw.extend(part)
    if len(raw)>=1024:break
   content=response.headers.get('content-type','').lower()
   valid=(content.startswith('image/') and bytes(raw).startswith((b'\x89PNG',b'\xff\xd8',b'RIFF'))) if kind=='image' else ('json' in content and bytes(raw).lstrip().startswith((b'{',b'[')))
   return key,bool(valid),'' if valid else 'Unexpected response (possibly a sign-in page)'
 except (httpx.HTTPError,OSError) as e:return key,False,(type(e).__name__+': '+str(e))[:220]
async def refresh(force=False):
 if strict() or not automatic():return snapshot()
 old=snapshot()
 if not force and old.get('checked_at') and time.time()-old['checked_at']<30:return old
 async with _lock:
  old=snapshot()
  if not force and old.get('checked_at') and time.time()-old['checked_at']<30:return old
  async with public_http_client([url for url,_ in PROBES.values()],timeout=httpx.Timeout(4,connect=3)) as client:
   result=await asyncio.gather(*[_probe(client,k,*p) for k,p in PROBES.items()])
  available={k:ok for k,ok,error in result}
  state={'mode':'auto','online':any(available.values()),'checked_at':time.time(),'maps':{k:available[k] for k in MAP_PROBES},'catalogs':{k:available[k] for k in CATALOG_PROBES},'errors':{k:error for k,ok,error in result if not ok}}
  path=state_file();path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp');temp.write_text(json.dumps(state));temp.replace(path)
  return state

def start_monitor():
    """Activate only once a user opens a workspace, never during installation."""
    global _monitor
    if strict() or not automatic():return
    if _monitor is None or _monitor.done():
        async def watch():
            while True:
                try:await refresh()
                except (OSError,ValueError,httpx.HTTPError):pass
                await asyncio.sleep(30)
        _monitor=asyncio.create_task(watch())

async def stop_monitor():
    global _monitor
    if _monitor is not None:
        _monitor.cancel()
        try:await _monitor
        except asyncio.CancelledError:pass
        _monitor=None
