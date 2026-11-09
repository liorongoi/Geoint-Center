"""Mandatory supplier read-only and local knowledge-containment policy."""
import ipaddress,os,socket
from pathlib import Path
from urllib.parse import unquote,urlsplit

for key,value in {'HF_HUB_DISABLE_TELEMETRY':'1','DO_NOT_TRACK':'1','WANDB_DISABLED':'true','WANDB_MODE':'disabled','COMET_MODE':'DISABLED'}.items():os.environ[key]=value

def require_internal_compute(url, *, resolve=True):
 p=urlsplit(url)
 if p.scheme not in ('http','https') or not p.hostname or p.username is not None or p.password is not None or p.query or p.fragment or '?' in url or '#' in url:raise ValueError('Use a local model service without URL credentials')
 trusted={'localhost','127.0.0.1','::1','host.docker.internal','ollama'}
 trusted.update(h.strip().lower() for h in os.getenv('SPECTRASCOPE_INTERNAL_COMPUTE_HOSTS','').split(',') if h.strip())
 if p.hostname.lower() not in trusted:raise ValueError('Knowledge-containment policy blocks external model services. Use the local model or an explicitly configured internal system endpoint.')
 if resolve:
  try:addresses={r[4][0] for r in socket.getaddrinfo(p.hostname,p.port or (443 if p.scheme=='https' else 80),type=socket.SOCK_STREAM)}
  except OSError as exc:raise ValueError('The approved internal model endpoint could not be resolved') from exc
  if not addresses or any(not (ipaddress.ip_address(a).is_loopback or ipaddress.ip_address(a).is_private) or ipaddress.ip_address(a).is_link_local for a in addresses):raise ValueError('Internal computation endpoints must resolve inside the approved local/private system boundary')
 return url

def supplier_operation(kind,*,verified_read_only=False):
 if kind not in ('read','query','inspect','list','download') or not verified_read_only:raise ValueError('Supplier systems are strictly read-only. This operation has no verified read-only contract and is blocked.')

def protect_output(path,source_roots):
 target=Path(path).resolve()
 for source in source_roots:
  root=Path(source).resolve()
  if target==root or target.is_relative_to(root):raise ValueError('Supplier/source folders are read-only. Choose a separate local output directory.')
 return target

def public_search_geometry(geometry):
 """Only bounded numeric WGS84 coordinates may enter a public search."""
 if not isinstance(geometry,dict):raise ValueError('AOI must be a GeoJSON geometry')
 kind=geometry.get('type')
 depth={'Point':0,'Polygon':2,'MultiPolygon':3}.get(kind) if isinstance(kind,str) else None
 count=0
 def valid(value,level):
  nonlocal count
  if not isinstance(value,list) or not value or len(value)>10000:return False
  if level:return all(valid(child,level-1) for child in value)
  count+=1
  return (count<=10000 and len(value)==2 and all(type(x) in (int,float) for x in value)
          and -180<=value[0]<=180 and -90<=value[1]<=90)
 if depth is None or not valid(geometry.get('coordinates'),depth):
  raise ValueError('Use a Point, Polygon or MultiPolygon with up to 10,000 finite 2D longitude/latitude positions')
 return {'type':kind,'coordinates':geometry['coordinates']}

PUBLIC_RASTER_ROUTES={
 'sentinel-2':{'download.dataspace.copernicus.eu':('/odata/v1/',),'datahub.creodias.eu':('/odata/v1/',)},
 'landsat':{'landsatlook.usgs.gov':('/data/',)},
 'eo1-hyperion':{'landsatlook.usgs.gov':('/data/',)},
 'cbers-4a-wpm':{'data.inpe.br':('/bdc/data/CB4A-WPM-L4-DN/',)},
}

def public_raster_url(catalog,url):
 """Permit only approved, unauthenticated public catalog raster routes."""
 p=urlsplit(url)
 path=unquote(p.path)
 if (p.scheme!='https' or p.port not in (None,443) or p.username is not None or p.password is not None
     or p.query or p.fragment or '\\' in path or '%' in path or '..' in path.split('/')
     or any(ord(c)<32 for c in path)
     or not any(path.startswith(prefix) for prefix in PUBLIC_RASTER_ROUTES.get(
         'sentinel-2' if catalog=='sentinel-2-l1c' else catalog,{}).get(p.hostname,()))):
  raise ValueError('Raster URL is outside the approved direct routes for this catalog. Import a saved local file.')
 return url
