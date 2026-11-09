"""Local, authenticated agent services. Supplier inputs are never opened for writing.

Inspection produces versioned local knowledge, never executable generated code.
Unknown operations and formats fail closed. Secrets are not persisted in records.
"""
import hashlib,ipaddress,json,math,os,re,secrets,sqlite3,time
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4
from contextlib import contextmanager
import numpy as np
from data_policy import supplier_operation

VECTOR_OUTPUT_LIMIT=64*1024**2

CAPABILITIES={
 'geolocation':{'name':'Image ↔ ground','available':['GeoTIFF/JP2 affine, GCP and RPC metadata','Independent checkpoints for embedded affine, GCP and RPC models','RPC00B RPB, GDAL RPC text, and full-scene or regular-tile DIMAP XML with 3D control-point validation','Pixel/ground conversion','Pixel Shapefile and pixel GeoJSON to geographic GeoJSON','Guided JSON and XML affine mappings, world files with EPSG or an explicit PRJ, and GDAL PAM affine and GCP sidecars'],'pending':['Unfamiliar binary/triangulation formats need examples and format-specific validation']},
 'catalog':{'name':'Catalog connections','available':['Read-only SQLite/GeoPackage, PostgreSQL, MongoDB, MySQL 8.4, SQL Server 2022 and 2025, and Oracle AI Database 26ai','STAC mounted imagery and OGC API Features catalogs','CSV and GeoJSON catalog snapshots','Explicit field mapping and approved local-image catalog import'],'pending':['Other database engines require their own verified read-only adapter or an existing standard catalog endpoint','Windows native driver validation','Remote STAC science-asset ingestion through the new connector']},
 'algorithm':{'name':'Algorithm onboarding','available':['Static Python AST, imports and function-parameter inspection','Task group and adapter-readiness checklist','Versioned package inventory and a reviewed data-only linear LULC runtime'],'pending':['Other executable model families require reviewed adapters. Inspection alone never publishes an algorithm']},
 'api':{'name':'API connections','available':['Local and internal OpenAPI document inspection','Query-only ArcGIS Enterprise FeatureServer discovery and layers','Internal MCP discovery, listed-resource reads and approved GeoJSON catalog imports','Private transport, pinned addresses and verified remote TLS'],'pending':['Vendor-specific REST read contracts and MCP tool execution; hints alone never authorize calls']}}

def private_transport(request):
 """Do not trust Host/X-Forwarded-* to permit cleartext private credentials."""
 try:
  address=ipaddress.ip_address(request.client.host);address=address.ipv4_mapped if isinstance(address,ipaddress.IPv6Address) and address.ipv4_mapped else address
 except (ValueError,AttributeError):return False
 if address.is_loopback:return True
 if address.is_link_local or address.is_multicast or address.is_unspecified or not address.is_private:return False
 proxies={p.strip() for p in os.getenv('GEO_LOCAL_PROXY_CLIENTS','').split(',') if p.strip()}
 return request.url.scheme=='https' or str(address) in proxies

def file_hash(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()


def pixel_geojson_features(document,collect):
 """Read an explicitly pixel-coordinate FeatureCollection, never a source CRS."""
 if (not isinstance(document,dict) or document.get('type')!='FeatureCollection'
     or 'crs' in document or not isinstance(document.get('features'),list)
     or len(document['features'])>2000):
  raise ValueError('Use a pixel FeatureCollection without a declared CRS and with at most 2000 features')

 def position(value):
  if (not isinstance(value,list) or len(value)!=2
      or any(type(v) not in (int,float) or type(v) is float and not math.isfinite(v) for v in value)):
   raise ValueError('Use finite two-dimensional pixel column,row pairs')
  return collect(value)

 def sequence(value,convert,minimum=1):
  if not isinstance(value,list) or len(value)<minimum:
   raise ValueError('Invalid or empty pixel geometry coordinates')
  return [convert(child) for child in value]

 def line(value):
  return sequence(value,position,2)

 def ring(value):
  result=sequence(value,position,4)
  if value[0]!=value[-1]:
   raise ValueError('Pixel polygon rings must be closed')
  return result

 def polygon(value):
  return sequence(value,ring)

 converters={'Point':position,'MultiPoint':lambda value:sequence(value,position),
             'LineString':line,'MultiLineString':lambda value:sequence(value,line),
             'Polygon':polygon,'MultiPolygon':lambda value:sequence(value,polygon)}
 features=[]
 for feature in document['features']:
  if not isinstance(feature,dict) or feature.get('type')!='Feature' or 'crs' in feature:
   raise ValueError('Expected pixel features without a declared CRS')
  geometry=feature.get('geometry')
  kind=geometry.get('type') if isinstance(geometry,dict) and 'crs' not in geometry else None
  if not isinstance(kind,str) or kind not in converters:
   raise ValueError('Use pixel point, line or polygon geometries, including their multi-part forms')
  properties=feature.get('properties')
  if properties is not None and not isinstance(properties,dict):
   raise ValueError('Pixel feature properties must be a JSON object or null')
  result={'type':'Feature','properties':properties,
          'geometry':{'type':kind,'coordinates':converters[kind](geometry.get('coordinates'))}}
  if 'id' in feature:
   identifier=feature['id']
   if type(identifier) not in (int,str) or isinstance(identifier,str) and len(identifier)>256:
    raise ValueError('Pixel feature IDs must be integers or strings up to 256 characters')
   result['id']=identifier
  features.append(result)
 return features

def control_pairs(checks,width,height,tolerance):
 if not 4<=len(checks)<=1000 or not 0<tolerance<=10:raise ValueError('Provide 4–1000 independent control points and a positive pixel tolerance <=10')
 try:
  pixels=np.asarray([c['pixel'] for c in checks],dtype=float);ground=np.asarray([c['ground'] for c in checks],dtype=float)
 except (KeyError,TypeError,OverflowError) as exc:raise ValueError('Provide pixel and ground coordinate pairs for each control point') from exc
 if pixels.shape!=(len(checks),2) or ground.shape!=pixels.shape or not np.isfinite(pixels).all() or not np.isfinite(ground).all():raise ValueError('Control points must contain finite pixel and longitude/latitude pairs')
 if (pixels<0).any() or (pixels>[width,height]).any() or (np.abs(ground)>[180,90]).any():raise ValueError('Control points lie outside the image or longitude/latitude domain')
 if len(np.unique(pixels,axis=0))<4 or len(np.unique(ground,axis=0))<4:raise ValueError('Provide at least four distinct control points')
 if np.linalg.matrix_rank(np.c_[pixels,np.ones(len(pixels))])<3:raise ValueError('Control points must span two image dimensions')
 return pixels,ground


def validate_rpc_points(rpc,width,height,checks,tolerance):
 import rasterio
 from rasterio.errors import RasterioError
 from rasterio.transform import RPCTransformer
 pixels,ground=control_pairs(checks,width,height,tolerance)
 heights=[c.get('height_metres') for c in checks]
 if any(type(h) not in (int,float) or not -12000<=h<=12000 for h in heights):
  raise ValueError('Supply an independent ellipsoidal height in metres for each RPC control point')
 try:
  with rasterio.Env(PROJ_NETWORK='OFF'),RPCTransformer(rpc) as transformer:
   rows,cols=transformer.rowcol(*ground.T,zs=heights,op=lambda value:value)
   inverse=transformer.xy(pixels[:,1],pixels[:,0],zs=heights,offset='ul')
 except (RasterioError,ValueError) as exc:raise ValueError('RPC control-point transformation failed') from exc
 errors=np.linalg.norm(np.column_stack((cols,rows))-pixels,axis=1)
 if not np.isfinite(errors).all() or not np.isfinite(inverse).all() or errors.max()>tolerance:
  raise ValueError('Independent control points exceed the requested pixel error tolerance')
 return {'control_points':checks,'maximum_pixel_error':float(errors.max()),
         'tolerance_pixels':tolerance,'scope':'Operator-provided 3D checkpoints'}


def suggest_affine_mapping(document):
 """Find one recognizable JSON field group. Geometry still requires control points."""
 aliases={'transform':{'transform','affine','affinecoefficients','geotransform'},
          'crs':{'crs','srs','epsg','spatialreference'},
          'width':{'width','imagewidth','rasterwidth','samples','columns'},
          'height':{'height','imageheight','rasterheight','lines','rows'}}
 pending=[(document,())];matches=[];visited=0
 while pending:
  group,prefix=pending.pop()
  if not isinstance(group,dict):continue
  visited+=1
  if visited>128 or len(prefix)>6 or len(group)>512:
   raise ValueError('JSON field discovery exceeded its limit; enter an explicit field mapping')
  paths={role:[] for role in aliases}
  for key,value in group.items():
   if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_-]{0,79}',key):continue
   if isinstance(value,dict):
    if visited+len(pending)>=128:raise ValueError('JSON field discovery exceeded its limit; enter an explicit field mapping')
    pending.append((value,(*prefix,key)))
   canonical=re.sub(r'[_-]','',key).lower()
   for role,options in aliases.items():
    if canonical in options:paths[role].append((key,canonical))
  if all(paths.values()):
   if any(len(values)!=1 for values in paths.values()):
    raise ValueError('Several JSON fields could match one role; enter an explicit field mapping')
   mapping={role:'.'.join((*prefix,values[0][0])) for role,values in paths.items()}
   matches.append((mapping,'gdal' if paths['transform'][0][1]=='geotransform' else 'affine'))
 if len(matches)!=1:
  raise ValueError('Expected one recognizable JSON affine field group; enter an explicit field mapping')
 return matches[0]


def affine_xml_fields(raw):
 """Read a bounded tree of literal element paths without resolving XML links."""
 from xml.etree import ElementTree
 try:text=raw.decode('utf-8-sig')
 except UnicodeError as exc:raise ValueError('Use a UTF-8 XML affine metadata file') from exc
 if re.search(r'<!\s*(?:DOCTYPE|ENTITY)\b',text,re.I):
  raise ValueError('XML affine metadata cannot contain DTDs or entity declarations')
 try:root=ElementTree.fromstring(text)
 except ElementTree.ParseError as exc:raise ValueError('Invalid XML affine metadata') from exc
 visited=0
 def collect(node,depth):
  nonlocal visited
  visited+=1
  if visited>512 or depth>8:raise ValueError('XML affine metadata exceeds the field or nesting limit')
  if not len(node):return (node.text or '').strip()
  fields={}
  for child in node:
   value=collect(child,depth+1)
   if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_-]{0,79}',child.tag):
    # Repeated tags are ambiguous, including when their values happen to agree.
    fields[child.tag]=None if child.tag in fields else value
  return fields
 if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_-]{0,79}',root.tag):
  raise ValueError('Use an XML root with a plain element name')
 return {root.tag:collect(root,0)}

def access_key(data):
 path=Path(data)/'services'/'access.key';path.parent.mkdir(parents=True,exist_ok=True)
 try:
  with path.open('x',encoding='utf-8') as stream:stream.write(secrets.token_hex(32))
  path.chmod(0o600)
 except FileExistsError:pass
 if path.is_symlink():raise ValueError('Service access key cannot be a symbolic link')
 return path.read_text().strip()

class Services:
 def __init__(self,database,data,roots):
  self.database,self.data=database,Path(data).resolve()
  self.roots=[]
  for path in roots:
   try:root=service_root(path)
   except (ValueError,OSError):continue
   if root not in self.roots:self.roots.append(root)
 def init(self):
  with self.database() as db:
   db.execute('CREATE TABLE IF NOT EXISTS service_knowledge(id TEXT PRIMARY KEY,kind TEXT NOT NULL,state TEXT NOT NULL,payload TEXT NOT NULL,created_at REAL NOT NULL)')
 def source_path(self,path):
  supplier_operation('inspect',verified_read_only=True)
  p=Path(path).expanduser()
  def linked(entry):return entry.is_symlink() or getattr(entry,'is_junction',lambda:False)()
  if not p.is_absolute() or linked(p):raise ValueError('Use an absolute source path without symbolic links or junctions')
  supplied=p;p=p.resolve()
  if not any(p.is_relative_to(root) for root in self.roots):raise ValueError('Source lies outside the configured read-only service roots')
  # Permit OS aliases such as macOS /var -> /private/var, but never follow a
  # user-created link in or into an approved source tree.
  if any(linked(parent) and any(parent.resolve().is_relative_to(root) or parent.parent.resolve().is_relative_to(root) for root in self.roots) for parent in supplied.parents):raise ValueError('Symbolic links inside source trees are not supported')
  if p.is_relative_to(self.data) and not any(p.is_relative_to(self.data/name) for name in ('local-rasters','local-imagery','downloads')):
   raise ValueError('Private system material is not an agent source')
  configured_db=os.getenv('SPECTRASCOPE_DB_PATH')
  if configured_db:
   db=Path(configured_db).resolve()
   if p.parent==db.parent and p.name in {db.name,db.name+'-wal',db.name+'-shm',db.name+'-journal'}:
    raise ValueError('The workspace database is not an agent source')
  return p
 def source(self,path,suffixes=None,max_bytes=None):
  p=self.source_path(path)
  if not p.is_file():raise ValueError('Source file is unavailable')
  if suffixes and p.suffix.lower() not in suffixes:raise ValueError('Unsupported source format; no automatic parser execution is allowed')
  if max_bytes is not None and p.stat().st_size>max_bytes:raise ValueError('Source exceeds this bounded inspection limit')
  return p
 def save(self,kind,payload,state='inspected',parent=None,db=None):
  if db is None:self.init()
  record={'id':str(uuid4()),'kind':kind,'state':state,'created_at':time.time(),'parent_id':parent,'knowledge_boundary':'local system only','supplier_access':'read-only',**payload}
  values=(record['id'],kind,state,json.dumps(record,allow_nan=False),record['created_at'])
  if db is not None:db.execute('INSERT INTO service_knowledge VALUES(?,?,?,?,?)',values)
  else:
   with self.database() as connection:connection.execute('INSERT INTO service_knowledge VALUES(?,?,?,?,?)',values)
  return record
 def get(self,identifier):
  self.init()
  with self.database() as db:row=db.execute('SELECT payload FROM service_knowledge WHERE id=?',(identifier,)).fetchone()
  if not row:raise ValueError('Saved service configuration not found')
  return self.read_record(row['payload'])
 def records(self):
  self.init()
  with self.database() as db:return [self.read_record(row['payload']) for row in db.execute('SELECT payload FROM service_knowledge ORDER BY created_at DESC LIMIT 100')]
 def read_record(self,payload):
  record=json.loads(payload)
  if record.get('kind')=='connection_preview' and record.get('driver')=='stac':
   from catalog_connections import stac_asset_inventory
   # Legacy inventory stored raw items. Project on read to preserve the
   # append-only record while keeping old asset URLs out of API responses.
   record['rows']=stac_asset_inventory(record.get('rows',[]),record['table'],100)
  return record
 def inspect_geolocation(self,path):
  p=self.source(path,{'.tif','.tiff','.jp2'})
  import rasterio
  from offline_policy import local_raster,raster_driver
  with rasterio.Env(PROJ_NETWORK='OFF',GDAL_PAM_ENABLED='NO',GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_GEOREF_SOURCES='INTERNAL'),rasterio.open(local_raster(str(p)),driver=raster_driver(str(p))) as src:
   gcps,gcp_crs=src.gcps
   modes=(['rpc'] if src.rpcs else [])+(['gcp'] if gcps and gcp_crs else [])+(['affine'] if src.crs and src.transform!=rasterio.Affine.identity() else [])
   info={'source':str(p),'source_sha256':file_hash(p),'width':src.width,'height':src.height,'crs':str(src.crs) if src.crs else None,'modes':modes,'transform':list(src.transform),'gcp_count':len(gcps),'gcp_crs':str(gcp_crs) if gcp_crs else None,'rpc':src.rpcs.to_dict() if src.rpcs else None,'pixel_convention':'Continuous column,row from upper-left corner; pixel centre is column+0.5,row+0.5','accuracy':'unmeasured; supplier metadata is not independent validation'}
  if not modes:raise ValueError('No supported geolocation model. Supply metadata or a validated custom recipe; coordinates will not be guessed.')
  return self.save('geolocation',info)
 def propose_embedded_validation(self,identifier,mode,checks,tolerance):
  from rasterio.rpc import RPC
  record=self.get(identifier)
  if (record['kind']!='geolocation' or record['state']!='inspected'
      or mode not in ('affine','gcp','rpc') or mode not in record['modes']):
   raise ValueError('Choose an inspected raster with embedded affine, GCP or RPC metadata')
  path=self.source(record['source'],{'.tif','.tiff','.jp2'})
  checksum=record['source_sha256']
  if file_hash(path)!=checksum:raise ValueError('Source changed; inspect the image again')
  width,height=record['width'],record['height']
  if mode=='affine':
   return self.validate_affine(path,checksum,record['transform'][:6],record['crs'],width,height,
                               checks,tolerance,{'format':'embedded-affine'},parent=identifier)
  if mode=='rpc':
   model={'rpc':record['rpc'],'crs':'EPSG:4326'}
   validation=validate_rpc_points(RPC(**record['rpc']),width,height,checks,tolerance)
  else:
   model,validation=self.validate_embedded_gcps(path,width,height,checks,tolerance)
  if file_hash(path)!=checksum:raise ValueError('Source changed during control-point validation')
  return self.save('geolocation',{'source':str(path),'source_sha256':checksum,'format':'embedded-'+mode,
    'modes':[mode],**model,'width':width,'height':height,
    'pixel_convention':record['pixel_convention'],'validation':validation},state='candidate',parent=identifier)
 def validate_embedded_gcps(self,path,width,height,checks,tolerance):
  import rasterio
  from offline_policy import local_raster,raster_driver
  with rasterio.Env(PROJ_NETWORK='OFF',GDAL_PAM_ENABLED='NO',GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_GEOREF_SOURCES='INTERNAL'):
   with rasterio.open(local_raster(str(path)),driver=raster_driver(str(path))) as src:
    gcps,crs=src.gcps
  return self.validate_gcp_points(gcps,crs,width,height,checks,tolerance)
 def validate_gcp_points(self,gcps,crs,width,height,checks,tolerance):
  import rasterio
  from rasterio.transform import GCPTransformer
  from rasterio.warp import transform
  pixels,ground=control_pairs(checks,width,height,tolerance)
  if any(c.get('height_metres') is not None for c in checks):
   raise ValueError('GCP checkpoint validation uses a planar model; omit heights')
  try:
   with rasterio.Env(PROJ_NETWORK='OFF'):
    if not crs or not 3<=len(gcps)<=1000:raise ValueError('Expected 3–1000 fitting GCPs with a coordinate reference system')
    fitting_pixels=np.array([(p.col,p.row) for p in gcps])
    if np.any(np.linalg.norm(pixels[:,None]-fitting_pixels,axis=2)<=1):
     raise ValueError('Independent checkpoints must be more than one pixel from the fitting GCPs')
    fitting_ground=np.column_stack(transform(crs,'EPSG:4326',[p.x for p in gcps],[p.y for p in gcps]))
    separation=np.abs(ground[:,None]-fitting_ground)
    separation[:,:,0]=np.minimum(separation[:,:,0],np.abs(360-separation[:,:,0]))
    if not np.isfinite(fitting_ground).all() or np.any(np.max(separation,axis=2)<=1e-8):
     raise ValueError('Independent checkpoints must have ground locations distinct from the fitting GCPs')
    x,y=transform('EPSG:4326',crs,*ground.T)
    with GCPTransformer(gcps) as transformer:
     rows,cols=transformer.rowcol(x,y,op=lambda value:value)
     inverse=transformer.xy(pixels[:,1],pixels[:,0],offset='ul')
    inverse=transform(crs,'EPSG:4326',*inverse)
  except rasterio.errors.RasterioError as exc:raise ValueError('GCP transformation failed') from exc
  errors=np.linalg.norm(np.column_stack((cols,rows))-pixels,axis=1)
  if not np.isfinite(errors).all() or not np.isfinite(inverse).all() or errors.max()>tolerance:
   raise ValueError('Independent control points exceed the requested pixel error tolerance')
  return ({'gcp_count':len(gcps),'gcp_crs':str(crs)},
          {'control_points':checks,'maximum_pixel_error':float(errors.max()),'tolerance_pixels':tolerance,
           'scope':'Operator-provided 2D checkpoints distinct from fitting GCPs'})
 def convert(self,identifier,points,direction,mode=None,height=None):
  import rasterio
  with rasterio.Env(PROJ_NETWORK='OFF',GDAL_PAM_ENABLED='NO',GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_GEOREF_SOURCES='INTERNAL'):return self._convert(identifier,points,direction,mode,height)
 def _convert(self,identifier,points,direction,mode=None,height=None):
  if direction not in ('image_to_ground','ground_to_image'):raise ValueError('Choose image_to_ground or ground_to_image')
  if not 1<=len(points)<=10000 or any(len(p)!=2 or not all(math.isfinite(v) for v in p) for p in points):raise ValueError('Supply 1–10000 finite coordinate pairs')
  record=self.get(identifier)
  if record['kind']!='geolocation' or record['state'] not in ('inspected','approved'):raise ValueError('Select a supported geolocation configuration')
  p=self.source(record['source'])
  if file_hash(p)!=record['source_sha256']:raise ValueError('Source changed; inspect and validate a new version')
  self.verify_dimap_tile_source(record)
  self.verify_world_projection(record)
  tile=record.get('dimap_tile')
  if tile and direction=='image_to_ground' and any(not (0<=x<=record['width'] and 0<=y<=record['height']) for x,y in points):
   raise ValueError('Pixel coordinates lie outside the selected DIMAP tile')
  import rasterio
  from rasterio.warp import transform
  from rasterio.transform import RPCTransformer,GCPTransformer
  from affine import Affine
  from offline_policy import local_raster,raster_driver
  mode=mode or record['modes'][0]
  if mode not in record['modes']:raise ValueError('The chosen transform is unavailable')
  xs,ys=map(list,zip(*points));results=[]
  if mode=='rpc' and record.get('format') in ('rpc00b-rpb','gdal-rpc-text','dimap-rpc-xml','embedded-rpc'):
   from rasterio.rpc import RPC
   if height is None or not math.isfinite(height):raise ValueError('RPC conversion requires an explicit ellipsoidal height in metres')
   with RPCTransformer(RPC(**record['rpc'])) as converter:
    if direction=='image_to_ground':
     gx,gy=converter.xy(ys,xs,zs=[height]*len(xs),offset='ul')
     results=list(zip(gx,gy))
    else:
     rows,cols=converter.rowcol(xs,ys,zs=[height]*len(xs),op=lambda x:x)
     results=list(zip(cols,rows))
  elif mode=='affine':
   affine=Affine(*record['transform'][:6]);crs=record['crs']
   if direction=='image_to_ground':
    values=[affine*p for p in points];gx,gy=transform(crs,'EPSG:4326',*[list(v) for v in zip(*values)]);results=list(map(list,zip(gx,gy)))
   else:
    gx,gy=transform('EPSG:4326',crs,xs,ys);results=[list(~affine*p) for p in zip(gx,gy)]
  elif record.get('format')=='gdal-pam-gcp':
   from rasterio.control import GroundControlPoint
   if height is not None:raise ValueError('This approved GCP model is planar; leave height blank')
   crs=record['gcp_crs']
   with GCPTransformer([GroundControlPoint(**point) for point in record['gcps']]) as converter:
    if direction=='image_to_ground':
     x,y=converter.xy(ys,xs,offset='ul')
     results=list(zip(*transform(crs,'EPSG:4326',x,y)))
    else:
     x,y=transform('EPSG:4326',crs,xs,ys)
     rows,cols=converter.rowcol(x,y,op=lambda value:value)
     results=list(zip(cols,rows))
  else:
   planar_gcp=record.get('format')=='embedded-gcp'
   if planar_gcp and height is not None:raise ValueError('This approved GCP model is planar; leave height blank')
   if not planar_gcp and (height is None or not math.isfinite(height)):raise ValueError('RPC/GCP conversion requires an explicit height in metres; no hidden terrain assumption')
   heights=None if planar_gcp else [height]*len(xs)
   with rasterio.open(local_raster(str(p)),driver=raster_driver(str(p))) as src:
    gcps,gcp_crs=src.gcps
    if mode=='rpc' and not src.rpcs:raise ValueError('Embedded RPC metadata is unavailable; validate the supplier RPC sidecar if available')
    with RPCTransformer(src.rpcs) if mode=='rpc' else GCPTransformer(gcps) as converter:
     if direction=='image_to_ground':
      gx,gy=converter.xy(ys,xs,zs=heights,offset='ul')
      if mode=='gcp':gx,gy=transform(gcp_crs,'EPSG:4326',gx,gy)
      results=list(map(list,zip(gx,gy)))
     else:
      if mode=='gcp':xs,ys=transform('EPSG:4326',gcp_crs,xs,ys)
      rows,cols=converter.rowcol(xs,ys,zs=heights,op=lambda x:x);results=list(map(list,zip(cols,rows)))
  if not np.isfinite(results).all():raise ValueError('Conversion failed or coordinates lie outside the model domain')
  if tile and direction=='ground_to_image' and any(not (0<=x<=record['width'] and 0<=y<=record['height']) for x,y in results):
   raise ValueError('Projected coordinates lie outside the selected DIMAP tile')
  result={'configuration':identifier,'mode':mode,'direction':direction,'points':results,'output_coordinates':'longitude,latitude · EPSG:4326' if direction=='image_to_ground' else record['pixel_convention'],'height_metres':height,'accuracy':'unmeasured','source_sha256':record['source_sha256']}
  if record.get('format') in ('embedded-affine','embedded-gcp','embedded-rpc','gdal-pam-gcp'):
   result['accuracy']='checked against operator-supplied control points'
   result['validation_maximum_pixel_error']=record['validation']['maximum_pixel_error']
  return result
 def reuse_affine(self,identifier,path,checks,tolerance):
  recipe=self.get(identifier)
  formats={'explicit-json-affine':'.json','explicit-xml-affine':'.xml'}
  if recipe.get('kind')!='geolocation' or recipe.get('state')!='approved' or recipe.get('format') not in formats:
   raise ValueError('Choose an approved JSON or XML affine mapping')
  return self.propose_affine(path,recipe['mapping'],checks,tolerance,parent=identifier,
                             order=recipe.get('coefficient_order','affine'),epsg_only=recipe.get('epsg_only',False),
                             expected_suffix=formats[recipe['format']])
 def suggest_affine(self,path,checks,tolerance):
  return self.propose_affine(path,None,checks,tolerance)
 def propose_affine(self,path,mapping,checks,tolerance,parent=None,order='affine',epsg_only=False,expected_suffix=None):
  from algorithm_inventory import bounded_json
  suggested=mapping is None
  is_xml=Path(path).suffix.lower()=='.xml'
  epsg_only=epsg_only or suggested or is_xml
  if order not in ('affine','gdal'):raise ValueError('Choose affine or GDAL coefficient order')
  if is_xml:
   limit=256*1024
   p=self.source(path,{expected_suffix or '.xml'},limit)
   with p.open('rb') as stream:raw=stream.read(limit+1)
   if len(raw)>limit:raise ValueError('XML affine metadata exceeds 256 KiB')
   doc,checksum=affine_xml_fields(raw),hashlib.sha256(raw).hexdigest()
  else:
   p=self.source(path,{expected_suffix or '.json'},2*1024**2)
   doc,checksum=bounded_json(p,2*1024**2,reject_duplicates=epsg_only)
  if suggested:mapping,order=suggest_affine_mapping(doc)
  if set(mapping)!={'transform','crs','width','height'}:raise ValueError('Map exactly transform, crs, width and height fields')
  def field(key):
   value=doc
   for part in key.split('.'):
    if not isinstance(value,dict) or part not in value:raise ValueError('Mapped field does not exist: '+key)
    value=value[part]
   if is_xml and not isinstance(value,str):raise ValueError('Mapped XML field must be one unambiguous text element: '+key)
   return value
  coefficients=field(mapping['transform'])
  crs,width,height=(field(mapping[key]) for key in ('crs','width','height'))
  if is_xml:
   try:coefficients=[float(value) for value in (coefficients.split(',') if ',' in coefficients else coefficients.split())]
   except ValueError as exc:raise ValueError('XML transform requires six plain numeric coefficients') from exc
   if not all(re.fullmatch(r'[1-9][0-9]{0,9}',value) for value in (width,height)):
    raise ValueError('XML image dimensions must be positive integers')
   width,height=int(width),int(height)
   if re.fullmatch(r'[1-9][0-9]{2,6}',crs):crs=f'EPSG:{crs}'
  if order=='gdal':
   if not isinstance(coefficients,list) or len(coefficients)!=6:raise ValueError('GDAL geotransform requires six numbers')
   c,a,b,f,d,e=coefficients;coefficients=[a,b,c,d,e,f]
  if epsg_only:
   if type(crs) is int and 100<=crs<=9999999:crs=f'EPSG:{crs}'
   if not isinstance(crs,str) or not re.fullmatch(r'EPSG:[1-9][0-9]{2,6}',crs,re.I):
    raise ValueError('Suggested mappings require an explicit EPSG code')
  return self.validate_affine(p,checksum,coefficients,crs,width,height,
                              checks,tolerance,{'format':'explicit-xml-affine' if is_xml else 'explicit-json-affine','mapping':mapping,
                              'coefficient_order':order,'epsg_only':epsg_only,'mapping_suggested':suggested},parent=parent)
 def verify_world_projection(self,record):
  projection=record.get('world_projection')
  if projection:
   path=self.source(projection['path'],{'.prj'},16*1024)
   if file_hash(path)!=projection['sha256']:
    raise ValueError('World-file projection changed; validate a new configuration')
 def propose_world_file(self,path,crs,width,height,checks,tolerance,projection_path=None):
  details={'format':'world-file-affine','source_pixel_convention':'Upper-left pixel centre'}
  if bool(crs)==bool(projection_path):
   raise ValueError('Supply one world-file CRS, either an EPSG code or a PRJ file')
  if projection_path:
   import rasterio
   from rasterio.crs import CRS
   projection=self.source(projection_path,{'.prj'},16*1024)
   with projection.open('rb') as stream:raw=stream.read(16*1024+1)
   try:
    wkt=raw.decode('utf-8-sig').strip()
    if len(raw)>16*1024 or not re.match(r'^(?:PROJCS|GEOGCS|PROJCRS|GEOGCRS|GEODCRS)\s*\[',wkt,re.I):
     raise ValueError
    with rasterio.Env(PROJ_NETWORK='OFF'):
     parsed=CRS.from_wkt(wkt)
     epsg=parsed.to_epsg(confidence_threshold=100)
     if not epsg or not (parsed.is_geographic or parsed.is_projected):
      raise ValueError
     canonical=CRS.from_epsg(epsg)
     # ESRI geographic WKT often omits the EPSG authority axis order.
     if parsed!=canonical and parsed!=CRS.from_wkt(canonical.to_wkt(version='WKT1_ESRI')):
      raise ValueError
   except (ValueError,rasterio.errors.RasterioError) as exc:
    raise ValueError('Use a bounded PRJ WKT file that matches a recognized geographic or projected EPSG CRS') from exc
   crs=f'EPSG:{epsg}'
   details['world_projection']={'path':str(projection),'sha256':hashlib.sha256(raw).hexdigest()}
  p=self.source(path,{'.tfw','.tifw','.jgw','.j2w','.jp2w','.pgw','.wld'},4096)
  with p.open('rb') as stream:content=stream.read(4097)
  if len(content)>4096:raise ValueError('World file exceeds 4096 bytes')
  fields=content.decode('ascii').split()
  if len(fields)!=6:raise ValueError('A world file requires exactly six numeric values')
  a,d,b,e,centre_x,centre_y=map(float,fields)
  # World files describe the centre of the upper-left pixel; the app uses its corner.
  coefficients=[a,b,centre_x-(a+b)/2,d,e,centre_y-(d+e)/2]
  return self.validate_affine(p,hashlib.sha256(content).hexdigest(),coefficients,crs,width,height,checks,tolerance,details)
 def propose_pam_affine(self,path,width,height,checks,tolerance):
  return self.propose_pam_geolocation(path,width,height,checks,tolerance,allow_gcp=False)
 def propose_pam_geolocation(self,path,width,height,checks,tolerance,allow_gcp=True):
  """Inspect dataset-level geolocation in a GDAL PAM sidecar."""
  from xml.etree import ElementTree
  from affine import Affine
  p=self.source(path,{'.xml'},64*1024)
  if not p.name.lower().endswith('.aux.xml'):raise ValueError('Select a GDAL .aux.xml sidecar')
  with p.open('rb') as stream:content=stream.read(64*1024+1)
  try:text=content.decode('utf-8-sig')
  except UnicodeError as exc:raise ValueError('Use a UTF-8 GDAL PAM file') from exc
  if len(content)>64*1024 or re.search(r'<!\s*(?:DOCTYPE|ENTITY)\b',text,re.I):
   raise ValueError('GDAL PAM must be bounded UTF-8 XML without DTDs or entities')
  try:root=ElementTree.fromstring(text)
  except ElementTree.ParseError as exc:raise ValueError('Invalid GDAL PAM XML') from exc
  if root.tag!='PAMDataset':raise ValueError('Expected a GDAL PAMDataset')
  fitting=root.findall('GCPList')
  if allow_gcp and fitting:
   if len(fitting)!=1 or root.find('GeoTransform') is not None:
    raise ValueError('Use a PAM sidecar with a single GCPList and no competing affine transform')
   return self.propose_pam_gcps(p,content,fitting[0],width,height,checks,tolerance)
  if len(root.findall('SRS'))!=1 or len(root.findall('GeoTransform'))!=1:
   raise ValueError('Expected one dataset-level SRS and GeoTransform in the GDAL PAM file')
  srs=root.find('SRS');geotransform=root.find('GeoTransform')
  if len(srs) or len(geotransform) or not (srs.text or '').strip():
   raise ValueError('GDAL PAM geolocation fields must be plain text')
  try:
   values=[float(value) for value in (geotransform.text or '').split(',')]
   if len(values)!=6:raise ValueError
   coefficients=list(Affine.from_gdal(*values))[:6]
  except ValueError as exc:raise ValueError('GDAL GeoTransform requires six comma-separated numbers') from exc
  return self.validate_affine(p,hashlib.sha256(content).hexdigest(),coefficients,srs.text.strip(),width,height,checks,tolerance,
                              {'format':'gdal-pam-affine','source_pixel_convention':'Upper-left pixel corner'})
 def propose_pam_gcps(self,path,content,fitting,width,height,checks,tolerance):
  import rasterio
  from rasterio.control import GroundControlPoint
  from rasterio.crs import CRS,epsg_treats_as_latlong,epsg_treats_as_northingeasting
  projection=fitting.get('Projection','')
  # Ignore optional legacy PROJ strings. Only a canonical offline EPSG CRS
  # may be used for conversion, so supplier parameters cannot select grids.
  wkt=re.sub(r',\s*EXTENSION\[\s*"PROJ4"\s*,\s*"[^"]*"\s*\]', '', projection,flags=re.I)
  if not re.match(r'^(PROJCS|GEOGCS|PROJCRS|GEOGCRS)\s*\[',wkt) or re.search(r'(EXTENSION|PARAMETERFILE)\s*\[',wkt,re.I):
   raise ValueError('PAM GCP projection must be a supported offline EPSG WKT')
  try:
   with rasterio.Env(PROJ_NETWORK='OFF'):
    epsg=CRS.from_wkt(wkt).to_epsg(confidence_threshold=100)
    if epsg is None:raise ValueError('PAM GCP projection must match a canonical offline EPSG CRS')
    crs=CRS.from_epsg(epsg)
  except rasterio.errors.CRSError as exc:raise ValueError('Invalid PAM GCP projection') from exc
  expected='2,1' if epsg_treats_as_latlong(crs) or epsg_treats_as_northingeasting(crs) else '1,2'
  if fitting.get('dataAxisToSRSAxisMapping',expected)!=expected:
   raise ValueError('Unsupported PAM GCP axis mapping; expected traditional x,y order')
  if not 3<=len(fitting)<=1000:raise ValueError('Expected 3–1000 PAM fitting GCPs')
  points=[]
  for element in fitting:
   if element.tag!='GCP' or len(element) or not {'Pixel','Line','X','Y'}.issubset(element.attrib):
    raise ValueError('PAM GCP entries require Pixel, Line, X and Y attributes')
   col,row,x,y=(float(element.get(name)) for name in ('Pixel','Line','X','Y'))
   if not all(math.isfinite(value) for value in (col,row,x,y)) or not (0<=col<=width and 0<=row<=height):
    raise ValueError('PAM GCPs must be finite and lie within the declared pixel dimensions')
   points.append({'col':col,'row':row,'x':x,'y':y})
  model,validation=self.validate_gcp_points([GroundControlPoint(**point) for point in points],crs,width,height,checks,tolerance)
  checksum=hashlib.sha256(content).hexdigest()
  if file_hash(path)!=checksum:raise ValueError('PAM source changed during validation')
  return self.save('geolocation',{'source':str(path),'source_sha256':checksum,'format':'gdal-pam-gcp','gcps':points,**model,
    'width':width,'height':height,'modes':['gcp'],'pixel_convention':'Continuous column,row from upper-left corner',
    'validation':validation},state='candidate')
 def verify_dimap_tile_source(self,record):
  tile=record.get('dimap_tile')
  if tile and file_hash(self.source(tile['metadata'],{'.xml'},512*1024))!=tile['metadata_sha256']:
   raise ValueError('DIMAP tile metadata changed; validate a new configuration')
 def propose_rpc_sidecar(self,path,width,height,checks,tolerance,tile_metadata=None,tile_name=None):
  from rpc_sidecar import parse_rpb,parse_rpc_text,parse_dimap_rpc,parse_dimap_tile
  p=self.source(path,{'.rpb','.rpc','.txt','.xml'},64*1024)
  if p.suffix.lower()=='.rpb':
   parser,format_id=parse_rpb,'rpc00b-rpb'
  elif p.suffix.lower()=='.rpc' or p.name.lower().endswith('_rpc.txt'):
   parser,format_id=parse_rpc_text,'gdal-rpc-text'
  elif p.suffix.lower()=='.xml' and p.name.lower().startswith('rpc_'):
   parser,format_id=parse_dimap_rpc,'dimap-rpc-xml'
  else:raise ValueError('Use an RPB, .rpc, _RPC.TXT or DIMAP RPC_*.XML sidecar')
  if (tile_metadata or tile_name) and (format_id!='dimap-rpc-xml' or not tile_metadata or not tile_name):
   raise ValueError('DIMAP tile conversion requires both product metadata and the tile filename')
  with p.open('rb') as stream:content=stream.read(64*1024+1)
  if len(content)>64*1024:raise ValueError('RPC sidecar exceeds 64 KiB')
  rpc=parser(content)
  provenance={}
  if format_id=='dimap-rpc-xml':
   rpc,profile=rpc
   provenance={'dimap_profile':profile,'source_pixel_scope':'Full scene'}
   if tile_metadata:
    metadata=self.source(tile_metadata,{'.xml'},512*1024)
    if metadata.name.casefold()!='dim_'+p.name[4:].casefold():
     raise ValueError('Use the corresponding DIM_*.XML product file for this RPC_*.XML sidecar')
    with metadata.open('rb') as stream:raw=stream.read(512*1024+1)
    if len(raw)>512*1024:raise ValueError('DIMAP product metadata exceeds 512 KiB')
    tile=parse_dimap_tile(raw,tile_name,profile)
    if (width,height)!=(tile['width'],tile['height']):
     raise ValueError(f"Selected DIMAP tile is {tile['width']} × {tile['height']} pixels; enter its actual dimensions")
    parameters=rpc.to_dict()
    parameters['samp_off']-=tile['column_offset']
    parameters['line_off']-=tile['row_offset']
    rpc=type(rpc)(**parameters)
    provenance.update(source_pixel_scope='Tile local',dimap_tile={**tile,'metadata':str(metadata),
                      'metadata_sha256':hashlib.sha256(raw).hexdigest()})
  if type(width) is not int or type(height) is not int or not (0<width<=2**31-1 and 0<height<=2**31-1):
   raise ValueError('Supply valid image dimensions')
  validation=validate_rpc_points(rpc,width,height,checks,tolerance)
  checksum=hashlib.sha256(content).hexdigest()
  if file_hash(p)!=checksum:raise ValueError('RPC source changed during inspection')
  self.verify_dimap_tile_source(provenance)
  return self.save('geolocation',{'source':str(p),'source_sha256':checksum,'format':format_id,**provenance,
    'modes':['rpc'],'rpc':rpc.to_dict(),'width':width,'height':height,'crs':'EPSG:4326',
    'pixel_convention':'Continuous column,row from upper-left tile corner' if provenance.get('dimap_tile') else 'Continuous column,row from upper-left corner','validation':validation},state='candidate')
 def validate_affine(self,path,checksum,coefficients,crs,width,height,checks,tolerance,details,parent=None):
  from affine import Affine
  import rasterio
  from rasterio.crs import CRS
  from rasterio.warp import transform
  if (not isinstance(coefficients,list) or len(coefficients)!=6
      or any(type(n) not in (int,float) or not -1e20<=n<=1e20 for n in coefficients)):
   raise ValueError('Affine recipe requires six finite numeric coefficients')
  affine=Affine(*coefficients)
  if type(width) is not int or type(height) is not int or not 0<width<=2**31-1 or not 0<height<=2**31-1 or abs(affine.determinant)<1e-20:
   raise ValueError('Invalid image dimensions or singular transform')
  pixels,ground=control_pairs(checks,width,height,tolerance)
  with rasterio.Env(PROJ_NETWORK='OFF'):
   try:
    crs=CRS.from_user_input(crs)
    gx,gy=transform('EPSG:4326',crs,*ground.T)
   except rasterio.errors.RasterioError as exc:raise ValueError('Invalid or unavailable offline coordinate reference system') from exc
  pred=np.asarray([~affine*p for p in zip(gx,gy)]);errors=np.linalg.norm(pred-pixels,axis=1)
  if not np.isfinite(errors).all() or np.max(errors)>tolerance:raise ValueError('Independent control points exceed the requested pixel error tolerance')
  if file_hash(path)!=checksum:raise ValueError('Source changed during recipe inspection')
  self.verify_world_projection(details)
  return self.save('geolocation',{'source':str(path),'source_sha256':checksum,**details,'modes':['affine'],'transform':list(affine),'crs':str(crs),'width':width,'height':height,'pixel_convention':'Continuous column,row from upper-left corner','validation':{'control_points':checks,'maximum_pixel_error':float(errors.max()),'tolerance_pixels':tolerance,'scope':'User-provided independent control points; no claim of universal format support'}},state='candidate',parent=parent)
 def convert_pixel_shapes(self,identifier,path,pixel_coordinates=False,mode=None,height=None):
  """Read bounded pixel vectors and create a new private GeoJSON.

  Coordinates are continuous column,row from the upper-left corner. They are
  not silently shifted to pixel centres. A declared source CRS is never used
  to reinterpret the operator's explicit pixel-coordinate confirmation.
  """
  if pixel_coordinates is not True:raise ValueError('Explicitly confirm that this file uses image column,row coordinates')
  limit=16*1024**2
  p=self.source(path,{'.shp','.geojson'},limit);points=[];features=[]
  def collect(coords):
   if not coords:return []
   if isinstance(coords[0],(int,float)):
    if len(coords)!=2:raise ValueError('Use two-dimensional pixel geometries; Z/M conversion is not supported')
    index=len(points);points.append(list(coords))
    if len(points)>10000:raise ValueError('This conversion supports at most 10000 vertices; split the input explicitly')
    return index
   return [collect(c) for c in coords]
  if p.suffix.lower()=='.geojson':
   from algorithm_inventory import bounded_json
   document,checksum=bounded_json(p,limit,reject_duplicates=True)
   hashes={str(p):checksum}
   features=pixel_geojson_features(document,collect)
  else:
   import shapefile
   sources=[p]
   for suffix in ('.shx','.dbf','.prj','.cpg'):
    companion=p.with_suffix(suffix)
    if suffix in ('.shx','.dbf') or companion.exists():sources.append(self.source(companion,{suffix},limit))
   hashes={str(s):file_hash(s) for s in sources}
   with shapefile.Reader(str(p),encoding='utf-8',encodingErrors='strict') as reader:
    if len(reader)>2000:raise ValueError('This conversion supports at most 2000 features')
    if reader.shapeType not in (shapefile.POINT,shapefile.POLYLINE,shapefile.POLYGON,shapefile.MULTIPOINT):raise ValueError('Use a 2D point, line or polygon Shapefile')
    for item in reader.iterShapeRecords():
     if item.shape.shapeType==shapefile.NULL:continue
     geometry=item.shape.__geo_interface__
     features.append({'type':'Feature','properties':item.record.as_dict(),'geometry':{'type':geometry['type'],'coordinates':collect(geometry['coordinates'])}})
  if not points:raise ValueError('Source contains no convertible pixel vertices')
  configuration=self.get(identifier)
  if configuration['kind']!='geolocation' or configuration['state'] not in ('inspected','approved'):
   raise ValueError('Select a supported geolocation configuration')
  if any(not 0<=x<=configuration['width'] or not 0<=y<=configuration['height'] for x,y in points):raise ValueError('Pixel vertices lie outside the configured image; select the correct image and coordinate convention')
  converted=self.convert(identifier,points,'image_to_ground',mode,height)
  if any(not -180<=x<=180 or not -90<=y<=90 for x,y in converted['points']):
   raise ValueError('Converted vertices fall outside the geographic longitude/latitude domain')
  def restore(value):return converted['points'][value] if isinstance(value,int) else [restore(v) for v in value]
  for feature in features:feature['geometry']['coordinates']=restore(feature['geometry']['coordinates'])
  if any(file_hash(self.source(source))!=digest for source,digest in hashes.items()):
   raise ValueError('Pixel vector source changed during conversion; no result was published')
  output_id=str(uuid4());folder=self.data/'services'/'vectors'/output_id
  if folder.resolve()!=folder:raise ValueError('Vector output requires a private non-symlink workspace directory')
  folder.mkdir(parents=True)
  target=folder/(p.stem+'-ground.geojson')
  # DBF dates are serialized as ISO strings; binary values are rejected.
  from datetime import date,datetime
  def scalar(value):
   if isinstance(value,(date,datetime)):return value.isoformat()
   raise ValueError('Unsupported source attribute type')
  document={'type':'FeatureCollection','features':features,'provenance':{'geolocation':identifier,'source_sha256':hashes,'image_sha256':converted['source_sha256'],'pixel_convention':'Continuous column,row from upper-left corner','height_metres':height,'mode':converted['mode'],'accuracy':'unmeasured; no independent ground validation','training_reference':False}}
  try:
   try:encoded=json.dumps(document,allow_nan=False,default=scalar).encode()
   except RecursionError as exc:raise ValueError('Source attributes exceed the supported JSON nesting limit') from exc
   if len(encoded)>VECTOR_OUTPUT_LIMIT:raise ValueError('Converted vector exceeds the 64 MiB output limit')
   target.write_bytes(encoded)
   return self.save('geographic_vector',{'source':str(p),'source_sha256':hashes,'configuration':identifier,'output':str(target),'output_sha256':file_hash(target),'feature_count':len(features),'vertex_count':len(points),'crs':'EPSG:4326','accuracy':converted['accuracy'],'training_reference':False},state='complete',parent=identifier)
  except BaseException:
   target.unlink(missing_ok=True);folder.rmdir();raise
 def geographic_vector_content(self,identifier):
  record=self.get(identifier)
  if record['kind']!='geographic_vector' or record['state']!='complete':
   raise ValueError('Choose a completed geographic vector result')
  root=self.data/'services'/'vectors'
  path=Path(record['output'])
  if (not path.is_relative_to(root) or len(path.relative_to(root).parts)!=2
      or path.resolve()!=path or path.suffix!='.geojson' or not path.is_file()):
   raise ValueError('Saved vector is unavailable in private workspace storage')
  with path.open('rb') as stream:content=stream.read(VECTOR_OUTPUT_LIMIT+1)
  if len(content)>VECTOR_OUTPUT_LIMIT or hashlib.sha256(content).hexdigest()!=record['output_sha256']:
   raise ValueError('Saved vector failed verification; convert the source again')
  return content
 def geographic_vectors(self,before=None):
  self.init()
  with self.database() as db:
   db.execute("CREATE INDEX IF NOT EXISTS service_knowledge_vector_order ON service_knowledge(created_at DESC,id DESC) WHERE kind='geographic_vector' AND state='complete'")
   query="SELECT payload FROM service_knowledge WHERE kind='geographic_vector' AND state='complete'"
   args=()
   if before is not None:
    cursor=db.execute("SELECT created_at FROM service_knowledge WHERE id=? AND kind='geographic_vector' AND state='complete'",(before,)).fetchone()
    if cursor is None:raise ValueError('Choose a cursor from the saved vector results')
    query+=" AND (created_at,id)<(?,?)"
    args=(cursor['created_at'],before)
   rows=db.execute(query+" ORDER BY created_at DESC,id DESC LIMIT 21",args).fetchall()
  items=[]
  for row in rows[:20]:
   record=json.loads(row['payload'])
   items.append({'id':record['id'],'name':Path(record['output']).name,
                 'feature_count':record['feature_count'],'created_at':record['created_at']})
  return {'items':items,'next_before':items[-1]['id'] if len(rows)>20 else None}
 def approve(self,identifier):
  record=self.get(identifier)
  if record['state']!='candidate':raise ValueError('Only a validated candidate can be approved')
  if file_hash(self.source(record['source']))!=record['source_sha256']:raise ValueError('Candidate source changed; inspect a new version')
  self.verify_dimap_tile_source(record)
  self.verify_world_projection(record)
  if record['kind']=='geolocation' and record.get('validation'):return self.save('geolocation',{k:v for k,v in record.items() if k not in ('id','state','created_at','parent_id')},state='approved',parent=identifier)
  raise ValueError('This inspection cannot be published as a runnable adapter. Implementation and qualification are still required.')
 @contextmanager
 def sqlite_source(self,path):
  p=self.source(path,{'.db','.sqlite','.sqlite3','.gpkg'})
  if p.with_name(p.name+'-wal').exists() and p.with_name(p.name+'-wal').stat().st_size:raise ValueError('Active WAL database: supply a supplier-created consistent read-only snapshot; this service never checkpoints or modifies it')
  if p.with_name(p.name+'-journal').exists():raise ValueError('Database has a rollback journal; provide a consistent snapshot')
  db=sqlite3.connect('file:'+quote(str(p),safe='/')+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row
  try:
   db.enable_load_extension(False);db.execute('PRAGMA trusted_schema=OFF');db.execute('PRAGMA query_only=ON');db.set_progress_handler(lambda:1 if time.monotonic()>deadline else 0,1000);deadline=time.monotonic()+5
   yield db
  finally:db.close()
 def inspect_catalog(self,path):
  p=self.source(path);before=file_hash(p)
  with self.sqlite_source(p) as db:
   tables=[]
   for row in db.execute("SELECT name FROM pragma_table_list WHERE schema='main' AND type='table' AND name NOT LIKE 'sqlite_%' LIMIT 501"):
    if len(tables)>=500:raise ValueError('Catalog exceeds 500 tables; use a narrower supplier snapshot')
    name=row['name'];quoted='"'+name.replace('"','""')+'"';columns=[{'name':c['name'],'type':c['type']} for c in db.execute('PRAGMA table_xinfo('+quoted+')') if not c['hidden']]
    tables.append({'name':name,'columns':columns})
  if file_hash(p)!=before:raise ValueError('Catalog changed during inspection; supply a stable consistent snapshot')
  return self.save('catalog',{'source':str(p),'source_sha256':before,'driver':'sqlite','tables':tables,'mapping_required':True})
 def preview_catalog(self,identifier,table,mapping):
  record=self.get(identifier)
  if record['kind']!='catalog':raise ValueError('Choose a catalog configuration')
  schema=next((t for t in record['tables'] if t['name']==table),None)
  if not schema:raise ValueError('Select a discovered table; arbitrary SQL and views are not accepted')
  if not mapping or set(mapping)-{'image_id','path','datetime','geometry','bands'} or 'path' not in mapping:raise ValueError('Map at least path, using supported image fields')
  if set(mapping.values())-{c['name'] for c in schema['columns']}:raise ValueError('Mapping refers to an unknown field')
  p=self.source(record['source'])
  if file_hash(p)!=record['source_sha256']:raise ValueError('Catalog changed; inspect its new schema before reading')
  def q(value):return '"'+value.replace('"','""')+'"'
  with self.sqlite_source(p) as db:
   # Bound each scalar before it reaches Python. Paths are never truncated.
   rows=[]
   columns=[q(name) for name in mapping.values()]
   projection=','.join(f"typeof({col}),length({col}),CASE WHEN typeof({col}) <> 'blob' THEN substr(CAST({col} AS TEXT),1,4097) END" for col in columns)
   for row in db.execute('SELECT '+projection+' FROM main.'+q(table)+' LIMIT 50'):
    converted={}
    for i,key in enumerate(mapping):
     kind,size,value=row[i*3:i*3+3]
     if kind=='blob':
      if key=='path':raise ValueError('The image path must be a text field')
      converted[key]={'type':'binary','bytes':size}
     elif value is not None and len(value)>4096:raise ValueError('A catalog field exceeds 4096 characters; no truncated preview was saved')
     else:converted[key]=value
    rows.append(converted)
  if file_hash(p)!=record['source_sha256']:raise ValueError('Catalog changed during preview; inspect its new revision')
  return self.save('connection_preview',{'source':str(p),'source_sha256':record['source_sha256'],'driver':'sqlite','table':table,'mapping':mapping,
              'rows':rows,'preview':rows,'connection_id':identifier,'paths_opened':False,'read_contract':'Immutable read-only SQLite/GeoPackage snapshot. Image paths are opened only for selected rows during import planning.'},state='preview',parent=identifier)

 def inspect_algorithm(self,path,task):
  if task not in ('lulc','segmentation','detection','anomaly','super_resolution'):raise ValueError('Choose a supported task group')
  if Path(path).suffix.lower()=='.json':
   from algorithm_inventory import inspect_package
   details=inspect_package(self,path,task)
  else:
   from algorithm_inventory import SOURCE_INSPECTION_LIMIT,source_interface
   p=self.source(path,{'.py'},SOURCE_INSPECTION_LIMIT)
   with p.open('rb') as stream:raw=stream.read(SOURCE_INSPECTION_LIMIT+1)
   if len(raw)>SOURCE_INSPECTION_LIMIT:raise ValueError('Source exceeds the declaration inspection budget')
   details={'source':str(p),'source_sha256':hashlib.sha256(raw).hexdigest(),**source_interface(raw)}
  return self.save('algorithm',{**details,'task':task,'source_executed':False,'placement':'onboarding candidate; not available in Training, Inference or pipelines','required_before_publication':['Reviewed license, code and dependencies','Offline runtime and hashed weights','Typed input/output and sensor/class schema','Real adapter execution, controls, progress and validation','Declare supported training and inherit correction policy','Explicit qualified version publication']},state='candidate')
 def verify_algorithm_inventory(self,identifier):
  from algorithm_inventory import inspect_package
  record=self.get(identifier)
  if record['kind']!='algorithm' or record['state'] not in ('candidate','inventory-verified') or not record.get('inventory_schema'):
   raise ValueError('Choose a saved algorithm package inventory')
  checked=inspect_package(self,record['source'],record['task'])
  if checked['source_sha256']!=record['source_sha256'] or checked['artifacts']!=record['artifacts']:
   raise ValueError('The algorithm package changed. Inspect a new candidate.')
  payload={k:v for k,v in record.items() if k not in ('id','kind','state','created_at','parent_id')}
  payload.update(source_interfaces=checked['source_interfaces'],integrity='Manifest and artifact hashes reverified',qualification='pending',source_executed=False)
  return self.save('algorithm',payload,state='inventory-verified',parent=identifier)
 def stage_algorithm(self,identifier,store,request):
  from algorithm_inventory import inspect_package
  from portable_lulc import MODEL,MAX_BYTES,read
  from mission_workspace import now
  _,job=store.mission(request.mission_id,request.job_id)
  if not job.get('images'):raise ValueError('Choose a job with independent validation imagery')
  record=self.get(identifier)
  if record['kind']!='algorithm' or record['state']!='inventory-verified' or record.get('task')!='lulc':
   raise ValueError('Verify a declared LULC algorithm inventory before preparing its runtime')
  checked=inspect_package(self,record['source'],'lulc')
  if checked['source_sha256']!=record['source_sha256'] or checked['artifacts']!=record['artifacts']:
   raise ValueError('The algorithm package changed. Inspect a new candidate.')
  configs=[a for a in checked['artifacts'] if a['role']=='config']
  if len(configs)!=1 or not any(a['role']=='license' for a in checked['artifacts']):
   raise ValueError('The portable adapter requires one declared model JSON and a license file')
  if not request.rights_confirmed or not request.independent_validation:
   raise ValueError('Confirm usage rights and that the validation imagery and labels were not used to fit or tune these weights')
  config=configs[0];identity=config['sha256']
  source=self.source(Path(record['source']).parent/config['path'],{'.json'},MAX_BYTES)
  raw,spec=read(source,identity)
  for label,targets in request.goals.per_class.items():
   if label not in spec['classes'] or any(key not in ('precision','recall','f1') or not 0<=value<=1 for key,value in targets.items()):
    raise ValueError('Per-class goals must name a declared class and valid precision, recall or F1 thresholds')
  identifier_model=str(uuid4())
  folder=self.data/'services'/'onboarded-models'/identifier_model
  if folder.resolve()!=folder:raise ValueError('Portable models require a private non-symlink workspace directory')
  folder.mkdir(parents=True,exist_ok=False)
  checkpoint=folder/'model.json'
  item={'id':identifier_model,'mission_id':request.mission_id,'job_id':request.job_id,
        'algorithm':MODEL,'name':record['name'],'task':'lulc','state':'training','classes':spec['classes'],
        'sensor_suitability':' · '.join(', '.join(spec[key]) for key in ('sensors','processing_levels','bands')),
        'portable_model':{'path':str(checkpoint),'sha256':identity,'inventory_id':identifier,
                          'inventory_sha256':record['source_sha256'],'independence_attested':True,'rights_confirmed':True},
        'qualification_mode':'goals','goals':request.goals.model_dump(),'created_at':now(),'history':[]}
  try:
   checkpoint.write_bytes(raw)
   store.put('workflow_models',item)
  except Exception:
   checkpoint.unlink(missing_ok=True);folder.rmdir()
   raise
  store.event('model.onboarded','Data-only model staged for independent validation',mission_id=request.mission_id,model_id=identifier_model,audit=True)
  return item
 def inspect_api(self,path):
  from algorithm_inventory import bounded_json
  p=self.source(path,{'.json'},2*1024**2);doc,checksum=bounded_json(p,2*1024**2)
  return self.save('api',{'source':str(p),'source_sha256':checksum,**openapi_inventory(doc),'network_requests':0},state='inspected')


def openapi_inventory(document):
 if not isinstance(document,dict) or not re.fullmatch(r'3\.\d+\.\d+(?:[-+][a-zA-Z0-9.-]+)?',str(document.get('openapi',''))):
  raise ValueError('Supply an OpenAPI 3 JSON document. Unknown specifications need a dedicated reviewed parser.')
 paths=document.get('paths',{})
 if not isinstance(paths,dict) or len(paths)>2000:raise ValueError('Use a scoped OpenAPI document with at most 2000 paths')
 operations=[]
 explicit_gets=set()
 for route,item in paths.items():
  if not isinstance(item,dict) or not route.startswith('/') or route.startswith('//') or len(route)>512 or '?' in route:continue
  for method,spec in item.items():
   if method.lower() not in ('get','head','post','put','patch','delete','options','trace'):continue
   ident=spec.get('operationId') if isinstance(spec,dict) else None
   operations.append({'path':route,'method':method.upper(),'operation_id':ident if isinstance(ident,str) and len(ident)<=150 else None,
                       'state':'unverified; execution blocked','reason':'An HTTP method or annotation does not establish a reviewed read-only contract'})
   if method=='get' and isinstance(spec,dict) and '$ref' not in spec and '$ref' not in item:
    explicit_gets.add(route)
   if len(operations)>2000:raise ValueError('Specification exceeds 2000 operations; supply a scoped document')
 adapter_hints=[]
 if '/collections' in explicit_gets and any(re.fullmatch(r'/collections/\{[A-Za-z_][A-Za-z0-9_]*\}/items',p) for p in explicit_gets):
  if '/search' in explicit_gets:adapter_hints.append('stac')
  if '/conformance' in explicit_gets:adapter_hints.extend(('ogc_features','ogc_records'))
 from catalog_arcgis import INVENTORY_DRIVER, SERVICE_PATH
 for route in explicit_gets:
  query=re.fullmatch(rf'({SERVICE_PATH})/\{{[A-Za-z_][A-Za-z0-9_]*\}}/query',route)
  if query and query[1] in explicit_gets:
   adapter_hints.append(INVENTORY_DRIVER)
   break
 return {'spec_version':document['openapi'],'operations':operations,'remote_refs_fetched':False,
         'adapter_hints':adapter_hints,
         'notice':'Operations remain blocked until a reviewed read contract is implemented.'}


def service_root(path):
 """Canonicalize an enrolled directory without accepting replacement links."""
 import sys
 path=Path(path).expanduser()
 if not path.is_absolute() or '..' in path.parts:raise ValueError('Use a specific absolute source directory')
 for part in (path,*path.parents):
  if not (part.is_symlink() or getattr(part,'is_junction',lambda:False)()):continue
  if sys.platform=='darwin' and part in (Path('/var'),Path('/tmp')) and part.resolve()==Path('/private')/part.name:continue
  raise ValueError('Approved source directories cannot be symbolic links or junctions')
 root=path.resolve()
 if root==Path(root.anchor):raise ValueError('Approve a specific directory instead of a volume root')
 return root


SOURCE_HINTS={
 'geolocation':('.tif','.tiff','.jp2','.tfw','.j2w','.wld','.aux.xml','.rpb','.rpc','_rpc.txt'),
 'catalog':('.sqlite','.sqlite3','.db','.gpkg','.csv','.geojson'),
 'algorithm':('.py','.json'),
 'api':('.json',),
}
SEARCH_COMMAND=re.compile(r'(?:find|look for|search for) (?:a |an |the )?(geolocation|triangulation(?: file)?|catalog|algorithm|api)(?: (.{1,160}))?',re.I)


def find_sources(services,kind,query):
 """Search names and approved folder entries. Candidate contents remain unopened."""
 from catalog_connections import DATABASE_ADAPTERS
 from urllib.parse import urlsplit
 candidates=[]
 term=query.casefold()
 checked=0
 truncated=False
 for record in services.records():
  category=record['kind']
  hostname=''
  if category=='connection':
   driver=record['driver']
   category='catalog' if driver in DATABASE_ADAPTERS or driver in ('stac','ogc_features','ogc_records','arcgis_features','arcgis_inventory','snapshot','geojson_snapshot') else 'api'
   hostname=urlsplit(record.get('endpoint','')).hostname or ''
  name=record.get('name') or Path(record.get('source') or '').name
  if category==kind and name and (term in name.casefold() or term in hostname.casefold()):
   candidates.append({'name':name,'saved_id':record['id']})
  if len(candidates)>=20:
   return {'kind':kind,'action':'choose','candidates':candidates,'truncated':True,'network_requests':0}
 for root in services.roots:
  try:
   path=services.source_path(root)
   with os.scandir(path) as entries:
    for entry in entries:
     checked+=1
     if checked>4000 or len(candidates)>=20:
      truncated=True
      break
     name=entry.name;folded=name.casefold()
     dimap_rpc=kind=='geolocation' and folded.startswith('rpc_') and folded.endswith('.xml')
     if name.startswith('.') or term not in folded or not (folded.endswith(SOURCE_HINTS[kind]) or dimap_rpc):continue
     if not entry.is_file(follow_symlinks=False):continue
     try:source=services.source_path(entry.path)
     except (OSError,ValueError):continue
     candidate={'name':name,'path':str(source)}
     if candidate not in candidates:candidates.append(candidate)
  except (OSError,ValueError):
   continue
  if truncated:break
 return {'kind':kind,'action':'choose','candidates':candidates,'truncated':truncated,'network_requests':0}


def discovery_plan(message):
 # An editable draft only. This parser never opens files or resolves hosts.
 from urllib.parse import urlsplit
 if len(message.splitlines())!=1 or len(message)>4096:raise ValueError('Enter one discovery request at a time')
 local=re.fullmatch(r'inspect (geolocation|catalog|algorithm|api) (.+)',message.strip(),re.I)
 if local:
  kind,path=local.group(1).lower(),local.group(2).strip()
  if len(path)>1 and path[0]==path[-1] and path[0] in ('"',"'"):path=path[1:-1]
  from pathlib import PureWindowsPath
  if not (Path(path).is_absolute() or PureWindowsPath(path).is_absolute()) or '://' in path:
   raise ValueError('Use an absolute source file path on the application computer')
  return {'kind':kind,'action':'inspect','path':path,'network_requests':0,'state':'draft'}
 remote=re.fullmatch(r'connect (postgres|mongo|mysql|mariadb|sqlserver|sqlserver2025|oracle|db2|stac|ogc_features|ogc_records|arcgis_features|arcgis_inventory|openapi|mcp) (\S+)',message.strip(),re.I)
 if remote:
  driver,address=remote.group(1).lower(),remote.group(2)
  expected={'postgres':'postgresql','mongo':'mongodb','mysql':'mysql','mariadb':'mariadb','sqlserver':'sqlserver','sqlserver2025':'sqlserver','oracle':'oracle','db2':'db2'}
  if '://' not in address:
   parsed=urlsplit('//'+address)
   try:ipaddress.ip_address(parsed.hostname or '')
   except ValueError as exc:raise ValueError('Enter an IP address or the complete service URL') from exc
   address=expected.get(driver,'https')+'://'+address
  url=urlsplit(address)
  if url.port is not None and not 1<=url.port<=65535:raise ValueError('Use a valid service port')
  if (not url.hostname or url.username is not None or url.password is not None or url.query or url.fragment
      or url.scheme not in ([expected[driver]] if driver in expected else ['http','https'])):
   raise ValueError('Use the adapter’s URL format without embedded credentials, queries or fragments')
  return {'kind':'catalog' if driver in expected or driver in ('stac','ogc_features','ogc_records','arcgis_features','arcgis_inventory') else 'api','action':'connect',
          'driver':driver,'endpoint':address,'network_requests':0,'state':'draft'}
 raise ValueError('Try “inspect algorithm /absolute/model.json” or “connect stac https://internal-host/catalog”')


async def interpret_discovery(message,planner):
 """Draft from an explicitly supplied target, without sending it to the model."""
 from urllib.parse import urlsplit
 if len(message.splitlines())!=1 or len(message)>4096:raise ValueError('Enter one discovery request at a time')
 targets=list(re.finditer(r'''["']([^"']+)["']|([a-z][a-z0-9+.-]*://[^\s<>"']+)''',message,re.I))
 if len(targets)!=1:raise ValueError('Supply one explicit endpoint or one quoted absolute file path')
 span=targets[0]
 target=span.group(1) or span.group(2)
 if '://' in target:
  url=urlsplit(target)
  if not url.hostname or url.username is not None or url.password is not None or url.query or url.fragment:
   raise ValueError('Use a source URL without credentials, queries or fragments')
  action='connect'
  choices=['postgres','mongo','mysql','mariadb','sqlserver','sqlserver2025','oracle','db2','stac','ogc_features','ogc_records','arcgis_features','arcgis_inventory','openapi','mcp']
 else:
  from pathlib import PureWindowsPath
  if not (Path(target).is_absolute() or PureWindowsPath(target).is_absolute()):
   raise ValueError('Use a quoted absolute source file path')
  action='inspect'
  choices=['geolocation','catalog','algorithm','api']
 config=planner.config()
 if (not config['enabled'] or config['provider']!='ollama'
     or urlsplit(config['base_url']).hostname not in ('localhost','127.0.0.1','::1')):
  raise ValueError('Conversational drafts require the configured loopback Ollama service. Typed discovery commands are also available.')
 redacted=message[:span.start()]+'[explicit target]'+message[span.end():]
 response=await planner.request(config,'POST','/api/chat',{'model':config['model'],'stream':False,'keep_alive':0,
     'format':{'type':'object','properties':{'choice':{'enum':choices+['clarify']}},'required':['choice'],'additionalProperties':False},
     'messages':[{'role':'system','content':'Classify this request into one allowed discovery adapter. Return clarify if uncertain. Only classify; never propose targets or operations.'},
                 {'role':'user','content':redacted}],
     'options':{'num_predict':96,'temperature':0},**({'think':False} if config['model'].startswith('qwen3.5:') else {})})
 try:
  raw=response['message']['content']
  value=json.loads(raw) if isinstance(raw,str) and len(raw)<=256 else {}
  choice=value.get('choice') if isinstance(value,dict) and set(value)=={'choice'} else None
 except (KeyError,TypeError,ValueError):
  choice=None
 if choice not in choices:raise ValueError('Choose a reviewed adapter explicitly. No source was contacted.')
 draft=discovery_plan(f'{action} {choice} {target}')
 return {**draft,'interpretation':'loopback-model','model_received_target':False}


async def interpret_lookup(message,planner):
 """The model can choose a local inventory filter, with no connector authority."""
 from urllib.parse import urlsplit
 if len(message.splitlines())!=1 or len(message)>4096:raise ValueError('Enter one discovery request at a time')
 config=planner.config()
 if (not config['enabled'] or config['provider']!='ollama'
     or urlsplit(config['base_url']).hostname not in ('localhost','127.0.0.1','::1')):
  raise ValueError('Use a command such as “find catalog archive” or configure loopback Ollama for conversational requests.')
 response=await planner.request(config,'POST','/api/chat',{'model':config['model'],'stream':False,'keep_alive':0,
    'format':{'type':'object','properties':{'kind':{'enum':[*SOURCE_HINTS,'clarify']},'name':{'type':'string'}},
              'required':['kind','name'],'additionalProperties':False},
    'messages':[{'role':'system','content':'Classify the requested source and copy only its distinctive name or filename. Omit words describing the type, such as imagery catalog, API or triangulation file. Example: Where is our Archive imagery catalog? yields kind catalog and name Archive. Use an empty name if no name was given. Choose clarify if the requested type is unclear.'},
                {'role':'user','content':message}],
    'options':{'num_predict':96,'temperature':0},**({'think':False} if config['model'].startswith('qwen3.5:') else {})})
 try:
  raw=response['message']['content']
  value=json.loads(raw) if isinstance(raw,str) and len(raw)<=1024 else {}
  if not isinstance(value,dict) or set(value)!={'kind','name'}:raise ValueError
  kind,name=value['kind'],value['name']
  if kind not in SOURCE_HINTS or not isinstance(name,str) or len(name)>160 or name.casefold() not in message.casefold():raise ValueError
 except (KeyError,TypeError,ValueError) as exc:
  raise ValueError('Specify a source type and optional name, such as “find catalog archive”.') from exc
 return kind,name.strip()


def register(backend):
 from fastapi import Header,HTTPException,Query
 from fastapi.responses import Response
 from pydantic import BaseModel,Field
 from starlette.concurrency import run_in_threadpool
 app=backend.app
 @app.middleware('http')
 async def protect_services_transport(request,call_next):
  if request.url.path.startswith('/api/services/') and request.url.path!='/api/services/capabilities' and not private_transport(request):
   from fastapi.responses import JSONResponse
   return JSONResponse({'detail':'Private services require loopback or internal HTTPS. A local container proxy must be explicitly configured.'},status_code=403)
  return await call_next(request)
 @app.on_event('startup')
 async def prepare_private_service_access():
  await run_in_threadpool(access_key,backend.DATA_DIR)
 def service():
  backend._init_database()
  with backend._database() as db:row=db.execute('SELECT payload FROM settings WHERE key=?',('service_source_roots',)).fetchone()
  roots=json.loads(os.getenv('GEO_SERVICE_SOURCE_ROOTS','[]')) or (json.loads(row['payload']) if row else []) or [str(backend.LOCAL_RASTER_ROOT),str(backend.DATA_DIR/'local-imagery')]
  return Services(backend._database,backend.DATA_DIR,roots)
 def authorize(value):
  if not value or not secrets.compare_digest(value,access_key(backend.DATA_DIR)):raise HTTPException(401,'Connect with this workspace’s local services access key')
 async def invoke(fn,*args):
  try:return await run_in_threadpool(fn,*args)
  except (ValueError,OSError,SyntaxError,sqlite3.Error) as exc:raise HTTPException(422,str(exc)) from exc
 class Inspect(BaseModel):
  kind:str
  path:str=Field(min_length=1,max_length=4096)
  task:str='lulc'
 class Conversion(BaseModel):
  points:list[list[float]]=Field(min_length=1,max_length=10000)
  direction:str
  mode:str|None=None
  height:float|None=None
 class Recipe(BaseModel):
  path:str
  mapping:dict[str,str]
  checks:list[dict]=Field(min_length=4,max_length=1000)
  tolerance:float=Field(gt=0,le=10)
  coefficient_order:str=Field(default='affine',pattern='^(affine|gdal)$')
 class ReuseRecipe(BaseModel):
  path:str=Field(min_length=1,max_length=4096)
  checks:list[dict]=Field(min_length=4,max_length=1000)
  tolerance:float=Field(gt=0,le=10)
 class AffineSidecar(BaseModel):
  path:str=Field(min_length=1,max_length=4096)
  width:int=Field(gt=0,le=2**31-1)
  height:int=Field(gt=0,le=2**31-1)
  checks:list[dict]=Field(min_length=4,max_length=1000)
  tolerance:float=Field(gt=0,le=10)
 class WorldFile(AffineSidecar):
  crs:str|None=Field(default=None,pattern=r'^EPSG:[1-9][0-9]{2,6}$')
  projection_path:str|None=Field(default=None,min_length=1,max_length=4096)
 class RPCSidecar(AffineSidecar):
  tile_metadata:str|None=Field(default=None,max_length=4096)
  tile_name:str|None=Field(default=None,max_length=255)
 class EmbeddedValidation(BaseModel):
  mode:str=Field(pattern='^(affine|gcp|rpc)$')
  checks:list[dict]=Field(min_length=4,max_length=1000)
  tolerance:float=Field(gt=0,le=10)
 class PixelShapes(BaseModel):
  path:str=Field(min_length=1,max_length=4096)
  pixel_coordinates:bool=False
  mode:str|None=None
  height:float|None=None
 class Mapping(BaseModel):
  table:str
  mapping:dict[str,str]
 class DiscoveryRequest(BaseModel):
  message:str=Field(min_length=1,max_length=4096)
  use_local_model:bool=False
 class RootRequest(BaseModel):
  path:str=Field(min_length=1,max_length=4096)
 @app.get('/api/services/capabilities')
 async def capabilities():
  from catalog_connections import DATABASE_ADAPTERS
  return {'agents':CAPABILITIES,'database_adapters':DATABASE_ADAPTERS,'policy':{'supplier_read_only':True,'knowledge_export':False,'external_llm':False},'authentication':'local workspace services key required','distributed_training':{'independent_models':['random_forest','svm','xgboost','unet'],'single_model':['random_forest','unet'],'neural_multi_node':True,'target_validation':'Native Mac MPS and Linux VM CPU exercised over private TLS on one host; physical multi-host and Windows CUDA remain unverified'}}
 @app.get('/api/services/knowledge')
 async def knowledge(x_geo_services_key:str|None=Header(default=None)):
  authorize(x_geo_services_key);s=service();return {'records':await invoke(s.records),'source_roots':[str(p) for p in s.roots],'roots_locked':bool(os.getenv('GEO_SERVICE_SOURCE_ROOTS'))}
 @app.post('/api/services/source-roots')
 async def add_root(request:RootRequest,x_geo_services_key:str|None=Header(default=None)):
  authorize(x_geo_services_key)
  if os.getenv('GEO_SERVICE_SOURCE_ROOTS'):raise HTTPException(403,'This deployment fixes service source roots; change the local deployment configuration')
  p=await invoke(service_root,request.path)
  if not p.is_dir():raise HTTPException(422,'Choose a specific existing source folder')
  roots=list(dict.fromkeys([str(r) for r in service().roots]+[str(p.resolve())]))
  if len(roots)>32:raise HTTPException(422,'At most 32 explicit read-only source roots are supported')
  with backend._database() as db:db.execute('INSERT INTO settings VALUES(?,?) ON CONFLICT(key) DO UPDATE SET payload=excluded.payload',('service_source_roots',json.dumps(roots)))
  backend.workflow_store.event('service.source_root_added','Read-only service source folder approved locally',path=str(p.resolve()),audit=True)
  return {'source_roots':roots,'access':'read-only'}
 @app.post('/api/services/discovery-plan')
 async def plan(request:DiscoveryRequest,x_geo_services_key:str|None=Header(default=None)):
  authorize(x_geo_services_key)
  lookup=SEARCH_COMMAND.fullmatch(request.message.strip())
  if lookup:
   kind=lookup[1].lower()
   return await invoke(find_sources,service(),'geolocation' if kind.startswith('triangulation') else kind,(lookup[2] or '').strip())
  if not request.use_local_model:return await invoke(discovery_plan,request.message)
  try:return discovery_plan(request.message)
  except ValueError:pass
  from insight_llm import Planner
  planner=Planner(backend._database,backend.DATA_DIR,backend.model_executor)
  try:
   if '://' not in request.message and not any(c in request.message for c in ('"',"'")):
    kind,query=await interpret_lookup(request.message,planner)
    return await invoke(find_sources,service(),kind,query)
   return await interpret_discovery(request.message,planner)
  except ValueError as exc:raise HTTPException(422,str(exc)) from exc
 @app.post('/api/services/inspect')
 async def inspect(request:Inspect,x_geo_services_key:str|None=Header(default=None)):
  authorize(x_geo_services_key);s=service()
  def inspect_catalog():
   from catalog_connections import Connections,FILE_CATALOGS
   if Path(request.path).suffix.lower() in FILE_CATALOGS:
    return Connections(s).snapshot(request.path)
   return s.inspect_catalog(request.path)
  actions={'geolocation':lambda:s.inspect_geolocation(request.path),'catalog':inspect_catalog,'algorithm':lambda:s.inspect_algorithm(request.path,request.task),'api':lambda:s.inspect_api(request.path)}
  if request.kind not in actions:raise HTTPException(422,'Choose one of the four supported services')
  result=await invoke(actions[request.kind]);backend.workflow_store.event('service.inspected','Read-only service inspection saved locally',service=request.kind,configuration=result['id']);return result
 @app.post('/api/services/{identifier}/convert')
 async def convert(identifier:str,request:Conversion,x_geo_services_key:str|None=Header(default=None)):
  authorize(x_geo_services_key);return await invoke(service().convert,identifier,request.points,request.direction,request.mode,request.height)
 @app.post('/api/services/{identifier}/pixel-shapes')
 async def pixel_shapes(identifier:str,request:PixelShapes,x_geo_services_key:str|None=Header(default=None)):
  authorize(x_geo_services_key);result=await invoke(service().convert_pixel_shapes,identifier,request.path,request.pixel_coordinates,request.mode,request.height)
  backend.workflow_store.event('service.vector_converted','Pixel vectors converted to a new geographic GeoJSON',configuration=identifier,output=result['id']);return result
 @app.get('/api/services/{identifier}/vector-output')
 async def vector_output(identifier:str,x_geo_services_key:str|None=Header(default=None)):
  authorize(x_geo_services_key)
  content=await invoke(service().geographic_vector_content,identifier)
  return Response(content,media_type='application/geo+json',
                  headers={'Cache-Control':'no-store','Content-Disposition':'attachment; filename="geographic-detections.geojson"','X-Content-Type-Options':'nosniff'})
 @app.get('/api/services/geographic-vectors')
 async def geographic_vectors(before:str|None=Query(default=None,min_length=1,max_length=36),
                              x_geo_services_key:str|None=Header(default=None)):
  authorize(x_geo_services_key)
  return await invoke(service().geographic_vectors,before)
 @app.post('/api/services/geolocation/recipe')
 async def recipe(request:Recipe,x_geo_services_key:str|None=Header(default=None)):
  authorize(x_geo_services_key);return await invoke(service().propose_affine,request.path,request.mapping,request.checks,request.tolerance,None,request.coefficient_order)
 @app.post('/api/services/geolocation/suggest-affine')
 async def suggest_recipe(request:ReuseRecipe,x_geo_services_key:str|None=Header(default=None)):
  authorize(x_geo_services_key);return await invoke(service().suggest_affine,request.path,request.checks,request.tolerance)
 @app.post('/api/services/{identifier}/verify-algorithm-inventory')
 async def verify_inventory(identifier:str,x_geo_services_key:str|None=Header(default=None)):
  authorize(x_geo_services_key)
  return await invoke(service().verify_algorithm_inventory,identifier)
 from workflow_api import Goals
 class StageAlgorithm(BaseModel):
  mission_id:str
  job_id:str
  rights_confirmed:bool=False
  independent_validation:bool=False
  goals:Goals=Field(default_factory=Goals)
 @app.post('/api/services/{identifier}/stage-algorithm')
 async def stage_algorithm(identifier:str,request:StageAlgorithm,x_geo_services_key:str|None=Header(default=None)):
  authorize(x_geo_services_key)
  return await invoke(service().stage_algorithm,identifier,backend.workflow_store,request)
 @app.post('/api/services/{identifier}/reuse-affine')
 async def reuse_recipe(identifier:str,request:ReuseRecipe,x_geo_services_key:str|None=Header(default=None)):
  authorize(x_geo_services_key)
  return await invoke(service().reuse_affine,identifier,request.path,request.checks,request.tolerance)
 @app.post('/api/services/geolocation/world-file')
 async def world_file(request:WorldFile,x_geo_services_key:str|None=Header(default=None)):
  authorize(x_geo_services_key)
  return await invoke(service().propose_world_file,request.path,request.crs,request.width,request.height,request.checks,request.tolerance,request.projection_path)
 @app.post('/api/services/geolocation/pam-affine')
 async def pam_affine(request:AffineSidecar,x_geo_services_key:str|None=Header(default=None)):
  authorize(x_geo_services_key)
  return await invoke(service().propose_pam_affine,request.path,request.width,request.height,request.checks,request.tolerance)
 @app.post('/api/services/geolocation/pam-geolocation')
 async def pam_geolocation(request:AffineSidecar,x_geo_services_key:str|None=Header(default=None)):
  authorize(x_geo_services_key)
  return await invoke(service().propose_pam_geolocation,request.path,request.width,request.height,request.checks,request.tolerance)
 @app.post('/api/services/geolocation/rpc-sidecar')
 async def rpc_sidecar(request:RPCSidecar,x_geo_services_key:str|None=Header(default=None)):
  authorize(x_geo_services_key)
  return await invoke(service().propose_rpc_sidecar,request.path,request.width,request.height,request.checks,request.tolerance,request.tile_metadata,request.tile_name)
 @app.post('/api/services/{identifier}/validate-embedded')
 async def validate_embedded(identifier:str,request:EmbeddedValidation,x_geo_services_key:str|None=Header(default=None)):
  authorize(x_geo_services_key)
  return await invoke(service().propose_embedded_validation,identifier,request.mode,request.checks,request.tolerance)
 @app.post('/api/services/{identifier}/approve')
 async def approve(identifier:str,x_geo_services_key:str|None=Header(default=None)):
  authorize(x_geo_services_key);return await invoke(service().approve,identifier)
 @app.post('/api/services/{identifier}/catalog-preview')
 async def preview(identifier:str,request:Mapping,x_geo_services_key:str|None=Header(default=None)):
  authorize(x_geo_services_key);return await invoke(service().preview_catalog,identifier,request.table,request.mapping)
