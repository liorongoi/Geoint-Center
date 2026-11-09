"""Known read-only live connectors. No arbitrary SQL, Mongo commands or API calls."""
import importlib.util
import csv,hashlib,importlib,io,ipaddress,json,math,os,re,socket,sys
import time
from pathlib import Path,PureWindowsPath
from contextlib import contextmanager
from datetime import date,datetime,time as utc_time,timezone
from itertools import islice
from urllib.parse import urlsplit
from uuid import uuid4
from agent_services import Services,file_hash
from internal_http import InternalHTTP
import catalog_arcgis

COMPONENT_HASHES={'mac-arm64': 'f539b6fefe2d5e3d699a5c347e4e17135361eff8568c57b481aa5f66d269a252', 'windows-x64': 'bf925a398baae6b251498e1715d0cbe9b8978e66e79cf4960f28dfadd0c9e330', 'linux-arm64': '1006642742e6a7a8832a06950528dda7ea838d9b45d5e5a1f8c9add23894bf45'}
COMPONENT_V2_HASHES={'mac-arm64':'acfa23266e500ba1468a7b43edc3e211c843c1742ee3e2767dd2f6fcaa35d034','windows-x64':'14aec774c745d82c38ad2ad47d452cd1356ba5346cce6c722d30e97859a03bb5','linux-arm64':'9cde77cf389195b4243ab2bcc07f003e825489a58aa76e9556aee07d94ceef2d'}
COMPONENT_V3_HASHES={
 'mac-arm64':'9e83e69ea2d2053964eed017039a79f27ce76bacc558b3b1bc9ddcb303a7ec48',
 'linux-arm64':'97e8843dce3a1bd76c4f085e35587b183260cc99fd405ec0c24284c0ef84a580',
 'windows-x64':'33c95335f499f0723306c398d1f826932b962592520026b1cf4503b9fd8d3849',
}
DATABASE_ADAPTERS={
 'postgres':{'label':'PostgreSQL','example':'postgresql://host:5432/database'},
 'mongo':{'label':'MongoDB','example':'mongodb://host:27017/database'},
 'mysql':{'label':'MySQL 8.4','example':'mysql://host:3306/database'},
 'mariadb':{'label':'MariaDB 11.4','example':'mariadb://host:3306/database','ca_required':True},
 'sqlserver':{'label':'SQL Server 2022','example':'sqlserver://host:1433/database','ca_required':True},
 'sqlserver2025':{'label':'SQL Server 2025','example':'sqlserver://host:1433/database','ca_required':True},
 'oracle':{'label':'Oracle AI Database 26ai','example':'oracle://host:2484/service','ca_required':True},
 'db2':{'label':'IBM Db2 LUW 12.1.3 · Apple Silicon Mac','example':'db2://127.0.0.1:50001/DATABASE','ca_required':True},
}
READ_ACTIONS={'find','listCollections','listCollectionNames','listIndexes','listDatabases','collStats','dbStats','dbHash','indexStats','changeStream','killCursors','planCacheRead','listSearchIndexes','listSearchIndexQueries'}

def postgis_schema(conn, type_oid):
    """Find the exact PostGIS-owned type and serializer overloads."""
    row = conn.execute("""
        SELECT n.nspname
        FROM pg_catalog.pg_extension AS e
        JOIN pg_catalog.pg_namespace AS n ON n.oid = e.extnamespace
        JOIN pg_catalog.pg_type AS t ON t.typnamespace = n.oid
        WHERE e.extname = 'postgis' AND e.extversion ~ '^3[.]5[.][0-9]+$'
          AND t.oid = %s AND t.typname = 'geometry' AND t.typtype = 'b'
          AND EXISTS (
            SELECT 1 FROM pg_catalog.pg_depend d
            WHERE d.classid = 'pg_catalog.pg_type'::pg_catalog.regclass
              AND d.objid = t.oid AND d.deptype = 'e'
              AND d.refclassid = 'pg_catalog.pg_extension'::pg_catalog.regclass
              AND d.refobjid = e.oid)
          AND 2 = (
            SELECT count(*) FROM pg_catalog.pg_proc p
            JOIN pg_catalog.pg_language l ON l.oid = p.prolang
            JOIN pg_catalog.pg_depend d ON d.objid = p.oid
            WHERE p.pronamespace = n.oid AND p.proargtypes[0] = t.oid
              AND l.lanname = 'c' AND p.probin = '$libdir/postgis-3'
              AND p.prokind = 'f' AND p.provolatile = 'i' AND NOT p.prosecdef
              AND p.proconfig IS NULL AND p.provariadic = 0
              AND d.classid = 'pg_catalog.pg_proc'::pg_catalog.regclass
              AND d.deptype = 'e' AND d.refobjid = e.oid
              AND d.refclassid = 'pg_catalog.pg_extension'::pg_catalog.regclass
              AND (
                (p.proname = 'st_srid' AND p.pronargs = 1 AND p.prorettype = 23)
                OR (p.proname = 'st_asgeojson' AND p.pronargs = 3 AND p.prorettype = 25
                    AND p.proargtypes[1] = 23 AND p.proargtypes[2] = 23)))
        """, (type_oid,)).fetchone()
    if not row:
        raise ValueError('Native geometry requires the reviewed PostGIS 3.5 type and serializers')
    return row[0]


def catalog_geometry(value):
    if value is None:
        return None
    if value == '!unsupported-srid':
        raise ValueError('PostGIS preview requires EPSG:4326 geometry. Map an existing WGS84 GeoJSON field for other source CRSs.')
    geometry = json.loads(value)
    if not isinstance(geometry, dict):
        raise ValueError('PostGIS did not return a GeoJSON geometry object')
    depth = {'Point':0, 'MultiPoint':1, 'LineString':1, 'MultiLineString':2,
             'Polygon':2, 'MultiPolygon':3}.get(geometry.get('type'))

    def valid(coords, level):
        if not isinstance(coords, list) or not coords:
            return False
        if level:
            return all(valid(child, level - 1) for child in coords)
        return (len(coords) == 2
                and all(type(item) in (float, int) and math.isfinite(item) for item in coords)
                and -180 <= coords[0] <= 180 and -90 <= coords[1] <= 90)

    if depth is None or not valid(geometry.get('coordinates'), depth):
        raise ValueError('PostGIS catalog geometry must contain finite 2D WGS84 coordinates')
    return {'type': geometry['type'], 'coordinates': geometry['coordinates']}


def verify_ogc_features(http, records=False):
 # Fixed standard route. Returned links never authorize additional operations.
 document=http.json('GET','/conformance')
 classes=document.get('conformsTo') if isinstance(document,dict) else None
 expected={'http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/core'}
 expected.add('http://www.opengis.net/spec/ogcapi-records-1/1.0/conf/records-api' if records
              else 'http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/geojson')
 if not isinstance(classes,list) or len(classes)>100 or not all(isinstance(x,str) for x in classes) or not expected.issubset(classes):
  raise ValueError('The configured API must declare the required OGC conformance classes')
 return sorted(expected)


def driver(name):
 import platform
 if name not in ('psycopg','pymongo','pymysql','pytds','OpenSSL','oracledb'):raise ValueError('Driver is outside the reviewed offline catalog component')
 tag=('windows-x64' if sys.platform=='win32' else 'mac-arm64' if sys.platform=='darwin' else 'linux-arm64' if platform.machine() in ('aarch64','arm64') else 'linux-x64')
 runtime=Path(sys.executable).resolve()
 roots=[os.getenv('GEO_CATALOG_COMPONENT')]
 for parent in (runtime.parent.parent,runtime.parent.parent.parent):
  roots.extend(str(parent/'components'/component) for component in ('catalog-connectors-1.2.0','catalog-connectors'))
 root=next((Path(p).resolve() for p in roots if p and (Path(p)/'component.json').is_file()),None)
 if root is None:raise ValueError('Install the matching offline catalog-connectors component; runtime package downloads are forbidden')
 identity=file_hash(root/'component.json')
 if identity not in (COMPONENT_HASHES.get(tag),COMPONENT_V2_HASHES.get(tag),COMPONENT_V3_HASHES.get(tag)):raise ValueError('Catalog connector component differs from the pinned platform inventory')
 if name=='pymysql' and identity not in (COMPONENT_V2_HASHES.get(tag),COMPONENT_V3_HASHES.get(tag)):raise ValueError('MySQL requires offline catalog connector component 1.1.0 or newer')
 if name in ('pytds','OpenSSL','oracledb') and identity!=COMPONENT_V3_HASHES.get(tag):raise ValueError('SQL Server and Oracle require offline catalog connector component 1.2.0')
 manifest=json.loads((root/'component.json').read_text())
 for entry in manifest['files']:
  path=root/entry['path']
  if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file() or file_hash(path)!=entry['sha256']:raise ValueError('Offline connector component is missing or changed')
 location=str(root/'python')
 if location not in sys.path:sys.path.insert(0,location)
 spec=importlib.util.find_spec(name)
 if spec is None or not spec.origin or not Path(spec.origin).resolve().is_relative_to(Path(location)):raise ValueError('A different database driver is already loaded or shadowing the verified component; restart with the correct offline runtime')
 return importlib.import_module(name)

def db_endpoint(endpoint,scheme):
 p=urlsplit(endpoint)
 if p.scheme!=scheme or not p.hostname or p.username is not None or p.password is not None or p.query or p.fragment or '\\' in endpoint or '?' in endpoint or '#' in endpoint or not p.path.strip('/'):raise ValueError('Use an internal '+scheme+'://host:port/database endpoint; put credentials in the protected fields')
 if not re.fullmatch(r'[A-Za-z0-9_.-]{1,120}',p.path.strip('/')):raise ValueError('Invalid database name')
 port=p.port or {'postgresql':5432,'mongodb':27017,'mysql':3306,'mariadb':3306,'sqlserver':1433,'oracle':2484}[scheme];addresses=list(dict.fromkeys(r[4][0] for r in socket.getaddrinfo(p.hostname,port,type=socket.SOCK_STREAM)))
 if not addresses or any(not (ipaddress.ip_address(a).is_loopback or ipaddress.ip_address(a).is_private) or ipaddress.ip_address(a).is_link_local or ipaddress.ip_address(a).is_multicast or ipaddress.ip_address(a).is_unspecified for a in addresses):raise ValueError('Database connections must stay inside the approved private system boundary')
 strict=os.getenv('SPECTRASCOPE_OFFLINE','').lower() in ('1','true','yes')
 if strict and any(not ipaddress.ip_address(a).is_loopback for a in addresses):raise ValueError('Strict offline-only launch blocks LAN services; use an explicitly enabled internal-network launch')
 return p,addresses[0],port

FILE_CATALOGS = {'.csv', '.geojson'}
FILE_DRIVERS = {'snapshot', 'geojson_snapshot'}
CATALOG_BYTES = 2 * 1024**2
MCP_CATALOG_DRIVER = 'mcp_geojson'
STAC_MOUNT_DRIVER = 'stac_mount'
STAC_MAX_MAPPED_BANDS = 128
STAC_MAX_INVENTORY_KEYS = 512
OGC_CATALOG_DRIVERS = {'ogc_features','ogc_records'}


def stac_mount_mapping(mapping):
 path=mapping['path']
 if (set(mapping)-{'path','image_id','datetime','geometry'}
     or any(mapping.get(role,field)!=field for role,field in
            (('image_id','id'),('datetime','properties.datetime'),('geometry','geometry')))):
  raise ValueError('Use the optional STAC fields id, properties.datetime and geometry')
 if re.fullmatch(r'(?:properties\.[A-Za-z0-9_:-]{1,100}|assets\.[A-Za-z0-9_:-]{1,100}\.href)',path):
  return path.split('.')
 entries=path.split(',')
 if not 2<=len(entries)<=STAC_MAX_MAPPED_BANDS or len(path)>24576:
  raise ValueError(f'Choose 2–{STAC_MAX_MAPPED_BANDS} distinct science assets for a band bundle')
 bands={};keys=set()
 for entry in entries:
  match=re.fullmatch(r'([A-Za-z][A-Za-z0-9_]{0,39})\s*=\s*assets\.([A-Za-z0-9_:-]{1,100})\.href',entry.strip())
  if not match or match[1].upper() in bands or match[2] in keys:
   raise ValueError('Map a mounted path or use distinct band mappings such as RED=assets.red.href, GREEN=assets.green.href')
  bands[match[1].upper()]=match[2];keys.add(match[2])
 return bands


def stac_band_asset(asset,band,key):
 if not isinstance(asset,dict):return None
 roles=asset.get('roles',[])
 if (not isinstance(roles,list) or any(not isinstance(role,str) for role in roles)
     or {role.lower() for role in roles}&{'thumbnail','overview','metadata','quality','mask','visual'}):return None
 path=stac_mounted_path(asset.get('href'))
 if not path:return None
 record={'name':band,'asset_key':key,'path':path}
 for field,prefix in (('raster:bands',''),('bands','raster:')):
  if field not in asset:continue
  values=asset[field]
  if not isinstance(values,list) or len(values)!=1 or not isinstance(values[0],dict):
   raise ValueError('Each bundled STAC asset must declare at most one raster band')
  for name in ('scale','offset'):
   if prefix+name not in values[0]:continue
   value=values[0][prefix+name]
   if (type(value) not in (int,float) or not -1e20<=value<=1e20
       or name=='scale' and value==0 or name in record and record[name]!=value):
    raise ValueError('Invalid or conflicting STAC band calibration')
   record[name]=value
  if 'nodata' in values[0]:
   value=values[0]['nodata']
   if (not (type(value) in (int,float) and -1e20<=value<=1e20 or value=='nan')
       or 'nodata' in record and record['nodata']!=value):
    raise ValueError('Invalid or conflicting STAC band nodata')
   record['nodata']=value
 for field,prefix in (('eo:bands',''),('bands','eo:')):
  if field not in asset:continue
  values=asset[field]
  if not isinstance(values,list) or len(values)!=1 or not isinstance(values[0],dict):
   raise ValueError('Each bundled STAC asset must describe one spectral band')
  for name,target in (('center_wavelength','wavelength_nm'),('full_width_half_max','fwhm_nm')):
   if prefix+name not in values[0]:continue
   value=values[0][prefix+name]
   if type(value) not in (int,float) or not 0<value<=1000:
    raise ValueError('STAC spectral wavelengths and bandwidths must be positive micrometre values')
   nm=value*1000
   if target in record and not math.isclose(record[target],nm,rel_tol=0,abs_tol=1e-9):
    raise ValueError('Conflicting STAC spectral declarations for the same asset')
   record[target]=nm
 return record


def stac_packed_asset(asset,key):
 """Project asset-level declarations onto the actual channel order."""
 declarations={field:asset[field] for field in ('bands','eo:bands','raster:bands') if field in asset}
 counts={len(values) for values in declarations.values() if isinstance(values,list)}
 if not counts or max(counts)<=1:return None
 count=max(counts)
 if not 2<=count<=512 or any(not isinstance(values,list) or len(values)!=count
                            or not all(isinstance(value,dict) for value in values) for values in declarations.values()):
  raise ValueError('Packed STAC metadata must describe the same ordered set of 2–512 channels')
 channels=[]
 for index in range(count):
  pieces={field:[values[index]] for field,values in declarations.items()}
  if any('name' in values[0] and not isinstance(values[0]['name'],str)
         for field,values in pieces.items() if field!='raster:bands'):
   raise ValueError('Packed STAC channel names must be short band identifiers')
  names={values[0]['name'] for field,values in pieces.items() if field!='raster:bands'
         and isinstance(values[0].get('name'),str)}
  if len(names)>1:raise ValueError('Conflicting STAC channel names at the same index')
  name=next(iter(names),None)
  if name is not None and not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,39}',name):
   raise ValueError('Packed STAC channel names must be short band identifiers')
  band=stac_band_asset({'href':asset['href'],'roles':asset.get('roles',[]),**pieces},name,key)
  if band is None:raise ValueError('Use a mounted science asset for packed STAC channels')
  channels.append({field:band[field] for field in ('name','wavelength_nm','fwhm_nm','scale','offset','nodata') if band.get(field) is not None})
 return channels


def stac_mount_rows(features,mapping,parts):
 rows=[];identifiers=set()
 for item in features:
  if (item.get('stac_version') not in ('1.0.0','1.1.0')
      or not isinstance(item.get('id'),str) or not 1<=len(item['id'])<=256 or item['id'] in identifiers
      or not isinstance(item.get('assets'),dict)
      or 'geometry' not in item):
   raise ValueError('Expected distinct STAC 1.0/1.1 items from the selected collection')
  identifiers.add(item['id'])
  if isinstance(parts,dict):
   assets=[stac_band_asset(item['assets'].get(key),band,key) for band,key in parts.items()]
   usable=all(assets)
   row={'bands':assets if usable else [],'item_id':item['id'],
        'path_status':f'{len(assets)} mounted science assets' if usable else 'A selected science asset is unavailable locally'}
  else:
   path=item
   for key in parts:path=path.get(key) if isinstance(path,dict) else None
   path=stac_mounted_path(path)
   row={'path':path,'path_status':'Ready to map to local storage' if path else 'No mounted path at this field'}
   if path and parts[0]=='assets':
    channels=stac_packed_asset(item['assets'][parts[1]],parts[1])
    if channels:
     row.update(item_id=item['id'],asset_key=parts[1],packed_channels=channels,
                path_status=f'{len(channels)} declared channels in one mounted raster')
  if 'image_id' in mapping:row['image_id']=item['id']
  if 'datetime' in mapping:
   timestamp=item['properties'].get('datetime')
   if timestamp is not None and (not isinstance(timestamp,str) or len(timestamp)>100):raise ValueError('Invalid STAC datetime')
   row['datetime']=timestamp
  if 'geometry' in mapping:row['geometry']=catalog_geometry(json.dumps(item['geometry'])) if item['geometry'] is not None else None
  rows.append(row)
 return rows


def stac_mounted_path(value):
 if (not isinstance(value,str) or not 1<=len(value)<=4096 or value!=value.strip()
     or any(ord(c)<32 or c in '?#' for c in value)):
  return None
 normalized=value.replace('\\','/')
 local=normalized[2:] if re.match(r'^[A-Za-z]:/',normalized) else normalized
 if ':' in local or local.lower().startswith('/vsi') or '..' in local.split('/'):
  return None
 return value


def stac_asset_inventory(items,collection,limit):
 """Retain identifiers and asset keys, never arbitrary STAC item content."""
 rows=[]
 for item in items[:limit]:
  if not isinstance(item,dict) or item.get('collection',collection)!=collection:
   continue
  identifier=item.get('id')
  if not isinstance(identifier,str) or not 1<=len(identifier)<=256 or any(ord(c)<32 for c in identifier):
   continue
  assets=item.get('assets')
  keys=list(assets) if isinstance(assets,dict) else item.get('asset_keys')
  if not isinstance(keys,list) or len(keys)>STAC_MAX_INVENTORY_KEYS:
   continue
  rows.append({'id':identifier,'asset_keys':sorted({key for key in keys if isinstance(key,str)
              and re.fullmatch(r'[A-Za-z0-9_:-]{1,100}',key)})})
 return rows


def utc_date_window(start,end):
 if not all(isinstance(value,str) and re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}',value) for value in (start,end)):
  raise ValueError('Supply both UTC dates in YYYY-MM-DD format')
 first,last=(date.fromisoformat(value) for value in (start,end))
 if first>last:raise ValueError('The end date must be on or after the start date')
 return (datetime.combine(first,utc_time.min,timezone.utc),
         datetime.combine(last,utc_time.max,timezone.utc))


def stac_collection_features(document,collection,limit,filtered):
 if (not isinstance(document,dict) or document.get('type')!='FeatureCollection'
     or 'crs' in document or not isinstance(document.get('features'),list)):
  raise ValueError('Expected a CRS84 STAC ItemCollection')
 features=document['features']
 if len(features)>limit:
  raise ValueError('The STAC supplier did not honor the bounded preview limit')
 for item in features:
  if (not isinstance(item,dict) or item.get('type')!='Feature' or 'crs' in item
      or item.get('collection',None if filtered else collection)!=collection
      or not isinstance(item.get('properties'),dict)):
   raise ValueError('Expected CRS84 STAC items from the selected collection')
 return features


def catalog_timestamp(value,utc_only=False):
 pattern=r'[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt][0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.([0-9]{1,9}))?(?:[Zz]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])'
 match=re.fullmatch(pattern,value) if isinstance(value,str) else None
 if not match or utc_only and not value.upper().endswith('Z'):
  raise ValueError('Filtered records require a valid timezone-qualified timestamp in the catalog time convention')
 nanoseconds=int((match[1] or '').ljust(9,'0')[6:])
 return datetime.fromisoformat(value.upper()),nanoseconds


def stac_verify_window(features,window):
 first,last=(window[0],0),(window[1],999)
 for item in features:
  properties=item['properties']
  instant=properties.get('datetime')
  if instant is not None:
   start=end=catalog_timestamp(instant)
  else:
   start,end=(catalog_timestamp(properties.get(key)) for key in ('start_datetime','end_datetime'))
  if start>end or start>last or end<first:
   raise ValueError('The STAC supplier returned an item outside the requested UTC date window')


def ogc_record_time(feature,window):
 value=feature.get('time')
 if not isinstance(value,dict):raise ValueError('Date-filtered OGC Records need a time object')
 time_fields={key:value[key] for key in ('date','timestamp','interval') if key in value}
 if not time_fields:raise ValueError('Date-filtered OGC Records need a date, timestamp or interval')
 day=utc_date_window(value['date'],value['date']) if 'date' in value else None
 instant=catalog_timestamp(value['timestamp'],utc_only=True) if 'timestamp' in value else None
 if day and instant and day[0].date()!=instant[0].date():
  raise ValueError('OGC record date and timestamp disagree')
 if 'interval' in value:
  interval=value['interval']
  if not isinstance(interval,list) or len(interval)!=2 or not all(isinstance(part,str) for part in interval) or interval==['..','..']:
   raise ValueError('OGC record time requires a two-endpoint interval with at least one bounded end')
  bounded=[part for part in interval if part!='..']
  date_only=[bool(re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}',part)) for part in bounded]
  if len(set(date_only))>1:raise ValueError('OGC record interval endpoints must have the same precision')
  points=[]
  for i,part in enumerate(interval):
   if part=='..':points.append(None)
   elif date_only[0]:points.append((utc_date_window(part,part)[i],999 if i else 0))
   else:points.append(catalog_timestamp(part,utc_only=True))
  start,end=points
  if start is not None and end is not None and start>end:
   raise ValueError('OGC record interval ends before it starts')
  if instant and (start is not None and instant<start or end is not None and instant>end):
   raise ValueError('OGC record timestamp is outside its interval')
  if day and (start is not None and start>(day[1],999) or end is not None and end<(day[0],0)):
   raise ValueError('OGC record date is outside its interval')
 elif instant:
  start=end=instant
 else:
  start,end=(day[0],0),(day[1],999)
 if start is not None and start>(window[1],999) or end is not None and end<(window[0],0):
  raise ValueError('The OGC supplier returned a record outside the requested UTC date window')
 return time_fields


def catalog_preview_bounds(bbox):
 if (len(bbox)!=4 or any(type(value) not in (float,int) or not math.isfinite(value) for value in bbox)
     or not (-180<=bbox[0]<bbox[2]<=180 and -90<=bbox[1]<bbox[3]<=90)):
  raise ValueError('Use a non-wrapping WGS84 rectangle with ordered west, south, east and north bounds')
 return bbox


def verify_catalog_bounds(features,bbox):
 from workflow_graph import geometries_intersect,polygon
 area=polygon(bbox)
 for item in features:
  raw=item.get('geometry') if isinstance(item,dict) else None
  if not isinstance(raw,dict) or raw.get('type') not in ('Point','Polygon','MultiPolygon'):
   raise ValueError('Area-filtered imagery needs a WGS84 point or polygon footprint')
  geometry=catalog_geometry(json.dumps(raw))
  if geometry['type']!='Point':
   polygons=[geometry['coordinates']] if geometry['type']=='Polygon' else geometry['coordinates']
   for part in polygons:
    for ring in part:
     if len(ring)<4 or ring[0]!=ring[-1] or any(abs(a[0]-b[0])>180 for a,b in zip(ring,ring[1:])):
      raise ValueError('Area-filtered footprints need closed rings split at the antimeridian')
  if not geometries_intersect(geometry,area):
   raise ValueError('The supplier returned a footprint outside the requested rectangle')


def geojson_snapshot(raw):
 try:
  doc=json.loads(raw.decode('utf-8-sig'))
 except (UnicodeError, RecursionError) as exc:
  raise ValueError('Expected a UTF-8 GeoJSON FeatureCollection') from exc
 if not isinstance(doc,dict) or doc.get('type')!='FeatureCollection' or 'crs' in doc or not isinstance(doc.get('features'),list) or len(doc['features'])>1000:
  raise ValueError('Expected a CRS84 GeoJSON FeatureCollection with at most 1,000 features')
 columns={'@id':'identifier','@geometry':'CRS84 geometry'};rows=[]
 for feature in doc['features']:
  if not isinstance(feature,dict) or feature.get('type')!='Feature' or 'crs' in feature or not isinstance(feature.get('properties'),dict):
   raise ValueError('Invalid GeoJSON catalog feature')
  properties=feature['properties']
  for name in properties:
   if not name.strip() or len(name)>120 or name.startswith('@') or any(ord(ch)<32 for ch in name):
    raise ValueError('Invalid GeoJSON catalog property name')
   columns[name]='property'
   if len(columns)>100:raise ValueError('Catalog exceeds 100 fields')
  identifier=feature.get('id')
  if identifier is not None and type(identifier) not in (str,int,float):raise ValueError('Invalid GeoJSON feature identifier')
  geometry=feature.get('geometry')
  if geometry is not None:geometry=catalog_geometry(json.dumps(geometry))
  rows.append({**properties,'@id':identifier,'@geometry':geometry})
 try:json.dumps(rows,allow_nan=False)
 except (ValueError,RecursionError) as exc:raise ValueError('GeoJSON properties must contain bounded JSON values') from exc
 return columns,rows


def mcp_geojson(contents, uri):
 if not isinstance(contents,list) or len(contents)!=1 or not isinstance(contents[0],dict):
  raise ValueError('A GeoJSON catalog resource must return one text item')
 item=contents[0]
 if item.get('uri')!=uri or item.get('mimeType')!='application/geo+json' or 'blob' in item or not isinstance(item.get('text'),str):
  raise ValueError('Expected one application/geo+json text item for the selected resource')
 raw=item['text'].encode('utf-8')
 if len(raw)>CATALOG_BYTES:
  raise ValueError('GeoJSON catalog resource exceeds 2 MiB')
 columns,rows=geojson_snapshot(raw)
 return columns,rows,hashlib.sha256(raw).hexdigest()


class Connections:
 def __init__(self,services):self.services=services;self.database=services.database;self.data=services.data
 def snapshot(self,path):
  source,identity,columns,_=self._snapshot(path,0)
  geojson=source.suffix.lower()=='.geojson'
  format_name='GeoJSON' if geojson else 'UTF-8 CSV'
  content={'tables':[{'schema':'snapshot','name':'imagery','columns':[{'name':name,'type':kind} for name,kind in columns.items()]}],'read_contract':f'Existing {format_name} catalog export, read-only and checked by hash. Image paths are opened only after row selection.'}
  return self.services.save('connection',{'name':source.name,'driver':'geojson_snapshot' if geojson else 'snapshot','source':str(source),'source_sha256':identity,'schema_sha256':hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest(),**content},state='inspected-read-only')
 def _snapshot(self,path,limit,expected=None):
  source=self.services.source(path,FILE_CATALOGS,CATALOG_BYTES)
  with source.open('rb') as stream:raw=stream.read(CATALOG_BYTES+1)
  if len(raw)>CATALOG_BYTES:raise ValueError('Catalog export exceeds 2 MiB; supply a narrower existing export')
  identity=hashlib.sha256(raw).hexdigest()
  if expected and identity!=expected:raise ValueError('Catalog export changed; inspect it again')
  if source.suffix.lower()=='.geojson':
   columns,rows=geojson_snapshot(raw)
   return source,identity,columns,rows[:limit]
  reader=csv.reader(io.StringIO(raw.decode('utf-8-sig'),newline=''),strict=True)
  try:
   columns=next(reader,[])
   if not 1<=len(columns)<=100 or len(set(columns))!=len(columns) or any(not c.strip() or len(c)>120 or any(ord(ch)<32 for ch in c) for c in columns):raise ValueError('Catalog CSV needs 1–100 unique named columns')
   rows=[]
   for row in islice(reader,limit):
    if len(row)!=len(columns) or any(len(v)>65536 or '\x00' in v for v in row):raise ValueError('Catalog CSV has an invalid row or oversized field')
    rows.append(dict(zip(columns,row)))
  except csv.Error as exc:raise ValueError('Invalid UTF-8 CSV catalog export') from exc
  return source,identity,dict.fromkeys(columns,'text'),rows
 def secret(self,value=None,identifier=None):
  from cryptography.fernet import Fernet
  folder=self.data/'services';folder.mkdir(parents=True,exist_ok=True);path=folder/'credentials.key'
  if identifier and not path.exists():raise ValueError('Credential encryption key is missing; restore its private backup. A replacement key will not be generated.')
  try:
   with path.open('xb') as stream:stream.write(Fernet.generate_key())
   path.chmod(0o600)
  except FileExistsError:pass
  if path.is_symlink():raise ValueError('Credential key cannot be a symbolic link')
  cipher=Fernet(path.read_bytes())
  with self.database() as db:
   db.execute('CREATE TABLE IF NOT EXISTS service_credentials(id TEXT PRIMARY KEY,value BLOB NOT NULL)')
   if value is not None:
    identifier=str(uuid4());db.execute('INSERT INTO service_credentials VALUES(?,?)',(identifier,cipher.encrypt(json.dumps(value).encode())));return identifier
   row=db.execute('SELECT value FROM service_credentials WHERE id=?',(identifier,)).fetchone()
  if not row:raise ValueError('Saved connection credential is unavailable')
  return json.loads(cipher.decrypt(row['value']))
 def headers(self,credential):
  scheme=credential.get('auth','none');token=credential.get('token','')
  if any(c in token for c in ('\r','\n')):raise ValueError('Invalid authentication token')
  if scheme not in ('none','bearer','api_key'):raise ValueError('Choose no authentication, bearer token or API key')
  return {'Authorization':'Bearer '+token} if scheme=='bearer' else {'X-API-Key':token} if scheme=='api_key' else {}
 @contextmanager
 def postgres(self,endpoint,credential,ca_file=None):
  pg=driver('psycopg');p,address,port=db_endpoint(endpoint,'postgresql');local=ipaddress.ip_address(address).is_loopback
  if not credential.get('username'):raise ValueError('Use an existing read-only PostgreSQL account')
  kwargs={'host':p.hostname,'hostaddr':address,'port':port,'dbname':p.path.strip('/'),'user':credential['username'],'password':credential.get('password',''),'connect_timeout':8,'options':'-c default_transaction_read_only=on -c statement_timeout=5000 -c lock_timeout=2000 -c search_path=pg_catalog','sslmode':'disable' if local else 'verify-full','prepare_threshold':None}
  if ca_file:kwargs['sslrootcert']=str(self.services.source(ca_file,{'.pem','.crt'},2*1024**2))
  conn=pg.connect(**kwargs)
  try:
   conn.read_only=True
   role=conn.execute('SELECT rolsuper,rolcreatedb,rolcreaterole,rolreplication FROM pg_catalog.pg_roles WHERE rolname=current_user').fetchone()
   if not role or any(role):raise ValueError('Privileged database accounts are blocked; supply an existing least-privilege read-only account')
   if conn.execute("SELECT current_setting('transaction_read_only')").fetchone()[0]!='on':raise ValueError('PostgreSQL did not enforce a read-only transaction')
   yield conn
  finally:conn.rollback();conn.close()
 @contextmanager
 def mongo(self,endpoint,credential,ca_file=None):
  mongo=driver('pymongo');p,address,port=db_endpoint(endpoint,'mongodb');local=ipaddress.ip_address(address).is_loopback
  if not credential.get('username'):raise ValueError('MongoDB requires an existing authenticated read-only account')
  kwargs={'host':address,'port':port,'username':credential['username'],'password':credential.get('password',''),'authSource':credential.get('auth_database') or p.path.strip('/'),'directConnection':True,'retryWrites':False,'retryReads':False,'serverSelectionTimeoutMS':8000,'connectTimeoutMS':8000,'socketTimeoutMS':6000,'maxPoolSize':1,'tls':not local}
  if ca_file:kwargs['tlsCAFile']=str(self.services.source(ca_file,{'.pem','.crt'},2*1024**2))
  client=mongo.MongoClient(**kwargs)
  try:
   status=client[p.path.strip('/')].command({'connectionStatus':1,'showPrivileges':True});info=status.get('authInfo',{});privileges=info.get('authenticatedUserPrivileges')
   if not info.get('authenticatedUsers') or not isinstance(privileges,list) or not privileges:raise ValueError('MongoDB did not return verifiable authenticated read privileges')
   actions={a for item in privileges for a in item.get('actions',[])}
   if actions-READ_ACTIONS:raise ValueError('MongoDB account has unapproved/write privileges; use a least-privilege read-only account')
   yield client[p.path.strip('/')]
  finally:client.close()
 def inspect(self,kind,endpoint,credentials,name,ca_file=None,*,validate_only=False):
  if not name.strip() or len(name)>120:raise ValueError('Name this internal connection')
  if validate_only and kind!='db2':raise ValueError('Validation without saving is available for Db2')
  if kind==catalog_arcgis.INVENTORY_DRIVER:
   ca=str(self.services.source(ca_file,{'.pem','.crt'},2*1024**2)) if ca_file else None
   content=catalog_arcgis.inventory(catalog_arcgis.client(endpoint,self.headers(credentials),ca,inventory=True))
   return self.services.save('connection',{'name':name,'driver':kind,'endpoint':endpoint,'ca_file':ca_file,
       'read_contract':'One FeatureServer metadata GET. Each layer must be verified separately before querying.',**content},state='inspected-read-only')
  if kind==catalog_arcgis.DRIVER:
   ca=str(self.services.source(ca_file,{'.pem','.crt'},2*1024**2)) if ca_file else None
   layer=catalog_arcgis.schema(catalog_arcgis.client(endpoint,self.headers(credentials),ca))
   content={'arcgis_layer':layer,'tables':[{'schema':'layer','name':'imagery','columns':layer['columns']}],
            'read_contract':catalog_arcgis.READ_CONTRACT}
  elif kind=='mcp':
   from mcp_readonly import ReadOnlyMCP
   ca=str(self.services.source(ca_file,{'.pem','.crt'},2*1024**2)) if ca_file else None;client=ReadOnlyMCP(endpoint,self.headers(credentials),ca);content=client.discover();credentials={**credentials,'mcp_session':client.session,'mcp_version':client.version};content['read_contract']='MCP initialization, discovery and explicitly selected listed-resource reads only. Tool calls/sampling/client requests are blocked.'
  elif kind in ('stac','ogc_features','ogc_records','openapi'):
   ca=str(self.services.source(ca_file,{'.pem','.crt'},2*1024**2)) if ca_file else None;http=InternalHTTP(endpoint,self.headers(credentials),ca,timeout=10)
   conformance=verify_ogc_features(http,records=kind=='ogc_records') if kind in ('ogc_features','ogc_records') else None
   document=http.json('GET','/openapi.json' if kind=='openapi' else '/collections')
   if kind!='openapi':
    collections=document.get('collections')
    if not isinstance(collections,list) or len(collections)>1000:raise ValueError('Not a bounded STAC collections document')
    discovered=[{'id':c['id'],'title':c.get('title',c['id'])} for c in collections if isinstance(c,dict) and isinstance(c.get('id'),str) and (kind!='ogc_records' or c.get('itemType')=='record')]
    content={'collections':discovered,'read_contract':'STAC collections and collection-items GET; no arbitrary operation execution'}
    if kind=='ogc_features':
     content.update(conformance=conformance,read_contract='OGC API Features Core fixed GET reads for conformance, collections and one bounded GeoJSON items page. Supplier links are ignored.')
    if kind=='ogc_records':
     content.update(conformance=conformance,read_contract='OGC API Records 1.0 fixed GET reads for conformance, record collections and one bounded GeoJSON records page. Supplier links are ignored.')
   else:
    from agent_services import openapi_inventory
    content={**openapi_inventory(document),'openapi':document['openapi'],
             'read_contract':'OpenAPI document retrieval only; discovered operations remain blocked'}
  elif kind=='postgres':
   with self.postgres(endpoint,credentials,ca_file) as conn:
    rows=conn.execute("SELECT n.nspname,c.relname,c.oid,pg_catalog.has_table_privilege(c.oid,'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER') FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace WHERE c.relkind='r' AND NOT c.relrowsecurity AND n.nspname NOT IN ('pg_catalog','information_schema') AND n.nspname NOT LIKE 'pg_toast%' AND pg_catalog.has_table_privilege(c.oid,'SELECT') LIMIT 501").fetchall()
    if len(rows)>500:raise ValueError('Database exposes too many tables; supply a narrower read-only role')
    tables=[]
    for schema,table,oid,writable in rows:
     if writable:raise ValueError('Catalog role has write privileges on a visible table; connection blocked')
     columns=[{'name':r[0],'type':r[1]} for r in conn.execute("SELECT attname,pg_catalog.format_type(atttypid,atttypmod) FROM pg_catalog.pg_attribute WHERE attrelid=%s AND attnum>0 AND NOT attisdropped AND attgenerated='' ORDER BY attnum",(oid,)).fetchall()]
     tables.append({'schema':schema,'name':table,'columns':columns})
    content={'tables':tables,'read_contract':'Read-only PostgreSQL transaction, nonprivileged read-only role, base tables only; views/foreign/RLS/generated columns excluded'}
  elif kind in ('mysql','mariadb'):
   adapter=importlib.import_module('catalog_'+kind)
   with adapter.session(self.services,endpoint,credentials,ca_file) as (conn,database):content={'tables':adapter.tables(conn,database),'read_contract':adapter.READ_CONTRACT}
  elif kind in ('sqlserver','sqlserver2025','oracle'):
   from catalog_commercial import request
   content=request(self.services,kind,endpoint,credentials,ca_file)
  elif kind=='db2':
   from catalog_db2 import catalog_request
   content=catalog_request(self.services,endpoint,credentials,ca_file)
   if validate_only:return {'driver':kind,'state':'validated-read-only',**content}
  elif kind=='mongo':
   with self.mongo(endpoint,credentials,ca_file) as db:
    collections=[]
    for item in db.list_collections(filter={'type':'collection'},nameOnly=False):
     if len(collections)>=500:raise ValueError('Too many collections; use a narrower account/database')
     if not item['name'].startswith('system.'):collections.append({'id':item['name']})
    content={'collections':collections,'read_contract':'Authenticated privilege allowlist; find/list only, no aggregation, JavaScript or arbitrary commands'}
  else:raise ValueError('No verified adapter for this database engine. Use an existing read-only STAC catalog or request a new engine contract.')
  secret_id=self.secret(value=credentials)
  ca_hash=content['ca_sha256'] if kind=='db2' else file_hash(self.services.source(ca_file,{'.pem','.crt'},2*1024**2)) if ca_file else None
  return self.services.save('connection',{'name':name,'driver':kind,'endpoint':endpoint,'ca_file':ca_file,'ca_sha256':ca_hash,'credential_id':secret_id,'schema_sha256':__import__('hashlib').sha256(json.dumps(content,sort_keys=True).encode()).hexdigest(),**content},state='connected-read-only')
 def preview(self,identifier,table,mapping,limit=50,item_id=None,start_date=None,end_date=None,bbox=None):
  r=self.services.get(identifier)
  mcp_catalog=r['kind']=='mcp_resource' and r['state']=='read' and r.get('driver')==MCP_CATALOG_DRIVER
  if not mcp_catalog and (r['kind']!='connection' or r['state']!=('inspected-read-only' if r['driver'] in FILE_DRIVERS else 'connected-read-only')):raise ValueError('Choose a verified read-only connection or inspected catalog')
  arcgis=r['driver']==catalog_arcgis.DRIVER
  spatial_driver=arcgis or r['driver'] in OGC_CATALOG_DRIVERS or r['driver']=='stac'
  exact_object_id=catalog_arcgis.object_id(item_id) if item_id is not None and arcgis else None
  if item_id is not None and not arcgis and (not spatial_driver or not mapping.get('path')
      or item_id in ('.','..') or not re.fullmatch(r'[A-Za-z0-9_.-]{1,256}',item_id)):
   raise ValueError('Exact lookup requires a STAC or OGC catalog and a literal ID containing letters, numbers, underscores, dots or hyphens')
  dated=start_date is not None or end_date is not None
  if dated or bbox is not None:
   if not spatial_driver or not mapping.get('path') or item_id is not None:
    raise ValueError('Date and area filters require a supported catalog preview without an exact item ID')
  if dated and r['driver'] not in ('stac','ogc_records',catalog_arcgis.DRIVER):
   raise ValueError('UTC date filtering requires STAC, OGC Records or a time-enabled ArcGIS layer')
  window=utc_date_window(start_date,end_date) if dated else None
  bbox=catalog_preview_bounds(bbox) if bbox is not None else None
  query={'limit':limit} if window or bbox else None
  if window:query['datetime']=f'{start_date}T00:00:00Z/{end_date}T23:59:59.999999999Z'
  if bbox:query['bbox']=','.join(format(value,'.17g') for value in bbox)
  credentials={} if mcp_catalog or r['driver'] in FILE_DRIVERS else self.secret(identifier=r['credential_id'])
  if r.get('ca_file') and file_hash(self.services.source(r['ca_file'],{'.pem','.crt'},2*1024**2))!=r['ca_sha256']:raise ValueError('Saved TLS trust material changed; inspect a new connection version')
  if not 1<=limit<=100:raise ValueError('Use a bounded preview limit from 1 to 100')
  if not mapping or set(mapping)-{'image_id','path','datetime','geometry','bands'} or 'path' not in mapping:raise ValueError('Map supported fields including the image path')
  geometry_format=None
  preview_driver=r['driver']
  if mcp_catalog:
   columns,records,identity=mcp_geojson(r['contents'],r['uri'])
   if identity!=r['resource_sha256']:raise ValueError('Saved MCP catalog snapshot failed its checksum')
   if table!='snapshot.imagery' or set(mapping.values())-columns.keys():raise ValueError('Choose mapped fields from the saved MCP catalog')
   rows=[{k:row.get(v) for k,v in mapping.items()} for row in records[:limit]]
   if mapping.get('geometry')=='@geometry':geometry_format='ogc-crs84'
  elif r['driver'] in FILE_DRIVERS:
   _,_,columns,records=self._snapshot(r['source'],limit,r['source_sha256'])
   if table!='snapshot.imagery' or set(mapping.values())-columns.keys():raise ValueError('Choose mapped columns from the catalog export')
   rows=[{k:row.get(v) for k,v in mapping.items()} for row in records]
   if r['driver']=='geojson_snapshot' and mapping.get('geometry')=='@geometry':geometry_format='ogc-crs84'
  elif r['driver']=='postgres':
   schema=next((t for t in r['tables'] if t['schema']+'.'+t['name']==table),None)
   if not schema or set(mapping.values())-{c['name'] for c in schema['columns']}:raise ValueError('Choose discovered base-table fields; arbitrary SQL is blocked')
   # Re-check privileges/schema on every connection, never trust stale mappings.
   with self.postgres(r['endpoint'],credentials,r.get('ca_file')) as conn:
    pg=driver('psycopg');sql=pg.sql
    current=conn.execute("SELECT c.oid,c.relkind,c.relrowsecurity,pg_catalog.has_table_privilege(c.oid,'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER') FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=%s AND c.relname=%s",(schema['schema'],schema['name'])).fetchone()
    if not current or current[1]!='r' or current[2] or current[3]:raise ValueError('Table type, policy or permissions changed; re-inspect')
    attributes=conn.execute("SELECT attname,pg_catalog.format_type(atttypid,atttypmod),atttypid FROM pg_catalog.pg_attribute WHERE attrelid=%s AND attnum>0 AND NOT attisdropped AND attgenerated='' ORDER BY attnum",(current[0],)).fetchall()
    columns=[{'name':name,'type':kind} for name,kind,_ in attributes]
    if columns!=schema['columns']:raise ValueError('Supplier schema changed; inspect a new version before preview/import')
    allowed_types={'text','character varying','character','date','timestamp without time zone','timestamp with time zone','json','jsonb','bigint','smallint','integer','real','double precision','numeric','boolean'}
    types={name:(kind,oid) for name,kind,oid in attributes}
    projections=[]
    for field,name in mapping.items():
     kind,oid=types[name]
     column=sql.Identifier(name)
     if any(kind==t or kind.startswith(t+'(') for t in allowed_types):
      projections.append(sql.SQL('pg_catalog.substr({}::pg_catalog.text,1,65537)').format(column))
     elif field=='geometry':
      namespace=sql.Identifier(postgis_schema(conn,oid))
      projections.append(sql.SQL("""
       CASE WHEN {col} IS NULL THEN NULL
            WHEN {ns}.st_srid({col}) = 4326
            THEN pg_catalog.substr({ns}.st_asgeojson({col}, 15, 0), 1, 65537)
            ELSE '!unsupported-srid' END
       """).format(ns=namespace,col=column))
      geometry_format='postgis-geojson-epsg4326'
     else:raise ValueError('Map ordinary scalar/JSON fields. Native PostGIS geometry is only supported for the geometry field.')
    statement=sql.SQL('SELECT {} FROM ONLY {}.{} LIMIT %s').format(sql.SQL(',').join(projections),sql.Identifier(schema['schema']),sql.Identifier(schema['name']));values=conn.execute(statement,(limit,)).fetchall();rows=[dict(zip(mapping.keys(),row)) for row in values]
    if any(isinstance(v,str) and len(v)>65536 for row in rows for v in row.values()):raise ValueError('Selected field exceeds the preview limit; no truncated mapping was saved')
    if geometry_format:
     for row in rows:row['geometry']=catalog_geometry(row['geometry'])
  elif r['driver'] in ('mysql','mariadb'):
   adapter=importlib.import_module('catalog_'+r['driver'])
   schema=next((t for t in r['tables'] if t['schema']+'.'+t['name']==table),None)
   if not schema:raise ValueError('Choose a discovered base table; arbitrary SQL is blocked')
   with adapter.session(self.services,r['endpoint'],credentials,r.get('ca_file')) as (conn,database):rows=adapter.preview(conn,database,schema,mapping,limit)
  elif r['driver'] in ('sqlserver','sqlserver2025','oracle'):
   from catalog_commercial import request
   schema=next((t for t in r['tables'] if t['schema']+'.'+t['name']==table),None)
   if not schema:raise ValueError('Choose a discovered base table')
   rows=request(self.services,r['driver'],r['endpoint'],credentials,r.get('ca_file'),schema=schema,mapping=mapping,limit=limit,expected_version=r.get('engine_version'))
  elif r['driver']=='db2':
   from catalog_db2 import catalog_request
   schema=next((t for t in r['tables'] if t['schema']+'.'+t['name']==table),None)
   if not schema:raise ValueError('Choose a discovered base table')
   rows=catalog_request(self.services,r['endpoint'],credentials,r['ca_file'],r['ca_sha256'],
                        table=schema,mapping=mapping,limit=limit)
  elif r['driver']=='mongo':
   if table not in {c['id'] for c in r['collections']} or any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]{0,119}',v) for v in mapping.values()):raise ValueError('Choose a discovered collection and ordinary field paths')
   with self.mongo(r['endpoint'],credentials,r.get('ca_file')) as db:
    cursor=db[table].find({},projection={v:1 for v in mapping.values()},limit=limit,max_time_ms=5000,batch_size=2)
    def field(document,path):
     for part in path.split('.'):
      if not isinstance(document,dict):return None
      document=document.get(part)
     return document
    rows=[];total=0
    try:
     for row in cursor:
      value={k:field(row,v) for k,v in mapping.items()};total+=len(json.dumps(value,default=str))
      if total>1024**2:raise ValueError('Selected MongoDB preview exceeds 1 MiB')
      rows.append(value)
    finally:cursor.close()
  elif r['driver']==catalog_arcgis.DRIVER:
   http=catalog_arcgis.client(r['endpoint'],self.headers(credentials),r.get('ca_file'))
   rows=catalog_arcgis.preview(http,r['arcgis_layer'],table,mapping,limit,exact_object_id,bbox,window)
   if mapping.get('geometry')=='@geometry':geometry_format='ogc-crs84'
  elif r['driver'] in OGC_CATALOG_DRIVERS:
   if table not in {c['id'] for c in r['collections']} or not re.fullmatch(r'[A-Za-z0-9_.-]{1,150}',table) or table in ('.','..'):
    raise ValueError('Choose an ordinary discovered feature collection')
   if any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_-]{0,119}',v) for v in mapping.values()):
    raise ValueError('Map simple GeoJSON property names, or the top-level id and geometry fields')
   http=InternalHTTP(r['endpoint'],{**self.headers(credentials),'Accept':'application/geo+json, application/json'},r.get('ca_file'),timeout=10)
   if verify_ogc_features(http,records=r['driver']=='ogc_records')!=r.get('conformance'):raise ValueError('OGC conformance changed; reconnect')
   path='/collections/'+table+'/items'
   if item_id is None and query is None:query={'limit':limit}
   connection,response=http.send('GET',path if item_id is None else path+'/'+item_id,query=query)
   try:
    if response.getheader('Content-Crs','').strip('<>') not in ('','http://www.opengis.net/def/crs/OGC/1.3/CRS84'):
     raise ValueError('OGC Features preview supports the default two-dimensional CRS84 geometry')
    content=response.read(2*1024**2+1)
    if len(content)>2*1024**2:raise ValueError('OGC Features page exceeds 2 MiB')
    doc=json.loads(content)
   finally:
    response.close()
    connection.close()
   if item_id is not None:
    returned_id=doc.get('id') if isinstance(doc,dict) else None
    if type(returned_id) not in (str,int) or str(returned_id)!=item_id or doc.get('collection',table)!=table:
     raise ValueError('OGC response does not match the requested feature ID')
    features=[doc]
   else:
    if not isinstance(doc,dict) or doc.get('type')!='FeatureCollection' or 'crs' in doc or not isinstance(doc.get('features'),list):
     raise ValueError('Expected a bounded CRS84 GeoJSON FeatureCollection')
    features=doc['features']
   if len(features)>limit:raise ValueError('The OGC supplier did not honor the bounded preview limit')
   if bbox:verify_catalog_bounds(features,bbox)
   rows=[]
   for feature in features:
    if not isinstance(feature,dict) or feature.get('type')!='Feature' or 'crs' in feature or not isinstance(feature.get('properties'),dict):
     raise ValueError('Invalid GeoJSON feature')
    if feature.get('collection',table)!=table:raise ValueError('OGC item belongs to a different collection')
    properties=feature['properties']
    if mapping['path'] not in properties:raise ValueError('The mapped local image-path property is missing')
    row={k:(feature.get(v) if (k,v) in {('image_id','id'),('geometry','geometry')} else properties.get(v)) for k,v in mapping.items()}
    row['path']=stac_mounted_path(row['path'])
    if window:row['record_time']=ogc_record_time(feature,window)
    if row.get('geometry') is not None:row['geometry']=catalog_geometry(json.dumps(row['geometry']))
    rows.append(row)
   geometry_format='ogc-crs84' if 'geometry' in mapping else None
  elif r['driver']=='stac':
   if table not in {c['id'] for c in r['collections']} or not re.fullmatch(r'[A-Za-z0-9_.-]{1,150}',table) or table in ('.','..'):raise ValueError('Choose a discovered STAC collection')
   parts=None if mapping['path']=='assets' else stac_mount_mapping(mapping)
   # Fixed standard endpoint; never follow returned links to another supplier.
   items_path='/collections/'+table+'/items'
   http=InternalHTTP(r['endpoint'],{**self.headers(credentials),'Accept':'application/geo+json, application/json'},r.get('ca_file'),timeout=10)
   crs84='http://www.opengis.net/def/crs/OGC/1.3/CRS84'
   if item_id is None:
    document=http.json('GET',items_path,query=query or {'limit':limit},content_crs=crs84)
   else:
    item=http.json('GET',items_path+'/'+item_id,content_crs=crs84)
    if not isinstance(item,dict) or item.get('id')!=item_id or item.get('collection')!=table:
     raise ValueError('STAC response does not match the requested item and collection')
    document={'type':'FeatureCollection','features':[item]}
   features=stac_collection_features(document,table,limit,filtered=bool(window or bbox))
   if window:stac_verify_window(features,window)
   if bbox:verify_catalog_bounds(features,bbox)
   if parts is None:
    rows=stac_asset_inventory(features,table,limit)
   else:
    rows=stac_mount_rows(features,mapping,parts)
    preview_driver=STAC_MOUNT_DRIVER
    if 'geometry' in mapping:geometry_format='ogc-crs84'
  else:raise ValueError('OpenAPI discovery does not authorize supplier operations')
  normalized=json.loads(json.dumps(rows,default=str))
  if len(json.dumps(normalized))>1024*1024:raise ValueError('Preview exceeds 1 MiB; choose narrower fields')
  provenance={'connection_id':identifier,'source_schema_sha256':r['schema_sha256']}
  if item_id is not None:provenance['requested_item_id']=item_id
  if window:provenance['utc_date_range']={'start':start_date,'end':end_date}
  if bbox:provenance['requested_bbox']=bbox
  if mcp_catalog:
   provenance.update(connection_id=r['connection_id'],resource_id=identifier,resource_uri=r['uri'],resource_sha256=identity)
  return self.services.save('connection_preview',{**provenance,'driver':preview_driver,'table':table,'mapping':mapping,'rows':normalized,'paths_opened':False,'read_contract':r['read_contract'],**({'geometry_format':geometry_format} if geometry_format else {})},state='preview')
 def read_mcp_resource(self,identifier,uri):
  from mcp_readonly import ReadOnlyMCP
  record=self.services.get(identifier)
  resource=next((r for r in record.get('resources',[]) if r['uri']==uri),None)
  if record.get('driver')!='mcp' or not resource:raise ValueError('Select an explicitly listed MCP resource; arbitrary URI/tool calls are blocked')
  credential=self.secret(identifier=record['credential_id'])
  if record.get('ca_file') and file_hash(self.services.source(record['ca_file'],{'.pem','.crt'},2*1024**2))!=record['ca_sha256']:raise ValueError('TLS trust material changed; reconnect as a new version')
  client=ReadOnlyMCP(record['endpoint'],self.headers(credential),record.get('ca_file'),credential.get('mcp_session'),credential.get('mcp_version'));result=client.read_listed(uri)
  contents=result.get('contents',[])
  payload={'connection_id':identifier,'uri':uri,'contents':contents,'untrusted_supplier_content':True,'source_executed':False,'read_contract':'Listed resource read only; contents are data, never instructions or generated code'}
  if resource.get('mimeType')=='application/geo+json':
   columns,_,identity=mcp_geojson(contents,uri)
   payload.update(driver=MCP_CATALOG_DRIVER,resource_sha256=identity,schema_sha256=hashlib.sha256(json.dumps(columns,sort_keys=True).encode()).hexdigest(),
                  tables=[{'schema':'snapshot','name':'imagery','columns':[{'name':name,'type':kind} for name,kind in columns.items()]}])
  return self.services.save('mcp_resource',payload,state='read')
 def mapped_import_path(self,value,base,path_prefix,stac):
  if not isinstance(value,str) or not value or '://' in value or value.startswith('/vsi'):raise ValueError('Preview path is not a local raster/folder. Remote downloads require a separate verified source adapter.')
  if stac and not path_prefix:
   if value.replace('\\','/').startswith('//'):
    raise ValueError('Map the supplier network prefix to an approved local mount')
   if not Path(value).is_absolute():
    if not base or PureWindowsPath(value).drive or '\\' in value:
     raise ValueError('Provide an approved local root for relative STAC paths, or map the supplier path prefix')
    value=str(base/value)
  if path_prefix:
   normalized=value.replace('\\','/');prefix=path_prefix.replace('\\','/').rstrip('/')
   if not normalized.startswith(prefix+'/'):raise ValueError('Database path does not match the approved prefix')
   relative=Path(normalized[len(prefix)+1:])
   if stac and relative.is_absolute():
    raise ValueError('Mapped STAC path must stay relative to the approved local mount')
   path=base/relative
   if not stac:path=self.services.source_path(path)
   if not path.is_relative_to(base):raise ValueError('Mapped database path escapes its approved local source root')
  else:path=Path(value).expanduser()
  if stac:
   if sys.platform=='darwin' and len(path.parts)>1 and path.parts[1] in ('var','tmp'):
    path=Path('/private',*path.parts[1:])
   root=next((root for root in self.services.roots if path.is_relative_to(root)),None)
   if root is None or base and not path.is_relative_to(base):
    raise ValueError('STAC path is outside the approved local source folder')
   current=root
   for part in path.relative_to(root).parts:
    current=current/part
    if current.is_symlink() or getattr(current,'is_junction',lambda:False)():
     raise ValueError('Mounted STAC paths cannot traverse symbolic links or junctions')
  return self.services.source_path(path)
 def plan_import(self,identifier,indices,path_prefix='',local_root=None):
  """Inspect approved local paths; do not open remote URLs or write suppliers."""
  from imagery_products import inspect,products,SUFFIXES,standalone_auxiliary,wavelength_nm
  from sensor_metadata import is_metadata,for_band
  r=self.services.get(identifier)
  if r['kind']!='connection_preview' or r['driver'] not in {*DATABASE_ADAPTERS,*FILE_DRIVERS,'ogc_features','ogc_records','sqlite',MCP_CATALOG_DRIVER,STAC_MOUNT_DRIVER,catalog_arcgis.DRIVER}:raise ValueError('Local path import requires a mapped catalog preview')
  self.verify_file_preview(r)
  if not indices or len(indices)>100 or len(set(indices))!=len(indices) or any(type(i)!=int or not 0<=i<len(r['rows']) for i in indices):raise ValueError('Choose distinct existing preview rows')
  if path_prefix and not local_root:raise ValueError('A supplier path prefix needs an explicit local source root')
  base=self.services.source_path(local_root) if local_root else None
  if base and not base.is_dir():raise ValueError('Choose an approved local source folder for path mapping')
  candidates={};items=[];issues=[];started=time.monotonic();total_files=0
  for index in indices:
   if time.monotonic()-started>60:raise ValueError('Catalog inspection exceeded its batch budget; choose fewer rows. No partial catalog was published.')
   row=r['rows'][index]
   stac=r['driver']==STAC_MOUNT_DRIVER
   bundle=stac and 'bands' in row
   packed=stac and 'packed_channels' in row
   manifest=None
   if bundle:
    if not row['bands'] or not base:raise ValueError('Choose a complete mounted STAC band row and an approved local root')
    base_path=base
    found=[self.mapped_import_path(asset['path'],base,path_prefix,True) for asset in row['bands']]
    if len(set(found))!=len(found) or any(not p.is_file() or standalone_auxiliary(p.name) for p in found):
     raise ValueError('Each STAC science asset must point to a distinct raster file')
   else:
    path=self.mapped_import_path(row.get('path'),base,path_prefix,stac)
    if packed and (not path.is_file() or standalone_auxiliary(path.name)):
     raise ValueError('Packed STAC channel metadata requires one mounted science raster')
    if path.is_dir():
     if path.is_relative_to(self.data):raise ValueError('Application-managed storage must not be indexed recursively')
     base_path=path.parent if path.suffix.upper()=='.SAFE' else path
     found=[]
     for folder,dirs,names in os.walk(path,followlinks=False):
      dirs[:]=[n for n in dirs if not (Path(folder)/n).is_symlink() and not getattr(Path(folder)/n,'is_junction',lambda:False)()]
      for name in names:
       p=Path(folder)/name
       if p.suffix.lower() in SUFFIXES and not standalone_auxiliary(p.relative_to(base_path).as_posix()):found.append(p)
       if len(found)>4096:raise ValueError('Selected product folder exceeds 4096 raster files; choose a narrower product')
    elif path.is_file():base_path=path.parent;found=[path]
    else:raise ValueError('Database image path is unavailable on this computer; provide a local prefix mapping')
   for p in found:
    total_files+=1
    if total_files>4096 or time.monotonic()-started>60:raise ValueError('Catalog inspection batch is too large; choose fewer source rows')
    self.services.source(p,SUFFIXES)
    # The product grouper discovers supplier calibration sidecars. Verify
    # those paths before it can read them, including SAFE-level metadata.
    parents=([q for q in p.parents if q.is_relative_to(base_path)] if bundle
             else [p.parent]+[q for q in p.parents if q.suffix.upper()=='.SAFE'])
    for parent in parents:
     for metadata in parent.iterdir():
      if is_metadata(metadata.name):self.services.source(metadata,{'.xml','.txt','.json'},2*1024**2)
   if bundle or packed:
    import rasterio
    with rasterio.Env(PROJ_NETWORK='OFF',GDAL_PAM_ENABLED='NO',GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
     records=[inspect(p,p.relative_to(base_path).as_posix()) for p in found]
   else:records=[inspect(p,p.relative_to(base_path).as_posix()) for p in found]
   if packed:
    record=records[0];channels=row['packed_channels']
    if record['count']!=len(channels):raise ValueError('STAC channel count differs from the mounted raster header')
    names=[];waves=[];calibration=[]
    for i,channel in enumerate(channels):
     native=wavelength_nm(record['channel_metadata'][i])
     declared=channel.get('wavelength_nm')
     if native is not None and declared is not None and not math.isclose(native,declared,rel_tol=0,abs_tol=.5):
      raise ValueError(f'STAC center wavelength conflicts with raster channel {i+1}')
     names.append(channel.get('name') or record['descriptions'][i] or f'band_{i+1}')
     waves.append(native if native is not None else declared)
     nodata='nan' if record['nodata_is_nan'] else record['nodata']
     if 'nodata' in channel and channel['nodata']!=nodata:
      raise ValueError(f'STAC nodata differs from raster channel {i+1}; use a raster with matching nodata')
     native_pair=(record['scales'][i],record['offsets'][i])
     declared_pair=(channel.get('scale',native_pair[0]),channel.get('offset',native_pair[1]))
     if declared_pair!=native_pair and native_pair!=(1,0):
      raise ValueError(f'STAC calibration differs from raster channel {i+1}; use a raster with matching native calibration')
     calibration.append(None if declared_pair==native_pair else {'scale':declared_pair[0],'offset':declared_pair[1]})
    identity_fields=[r['connection_id'],r['table'],row['item_id'],row['asset_key']]
    if any(calibration):
     if for_band(Path(record['path']),base_path):raise ValueError('Choose one radiometric calibration source for the packed STAC asset')
     identity_fields.append(calibration)
    identity=hashlib.sha256(json.dumps(identity_fields,sort_keys=True).encode()).hexdigest()
    tags=record['tags']
    manifest=[(base_path/('stac-packed-'+identity+'.json'),{'name':row['item_id'],
      'bands':[{'name':row['asset_key'],'path':record['relative_path'],'band_names':names,'wavelength_nm':waves}],
      'sensor':tags.get('sensor',''),'processing_level':tags.get('processing_level','unknown'),
      'datetime':row.get('datetime'),'datetime_source':'STAC item properties.datetime'})]
   if bundle:
    entries=[]
    for record,asset in zip(records,row['bands']):
     if record['count']!=1:raise ValueError('Each selected STAC science asset must have exactly one raster channel')
     if 'nodata' in asset and asset['nodata']!=('nan' if record['nodata_is_nan'] else record['nodata']):
      raise ValueError('STAC band nodata conflicts with the raster header')
     if any(field in asset for field in ('scale','offset')) and for_band(Path(record['path']),base_path):
      raise ValueError('Choose one radiometric calibration source for the STAC band bundle')
     for field,default,header in (('scale',1,'scales'),('offset',0,'offsets')):
      if field in asset and record[header][0] not in (default,asset[field]):
       raise ValueError('STAC band calibration conflicts with the raster header')
     entries.append({'name':asset['name'],'path':record['relative_path'],
                     **{key:asset[key] for key in ('scale','offset') if key in asset}})
     if 'wavelength_nm' in asset:
      native=wavelength_nm(record['channel_metadata'][0])
      if native is not None and not math.isclose(native,asset['wavelength_nm'],rel_tol=0,abs_tol=.5):
       raise ValueError('STAC center wavelength conflicts with the raster header')
      entries[-1]['wavelength_nm']=[native if native is not None else asset['wavelength_nm']]
    sensors={rec['tags'].get('sensor','') for rec in records}
    levels={rec['tags'].get('processing_level','unknown') for rec in records}
    if len(sensors)>1 or len(levels)>1:raise ValueError('STAC band files declare inconsistent sensor or processing level')
    identity=hashlib.sha256(json.dumps([r['table'],row['item_id'],str(base_path)]).encode()).hexdigest()
    # Build a local grouping recipe in memory. No manifest is written to the mount.
    manifest=[(base_path/('stac-'+identity+'.json'),
               {'name':row['item_id'],'bands':entries,'sensor':sensors.pop(),
                'processing_level':levels.pop(),'datetime':row.get('datetime'),'datetime_source':'STAC item properties.datetime'})]
   grouped,warnings=products(records,base_path,manifest)
   if (bundle or packed) and (len(grouped)!=1 or grouped[0]['status']!='ready'):
    raise ValueError('STAC band product needs review: '+('; '.join(grouped[0]['warnings']) if grouped else 'no valid grouped product'))
   if not grouped:raise ValueError('No supported logical image product found for the selected database row')
   issues.extend(warnings)
   for product in grouped:
    if product['id'] in candidates:continue
    product['catalog_connection']={'connection_id':r['connection_id'],'preview_id':identifier,'row':index,'external_image_id':r['rows'][index].get('image_id'),'external_datetime':r['rows'][index].get('datetime'),'supplier_names_preserved':True};items.append(product);candidates[product['id']]=True
    if bundle:product['catalog_connection']['stac_assets']=[{k:a[k] for k in ('name','asset_key','wavelength_nm','fwhm_nm') if k in a} for a in row['bands']]
    if packed:
     product['catalog_connection']['stac_packed_asset']={'asset_key':row['asset_key'],'channels':row['packed_channels']}
     if any(calibration):product['sources'][0]['stac_channel_calibration']=calibration
    if r['driver']=='ogc_records' and row.get('record_time'):
     product['catalog_connection']['record_time']=row['record_time']
    if r['driver']==catalog_arcgis.DRIVER:
     if 'requested_item_id' in r:product['catalog_connection']['arcgis_object_id']=catalog_arcgis.object_id(r['requested_item_id'])
     if 'requested_bbox' in r:product['catalog_connection']['arcgis_requested_bbox']=r['requested_bbox']
     if row.get('layer_time_utc'):
      product['catalog_connection']['arcgis_layer_time_utc']=row['layer_time_utc']
      product['catalog_connection']['arcgis_utc_date_range']=r['utc_date_range']
    if r['driver']==MCP_CATALOG_DRIVER:
     product['catalog_connection'].update({k:r[k] for k in ('resource_id','resource_uri','resource_sha256')})
    if r.get('geometry_format') in ('postgis-geojson-epsg4326','ogc-crs84'):
     product['catalog_connection']['supplier_geometry']=r['rows'][index]['geometry']
     product['catalog_connection']['supplier_geometry_crs']='OGC:CRS84' if r['geometry_format']=='ogc-crs84' else 'EPSG:4326'
  return self.services.save('catalog_import_plan',{'source_preview':identifier,'items':items,'issues':issues,'files':[{k:s[k] for k in ('path','sha256')} for item in items for s in item['sources']],'operation':'index-in-place; immutable copies are made later when selected into a mission','monitoring_enabled':False},state='candidate')
 def verify_file_preview(self,preview):
  if preview['driver'] in FILE_DRIVERS:
   source=self.services.get(preview['connection_id'])
   self._snapshot(source['source'],0,source['source_sha256'])
  if preview['driver']!='sqlite':return
  source=self.services.source(preview['source'])
  with self.services.sqlite_source(source):
   if file_hash(source)!=preview['source_sha256']:raise ValueError('SQLite catalog changed since preview; inspect its new revision')
 def approve_import(self,identifier,catalog):
  plan=self.services.get(identifier)
  if plan['kind']!='catalog_import_plan' or plan['state']!='candidate':raise ValueError('Choose a saved catalog import plan')
  preview=self.services.get(plan['source_preview'])
  self.verify_file_preview(preview)
  for entry in plan['files']:
   if file_hash(self.services.source(entry['path']))!=entry['sha256']:raise ValueError('A source band changed after preview; prepare a new import plan')
  # Catalog mutation is local only. Supplier files/database are never modified.
  prior={i['id']:i.get('watch_id') for i in catalog.images(include_history=True)};groups={}
  for item in plan['items']:groups.setdefault(prior.get(item['id']),[]).append(item)
  with self.database() as db:
   for watch,items in groups.items():catalog.save_products(items,watch_id=watch,db=db)
   return self.services.save('catalog_import_receipt',{'plan_id':identifier,'image_ids':[i['id'] for i in plan['items']],'images':len(plan['items']),'monitoring_enabled':False},state='approved',parent=identifier,db=db)

def register(backend):
 from fastapi import Header,HTTPException
 from pydantic import BaseModel,Field,StrictFloat
 from starlette.concurrency import run_in_threadpool
 from agent_services import access_key
 import secrets
 def connection():
  roots=json.loads(os.getenv('GEO_SERVICE_SOURCE_ROOTS','[]')) or [str(backend.LOCAL_RASTER_ROOT)]
  with backend._database() as db:row=db.execute("SELECT payload FROM settings WHERE key='service_source_roots'").fetchone()
  if row and not os.getenv('GEO_SERVICE_SOURCE_ROOTS'):roots=json.loads(row['payload'])
  return Connections(Services(backend._database,backend.DATA_DIR,roots))
 def authorize(key):
  if not key or not secrets.compare_digest(key,access_key(backend.DATA_DIR)):raise HTTPException(401,'Local services key required')
 class Connect(BaseModel):
  kind:str
  name:str=Field(min_length=1,max_length=120)
  endpoint:str=Field(min_length=1,max_length=2048)
  credentials:dict=Field(default_factory=dict)
  ca_file:str|None=None
  validate_only:bool=False
 class Preview(BaseModel):
  table:str
  mapping:dict[str,str]
  limit:int=Field(default=50,ge=1,le=100)
  item_id:str|None=Field(default=None,min_length=1,max_length=256)
  start_date:str|None=Field(default=None,min_length=10,max_length=10)
  end_date:str|None=Field(default=None,min_length=10,max_length=10)
  bbox:list[StrictFloat]|None=Field(default=None,min_length=4,max_length=4)
 class ImportPlan(BaseModel):
  rows:list[int]=Field(min_length=1,max_length=100)
  path_prefix:str=Field(default='',max_length=4096)
  local_root:str|None=None
 class ResourceRead(BaseModel):
  uri:str=Field(min_length=1,max_length=4096)
 class EngineRequest(BaseModel):
  engine:str=Field(min_length=1,max_length=80)
  version:str=Field(default='',max_length=80)
 @backend.app.post('/api/services/catalog-engine-requests')
 async def request_engine(body:EngineRequest,x_geo_services_key:str|None=Header(default=None)):
  authorize(x_geo_services_key)
  engine=body.engine.strip();version=body.version.strip()
  if not engine or not re.fullmatch(r'[\w .+()-]{1,80}',engine) or version and not re.fullmatch(r'[\w .+()-]{1,80}',version):raise HTTPException(422,'Use product and version names only; do not enter credentials or connection strings')
  return await run_in_threadpool(connection().services.save,'catalog_engine_request',{'name':engine,'engine_version':version,'network_requests':0,'read_contract':'Pending verified adapter. Supply an existing read-only catalog endpoint, or validate an offline driver and engine-specific privilege, TLS and bounded read checks on an isolated target.'},'awaiting-adapter')
 @backend.app.post('/api/services/connections')
 async def connect(body:Connect,x_geo_services_key:str|None=Header(default=None)):
  authorize(x_geo_services_key)
  try:return await run_in_threadpool(connection().inspect,body.kind,body.endpoint,body.credentials,body.name,body.ca_file,validate_only=body.validate_only)
  except Exception as exc:
   # Driver errors can contain credentials/connection strings. Never echo them.
   if isinstance(exc,ValueError):raise HTTPException(422,str(exc)) from exc
   raise HTTPException(422,'Read-only connection failed. Check the private endpoint, verified TLS, offline driver component and read-only account; no supplier changes were attempted.') from exc
 @backend.app.post('/api/services/connections/{identifier}/preview')
 async def preview(identifier:str,body:Preview,x_geo_services_key:str|None=Header(default=None)):
  authorize(x_geo_services_key)
  try:return await run_in_threadpool(connection().preview,identifier,body.table,body.mapping,body.limit,body.item_id,body.start_date,body.end_date,body.bbox)
  except ValueError as exc:raise HTTPException(422,str(exc)) from exc
  except Exception as exc:raise HTTPException(422,'Read-only preview failed; no supplier changes were attempted. Check schema, permissions and connection health.') from exc
 @backend.app.post('/api/services/connections/{identifier}/plan-import')
 async def plan_import(identifier:str,body:ImportPlan,x_geo_services_key:str|None=Header(default=None)):
  authorize(x_geo_services_key)
  try:return await run_in_threadpool(connection().plan_import,identifier,body.rows,body.path_prefix,body.local_root)
  except (ValueError,OSError) as exc:raise HTTPException(422,str(exc)) from exc
 @backend.app.post('/api/services/connections/{identifier}/approve-import')
 async def approve_import(identifier:str,x_geo_services_key:str|None=Header(default=None)):
  authorize(x_geo_services_key)
  try:return await run_in_threadpool(connection().approve_import,identifier,backend.search_local_imagery.catalog)
  except (ValueError,OSError) as exc:raise HTTPException(422,str(exc)) from exc
 @backend.app.post('/api/services/connections/{identifier}/resource')
 async def resource(identifier:str,body:ResourceRead,x_geo_services_key:str|None=Header(default=None)):
  authorize(x_geo_services_key)
  try:return await run_in_threadpool(connection().read_mcp_resource,identifier,body.uri)
  except ValueError as exc:raise HTTPException(422,str(exc)) from exc
  except Exception as exc:raise HTTPException(422,'Read-only MCP resource read failed; reconnect or check the approved resource. No tool was executed.') from exc
