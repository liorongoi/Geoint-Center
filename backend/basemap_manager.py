"""Browser-fetched basemap service definitions and bounded image basemaps."""
from __future__ import annotations
import io
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from uuid import uuid4
from offline_policy import enabled as offline
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from PIL import Image
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

class Provider(BaseModel):
    name: str = Field(min_length=1,max_length=100)
    service_type: str = Field(pattern='^(wms|wmts|arcgis-cached|arcgis-dynamic|xyz|tilejson|style)$')
    url: str = Field(min_length=1,max_length=4096)
    layers: str = Field(default='',max_length=500)
    version: str = Field(default='1.3.0',pattern='^(1.3.0|1.1.1)$')
    style: str = Field(default='',max_length=200)
    matrix_set: str = Field(default='GoogleMapsCompatible',max_length=200)
    matrix_prefix: str = Field(default='',max_length=200)
    tile_size: int = Field(default=256,ge=64,le=2048)
    max_zoom: int = Field(default=22,ge=0,le=24)
    attribution: str = Field(default='',max_length=500)


def normalized(p):
    url=p.url.strip();u=urlsplit(url);_=u.port
    if u.scheme not in ('http','https') or not u.hostname or u.username or u.password or u.fragment or any(c.isspace() or ord(c)<32 for c in url) or any(c in url for c in '\\<>"\'') or '{' in u.netloc:
        raise ValueError('Use an HTTP(S) URL without embedded credentials, spaces or a fragment')
    if not p.name.strip() or p.tile_size & (p.tile_size-1): raise ValueError('Give a name and a power-of-two tile size')
    def query(values):
        keys={k.lower() for k in values};pairs=[(k,v) for k,v in parse_qsl(u.query,keep_blank_values=True) if k.lower() not in keys]
        return urlunsplit(u._replace(query=urlencode(pairs+list(values.items()),safe='{}:,')))
    definition={'label':p.name.strip(),'kind':'raster','url':url,'tile_size':p.tile_size,'max_zoom':p.max_zoom,'attribution':p.attribution}
    if p.service_type=='wms':
        if not p.layers: raise ValueError('Enter the WMS layer name(s)')
        definition['url']=query({'SERVICE':'WMS','REQUEST':'GetMap','VERSION':p.version,'LAYERS':p.layers,'STYLES':p.style,'FORMAT':'image/png','TRANSPARENT':'TRUE','WIDTH':str(p.tile_size),'HEIGHT':str(p.tile_size),'BBOX':'{bbox-epsg-3857}',('CRS' if p.version=='1.3.0' else 'SRS'):'EPSG:3857'})
    elif p.service_type=='wmts':
        if not p.layers or not p.matrix_set: raise ValueError('Enter the WMTS layer and Web Mercator matrix set')
        definition['url']=query({'SERVICE':'WMTS','REQUEST':'GetTile','VERSION':'1.0.0','LAYER':p.layers,'STYLE':p.style or 'default','FORMAT':'image/png','TILEMATRIXSET':p.matrix_set,'TILEMATRIX':p.matrix_prefix+'{z}','TILECOL':'{x}','TILEROW':'{y}'})
    elif p.service_type.startswith('arcgis'):
        if not u.path.rstrip('/').lower().endswith('/mapserver'): raise ValueError('Use an ArcGIS MapServer URL ending in /MapServer')
        if p.service_type=='arcgis-cached': definition['url']=urlunsplit(u._replace(path=u.path.rstrip('/')+'/tile/{z}/{y}/{x}'))
        else:
            u=u._replace(path=u.path.rstrip('/')+'/export')
            params={'f':'image','bbox':'{bbox-epsg-3857}','bboxSR':'3857','imageSR':'3857','size':f'{p.tile_size},{p.tile_size}','format':'png32','transparent':'true'}
            if p.layers: params['layers']='show:'+p.layers
            definition['url']=query(params)
    elif p.service_type=='xyz':
        if not all(x in url for x in ('{z}','{x}','{y}')): raise ValueError('XYZ URL must include {z}, {x} and {y}')
    elif p.service_type=='tilejson': definition['tilejson']=True
    elif p.service_type=='style': definition['kind']='vector'
    return definition


def register(app,database,data,map_client,ensure_map,tool_data,mc2_url):
    router=APIRouter(prefix='/api/map-providers',tags=['Custom basemaps'])
    def saved():
        with database() as db:
            rows=db.execute("SELECT payload FROM settings WHERE key LIKE 'basemap:custom-%'").fetchall()
            active=db.execute("SELECT payload FROM settings WHERE key='basemap_active'").fetchone()
        return [json.loads(r['payload']) for r in rows if not offline() or json.loads(r['payload']).get('definition',{}).get('image_bounds')],json.loads(active['payload']) if active else ('offline' if offline() else 'satellite')
    async def call(tool,args):
        result=await map_client.call(tool,args);d=tool_data(result)
        if getattr(result,'isError',False) or (isinstance(result,dict) and result.get('isError')) or d.get('error'): raise HTTPException(502,'Map provider update failed. '+str(d.get('error','Update the MC² service to support custom basemaps.'))[:200])
        return d
    async def install(record):
        state=await ensure_map()
        await call('configure_basemap',{'map_id':state['map_id'],'key':record['id'],'definition':record['definition']})
        await call('set_basemap',{'map_id':state['map_id'],'basemap':record['id']})
        with database() as db:
            db.execute('INSERT INTO settings VALUES(?,?) ON CONFLICT(key) DO UPDATE SET payload=excluded.payload',('basemap:'+record['id'],json.dumps(record)))
            db.execute("INSERT INTO settings VALUES('basemap_active',?) ON CONFLICT(key) DO UPDATE SET payload=excluded.payload",(json.dumps(record['id']),))
        return {'provider':record,'map_url':state['url'],'reload_map':True}
    @router.get('')
    async def providers():
        rows,active=saved();return {'providers':rows,'active':active}
    @router.post('')
    async def add(request:Provider):
        if offline(): raise HTTPException(409,"Remote map services are disabled in the offline desktop edition. Import a local georeferenced basemap instead.")
        rows,_=saved()
        if len(rows)>=40: raise HTTPException(422,'Remove an unused custom basemap before adding another (limit 40)')
        try: definition=normalized(request)
        except ValueError as e: raise HTTPException(422,str(e))
        return await install({'id':'custom-'+uuid4().hex,**request.model_dump(),'definition':definition})
    @router.post('/{key}/activate')
    async def activate(key:str):
        rows,_=saved();record=next((r for r in rows if r['id']==key),None)
        if not record: raise HTTPException(404,'Saved basemap not found')
        return await install(record)
    @router.delete('/{key}')
    async def remove(key:str):
        rows,active=saved();record=next((r for r in rows if r['id']==key),None)
        if not record: raise HTTPException(404,'Saved basemap not found')
        state=await ensure_map()
        if active==key: await call('set_basemap',{'map_id':state['map_id'],'basemap':'offline' if offline() else 'satellite'})
        await call('configure_basemap',{'map_id':state['map_id'],'key':key,'remove':True})
        with database() as db:
            db.execute('DELETE FROM settings WHERE key=?',('basemap:'+key,))
            if active==key: db.execute("UPDATE settings SET payload=? WHERE key='basemap_active'",(json.dumps('offline' if offline() else 'satellite'),))
        return {'ok':True,'map_url':state['url'],'reload_map':True}
    @router.post('/image/upload')
    async def upload(file:UploadFile=File(...),name:str=Form(...,min_length=1,max_length=100),bands:str=Form('1,2,3',max_length=30)):
        rows,_=saved()
        if len(rows)>=40: raise HTTPException(422,'The custom basemap limit is 40')
        raw=bytearray()
        try:
            while chunk:=await file.read(1024*1024):
                raw.extend(chunk)
                if len(raw)>64*1024*1024: raise HTTPException(413,'Basemap upload limit is 64 MiB; use a tiled service for larger imagery')
        finally: await file.close()
        def prepare():
            indices=[int(v.strip()) for v in bands.split(',')]
            if len(indices)==1: indices*=3
            if len(indices)!=3: raise ValueError('Choose one grayscale band or three RGB band numbers')
            with rasterio.MemoryFile(bytes(raw)) as mem:
                with mem.open(driver='GTiff') as src:
                    if not src.crs or src.width*src.height>64_000_000 or src.count>512: raise ValueError('Use a georeferenced TIFF up to 64 million source pixels and 512 bands')
                    if min(indices)<1 or max(indices)>src.count: raise ValueError('Band numbers exceed the TIFF band count')
                    with WarpedVRT(src,crs='EPSG:3857',resampling=Resampling.bilinear) as vrt:
                        scale=min(1,1024/max(vrt.width,vrt.height));h=max(1,round(vrt.height*scale));w=max(1,round(vrt.width*scale))
                        arr=vrt.read(indices,out_shape=(3,h,w),masked=True).astype('float64')
                        bounds=list(transform_bounds(vrt.crs,'EPSG:4326',*vrt.bounds))
            if not np.isfinite(bounds).all() or not -180<=bounds[0]<bounds[2]<=180 or not -85.051129<=bounds[1]<bounds[3]<=85.051129: raise ValueError('Basemap bounds must fit the Web Mercator world and not cross the antimeridian')
            valid=~np.ma.getmaskarray(arr).any(axis=0)&np.isfinite(arr.data).all(axis=0)
            if not valid.any(): raise ValueError('No valid pixels in the selected bands')
            rgb=np.zeros((h,w,4),dtype='uint8')
            for i in range(3):
                a=arr.data[i];lo,hi=np.percentile(a[valid],[2,98]);scaled=np.clip((a-lo)/max(hi-lo,1e-12),0,1)*254+1
                rgb[:,:,i]=np.where(valid,np.nan_to_num(scaled),0).astype('uint8')
            rgb[:,:,3]=valid.astype('uint8')*255;out=io.BytesIO();Image.fromarray(rgb).save(out,format='PNG');return out.getvalue(),bounds
        from resource_admission import run_reserved
        try: png,bounds=await run_in_threadpool(run_reserved,'Local basemap preparation',2,prepare)
        except (ValueError,rasterio.errors.RasterioError) as e: raise HTTPException(422,str(e))
        state=await ensure_map()
        from mc2_client import private_http_client
        async with private_http_client(mc2_url,timeout=60) as client:
            r=await client.post(f'{mc2_url}/api/maps/{state["map_id"]}/basemap-image',files={'file':('preview.png',png,'image/png')})
            if r.is_error: raise HTTPException(502,'Could not save the basemap preview. Rebuild MC² along with the app.')
            image_url=r.json()['image_url']
        return await install({'id':'custom-'+uuid4().hex,'name':name,'service_type':'image','filename':Path(file.filename or 'basemap.tif').name,
                              'definition':{'label':name,'kind':'raster','url':image_url,'image_bounds':bounds,'attribution':'User-provided basemap','opacity':1}})
    app.include_router(router)
