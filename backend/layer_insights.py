"""Immutable result library and approved, bounded raster-algebra plans.

Chat is a deterministic command interpreter, never arbitrary Python/SQL or an LLM.
Raster pixels live beside SQLite; content hashes and lineage live in SQLite.
"""
from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from insight_llm import Planner
from uuid import uuid4

import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

MAX_PIXELS = 4_000_000
MAX_BYTES = 128 * 1024 * 1024
MAX_LIBRARY_BYTES = 8 * 1024**3
OPERATIONS = {
    'intersection': 'Intersection of binary masks (1 where both are 1; otherwise 0).',
    'union': 'Union of binary masks (1 where either is 1; otherwise 0).',
    'difference': 'Subtract numeric B from A (A − B).',
    'mean': 'Arithmetic mean of numeric A and B.',
    'change': 'Class change: 1 where A and B differ; 0 where they agree. Requires identical class legends.',
    'cross': 'Cross-classify A and B into a new class for each observed pair; preserve a pair legend.',
    'mask': 'Keep values from A only where binary mask B equals 1; other pixels become nodata.',
    'threshold': 'Create a binary mask using the stated comparison and threshold on A.',
    'class': 'Extract a single class value from A as a binary mask.',
}


def now(): return datetime.now(timezone.utc).isoformat()
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def clean_legend(legend):
    if not isinstance(legend,list) or len(legend)>4096: raise ValueError('Class legend must be a JSON array with at most 4096 entries')
    result=[]; seen=set()
    for entry in legend:
        if not isinstance(entry,dict): raise ValueError('Each legend entry must contain value and label')
        value=entry.get('value');label=entry.get('label')
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value!=int(value) or abs(value)>2**31 or value in seen: raise ValueError('Legend class values must be unique integer codes within the signed 32-bit range')
        if not isinstance(label,str) or not 1<=len(label)<=200: raise ValueError('Each class needs a label of 1–200 characters')
        item={'value':int(value),'label':label}
        color=entry.get('color')
        if color is not None:
            if not isinstance(color,str) or not re.fullmatch(r'#[0-9a-fA-F]{6}',color): raise ValueError('Legend colors use #RRGGBB')
            item['color']=color
        for key in ('a','b'):
            if key in entry: item[key]=entry[key]
        seen.add(value);result.append(item)
    return result


class SaveSource(BaseModel):
    source_id: str = Field(max_length=200)
    name: str = Field(min_length=1, max_length=120)
    kind: str = Field(default='continuous', pattern='^(continuous|categorical|binary)$')


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    layer_ids: list[str] = Field(default_factory=list, max_length=2)
    name: str = Field(default='Insight layer', min_length=1, max_length=120)
    previous_plan_id: str | None = Field(default=None, max_length=64)
    grid_max_pixels: int | None = Field(default=None,ge=10000,le=MAX_PIXELS)


class ImportPath(BaseModel):
    path: str = Field(min_length=1, max_length=1024)
    name: str = Field(min_length=1, max_length=120)
    kind: str = Field(pattern='^(continuous|categorical|binary)$')


def register(app, database, data, map_client, ensure_map, tool_data, mc2_url, analysis_executor=None):
    router = APIRouter(prefix='/api/layers', tags=['Layers & insights'])
    root = (data / 'layer-library').resolve()
    catalog = (data / 'layer-catalog').resolve()
    local = (data / 'local-rasters').resolve()
    worker = analysis_executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix='insight')
    planner = Planner(database, data, analysis_executor)
    planner.register(app)
    def init():
        root.mkdir(parents=True, exist_ok=True); catalog.mkdir(parents=True, exist_ok=True)
        with database() as db:
            db.execute('CREATE TABLE IF NOT EXISTS saved_layers (id TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS insight_plans (id TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at TEXT NOT NULL)')
    def put(table, item):
        with database() as db: db.execute(f'INSERT INTO {table} VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload', (item['id'],json.dumps(item),item['created_at']))
    def get(table, identifier):
        init()
        with database() as db: row = db.execute(f'SELECT payload FROM {table} WHERE id=?',(identifier,)).fetchone()
        if not row: raise HTTPException(404,'Layer or plan not found')
        return json.loads(row['payload'])
    def public(layer):
        return {k:v for k,v in layer.items() if k!='path'} | {'download_url':f'/api/layers/{layer["id"]}/download'}
    def path_for(layer):
        path = (root/layer['id']/'layer.tif').resolve()
        if not path.is_relative_to(root) or not path.is_file(): raise HTTPException(404,'Saved raster is unavailable')
        return path
    def inspect(path):
        if path.stat().st_size > MAX_LIBRARY_BYTES: raise ValueError('Saved layer exceeds the 8 GiB local library file limit')
        with rasterio.open(path, driver='GTiff') as src:
            if any(Path(f).resolve()!=path.resolve() for f in src.files): raise ValueError('Use a self-contained GeoTIFF; embed external masks, auxiliary metadata and overviews before import')
            if not src.crs: raise ValueError('GeoTIFF must include its CRS and georeferencing')
            if src.count!=1: raise ValueError('Save one analytical band at a time; RGB display composites are not analytical layers')
            if src.width*src.height>500_000_000: raise ValueError('Layer exceeds the 500 million pixel library safety limit')
            if not np.isfinite(tuple(src.transform)).all() or src.transform.a==0 or src.transform.e==0: raise ValueError('Invalid raster transform')
            if np.dtype(src.dtypes[0]).kind not in 'uif': raise ValueError('Only real numeric rasters are supported')
            bounds = list(transform_bounds(src.crs,'EPSG:4326',*src.bounds,densify_pts=21))
            if not np.isfinite(bounds).all() or not -180<=bounds[0]<bounds[2]<=180 or not -90<=bounds[1]<bounds[3]<=90:
                raise ValueError('Use a valid geographic footprint that does not cross the antimeridian; reproject or split this raster first')
            return {'crs':str(src.crs),'transform':list(src.transform),'width':src.width,'height':src.height,
                    'dtype':src.dtypes[0],'nodata':str(src.nodata),'bbox':bounds,'band':1,
                    'scale':src.scales[0],'offset':src.offsets[0],'units':src.units[0],
                    'description':src.descriptions[0]}
    def save_file(source, name, kind, provenance, legend=None):
        init(); identifier=uuid4().hex; folder=root/identifier; folder.mkdir()
        try:
            inspect(source)
            legend=clean_legend(legend or [])
            if kind=='binary' and not legend: legend=[{'value':0,'label':'False'},{'value':1,'label':'True'}]
            # Reject half-written/changing outputs instead of saving a mixed snapshot.
            initial_hash=digest(source)
            target=folder/'layer.tif'; shutil.copyfile(source,target)
            info=inspect(target)
            saved_hash=digest(target)
            if saved_hash!=initial_hash or digest(source)!=initial_hash: raise ValueError('Source changed while saving; wait for its operation to finish and retry')
            with rasterio.open(target) as src:
                has_values=False
                for _,window in src.block_windows(1):
                    values=src.read(1,window=window,masked=True).compressed();values=values[np.isfinite(values)]
                    if not len(values):continue
                    has_values=True
                    if kind=='binary' and not np.isin(values,[0,1]).all():raise ValueError('Binary layers must contain only 0 and 1, excluding nodata')
                    if kind=='categorical' and (not np.equal(values,np.floor(values)).all() or np.abs(values).max()>2**31):raise ValueError('Categorical layers require integer class codes within the signed 32-bit range')
                if not has_values:raise ValueError('Layer contains no valid finite pixels')
            item={'id':identifier,'name':name,'kind':kind,'created_at':now(),'sha256':saved_hash,
                  'grid':info,'legend':legend,'provenance':provenance}
            put('saved_layers',item); return item
        except Exception:
            shutil.rmtree(folder,ignore_errors=True); raise
    def candidates():
        init(); items=[]
        with database() as db:
            workspace=db.execute('SELECT payload FROM workspace WHERE id=1').fetchone()
            missions=json.loads(workspace['payload']).get('missions',[]) if workspace else []
            mission_names={m['id']:m.get('name',m['id']) for m in missions}
            job_names={(m['id'],j['id']):j.get('name',j['id']) for m in missions for j in m.get('jobs',[])}
            for table, family, dirname in [('inference_runs','lulc','results'),('anomaly_runs','anomaly','anomalies'),('segmentation_runs','segmentation','segmentations')]:
                for row in db.execute(f'SELECT payload FROM {table} ORDER BY rowid DESC'):
                    run=json.loads(row['payload']); rid=run['id']
                    if run.get('status') not in ('complete','review-ready'): continue
                    folder=(data/dirname/rid).resolve()
                    if not folder.is_relative_to((data/dirname).resolve()): continue
                    def add(filename,label,kind,legend=None,image=None):
                        p=(folder/filename).resolve()
                        if not p.is_relative_to(folder) or not p.is_file() or p.suffix!='.tif': return
                        sid=hashlib.sha256(f'{family}/{rid}/{filename}'.encode()).hexdigest()
                        items.append({'source_id':sid,'name':label,'kind':kind,'legend':legend or [],'family':family,
                                      'mission_id':run.get('mission_id'),'job_id':run.get('job_id'),'run_id':rid,'image_id':image,
                                      'mission_name':mission_names.get(run.get('mission_id'),run.get('mission_id')),
                                      'job_name':job_names.get((run.get('mission_id'),run.get('job_id')),run.get('job_id')),
                                      'filename':filename,'path':str(p)})
                    if family=='lulc':
                        for index,output in enumerate(run.get('outputs',[])):
                            files=dict(output.get('files',{}))
                            if (folder/f'{index}-reviewed.tif').is_file(): files['reviewed']=f'{index}-reviewed.tif'
                            for key,url in files.items():
                                if key in ('agreement','disagreement') and output.get('agreement_summary',{}).get('models_compared',0)<2: continue
                                filename=url.rsplit('/',1)[-1]
                                kind='binary' if key in ('disagreement','agreement') else 'continuous' if key=='confidence' else 'categorical'
                                add(filename,f'{output.get("image_id",rid)} · {key}',kind,output.get('class_legend') if kind=='categorical' else None,output.get('image_id'))
                    elif family=='anomaly':
                        for i,output in enumerate(run.get('outputs',[])):
                            for mid,det in output.get('detectors',{}).items():
                                for key in ('scores','percentiles'):
                                    if det.get(key): add(det[key].rsplit('/',1)[-1],f'{output.get("image_id",i)} · {mid} {key}','continuous',image=output.get('image_id'))
                            view=output.get('default_view',{})
                            for key in ('mask','votes'):
                                if view.get(key): add(view[key].rsplit('/',1)[-1],f'{output.get("image_id",i)} · anomaly {key}','binary' if key=='mask' else 'continuous',image=output.get('image_id'))
                        for vr in db.execute('SELECT payload FROM anomaly_views WHERE run_id=?',(rid,)):
                            view=json.loads(vr['payload'])
                            for key in ('mask','votes'):
                                if view.get(key): add(view[key].rsplit('/',1)[-1],f'{rid[:8]} · threshold {view.get("percentile")} · {key}','binary' if key=='mask' else 'continuous')
                    else:
                        for c in run.get('candidates',[]):
                            add(c['mask'].rsplit('/',1)[-1],f'SAM proposal {c["id"]} · {rid[:8]}','binary',image=run.get('image_id'))
        # Duplicate threshold records may refer to the same artifact.
        return list({i['source_id']:i for i in items}.values())

    @router.get('')
    async def library():
        init()
        with database() as db: layers=[public(json.loads(r['payload'])) for r in db.execute('SELECT payload FROM saved_layers ORDER BY created_at DESC')]
        files=[]
        for p in list(catalog.glob('*.tif'))[:500]:
            known=next((l for l in layers if l['id']==p.stem),None)
            files.append({'path':str(p.relative_to(data.resolve())), 'name':known['name'] if known else p.name,
                          'kind':known['kind'] if known else 'continuous'})
        return {'layers':layers,'catalog':files,'catalog_directory':str(catalog),'import_roots':[str(catalog),str(local)],'operations':OPERATIONS}

    @router.get('/sources')
    async def sources(): return {'sources':[{k:v for k,v in i.items() if k!='path'} for i in await run_in_threadpool(candidates)]}

    @router.post('/save')
    async def save(request:SaveSource):
        source=next((i for i in await run_in_threadpool(candidates) if i['source_id']==request.source_id),None)
        if not source: raise HTTPException(404,'Source result not found')
        try: item=await run_in_threadpool(save_file,Path(source['path']),request.name,source['kind'],{k:v for k,v in source.items() if k!='path'},source['legend'])
        except (ValueError,rasterio.errors.RasterioError) as e: raise HTTPException(422,str(e))
        return public(item)

    @router.post('/import-path')
    async def import_path(request:ImportPath):
        init(); p=Path(request.path).expanduser()
        if not p.is_absolute(): p=data/p
        p=p.resolve()
        if not any(p.is_relative_to(r) for r in (catalog,local)) or p.suffix.lower() not in ('.tif','.tiff') or not p.is_file():
            raise HTTPException(422,'Choose a GeoTIFF under the displayed catalog or local-rasters folder, or upload from your computer')
        sidecar=p.with_suffix('.json'); legend=[]
        if sidecar.is_file() and sidecar.stat().st_size<1024*1024:
            try: legend=json.loads(sidecar.read_text()).get('legend',[])
            except (ValueError,AttributeError): pass
        try: item=await run_in_threadpool(save_file,p,request.name,request.kind,{'source':'local catalog','filename':p.name},legend)
        except (ValueError,rasterio.errors.RasterioError) as e: raise HTTPException(422,str(e))
        return public(item)

    @router.post('/upload')
    async def upload(file:UploadFile=File(...),name:str=Form(...),kind:str=Form('continuous'),legend:str=Form('[]',max_length=250000)):
        init()
        if kind not in ('continuous','categorical','binary') or not 1<=len(name)<=120: raise HTTPException(422,'Choose a valid name and layer type')
        temp=root/(uuid4().hex+'.upload')
        try:
            count=0
            with temp.open('wb') as f:
                while chunk:=await file.read(1024*1024):
                    count+=len(chunk)
                    if count>MAX_BYTES: raise HTTPException(413,'Upload limit is 128 MiB')
                    f.write(chunk)
            item=await run_in_threadpool(save_file,temp,name,kind,{'source':'user upload','filename':Path(file.filename or 'layer.tif').name},json.loads(legend))
            return public(item)
        except (ValueError,rasterio.errors.RasterioError) as e: raise HTTPException(422,str(e))
        finally: temp.unlink(missing_ok=True); await file.close()

    @router.get('/{identifier}/download')
    async def download(identifier:str):
        item=get('saved_layers',identifier); return FileResponse(path_for(item),media_type='image/tiff',filename=f'{identifier}.tif')

    @router.get('/{identifier}/metadata')
    async def metadata(identifier:str): return public(get('saved_layers',identifier))

    @router.post('/{identifier}/export')
    async def export(identifier:str):
        item=get('saved_layers',identifier); p=path_for(item); init()
        def write():
            target=catalog/f'{identifier}.tif'
            if not target.exists(): shutil.copyfile(p,target)
            if digest(target)!=item['sha256']: raise ValueError('Catalog file differs from the saved layer; export stopped without overwriting it')
            sidecar=target.with_suffix('.json')
            if not sidecar.exists(): sidecar.write_text(json.dumps(public(item),indent=2))
            return {'path':str(target),'catalog_path':str(target.relative_to(data.resolve())),'sha256':item['sha256']}
        try: return await run_in_threadpool(write)
        except ValueError as e: raise HTTPException(409,str(e))

    def interpret(message):
        s=message.strip().lower().rstrip('.!?').strip()
        s=re.sub(r'^(please\s+)?(create |make |show |compute |calculate )?(a |an |the )?', '', s)
        rules=[(r'(intersection|intersect)( of)?( (a and b|a with b|a,? b|the masks))?', 'intersection'),
               (r'union( of)?( (a and b|a with b|a,? b|the masks))?', 'union'),
               (r'(subtract b from a|a\s*-\s*b|difference( between a and b)?)','difference'),
               (r'(mean|average)( of)?( (a and b|a,? b))?', 'mean'),
               (r'(change( detection| map)?|changes between a and b|where a and b differ)', 'change'),
               (r'(cross.classify|cross classification|cross)( a and b)?', 'cross'),
               (r'(mask a (by|with) b|keep a where b (is|equals|=) 1)', 'mask')]
        for pattern,op in rules:
            if re.fullmatch(pattern,s): return {'operation':op}
        m=re.fullmatch(r'(?:threshold )?a\s*(>=|<=|>|<|==|=|!=)\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)',s)
        if m: return {'operation':'threshold','comparison':'==' if m[1]=='=' else m[1],'value':float(m[2])}
        m=re.fullmatch(r'(?:extract |select )?(?:class )(-?\d+)(?: from a| in a)?',s)
        if m: return {'operation':'class','value':int(m[1])}
        raise ValueError('I could not interpret that safely. Use one of the examples, such as “subtract B from A”, “mask A with B”, “cross-classify A and B”, “A >= 95”, or “extract class 6 from A”. No operation has been scheduled.')

    @router.post('/chat/plan')
    async def plan(request:ChatRequest):
        layers=[get('saved_layers',i) for i in request.layer_ids]
        engine={'provider':'local-command-interpreter','model':None}
        routing='Recognized an explicit supported command; no LLM call. Raster engine runs only after approval.'
        try:
            try: command=interpret(request.message)
            except ValueError:
                if not planner.config()['enabled']: raise
                previous=None
                if request.previous_plan_id:
                    old=get('insight_plans',request.previous_plan_id)
                    previous={k:old[k] for k in ('message','summary','operation','inputs')}
                command,engine=await planner.propose(request.message,layers,previous)
                routing='Conversational request sent to the configured LLM for interpretation. Raster execution remains local and requires approval.'
                if command is None:
                    return {'kind':'reply','reply':engine['interpretation'],'planner':engine,
                            'routing':'LLM explanation or clarification only. No spatial operation was scheduled.'}
        except ValueError as e: raise HTTPException(422,str(e))
        unary=command['operation'] in ('threshold','class')
        if len(request.layer_ids)!=(1 if unary else 2) or len(set(request.layer_ids))!=len(request.layer_ids):
            raise HTTPException(422,'Select exactly one layer for threshold/class extraction, or two distinct layers for other operations')
        op=command['operation']
        if op in ('intersection','union') and any(l['kind']!='binary' for l in layers): raise HTTPException(422,'First extract classes or threshold values to create binary masks')
        if op in ('difference','mean') and any(l['kind']!='continuous' for l in layers): raise HTTPException(422,'Numeric arithmetic requires continuous layers, not class codes or binary masks')
        if op in ('change','cross','class') and any(l['kind']!='categorical' for l in layers): raise HTTPException(422,'This operation requires categorical layers')
        if op=='change' and (not layers[0]['legend'] or layers[0]['legend']!=layers[1]['legend']): raise HTTPException(422,'Change detection requires identical, recorded class legends. Use cross-classification for different taxonomies.')
        if op=='mask' and layers[1]['kind']!='binary': raise HTTPException(422,'B must be a binary mask')
        if op=='threshold' and layers[0]['kind']!='continuous': raise HTTPException(422,'Thresholds require a continuous layer; use class extraction for categorical maps')
        if 'value' in command and not np.isfinite(command['value']): raise HTTPException(422,'Threshold must be finite')
        summary=OPERATIONS[op]
        if 'value' in command: summary+=f' Value: {command.get("comparison","class =")} {command["value"]}.'
        grid=layers[0]['grid'];width,height=grid['width'],grid['height']
        budget=request.grid_max_pixels
        if width*height>MAX_PIXELS and budget is None:
            raise HTTPException(422,'Select a smaller Processing grid below the chat before preparing a plan. Originals remain unchanged; the plan will show the exact resolution before approval.')
        factor=max(1,(width*height/(budget or MAX_PIXELS))**.5)
        out_width,out_height=max(1,int(width/factor)),max(1,int(height/factor))
        affine=rasterio.Affine(*grid['transform'][:6])*rasterio.Affine.scale(width/out_width,height/out_height)
        output_grid={'crs':grid['crs'],'width':out_width,'height':out_height,'transform':list(affine)[:6]}
        reduced=(out_width,out_height)!=(width,height)
        grid_policy=f"Output: {out_width} × {out_height} pixels in {grid['crs']}; pixel sizes {abs(affine.a):.8g} × {abs(affine.e):.8g} CRS units. "+('Explicitly reduced processing grid; original saved layers remain unchanged. ' if reduced else 'Native A grid retained. ')+"All classes/masks use nearest neighbor; continuous values use bilinear. Match B to this exact grid. No displacement estimation."
        p={'id':uuid4().hex,'created_at':now(),'status':'awaiting_approval','message':request.message,'name':request.name,
           **command,'planner':engine,'routing':routing,'execution_engine':'local raster engine',
           'layer_ids':request.layer_ids,'input_hashes':[l['sha256'] for l in layers],
           'summary':summary,'inputs':[{'alias':chr(65+i),'id':l['id'],'name':l['name'],'kind':l['kind'],'units':l['grid'].get('units'),'scale':l['grid'].get('scale',1),'offset':l['grid'].get('offset',0)} for i,l in enumerate(layers)],
           'grid_policy':grid_policy,'output_grid':output_grid,'grid_reduced':reduced,'resampling':{l['id']:'bilinear' if l['kind']=='continuous' else 'nearest' for l in layers},
           'nodata_policy':'Evaluate only pixels where every input has finite valid data on A’s grid. Non-overlap is nodata, including union. Mask output is nodata wherever B is 0.',
           'limitations':'Operations use raw stored values; scale/offset tags are not applied and units are not harmonized. Check the input units and scale below. Single-band rasters; the approved processing grid is limited to 4 million pixels. Reduced grids may omit small regions. Full original inputs are preserved. Derived layers are not ground truth and do not trigger model training.'}
        put('insight_plans',p); return p

    def execute(p):
        from contextlib import ExitStack
        from resource_admission import reserve
        resources = ExitStack()
        temp=root/(p['id']+'.tif')
        try:
            resources.enter_context(reserve('Insights raster operation',2))
            p.update(status='running',progress=10,stage='Reading reference grid');put('insight_plans',p)
            layers=[get('saved_layers',i) for i in p['layer_ids']]
            arrays=[]
            for i,l in enumerate(layers):
                source=path_for(l)
                if digest(source)!=p['input_hashes'][i]: raise ValueError('Input changed since approval was requested')
                with rasterio.open(source) as src:
                    if i==0:
                        profile=src.profile.copy()
                        if p.get('output_grid'):
                            grid=p['output_grid'];profile.update(crs=grid['crs'],transform=rasterio.Affine(*grid['transform']),width=grid['width'],height=grid['height'])
                        if profile['width']*profile['height']>MAX_PIXELS:raise ValueError('Approved grid exceeds processing budget; prepare a new plan with a smaller Processing grid')
                    with WarpedVRT(src,crs=profile['crs'],transform=profile['transform'],width=profile['width'],height=profile['height'],dtype='float64',nodata=float('nan'),resampling=Resampling.bilinear if l['kind']=='continuous' else Resampling.nearest) as vrt:
                        a=vrt.read(1,masked=True)
                    arrays.append(a.filled(np.nan))
            p.update(progress=50,stage='Applying approved raster operation');put('insight_plans',p)
            a=arrays[0]; b=arrays[-1]; valid=np.logical_and.reduce([np.isfinite(x) for x in arrays]); op=p['operation'];legend=[];kind='binary'
            if not valid.any(): raise ValueError('Inputs have no valid overlapping pixels on A’s grid')
            if op=='intersection': result=(a==1)&(b==1)
            elif op=='union': result=(a==1)|(b==1)
            elif op=='difference': result=a-b;kind='continuous'
            elif op=='mean': result=(a+b)/2;kind='continuous'
            elif op=='change': result=a!=b
            elif op=='mask': result=a;valid&=b==1;kind=layers[0]['kind'];legend=layers[0]['legend']
            elif op=='class': result=a==p['value']
            elif op=='threshold': result={'>':np.greater,'>=':np.greater_equal,'<':np.less,'<=':np.less_equal,'==':np.equal,'!=':np.not_equal}[p['comparison']](a,p['value'])
            elif op=='cross':
                pairs,inverse=np.unique(np.stack([a[valid],b[valid]],axis=1),axis=0,return_inverse=True)
                if len(pairs)>4096: raise ValueError('Cross classification would exceed 4096 class pairs; subset or simplify classes')
                result=np.zeros(a.shape);result[valid]=inverse+1;kind='categorical'
                def label(l,v): return next((str(c.get('label',v)) for c in l['legend'] if c.get('value')==v),str(int(v)))
                legend=[{'value':i+1,'label':f'{label(layers[0],x)} × {label(layers[1],y)}','a':int(x),'b':int(y)} for i,(x,y) in enumerate(pairs)]
            valid &= np.isfinite(result)
            if not valid.any(): raise ValueError('Approved operation produced no valid pixels')
            if kind=='binary': legend=[{'value':0,'label':'False'},{'value':1,'label':'True'}]
            profile.update(driver='GTiff',count=1,dtype='float64',nodata=float('nan'),compress='deflate')
            # Source tiling can be incompatible with a new single-band dtype/profile.
            for k in ('photometric','blockxsize','blockysize','tiled','interleave'): profile.pop(k,None)
            with rasterio.open(temp,'w',**profile) as dst:
                dst.write(np.where(valid,result,np.nan).astype('float64'),1)
                dst.update_tags(INSIGHT_PLAN=p['id'],INPUTS=json.dumps(p['layer_ids']),OPERATION=op)
                if op=='mask':
                    dst.scales=(layers[0]['grid'].get('scale',1),);dst.offsets=(layers[0]['grid'].get('offset',0),)
                    if layers[0]['grid'].get('units'): dst.set_band_unit(1,layers[0]['grid']['units'])
            p.update(progress=85,stage='Saving immutable layer and lineage');put('insight_plans',p)
            item=save_file(temp,p['name'],kind,{'source':'approved insight','plan_id':p['id'],
                'input_layers':p['layer_ids'],'input_hashes':p['input_hashes'],'operation':op,
                'parameters':{k:p[k] for k in ('comparison','value') if k in p},'chat':p['message'],
                'planner':p['planner'],'routing':p['routing'],'execution_engine':'local raster engine',
                'inputs':[{'id':l['id'],'name':l['name'],'sha256':l['sha256'],'kind':l['kind'],'grid':l['grid']} for l in layers],
                'grid_policy':p['grid_policy'],'output_grid':p.get('output_grid'),'resampling':p.get('resampling'),'nodata_policy':p['nodata_policy'],'approved_at':p['approved_at']},legend)
            p.update(status='complete',progress=100,stage='Result saved',output_layer_id=item['id'],valid_pixels=int(valid.sum()))
        except Exception as e: p.update(status='failed',stage='Operation failed',error=str(e)[:1000])
        finally:
            resources.close();temp.unlink(missing_ok=True);put('insight_plans',p)

    @router.post('/plans/{identifier}/approve')
    async def approve(identifier:str):
        p=get('insight_plans',identifier)
        if p['status']!='awaiting_approval': raise HTTPException(409,'This plan has already been approved or closed')
        # Compare-and-set reserves this exact, immutable plan once.
        original=json.dumps(p);p.update(status='queued',progress=0,stage='Queued',approved_at=now())
        with database() as db:
            updated=db.execute('UPDATE insight_plans SET payload=? WHERE id=? AND payload=?',(json.dumps(p),identifier,original)).rowcount
        if not updated: raise HTTPException(409,'Plan already approved')
        asyncio.get_running_loop().run_in_executor(worker,execute,p)
        return p

    @router.post('/plans/{identifier}/cancel')
    async def cancel(identifier:str):
        p=get('insight_plans',identifier)
        if p['status']!='awaiting_approval': raise HTTPException(409,'Only an unapproved plan can be cancelled')
        original=json.dumps(p);p.update(status='cancelled')
        with database() as db:
            n=db.execute('UPDATE insight_plans SET payload=? WHERE id=? AND payload=?',(json.dumps(p),identifier,original)).rowcount
        if not n: raise HTTPException(409,'Plan already approved')
        return p

    @router.get('/plans/history')
    async def history():
        init()
        with database() as db: return {'plans':[json.loads(r['payload']) for r in db.execute('SELECT payload FROM insight_plans ORDER BY created_at DESC LIMIT 100')]}

    @router.get('/plans/{identifier}')
    async def status(identifier:str): return get('insight_plans',identifier)

    @router.post('/{identifier}/map')
    async def show(identifier:str):
        item=get('saved_layers',identifier);path=path_for(item);state=await ensure_map()
        display_legend=[]
        if item['kind'] in ('binary','categorical'):
            def preview():
                import colorsys
                with rasterio.open(path) as src:
                    factor=min(1,1024/max(src.width,src.height));height,width=max(1,round(src.height*factor)),max(1,round(src.width*factor))
                    values=src.read(1,out_shape=(height,width),resampling=Resampling.nearest,masked=True).astype('float64');valid=~np.ma.getmaskarray(values)&np.isfinite(values.data)
                    unique=np.unique(values.data[valid])
                    if len(unique)>4096: raise ValueError('More than 4096 classes; export the layer for detailed GIS visualization')
                    rgb=np.zeros((3,height,width),dtype='uint8');legend=[]
                    for i,v in enumerate(unique):
                        spec=next((l for l in item['legend'] if l.get('value')==v),{})
                        color=spec.get('color','')
                        if not re.fullmatch(r'#[0-9a-fA-F]{6}',color):
                            color=('#264d58' if v==0 else '#88e6ba') if item['kind']=='binary' else '#'+''.join(f'{round(c*255):02x}' for c in colorsys.hsv_to_rgb((i*.61803398875)%1,.5,.95))
                        mask=valid&(values.data==v)
                        for b in range(3): rgb[b,mask]=int(color[1+b*2:3+b*2],16)
                        legend.append({'value':int(v),'label':spec.get('label',str(int(v))),'color':color})
                    profile=src.profile.copy();profile.update(count=3,dtype='uint8',nodata=None,compress='deflate',height=height,width=width,transform=src.transform*rasterio.Affine.scale(src.width/width,src.height/height))
                    for k in ('photometric','blockxsize','blockysize','tiled','interleave'): profile.pop(k,None)
                    preview_path=path.with_name('display.tif')
                    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                        with rasterio.open(preview_path,'w',**profile) as dst: dst.write(rgb);dst.write_mask(valid.astype('uint8')*255)
                    return preview_path,legend
            from resource_admission import run_reserved
            try: path,display_legend=await run_in_threadpool(run_reserved,'Result map preparation',2,preview)
            except ValueError as e: raise HTTPException(422,str(e))
        from mc2_client import private_http_client
        async with private_http_client(mc2_url,timeout=180) as client:
            with path.open('rb') as stream:
                r=await client.post(f'{mc2_url}/api/maps/{state["map_id"]}/upload-geotiff',files={'file':(path.name,stream,'image/tiff')},
                                    data={'mode':'rgb' if display_legend else 'singleband','bands':'1,2,3','band':'1','colormap':'viridis','name':f'Saved layer · {item["name"]} · {identifier[:8]}'})
            if r.is_error: raise HTTPException(502,r.text[:250])
        await map_client.call('zoom',{'map_id':state['map_id'],'target':'bbox','bbox':item['grid']['bbox']})
        return {'ok':True,'map_url':state['url'],'layer':public(item),'display_legend':display_legend,
                'rendering':'Class colors shown in the legend; exported GeoTIFF retains the original class codes.' if display_legend else 'Viridis display stretch; download contains original numeric values. Legend values are listed in the library.'}

    @app.on_event('startup')
    async def recover():
        init()
        with database() as db:
            for row in db.execute('SELECT payload FROM insight_plans').fetchall():
                p=json.loads(row['payload'])
                if p['status'] in ('running','queued'):
                    p.update(status='failed',stage='Interrupted by application restart',error='Create a new plan to retry; original inputs are unchanged.')
                    db.execute('UPDATE insight_plans SET payload=? WHERE id=?',(json.dumps(p),p['id']))
    app.include_router(router)
