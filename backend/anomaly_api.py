"""Durable anomaly runs, map review and policy-driven background refits."""
import mission_control
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

import numpy as np
import rasterio
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from rasterio.features import rasterize, is_valid_geom, bounds as geometry_bounds
from rasterio.warp import transform, transform_geom

from algorithm_registry import ANOMALY_DETAILS
from anomaly import execute, make_view
import retraining


class AnomalyRequest(BaseModel):
    mission_id: str
    job_id: str
    image_ids: list[str] = Field(min_length=1)
    models: list[str] = Field(default_factory=lambda:list(ANOMALY_DETAILS), min_length=1)
    max_size: int = Field(default=192,ge=32,le=1024)
    percentile: float = Field(default=99.5,ge=50,lt=100)
    minimum_votes: int = Field(default=2,ge=1,le=4)
    outer_window: int = Field(default=63,ge=15,le=255)
    guard_window: int = Field(default=31,ge=3,le=127)
    epochs: int = Field(default=15,ge=1,le=200)
    background_mode: str = Field(default='scene',pattern='^(scene|reviewed_normal)$')
    geometry: dict | None = None


class AnomalyViewRequest(BaseModel):
    image_index: int = Field(ge=0)
    percentile: float = Field(default=99.5,ge=50,lt=100)
    minimum_votes: int = Field(default=2,ge=1,le=4)


class AnomalyReviewRequest(BaseModel):
    image_index: int = Field(ge=0)
    view_id: str
    label: str = Field(pattern='^(normal|anomaly)$')
    role: str = Field(default='train',pattern='^(train|validation)$')
    confidence: int = Field(default=3,ge=1,le=5)
    geometry: dict
    notes: str = Field(default='',max_length=500)


def now():return datetime.now(timezone.utc).isoformat()


def initialize(conn):
    conn.execute('CREATE TABLE IF NOT EXISTS anomaly_runs(id TEXT PRIMARY KEY,mission_id TEXT,job_id TEXT,payload TEXT,created_at TEXT)')
    conn.execute('CREATE TABLE IF NOT EXISTS anomaly_reviews(id TEXT PRIMARY KEY,run_id TEXT,image_index INTEGER,payload TEXT,created_at TEXT)')
    conn.execute('CREATE TABLE IF NOT EXISTS anomaly_views(id TEXT PRIMARY KEY,run_id TEXT,payload TEXT,created_at TEXT)')
    conn.execute('CREATE TABLE IF NOT EXISTS anomaly_training_batches(id TEXT PRIMARY KEY,mission_id TEXT,job_id TEXT,model_id TEXT,evidence_key TEXT,payload TEXT,UNIQUE(mission_id,job_id,model_id,evidence_key))')


def valid_geometry(geometry):
    if geometry.get('type') not in ('Point','Polygon','MultiPolygon') or not is_valid_geom(geometry):raise HTTPException(422,'Use a valid point or polygon')
    b=geometry_bounds(geometry)
    if not np.isfinite(b).all() or b[0]<-180 or b[2]>180 or b[1]<-90 or b[3]>90:raise HTTPException(422,'Geometry must use longitude/latitude')


def register(app,database,data,executor,map_client,ensure_map,tool_data,mc2_url):
    router=APIRouter(prefix='/api/anomalies')
    lock=asyncio.Lock()
    def job_context(mission_id,job_id):
        with database() as db:
            row=db.execute('SELECT payload FROM workspace WHERE id=1').fetchone()
            missions=json.loads(row['payload']).get('missions',[]) if row else []
            mission=next((m for m in missions if m['id']==mission_id),{})
            job=next((j for j in mission.get('jobs',[]) if j['id']==job_id),None)
            if not job:raise HTTPException(404,'Mission/job not found')
            access={}
            for row in db.execute('SELECT payload FROM raster_access WHERE mission_id=? AND job_id=? ORDER BY created_at DESC,rowid DESC',(mission_id,job_id)):
                record=json.loads(row['payload']);access.setdefault(record['image_id'],record)
            reviews=[]
            for row in db.execute('SELECT payload FROM anomaly_reviews ORDER BY created_at,rowid'):
                r=json.loads(row['payload'])
                if r['mission_id']==mission_id and r['job_id']==job_id:reviews.append(r)
        return job,access,reviews

    def persist(run):
        with database() as db:db.execute('UPDATE anomaly_runs SET payload=? WHERE id=?',(json.dumps(run),run['id']))
        folder=data/'anomalies'/run['id'];folder.mkdir(parents=True,exist_ok=True)
        (folder/'report.json').write_text(json.dumps(run,indent=2))

    def get_run(run_id):
        with database() as db:row=db.execute('SELECT payload FROM anomaly_runs WHERE id=?',(run_id,)).fetchone()
        if not row:raise HTTPException(404,'Anomaly run not found')
        return json.loads(row['payload'])

    def get_view(run,view_id):
        with database() as db:row=db.execute('SELECT payload FROM anomaly_views WHERE id=? AND run_id=?',(view_id,run['id'])).fetchone()
        if row:return json.loads(row['payload'])
        for output in run.get('outputs',[]):
            if output.get('default_view',{}).get('id')==view_id:return output['default_view']
        raise HTTPException(404,'Saved threshold view not found')

    def read_ranks(run,index):
        if not 0<=index<len(run.get('outputs',[])):raise HTTPException(404,'Anomaly image not found')
        ranks={};profile=None
        for mid in run['outputs'][index]['detectors']:
            with rasterio.open(data/'anomalies'/run['id']/f'{index}-{mid}-percentiles.tif') as src:
                ranks[mid]=src.read(1);profile=src.profile.copy()
        if not ranks:raise HTTPException(422,'Detectors have not finished for this image')
        return ranks,profile

    def burn(rows,profile):
        return rasterize([(transform_geom('EPSG:4326',profile['crs'],r['geometry']),i+1) for i,r in enumerate(rows)],
            out_shape=(profile['height'],profile['width']),transform=profile['transform'],fill=0,dtype='int32') if rows else np.zeros((profile['height'],profile['width']),dtype='int32')

    def validation(run,view):
        index=view['image_index'];ranks,profile=read_ranks(run,index);valid=next(iter(ranks.values()))>=0
        with database() as db:rows=[json.loads(row['payload']) for row in db.execute('SELECT payload FROM anomaly_reviews ORDER BY created_at,rowid')]
        refs=[r for r in rows if r['mission_id']==run['mission_id'] and r['job_id']==run['job_id'] and r['image_id']==run['outputs'][index]['image_id'] and r['role']=='validation']
        owner=burn(refs,profile);truth=np.zeros(valid.shape,dtype=bool)
        for i,r in enumerate(refs,1):truth[owner==i]=r['label']=='anomaly'
        evaluated=(owner>0)&valid
        if not evaluated.any():return view['metrics']
        def score(pred):
            tp=int((pred&truth&evaluated).sum());fp=int((pred&~truth&evaluated).sum());fn=int((~pred&truth&evaluated).sum())
            return {'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
                'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,'true_positive':tp,'false_positive':fp,'false_negative':fn}
        flags={mid:(rank>=view['percentile'])&valid for mid,rank in ranks.items()}
        selected=sum(flags.values())>=view['minimum_votes']
        return {'status':'measured','pixels':int(evaluated.sum()),'anomaly_reference_pixels':int((truth&evaluated).sum()),
            **score(selected),'detectors':{mid:score(flag) for mid,flag in flags.items()},
            'scope':'Human-designated validation regions only; score-driven review selection may bias these metrics. Not whole-scene benchmark accuracy.'}

    def readiness(mission_id,job_id):
        job,access,reviews=job_context(mission_id,job_id)
        with database() as db:
            policy=retraining.get_policy(db)
            runs=[json.loads(row['payload']) for row in db.execute('SELECT payload FROM anomaly_runs WHERE mission_id=? AND job_id=? ORDER BY created_at DESC,rowid DESC',(mission_id,job_id))]
            batches=[json.loads(row['payload']) for row in db.execute('SELECT payload FROM anomaly_training_batches WHERE mission_id=? AND job_id=?',(mission_id,job_id))]
        result=[]
        protected={i['id'] for i in job.get('images',[]) if i.get('benchmark_split') in ('validation','test')}
        for mid,spec in ANOMALY_DETAILS.items():
            consumed=set()
            for run in runs:
                if run.get('background_mode')=='reviewed_normal' and run['status']=='complete' and mid in run['requested_models']:consumed.update(run.get('label_ids',[]))
            latest={}
            for run in runs:
                if run['status']!='complete' or run.get('candidate_version'):continue
                for index,output in enumerate(run.get('outputs',[])):
                    if mid in output['detectors']:latest.setdefault(output['image_id'],(run,index))
            ids=set();image_ids=set();pixels=0;normal_counts={};baselines={}
            for image_id,(baseline,index) in latest.items():
                if image_id in protected:continue
                ranks,profile=read_ranks(baseline,index);valid=ranks[mid]>=0
                refs=[r for r in reviews if r['image_id']==image_id]
                owner=burn(refs,profile);held=burn([r for r in refs if r['role']=='validation'],profile)>0
                normal=np.zeros(valid.shape,dtype=bool)
                for i,r in enumerate(refs,1):
                    mask=(owner==i)&valid&~held
                    if r['role']!='train' or r['confidence']<policy['minimum_confidence']:continue
                    if r['label']=='normal':normal|=mask
                    if r['id'] in consumed:continue
                    wrong=(ranks[mid]>=r['percentile'])!=(r['label']=='anomaly')
                    count=int((mask&wrong).sum())
                    if count:ids.add(r['id']);image_ids.add(image_id);pixels+=count
                normal_counts[image_id]=int(normal.sum());baselines[image_id]=baseline['id']
            key=hashlib.sha256(json.dumps({'reviews':sorted(ids),'baselines':baselines,'policy':policy['version']},sort_keys=True).encode()).hexdigest()
            waiting=[]
            if not spec.get('trainable',False):waiting.append('This detector has no supported training adapter')
            if not policy['enabled']:waiting.append('Automatic training disabled')
            if len(ids)<policy['minimum_regions']:waiting.append(f'{len(ids)}/{policy["minimum_regions"]} corrected regions')
            if pixels<policy['minimum_corrected_pixels']:waiting.append(f'{pixels}/{policy["minimum_corrected_pixels"]} unique corrected pixels')
            if len(image_ids)<policy['minimum_images']:waiting.append(f'{len(image_ids)}/{policy["minimum_images"]} images')
            if not image_ids or any(normal_counts.get(i,0)<32 for i in image_ids):waiting.append('Need 32 approved normal-background pixels on every training image')
            if any(access.get(i,{}).get('status') not in ('stream-ready','downloaded','local-ready') for i in image_ids):waiting.append('Configure raster access')
            active=next((r for r in runs if r['status'] in ('queued','running','pausing','paused','stopping') and mid in r['requested_models']),None)
            if active:waiting.append('A detector run is already active')
            if any(b['model_id']==mid and b['evidence_key']==key for b in batches):waiting.append('This evidence already has a training attempt')
            result.append({'model_id':mid,'name':spec['name'],'ready':not waiting,'regions':len(ids),'corrected_pixels':pixels,
                'images':len(image_ids),'image_ids':sorted(image_ids),'review_ids':sorted(ids),'evidence_key':key,
                'normal_background_pixels':normal_counts,'waiting_for':waiting,'active_run_id':active['id'] if active else None,
                'max_size':max((r['max_size'] for r,_ in latest.values()),default=192)})
        return {'policy':policy,'algorithms':result}

    async def queue(request,evidence=None):
        if not mission_control.allowed(database,request.mission_id):
            raise HTTPException(409,'Resume this mission before starting another run')
        job,access,reviews=job_context(request.mission_id,request.job_id)
        unknown=set(request.models)-set(ANOMALY_DETAILS)
        if unknown:raise HTTPException(422,'Unknown anomaly detector')
        images=[i for i in job.get('images',[]) if i['id'] in request.image_ids]
        if len(images)!=len(set(request.image_ids)):raise HTTPException(422,'Choose images from this job')
        if any(i.get('benchmark_split') in ('validation','test') for i in images):
            raise HTTPException(422,'These detectors fit background statistics. Benchmark validation/test images are protected; choose training or non-benchmark imagery.')
        if request.outer_window%2!=1 or request.guard_window%2!=1 or request.guard_window<17 or request.outer_window<=request.guard_window:
            raise HTTPException(422,'Local RX needs odd windows: guard at least 17 pixels and outer larger than guard')
        if request.geometry:valid_geometry(request.geometry)
        if any(access.get(i['id'],{}).get('status') not in ('stream-ready','downloaded','local-ready') for i in images):raise HTTPException(422,'Configure readable raster access for every selected image')
        with database() as db:policy=retraining.get_policy(db)
        run={'id':str(uuid4()),'created_at':now(),**request.model_dump(),'requested_models':list(dict.fromkeys(request.models)),
            'geometry':request.geometry or job.get('aoi'),
            'status':'queued','stage':'Waiting for anomaly worker','progress':0,'outputs':[],
            'candidate_version':request.background_mode=='reviewed_normal','trigger':'correction_threshold' if evidence else 'manual',
            'run_kind':'training' if request.background_mode=='reviewed_normal' else 'anomaly_detection',
            'training_policy':policy,'label_ids':[r['id'] for r in reviews if r['role']=='train' and r['confidence']>=policy['minimum_confidence'] and r['image_id'] in request.image_ids],
            'interpretation':'Empirical within-image anomaly percentiles, not class probabilities. Detector agreement is not correctness.',
            'band_policy':'Every connected science channel, no PCA or spectral feature subset. Quality/alpha channels are excluded explicitly.',
            'reference_policy':'Validation geometries and confirmed anomalies excluded from fitting; reviewed mode uses only approved normal references. Benchmark holdout images are blocked from fitting.'}
        if evidence:run['correction_evidence']=evidence
        with database() as db:
            db.execute('INSERT INTO anomaly_runs VALUES(?,?,?,?,?)',(run['id'],run['mission_id'],run['job_id'],json.dumps(run),run['created_at']))
            if evidence:
                batch={**evidence,'run_id':run['id']}
                db.execute('INSERT INTO anomaly_training_batches VALUES(?,?,?,?,?,?)',(str(uuid4()),run['mission_id'],run['job_id'],evidence['model_id'],evidence['evidence_key'],json.dumps(batch)))
        loop=asyncio.get_running_loop()
        def worker():
            result=mission_control.execute(database,'anomaly_runs',run,execute,persist,images,access,reviews,data)
            if result['status']=='complete':loop.call_soon_threadsafe(lambda:asyncio.create_task(check_auto(run['mission_id'],run['job_id'])))
        executor.submit(worker)
        return run

    async def check_auto(mission_id,job_id):
        if not mission_control.allowed(database,mission_id):return {'status':'mission_suspended','runs':[]}
        async with lock:
            from starlette.concurrency import run_in_threadpool
            status=await run_in_threadpool(readiness,mission_id,job_id);queued=[]
            for item in status['algorithms']:
                if not item['ready']:continue
                try:queued.append(await queue(AnomalyRequest(mission_id=mission_id,job_id=job_id,image_ids=item['image_ids'],models=[item['model_id']],background_mode='reviewed_normal',max_size=item['max_size']),item))
                except sqlite3.IntegrityError:continue
            return {'status':'training_started' if queued else 'collecting_evidence','runs':queued}

    @router.get('/models')
    async def models():return {'models':ANOMALY_DETAILS}

    @router.post('/runs')
    async def create(request:AnomalyRequest):return await queue(request)

    @router.get('/runs')
    async def listing(mission_id:str,job_id:str|None=None):
        with database() as db:
            rows=db.execute('SELECT payload FROM anomaly_runs WHERE mission_id=? ORDER BY created_at DESC,rowid DESC',(mission_id,))
            runs=[json.loads(r['payload']) for r in rows]
        for run in runs:
            for index,output in enumerate(run.get('outputs',[])):
                if output.get('default_view'):output['default_view']=get_view(run,output['default_view']['id'])
                with database() as db:
                    saved=[json.loads(r['payload']) for r in db.execute('SELECT payload FROM anomaly_views WHERE run_id=? ORDER BY created_at DESC,rowid DESC',(run['id'],))]
                output['latest_view']=next((v for v in saved if v['image_index']==index),output.get('default_view'))
        return [r for r in runs if not job_id or r['job_id']==job_id]

    @router.get('/readiness')
    async def ready(mission_id:str,job_id:str):
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(readiness,mission_id,job_id)

    @router.get('/runs/{run_id}')
    async def detail(run_id:str):
        run=get_run(run_id)
        for output in run.get('outputs',[]):
            if output.get('default_view'):output['default_view']=get_view(run,output['default_view']['id'])
        return run

    @router.get('/runs/{run_id}/files/{filename}')
    async def artifact(run_id:str,filename:str):
        get_run(run_id);root=(data/'anomalies'/run_id).resolve();path=(root/filename).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.suffix not in ('.json','.png','.tif'):raise HTTPException(404,'Artifact not found')
        return FileResponse(path,media_type='image/png' if path.suffix=='.png' else 'image/tiff' if path.suffix=='.tif' else 'application/json')

    @router.post('/runs/{run_id}/views')
    async def threshold(run_id:str,request:AnomalyViewRequest):
        run=get_run(run_id)
        if run['status']!='complete':raise HTTPException(422,'Wait for this run to finish')
        if request.image_index>=len(run['outputs']):raise HTTPException(404,'Image not found')
        from starlette.concurrency import run_in_threadpool
        view=await run_in_threadpool(make_view,run,request.image_index,data,request.percentile,request.minimum_votes)
        view['metrics']=validation(run,view)
        with database() as db:db.execute('INSERT INTO anomaly_views VALUES(?,?,?,?)',(view['id'],run_id,json.dumps(view),now()))
        (data/'anomalies'/run_id/f'view-{view["id"]}.json').write_text(json.dumps(view,indent=2))
        return view

    @router.get('/runs/{run_id}/views/{view_id}/inspect')
    async def inspect(run_id:str,view_id:str,longitude:float,latitude:float):
        if not (-180<=longitude<=180 and -90<=latitude<=90):raise HTTPException(422,'Invalid coordinates')
        run=get_run(run_id);view=get_view(run,view_id);index=view['image_index'];ranks,profile=read_ranks(run,index)
        xs,ys=transform('EPSG:4326',profile['crs'],[longitude],[latitude]);col,row=(~profile['transform'])*(xs[0],ys[0]);col,row=int(np.floor(col)),int(np.floor(row))
        if not (0<=row<profile['height'] and 0<=col<profile['width']) or next(iter(ranks.values()))[row,col]<0:raise HTTPException(422,'Click a valid pixel inside this anomaly image')
        corners=[profile['transform']*(x,y) for x,y in [(col,row),(col+1,row),(col+1,row+1),(col,row+1),(col,row)]]
        votes=[]
        for mid,rank in ranks.items():
            with rasterio.open(data/'anomalies'/run_id/f'{index}-{mid}-scores.tif') as src:score=float(src.read(1,window=rasterio.windows.Window(col,row,1,1))[0,0])
            votes.append({'id':mid,'name':ANOMALY_DETAILS[mid]['name'],'score':score,'percentile':float(rank[row,col]),'flagged':bool(rank[row,col]>=view['percentile'])})
        return {'longitude':longitude,'latitude':latitude,'row':row,'column':col,'votes':votes,
            'flagged':sum(v['flagged'] for v in votes)>=view['minimum_votes'],
            'geometry':transform_geom(profile['crs'],'EPSG:4326',{'type':'Polygon','coordinates':[corners]})}

    @router.post('/runs/{run_id}/views/{view_id}/map')
    async def map_view(run_id:str,view_id:str,layer:str='votes'):
        run=get_run(run_id);view=get_view(run,view_id);index=view['image_index']
        if layer not in ('votes','mask',*run['outputs'][index]['detectors']):raise HTTPException(422,'Unknown anomaly layer')
        filename=f'{view["prefix"]}-{layer}-rgb.tif' if layer in ('votes','mask') else f'{index}-{layer}-rgb.tif'
        path=data/'anomalies'/run_id/filename;state=await ensure_map()
        current=tool_data(await map_client.call('list_assets',{'map_id':state['map_id']}))
        from mc2_client import private_http_client
        async with private_http_client(mc2_url,timeout=180) as client:
            with path.open('rb') as stream:
                response=await client.post(f'{mc2_url}/api/maps/{state["map_id"]}/upload-geotiff',files={'file':(filename,stream,'image/tiff')},
                    data={'mode':'rgb','bands':'1,2,3','name':f'Anomaly result · {run_id} · {layer}'})
            if response.is_error:raise HTTPException(502,response.text[:250])
        for asset in current.get('assets',[]):
            if str(asset.get('name','')).startswith(('Anomaly result · ','Analysis consensus · ','Analysis review · ','Analysis reviewed · ','Analysis agreement · ','Analysis disagreement · ','SAM proposal · ')):
                await map_client.call('delete_asset',{'map_id':state['map_id'],'asset_id':asset['asset_id']})
        await map_client.call('zoom',{'map_id':state['map_id'],'target':'bbox','bbox':run['outputs'][index]['bbox']})
        return {'ok':True}

    @router.post('/runs/{run_id}/reviews')
    async def save_review(run_id:str,request:AnomalyReviewRequest):
        run=get_run(run_id);view=get_view(run,request.view_id)
        if view['image_index']!=request.image_index:raise HTTPException(422,'View belongs to another image')
        valid_geometry(request.geometry);ranks,profile=read_ranks(run,request.image_index)
        region=(burn([{'geometry':request.geometry}],profile)>0)&(next(iter(ranks.values()))>=0)
        if not region.any():raise HTTPException(422,'Reference contains no valid anomaly pixels')
        record={'id':str(uuid4()),'run_id':run_id,'mission_id':run['mission_id'],'job_id':run['job_id'],
            'image_id':run['outputs'][request.image_index]['image_id'],**request.model_dump(),'created_at':now(),
            'percentile':view['percentile'],'minimum_votes':view['minimum_votes'],'pixels':int(region.sum()),'origin':'human_anomaly_review'}
        with database() as db:db.execute('INSERT INTO anomaly_reviews VALUES(?,?,?,?,?)',(record['id'],run_id,request.image_index,json.dumps(record),record['created_at']))
        folder=data/'anomalies'/run_id
        with database() as db:history=[json.loads(r['payload']) for r in db.execute('SELECT payload FROM anomaly_reviews WHERE run_id=? ORDER BY created_at,rowid',(run_id,))]
        (folder/'reviews.json').write_text(json.dumps(history,indent=2))
        metrics=validation(run,view);view['metrics']=metrics
        with database() as db:db.execute('INSERT INTO anomaly_views VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload',
            (view['id'],run_id,json.dumps(view),now()))
        (folder/f'validation-{record["id"]}.json').write_text(json.dumps({'view':view,'review_id':record['id'],'metrics':metrics},indent=2))
        automatic=await check_auto(run['mission_id'],run['job_id'])
        return {'review':record,'metrics':metrics,'automatic_training':automatic}

    @router.get('/runs/{run_id}/reviews')
    async def reviews(run_id:str):
        get_run(run_id)
        with database() as db:return [json.loads(r['payload']) for r in db.execute('SELECT payload FROM anomaly_reviews WHERE run_id=? ORDER BY created_at,rowid',(run_id,))]

    app.include_router(router)
    return check_auto
