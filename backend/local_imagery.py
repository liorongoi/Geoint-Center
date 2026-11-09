"""Revisioned local imagery catalog and checkbox-controlled folder monitoring."""
from __future__ import annotations
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import threading
import time
from uuid import uuid4
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from sensor_metadata import is_metadata, MAX_METADATA_BYTES
from imagery_products import SUFFIXES, inspect, products, checksum, now, temporal_hls, inferred_key, auxiliary_raster, catalog_image_product, standalone_auxiliary

class WatchRequest(BaseModel):
    path: str
    enabled: bool  # Mandatory checkbox state, never inferred or reset.
    recursive: bool = True
    interval_seconds: int = Field(default=30, ge=10, le=3600)
    settle_seconds: int = Field(default=15, ge=2, le=600)

class WatchToggle(BaseModel):
    enabled: bool

class GroupReview(BaseModel):
    sensor: str | None = Field(default=None, pattern='^(hls-l30|hls-s30|maxar|cbers-4a-wpm)$')
    sources: list[dict] = Field(min_length=1, max_length=4096)
    acknowledge_warnings: bool = False

class HLSBundleRequest(BaseModel):
    image_ids: list[str] = Field(min_length=3,max_length=3)

class Catalog:
    def __init__(self, database, root):
        self.database, self.root = database, Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock, self.snapshot_lock = threading.Lock(), threading.Lock()
        self.stop, self.task = threading.Event(), None
        with self.database() as db:
            db.execute('CREATE TABLE IF NOT EXISTS local_imagery (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS imagery_watches (id TEXT PRIMARY KEY, path TEXT NOT NULL UNIQUE, payload TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS imagery_watch_files (watch_id TEXT NOT NULL, path TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(watch_id,path))')
            db.execute('CREATE TABLE IF NOT EXISTS imagery_group_reviews (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')

    def watches(self):
        with self.database() as db:
            return [json.loads(r['payload']) for r in db.execute('SELECT payload FROM imagery_watches ORDER BY path')]

    def watch(self, ident):
        with self.database() as db:
            row = db.execute('SELECT payload FROM imagery_watches WHERE id=?',(ident,)).fetchone()
        if not row: raise ValueError('Monitored folder was not found')
        return json.loads(row['payload'])

    def update_watch(self, ident, generation_guard=None, **changes):
        with self.database() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT payload FROM imagery_watches WHERE id=?',(ident,)).fetchone()
            if row:
                current=json.loads(row['payload'])
                if generation_guard and current['generation']!=generation_guard:return current
                value = {**current, **changes}
                db.execute('UPDATE imagery_watches SET payload=? WHERE id=?',(json.dumps(value),ident))
                return value

    def add_watch(self, request):
        supplied = request.path.strip()
        if not supplied or supplied.startswith(('\\\\','//','/vsi')) or '://' in supplied:
            raise ValueError('Choose an absolute local folder on the computer running the app')
        p = Path(supplied).expanduser()
        if not p.is_absolute() or not p.is_dir() or p.is_symlink():
            raise ValueError('The local folder must exist and must not be a symbolic link')
        p = p.resolve()
        configured=os.getenv('LOCAL_WATCH_ROOTS')
        if configured:
            allowed=[Path(value).resolve() for value in json.loads(configured)]
            if not any(p.is_relative_to(parent) for parent in allowed):
                raise ValueError('This deployment permits monitoring only under: '+', '.join(str(q) for q in allowed))
        if os.name=='nt':
            import ctypes
            if ctypes.windll.kernel32.GetDriveTypeW(str(p.anchor))==4:
                raise ValueError('Offline folder monitoring requires a local drive, not a mapped network share')
        if p.is_relative_to(self.root): raise ValueError('Application-managed storage cannot monitor itself')
        for existing in self.watches():
            q = Path(existing['path'])
            if p == q: raise ValueError('This folder is already listed; change its checkbox instead')
            if p.is_relative_to(q) or q.is_relative_to(p):
                raise ValueError('Monitored folders cannot overlap; use one recursive parent folder')
        value = {**request.model_dump(),'id':uuid4().hex,'path':str(p),'generation':uuid4().hex,
                 'status':'waiting' if request.enabled else 'paused','last_scan':None,
                 'last_attempt':None,'error':None,'issues':[],'products':0}
        with self.database() as db:
            db.execute('INSERT INTO imagery_watches VALUES(?,?,?)',(value['id'],str(p),json.dumps(value)))
        return value

    def images(self, include_history=False):
        with self.database() as db:
            values = [json.loads(r['payload']) for r in db.execute('SELECT payload FROM local_imagery')]
        return [i for i in values if include_history or (i.get('active',True) and catalog_image_product(i))]

    def remove_watch(self,ident):
        with self.database() as db:
            db.execute('BEGIN IMMEDIATE')
            for row in db.execute('SELECT id,payload FROM local_imagery').fetchall():
                image=json.loads(row['payload'])
                if image.get('watch_id')==ident:
                    image.update(active=False,inactive_reason='Folder removed from monitored catalog; originals and job snapshots retained')
                    db.execute('UPDATE local_imagery SET payload=? WHERE id=?',(json.dumps(image),row['id']))
            db.execute('DELETE FROM imagery_watch_files WHERE watch_id=?',(ident,))
            db.execute('DELETE FROM imagery_watches WHERE id=?',(ident,))

    def save_products(self, items, watch_id=None, db=None):
        def save(conn):
            for item in items:
                item = {**item,'watch_id':watch_id}
                previous=conn.execute('SELECT payload FROM local_imagery WHERE id=?',(item['id'],)).fetchone()
                if previous and item.get('date_basis','').startswith('Catalog import time'):
                    old=json.loads(previous['payload']);item['datetime']=old['datetime'];item['date_basis']=old['date_basis']
                row = conn.execute('SELECT payload FROM imagery_group_reviews WHERE id=?',(item['id'],)).fetchone()
                if row: item.update(json.loads(row['payload']))
                conn.execute('INSERT INTO local_imagery VALUES(?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload',(item['id'],json.dumps(item)))
        if db is not None: save(db)
        else:
            with self.database() as conn: save(conn)

    def scan(self, ident):
        if not self.lock.acquire(blocking=False): return
        generation = None
        try:
            watch = self.watch(ident)
            if not watch['enabled']: return
            generation = watch['generation']
            self.update_watch(ident,generation_guard=generation,status='scanning',last_attempt=now(),error=None)
            base = Path(watch['path'])
            if not base.is_dir(): raise ValueError('Folder unavailable; previous catalog retained')
            with self.database() as db:
                cache = {r['path']:json.loads(r['payload']) for r in db.execute('SELECT path,payload FROM imagery_watch_files WHERE watch_id=?',(ident,))}
            candidates, manifests, errors, states, walk_errors = [],[],[],[],[]
            blocked_groups=set();blocked_folders=[]
            resolved_base=base.resolve()
            next_control_check=0.
            for folder, directories, names in os.walk(base,followlinks=False,onerror=lambda e:walk_errors.append(str(e))):
                directories[:] = [n for n in directories if not n.startswith('.') and not (Path(folder)/n).is_symlink() and not getattr(Path(folder)/n,'is_junction',lambda:False)() and not (Path(folder)/n).resolve().is_relative_to(self.root)] if watch['recursive'] else []
                for name in sorted(names):
                    if self.stop.is_set():return
                    # Bound cancellation latency without opening SQLite for every file.
                    if time.monotonic()>=next_control_check:
                        current=self.watch(ident)
                        if not current['enabled'] or current['generation']!=generation:return
                        next_control_check=time.monotonic()+.25
                    if name.startswith('.') or (Path(name).suffix.lower() not in SUFFIXES and name!='spectrascope-image.json' and not is_metadata(name)):continue
                    p = Path(folder)/name
                    if p.is_symlink() or not p.resolve().is_relative_to(resolved_base): continue
                    if is_metadata(name):
                        try:
                            stat=p.stat();states.append([str(p.relative_to(base)),stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns])
                            if stat.st_size>MAX_METADATA_BYTES:raise ValueError('Supplier metadata exceeds 2 MB')
                            if time.time()-stat.st_mtime<watch['settle_seconds']:raise ValueError('Supplier metadata is still being copied')
                            states[-1].append(hashlib.sha256(p.read_bytes()).hexdigest())
                        except Exception as exc:
                            errors.append({'path':str(p),'error':str(exc)});blocked_folders.append(p.parent)
                        continue
                    if name=='spectrascope-image.json':
                        try:
                            stat=p.stat();states.append([str(p.relative_to(base)),stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns])
                            if stat.st_size>2*1024*1024: raise ValueError('Band manifest exceeds 2 MB')
                            if time.time()-stat.st_mtime<watch['settle_seconds']: raise ValueError('Manifest is still being copied')
                            content=p.read_bytes();states[-1].append(hashlib.sha256(content).hexdigest())
                            manifests.append((p,json.loads(content.decode('utf-8'))))
                        except Exception as exc:
                            errors.append({'path':str(p),'error':str(exc)})
                            blocked_folders.append(p.parent)
                        continue
                    if p.suffix.lower() not in SUFFIXES: continue
                    rel = p.relative_to(base).as_posix()
                    if standalone_auxiliary(rel):continue
                    try:
                        stat=p.stat();states.append([rel,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,stat.st_ino,stat.st_dev])
                        if time.time()-stat.st_mtime<watch['settle_seconds']: raise ValueError('File is still settling; indexing will retry')
                        saved=cache.get(rel)
                        if watch.get('force_full_verify') or saved and time.time()-saved.get('verified_at',0)>300:saved=None
                        item=inspect(p,rel,saved)
                        states[-1].append(item['sha256'])
                        candidates.append(item)
                    except Exception as exc:
                        errors.append({'path':str(p),'error':str(exc)})
                        # Hold the whole affected acquisition, rather than
                        # publishing its other bands as a shorter product.
                        failed=cache.get(rel) or {'relative_path':rel,'count':1,'tags':{},'descriptions':[None]}
                        blocked_groups.add(inferred_key(failed)[0])
                if not watch['recursive']: break
            if walk_errors: raise ValueError('Incomplete directory scan; previous catalog retained: '+'; '.join(walk_errors[:3]))
            signature=hashlib.sha256(json.dumps(sorted(states),sort_keys=True).encode()).hexdigest()
            if signature!=watch.get('observed_signature'):
                current=self.watch(ident)
                if not current['enabled'] or current['generation']!=generation:return
                self.update_watch(ident,generation_guard=generation,status='settling' if states else 'empty',observed_signature=signature,issues=errors,last_scan=now(),files_seen=len(states),force_full_verify=False)
                with self.database() as db:
                    for f in candidates:
                        db.execute('INSERT INTO imagery_watch_files VALUES(?,?,?) ON CONFLICT(watch_id,path) DO UPDATE SET payload=excluded.payload',(ident,f['relative_path'],json.dumps(f)))
                return
            candidates=[f for f in candidates if inferred_key(f)[0] not in blocked_groups and not any(Path(f['path']).is_relative_to(q) for q in blocked_folders)]
            unchanged=signature==watch.get('published_signature')
            if not unchanged:
                items, grouping_issues=products(candidates,base,manifests)
                grouping_issues=errors+grouping_issues
                for item in items:item['id']='local-'+hashlib.sha256((ident+item['sha256']).encode()).hexdigest()[:24]
            with self.database() as db:
                db.execute('BEGIN IMMEDIATE')
                row=db.execute('SELECT payload FROM imagery_watches WHERE id=?',(ident,)).fetchone()
                if not row:return
                current=json.loads(row['payload'])
                if not current['enabled'] or current['generation']!=generation:return
                if unchanged:
                    # Refresh verification times after periodic full hash checks, but
                    # leave product rows and human grouping decisions untouched.
                    for f in candidates:
                        if f!=cache.get(f['relative_path']):
                            db.execute('UPDATE imagery_watch_files SET payload=? WHERE watch_id=? AND path=?',(json.dumps(f),ident,f['relative_path']))
                    current.update(status=('ready-with-issues' if current.get('published_issues') else 'ready') if current.get('products') else ('attention' if current.get('published_issues') else 'empty'),error=None,issues=current.get('published_issues',[]),last_scan=now(),force_full_verify=False)
                    db.execute('UPDATE imagery_watches SET payload=? WHERE id=?',(json.dumps(current),ident))
                    return
                for row in db.execute('SELECT id,payload FROM local_imagery').fetchall():
                    item=json.loads(row['payload'])
                    if item.get('watch_id')==ident:
                        failed_paths={str(Path(e['path']).resolve()) for e in errors}
                        if any(str(Path(s.get('path','')).resolve()) in failed_paths or inferred_key(s)[0] in blocked_groups or any(Path(s.get('path','')).is_relative_to(q) for q in blocked_folders) for s in item.get('sources',[])):
                            item.update(status='needs-review',monitor_error='A source band is unreadable or still being copied; previous catalog metadata retained.')
                            db.execute('UPDATE local_imagery SET payload=? WHERE id=?',(json.dumps(item),row['id']))
                            continue
                        item.update(active=False,inactive_reason='Replaced or removed in monitored folder; job snapshots retained')
                        db.execute('UPDATE local_imagery SET payload=? WHERE id=?',(json.dumps(item),row['id']))
                self.save_products(items,ident,db)
                for f in candidates:
                    db.execute('INSERT INTO imagery_watch_files VALUES(?,?,?) ON CONFLICT(watch_id,path) DO UPDATE SET payload=excluded.payload',(ident,f['relative_path'],json.dumps(f)))
                active=[json.loads(r['payload']) for r in db.execute('SELECT payload FROM local_imagery').fetchall()]
                active=[i for i in active if i.get('active',True) and i.get('watch_id')==ident]
                current.update(status=('ready-with-issues' if grouping_issues else 'ready') if active else ('attention' if grouping_issues else 'empty'),error=None,issues=grouping_issues,published_issues=grouping_issues,last_scan=now(),files_seen=len(states),products=len(active),review_count=sum(i['status']!='ready' for i in active),observed_signature=signature,published_signature=signature,force_full_verify=False)
                db.execute('UPDATE imagery_watches SET payload=? WHERE id=?',(json.dumps(current),ident))
        except Exception as exc:
            current=self.watch(ident)
            if current['enabled'] and current['generation']==generation:
                self.update_watch(ident,generation_guard=generation,status='error',error=str(exc),last_attempt=now())
        finally:self.lock.release()

    async def loop(self):
        while not self.stop.is_set():
            try:
                for watch in await asyncio.to_thread(self.watches):
                    if not watch['enabled']:continue
                    last=watch.get('last_attempt')
                    elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(last)).total_seconds() if last else float('inf')
                    interval=2 if watch.get('status')=='settling' else watch['interval_seconds']
                    if elapsed>=interval:await asyncio.to_thread(self.scan,watch['id'])
            except Exception:logging.getLogger(__name__).exception('Folder monitor will retry after an index error')
            await asyncio.sleep(2)

    def snapshot(self,image):
        sources=image.get('sources',[])
        # A verified repair also applies to future jobs selecting the exact
        # same catalog product. Match every band hash and identity; never
        # transfer calibration to changed data or overwrite explicit settings.
        if sources and all(s.get(key) is None for s in sources
                           for key in ('radiometry','scale_override','offset_override','stac_channel_calibration')):
            with self.database() as db:
                table=db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='raster_access'").fetchone()
                previous=db.execute('SELECT payload FROM raster_access WHERE image_id=? ORDER BY created_at DESC,rowid DESC',(image['id'],)).fetchall() if table else []
            identity=lambda s:(s.get('sha256'),tuple(s.get('band_names',[])))
            wanted={identity(s) for s in sources}
            for row in previous:
                known=json.loads(row['payload']).get('local_sources',[])
                if len(known)!=len(sources) or not all(s.get('radiometry') for s in known) or {identity(s) for s in known}!=wanted:continue
                lookup={identity(s):s for s in known}
                sources=[{**s,**{key:lookup[identity(s)].get(key) for key in ('scale_override','offset_override','radiometry')}} for s in sources]
                break
        if not sources and image.get('local_relative_path'):
            legacy=(self.root/image['local_relative_path']).resolve()
            if not legacy.is_relative_to(self.root) or not legacy.is_file():raise ValueError('The legacy local raster is missing or outside managed storage')
            return [{'key':'stack','path':str(legacy),'band_names':image['band_names'],'sha256':image.get('sha256')}]
        frozen=[]
        with self.snapshot_lock:
            for source in sources:
                original=Path(source['path'])
                target=self.root/'snapshots'/source['sha256'][:2]/(source['sha256']+original.suffix.lower())
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():
                    temp=target.with_name(target.name+'.'+uuid4().hex+'.part')
                    try:
                        from offline_policy import local_raster
                        local_raster(str(original))
                        shutil.copyfile(original,temp)
                        if checksum(temp)!=source['sha256']:raise ValueError('A source band changed since cataloging. Select its updated revision: '+original.name)
                        temp.replace(target)
                    finally:temp.unlink(missing_ok=True)
                if checksum(target)!=source['sha256']:raise ValueError('A saved band snapshot is damaged; original outputs were preserved')
                radiometry=source.get('radiometry')
                if radiometry:
                    metadata=Path(radiometry['metadata_path']);metadata_target=self.root/'snapshots'/'metadata'/(radiometry['metadata_sha256']+metadata.suffix.lower())
                    metadata_target.parent.mkdir(parents=True,exist_ok=True)
                    if not metadata_target.exists():
                        content=metadata.read_bytes()
                        if hashlib.sha256(content).hexdigest()!=radiometry['metadata_sha256']:raise ValueError('Supplier calibration metadata changed; re-index the product')
                        temporary=metadata_target.with_suffix('.'+uuid4().hex+'.part');temporary.write_bytes(content);temporary.replace(metadata_target)
                    if checksum(metadata_target)!=radiometry['metadata_sha256']:raise ValueError('Saved calibration metadata is damaged')
                    radiometry={**radiometry,'metadata_path':str(metadata_target)}
                frozen.append({**source,'radiometry':radiometry,'path':str(target),'original_path':source['path']})
        return frozen


def register(app,database,root):
    catalog=Catalog(database,root)
    app.state.imagery_catalog=catalog
    router=APIRouter(prefix='/api/local-imagery',tags=['Local imagery'])

    async def import_batch(files,relative_paths):
        folder=catalog.root/'imports'/uuid4().hex
        folder.mkdir(parents=True)
        indexed,manifests=[],[]
        try:
            paths=json.loads(relative_paths) if relative_paths else [f.filename or 'image.tif' for f in files]
            if not isinstance(paths,list) or len(paths)!=len(files):raise ValueError('Upload paths must match the file count')
            used=set()
            for file,name in zip(files,paths):
                if not isinstance(name,str) or '\\' in name or ':' in name:raise ValueError('Upload names must be relative folder paths')
                rel=Path(name)
                if rel.is_absolute() or '..' in rel.parts or name in used:raise ValueError('Upload paths must be unique and remain inside the selected folder')
                used.add(name)
                if any(part.startswith('.') for part in rel.parts):continue
                if rel.suffix.lower() not in SUFFIXES and rel.name!='spectrascope-image.json' and not is_metadata(rel.name):continue
                if rel.suffix.lower() in SUFFIXES and standalone_auxiliary(rel.as_posix()):continue
                target=folder/rel;target.parent.mkdir(parents=True,exist_ok=True);size=0
                with target.open('wb') as output:
                    while chunk:=await file.read(1024*1024):
                        size+=len(chunk)
                        if size>(MAX_METADATA_BYTES if rel.name=='spectrascope-image.json' or is_metadata(rel.name) else 5*1024**3):raise ValueError('Upload exceeds 5 GB per raster or 2 MB per manifest; use folder monitoring for larger local files')
                        output.write(chunk)
                if rel.name=='spectrascope-image.json':manifests.append((target,json.loads(target.read_text(encoding='utf-8'))))
                elif not is_metadata(rel.name):indexed.append(await run_in_threadpool(inspect,target,rel.as_posix()))
            if not indexed:raise ValueError('Select GeoTIFF/JP2 rasters or a folder containing them')
            items,issues=await run_in_threadpool(products,indexed,folder,manifests)
            await run_in_threadpool(catalog.save_products,items)
            return {'items':items,'issues':issues,'image':items[0] if len(items)==1 else None}
        except Exception as exc:
            shutil.rmtree(folder,ignore_errors=True)
            raise HTTPException(422,str(exc)) from exc
        finally:
            for file in files:await file.close()

    @router.post('')
    async def upload(file:UploadFile=File(...)):
        return await import_batch([file],None)

    @router.post('/batch')
    async def batch(files:list[UploadFile]=File(...),relative_paths:str=Form('')):
        return await import_batch(files,relative_paths)

    @router.get('')
    def list_images():return {'items':catalog.images()}

    @router.post('/hls-bundles')
    def create_hls_bundle(request:HLSBundleRequest):
        from imagery_products import hls_bundle
        by_id={image['id']:image for image in catalog.images()}
        try:
            if len(set(request.image_ids))!=3:raise ValueError('Choose three different acquisitions')
            image=hls_bundle([by_id[ident] for ident in request.image_ids])
            catalog.save_products([image])
        except (ValueError,KeyError,OSError) as exc:
            raise HTTPException(422,'A selected acquisition is unavailable' if isinstance(exc,KeyError) else str(exc)) from exc
        return {'image':image}

    @router.get('/watches')
    def watches():return {'items':catalog.watches(),'allowed_roots':json.loads(os.getenv('LOCAL_WATCH_ROOTS','[]')),'platform':os.name,'path_scope':'Use a folder path visible to this local service. In Docker, use the path of a mounted folder.'}

    @router.post('/watches')
    def add(request:WatchRequest):
        try:return {'watch':catalog.add_watch(request)}
        except Exception as exc:raise HTTPException(422,str(exc)) from exc

    @router.patch('/watches/{ident}')
    def toggle(ident:str,request:WatchToggle):
        try:
            catalog.watch(ident)
            return {'watch':catalog.update_watch(ident,enabled=request.enabled,generation=uuid4().hex,status='waiting' if request.enabled else 'paused',last_attempt=None,force_full_verify=True)}
        except Exception as exc:raise HTTPException(422,str(exc)) from exc

    @router.post('/watches/{ident}/scan')
    async def scan(ident:str):
        try:
            if not catalog.watch(ident)['enabled']:raise ValueError('Check Monitor this folder before scanning')
            await run_in_threadpool(catalog.scan,ident)
            return {'watch':catalog.watch(ident)}
        except Exception as exc:raise HTTPException(422,str(exc)) from exc

    @router.delete('/watches/{ident}')
    def remove(ident:str):
        catalog.remove_watch(ident)
        return {'ok':True,'message':'Monitoring entry removed; original files and existing job snapshots were preserved'}

    @router.post('/groups/{ident}/review')
    def review(ident:str,request:GroupReview):
        if not request.acknowledge_warnings:raise HTTPException(422,'Confirm explicitly that these bands form one acquisition')
        with database() as db:
            row=db.execute('SELECT payload FROM local_imagery WHERE id=?',(ident,)).fetchone()
            if not row:raise HTTPException(404,'Image group was not found')
            image=json.loads(row['payload'])
            if not image.get('active',True):raise HTTPException(409,'This product was replaced or removed. Refresh the library and review its current revision.')
            choices={s['relative_path']:s for s in image.get('sources',[])+image.get('alternate_sources',[])}
            selected,names=[],[]
            for entry in request.sources:
                source=choices.get(entry.get('relative_path'));channels=entry.get('band_names')
                if not source or not isinstance(channels,list) or len(channels)!=source['count'] or not all(isinstance(n,str) and n.strip() for n in channels):raise HTTPException(422,'Select indexed files and name every channel')
                selected.append({**source,'key':source.get('key') or source['relative_path'],'band_names':[n.strip() for n in channels]});names.extend(n.strip() for n in channels)
            temporal=temporal_hls(image.get('processing_level'),names,image.get('temporal_dates'))
            if len({s['relative_path'] for s in selected})!=len(selected) or (len(set(n.upper() for n in names))!=len(names) and not temporal):raise HTTPException(422,'Channel names must be unique, except for a declared 18-band HLS product with three actual temporal_dates')
            from raster_pipeline import canonical_band
            sensor=image.get('sensor','')
            if request.sensor and sensor and request.sensor!=sensor:raise HTTPException(422,'This source already has a declared sensor')
            sensor=request.sensor or sensor
            canonical=[canonical_band(n,sensor) for n in names]
            if len(set(canonical))!=len(canonical) and not temporal:raise HTTPException(422,'Channel names become duplicates under the selected sensor mapping; assign unambiguous names')
            if max(s['bbox'][0] for s in selected)>=min(s['bbox'][2] for s in selected) or max(s['bbox'][1] for s in selected)>=min(s['bbox'][3] for s in selected):raise HTTPException(422,'The selected bands do not overlap spatially')
            decision={'sources':selected,'band_names':names,'raster_bands':len(names),'status':'ready','reviewed_at':now(),'grouping_approved':True}
            if sensor:decision.update(sensor=sensor,platform=sensor)
            decision['review_revision']=hashlib.sha256(json.dumps([sensor,[{'sha256':s['sha256'],'band_names':s['band_names']} for s in selected]],sort_keys=True).encode()).hexdigest()
            w=min(s['bbox'][0] for s in selected);s=min(s['bbox'][1] for s in selected);e=max(s['bbox'][2] for s in selected);n=max(s['bbox'][3] for s in selected)
            decision.update(bbox=[w,s,e,n],geometry={'type':'Polygon','coordinates':[[[w,s],[e,s],[e,n],[w,n],[w,s]]]})
            db.execute('INSERT INTO imagery_group_reviews VALUES(?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload',(ident,json.dumps(decision)))
            image.update(decision)
            db.execute('UPDATE local_imagery SET payload=? WHERE id=?',(json.dumps(image),ident))
        return {'image':image}

    def reset_monitor():
        for watch in catalog.watches():
            if watch['enabled']:catalog.update_watch(watch['id'],generation_guard=watch['generation'],force_full_verify=True,last_attempt=None)

    @app.on_event('startup')
    async def start_monitor():
        from mac_web_gate import pending, wait_for_commit
        catalog.stop.clear()
        if not pending():
            reset_monitor()
            catalog.task=asyncio.create_task(catalog.loop())
            return
        async def deferred():
            await wait_for_commit()
            reset_monitor()
            await catalog.loop()
        catalog.task=asyncio.create_task(deferred())

    @app.on_event('shutdown')
    async def stop_monitor():
        from mac_web_gate import pending
        catalog.stop.set()
        if catalog.task:
            if pending():catalog.task.cancel()
            await asyncio.gather(catalog.task,return_exceptions=True)

    app.include_router(router)

    def search(geometry,start,end):
        matches=[]
        for image in catalog.images():
            if image.get('status','ready')!='ready' or not start.isoformat()<=image['datetime'][:10]<=end.isoformat():continue
            w,s,e,n=image['bbox']
            if geometry['type']=='Point':
                x,y=geometry['coordinates'];intersects=w<=x<=e and s<=y<=n
            else:intersects=bool(rasterize([(geometry,1)],out_shape=(64,64),transform=from_bounds(w,s,e,n,64,64),all_touched=True).any())
            if intersects:matches.append(image)
        return matches

    def attach(db,missions,at):
        pending=[]
        for mission in missions:
            for job in mission.get('jobs',[]):
                for image in job.get('images',[]):
                    if image.get('catalog')!='local':continue
                    if db.execute('SELECT 1 FROM raster_access WHERE mission_id=? AND job_id=? AND image_id=?',(mission['id'],job['id'],image['id'])).fetchone():continue
                    row=db.execute('SELECT payload FROM local_imagery WHERE id=?',(image['id'],)).fetchone()
                    if not row:raise ValueError('This local image is no longer in the catalog')
                    original=json.loads(row['payload'])
                    if original.get('status','ready')!='ready':raise ValueError('Review the band grouping before adding this image to a job')
                    frozen=catalog.snapshot(original)
                    record={'id':uuid4().hex,'mission_id':mission['id'],'job_id':job['id'],'image_id':image['id'],'mode':'local_path','local_path':frozen[0]['path'],'local_sources':frozen,'status':'local-ready','created_at':at,'band_names':original['band_names'],'sensor':original.get('sensor',''),'processing_level':original.get('processing_level','unknown'),'temporal_dates':original.get('temporal_dates',[]),'source_revision':original.get('review_revision') or original.get('sha256'),'grouping_reviewed_at':original.get('reviewed_at'),'message':'All bands preserved in a revisioned local job snapshot.'}
                    pending.append((record['id'],mission['id'],job['id'],image['id'],json.dumps(record),at))
        if pending:
            if not db.in_transaction:db.execute('BEGIN IMMEDIATE')
            pending=[r for r in pending if not db.execute('SELECT 1 FROM raster_access WHERE mission_id=? AND job_id=? AND image_id=?',(r[1],r[2],r[3])).fetchone()]
            db.executemany('INSERT INTO raster_access VALUES(?,?,?,?,?,?)',pending)
    search.catalog=catalog
    return search,attach
