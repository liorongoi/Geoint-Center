"""Durable cooperative controls. Never kill a thread while it writes scientific data."""
import json
import os
import time
from datetime import datetime, timezone


class MissionStopped(BaseException):
    """Bypass model-error handlers; the managed worker records cancellation."""


ACTIVE = {'queued', 'running', 'pausing', 'paused', 'blocked', 'stopping'}
PROGRESS_OBSERVER = None
TABLES = ('inference_runs', 'anomaly_runs', 'segmentation_runs')


def initialize(db):
    db.execute('CREATE TABLE IF NOT EXISTS mission_controls (mission_id TEXT PRIMARY KEY, action TEXT NOT NULL, updated_at TEXT NOT NULL)')


def action(database, mission_id):
    with database() as db:
        row = db.execute('SELECT action FROM mission_controls WHERE mission_id=?', (mission_id,)).fetchone()
    return row['action'] if row else 'resume'


def allowed(database, mission_id):
    return action(database, mission_id) == 'resume'


def request(database, mission_id, command):
    if command not in ('pause', 'resume', 'stop', 'delete'):
        raise ValueError('Choose pause, resume, stop, or delete')
    with database() as db:
        db.execute('BEGIN IMMEDIATE')
        workspace = db.execute('SELECT payload FROM workspace WHERE id=1').fetchone()
        if not workspace or not any(m['id'] == mission_id for m in json.loads(workspace['payload']).get('missions', [])):
            raise LookupError('Mission not found')
        old = db.execute('SELECT action FROM mission_controls WHERE mission_id=?', (mission_id,)).fetchone()
        if old and old['action'] == 'delete':
            raise ValueError('This mission has been deleted from the workspace')
        db.execute('INSERT INTO mission_controls VALUES(?,?,?) ON CONFLICT(mission_id) DO UPDATE SET action=excluded.action,updated_at=excluded.updated_at',
                   (mission_id, command, datetime.now(timezone.utc).isoformat()))
        changed = 0
        for table in TABLES:
            for row in db.execute(f'SELECT id,payload FROM {table}').fetchall():
                run = json.loads(row['payload'])
                if run.get('mission_id') != mission_id or run.get('status') not in ACTIVE:
                    continue
                if command in ('stop', 'delete'):
                    run.update(control_stop=True, status='stopping', stage='Stop requested; waiting for a safe processing checkpoint')
                elif command == 'pause' and not run.get('control_stop'):
                    run.update(status='pausing', stage='Pause requested; waiting for a safe processing checkpoint')
                db.execute(f'UPDATE {table} SET payload=? WHERE id=?', (json.dumps(run), row['id']))
                changed += 1
    return {'action': command, 'affected_runs': changed, 'data_preserved': True}


def controlled(database, table):
    """Gate each progress checkpoint and prevent stale worker writes undoing Stop."""
    def checkpoint(run):
        original_status, original_stage = run.get('status'), run.get('stage')
        while True:
            # Read and write under one lock so a control request cannot be lost.
            with database() as db:
                db.execute('BEGIN IMMEDIATE')
                control = db.execute('SELECT action FROM mission_controls WHERE mission_id=?', (run['mission_id'],)).fetchone()
                saved = db.execute(f'SELECT payload FROM {table} WHERE id=?', (run['id'],)).fetchone()
                state = control['action'] if control else 'resume'
                stored=json.loads(saved['payload']) if saved else {}
                if 'control_action' in stored:run['control_action']=stored['control_action']
                if stored.get('archive_after_stop'):run['archive_after_stop']=True
                if state=='resume':state=stored.get('control_action','resume')
                blocked=False
                if run.get('qualified_releases'):
                    for identifier,release_id in run['qualified_releases'].items():
                        profile=db.execute('SELECT payload FROM workflow_models WHERE id=?',(identifier,)).fetchone()
                        profile=json.loads(profile['payload']) if profile else {}
                        if profile.get('state')!='qualified':blocked=True
                        elif profile.get('release',{}).get('id')!=release_id:
                            run['stop_reason']='Qualified model version changed; start a new run with its new release'
                            raise MissionStopped()
                if blocked and state=='resume':state='model_blocked'
                if state in ('stop', 'delete') or (saved and json.loads(saved['payload']).get('control_stop')):
                    raise MissionStopped()
                if state in ('pause','model_blocked'):
                    run.update(status='blocked' if state=='model_blocked' else 'paused',stage='Waiting for configured model qualification' if state=='model_blocked' else 'Paused at a safe processing checkpoint')
                else:
                    run.update(status=original_status, stage=original_stage)
                db.execute(f'UPDATE {table} SET payload=? WHERE id=?', (json.dumps(run, default=str), run['id']))
            if PROGRESS_OBSERVER:PROGRESS_OBSERVER(run)
            if state not in ('pause','model_blocked'):
                # The table write above is authoritative. Side effects of the
                # ordinary persister are performed by the managed worker below.
                return
            time.sleep(.25)
    return checkpoint


def execute(database, table, run, worker, persist, *args, **kwargs):
    from resource_admission import ResourceUnavailable, accelerator, analysis_gib, reserve
    from portable_lulc import MODEL as PORTABLE
    checkpoint = controlled(database, table)
    try:
        checkpoint(run)
        if worker.__module__ == 'native_gpu_bridge':
            # The execution host owns this reservation, not the coordinator.
            result = worker(run, *args, persist=checkpoint, **kwargs)
        else:
            if os.getenv('GEO_RESOURCE_ROOT'):
                run.update(stage='Waiting for shared host resources')
            uses_gpu=run.get('requested_models')!=[PORTABLE] and accelerator()
            with reserve('Local analysis', analysis_gib(), gpu=uses_gpu, checkpoint=lambda:checkpoint(run)):
                result = worker(run, *args, persist=checkpoint, **kwargs)
    except MissionStopped:
        run.update(status='cancelled', control_stop=True, stage=run.get('stop_reason','Stopped; completed artifacts preserved'))
        if run.get('archive_after_stop'):run['archived']=True
        for model in run.get('models', []):
            if model.get('status') in ACTIVE:
                model.update(status='cancelled')
        result = run
    except ResourceUnavailable as exc:
        run.update(status='failed',stage='Shared host capacity unavailable',error=str(exc))
        result = run
    # Sync secondary training records and completion hooks only after the gate.
    persist(run)
    return result
