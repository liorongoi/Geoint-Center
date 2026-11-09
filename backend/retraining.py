"""Persisted, event-driven training decisions from human correction evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom

from algorithm_registry import MODEL_DETAILS, algorithm_training_rule

DEFAULT_POLICY = {
    'version':1, 'enabled':True, 'minimum_regions':20, 'minimum_corrected_pixels':500,
    'minimum_images':2, 'minimum_confidence':3, 'notification':'training_started_only',
    'applies_to':'All registered current and future trainable algorithms',
    'promotion':'Save a separate candidate version; preserve original run outputs',
}


def initialize(conn):
    conn.execute("INSERT OR IGNORE INTO settings(key,payload) VALUES('retraining_policy',?)", (json.dumps(DEFAULT_POLICY),))
    conn.execute('''CREATE TABLE IF NOT EXISTS auto_training_batches (
        id TEXT PRIMARY KEY, mission_id TEXT NOT NULL, job_id TEXT NOT NULL,
        model_id TEXT NOT NULL, evidence_key TEXT NOT NULL, payload TEXT NOT NULL,
        UNIQUE(mission_id,job_id,model_id,evidence_key))''')


def get_policy(conn):
    row = conn.execute("SELECT payload FROM settings WHERE key='retraining_policy'").fetchone()
    return {**DEFAULT_POLICY, **(json.loads(row['payload']) if row else {})}


def evaluate(database, data_dir: Path, mission_id: str, job_id: str) -> dict:
    with database() as conn:
        policy = get_policy(conn)
        reviews = []
        # mission/job are in the auditable payload of the original review table.
        for row in conn.execute('SELECT payload FROM manual_reviews ORDER BY created_at,rowid'):
            item=json.loads(row['payload'])
            if item['mission_id']==mission_id and item['job_id']==job_id:
                reviews.append(item)
        labels = [json.loads(row['payload']) for row in conn.execute(
            'SELECT payload FROM ground_truth WHERE mission_id=? AND job_id=? ORDER BY created_at,rowid', (mission_id,job_id))]
        recorded={item['id'] for item in reviews}
        reviews.extend(label for label in labels if label.get('origin')=='human_reference' and label['id'] not in recorded)
        reviews.sort(key=lambda item:item['created_at'])
        runs = [json.loads(row['payload']) for row in conn.execute(
            'SELECT payload FROM inference_runs WHERE mission_id=? AND job_id=? ORDER BY created_at DESC,rowid DESC', (mission_id,job_id))]
        batches = [json.loads(row['payload']) for row in conn.execute(
            'SELECT payload FROM auto_training_batches WHERE mission_id=? AND job_id=?',(mission_id,job_id))]
        workspace=conn.execute('SELECT payload FROM workspace WHERE id=1').fetchone()
        missions=json.loads(workspace['payload']).get('missions',[]) if workspace else []
        mission=next((m for m in missions if m['id']==mission_id),{})
        job=next((j for j in mission.get('jobs',[]) if j['id']==job_id),{})
        access={}
        for row in conn.execute('SELECT payload FROM raster_access WHERE mission_id=? AND job_id=? ORDER BY created_at DESC,rowid DESC',(mission_id,job_id)):
            record=json.loads(row['payload']);access.setdefault(record['image_id'],record)
    by_id={run['id']:run for run in runs}
    training_classes=sorted({item['label'] for item in labels if item.get('role','train')=='train'})
    all_images={image['id'] for image in job.get('images',[])}
    raster_ready=bool(all_images) and all(access.get(image,{}).get('status') in ('stream-ready','local-ready','downloaded') for image in all_images)
    result=[]
    for model_id,spec in MODEL_DETAILS.items():
        rule=algorithm_training_rule(model_id)
        consumed=set()
        for run in runs:
            if run.get('run_kind')=='training' and run.get('status') in ('complete','partial') and any(m['id']==model_id and m.get('status')=='complete' for m in run.get('models',[])):
                consumed.update(run.get('label_ids',[]))
        for batch in batches:
            run=by_id.get(batch['run_id'],{})
            if batch['model_id']==model_id and run.get('status') in ('complete','partial') and any(m['id']==model_id and m.get('status')=='complete' for m in run.get('models',[])):
                consumed.update(batch['review_ids'])
        active=next((run for run in runs if run.get('run_kind')=='training' and run.get('status') in ('queued','running','pausing','paused','stopping') and model_id in run.get('requested_models',[])),None)
        latest={}
        for run in runs:
            if run.get('status') not in ('complete','partial') or not any(m['id']==model_id and m.get('status')=='complete' for m in run.get('models',[])):
                continue
            for index,output in enumerate(run.get('outputs',[])):
                if output['image_id'] in all_images:
                    latest.setdefault(output['image_id'],(run,index))
        regions=set();images=set();pixels=0;class_counts={};training_counts={name:0 for name in training_classes};baselines={}
        for image_id,(run,index) in latest.items():
            path=(data_dir/'results'/run['id']/f'{index}-{model_id}.tif').resolve()
            if not path.is_relative_to((data_dir/'results').resolve()) or not path.is_file():
                continue
            applicable=[review for review in reviews if not review.get('image_id') or review['image_id']==image_id]
            with rasterio.open(path) as source:
                predicted=source.read(1);valid=(predicted>0)&(source.read_masks(1)>0)
                profile=source.profile
            def burn(shapes,fill=0):
                return rasterize(shapes,out_shape=predicted.shape,transform=profile['transform'],fill=fill,dtype='int32') if shapes else np.full(predicted.shape,fill,dtype='int32')
            def projected(item):
                return transform_geom('EPSG:4326',profile['crs'],item['geometry'])
            relevant_labels=[label for label in labels if label.get('geometry') and (not label.get('image_id') or label['image_id']==image_id)]
            heldout=burn([(projected(item),1) for item in relevant_labels if item.get('role')=='validation'])>0
            if any(image.get('id')==image_id and image.get('benchmark_split') in ('validation','test') for image in job.get('images',[])):
                heldout[:]=True
            if not any(label.get('role')=='validation' for label in labels):
                heldout |= np.indices(predicted.shape)[1]>=int(predicted.shape[1]*.65)
            train=burn([(projected(item),training_classes.index(item['label'])) for item in relevant_labels if item.get('role','train')=='train'],fill=-1)
            for i,name in enumerate(training_classes):
                training_counts[name]+=int(((train==i)&valid&~heldout).sum())
            if not applicable:
                continue
            # Last decision owns a pixel. Repeating or overlapping a shape cannot multiply evidence.
            owner=burn([(projected(item),i+1) for i,item in enumerate(applicable)])
            baselines[image_id]=run['id']
            for number in np.unique(owner):
                if number==0:
                    continue
                review=applicable[int(number)-1]
                if review['id'] in consumed or review.get('role','train')!='train' or review.get('confidence',3)<policy['minimum_confidence']:
                    continue
                comparison=np.zeros_like(valid)
                for label_value,name in enumerate(run.get('classes',[]),1):
                    if name!=review['label']:
                        comparison|=predicted==label_value
                mask=(owner==number)&valid&~heldout&comparison
                count=int(mask.sum())
                if count:
                    regions.add(review['id']);images.add(image_id);pixels+=count
                    class_counts[review['label']]=class_counts.get(review['label'],0)+count
        evidence_key=hashlib.sha256(json.dumps({'reviews':sorted(regions),'baselines':baselines,'version':policy['version']},sort_keys=True).encode()).hexdigest()
        prior=next((batch for batch in batches if batch['model_id']==model_id and batch['evidence_key']==evidence_key),None)
        reasons=[]
        if not policy['enabled'] or not rule['enabled']:reasons.append('Automatic training disabled')
        if len(regions)<policy['minimum_regions']:reasons.append(f'{len(regions)}/{policy["minimum_regions"]} distinct reviewed regions')
        if pixels<policy['minimum_corrected_pixels']:reasons.append(f'{pixels}/{policy["minimum_corrected_pixels"]} unique corrected analysis pixels')
        if len(images)<policy['minimum_images']:reasons.append(f'{len(images)}/{policy["minimum_images"]} images')
        if len(training_classes)<2 or any(count<rule['minimum_training_pixels_per_class'] for count in training_counts.values()):reasons.append('Need at least two training classes with sufficient valid pixels')
        if not raster_ready:reasons.append('Configure readable raster access for all job images')
        if active:reasons.append('A separate training run for this algorithm is already active')
        if prior:reasons.append('This evidence batch already has a training attempt; no duplicate run')
        result.append({'model_id':model_id,'name':spec['name'],'training_mode':rule['mode'],
            'regions':len(regions),'corrected_pixels':pixels,'images':len(images),
            'class_counts':class_counts,'training_class_counts':training_counts,
            'review_ids':sorted(regions),'label_ids':[label['id'] for label in labels],
            'evidence_key':evidence_key,'baselines':baselines,'ready':not reasons,'waiting_for':reasons,
            'active_run_id':active['id'] if active else None})
    return {'policy':policy,'mission_id':mission_id,'job_id':job_id,'algorithms':result}
