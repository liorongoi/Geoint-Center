"""Persisted background jobs connecting raster access, labels, workers and artifacts."""
from __future__ import annotations
import json
import time
import re
from pathlib import Path
from typing import Callable

import numpy as np
import rasterio
from PIL import Image
from rasterio.warp import transform_bounds

from model_workers import consensus_maps, predict_worker, score_predictions, train_worker, ENSEMBLE_MEMBERS
from raster_pipeline import coregister, label_mask, read_stack, write_raster
from algorithm_registry import MODEL_DETAILS
from pretrained_lulc import PRETRAINED, validate_input as validate_pretrained, checkpoint as pretrained_checkpoint
import portable_lulc

MODEL_NAMES={key:value['name'] for key,value in MODEL_DETAILS.items()}
MODEL_NAMES[portable_lulc.MODEL]=portable_lulc.DETAILS['name']
PALETTE=np.array([[78,153,99],[62,133,222],[214,173,80],[168,112,196],[228,105,87],[102,207,192],[153,160,168],[232,204,154],
                 [23,90,61],[202,74,151],[95,70,158],[15,104,153],[180,90,35],[189,218,85],[244,142,180],
                 [119,82,59],[86,200,235],[233,226,89],[222,59,59],[39,72,106]],dtype='uint8')
AGREEMENT_COLOR = [70, 203, 170]
DISAGREEMENT_COLOR = [255, 126, 56]


def prepare_output_review(run: dict, index: int, data_dir: Path) -> dict:
    """Build explicit agreement views from saved predictions, including older real runs."""
    root = (data_dir / 'results').resolve()
    folder = (root / run['id']).resolve()
    if not folder.is_relative_to(root):
        raise ValueError('Invalid result path')
    base = f'/api/results/{run["id"]}/{index}'
    legend = [{'label': name, 'value': i + 1,
               'color': '#%02x%02x%02x' % tuple(PALETTE[i % len(PALETTE)])}
              for i, name in enumerate(run.get('classes', []))]
    result = {'class_legend': legend}
    compared = sum(model.get('status') == 'complete' for model in run.get('models', []))
    cache = folder / f'{index}-review.json'
    if cache.is_file():
        result = json.loads(cache.read_text())
        if 'class_distribution' in result:
            return result
    with rasterio.open(folder / f'{index}-consensus.tif') as source:
        # Label 0 is nodata, not agreement. Also honor the source mask.
        classified = source.read(1)
        valid = (classified > 0) & (source.dataset_mask() > 0)
        profile = source.profile.copy()
    total = int(valid.sum())
    result['class_distribution'] = [
        {**item, 'pixels': int(((classified == item['value']) & valid).sum()),
         'percent': round(100 * int(((classified == item['value']) & valid).sum()) / total, 2) if total else None}
        for item in legend
    ]
    result['classification_summary'] = {
        'valid_pixels': total, 'nodata_pixels': int(valid.size - total),
        'configured_classes': len(legend),
        'predicted_classes': sum(item['pixels'] > 0 for item in result['class_distribution']),
        'source': 'Final classification raster; percentages exclude nodata',
    }
    if compared < 2:
        result['agreement_summary'] = {'models_compared': compared, 'agreement_percent': None}
        cache.write_text(json.dumps(result, indent=2))
        return result
    if result.get('review_preview'):
        # Upgrade existing real reports without rerunning models or recreating overlays.
        cache.write_text(json.dumps(result, indent=2))
        return result
    with rasterio.open(folder / f'{index}-disagreement.tif') as source:
        disagreement = (source.read(1) == 1) & valid
    agreement = valid & ~disagreement
    agree_count = int(agreement.sum())
    result.update(agreement_summary={
        'models_compared': compared, 'valid_pixels': total,
        'agreement_pixels': agree_count, 'disagreement_pixels': int(disagreement.sum()),
        'agreement_percent': round(100 * agree_count / total, 2) if total else None,
        'disagreement_percent': round(100 * int(disagreement.sum()) / total, 2) if total else None,
        'nodata_pixels': int(valid.size - total),
    }, review_preview=f'{base}-review.png', agreement_preview=f'{base}-agreement.png',
       agreement_file=f'{base}-agreement.tif')
    mask = np.full(valid.shape, 255, dtype='uint8')
    mask[valid] = agreement[valid].astype('uint8')
    with rasterio.open(folder / f'{index}-agreement.tif', 'w',
                       **{**profile, 'dtype': 'uint8', 'count': 1, 'nodata': 255}) as target:
        target.write(mask, 1)
        target.update_tags(description='1=all models agree; 0=models disagree; 255=no valid data', models_compared=compared)
    for name, show_disagreement in [('agreement', False), ('review', True)]:
        rgb = np.zeros((*valid.shape, 3), dtype='uint8')
        rgb[agreement] = AGREEMENT_COLOR
        if show_disagreement:
            rgb[disagreement] = DISAGREEMENT_COLOR
        visible = valid if show_disagreement else agreement
        rgba = np.dstack((rgb, visible.astype('uint8') * 220))
        Image.fromarray(rgba).save(folder / f'{index}-{name}.png')
        with rasterio.open(folder / f'{index}-{name}-rgb.tif', 'w',
                           **{**profile, 'count': 3, 'dtype': 'uint8', 'nodata': 0}) as target:
            target.write(np.moveaxis(rgb, -1, 0))
    cache.write_text(json.dumps(result, indent=2))
    return result


def _preview(path, labels, valid):
    rgba=np.zeros((*labels.shape,4),dtype='uint8')
    rgba[:,:,:3]=PALETTE[np.maximum(labels,0)%len(PALETTE)]
    rgba[:,:,3]=valid.astype('uint8')*190
    Image.fromarray(rgba).save(path)


def execute_job(run: dict, images: list[dict], access: dict, labels: list[dict], data_dir: Path,
                persist: Callable, epochs: int=20, max_size: int=384, geometry: dict|None=None, initial_checkpoints: dict|None=None, unet_trainer: Callable|None=None):
    if run.get('run_kind')=='operational_inference':
        from mission_workspace import sha
        for release in run.get('qualification',{}).values():
            for path,expected in release.get('files',{}).items():
                if not Path(path).is_file() or sha(path)!=expected:
                    run.update(status='failed',stage='Qualified model files changed',error='Qualified model integrity check failed');persist(run);return run
    native_dynamic_world = run['requested_models']==['dynamicworld'] and run.get('run_kind')!='training'
    native_prithvi = run['requested_models']==['prithvi_crop'] and run.get('run_kind')=='operational_inference'
    if native_dynamic_world or native_prithvi:
        from native_lulc import execute
        return execute(run,images,access,labels,data_dir,persist,max_size=max_size,geometry=geometry,initial_checkpoints=initial_checkpoints)
    start=time.perf_counter()
    operational=run.get('run_kind')=='operational_inference'
    portable=run['requested_models']==[portable_lulc.MODEL]
    fixed=operational or portable
    if 'ensemble' in run['requested_models'] and not operational:
        run['user_requested_models']=list(run['requested_models'])
        run['requested_models']=list(dict.fromkeys([*ENSEMBLE_MEMBERS,*run['requested_models']]))
    out=data_dir/'results'/run['id'];out.mkdir(parents=True,exist_ok=True)
    run.update(status='running',progress=2,stage='Reading source rasters',models=[],outputs=[])
    persist(run)
    try:
        pretrained=[model for model in run['requested_models'] if model in PRETRAINED]
        if pretrained and (len(pretrained)!=1 or len(run['requested_models'])!=1):
            raise ValueError('Run one native pretrained LULC taxonomy per job run. Use separate runs for models with different class systems; the trained ensemble uses its shared job taxonomy.')
        if pretrained:
            run['user_combination']=run.get('combination');run['combination']='separate'
        if portable:
            validation_images={image['id'] for image in images if image.get('benchmark_split') not in ('train','test')}
            labels=[ref for ref in labels if ref.get('role')=='validation' and ref.get('image_id') in validation_images
                    and ref.get('confidence',3)>=3 and ref.get('origin','human_reference')=='human_reference'
                    and ref.get('benchmark_split') not in ('train','test')]
            run['validation_label_ids']=[ref['id'] for ref in labels]
        has_pseudo=any(label.get('origin')=='reviewed_pseudo_label' for label in labels)
        if any(label.get('origin')=='reviewed_pseudo_label' and label.get('role')!='train' for label in labels):raise ValueError('Teacher-derived pseudo-labels cannot be used for validation or testing')
        if pretrained and run.get('run_kind')=='training' and has_pseudo:
            raise ValueError('Dynamic World starting labels support RF, SVM, XGBoost, generic U-Net, frozen-encoder Prithvi and their ensemble, not incompatible pretrained heads')
        run['training_label_provenance']=[{'id':l['id'],'origin':l.get('origin','human_reference'),'role':l.get('role','train'),'source':l.get('source'),'class':l['label']} for l in labels]
        protected={image['id'] for image in images if image.get('benchmark_split') in ('validation','test')}
        classes=run['qualified_classes'] if fixed else PRETRAINED[pretrained[0]]['classes'] if pretrained else sorted({label['label'] for label in labels if label.get('role','train')=='train' and label.get('confidence',3)>=3 and label.get('benchmark_split') not in ('validation','test') and (not label.get('image_id') or label.get('image_id') in {image['id'] for image in images}) and label.get('image_id') not in protected})
        if len(classes)<2:
            raise ValueError('Add ground-truth polygons or points for at least two land-cover classes in this job')
        stacks=[];training=[];validation=[];testing=[];alignment=[];read_provenance=[]
        for image_index, image in enumerate(images):
            record=access.get(image['id'])
            if not record:
                raise ValueError(f"Configure streaming, download, or a local raster for {image['id']}")
            stack=read_stack(record,image,max_size=max_size,geometry=geometry,
                             max_elements=max(1,portable_lulc.MAX_ELEMENTS//len(images)) if portable else None)
            if pretrained:validate_pretrained(pretrained[0],stack,image,record)
            if run['band_alignment']['enabled']:
                shifts=coregister(stack)
                alignment.append({'image_id':image['id'],'bands':shifts})
            applicable=[label for label in labels if not label.get('image_id') or label['image_id']==image['id']]
            train_labels=[label for label in applicable if label.get('role','train')=='train' and label.get('confidence',3)>=3 and label.get('benchmark_split') not in ('validation','test')]
            val_labels=[label for label in applicable if label.get('role')=='validation' and label.get('benchmark_split')!='test']
            test_labels=[label for label in applicable if label.get('benchmark_split')=='test']
            train_mask=label_mask(stack,train_labels,classes)
            val_mask=label_mask(stack,val_labels,classes)
            test_mask=label_mask(stack,test_labels,classes)
            if image['id'] in protected:train_mask[:]=-1
            all_heldout=[label for label in applicable if label.get('role')=='validation' or label.get('benchmark_split') in ('validation','test')]
            excluded=label_mask(stack,[{**label,'label':'__excluded__'} for label in all_heldout],['__excluded__'])>=0
            train_mask[excluded]=-1
            if portable:
                run['validation_protocol']='Fixed portable weights. Scored only image-specific human validation references. Independence from supplier fitting and tuning was attested by the operator.'
            elif operational:
                run['validation_protocol']='Qualified-model inference without fitting. Only independent validation/test labels are scored.'
            elif has_pseudo:
                run['validation_protocol']='Reviewed Dynamic World pseudo-labels are training targets only. Only separate human validation/test references are scored; no teacher-derived holdout is reported as accuracy.'
            elif not any(label.get('role')=='validation' for label in labels):
                # Reserve the eastern 25% and remove a 10% buffer from training.
                x=np.indices(stack.valid.shape)[1];width=stack.valid.shape[1]
                val_mask=train_mask.copy();val_mask[x<int(width*.75)]=-1
                train_mask[x>=int(width*.65)]=-1
                run['validation_protocol']='Internal spatial holdout: eastern 25%, 10% buffer; no external test set'
            else:
                # Validation always wins if submitted labels overlap.
                train_mask[(val_mask>=0)|(test_mask>=0)]=-1
                run['validation_protocol']='User-designated validation geometries excluded from training'
            stacks.append((image,stack));validation.append(val_mask);testing.append(test_mask)
            if np.any(train_mask>=0):
                training.append((stack,train_mask,excluded | (val_mask>=0) | (test_mask>=0)) if unet_trainer else (stack,train_mask))
            read_provenance.append({'image_id':image['id'],'bands':stack.bands,'analysis_shape':list(stack.pixels.shape),
                                    'valid_pixels':int(stack.valid.sum()),'sources':stack.provenance,
                                    'crs':str(stack.profile['crs']),'transform':list(stack.profile['transform'])})
            run.update(progress=2+int(12*(image_index+1)/len(images)),
                       stage=f'Read image {image_index+1} of {len(images)} · {image["id"]}')
            persist(run)
        if portable and sum(stack.valid.size for _,stack in stacks)*len(classes)>portable_lulc.MAX_ELEMENTS:
            raise ValueError('Portable model output exceeds the analysis budget. Choose a smaller grid, AOI or image batch.')
        if not fixed and not training and (not pretrained or run.get('run_kind')=='training'):
            raise ValueError('No training labels intersect valid source pixels')
        run.update(classes=classes,band_alignment={**run['band_alignment'],'records':alignment},raster_provenance=read_provenance,
                   analysis_grid_limit=max_size,progress=15,stage='Loading fixed pixel classifier' if fixed or pretrained and run.get('run_kind')!='training' else 'Training models on reviewed labels')
        persist(run)
        probabilities=[[] for _ in stacks]
        requested=run['requested_models']
        run['mission_complexity']={'spectral_bands':[len(stack.bands) for _,stack in stacks],
                                   'image_count':len(stacks),'training_labeled_pixels':sum(int((sample[1]>=0).sum()) for sample in training),
                                   'class_count':len(classes),'analysis_grid_limit':max_size,
                                   'temporal_mode':'Independent acquisitions; temporal fusion is not enabled'}
        for mi,model_id in enumerate(requested):
            rationale={
                'random_forest':'Balanced tree baseline across every calibrated band; robust candidate for limited reference data',
                'svm':'Scaled spectral features with an RBF decision boundary; candidate for high-dimensional and small-sample classification',
                'xgboost':'Boosted trees model non-linear interactions between calibrated bands',
                'unet':'Learns local spatial context from label masks and all supplied bands; useful for connected fields and boundaries',
                'prithvi':'Official pretrained Prithvi-EO-2.0-300M spatial encoder; six physically mapped reflectance bands plus a supervised RF head',
                'ensemble':'Derived classifier: equal soft vote of compatible fitted base members; fixed weights use no validation/test labels. Member predictions remain separately inspectable.',
            }
            detail={'id':model_id,'name':MODEL_NAMES[model_id],'status':'running',
                    'why':rationale.get(model_id,(portable_lulc.DETAILS if portable else MODEL_DETAILS[model_id]).get('reason','Registered classification algorithm'))}
            run['models'].append(detail)
            inference_only=fixed or (model_id in PRETRAINED and run.get('run_kind')!='training')
            run.update(stage=f'{"Loading pretrained" if inference_only else "Training"} {MODEL_NAMES[model_id]}',progress=15+int(70*mi/len(requested)));persist(run)
            try:
                checkpoint=out/f'{model_id}.joblib'
                phase = 'training'
                prediction_index = 0
                def progress(stage,percent):
                    match=re.search(r'epoch (\d+).*loss ([\deE+.\-]+)',stage)
                    if match:
                        point={'epoch':int(match[1]),'loss':float(match[2]),'scope':'last labeled training batch'}
                        curve=detail.setdefault('learning_curve',[])
                        if not curve or curve[-1]['epoch']!=point['epoch']:curve.append(point)
                    fraction = .65*percent/100 if phase=='training' else .65+.30*(prediction_index+percent/100)/len(stacks)
                    run['progress'] = max(run['progress'], min(84,15+int(70*(mi+fraction)/len(requested))))
                    detail['progress'] = min(99,int(100*fraction))
                    run['stage']=f'{MODEL_NAMES[model_id]} · {stage} {percent}%';persist(run)
                previous=Path(initial_checkpoints[model_id]) if initial_checkpoints and model_id in initial_checkpoints else None
                if portable:
                    model=run['portable_model']
                    source=Path(run['qualified_checkpoints'][model_id] if operational else model['path'])
                    raw,spec=portable_lulc.read(source,model['sha256'])
                    if spec['classes']!=classes:raise ValueError('Portable model taxonomy differs from the configured class order')
                    checkpoint=source if operational else out/'portable-model.json'
                    if not operational:checkpoint.write_bytes(raw)
                    trained={'path':str(checkpoint),'sha256':model['sha256'],'inventory_id':model['inventory_id'],
                             'training_seconds':0,'device':'cpu','initialization':'Fixed data-only coefficients; no fitting',
                             'score_semantics':'Softmax scores are not calibrated probabilities','implementation':MODEL_NAMES[model_id]}
                elif operational and model_id not in PRETRAINED:
                    checkpoint=Path(run['qualified_checkpoints'][model_id])
                    if not checkpoint.is_file():raise ValueError('Qualified checkpoint is missing')
                    trained={'path':str(checkpoint),'training_seconds':0,'initialization':'Qualified immutable checkpoint; no fitting','implementation':MODEL_NAMES[model_id]}
                elif inference_only:
                    trained=pretrained_checkpoint(model_id,stacks[0][1],checkpoint,data_dir/'models',previous)
                elif model_id=='unet' and unet_trainer:
                    trained=unet_trainer(training,classes,checkpoint,previous)
                else:
                    trained=train_worker(model_id,training,classes,checkpoint,data_dir/'models'/'prithvi',epochs=epochs,progress=progress,
                                     initial_checkpoint=Path(initial_checkpoints[model_id]) if initial_checkpoints and model_id in initial_checkpoints else None,
                                     member_checkpoints={m['id']:m['training']['path'] for m in run['models'] if m.get('status')=='complete' and m['id'] in ENSEMBLE_MEMBERS} if model_id=='ensemble' else None,
                                     **({'forest_spec':run['distributed_forest']} if run.get('distributed_forest') else {}))
                scores=[]
                phase = 'prediction'
                for si,(image,stack) in enumerate(stacks):
                    prediction_index = si
                    run['stage']=f'{MODEL_NAMES[model_id]} · predicting image {si+1} of {len(stacks)}';persist(run)
                    if model_id=='ensemble' and not operational:
                        prob=np.mean(probabilities[si],axis=0).astype('float32')
                        predicted=prob.argmax(0).astype('int32');predicted[~stack.valid]=-1
                        performance={'prediction_seconds':0,'implementation':'Reused fitted member probabilities; equal soft vote'}
                    elif portable:
                        started=time.perf_counter()
                        source_access=access[image['id']]
                        sensor=source_access.get('sensor') or image.get('sensor') or image.get('catalog')
                        level=source_access.get('processing_level') or image.get('processing_level')
                        predicted,prob=portable_lulc.predict(spec,stack,sensor,level,progress)
                        performance={'prediction_seconds':time.perf_counter()-started,'implementation':'Portable linear classifier · CPU'}
                    else:
                        predicted,prob,performance=predict_worker(checkpoint,stack,data_dir/'models'/'prithvi',progress)
                    # Append only after all predictions succeed, preserving per-image model correspondence.
                    scores.append((predicted,prob,performance))
                    progress('prediction',100)
                truths=[];preds=[]
                for si,((image,stack),(predicted,prob,performance)) in enumerate(zip(stacks,scores)):
                    probabilities[si].append(prob)
                    truths.append(validation[si].ravel());preds.append(predicted.ravel())
                    write_raster(out/f'{si}-{model_id}.tif',(predicted+1).astype('uint16'),stack)
                metrics=score_predictions(np.concatenate(truths),np.concatenate(preds),classes)
                detail.update(status='complete',progress=100,training=trained,metrics=metrics,
                              prediction_seconds=round(sum(score[2]['prediction_seconds'] for score in scores),3))
                if any(np.any(mask>=0) for mask in testing):
                    detail['test_metrics']=score_predictions(np.concatenate([mask.ravel() for mask in testing]),np.concatenate(preds),classes)
            except Exception as exc:
                detail.update(status='failed',error=str(exc))
            run['progress']=max(run['progress'],15+int(70*(mi+1)/len(requested)))
            persist(run)
        succeeded=sum(model['status']=='complete' for model in run['models'])
        if not succeeded:
            raise ValueError('Every selected worker failed. See each model error for the required input or runtime change.')
        all_valid=0;all_disagree=0;final_predictions=[]
        for si,(image,stack) in enumerate(stacks):
            run.update(progress=85+int(13*si/len(stacks)),stage=f'Writing result maps · image {si+1} of {len(stacks)}');persist(run)
            consensus,disagree,confidence=consensus_maps(probabilities[si],stack.valid)
            if run.get('combination')=='adaptive':
                completed=[model for model in run['models'] if model['status']=='complete']
                scored=[(index,model['metrics']['macro_f1']) for index,model in enumerate(completed) if model.get('metrics',{}).get('status')=='measured']
                if scored:
                    selected=max(scored,key=lambda item:item[1])[0]
                    consensus=probabilities[si][selected].argmax(0).astype('int32');consensus[~stack.valid]=-1
                    confidence=probabilities[si][selected].max(0)
                    run['selected_model']=completed[selected]['id']
                    run['selection_reason']='Highest macro F1 on the reported validation set; disagreement still compares all successful workers'
                else:
                    run['selection_reason']='Validation metrics unavailable; equal probability averaging, requiring human review'
            all_valid+=int(stack.valid.sum());all_disagree+=int(disagree.sum())
            final_predictions.append(consensus.ravel())
            write_raster(out/f'{si}-consensus.tif',(consensus+1).astype('uint16'),stack)
            write_raster(out/f'{si}-disagreement.tif',disagree.astype('uint8'),stack,nodata=255)
            write_raster(out/f'{si}-confidence.tif',confidence.astype('float32'),stack,nodata=-1)
            _preview(out/f'{si}-consensus.png',consensus,stack.valid)
            rgb=PALETTE[np.maximum(consensus,0)%len(PALETTE)].copy();rgb[~stack.valid]=0
            write_raster(out/f'{si}-consensus-rgb.tif',np.moveaxis(rgb,-1,0),stack)
            review_rgb=np.zeros_like(rgb);review_rgb[disagree]=[255,126,56]
            write_raster(out/f'{si}-disagreement-rgb.tif',np.moveaxis(review_rgb,-1,0),stack)
            overlay=np.zeros((*disagree.shape,4),dtype='uint8');overlay[disagree]=[255,126,56,220]
            Image.fromarray(overlay).save(out/f'{si}-disagreement.png')
            bounds=transform_bounds(stack.profile['crs'],'EPSG:4326',*__import__('rasterio').transform.array_bounds(stack.profile['height'],stack.profile['width'],stack.profile['transform']))
            files={name:f'/api/results/{run["id"]}/{si}-{name}.{ext}' for name,ext in [('consensus','tif'),('disagreement','tif'),('confidence','tif')]}
            for model in run['models']:
                if model['status']=='complete':files[model['id']]=f'/api/results/{run["id"]}/{si}-{model["id"]}.tif'
            run['outputs'].append({'image_id':image['id'],'files':files,'preview':f'/api/results/{run["id"]}/{si}-consensus.png',
                                   'disagreement_preview':f'/api/results/{run["id"]}/{si}-disagreement.png',
                                   'bbox':list(bounds),'metrics':score_predictions(validation[si],consensus,classes)})
            if np.any(testing[si]>=0):
                run['outputs'][-1]['test_metrics']=score_predictions(testing[si],consensus,classes)
            review = prepare_output_review(run, si, data_dir)
            run['outputs'][-1].update(review)
            if review.get('agreement_file'):
                files['agreement'] = review['agreement_file']
        agreement=round(100*(1-all_disagree/max(1,all_valid)),2) if succeeded>1 else None
        if any(np.any(mask>=0) for mask in testing):
            run['test_metrics']=score_predictions(np.concatenate([mask.ravel() for mask in testing]),np.concatenate(final_predictions),classes)
            run['validation_protocol']='Official validation split selects the model; separate official test images score the selected agent. Neither split enters fitting.'
        if pretrained and run.get('run_kind')!='training':
            run['validation_protocol']='Inference with the released pretrained encoder and classification head. No local fitting. References only score predictions; unverified pretraining overlap is not excluded.'
        run.update(status='complete' if succeeded==len(requested) else 'partial',progress=100,stage='Outputs ready for review',
                   elapsed_seconds=round(time.perf_counter()-start,3),
                   consensus={'agreement_percent':agreement,'disagreement_pixels':all_disagree,'valid_pixels':all_valid,
                              'models_compared':succeeded,'needs_user_review':bool(all_disagree) if succeeded>1 else None,
                              'message':f'Green: all {succeeded} models predict the same class. Orange: at least one predicts a different class. Agreement does not measure accuracy; transparent areas have no valid data.' if succeeded>1 else 'Only one worker completed; model agreement is unmeasured.'},
                   data_status='Source pixels processed; GeoTIFF outputs and model checkpoints saved locally')
    except Exception as exc:
        run.update(status='needs_input' if not run.get('models') else 'failed',stage=str(exc),error=str(exc),
                   consensus={'agreement_percent':None,'needs_user_review':None,'message':'No model agreement can be computed without prediction maps.'})
    run['elapsed_seconds']=round(time.perf_counter()-start,3)
    persist(run)
    (out/'report.json').write_text(json.dumps(run,indent=2,default=str))
    return run
