"""Full-spectrum anomaly workers and immutable threshold views."""
from execution_policy import torch_device
from copy import deepcopy
import json
from pathlib import Path
import time
from uuid import uuid4

import joblib
import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize
from rasterio.warp import transform_geom, transform_bounds
from threadpoolctl import threadpool_limits

from algorithm_registry import ANOMALY_DETAILS
from raster_pipeline import read_stack, write_raster
from pretrained_lulc import sha256


def review_masks(stack, reviews, confidence=3):
    """Latest reference owns a pixel; validation always excludes fitting."""
    def burn(rows):
        return rasterize([(transform_geom('EPSG:4326',stack.profile['crs'],r['geometry']),i+1)
            for i,r in enumerate(rows)],out_shape=stack.valid.shape,transform=stack.profile['transform'],fill=0,dtype='int32') if rows else np.zeros(stack.valid.shape,dtype='int32')
    owner=burn(reviews)
    held=burn([r for r in reviews if r.get('role')=='validation'])>0
    normal=np.zeros_like(stack.valid)
    anomaly=np.zeros_like(stack.valid)
    for i,r in enumerate(reviews,1):
        if r.get('role','train')=='train' and r.get('confidence',3)>=confidence:
            (normal if r['label']=='normal' else anomaly)[owner==i]=True
    return normal&~held&stack.valid,anomaly&~held&stack.valid,held


def _network(bands):
    import torch.nn as nn
    width=min(128,max(16,bands*2));latent=min(32,max(2,bands//4))
    return nn.Sequential(nn.Linear(bands,width),nn.GELU(),nn.Linear(width,latent),nn.GELU(),
                         nn.Linear(latent,width),nn.GELU(),nn.Linear(width,bands))


def fit_autoencoder(background, destination, epochs, progress, previous=None):
    import torch
    torch.set_num_threads(2);torch.manual_seed(1729)
    rng=np.random.default_rng(1729)
    sample=background[rng.choice(len(background),min(12000,len(background)),replace=False)].astype('float32')
    order=rng.permutation(len(sample));cut=max(1,int(len(sample)*.8))
    train=torch.from_numpy(sample[order[:cut]]);val=torch.from_numpy(sample[order[cut:]])
    model=_network(sample.shape[1]).to(torch_device());val=val.to(torch_device());continued=False
    if previous and Path(previous).is_file():
        model.load_state_dict(torch.load(previous,map_location='cpu',weights_only=True));continued=True
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    history=[];best=float('inf');best_state=None;stale=0
    for epoch in range(epochs):
        model.train();losses=[]
        for batch in torch.randperm(len(train)).split(256):
            values=train[batch].to(torch_device());loss=torch.nn.functional.smooth_l1_loss(model(values),values)
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():v=float(torch.nn.functional.mse_loss(model(val),val))
        history.append({'epoch':epoch+1,'train_huber':float(np.mean(losses)),'optimization_holdout_mse':v})
        if v<best:best=v;best_state=deepcopy(model.state_dict());stale=0
        else:stale+=1
        progress(round(100*(epoch+1)/epochs),f'Autoencoder epoch {epoch+1}/{epochs}')
        if stale>=4:break
    model.load_state_dict(best_state);model.eval();torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},destination)
    return model,{'epochs_completed':len(history),'history':history,'continued_checkpoint':continued,
        'background_samples':len(sample),'optimization_holdout':'Random 20% of selected background samples, used only for reconstruction early stopping; not detection accuracy'}


def local_rx(z, valid, background_mask, fallback, progress, outer=63, guard=31, cell=16):
    from scipy.linalg import cho_factor, cho_solve
    from sklearn.covariance import OAS
    h,w=valid.shape;result=np.full((h,w),np.nan);fallback_cells=0;cells=0
    rng=np.random.default_rng(1729)
    for y in range(0,h,cell):
        for x in range(0,w,cell):
            y1,x1=min(h,y+cell),min(w,x+cell)
            cy,cx=(y+y1)//2,(x+x1)//2
            ya,yb=max(0,cy-outer//2),min(h,cy+outer//2+1)
            xa,xb=max(0,cx-outer//2),min(w,cx+outer//2+1)
            ring=background_mask[ya:yb,xa:xb].copy()
            iy0,iy1=max(ya,cy-guard//2),min(yb,cy+guard//2+1)
            ix0,ix1=max(xa,cx-guard//2),min(xb,cx+guard//2+1)
            ring[iy0-ya:iy1-ya,ix0-xa:ix1-xa]=False
            samples=z[ya:yb,xa:xb][ring]
            if len(samples)>=32:
                if len(samples)>512:samples=samples[rng.choice(len(samples),512,replace=False)]
                estimator=OAS(store_precision=False).fit(samples)
                center=estimator.location_;cov=estimator.covariance_
            else:
                center=fallback.location_;cov=fallback.covariance_;fallback_cells+=1
            target=z[y:y1,x:x1]-center
            flat=target.reshape(-1,z.shape[-1])
            factor=cho_factor(cov+np.eye(z.shape[-1])*1e-6,check_finite=False)
            values=np.einsum('ij,ij->i',flat,cho_solve(factor,flat.T,check_finite=False).T).reshape(y1-y,x1-x)
            result[y:y1,x:x1]=values;cells+=1
        progress(round(100*min(h,y+cell)/h),'Local RX background windows')
    result[~valid]=np.nan
    return result,{'outer_window':outer,'guard_window':guard,'score_cell':cell,'background_sample_cap':512,
        'covariance':'OAS shrinkage per cell','fallback_cells':fallback_cells,'cells':cells,
        'approximation':'Each 16×16 score cell shares one local ring covariance; every pixel retains its full spectral vector. Insufficient rings use the saved global covariance.'}


def percentile_map(scores,valid):
    from scipy.stats import rankdata
    result=np.full(valid.shape,-1,dtype='float32')
    result[valid]=(100*rankdata(scores[valid],method='average')/(int(valid.sum())+1)).astype('float32')
    return result


def heat_rgb(percentiles,valid):
    stops=[0,50,90,98,100];colors=np.array([[10,29,39],[22,70,87],[45,140,138],[255,181,62],[251,57,88]])
    rgb=np.stack([np.interp(percentiles,stops,colors[:,i]) for i in range(3)],axis=-1).astype('uint8')
    rgb[~valid]=0
    return rgb


def make_view(run,index,data,percentile=99.5,min_votes=2):
    """New files for each threshold version; saved detector scores never change."""
    folder=data/'anomalies'/run['id'];output=run['outputs'][index]
    ids=list(output['detectors']);view_id=str(uuid4());base=f'{index}-view-{view_id}'
    with rasterio.open(folder/f'{index}-{ids[0]}-percentiles.tif') as src:
        valid=src.read(1)>=0;profile=src.profile.copy()
    votes=np.zeros(valid.shape,dtype='uint8');counts={};flags={}
    for mid in ids:
        with rasterio.open(folder/f'{index}-{mid}-percentiles.tif') as src:rank=src.read(1)
        flag=(rank>=percentile)&valid;votes+=flag;counts[mid]=int(flag.sum());flags[mid]=flag
    min_votes=min(min_votes,len(ids));selected=(votes>=min_votes)&valid
    binary=np.where(valid,selected.astype('uint8'),255).astype('uint8')
    def save(name,array,dtype,nodata):
        with rasterio.open(folder/f'{base}-{name}.tif','w',**{**profile,'count':1,'dtype':dtype,'nodata':nodata}) as dst:dst.write(array,1)
    save('mask',binary,'uint8',255);save('votes',np.where(valid,votes,255).astype('uint8'),'uint8',255)
    vote_colors=np.array([[16,47,54],[246,184,71],[250,127,60],[232,83,160],[252,61,83]],dtype='uint8')
    rgb=vote_colors[votes];rgb[~valid]=0
    for layer,colors in [('votes',rgb),('mask',np.where(selected[:,:,None],np.array([252,76,98]),0).astype('uint8'))]:
        Image.fromarray(np.dstack((colors,(valid if layer=='votes' else selected).astype('uint8')*220))).save(folder/f'{base}-{layer}.png')
        with rasterio.open(folder/f'{base}-{layer}-rgb.tif','w',**{**profile,'count':3,'dtype':'uint8','nodata':0}) as dst:dst.write(np.moveaxis(colors,-1,0))
    url=f'/api/anomalies/runs/{run["id"]}/files/'
    return {'id':view_id,'image_index':index,'percentile':percentile,'minimum_votes':min_votes,'detector_count':len(ids),
        'valid_pixels':int(valid.sum()),'flagged_pixels':int(selected.sum()),'detector_counts':counts,
        'flagged_percent':round(100*selected.sum()/max(1,valid.sum()),3),'prefix':base,
        'vote_distribution':{str(n):int(((votes==n)&valid).sum()) for n in range(len(ids)+1)},
        'agreement_percent':round(100*int((((votes==0)|(votes==len(ids)))&valid).sum())/max(1,int(valid.sum())),2),
        'preview':url+f'{base}-votes.png','mask_preview':url+f'{base}-mask.png',
        'mask':url+f'{base}-mask.tif','votes':url+f'{base}-votes.tif',
        'metrics':{'status':'unmeasured','precision':None,'recall':None,'f1':None,'reason':'Requires human validation references; agreement and percentile are not accuracy or probability.'}}


def execute(run,images,access,reviews,data,persist,initial=None):
    from sklearn.covariance import LedoitWolf
    from sklearn.ensemble import IsolationForest
    start=time.perf_counter();folder=data/'anomalies'/run['id'];folder.mkdir(parents=True,exist_ok=True)
    run.update(status='running',progress=1,stage='Reading all science bands',outputs=[])
    persist(run)
    try:
        (folder/'input-snapshot.json').write_text(json.dumps({'images':images,'access':access,'reviews':reviews},indent=2))
        for index,image in enumerate(images):
            begin=100*index/len(images);span=100/len(images)
            def progress(value,stage):
                run.update(progress=min(99,round(begin+span*value/100)),stage=f'Image {index+1}/{len(images)} · {stage}');persist(run)
            stack=read_stack(access[image['id']],image,max_size=run['max_size'],geometry=run.get('geometry'),science_only=True,max_elements=30_000_000)
            applicable=[r for r in reviews if r['image_id']==image['id']]
            normal,positive,held=review_masks(stack,applicable,run.get('training_policy',{}).get('minimum_confidence',3))
            eligible=normal if run.get('background_mode')=='reviewed_normal' else stack.valid&~held&~positive
            if eligible.sum()<32:raise ValueError(f'{image["id"]}: need at least 32 valid background pixels outside validation/confirmed anomalies')
            cube=np.moveaxis(stack.pixels,0,-1).astype('float64');bands=len(stack.bands)
            background=cube[eligible]
            mean=background.mean(0);scale=background.std(0);constant=scale<=np.maximum(np.abs(mean)*1e-12,1e-12)
            if constant.all():raise ValueError('All science bands are constant in the background')
            scale[constant]=1
            z=np.nan_to_num((cube-mean)/scale);background=z[eligible]
            rng=np.random.default_rng(1729)
            sampled=background[rng.choice(len(background),min(6000,len(background)),replace=False)]
            progress(7,f'Fitting background with all {bands} bands')
            with threadpool_limits(limits=2):global_model=LedoitWolf().fit(sampled)
            output={'image_id':image['id'],'bands':stack.bands,'band_count':bands,
                'band_inventory':[{'channel':i+1,'name':b,'constant_in_background':bool(constant[i])} for i,b in enumerate(stack.bands)],
                'excluded_bands':[b for p in stack.provenance for b in p.get('bands',[]) if b.get('excluded')],
                'sources':stack.provenance,'shape':list(stack.pixels.shape),'crs':str(stack.profile['crs']),
                'transform':list(stack.profile['transform']),'valid_pixels':int(stack.valid.sum()),
                'background_pixels':int(eligible.sum()),'detectors':{},
                'bbox':list(transform_bounds(stack.profile['crs'],'EPSG:4326',*rasterio.transform.array_bounds(stack.profile['height'],stack.profile['width'],stack.profile['transform'])))}
            run['outputs'].append(output)
            for mi,mid in enumerate(run['requested_models']):
                def step(value,stage):progress(10+80*(mi+value/100)/len(run['requested_models']),stage)
                step(0,f'Running {ANOMALY_DETAILS[mid]["name"]}')
                model_path=folder/f'{index}-{mid}.joblib';extra={}
                with threadpool_limits(limits=2):
                    if mid=='rx':
                        values=z[stack.valid]-global_model.location_
                        scores=np.zeros(stack.valid.shape);scores[stack.valid]=np.einsum('ij,jk,ik->i',values,global_model.precision_,values,optimize=True)
                        fitted=global_model;extra={'covariance':'Ledoit-Wolf shrinkage','shrinkage':float(global_model.shrinkage_)}
                    elif mid=='local_rx':
                        scores,extra=local_rx(z,stack.valid,eligible,global_model,step,run['outer_window'],run['guard_window'])
                        fitted=global_model
                    elif mid=='isolation_forest':
                        fitted=IsolationForest(n_estimators=128,max_samples=min(256,len(sampled)),max_features=1.0,random_state=1729,n_jobs=2).fit(sampled)
                        scores=np.zeros(stack.valid.shape);scores[stack.valid]=-fitted.score_samples(z[stack.valid])
                        extra={'trees':128,'features_available_to_each_tree':bands}
                    elif mid=='autoencoder':
                        import torch
                        ae_path=folder/f'{index}-autoencoder.pt';previous=None
                        if initial and image['id'] in initial:
                            old=joblib.load(initial[image['id']])
                            # Normalization is checkpoint-specific; start afresh if it changed.
                            if old['bands']==stack.bands and np.allclose(old['mean'],mean) and np.allclose(old['scale'],scale):previous=old.get('ae_path')
                        model,extra=fit_autoencoder(background,ae_path,run['epochs'],step,previous)
                        scores=np.zeros(stack.valid.shape);values=z[stack.valid].astype('float32');reconstruction=[]
                        with torch.inference_mode():
                            for offset in range(0,len(values),4096):
                                batch=torch.from_numpy(values[offset:offset+4096]).to(next(model.parameters()).device);reconstruction.append(((model(batch)-batch)**2).mean(1).cpu().numpy())
                        scores[stack.valid]=np.concatenate(reconstruction)
                        fitted=None;extra['ae_path']=str(ae_path)
                    else:raise ValueError(f'No anomaly adapter is registered for {mid}')
                if not np.isfinite(scores[stack.valid]).all():raise ValueError(f'{mid}: non-finite anomaly scores')
                scores=scores.astype('float32');scores[~stack.valid]=-1
                ranks=percentile_map(scores,stack.valid)
                record={'model_id':mid,'bands':stack.bands,'mean':mean,'scale':scale,'fitted':fitted,**extra}
                joblib.dump(record,model_path)
                write_raster(folder/f'{index}-{mid}-scores.tif',scores,stack,nodata=-1)
                write_raster(folder/f'{index}-{mid}-percentiles.tif',ranks,stack,nodata=-1)
                colors=heat_rgb(ranks,stack.valid)
                Image.fromarray(np.dstack((colors,stack.valid.astype('uint8')*220))).save(folder/f'{index}-{mid}.png')
                write_raster(folder/f'{index}-{mid}-rgb.tif',np.moveaxis(colors,-1,0),stack,nodata=0)
                url=f'/api/anomalies/runs/{run["id"]}/files/{index}-{mid}'
                output['detectors'][mid]={'id':mid,'name':ANOMALY_DETAILS[mid]['name'],'checkpoint':str(model_path),
                    'checkpoint_sha256':sha256(model_path),
                    'network_sha256':sha256(Path(extra['ae_path'])) if 'ae_path' in extra else None,
                    'preview':url+'.png','scores':url+'-scores.tif','percentiles':url+'-percentiles.tif',
                    'score_min':float(scores[stack.valid].min()),'score_max':float(scores[stack.valid].max()),
                    'score_definition':'Squared regularized Mahalanobis distance' if mid in ('rx','local_rx') else 'Negative Isolation Forest score_samples' if mid=='isolation_forest' else 'Mean squared reconstruction error across every standardized science band',
                    'fit':{k:v for k,v in extra.items() if k!='ae_path'}}
                step(100,f'{ANOMALY_DETAILS[mid]["name"]} finished')
            output['default_view']=make_view(run,index,data,run['percentile'],run['minimum_votes'])
            progress(100,'Maps and checkpoints saved')
        run.update(status='complete',progress=100,stage='Anomaly maps ready for review')
    except Exception as exc:run.update(status='failed',error=str(exc),stage=str(exc))
    run['elapsed_seconds']=round(time.perf_counter()-start,3);persist(run)
    return run
