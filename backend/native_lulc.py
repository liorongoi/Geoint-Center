"""Windowed native-resolution pretrained land-cover inference; never shrink a scene to fit RAM."""
from contextlib import ExitStack
import json
import math
from pathlib import Path
import time
import joblib
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import bounds as geometry_bounds
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds, transform_geom
from rasterio.windows import Window, from_bounds, transform as window_transform
from PIL import Image
from offline_policy import local_raster, raster_driver
from raster_pipeline import source_assets, read_stack, label_mask
import pretrained_lulc as models


def native_grid(record, geometry=None, model_id="dynamicworld"):
    """Use an actual source band at the model resolution and snap the AOI to that grid."""
    spec=models.PRETRAINED[model_id];resolution=spec["native_resolution"]
    with ExitStack() as context:
        sources=[context.enter_context(rasterio.open(local_raster(path),**({'driver':raster_driver(path)} if record['mode'] in ('local_path','download') else {}))) for _,path in source_assets(record)]
        candidates=[s for s in sources if s.crs and s.crs.is_projected and
                    np.allclose([abs(s.transform.a)*s.crs.linear_units_factor[1],abs(s.transform.e)*s.crs.linear_units_factor[1]],[resolution,resolution],atol=.05) and
                    s.transform.b==0 and s.transform.d==0]
        if not candidates:raise ValueError(f'{spec["name"]} needs a georeferenced native {resolution} m source band. Check the imported product.')
        src=candidates[0];window=Window(0,0,src.width,src.height)
        if geometry and geometry.get('type')!='Point':
            box=geometry_bounds(transform_geom('EPSG:4326',src.crs,geometry))
            if box[0]>=src.bounds.right or box[2]<=src.bounds.left or box[1]>=src.bounds.top or box[3]<=src.bounds.bottom:
                raise ValueError('AOI does not intersect this image')
            w=from_bounds(max(box[0],src.bounds.left),max(box[1],src.bounds.bottom),min(box[2],src.bounds.right),min(box[3],src.bounds.top),src.transform)
            left,top=math.floor(w.col_off),math.floor(w.row_off)
            window=Window(left,top,math.ceil(w.col_off+w.width)-left,math.ceil(w.row_off+w.height)-top).intersection(window)
        return {'crs':src.crs,'transform':window_transform(window,src.transform),'width':int(window.width),'height':int(window.height)}


def metrics(matrix, classes):
    if not matrix.sum():return {'status':'unmeasured','reason':'No independent validation references in this image'}
    support=matrix.sum(1);predicted=matrix.sum(0);tp=np.diag(matrix)
    divide=lambda a,b:np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b!=0)
    precision=divide(tp,predicted);recall=divide(tp,support);f1=divide(2*precision*recall,precision+recall)
    return {'status':'measured','pixels':int(matrix.sum()),'macro_precision':float(precision.mean()),'macro_recall':float(recall.mean()),
            'macro_f1':float(f1.mean()),'mean_iou':float(divide(tp,support+predicted-tp).mean()),'confusion_matrix':matrix.tolist(),
            'per_class':{name:{'precision':float(precision[i]),'recall':float(recall[i]),'f1-score':float(f1[i]),'support':int(support[i])} for i,name in enumerate(classes)},
            'macro_definition':f'All {len(classes)} native classes; zero for undefined precision/recall. Scores describe supplied validation regions only.'}


def execute(run, images, access, labels, data_dir, persist, max_size=384, geometry=None, initial_checkpoints=None):
    from job_runner import PALETTE
    started=time.perf_counter();out=Path(data_dir)/'results'/run['id'];out.mkdir(parents=True,exist_ok=True)
    model_id=run['requested_models'][0];spec=models.PRETRAINED[model_id]
    classes=spec['classes'];class_count=len(classes);resolution=spec['native_resolution']
    crop=model_id=='prithvi_crop'
    tile_size=160 if crop else max(64,min(384,int(max_size)))
    context_pixels=32
    run.update(status='running',stage=f'Preparing native {resolution} m raster windows',progress=1,outputs=[],models=[],classes=classes,
               combination='separate',grid_policy={'resolution_metres':resolution,'tile_size':tile_size,'context_pixels':context_pixels,'resampling':'Science bands bilinear; masks nearest; original bands preserved','storage':'Windowed GeoTIFF outputs on the native source grid'},
               band_alignment={'enabled':False,'method':'Supplier georeferencing onto native 10 m grid; no additional band shifts'})
    persist(run)
    try:
        total_confusion=np.zeros((class_count,class_count),dtype='int64');record=None;training=None
        for image_index,image in enumerate(images):
            source=access.get(image['id'])
            if not source:raise ValueError('Configure raster access for '+image['id'])
            grid=native_grid(source,geometry,model_id);height,width=grid['height'],grid['width']
            total_tiles=math.ceil(height/tile_size)*math.ceil(width/tile_size);done=0;valid_count=0
            confusion=np.zeros((class_count,class_count),dtype='int64');counts=np.zeros(class_count,dtype='int64');provenance=[]
            profile={**grid,'driver':'GTiff','count':1,'tiled':True,'blockxsize':256,'blockysize':256,'compress':'deflate','BIGTIFF':'IF_SAFER'}
            with ExitStack() as context:
                targets={}
                for name,dtype,nodata in [('consensus','uint16',0),(model_id,'uint16',0),('confidence','float32',-1),('disagreement','uint8',255)]:
                    targets[name]=context.enter_context(rasterio.open(out/f'{image_index}-{name}.tif','w',**profile,dtype=dtype,nodata=nodata))
                for y in range(0,height,tile_size):
                    for x in range(0,width,tile_size):
                        h,w=min(tile_size,height-y),min(tile_size,width-x)
                        y0,x0=max(0,y-context_pixels),max(0,x-context_pixels);y1,x1=min(height,y+h+context_pixels),min(width,x+w+context_pixels)
                        window=Window(x0,y0,x1-x0,y1-y0)
                        tile_grid={**grid,'width':int(window.width),'height':int(window.height),'transform':window_transform(window,grid['transform'])}
                        run.update(stage=f'Image {image_index+1}/{len(images)} · native tile {done+1}/{total_tiles}',progress=min(98,int(98*(image_index+done/total_tiles)/len(images))))
                        persist(run)
                        stack=None
                        try:
                            stack=read_stack(source,image,geometry=geometry,target_profile=tile_grid,verify_sources=done==0,science_only=True)
                        except ValueError as exc:
                            if str(exc)!='No valid pixels in the selected AOI':raise
                        if stack is not None:
                            models.validate_input(model_id,stack,image,source)
                            if record is None:
                                checkpoint=out/f'{model_id}.joblib'
                                training=models.checkpoint(model_id,stack,checkpoint,Path(data_dir)/'models',(initial_checkpoints or {}).get(model_id))
                                record=joblib.load(checkpoint)
                            provenance=stack.provenance
                            probability=models.predict(record,stack,lambda stage,pct:persist(run))
                            sy,sx=y-y0,x-x0
                            valid=stack.valid[sy:sy+h,sx:sx+w];probability=probability[:,sy:sy+h,sx:sx+w]
                            classified=(np.argmax(probability,axis=0)+1).astype('uint16');classified[~valid]=0
                            confidence=np.max(probability,axis=0);confidence[~valid]=-1
                            refs=[r for r in labels if r.get('role')=='validation' and (not r.get('image_id') or r['image_id']==image['id']) and r.get('benchmark_split')!='test']
                            truth=label_mask(stack,refs,classes)[sy:sy+h,sx:sx+w];known=valid&(truth>=0)
                            confusion+=np.bincount((truth[known]*class_count+classified[known]-1).astype('int64'),minlength=class_count**2).reshape(class_count,class_count)
                        else:
                            valid=np.zeros((h,w),bool);classified=np.zeros((h,w),'uint16');confidence=np.full((h,w),-1,'float32')
                        core=Window(x,y,w,h)
                        for name in ('consensus',model_id):targets[name].write(classified,1,window=core)
                        targets['confidence'].write(confidence,1,window=core)
                        targets['disagreement'].write(np.where(valid,0,255).astype('uint8'),1,window=core)
                        valid_count+=int(valid.sum());counts+=np.bincount(classified[valid]-1,minlength=class_count);done+=1
            if not valid_count:raise ValueError('No valid pixels in the selected AOI')
            # Display only: bounded previews and map RGB never replace science rasters.
            with rasterio.open(out/f'{image_index}-consensus.tif') as src:
                scale=min(1,1024/max(width,height));shape=(max(1,round(height*scale)),max(1,round(width*scale)))
                display=src.read(1,out_shape=shape,resampling=Resampling.nearest)
                transform=src.transform*rasterio.Affine.scale(width/shape[1],height/shape[0])
                rgb=PALETTE[np.maximum(display,1)-1];rgb[display==0]=0
                rgba=np.dstack((rgb,(display>0).astype('uint8')*220));Image.fromarray(rgba).save(out/f'{image_index}-consensus.png')
                with rasterio.open(out/f'{image_index}-consensus-rgb.tif','w',driver='GTiff',width=shape[1],height=shape[0],count=3,dtype='uint8',crs=grid['crs'],transform=transform,nodata=0) as dst:dst.write(np.moveaxis(rgb,-1,0))
            result_metrics=metrics(confusion,classes);total_confusion+=confusion
            run['outputs'].append({'image_id':image['id'],'bbox':list(transform_bounds(grid['crs'],'EPSG:4326',*array_bounds(height,width,grid['transform']))),
                'files':{name:f'/api/results/{run["id"]}/{image_index}-{name}.tif' for name in targets},'preview':f'/api/results/{run["id"]}/{image_index}-consensus.png',
                'grid':{'width':width,'height':height,'resolution_metres':resolution,'crs':str(grid['crs'])},'metrics':result_metrics,'source_provenance':provenance})
            run['models']=[{'id':model_id,'name':spec['name'],'status':'complete','training':training,'metrics':metrics(total_confusion,classes),'why':f'Windowed {resolution} m inference retains the full selected area and all {class_count} native classes'}]
            legend=[{'label':name,'value':i+1,'color':'#%02x%02x%02x'%tuple(PALETTE[i])} for i,name in enumerate(classes)]
            summary={'class_legend':legend,'class_distribution':[{**item,'pixels':int(counts[i]),'percent':round(100*int(counts[i])/valid_count,2)} for i,item in enumerate(legend)],
                     'classification_summary':{'valid_pixels':valid_count,'nodata_pixels':height*width-valid_count,'configured_classes':class_count,'predicted_classes':int((counts>0).sum()),'source':'Final classification raster; windowed counts, nodata excluded'},
                     'agreement_summary':{'models_compared':1,'agreement_percent':None}}
            (out/f'{image_index}-review.json').write_text(json.dumps(summary,indent=2))
            run['outputs'][-1].update(summary)
        run.update(status='complete',progress=100,stage=f'Native {resolution} m outputs ready',data_status='Full selected AOI processed in bounded windows',
                   consensus={'models_compared':1,'agreement_percent':None,'needs_user_review':None,'message':'One pretrained classifier; agreement is unmeasured'},validation_protocol='Released pretrained model, no local fitting. Metrics use supplied human validation only; pretraining overlap unverified.')
    except Exception as exc:
        run.update(status='failed',stage=str(exc),error=str(exc))
    run['elapsed_seconds']=round(time.perf_counter()-started,3);persist(run)
    (out/'report.json').write_text(json.dumps(run,indent=2,default=str))
    return run
