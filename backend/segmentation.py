"""Prompted SAM 3 Tracker masks with a georeferenced raster adapter.

Proposals are not class votes or ground truth. Only explicit human approval
creates a reference through the existing correction workflow.
"""
import json
from pathlib import Path
import time
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes, bounds as geometry_bounds
from rasterio.warp import reproject, transform, transform_geom
from PIL import Image
from raster_pipeline import read_stack, write_raster
from algorithm_registry import SEGMENTATION_DETAILS

from sam3_runtime import prepare, Predictor
SPEC = {**SEGMENTATION_DETAILS['sam3_tracker'],'id':'sam3_tracker',
        'source':'https://github.com/facebookresearch/sam3','license':'SAM License',
        'input':'Three explicitly mapped display bands; default RED/GREEN/BLUE',
        'trainable':False,'training_status':'SAM 3 mask inference only; supervised fine-tuning is not implemented. Explicitly classified and approved masks feed the shared correction policy.'}


def execute(task, data, reference, image, access, persist):
    started = time.perf_counter()
    folder = data/'segmentations'/task['id']
    folder.mkdir(parents=True,exist_ok=True)
    def progress(value, stage):
        task.update(status='running',progress=value,stage=stage)
        persist(task)
    try:
        if task.get('model','sam3_tracker')!='sam3_tracker':raise ValueError('This saved task was created for SAM 2.1. Its previous outputs are preserved; start a new SAM 3 segmentation instead of resuming it with different weights.')
        progress(2,'Reading georeferenced source bands')
        with rasterio.open(reference) as src:
            b=src.bounds
            area=transform_geom(src.crs,'EPSG:4326',{'type':'Polygon','coordinates':[[
                [b.left,b.bottom],[b.right,b.bottom],[b.right,b.top],[b.left,b.top],[b.left,b.bottom]]]})
            ref_valid=(src.read(1)>0)&(src.read_masks(1)>0)
            profile=src.profile.copy()
        stack=read_stack(access,image,max_size=1024,geometry=area)
        bands=task.get('rgb_bands') or ['RED','GREEN','BLUE']
        if len(bands)!=3 or any(b not in stack.bands for b in bands):
            raise ValueError(f'Choose three available display bands. Available: {", ".join(stack.bands)}')
        valid=np.zeros(stack.valid.shape,dtype='uint8')
        reproject(ref_valid.astype('uint8'),valid,src_transform=profile['transform'],src_crs=profile['crs'],
                  dst_transform=stack.profile['transform'],dst_crs=stack.profile['crs'],resampling=Resampling.nearest)
        stack.valid &= valid.astype(bool)
        if not stack.valid.any():raise ValueError('No valid pixels are available in this image region')
        rgb=np.zeros((*stack.valid.shape,3),dtype='uint8');stretch=[]
        for i,band in enumerate(bands):
            values=stack.pixels[stack.bands.index(band)]
            low,high=np.percentile(values[stack.valid],[2,98])
            rgb[:,:,i]=(np.clip((np.nan_to_num(values)-low)/max(float(high-low),1e-8),0,1)*255).astype('uint8')
            stretch.append({'band':band,'low':float(low),'high':float(high)})
        rgb[~stack.valid]=0
        point_coords=[];point_labels=[]
        inverse=~stack.profile['transform']
        for point in task.get('points',[]):
            xs,ys=transform('EPSG:4326',stack.profile['crs'],[point['longitude']],[point['latitude']])
            col,row=inverse*(xs[0],ys[0])
            if not (0<=col<rgb.shape[1] and 0<=row<rgb.shape[0]):
                raise ValueError('All segmentation points must lie inside the image')
            if not stack.valid[int(row),int(col)]:raise ValueError('Prompt point has no valid classified source pixels')
            point_coords.append([col,row]);point_labels.append(point['label'])
        box=None
        if task.get('box_geometry'):
            box=np.array(geometry_bounds(transform_geom('EPSG:4326',stack.profile['crs'],task['box_geometry'])))
            corners=[inverse*(box[0],box[1]),inverse*(box[2],box[3])]
            xs,ys=zip(*corners)
            box=np.array([max(0,min(xs)),max(0,min(ys)),min(rgb.shape[1],max(xs)),min(rgb.shape[0],max(ys))],dtype='float32')
            if box[0]>=box[2] or box[1]>=box[3]:raise ValueError('Draw a box intersecting this image')
        weights, provenance=prepare(data,progress)
        provenance={**SPEC,**provenance}
        progress(35,'Loading local SAM 3 sessions')
        predictor=Predictor(weights)
        masks,scores=predictor.predict(rgb,point_coords,point_labels,box,progress)
        provenance['execution_provider']=predictor.provider
        progress(90,'Saving georeferenced mask proposals')
        candidates=[]
        Image.fromarray(rgb).save(folder/'source.png')
        for index in np.argsort(-np.asarray(scores).reshape(-1)):
            mask=np.asarray(masks[index],dtype=bool)&stack.valid
            if not mask.any():continue
            candidate_id=len(candidates)
            polygons=[transform_geom(stack.profile['crs'],'EPSG:4326',g)['coordinates']
                      for g,_ in shapes(mask.astype('uint8'),mask=mask,transform=stack.profile['transform'])]
            geometry={'type':'MultiPolygon','coordinates':polygons}
            prefix=f'candidate-{candidate_id}'
            write_raster(folder/f'{prefix}.tif',np.where(stack.valid,mask.astype('uint8'),255),stack,nodata=255)
            (folder/f'{prefix}.geojson').write_text(json.dumps({'type':'FeatureCollection','features':[
                {'type':'Feature','properties':{'model':'SAM 3 Tracker','proposal':True},'geometry':geometry}]}))
            preview=rgb.copy()
            preview[mask]=(preview[mask]*.4+np.array([102,217,191])*.6).astype('uint8')
            Image.fromarray(preview).save(folder/f'{prefix}.png')
            candidates.append({'id':candidate_id,'pixels':int(mask.sum()),'predicted_mask_quality':float(scores[index]),
                'geometry':geometry,'preview':f'/api/segmentations/{task["id"]}/{prefix}.png',
                'mask':f'/api/segmentations/{task["id"]}/{prefix}.tif',
                'geojson':f'/api/segmentations/{task["id"]}/{prefix}.geojson'})
        if not candidates:raise ValueError('No mask found for these prompts. Try a different point or box.')
        task.update(status='complete',progress=100,stage='Proposals ready for human review',candidates=candidates,
            provenance=provenance,rgb_stretch=stretch,input_sources=stack.provenance,
            grid={'crs':str(stack.profile['crs']),'shape':list(stack.valid.shape),'transform':list(stack.profile['transform'])},
            accuracy={'status':'unmeasured','message':'SAM quality scores are model estimates, not measured Precision/Recall/F1.'},
            training={'enabled':False,'reason':SPEC['training_status']},elapsed_seconds=round(time.perf_counter()-started,3))
    except Exception as exc:
        task.update(status='failed',stage=str(exc),error=str(exc))
    persist(task)
    return task
