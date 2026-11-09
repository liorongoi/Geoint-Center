"""Shared input and training-reference scope checks, before any batch is queued."""
import numpy as np,rasterio
from raster_pipeline import read_stack,RasterStack,source_assets
from offline_policy import local_raster,raster_driver

def training_references(labels,classes,selected_ids=None):
 return [r for r in labels if r.get('label') in classes and r.get('role','train')=='train' and r.get('confidence',3)>=3 and r.get('benchmark_split') not in ('validation','test') and (selected_ids is None or r['id'] in selected_ids)]

def preflight(profile,record,image,geometry=None,data=None):
 from dsen2 import MODELS, preflight as check_dsen2
 if profile['algorithm'] in MODELS:
  return check_dsen2(profile['algorithm'],record,geometry,data)
 from pretrained_lulc import PRETRAINED,validate_input
 stack=read_stack(record,image,max_size=32,geometry=geometry)
 if not stack.valid.any():raise ValueError('No valid source pixels in selected area: '+image.get('name',image['id']))
 if profile['algorithm'] in PRETRAINED:
  # Validate sampled radiometry/band coverage against the actual source grid,
  # independently of the deliberately small preflight sampling grid.
  grids=[]
  for _,path in source_assets(record):
   with rasterio.open(local_raster(path),**({'driver':raster_driver(path)} if record['mode'] in ('local_path','download') else {})) as src:
    grids.append({'crs':src.crs,'transform':src.transform,'width':32,'height':32})
  native=min(grids,key=lambda g:abs(g['transform'].a*g['transform'].e))
  checked=RasterStack(stack.pixels,stack.valid,native,stack.bands,stack.provenance)
  validate_input(profile['algorithm'],checked,image,record)
 elif profile.get('task') in ('segmentation','detection'):
  missing=set(('RED','GREEN','BLUE'))-set(stack.bands)
  if missing:raise ValueError('Map source RGB channels before '+profile['name']+': missing '+', '.join(sorted(missing)))
 return {'bands':stack.bands,'radiometry_sampled':True}
