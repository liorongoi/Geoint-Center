"""Local-only aerial-trained Oriented R-CNN, separate from semantic classification."""
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from rasterio.warp import transform_geom
from execution_policy import torch_device,available_ram_bytes
from raster_pipeline import read_stack

MODEL_ID='oriented_rcnn_dota'
COMPONENT_SHA256='62b84b18294a54c6ef4be2dc7ca0e7e9e67d27cf556980d8c09ff9c4e4cafda9'
CLASSES=['baseball-diamond','basketball-court','bridge','ground-track-field','harbor','helicopter','large-vehicle','plane','roundabout','ship','small-vehicle','soccer-ball-field','storage-tank','swimming-pool','tennis-court']
DETAILS={MODEL_ID:{'name':'Oriented R-CNN · aerial DOTA pretrained','task':'detection','classes':CLASSES,'trainable':False,'training_mode':'unsupported','reason':'Pretrained RPN-based oriented object detector; supervised detector fine-tuning is not implemented in this adapter.','satellite_suitability':{'requirements':'RGB optical imagery; oriented boxes for 15 DOTA object categories. Best suited to high-resolution aerial/satellite imagery.','limitations':'Sentinel/Landsat resolution cannot reliably resolve small vehicles. Uses an explicit RGB display stretch, not every science band. No super-resolution or invented detail.'}}}

def component():
 roots=[os.getenv('SPECTRASCOPE_DETECTION_COMPONENT'),str(Path(os.getenv('SPECTRASCOPE_DATA_DIR','data/spectrascope'))/'components/oriented-rcnn')]
 bundled=os.getenv('SPECTRASCOPE_BUNDLED_MODELS')
 if bundled:roots.append(Path(bundled).parent/'components/oriented-rcnn')
 roots += [str(Path(sys.executable).resolve().parent.parent/'components/oriented-rcnn'),str(Path(__file__).parent/'desktop/payload/mac-arm64/components/oriented-rcnn')]
 return next((Path(p).resolve() for p in roots if p and (Path(p)/'component.json').is_file()),None)

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def prepare():
 root=component()
 if root is None:raise ValueError('Install the separate offline Oriented R-CNN component. Runtime downloads are disabled.')
 if digest(root/'component.json')!=COMPONENT_SHA256:raise ValueError('Detector component manifest differs from the pinned offline release')
 meta=json.loads((root/'component.json').read_text())
 for entry in meta['files']:
  p=(root/entry['path']).resolve()
  if not p.is_relative_to(root) or not p.is_file() or digest(p)!=entry['sha256']:raise ValueError('Detector component is missing or damaged: '+entry['path'])
 sys.path.insert(0,str(root/'python'))
 return root,meta

def load_detector(progress, fcos_pack=None):
 available=available_ram_bytes()
 if available is not None and available<3*1024**3:raise ValueError('Oriented R-CNN needs at least 3 GiB of available RAM for bounded CPU inference. Close other workloads and retry.')
 root,meta=prepare();progress(25,'Loading verified local aerial detector')
 import torch
 from oriented_det.runtime.checkpoint import load_model_from_checkpoint
 from oriented_det.runtime.inference import run_inference_auto,apply_nms_to_detections
 from oriented_det.train.config import get_preprocessing_params
 torch.set_num_threads(min(4,os.cpu_count() or 1));device=str(torch_device())
 if fcos_pack is None:
  checkpoint=(root/'model'/meta['model_file']).resolve();config=(root/'model/config.json').resolve()
 else:
  pack=Path(fcos_pack)
  checkpoint=pack/'model.pth';config=pack/'config.json'
  expected={checkpoint:'a87b6dba593272599d7e905ab5f86c7218a29944932da3d68fccb07c774ef86a',
            config:'e22fe28394fa7693577b043495618b8f1f4c8f3c19688b7e9febdb5376989eb4'}
  for path,checksum in expected.items():
   if path.is_symlink() or not path.is_file() or digest(path)!=checksum:
    raise ValueError('Install the verified local Rotated FCOS candidate pack')
  checkpoint=checkpoint.resolve();config=config.resolve()
  meta={'component':'rotated-fcos-research','model_sha256':expected[pack/'model.pth'],'config_sha256':expected[pack/'config.json']}
 # Absolute existing paths prevent the upstream loader's download fallback.
 model,cfg,names=load_model_from_checkpoint(checkpoint,config,device=device)
 if names!=CLASSES:raise ValueError('Detector class order differs from its pinned published configuration')
 def infer(rgb,score_threshold=.4):
  with torch.inference_mode():dets=run_inference_auto(image=Image.fromarray(rgb),model=model,device=device,preprocessing=get_preprocessing_params(cfg),score_threshold=score_threshold,class_names=names)
  return apply_nms_to_detections(dets,iou_threshold=.3)
 return infer,lambda detections:apply_nms_to_detections(detections,iou_threshold=.3),{**meta,'files':None,'device':device,'score_threshold':.4,'nms_iou':.3,'nms_backend':'Upstream exact convex polygon clipping CPU fallback; not benchmark-equivalent to MMRotate compiled kernels','published_model':'DOTA aerial-trained Rotated FCOS candidate' if fcos_pack else 'DOTA aerial-trained ResNet50/FPN Oriented R-CNN; practical baseline'}

def detect(rgb,progress):
 infer,merge,meta=load_detector(progress)
 progress(45,'Running RPN and oriented region classification')
 return infer(rgb),meta

def native_grid(access,geometry=None):
 """Metadata-only native AOI grid. Never shrink small objects to fit a preview."""
 import rasterio
 from rasterio.features import geometry_window
 from rasterio.windows import Window
 from raster_pipeline import source_assets
 from offline_policy import local_raster,raster_driver
 from contextlib import ExitStack
 with ExitStack() as contexts:
  sources=[contexts.enter_context(rasterio.open(local_raster(p),**({'driver':raster_driver(p)} if access['mode'] in ('local_path','download') else {}))) for _,p in source_assets(access)]
  if not sources:raise ValueError('No source raster configured')
  first=sources[0]
  if all(s.crs==first.crs for s in sources):first=min(sources,key=lambda s:abs(s.transform.determinant))
  if not first.crs:raise ValueError('Object detection requires a georeferenced image')
  window=Window(0,0,first.width,first.height)
  if geometry and geometry.get('type')!='Point':
   window=geometry_window(first,[transform_geom('EPSG:4326',first.crs,geometry)]).intersection(window)
  if first.transform.b or first.transform.d:raise ValueError('Rotated raster grids require an explicit north-up source preparation step before tiled detection')
  return {'width':int(window.width),'height':int(window.height),'crs':first.crs,'transform':first.window_transform(window)},all(all(t=='uint8' for t in s.dtypes) for s in sources)

def tile_windows(width,height,size=1024,overlap=128):
 """Bounded overlapping windows, including every edge pixel without tiny slivers."""
 from rasterio.windows import Window
 def offsets(length):
  result=list(range(0,max(1,length-size+1),size-overlap))
  if result[-1]+size<length:result.append(max(0,length-size))
  return result
 xs,ys=offsets(width),offsets(height)
 if len(xs)*len(ys)>4096:raise ValueError('Detection area exceeds 4096 native tiles. Choose a smaller AOI; the image will not be silently downsampled.')
 return [Window(x,y,min(size,width-x),min(size,height-y)) for y in ys for x in xs]

def display_rgb(stack,stretch):
 rgb=np.zeros((*stack.valid.shape,3),'uint8')
 for k,s in enumerate(stretch):
  values=stack.pixels[stack.bands.index(s['band'])]
  rgb[:,:,k]=(255*np.clip((np.nan_to_num(values)-s['low'])/max(s['high']-s['low'],1e-8),0,1)).astype('uint8')
 rgb[~stack.valid]=0
 return rgb

def detection_stretch(stack,is_byte):
 bands=['RED','GREEN','BLUE']
 if any(b not in stack.bands for b in bands):raise ValueError('Map RED, GREEN and BLUE science channels before object detection')
 stretch=[]
 for band in bands:
  values=stack.pixels[stack.bands.index(band)]
  lo,hi=(0,255) if is_byte else np.percentile(values[stack.valid],[2,98])
  stretch.append({'band':band,'low':float(lo),'high':float(hi)})
 return stretch

def execute(task,data,reference,image,access,persist,support=None,candidate=None):
 start=time.monotonic();folder=Path(data)/'segmentations'/task['id'];folder.mkdir(parents=True,exist_ok=True)
 def progress(value,stage):task.update(status='running',progress=value,stage=stage);persist(task)
 try:
  progress(2,'Reading mapped source RGB bands')
  grid,is_byte=native_grid(access,task.get('geometry'))
  windows=tile_windows(grid['width'],grid['height'])
  stack=read_stack(access,image,max_size=1024,geometry=task.get('geometry'))
  if not stack.valid.any():raise ValueError('No valid source pixels in the selected area')
  bands=['RED','GREEN','BLUE'];stretch=detection_stretch(stack,is_byte)
  rgb=display_rgb(stack,stretch);Image.fromarray(rgb).save(folder/'source.png')
  from detection_experiments import CLASSES as AIRCRAFT_CLASSES, MODEL_ID as AIRCRAFT_ID, load as load_aircraft
  research=task.get('model')==AIRCRAFT_ID
  classify=None
  if research:
   progress(15,'Verifying frozen aircraft references and loading the local encoder')
   if support:
    from detection_experiments import fit_candidate
    classify,configured_classes,aircraft_provenance=fit_candidate(data,support,access,folder,progress)
    task['candidate_fit']=True
   elif candidate:
    from detection_experiments import load_candidate
    candidate_path,candidate_hash=candidate
    classify,configured_classes,aircraft_provenance=load_candidate(data,candidate_path,candidate_hash,access)
   else:
    classify,aircraft_provenance=load_aircraft(data)
    configured_classes=list(AIRCRAFT_CLASSES)
  infer,merge,provenance=load_detector(progress);detections=[]
  if research:provenance['fewshot']=aircraft_provenance
  from dataclasses import replace
  from affine import Affine
  for index,window in enumerate(windows):
   progress(30+55*index/len(windows),f'Detecting native-resolution tile {index+1} of {len(windows)}')
   profile={**grid,'width':int(window.width),'height':int(window.height),'transform':grid['transform']*Affine.translation(window.col_off,window.row_off)}
   tile=read_stack(access,image,geometry=task.get('geometry'),target_profile=profile,verify_sources=False)
   if not tile.valid.any():continue
   rgb_tile=display_rgb(tile,stretch)
   proposals=infer(rgb_tile)
   if classify:proposals=classify(rgb_tile,proposals,None if support or candidate else CLASSES.index('plane')+1)
   for detection in proposals:
    box=detection['rbox'];col,row=int(np.floor(box.cx)),int(np.floor(box.cy))
    if not 0<=row<tile.valid.shape[0] or not 0<=col<tile.valid.shape[1] or not tile.valid[row,col]:continue
    detections.append({**detection,'rbox':replace(box,cx=box.cx+window.col_off,cy=box.cy+window.row_off)})
   if len(detections)>10000:raise ValueError('More than 10,000 overlapping candidates. Select a smaller AOI before global oriented-box suppression.')
  progress(87,'Removing duplicate detections between overlapping tiles')
  detections=merge(detections);provenance['tiling']={'native_resolution':True,'tiles':len(windows),'size':1024,'overlap':128,'input_preview_only':True}
  progress(90,'Saving oriented boxes and source-image review')
  classes=configured_classes if research else CLASSES
  features=[];preview=Image.fromarray(rgb);draw=ImageDraw.Draw(preview)
  for detection in detections:
   box=detection['rbox'];pixel=list(box.corners());pixel.append(pixel[0]);world=[grid['transform']*p for p in pixel];geometry=transform_geom(grid['crs'],'EPSG:4326',{'type':'Polygon','coordinates':[world]})
   label=detection['subtype'] if research else CLASSES[detection['label']-1]
   properties={'class_id':classes.index(label)+1,'label':label,'class_name':label,'score':float(detection['score']),
               'model':AIRCRAFT_ID if research else MODEL_ID,'prediction_not_ground_truth':True}
   if research:
    properties.update(subtype_margin=float(detection['subtype_margin']),pixel_corners=pixel[:-1])
    if support or candidate:properties.update(baseline_class=detection['baseline_class'],baseline_margin=float(detection['baseline_margin']))
   feature={'type':'Feature','geometry':geometry,'properties':properties,'id':len(features)};features.append(feature)
   review=[~stack.profile['transform']*p for p in world];draw.line(review,fill='#49f4ba',width=2)
   draw.text(review[0],label if research else label+' '+str(round(100*detection['score']))+'%',fill='#ffff40')
  preview.save(folder/'detections.png');(folder/'detections.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features},separators=(',',':')))
  source_profile={k:(list(v) if k=='transform' else str(v) if k=='crs' else v) for k,v in grid.items() if k in ('crs','transform','width','height')}
  task.update(status='complete',progress=100,stage=f'{len(features)} object detections saved',detections=features,preview=f'/api/segmentations/{task["id"]}/detections.png',geojson=f'/api/segmentations/{task["id"]}/detections.geojson',elapsed_seconds=round(time.monotonic()-start,2),provenance=provenance,classes=classes,grid=source_profile,display={'rgb_bands':bands,'stretch':stretch,'excluded_science_bands':[b for b in stack.bands if b not in bands]},validation={'status':'unmeasured','reason':'No independent object-detection validation dataset evaluated for this run'})
  persist(task);return task
 except Exception as exc:task.update(status='failed',stage='Object detection needs attention',error=str(exc));persist(task);return task
