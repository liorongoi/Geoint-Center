"""Task-specific pretrained pixel classifiers, using their native class heads."""
from __future__ import annotations

from execution_policy import torch_device, configure_tensorflow
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time
import httpx
from offline_policy import require_model_download as require_online
import joblib
import numpy as np

DW_REVISION='0dd288fae5fb5291d988e5ea3c12eb91df481ec5'
DW_CLASSES=['Water','Trees','Grass','Flooded vegetation','Crops','Shrub and scrub','Built-up','Bare ground','Snow and ice']
DW_BANDS=['BLUE','GREEN','RED','B5','B6','B7','B8','SWIR1','SWIR2']
CROP_CLASSES=['Natural Vegetation','Forest','Corn','Soybeans','Wetlands','Developed/Barren','Open Water','Winter Wheat','Alfalfa','Fallow/Idle Cropland','Cotton','Sorghum','Other']
OEM_CLASSES=['Bareland','Rangeland','Developed space','Road','Tree','Water','Agriculture','Building']
HLS_BANDS=['BLUE','GREEN','RED','NIR','SWIR1','SWIR2']
PRETRAINED={'dynamicworld':{'name':'Dynamic World · pretrained LULC','classes':DW_CLASSES,
    'source':'https://github.com/google/dynamicworld','revision':DW_REVISION,'bands':DW_BANDS,
    'input':'Sentinel-2 L1C TOA reflectance, nine bands, native 10m grid', 'native_resolution':10},
 'prithvi_crop':{'name':'Prithvi crop/LULC · pretrained head','classes':CROP_CLASSES,
    'source':'https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification',
    'bands':HLS_BANDS*3,'input':'HLS 30m, 18 bands: BLUE/GREEN/RED/NIR/SWIR1/SWIR2 for each of three real dates', 'native_resolution':30},
 'openearthmap':{'name':'OpenEarthMap · pretrained U-Net','classes':OEM_CLASSES,'source':'https://open-earth-map.org/code.html',
    'bands':['RED','GREEN','BLUE'],'input':'Very high resolution RGB, 0.25–0.5m, eight land-cover classes'}}
NORM=np.array([[1.7417268007636313,2.023298706048351],[1.7261204997060209,2.038905204308012],
 [1.6798346251414997,2.179592821212937],[1.7734969472909623,2.2890068333026603],
 [2.289154079164943,2.6171674549378166],[2.382939712192371,2.773418590375327],
 [2.3828939530384052,2.7578332604178284],[2.1952484264967844,2.789092484314204],
 [1.554812948247501,2.4140534947492487]],dtype='float32')

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

CROP_REVISION='b53a88b8da673800b67c34a98a527b77076e7035'

def prepare_torch(model_id,cache):
    from model_packs import ensure_model_pack
    pack='openearthmap' if model_id=='openearthmap' else 'prithvi-crop'
    ensure_model_pack(cache/pack,pack)
    if model_id=='openearthmap':
        path=cache/'openearthmap'/'released.pth';path.parent.mkdir(parents=True,exist_ok=True)
        if not path.is_file():
            require_online('Downloading OpenEarthMap weights')
            import gdown
            temp=path.with_suffix('.part');result=gdown.download(id='1j6Os_c4RmiJlJd6EsPTGhJizVlfLeiZv',output=str(temp),quiet=True)
            if not result:raise ValueError('Official OpenEarthMap checkpoint download failed')
            temp.replace(path)
        info={'source':PRETRAINED[model_id]['source'],'weights_url':'https://drive.google.com/file/d/1j6Os_c4RmiJlJd6EsPTGhJizVlfLeiZv/view',
              'sha256':sha256(path),'initialization':'Released pretrained U-Net/EfficientNet-B4 encoder and eight-class semantic head',
              'normalization':'RGB /255; original ninth background channel excluded as in official inference example',
              'implementation':'OpenEarthMap U-Net EfficientNet-B4 with SCSE decoder; all pretrained tensors retained'}
    else:
        from safetensors.torch import save_file
        path=cache/'prithvi-crop'/'model.safetensors';path.parent.mkdir(parents=True,exist_ok=True)
        if not path.is_file():
            import torch
            raw=path.parent/'released.pth'
            if not raw.is_file():
                require_online('Downloading Prithvi crop weights')
                from huggingface_hub import hf_hub_download
                raw=Path(hf_hub_download('ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification',
                    'multi_temporal_crop_classification_Prithvi_100M.pth',revision=CROP_REVISION,cache_dir=path.parent/'hf-cache'))
            safe=[(np._core.multiarray.scalar,'numpy.core.multiarray.scalar'),np.dtype,np.dtypes.Float64DType,np.dtypes.Float32DType,np.dtypes.Int64DType]
            with torch.serialization.safe_globals(safe):state=torch.load(raw,map_location='cpu',weights_only=True)['state_dict']
            save_file({k:v.contiguous() for k,v in state.items()},str(path))
        info={'source':PRETRAINED[model_id]['source'],'revision':CROP_REVISION,'sha256':sha256(path),
              'initialization':'Released pretrained Prithvi encoder, temporal neck and 13-class LULC/crop head',
              'normalization':'Official 18-band mean/std and six-band × three-frame reshape; 224px windows',
              'implementation':'NASA-IMPACT TemporalViTEncoder, pretrained temporal neck and FCN heads; all tensors retained'}
    return path,info

@lru_cache(maxsize=1)
def loaded_torch(model_id,path):
    import torch
    torch.set_num_threads(2)
    if model_id=='openearthmap':
        import segmentation_models_pytorch as smp
        model=smp.Unet(encoder_name='efficientnet-b4',encoder_weights=None,in_channels=3,classes=9,decoder_attention_type='scse')
        state=torch.load(path,map_location='cpu',weights_only=True)
    else:
        from pretrained_networks import PrithviCrop
        from safetensors.torch import load_file
        model=PrithviCrop();state=load_file(path)
    model.load_state_dict(state,strict=True);return model.to(torch_device()).eval()

def normalize_torch(model_id,stack):
    if model_id=='openearthmap':
        pixels=stack.pixels[[stack.bands.index(b) for b in ['RED','GREEN','BLUE']]].copy()
        if np.nanpercentile(pixels,99)>1.5:pixels/=255
        return np.clip(np.nan_to_num(pixels),0,1)
    mean=np.tile(np.array([494.905781,815.239594,924.335066,2968.881459,2634.621962,1739.579917],dtype='float32'),3)
    std=np.tile(np.array([284.925432,357.84876,575.566823,896.601013,951.900334,921.407808],dtype='float32'),3)
    return np.nan_to_num((stack.pixels*10000-mean[:,None,None])/std[:,None,None])

def torch_logits(model_id,model,crop,legacy_prithvi_layout=False):
    import torch
    if model_id=='prithvi_crop':
        # Raster stacks store all six bands of each date consecutively.
        # Conv3d expects [batch, band, time, row, column].
        temporal=crop.reshape(1,6,3,224,224) if legacy_prithvi_layout else crop.reshape(3,6,224,224).transpose(1,0,2,3)[None]
        logits=model(torch.from_numpy(np.ascontiguousarray(temporal)).to(next(model.parameters()).device))
    else:
        logits=model(torch.from_numpy(crop[None]).to(next(model.parameters()).device))[:,1:]
    return logits

def predict_torch(record,stack,progress):
    import torch
    model_id=record['model_id'];model=loaded_torch(model_id,record['model_path']);pixels=normalize_torch(model_id,stack)
    h,w=stack.valid.shape;size=224 if model_id=='prithvi_crop' else 512
    stride=size//2;count=np.zeros((h,w),dtype='float32');logits=np.zeros((len(record['classes']),h,w),dtype='float32')
    legacy_layout=model_id=='prithvi_crop' and not record.get('prithvi_layout') and record.get('initialization','').startswith('Fine-tuned')
    ys=list(range(0,max(1,h-size+1),stride));xs=list(range(0,max(1,w-size+1),stride))
    if h>size and ys[-1]!=h-size:ys.append(h-size)
    if w>size and xs[-1]!=w-size:xs.append(w-size)
    with torch.inference_mode():
        for yi,y in enumerate(ys):
            for x in xs:
                crop=pixels[:,y:y+size,x:x+size];ch,cw=crop.shape[-2:]
                crop=np.pad(crop,((0,0),(0,size-ch),(0,size-cw)),mode='edge')
                result=torch_logits(model_id,model,crop,legacy_layout)[0,:,:ch,:cw].cpu().numpy()
                logits[:,y:y+ch,x:x+cw]+=result;count[y:y+ch,x:x+cw]+=1
            progress('pretrained_inference',round(100*(yi+1)/len(ys)))
    probabilities=torch.from_numpy(logits/np.maximum(count,1)[None]).softmax(0).numpy()
    probabilities[:,~stack.valid]=0;return probabilities

def fine_tune_torch(model_id,samples,classes,destination,cache,epochs,progress,initial_checkpoint):
    import torch
    import torch.nn.functional as F
    from copy import deepcopy
    from safetensors.torch import save_file
    if classes!=PRETRAINED[model_id]['classes']:raise ValueError('Fine-tuning preserves the pretrained native class system')
    path,_=prepare_torch(model_id,cache)
    if initial_checkpoint:
        previous=joblib.load(initial_checkpoint)
        if model_id=='prithvi_crop' and not previous.get('prithvi_layout') and previous.get('initialization','').startswith('Fine-tuned'):
            raise ValueError('This earlier Prithvi candidate used a different temporal layout. Start from the pretrained base and qualify a new version.')
        path=Path(previous['model_path'])
    model=deepcopy(loaded_torch(model_id,str(path)));model.train()
    # Keep BN statistics fixed for sparse, small correction batches.
    for module in model.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
    optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=1e-5)
    size=224 if model_id=='prithvi_crop' else 256;started=time.perf_counter();pixels=0
    for epoch in range(epochs):
        for stack,mask in samples:
            values=normalize_torch(model_id,stack)
            for y in range(0,mask.shape[0],size):
                for x in range(0,mask.shape[1],size):
                    target=mask[y:y+size,x:x+size];ch,cw=target.shape
                    if not (target>=0).any():continue
                    crop=values[:,y:y+size,x:x+size];crop=np.pad(crop,((0,0),(0,size-ch),(0,size-cw)),mode='edge')
                    logits=torch_logits(model_id,model,crop)[:,:,:ch,:cw]
                    loss=F.cross_entropy(logits,torch.from_numpy(target[None].astype('int64')).to(next(model.parameters()).device),ignore_index=-1)
                    optimizer.zero_grad();loss.backward();optimizer.step()
                    if epoch==0:pixels+=int((target>=0).sum())
        progress(f'epoch {epoch+1} loss {float(loss.detach().cpu()):.8g}',round(100*(epoch+1)/epochs))
    target=destination.parent/(destination.stem+('.safetensors' if model_id=='prithvi_crop' else '.pth'))
    state={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
    if model_id=='prithvi_crop':save_file(state,str(target))
    else:torch.save(state,target)
    info=checkpoint(model_id,samples[0][0],destination,cache);record=joblib.load(destination)
    record.update(model_path=str(target),initialization='Fine-tuned released pretrained pixel classifier; separate local version')
    joblib.dump(record,destination)
    return {**info,'initialization':record['initialization'],'training_pixels':pixels,'epochs':epochs,'training_seconds':round(time.perf_counter()-started,3)}

def prepare_dynamicworld(cache):
    from model_packs import ensure_model_pack
    ensure_model_pack(cache/'dynamicworld','dynamicworld')
    root=cache/'dynamicworld'/DW_REVISION;manifest=[]
    for name in ['saved_model.pb','variables/variables.data-00000-of-00001','variables/variables.index']:
        path=root/'forward'/name;url=f'https://raw.githubusercontent.com/google/dynamicworld/{DW_REVISION}/model/forward/{name}'
        if not path.is_file():
            require_online('Downloading Dynamic World weights')
            path.parent.mkdir(parents=True,exist_ok=True)
            response=httpx.get(url,timeout=90,follow_redirects=True);response.raise_for_status()
            temp=path.with_suffix(path.suffix+'.part');temp.write_bytes(response.content);temp.replace(path)
        manifest.append({'file':name,'source':url,'sha256':sha256(path)})
    info={'source':'https://github.com/google/dynamicworld','revision':DW_REVISION,'files':manifest,
          'initialization':'Official pretrained LULC encoder and nine-class head; no local fitting',
          'license':'Apache-2.0','normalization':'Official log/sigmoid transform of Sentinel-2 L1C DN (reflectance ×10000)',
          'classes':DW_CLASSES}
    (root/'provenance.json').write_text(json.dumps(info,indent=2));return root/'forward',info

def validate_input(model_id,stack,image,access):
    spec=PRETRAINED[model_id];missing=[b for b in spec['bands'] if b not in stack.bands]
    if missing:raise ValueError(f'{spec["name"]}: missing bands {", ".join(missing)}. Required input: {spec["input"]}')
    level=str(access.get('processing_level') or image.get('processing_level') or '').upper()
    if model_id=='dynamicworld' and level not in ('L1C','LEVEL-1C') and image.get('catalog')!='sentinel-2-l1c':
        raise ValueError('Dynamic World requires confirmed Sentinel-2 L1C TOA input. L2A surface reflectance and Landsat are incompatible; choose the L1C catalog or declare the actual local product level.')
    if model_id=='prithvi_crop':
        if level!='HLS' or stack.bands!=HLS_BANDS*3:raise ValueError('Prithvi crop requires the actual 18-band HLS stack in six-band-per-date order, not repeated copies of one image')
        dates=access.get('temporal_dates') or image.get('temporal_dates') or []
        if len(dates)!=3 or len(set(dates))!=3:raise ValueError('Provide three distinct actual acquisition dates in Raster access for the temporal HLS stack')
    transform=stack.profile['transform']
    if not stack.profile['crs'].is_projected:raise ValueError('Pretrained input needs a projected CRS with metre units')
    factor=stack.profile['crs'].linear_units_factor[1]
    resolution=[abs(transform.a)*factor,abs(transform.e)*factor]
    if model_id=='openearthmap':
        if not all(.24<=v<=.51 for v in resolution):raise ValueError('OpenEarthMap requires native 0.25–0.5m RGB imagery; Sentinel/Landsat pixels are too coarse')
        if np.nanpercentile(np.abs(stack.pixels),99)>255:raise ValueError('OpenEarthMap expects RGB intensities in 0..255 or normalized 0..1')
        return
    target=spec['native_resolution']
    if not np.allclose(resolution,[target,target],atol=.05):raise ValueError(f'{spec["name"]} requires a native {target}m grid. Reduce the AOI or increase the analysis grid.')
    if np.nanpercentile(np.abs(stack.pixels),99)>3:raise ValueError('The source appears to contain encoded digital numbers rather than reflectance. For a local Sentinel SAFE product, import MTD_MSIL1C.xml with its bands, or attach the matching supplier metadata in Source data. Do not change the analysis grid to fix calibration.')

def checkpoint(model_id,stack,destination,cache,previous=None):
    start=time.perf_counter()
    if previous and Path(previous).is_file():
        record=joblib.load(previous)
        if record.get('model_id')==model_id and record.get('bands')==stack.bands:
            joblib.dump(record,destination)
            return {'path':str(destination),'initialization':'Saved compatible job model version','implementation':PRETRAINED[model_id]['name'],'training_seconds':0,'continued_from':str(previous)}
    if model_id=='dynamicworld':model,info=prepare_dynamicworld(cache)
    else:model,info=prepare_torch(model_id,cache)
    record={'model_id':model_id,'bands':stack.bands,'classes':PRETRAINED[model_id]['classes'],
            'model_path':str(model),'implementation':PRETRAINED[model_id]['name'],**info}
    if model_id=='prithvi_crop':record['prithvi_layout']='date-major-stack-to-band-time-v1'
    destination.parent.mkdir(parents=True,exist_ok=True);joblib.dump(record,destination)
    return {'path':str(destination),'training_seconds':0,'preparation_seconds':round(time.perf_counter()-start,3),
            'implementation':record['implementation'],**info}

def tf_runtime():
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
    return configure_tensorflow()

@lru_cache(maxsize=2)
def loaded_tf(path):return tf_runtime().saved_model.load(path)

def normalize_dw(stack):
    pixels=np.moveaxis(stack.pixels[[stack.bands.index(b) for b in DW_BANDS]],0,-1)
    # The app reader already converted integer DN to physical reflectance.
    transformed=(np.log1p(np.maximum(np.nan_to_num(pixels),0)*50)-NORM[:,0])/NORM[:,1]
    return (1/(1+np.exp(-np.clip(transformed*5-1,-60,60)))).astype('float32')

def predict(record,stack,progress):
    if record['model_id']!='dynamicworld':return predict_torch(record,stack,progress)
    tf=tf_runtime();model=loaded_tf(record['model_path']);values=normalize_dw(stack)
    h,w=stack.valid.shape;probabilities=np.zeros((9,h,w),dtype='float32')
    # Native-resolution tiles with context; trim context to reduce boundary seams.
    for y in range(0,h,256):
        for x in range(0,w,256):
            y0,x0=max(0,y-32),max(0,x-32);y1,x1=min(h,y+288),min(w,x+288)
            tile=values[y0:y1,x0:x1];th,tw=tile.shape[:2]
            tile=np.pad(tile,((0,(-th)%4),(0,(-tw)%4),(0,0)),mode='edge')
            logits=model(tf.convert_to_tensor(tile[None]));prob=tf.nn.softmax(logits,axis=-1).numpy()[0]
            hh,ww=min(256,h-y),min(256,w-x)
            probabilities[:,y:y+hh,x:x+ww]=np.moveaxis(prob[y-y0:y-y0+hh,x-x0:x-x0+ww],-1,0)
        progress('pretrained_inference',min(100,round(100*(y+256)/h)))
    probabilities[:,~stack.valid]=0
    return probabilities

def fine_tune(model_id,samples,classes,destination,cache,epochs,progress,initial_checkpoint=None):
    if model_id!='dynamicworld':return fine_tune_torch(model_id,samples,classes,destination,cache,epochs,progress,initial_checkpoint)
    if classes!=DW_CLASSES:raise ValueError('Pretrained fine-tuning preserves the native nine-class taxonomy')
    tf=tf_runtime();source,_=prepare_dynamicworld(cache)
    if initial_checkpoint:source=Path(joblib.load(initial_checkpoint)['model_path'])
    # Load a fresh instance: never mutate the cached original pretrained model.
    model=tf.saved_model.load(str(source));optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5)
    start=time.perf_counter();pixels=0
    for epoch in range(epochs):
        for stack,mask in samples:
            values=normalize_dw(stack);h,w=mask.shape
            for y in range(0,h,224):
                for x in range(0,w,224):
                    target=mask[y:y+224,x:x+224];keep=target>=0
                    if not keep.any():continue
                    crop=values[y:y+224,x:x+224];ch,cw=crop.shape[:2]
                    crop=np.pad(crop,((0,(-ch)%4),(0,(-cw)%4),(0,0)),mode='edge')
                    with tf.GradientTape() as tape:
                        logits=model(tf.convert_to_tensor(crop[None]))[0,:ch,:cw]
                        loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target[keep],logits=tf.boolean_mask(logits,keep)))
                    gradients=tape.gradient(loss,model.trainable_variables)
                    pairs=[(g,v) for g,v in zip(gradients,model.trainable_variables) if g is not None]
                    if not pairs:raise ValueError('Released SavedModel did not expose trainable gradients')
                    optimizer.apply_gradients(pairs)
                    if epoch==0:pixels+=int(keep.sum())
        progress(f'epoch {epoch+1} loss {float(loss.numpy()):.8g}',round(100*(epoch+1)/epochs))
    target=destination.parent/(destination.stem+'-savedmodel');tf.saved_model.save(model,str(target))
    info=checkpoint(model_id,samples[0][0],destination,cache)
    record=joblib.load(destination);record.update(model_path=str(target),initialization='Fine-tuned official Dynamic World encoder and head; separate local version')
    joblib.dump(record,destination)
    return {**info,'initialization':record['initialization'],'training_pixels':pixels,'epochs':epochs,'training_seconds':round(time.perf_counter()-start,3)}
