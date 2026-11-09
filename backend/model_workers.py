"""Executable LULC workers; every prediction and metric comes from source pixels."""
from __future__ import annotations

from execution_policy import torch_device

import importlib.util
import json
import os
import time
import shutil
import hashlib
from pathlib import Path
from typing import Callable

import joblib
import numpy as np

SEED = 42
PRITHVI_REPO = 'ibm-nasa-geospatial/Prithvi-EO-2.0-300M'
PRITHVI_REVISION = '9eb1b1102806593963daa333bcc491b1c6f8562f'
PRITHVI_BANDS = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']
ENSEMBLE_MEMBERS = ('random_forest','svm','xgboost','unet','prithvi')

def build_ensemble(checkpoints: dict, destination: Path, classes: list[str], bands: list[str]) -> dict:
    """Bundle fitted members and fixed equal weights; never read validation/test labels."""
    start=time.perf_counter();members=[]
    folder=destination.parent/(destination.stem+'-members');folder.mkdir(parents=True,exist_ok=True)
    for model_id in ENSEMBLE_MEMBERS:
        if model_id not in checkpoints:continue
        source=Path(checkpoints[model_id]);record=joblib.load(source)
        if record['classes']!=classes or record['bands']!=bands:
            raise ValueError(f'Ensemble member {model_id} has incompatible classes or bands')
        target=folder/f'{model_id}.joblib'
        if source.resolve()!=target.resolve():shutil.copy2(source,target)
        members.append({'id':model_id,'path':str(target.relative_to(destination.parent)),
                        'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                        'implementation':record['implementation']})
    if len(members)<2:raise ValueError('Ensemble requires at least two compatible successful members')
    for member in members:member['weight']=1/len(members)
    implementation='Equal probability soft voting; weights fixed before validation/test evaluation'
    joblib.dump({'model_id':'ensemble','classes':classes,'bands':bands,'members':members,
                 'implementation':implementation,'seed':SEED},destination)
    return {'model_id':'ensemble','path':str(destination),'training_seconds':round(time.perf_counter()-start,3),
            'implementation':implementation,'members':members,'training_mode':'refit_members',
            'initialization':'Derived classifier using saved member models; fixed equal weights',
            'note':'Member models refit or continue on approved training references; Prithvi encoder remains frozen'}


def score_predictions(truth: np.ndarray, predicted: np.ndarray, classes: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, jaccard_score, classification_report
    keep = truth >= 0
    if not keep.any():
        return {'status': 'unmeasured', 'reason': 'No independent validation labels intersect this image'}
    y, p = truth[keep], predicted[keep]
    ids = list(range(len(classes)))
    per_class=classification_report(y,p,labels=ids,target_names=classes,output_dict=True,zero_division=0)
    return {'status': 'measured', 'pixels': int(len(y)), 'accuracy': float(accuracy_score(y, p)),
            'macro_precision':float(per_class['macro avg']['precision']),
            'macro_recall':float(per_class['macro avg']['recall']),
            'macro_f1': float(f1_score(y, p, labels=ids, average='macro', zero_division=0)),
            'mean_iou': float(jaccard_score(y, p, labels=ids, average='macro', zero_division=0)),
            'confusion_matrix': confusion_matrix(y, p, labels=ids).tolist(),
            'per_class': per_class}


def prepare_prithvi(cache: Path) -> dict:
    from model_packs import ensure_model_pack
    ensure_model_pack(cache,'prithvi')
    from huggingface_hub import hf_hub_download
    cache.mkdir(parents=True, exist_ok=True)
    filenames = ['prithvi_mae.py', 'config.json', 'Prithvi_EO_V2_300M.pt']
    from offline_policy import models_local_only as enabled
    files = {name: hf_hub_download(PRITHVI_REPO, name, revision=PRITHVI_REVISION, cache_dir=str(cache), local_files_only=enabled()) for name in filenames}
    return {'repository': PRITHVI_REPO, 'revision': PRITHVI_REVISION, 'files': files}


def _prithvi_encoder(cache: Path):
    import torch
    info = prepare_prithvi(cache)
    spec = importlib.util.spec_from_file_location('spectrascope_prithvi', info['files']['prithvi_mae.py'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads(Path(info['files']['config.json']).read_text())['pretrained_cfg']
    # Only an encoder is needed for downstream supervised pixel classification.
    encoder = module.PrithviViT(**{**config, 'num_frames': 1}).eval()
    weights = torch.load(info['files']['Prithvi_EO_V2_300M.pt'], map_location='cpu', weights_only=True, mmap=True)
    weights = weights.get('state_dict', weights)
    state = {key.removeprefix('encoder.'): value for key, value in weights.items() if key.startswith('encoder.')}
    if not state:
        state = {key: value for key, value in weights.items() if key in encoder.state_dict()}
    state.pop('pos_embed', None)  # recomputed sinusoidal positions for one frame
    result = encoder.load_state_dict(state, strict=False)
    if set(result.missing_keys) - {'pos_embed'} or result.unexpected_keys:
        raise ValueError(f'Pretrained encoder mismatch: {result}')
    return encoder, config, info


def prithvi_features(pixels: np.ndarray, bands: list[str], cache: Path, progress: Callable = lambda *a: None) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    missing = [name for name in PRITHVI_BANDS if name not in bands]
    if missing:
        raise ValueError('Prithvi requires mapped reflectance bands: ' + ', '.join(missing) + '. Set band names before running.')
    values = pixels[[bands.index(name) for name in PRITHVI_BANDS]].copy()
    if np.nanpercentile(np.abs(values), 99) > 3:
        raise ValueError('Prithvi inputs must be reflectance units; configure raster scale/offset before running')
    encoder, config, info = _prithvi_encoder(cache)
    device = torch_device(); encoder = encoder.to(device)
    torch.set_num_threads(max(1, int(os.getenv('MODEL_THREADS', '2'))))
    _, height, width = values.shape
    output = np.zeros((38, height, width), dtype='float32')
    output[:6] = values
    mean = torch.tensor(config['mean'], device=device).view(1,6,1,1) / 10000
    std = torch.tensor(config['std'], device=device).view(1,6,1,1) / 10000
    with torch.inference_mode():
        for y in range(0, height, 64):
            for x in range(0, width, 64):
                tile = np.nan_to_num(values[:,y:y+64,x:x+64])
                h,w = tile.shape[-2:]
                tensor = torch.from_numpy(tile[None]).to(device)
                tensor = F.pad(tensor, (0, (16-w%16)%16, 0, (16-h%16)%16), mode='replicate')
                tensor = (tensor - mean) / std
                # forward with zero masking retains every patch token in raster order.
                # forward_features avoids MAE random-token shuffling, even at mask_ratio=0.
                features = encoder.forward_features(tensor)[-1][:,1:]
                hp,wp = tensor.shape[-2]//16, tensor.shape[-1]//16
                # Fixed channel pooling preserves spatial pretrained features without
                # fitting a projection on held-out labels (1024 -> 32 channels).
                features = features.reshape(1,hp,wp,32,32).mean(-1).permute(0,3,1,2)
                features = F.interpolate(features, size=tensor.shape[-2:], mode='bilinear', align_corners=False)
                output[6:,y:y+h,x:x+w] = features[0,:,:h,:w].cpu().numpy()
            progress('foundation_features', min(95, int(100*(y+64)/height)))
    del encoder
    return output


def _unet(channels: int, classes: int):
    import torch
    from torch import nn
    import torch.nn.functional as F
    def block(a,b):
        return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.ReLU())
    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.down1=block(channels,16); self.down2=block(16,32); self.bridge=block(32,64)
            self.up2=block(96,32); self.up1=block(48,16); self.output=nn.Conv2d(16,classes,1)
        def forward(self,x):
            a=self.down1(x); b=self.down2(F.max_pool2d(a,2)); c=self.bridge(F.max_pool2d(b,2))
            d=self.up2(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
            return self.output(self.up1(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1)))
    return UNet()


def _balanced_samples(pixels, labels, cap=12000):
    features = np.moveaxis(pixels,0,-1).reshape(-1,pixels.shape[0])
    target = labels.ravel()
    rng = np.random.default_rng(SEED)
    classes = np.unique(target[target>=0])
    if not len(classes):
        raise ValueError('Training sample has no labeled pixels')
    indices=[]
    for value in classes:
        choices=np.flatnonzero((target==value)&np.all(np.isfinite(features),axis=1))
        if not len(choices):
            continue
        indices.extend(rng.choice(choices,min(len(choices),max(5,cap//len(classes))),replace=False).tolist())
    return features[indices],target[indices]


def train_worker(model_id: str, samples: list[tuple], classes: list[str], destination: Path, cache: Path,
                 epochs: int = 20, progress: Callable = lambda *a: None, initial_checkpoint: Path | None = None,
                 member_checkpoints: dict | None = None, forest_spec: dict | None = None) -> dict:
    """Train on explicit training masks. Held-out masks never enter this function."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    start=time.perf_counter()
    from pretrained_lulc import PRETRAINED, fine_tune
    if model_id in PRETRAINED:
        return fine_tune(model_id,samples,classes,destination,cache.parent,epochs,progress,initial_checkpoint)
    bands=samples[0][0].bands
    if model_id=='ensemble':
        checkpoints=dict(member_checkpoints or {});failures={}
        if member_checkpoints is None:
            previous=joblib.load(initial_checkpoint) if initial_checkpoint and initial_checkpoint.is_file() else {}
            old={item['id']:initial_checkpoint.parent/item['path'] for item in previous.get('members',[])}
            for index,member in enumerate(ENSEMBLE_MEMBERS):
                try:
                    target=destination.parent/(destination.stem+'-members')/f'{member}.joblib'
                    train_worker(member,samples,classes,target,cache,epochs,
                        lambda phase,value:progress(f'{member} {phase}',int(100*(index+value/100)/len(ENSEMBLE_MEMBERS))),old.get(member))
                    checkpoints[member]=target
                except Exception as exc:failures[member]=str(exc)
        result=build_ensemble(checkpoints,destination,classes,bands)
        result.update(training_seconds=round(time.perf_counter()-start,3),skipped_members=failures)
        progress('ensemble_ready',100)
        return result
    if any(stack.bands != bands for stack,_ in samples):
        raise ValueError('Training images have different band schemas. Use separate jobs or matching band maps.')
    transformed=[]
    for stack,mask in samples:
        pixels=prithvi_features(stack.pixels,bands,cache,progress) if model_id=='prithvi' else stack.pixels
        transformed.append((pixels,mask))
    pairs=[_balanced_samples(pixels,mask) for pixels,mask in transformed]
    X=np.concatenate([pair[0] for pair in pairs]); y=np.concatenate([pair[1] for pair in pairs])
    if len(np.unique(y))<2:
        raise ValueError('Training pixels across all images must contain at least two classes')
    if set(np.unique(y)) != set(range(len(classes))):
        raise ValueError('Each class needs valid training pixels')
    if any(np.count_nonzero(y==value)<5 for value in range(len(classes))):
        raise ValueError('Each class needs at least five valid training pixels across the training images')
    record={'model_id':model_id,'bands':bands,'classes':classes,'seed':SEED,'training_pixels':int(len(y)),
            'initialization':'Official pretrained encoder frozen; new supervised classifier head' if model_id=='prithvi' else 'Random initialization; trained from scratch' if model_id=='unet' else 'Fitted from training labels; no pretrained estimator',
            'implementation':'Prithvi-EO-2.0-300M frozen encoder + supervised RF head' if model_id=='prithvi' else model_id}
    if model_id in ('random_forest','prithvi'):
        trees,forest_seed=128,SEED
        if forest_spec:
            if model_id!='random_forest' or set(forest_spec)!={'trees','seed','shard','total_shards'} or any(type(v)!=int for v in forest_spec.values()) or not 1<=forest_spec['trees']<=128 or not 0<=forest_spec['shard']<forest_spec['total_shards']<=8:raise ValueError('Invalid typed Random Forest shard configuration')
            trees,forest_seed=forest_spec['trees'],forest_spec['seed'];record['distributed_forest']=forest_spec
        record['training_data_sha256']=hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest();record['sampling_seed']=SEED
        model=RandomForestClassifier(n_estimators=trees,max_depth=24,class_weight='balanced',random_state=forest_seed,n_jobs=2)
    elif model_id=='svm':
        if len(y)>6000:
            rng=np.random.default_rng(SEED);indices=[]
            for value in np.unique(y):
                choices=np.flatnonzero(y==value)
                indices.extend(rng.choice(choices,min(len(choices),6000//len(classes)),replace=False).tolist())
            X=X[indices];y=y[indices]
        model=make_pipeline(StandardScaler(),SVC(C=10,gamma='scale',probability=True,class_weight='balanced',random_state=SEED))
    elif model_id=='xgboost':
        from xgboost import XGBClassifier
        model=XGBClassifier(n_estimators=150,max_depth=6,learning_rate=.08,subsample=.9,colsample_bytree=.9,tree_method='hist',random_state=SEED,n_jobs=2)
    elif model_id=='unet':
        import torch
        import torch.nn.functional as F
        torch.manual_seed(SEED);torch.set_num_threads(2)
        mean=X.mean(0);std=np.maximum(X.std(0),1e-6)
        model=_unet(len(bands),len(classes)).to(torch_device())
        if initial_checkpoint and initial_checkpoint.is_file():
            previous=joblib.load(initial_checkpoint)
            if previous.get('bands')==bands and previous.get('classes')==classes and 'state_dict' in previous:
                model.load_state_dict(previous['state_dict']);record['continued_from']=str(initial_checkpoint)
                record['initialization']='Continued compatible local U-Net checkpoint'
        optimizer=torch.optim.Adam(model.parameters(),lr=.002)
        for epoch in range(epochs):
            for pixels,mask in transformed:
                normalized=np.nan_to_num((pixels-mean[:,None,None])/std[:,None,None])
                logits=model(torch.from_numpy(normalized[None].astype('float32')).to(torch_device()))
                loss=F.cross_entropy(logits,torch.from_numpy(mask[None].astype('int64')).to(torch_device()),ignore_index=-1)
                optimizer.zero_grad();loss.backward();optimizer.step()
            point={'epoch':epoch+1,'loss':float(loss.detach().cpu()),'scope':'last labeled training batch'}
            record.setdefault('learning_curve',[]).append(point)
            progress(f'epoch {epoch+1} loss {point["loss"]:.8g}',int(100*(epoch+1)/epochs))
        record.update({'state_dict':{key:value.detach().cpu() for key,value in model.state_dict().items()},
                       'mean':mean,'std':std,'epochs':epochs,'final_training_loss':float(loss.detach())})
        model=None
    else:
        raise ValueError(f'Unsupported model {model_id}')
    if model is not None:
        model.fit(X,y);record['estimator']=model
    if model_id=='prithvi':
        record.update({'pretrained_repository':PRITHVI_REPO,'pretrained_revision':PRITHVI_REVISION,'fine_tuning_scope':'downstream RF head; foundation encoder frozen'})
    destination.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(record,destination)
    return {'model_id':model_id,'path':str(destination),'training_seconds':round(time.perf_counter()-start,3),
            'training_pixels':record['training_pixels'],'implementation':record['implementation'],'initialization':record['initialization'],'learning_curve':record.get('learning_curve',[])}


def predict_worker(checkpoint: Path, stack, cache: Path, progress: Callable = lambda *a: None) -> tuple[np.ndarray,np.ndarray,dict]:
    start=time.perf_counter()
    record=joblib.load(checkpoint)
    if record['bands']!=stack.bands:
        raise ValueError('Image band schema differs from the trained model. Retrain or fix band mapping.')
    model_id=record['model_id']
    from pretrained_lulc import PRETRAINED, predict as pretrained_predict
    if model_id in PRETRAINED:
        probabilities=pretrained_predict(record,stack,progress)
    elif model_id=='ensemble':
        probabilities=None
        for index,member in enumerate(record['members']):
            _,prob,_=predict_worker(checkpoint.parent/member['path'],stack,cache,
                lambda stage,value:progress(f'{member["id"]} {stage}',int(100*(index+value/100)/len(record['members']))))
            probabilities=prob*member['weight'] if probabilities is None else probabilities+prob*member['weight']
            progress('ensemble_member',int(100*(index+1)/len(record['members'])))
    elif model_id=='unet':
        import torch
        model=_unet(len(stack.bands),len(record['classes'])).to(torch_device());model.load_state_dict(record['state_dict']);model.eval()
        pixels=np.nan_to_num((stack.pixels-record['mean'][:,None,None])/record['std'][:,None,None])
        with torch.inference_mode():
            probabilities=model(torch.from_numpy(pixels[None].astype('float32')).to(torch_device())).softmax(1)[0].cpu().numpy()
    else:
        pixels=prithvi_features(stack.pixels,stack.bands,cache,progress) if model_id=='prithvi' else stack.pixels
        features=np.nan_to_num(np.moveaxis(pixels,0,-1).reshape(-1,pixels.shape[0]))
        chunks=[record['estimator'].predict_proba(features[i:i+8192]) for i in range(0,len(features),8192)]
        probabilities=np.concatenate(chunks).T.reshape(len(record['classes']),*stack.valid.shape).astype('float32')
    predicted=probabilities.argmax(0).astype('int32')
    predicted[~stack.valid]=-1;probabilities[:,~stack.valid]=0
    return predicted,probabilities,{'prediction_seconds':round(time.perf_counter()-start,3),'implementation':record['implementation']}


def consensus_maps(probabilities: list[np.ndarray], valid: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    if not probabilities:
        raise ValueError('No model predictions to combine')
    labels=np.stack([prob.argmax(0) for prob in probabilities])
    disagreement=np.any(labels!=labels[0],axis=0)&valid
    mean=np.mean(probabilities,axis=0)
    result=mean.argmax(0).astype('int32');result[~valid]=-1
    confidence=mean.max(0);confidence[~valid]=0
    return result,disagreement,confidence
