"""Local-only SAM 3 Tracker ONNX inference. No account, telemetry or downloads."""
from functools import lru_cache
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import threading
import numpy as np
from PIL import Image

REVISION='429305c8a5b3de597243d919a07e4e6bdcd00ef7'
REPOSITORY='onnx-community/sam3-tracker-ONNX'
FILES={
 'onnx/vision_encoder_quantized.onnx':'4446e8b0b1c3b3a08887a1ce98759ae5cef9fc45a61751f4dc42f9dbe76b43c4',
 'onnx/vision_encoder_quantized.onnx_data':'fd98c9a56d588722ec676f7db67dd7f714ea90bd804b5efc011fd0a3284042e1',
 'onnx/prompt_encoder_mask_decoder.onnx':'4f9ac85291d634ae36a21ce940e3c09671cc05b6511966e5d3d96988b12b95f8',
 'onnx/prompt_encoder_mask_decoder.onnx_data':'2d870726d484cb496760fd139c21f115cf1b945c6b69583489faa2ac79f1d2ae'}
_LOCK=threading.RLock()
_DLL_DIRS=[]


def component_roots():
    roots=[]
    if os.getenv('SPECTRASCOPE_SAM3_COMPONENT'):roots.append(Path(os.environ['SPECTRASCOPE_SAM3_COMPONENT']))
    if os.getenv('SPECTRASCOPE_BUNDLED_MODELS'):roots.append(Path(os.environ['SPECTRASCOPE_BUNDLED_MODELS']).parent/'components/sam3')
    roots.append(Path(os.getenv('SPECTRASCOPE_DATA_DIR',str(Path(__file__).resolve().parent/'data/spectrascope')))/'components/sam3')
    roots += [Path(__file__).resolve().parent.parent/'components/sam3',Path(__file__).resolve().parent/'desktop/payload/mac-arm64/components/sam3']
    return roots


def runtime_available():
    return importlib.util.find_spec('onnxruntime') is not None or any((p/'python/onnxruntime/__init__.py').is_file() for p in component_roots())


def model_source():
    if os.getenv('SPECTRASCOPE_BUNDLED_MODELS'):
        path=Path(os.environ['SPECTRASCOPE_BUNDLED_MODELS'])/'sam3-tracker'
        if path.is_dir():return path
    for root in component_roots():
        path=root/'models/sam3-tracker'
        if path.is_dir():return path
    return None


def weights_available(data):
    return all((Path(data)/'models/sam3-tracker'/name).is_file() for name in FILES) or bool((source:=model_source()) and all((source/name).is_file() for name in FILES))


def ort_runtime():
    if importlib.util.find_spec('onnxruntime') is None:
        path=next((p/'python' for p in component_roots() if (p/'python/onnxruntime/__init__.py').is_file()),None)
        if path is None:raise ValueError('SAM 3 runtime is missing. Install the separate offline SAM 3 component, then restart local services. No files will be downloaded.')
        if os.name=='nt':_DLL_DIRS.append(os.add_dll_directory(str(path/'onnxruntime/capi')))
        sys.path.insert(0,str(path))
    import onnxruntime as ort
    ort.disable_telemetry_events()
    return ort


@lru_cache(maxsize=8)
def _verify(path,stamp):
    root=Path(path)
    for name,expected in FILES.items():
        file=root/name
        if file.is_symlink() or not file.resolve().is_relative_to(root.resolve()):raise ValueError('Invalid SAM 3 model path')
        digest=hashlib.sha256()
        with file.open('rb') as source:
            for block in iter(lambda:source.read(8*1024*1024),b''):digest.update(block)
        if digest.hexdigest()!=expected:raise ValueError('SAM 3 model hash mismatch: '+name+'. Restore the offline component.')
    return True


def prepare(data,progress):
    destination=Path(data)/'models/sam3-tracker'
    if not all((destination/name).is_file() for name in FILES):
        from model_packs import ensure_model_pack
        progress(22,'Preparing the local SAM 3 model pack')
        ensure_model_pack(destination,'sam3-tracker')
    if not all((destination/name).is_file() for name in FILES):raise ValueError('SAM 3 model pack is incomplete. Install the offline SAM 3 component; no download fallback is allowed.')
    progress(28,'Verifying SAM 3 model files')
    stamp=tuple((name,(destination/name).stat().st_size,(destination/name).stat().st_mtime_ns,(destination/name).stat().st_ctime_ns) for name in FILES)
    _verify(str(destination),stamp)
    return destination,{'source':'https://github.com/facebookresearch/sam3','distribution':'https://huggingface.co/'+REPOSITORY,'revision':REVISION,'license':'SAM License','weights_sha256':FILES,'implementation':'SAM 3 Tracker ONNX: quantized INT8 vision encoder and FP32 prompt/mask decoder','prompts':'Foreground/background points and a box; no text-concept model in this adapter','checkpoint_schema':'sam3-tracker-onnx-v1'}


def require_cpu_memory(minimum_gib):
    from execution_policy import available_ram_bytes
    available=available_ram_bytes()
    if available is not None and available<minimum_gib*1024**3:
        raise ValueError(f'SAM 3 needs more free RAM for CPU inference ({available/1024**3:.1f} GiB available; about 8 GiB recommended). Close other heavy applications and retry. No lower-quality model is substituted.')


@lru_cache(maxsize=1)
def load_model(path):
    ort=ort_runtime()
    from execution_policy import onnx_providers
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    options.enable_cpu_mem_arena=False;options.enable_mem_pattern=False;options.log_severity_level=3
    providers=onnx_providers(ort.get_available_providers())
    if providers==['CPUExecutionProvider']:require_cpu_memory(7.5)
    elif os.name=='nt':ort.preload_dlls()
    encoder=ort.InferenceSession(str(Path(path)/'onnx/vision_encoder_quantized.onnx'),sess_options=options,providers=providers)
    decoder=ort.InferenceSession(str(Path(path)/'onnx/prompt_encoder_mask_decoder.onnx'),sess_options=options,providers=providers)
    return encoder,decoder


class Predictor:
    """Private embeddings per request; shared sessions are serialized to bound RAM."""
    def __init__(self,path):
        with _LOCK:self.encoder,self.decoder=load_model(str(path))
        self.provider=self.encoder.get_providers()[0]
    def predict(self,rgb,points,labels,box,progress):
        import torch
        from torch.nn.functional import interpolate
        height,width=rgb.shape[:2]
        resized=np.asarray(Image.fromarray(rgb).resize((1008,1008),Image.Resampling.BILINEAR),dtype='float32')
        values=np.ascontiguousarray(resized.transpose(2,0,1)[None]/127.5-1)
        with _LOCK:
            if self.provider=='CPUExecutionProvider':require_cpu_memory(6.5)
            progress(45,'SAM 3: encoding the image on '+self.provider)
            embeddings=self.encoder.run(None,{'pixel_values':values})
            feed={output.name:value for output,value in zip(self.encoder.get_outputs(),embeddings)}
            scale=np.array([1008/width,1008/height],dtype='float32')
            feed['input_points']=np.asarray(points,dtype='float32').reshape(1,1,-1,2)*scale
            feed['input_labels']=np.asarray(labels,dtype='int64').reshape(1,1,-1)
            feed['input_boxes']=np.asarray(box,dtype='float32').reshape(1,1,4)*np.tile(scale,2) if box is not None else np.zeros((1,0,4),dtype='float32')
            progress(75,'SAM 3: predicting prompted boundaries')
            outputs=dict(zip([o.name for o in self.decoder.get_outputs()],self.decoder.run(None,feed)))
        masks=outputs['pred_masks'][0,0];scores=outputs['iou_scores'][0,0]
        if not np.isfinite(masks).all() or not np.isfinite(scores).all():raise ValueError('SAM 3 returned non-finite predictions')
        masks=interpolate(torch.from_numpy(masks[:,None]),size=(height,width),mode='bilinear',align_corners=False)[:,0].numpy()>0
        if box is not None:
            best=int(np.argmax(scores));masks=masks[best:best+1];scores=scores[best:best+1]
        return masks,scores
