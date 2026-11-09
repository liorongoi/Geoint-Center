"""Frozen DINOv2 embeddings and few-shot centroids for true oriented boxes."""
from pathlib import Path
import hashlib

import numpy as np
from PIL import Image

from rotated_fewshot import oriented_chip

REVISION = '4610ca143709d58a633b6397a74412c2c3842454'
SHA256 = '04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081'
SOURCE = 'https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m'
SIDE = 84
BATCH = 16


def encoder(weights_path):
    import torch
    import timm
    from safetensors.torch import load
    from execution_policy import torch_device

    weights = Path(weights_path)
    if weights.is_symlink() or not weights.is_file():
        raise ValueError('Install the pinned local DINOv2 SafeTensors checkpoint')
    with weights.open('rb') as stream:
        content = stream.read(88_240_511)
    if len(content)!=88_240_510 or hashlib.sha256(content).hexdigest()!=SHA256:
        raise ValueError('DINOv2 checkpoint differs from the pinned release')
    torch.set_num_threads(4)
    model = timm.create_model('vit_small_patch14_dinov2', pretrained=False,
                              num_classes=0, dynamic_img_size=True)
    model.load_state_dict(load(content),strict=True)
    device = torch_device()
    model.to(device).eval()
    mean = torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
    std = torch.tensor([.229,.224,.225],device=device)[None,:,None,None]

    def embed(image, corners):
        if not 1<=len(corners)<=10000:
            raise ValueError('Use 1–10000 oriented boxes per embedding call')
        results = []
        with torch.inference_mode():
            for offset in range(0,len(corners),BATCH):
                chips = [oriented_chip(image,quad,SIDE)[0] for quad in corners[offset:offset+BATCH]]
                # Opposite long-edge starts represent the same physical box.
                augmented = chips + [chip.transpose(Image.Transpose.ROTATE_180) for chip in chips]
                pixels = torch.from_numpy(np.stack([np.asarray(chip) for chip in augmented])).to(device)
                x = (pixels.permute(0,3,1,2).float()/255-mean)/std
                tokens = model.forward_features(x)
                features = torch.nn.functional.normalize(torch.cat((tokens[:,0],tokens[:,1:].mean(1)),dim=1))
                first, opposite = features.chunk(2)
                results.append(torch.nn.functional.normalize(first+opposite).cpu().numpy())
        output = np.concatenate(results)
        if not np.isfinite(output).all():
            raise ValueError('DINOv2 generated nonfinite features')
        return output

    return embed,str(device)


def fit_vectors(features, labels, shots):
    if type(shots) is not int or not 1<=shots<=10:
        raise ValueError('Use one to ten reference objects per class')
    features=np.asarray(features)
    if features.shape!=(len(labels),768) or not np.isfinite(features).all():
        raise ValueError('Invalid DINOv2 feature matrix')
    names=sorted(set(labels))
    if not 2<=len(names)<=30 or any(labels.count(name)!=shots for name in names):
        raise ValueError('Supply exactly the declared number of examples for each class')
    prototypes=np.stack([features[np.asarray(labels)==name].mean(0) for name in names])
    magnitude=np.linalg.norm(prototypes,axis=1,keepdims=True)
    if (magnitude<=1e-8).any():
        raise ValueError('Degenerate DINOv2 prototype')
    return names,prototypes/magnitude


def predict_vectors(features, model, score_penalty=None, exemplars=None, neighbors=0):
    names,prototypes=model
    if exemplars is None:
        similarities=features @ prototypes.T
    else:
        # Keep class diversity that averaging every support embedding can hide.
        count=min(neighbors,exemplars.shape[1])
        scores=np.einsum('nd,csd->ncs',features,exemplars)
        similarities=np.sort(scores,axis=2)[:,:,-count:].mean(axis=2)
    if score_penalty is not None:
        similarities=similarities-score_penalty
    rank=np.argsort(similarities,axis=1)
    best,second=rank[:,-1],rank[:,-2]
    scores=similarities[np.arange(len(best)),best]-similarities[np.arange(len(best)),second]
    return [(names[index],float(score)) for index,score in zip(best,scores)]
