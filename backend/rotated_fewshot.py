"""Local research baseline for reclassifying existing oriented proposals.

Fits class prototypes from a few oriented reference chips. The proposal
detector stays frozen and limits localization and novel-object recall.
"""
import numpy as np
from PIL import Image


def oriented_chip(image, corners, size):
    points = np.asarray(corners,dtype='float64')
    if points.shape != (4,2) or not np.isfinite(points).all():
        raise ValueError('Supply four finite oriented corners in boundary order')
    edges = np.roll(points,-1,axis=0)-points
    length = np.linalg.norm(edges,axis=1)
    next_edges = np.roll(edges,-1,axis=0)
    cross = edges[:,0]*next_edges[:,1]-edges[:,1]*next_edges[:,0]
    if (length<2).any() or not (np.all(cross>0) or np.all(cross<0)):
        raise ValueError('Use a nondegenerate convex oriented box')
    if (points<0).any() or (points>[image.width,image.height]).any():
        raise ValueError('The oriented box extends outside the source image')
    if cross[0]<0:
        points=points[[0,3,2,1]]
        length=length[[3,2,1,0]]
    if length[0]+length[2]<length[1]+length[3]:
        points=np.roll(points,1,axis=0)
    patch=image.transform((size,size),Image.Transform.QUAD,tuple(points[[0,3,2,1]].ravel()),Image.Resampling.BILINEAR)
    return patch.convert('RGB'), min(length)/max(length)


def descriptor(image,corners):
    patch, aspect=oriented_chip(image,corners,32)
    rgb=np.asarray(patch.convert('RGB'),dtype='float32')/255
    gray=rgb.mean(axis=2)
    dy,dx=np.gradient(gray)
    magnitude=np.hypot(dx,dy)
    angles=(np.arctan2(dy,dx)%np.pi)*(8/np.pi)
    cells=[]
    for row in (0,8,16,24):
        for col in (0,8,16,24):
            area=np.s_[row:row+8,col:col+8]
            bins=np.minimum(angles[area].astype(int),7).ravel()
            vector=np.bincount(bins,magnitude[area].ravel(),minlength=8)
            cells.append(vector/max(np.linalg.norm(vector),.01))
    hog=np.asarray(cells).reshape(4,4,8)
    # The same box can start at either end of its long axis.
    hog=(hog+hog[::-1,::-1])/2
    color=[np.histogram(rgb[:,:,c],bins=8,range=(0,1))[0]/1024 for c in range(3)]
    values=np.concatenate([hog.ravel(),*color,[aspect]])
    return values/max(np.linalg.norm(values),1e-8)


def fit(references, shots):
    """References are supplied as (PIL image, ordered corners, class name)."""
    if type(shots) is not int or not 1<=shots<=10:
        raise ValueError('Use 1–10 shots per class')
    groups={}
    for image,corners,label in references:
        groups.setdefault(label,[]).append(descriptor(image,corners))
    if not 2<=len(groups)<=30 or any(len(group)!=shots for group in groups.values()):
        raise ValueError('Supply exactly the requested shot count for each of 2–30 classes')
    names=sorted(groups)
    prototypes=np.stack([np.mean(groups[name],axis=0) for name in names])
    return {'classes':names,'shots':shots,'prototypes':prototypes}


def classify(model,image,corners):
    prototypes=model['prototypes']
    distance=np.sum((prototypes-descriptor(image,corners))**2,axis=1)
    rank=np.argsort(distance)
    return model['classes'][rank[0]],float(distance[rank[1]]-distance[rank[0]])


def score_oriented(truth, predictions, classes):
    from oriented_det.ops.iou import polygon_iou

    result={}
    for category in classes:
        total=sum(label==category for rows in truth.values() for _,label in rows)
        matched=set()
        hits=[]
        selected=sorted((p for p in predictions if p[2]==category),key=lambda p:-p[3])
        for scene,corners,_,_ in selected:
            candidates=[(polygon_iou(corners,q,intersection_backend='python'),i)
                        for i,(q,label) in enumerate(truth[scene]) if label==category and (scene,i) not in matched]
            best=max(candidates,default=(0,None))
            hits.append(best[0]>=.5)
            if hits[-1]:matched.add((scene,best[1]))
        tp=np.cumsum(hits,dtype=float)
        precision=tp/np.arange(1,len(hits)+1)
        recall=tp/max(total,1)
        ap=float(np.sum(np.diff(np.r_[0,recall])*np.maximum.accumulate(precision[::-1])[::-1])) if len(hits) else 0.
        p,r=len(matched)/max(len(hits),1),len(matched)/max(total,1)
        result[category]={'reference_objects':total,'predictions':len(hits),'tp':len(matched),
                          'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.,'ap50':ap}
    return result

