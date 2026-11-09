"""Bounded multi-scale proposal recovery for offline aircraft evaluation."""
from dataclasses import replace

import numpy as np
from PIL import Image


def aircraft_proposals(rgb, proposals, infer, merge, plane_label, scales, threshold):
    """Combine original boxes and zoomed aircraft boxes in source pixel space."""
    if (not 1 <= len(scales) <= 2 or any(type(s) is not int or s not in (2,3) for s in scales)
            or len(set(scales)) != len(scales) or not .1 <= threshold <= .4):
        raise ValueError('Use one or two bounded aircraft zooms and a valid score floor')
    height,width=rgb.shape[:2]
    if max(height,width)*max(scales)>2048:
        raise ValueError('Aircraft zoom evaluation needs source chips bounded to 2048 upsampled pixels')
    candidates=[p for p in proposals if p['label']==plane_label]
    tile=Image.fromarray(rgb)
    for scale in scales:
        zoomed=np.asarray(tile.resize((width*scale,height*scale),Image.Resampling.BILINEAR))
        for item in infer(zoomed,score_threshold=threshold):
            if item['label']!=plane_label:
                continue
            box=item['rbox']
            remapped=replace(box,cx=box.cx/scale,cy=box.cy/scale,width=box.width/scale,height=box.height/scale)
            if (0<=remapped.cx<width and 0<=remapped.cy<height
                    and all(0<=x<=width and 0<=y<=height for x,y in remapped.corners())):
                candidates.append({**item,'rbox':remapped})
    return [p for p in proposals if p['label']!=plane_label]+merge(candidates)
