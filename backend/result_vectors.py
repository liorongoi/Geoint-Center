"""Pixel-exact categorical polygon export, never a training reference."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_geom
from runtime_profiles import performance_profile


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def polygon_export(source, destination, classes, lineage, progress=lambda *a:None,class_values=None):
    """Stream four-connected class polygons; no smoothing, fitting or class changes."""
    source=Path(source);destination=Path(destination);partial=destination.with_suffix('.partial')
    mapping=dict(zip(class_values if class_values is not None else range(1,len(classes)+1),classes))
    before=digest(source);count=0;started=time.monotonic();progress(5,'Reading classification grid',0)
    try:
        with rasterio.open(source) as src:
            if src.count!=1 or not src.crs:raise ValueError('A single-band georeferenced class raster is required')
            # A categorical scan needs one byte of validity per pixel, not a
            # full multiband float analysis cube. Permit a standard Sentinel tile
            # in the 16 GB tier while bounding the mask to at most 160 MiB.
            maximum=min(160_000_000,performance_profile()['max_raster_elements']*4)
            if src.width*src.height>maximum:
                raise ValueError('This class raster exceeds the safe polygon export budget for this machine. Run a smaller AOI before exporting; the raster will not be downsampled silently.')
            profile={'crs':str(src.crs),'transform':list(src.transform)[:6],'width':src.width,'height':src.height,'nodata':src.nodata}
            valid=src.read_masks(1).astype(bool)
            provenance={**lineage,'source_sha256':before,'grid':profile,'connectivity':4,'simplified':False,'taxonomy':classes,'label_role':'prediction','training_reference':False}
            with open(partial,'w',encoding='utf-8') as output:
                output.write('{"type":"FeatureCollection","spectrascope":'+json.dumps(provenance)+',"features":[')
                progress(10,'Tracing class boundaries · polygon count shown as work advances',0)
                for geometry,value in shapes(rasterio.band(src,1),mask=valid,connectivity=4,transform=src.transform):
                    value=int(value)
                    if value not in mapping:raise ValueError(f'Raster class {value} has no class name in this run')
                    feature={'type':'Feature','id':count+1,'properties':{'class_id':value,'class_name':mapping[value]},'geometry':transform_geom(src.crs,'EPSG:4326',geometry)}
                    if count:output.write(',')
                    output.write(json.dumps(feature,separators=(',',':'),allow_nan=False));count+=1
                    if count%1000==0:
                        progress(10,f'Tracing boundaries · {count:,} polygons',count)
                    if count>1_000_000 or output.tell()>256*1024*1024:raise ValueError('Vector export exceeds one million polygons or 256 MiB. Use a smaller AOI; no classes were removed or simplified.')
                output.write(']}')
        progress(90,'Verifying original raster and vector export',count)
        if digest(source)!=before:raise ValueError('Source raster changed during export; export stopped to preserve lineage')
        partial.replace(destination)
        return {'polygon_count':count,'sha256':digest(destination),'source_sha256':before,'bytes':destination.stat().st_size,'seconds':round(time.monotonic()-started,3),'provenance':provenance}
    except Exception:
        partial.unlink(missing_ok=True)
        raise
