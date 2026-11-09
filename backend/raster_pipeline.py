"""Read calibrated, band-aware raster stacks from local files and remote catalogs."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math
import os
import re
from offline_policy import local_raster, raster_driver
from runtime_profiles import performance_profile

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_geom
from rasterio.windows import Window, from_bounds as window_from_bounds


@dataclass
class RasterStack:
    pixels: np.ndarray
    valid: np.ndarray
    profile: dict
    bands: list[str]
    provenance: list[dict]


def source_assets(record: dict) -> list[tuple[str, str]]:
    if record['mode'] == 'local_path':
        if record.get('local_sources'):
            return [(item['key'], item['path']) for item in record['local_sources']]
        return [('stack', record['local_path'])]
    if record['mode'] == 'download':
        return [(item['key'], item['path']) for item in record.get('files', [])]
    return [(item['key'], item['url']) for item in record.get('asset_urls', [])]


def canonical_band(name: str, catalog: str = '') -> str:
    key = name.strip().lower().replace('-', '_')
    aliases = {'blue': 'BLUE', 'green': 'GREEN', 'red': 'RED', 'nir': 'NIR', 'nir08': 'NIR',
               'nir09': 'NIR', 'swir16': 'SWIR1', 'swir22': 'SWIR2', 'swir1': 'SWIR1', 'swir2': 'SWIR2'}
    if re.fullmatch(r'st_b\d+',key):return key.upper()
    if key in aliases:
        return aliases[key]
    if catalog == 'cbers-4a-wpm' and re.fullmatch(r'band[0-4]',key):
        return ('PAN','BLUE','GREEN','RED','NIR')[int(key[-1])]
    band = re.search(r'(?:^|_)(?:sr_)?b(0?[1-9]|1[0-2]|8a)(?:_|$)', key)
    if band:
        number = band.group(1).lstrip('0').upper()
        mapping = ({'2': 'BLUE', '3': 'GREEN', '4': 'RED', '8A': 'NIR', '11': 'SWIR1', '12': 'SWIR2'}
                   if catalog in ('sentinel-2','sentinel-2-l1c','hls-s30') else {'2': 'BLUE', '3': 'GREEN', '4': 'RED', '5': 'NIR', '6': 'SWIR1', '7': 'SWIR2'} if catalog in ('landsat','hls-l30') else {})
        return mapping.get(number, f'B{number}')
    return name.upper()


def read_stack(record: dict, image: dict, max_size: int = 384, geometry: dict | None = None, science_only: bool = False, max_elements: int | None = None, target_profile: dict | None = None, verify_sources: bool = True) -> RasterStack:
    """Windowed read; resample different-resolution bands onto one documented grid.

    max_size bounds the analysis grid explicitly. Full original rasters stay intact.
    Masks use nearest-neighbor; science values use bilinear resampling.
    """
    sources = source_assets(record)
    resources=performance_profile()
    if max_elements is None:
        max_elements = resources['max_raster_elements']
    else:
        max_elements=min(max_elements,resources['max_raster_elements']) if max_elements>0 else resources['max_raster_elements']
    if not sources:
        raise ValueError('No raster sources configured for this image')
    if record['mode'] == 'stream' and os.getenv('RASTER_HTTP_HEADERS'):
        raise ValueError('Global raster credentials are disabled for remote streaming. Use an approved local copy.')
    with rasterio.Env(GDAL_HTTP_TIMEOUT='45', GDAL_HTTP_MAX_RETRY='2',
                      GDAL_CACHEMAX=resources['gdal_cache_mb']*1024**2,
                      GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_HEADERS='',
                      GDAL_HTTP_MULTIRANGE='SERIAL', GDAL_HTTP_PROXY=''), ExitStack() as contexts:
        local_schema = record.get('local_sources', [])
        if local_schema and verify_sources:
            from imagery_products import checksum
            verified_metadata=set()
            for item in local_schema:
                if item.get('sha256') and checksum(Path(local_raster(item['path']))) != item['sha256']:
                    raise ValueError('A saved band snapshot changed; inference was stopped to preserve input provenance')
                metadata=item.get('radiometry')
                if metadata and metadata['metadata_sha256'] not in verified_metadata:
                    path=Path(metadata['metadata_path'])
                    if path.is_symlink() or not path.is_file() or path.stat().st_size>2*1024*1024 or checksum(path)!=metadata['metadata_sha256']:
                        raise ValueError('Saved supplier calibration metadata changed or is unavailable')
                    verified_metadata.add(metadata['metadata_sha256'])
        if record['mode']=='stream':
            from public_raster_transport import raster_range_gateway
            paths=contexts.enter_context(raster_range_gateway(image.get('catalog'),[path for _,path in sources]))
            opened=[contexts.enter_context(rasterio.open(path)) for path in paths]
        else:
            opened=[contexts.enter_context(rasterio.open(local_raster(path),driver=raster_driver(path))) for _,path in sources]
        first=opened[0]
        # Pick the finest native band grid when the product bands share a CRS.
        # A leading 60 m coastal band must not discard 10 m AOI detail.
        if all(src.crs==first.crs for src in opened):
            first=min(opened,key=lambda src:abs(src.transform.a*src.transform.e))
        if not first.crs:
            raise ValueError('Raster has no CRS. Supply a georeferenced GeoTIFF for map-based labels.')
        bounds = first.bounds
        if geometry and geometry.get('type') != 'Point':
            from rasterio.features import bounds as geometry_bounds
            projected = transform_geom('EPSG:4326', first.crs, geometry)
            a = geometry_bounds(projected)
            bounds = (max(a[0], bounds.left), max(a[1], bounds.bottom), min(a[2], bounds.right), min(a[3], bounds.top))
        if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise ValueError('AOI does not intersect this raster')
        win = window_from_bounds(*bounds, transform=first.transform).intersection(Window(0, 0, first.width, first.height))
        factor = min(1., max_size / max(win.width, win.height))
        width, height = max(1, round(win.width * factor)), max(1, round(win.height * factor))
        if target_profile is not None:
            width, height = target_profile['width'], target_profile['height']
        cells=width*height
        channels=sum(src.count for src in opened)
        if cells*channels>max_elements:
            raise ValueError('This all-band cube exceeds the local analysis memory budget. Choose a smaller grid or AOI; no spectral bands were dropped.')
        transform = from_bounds(*bounds, width, height)
        target_crs = first.crs
        if target_profile is not None:
            transform, target_crs = target_profile['transform'], target_profile['crs']
        # Keep one scratch band rather than a second cube for a multiband source.
        # This estimates stack construction only. Model-specific limits still apply.
        working_bytes=cells*(channels*4+32)+resources['gdal_cache_mb']*1024**2
        if cells*channels*4>=128*1024**2:
            from execution_policy import available_ram_bytes
            available=available_ram_bytes()
            if available is None or available<working_bytes+1024**3:
                raise ValueError('The complete raster stack cannot be built with current free memory. Retry when the machine is less busy or choose an explicit smaller AOI or grid.')
        pixels=np.empty((channels,height,width),dtype='float32')
        names, provenance = [], []
        valid=np.ones((height,width),dtype=bool)
        catalog = image.get('catalog', '')
        band_sensor = record.get('sensor') or image.get('sensor') or catalog
        explicit = record.get('band_names') or image.get('band_names') or []
        channel_offset = 0
        for source_number, ((key, path),src) in enumerate(zip(sources,opened)):
            schema = local_schema[source_number] if local_schema else {}
            channel_calibration=schema.get('stac_channel_calibration')
            if channel_calibration is not None and (not isinstance(channel_calibration,list) or len(channel_calibration)!=src.count
                    or any(schema.get(key) is not None for key in ('scale_override','offset_override','radiometry'))):
                raise ValueError('Conflicting or incomplete per-channel STAC calibration')
            source_names = schema.get('band_names', [])
            if source_names and len(source_names) != src.count:
                raise ValueError(f'Band schema changed for {key}; expected {len(source_names)} channels, found {src.count}')
            if not src.crs:
                raise ValueError(f'Band {key} has no coordinate reference system')
            nodata_options={'src_nodata':0,'nodata':float('nan'),'dtype':'float32'} if schema.get('radiometry') else {}
            vrt = contexts.enter_context(WarpedVRT(src, crs=target_crs, transform=transform, width=width, height=height,
                                                 resampling=Resampling.bilinear, **nodata_options))
            band_provenance=[]
            for index in range(src.count):
                raw_name = source_names[index] if source_names else explicit[channel_offset+index] if len(explicit)==channels else src.descriptions[index] or (key if src.count == 1 else f'band_{index+1}')
                if science_only and (src.colorinterp[index]==rasterio.enums.ColorInterp.alpha or re.search(r'(^|[_\s-])(qa\w*|bqa|scl|mask|alpha|cloud\w*|aot|wvp|tci|visual|zenith|azimuth)([_\s-]|$)',raw_name,re.I)):
                    band_provenance.append({'source_band':index+1,'name':raw_name,'excluded':'Quality, atmospheric retrieval, geometry or display channel; not a measured spectral band'})
                    continue
                name = canonical_band(raw_name, band_sensor)
                data = vrt.read(index+1,masked=True).astype('float32',copy=False)
                values = data.filled(np.nan)
                scale, offset = src.scales[index], src.offsets[index]
                if schema.get('scale_override') is not None: scale = float(schema['scale_override'])
                if schema.get('offset_override') is not None: offset = float(schema['offset_override'])
                metadata = (image.get('asset_metadata') or {}).get(key, {})
                raster_bands = metadata.get('raster:bands') or [{}]
                raster_band = raster_bands[index] if index<len(raster_bands) else raster_bands[0]
                scale = raster_band.get('scale', scale)
                offset = raster_band.get('offset', offset)
                if channel_calibration is not None and channel_calibration[index] is not None:
                    values_for_band=channel_calibration[index]
                    if (not isinstance(values_for_band,dict) or set(values_for_band)!={'scale','offset'}
                            or any(type(value) not in (int,float) or not math.isfinite(value) for value in values_for_band.values())
                            or values_for_band['scale']==0):
                        raise ValueError(f'Invalid saved STAC calibration for {name}')
                    if (scale,offset)!=(1,0):
                        raise ValueError(f'Conflicting native or catalog calibration for {name}')
                    scale,offset=values_for_band['scale'],values_for_band['offset']
                if scale == 1 and offset == 0 and catalog == 'landsat' and ('sr_' in key.lower() or key.lower() in ('blue','green','red','nir08','swir16','swir22')):
                    scale, offset = .0000275, -.2
                values = values * scale + offset
                pixels[len(names)]=values
                valid &= np.isfinite(values) & ~np.ma.getmaskarray(data)
                names.append(name)
                wavelength=schema.get('wavelength_nm')
                if isinstance(wavelength,list):wavelength=wavelength[index] if index<len(wavelength) else None
                band_provenance.append({'source_band':index+1,'name':name,'scale':scale,'offset':offset,
                                        'wavelength_nm':wavelength})
                if channel_calibration is not None and channel_calibration[index] is not None:
                    band_provenance[-1]['calibration_source']='STAC asset channel'
            provenance.append({'asset': key, 'source': path, 'source_shape': [src.count, src.height, src.width],
                               'calibration_metadata':schema.get('radiometry'),
                               'sha256':schema.get('sha256'), 'original_source':schema.get('original_path'),
                               'scale': next((b['scale'] for b in band_provenance if 'scale' in b),None),
                               'offset': next((b['offset'] for b in band_provenance if 'offset' in b),None), 'bands':band_provenance})
            channel_offset += src.count
        if not names:raise ValueError('No science bands remain after excluding quality/alpha channels')
        if local_schema and len(set(names))!=len(names) and str(record.get('processing_level') or image.get('processing_level','')).upper()!='HLS':
            raise ValueError('Band names are ambiguous after sensor mapping; review the channel names before inference')
        pixels=pixels[:len(names)]
        if geometry and geometry.get('type') != 'Point':
            projected = transform_geom('EPSG:4326', target_crs, geometry)
            valid &= rasterize([(projected, 1)], out_shape=(height, width), transform=transform).astype(bool)
        if not valid.any():
            raise ValueError('No valid pixels in the selected AOI')
        profile = {'driver': 'GTiff', 'width': width, 'height': height, 'crs': target_crs,
                   'transform': transform, 'count': len(names), 'dtype': 'float32'}
    return RasterStack(pixels, valid, profile, names, provenance)


def label_mask(stack: RasterStack, labels: list[dict], classes: list[str]) -> np.ndarray:
    mask = np.full(stack.valid.shape, -1, dtype='int32')
    # Teacher-derived references are applied first. Explicit human classifications
    # retain priority even when the pseudo-label approval was saved later.
    ordered=sorted(labels,key=lambda label:label.get('origin')!='reviewed_pseudo_label')
    cache={};human_shapes=[]
    for label in ordered:
        if not label.get('geometry') or label['label'] not in classes:
            continue
        geom = transform_geom('EPSG:4326', stack.profile['crs'], label['geometry'])
        if label.get('origin')!='reviewed_pseudo_label':
            human_shapes.append((geom,classes.index(label['label'])));continue
        region=rasterize([(geom,1)],out_shape=stack.valid.shape,transform=stack.profile['transform'],fill=0,dtype='uint8').astype(bool)
        if label.get('origin')=='reviewed_pseudo_label':
            from pseudo_labels import read_reference,signature
            reference=label['raster_reference']
            key=signature({k:v for k,v in reference.items() if k!='class_value'})
            if key not in cache:cache[key]=read_reference(reference,stack)
            region&=cache[key]==reference['class_value']
        mask[region]=classes.index(label['label'])
    if human_shapes:
        human=rasterize(human_shapes,out_shape=stack.valid.shape,transform=stack.profile['transform'],fill=-1,dtype='int32')
        mask[human>=0]=human[human>=0]
    mask[~stack.valid] = -1
    return mask


def coregister(stack: RasterStack, max_shift: int = 5) -> list[dict]:
    """Conservative translation correction. Reject weak or degraded matches."""
    from scipy.ndimage import shift, sobel
    ref_index = stack.bands.index('NIR') if 'NIR' in stack.bands else 0
    reference = np.nan_to_num(stack.pixels[ref_index])
    ref_edge = np.hypot(sobel(reference, 0), sobel(reference, 1))
    records = []
    for index, values in enumerate(stack.pixels):
        if index == ref_index:
            records.append({'band': stack.bands[index], 'reference': True, 'applied': False, 'shift_px': [0, 0]})
            continue
        moving = np.nan_to_num(values)
        edge = np.hypot(sobel(moving, 0), sobel(moving, 1))
        cross = np.fft.fft2(ref_edge - ref_edge.mean()) * np.conj(np.fft.fft2(edge - edge.mean()))
        cross /= np.maximum(np.abs(cross), 1e-9)
        surface = np.abs(np.fft.ifft2(cross))
        dy, dx = np.unravel_index(np.argmax(surface), surface.shape)
        dy = int(dy if dy <= surface.shape[0] // 2 else dy - surface.shape[0])
        dx = int(dx if dx <= surface.shape[1] // 2 else dx - surface.shape[1])
        quality = float(surface.max() / max(surface.mean(), 1e-9))
        before = float(np.corrcoef(ref_edge.ravel(), edge.ravel())[0, 1]) if edge.std() and ref_edge.std() else 0.
        adjusted = shift(edge, (dy, dx), order=0, mode='constant')
        after = float(np.corrcoef(ref_edge.ravel(), adjusted.ravel())[0, 1]) if adjusted.std() and ref_edge.std() else 0.
        apply = quality > 8 and max(abs(dy), abs(dx)) <= max_shift and after > before + .02
        if apply:
            stack.pixels[index] = shift(moving, (dy, dx), order=0, mode='constant', cval=np.nan)
        records.append({'band': stack.bands[index], 'shift_px': [dy, dx], 'peak_ratio': quality,
                        'correlation_before': before, 'correlation_after': after, 'applied': bool(apply)})
    stack.valid &= np.all(np.isfinite(stack.pixels), axis=0)
    return records


def write_raster(path: Path, values: np.ndarray, stack: RasterStack, nodata: float | int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = values if values.ndim == 3 else values[None]
    profile = {**stack.profile, 'count': array.shape[0], 'dtype': str(array.dtype), 'nodata': nodata, 'compress': 'deflate'}
    with rasterio.open(path, 'w', **profile) as target:
        target.write(array)
