"""Offline DSen2 numerical inference using pinned published tensors."""
import hashlib
from functools import lru_cache

from execution_policy import configure_tensorflow

WEIGHTS = {
    20: ("s2_032_lr_1e-04.hdf5", "2dab3725ecb1a0f69116a864ffcb18139177b5287af1dff2495505d4c79322bb", 10, 6),
    60: ("s2_030_lr_1e-05.hdf5", "d282eb43915383344e80663de5b1bed1bf0009d57fbf38804e39759d1c1824cf", 12, 2),
}


def load_predictor(checkpoint, expected_hash, channels, outputs):
    import h5py

    tf = configure_tensorflow()
    layers = []
    # Consume numeric tensors from the checked release. HDF5 model_config
    # contains legacy serialized Lambda functions and is deliberately ignored.
    with checkpoint.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected_hash:
            raise ValueError("DSen2 checkpoint does not match the pinned research release")
        stream.seek(0)
        with h5py.File(stream, "r") as model:
            for index in range(1, 15):
                name = f"conv2d_{index}"
                group = model["model_weights"][name][name]
                kernel = group["kernel:0"][:]
                bias = group["bias:0"][:]
                input_channels = channels if index == 1 else 128
                output_channels = outputs if index == 14 else 128
                if kernel.shape != (3, 3, input_channels, output_channels) or bias.shape != (output_channels,):
                    raise ValueError("Unexpected DSen2 tensor layout")
                layers.append((tf.constant(kernel), tf.constant(bias)))

    def convolution(value, index):
        kernel, bias = layers[index]
        return tf.nn.bias_add(tf.nn.conv2d(value, kernel, strides=1, padding="SAME"), bias)

    def predict(patch):
        source = tf.convert_to_tensor(patch[None, ...] / 2000, dtype=tf.float32)
        hidden = tf.nn.relu(convolution(source, 0))
        for index in range(1, 13, 2):
            correction = convolution(tf.nn.relu(convolution(hidden, index)), index + 1)
            hidden = hidden + 0.1 * correction
        estimate = (convolution(hidden, 13) + source[..., -outputs:]) * 2000
        return estimate.numpy()[0], estimate.device

    return predict


BANDS = {10: ('B02', 'B03', 'B04', 'B08'),
         20: ('B05', 'B06', 'B07', 'B8A', 'B11', 'B12'),
         60: ('B01', 'B09')}
MODELS = {'dsen2_20': 20, 'dsen2_60': 60}
DETAILS = {
    model: {
        'name': f'DSen2 · Sentinel-2 {resolution} m bands',
        'task': 'super_resolution', 'trainable': False, 'classes': [],
        'training_mode': 'unsupported', 'revision': WEIGHTS[resolution][1][:16],
        'satellite_suitability': {
            'requirements': f'Local Sentinel-2 L1C SAFE product with native {"10/20" if resolution==20 else "10/20/60"} m grids, '
                            'verified calibration XML and all required bands.',
            'limitations': 'A model estimate on a 10 m grid. Original sensor resolution is retained in the report.',
        },
    } for model, resolution in MODELS.items()
}


def checkpoint(model, data=None):
    from importlib.util import find_spec
    import os
    from pathlib import Path

    name, checksum, _, _ = WEIGHTS[MODELS[model]]
    roots = [os.getenv('SPECTRASCOPE_DSEN2_MODELS'),
             (Path(data) / 'components' / 'dsen2') if data is not None else None]
    if os.getenv('SPECTRASCOPE_BUNDLED_MODELS'):
        roots.append(Path(os.environ['SPECTRASCOPE_BUNDLED_MODELS']) / 'dsen2')
    if find_spec('tensorflow') is None or find_spec('h5py') is None:
        return None
    for root in filter(None, roots):
        if str(root).startswith((chr(92) * 2, '//', '/vsi')) or '://' in str(root):
            continue
        path = Path(root) / name
        try:
            if path.is_file() and not path.is_symlink():
                status = path.stat()
                revision = (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)
                if _checked_weight(path, revision, checksum):
                    return path
        except OSError:
            continue
    return None


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@lru_cache(maxsize=16)
def _checked_weight(path, _revision, checksum):
    return digest(path) == checksum


def open_inputs(model, record, contexts, verify=False):
    import math
    from pathlib import Path

    import rasterio
    from offline_policy import local_raster, raster_driver
    from sensor_metadata import band_name

    resolution = MODELS[model]
    required = BANDS[10] + BANDS[20] + (BANDS[60] if resolution == 60 else ())
    if record.get('mode') != 'local_path':
        raise ValueError('DSen2 requires a local Sentinel-2 L1C SAFE snapshot with its calibration XML')
    selected = {}
    verified = set()
    for asset in record.get('local_sources', []):
        names = asset.get('band_names', [])
        if len(names) != 1 or (band := band_name(names[0])) not in required:
            continue
        if band in selected:
            raise ValueError(f'DSen2 found multiple native sources for {band}')
        calibration = asset.get('radiometry') or {}
        if calibration.get('processing_level') != 'L1C' or not calibration.get('metadata_sha256') or not asset.get('sha256'):
            raise ValueError(f'DSen2 requires a verified Sentinel-2 L1C SAFE band and calibration XML for {band}')
        scale, offset = asset.get('scale_override'), asset.get('offset_override')
        if scale is None or offset is None or not math.isfinite(scale) or scale <= 0 or not math.isfinite(offset):
            raise ValueError(f'DSen2 requires the declared reflectance scale and offset for {band}')
        path = Path(local_raster(asset['path']))
        if verify:
            if digest(path) != asset['sha256']:
                raise ValueError(f'Saved DSen2 source band changed: {band}')
            metadata_hash = calibration['metadata_sha256']
            if metadata_hash not in verified:
                metadata = Path(calibration['metadata_path'])
                if metadata.is_symlink() or metadata.stat().st_size > 2 * 1024**2 or digest(metadata) != metadata_hash:
                    raise ValueError('Saved DSen2 calibration XML changed or is unavailable')
                verified.add(metadata_hash)
        src = contexts.enter_context(rasterio.open(path, driver=raster_driver(str(path))))
        if src.count != 1 or src.dtypes != ('uint16',):
            raise ValueError(f'DSen2 expects original Sentinel-2 uint16 samples for {band}')
        selected[band] = (src, scale, offset, asset)
    missing = set(required) - selected.keys()
    if missing:
        raise ValueError('DSen2 needs these native bands: ' + ', '.join(sorted(missing)))
    if len({value[3]['radiometry']['metadata_sha256'] for value in selected.values()}) != 1:
        raise ValueError('DSen2 bands must come from the same Sentinel-2 SAFE product')
    guide = selected[BANDS[10][0]][0]
    if not guide.crs or not guide.crs.is_projected or guide.crs.linear_units_factor[1] != 1:
        raise ValueError('DSen2 requires a projected Sentinel-2 grid in metres')
    for metres, bands in BANDS.items():
        for band in bands:
            if band not in selected:
                continue
            src = selected[band][0]
            t = src.transform
            if (src.crs != guide.crs or (t.a, t.b, t.d, t.e) != (metres, 0, 0, -metres)
                    or t.c != guide.transform.c or t.f != guide.transform.f
                    or src.width * metres != guide.width * 10 or src.height * metres != guide.height * 10):
                raise ValueError(f'DSen2 needs coextensive, aligned native Sentinel-2 grids; check {band}')
    return guide, {band: selected[band] for band in required}


def area(guide, geometry):
    from math import ceil, floor
    from rasterio.features import bounds
    from rasterio.warp import transform_geom
    from rasterio.windows import Window, from_bounds

    full = Window(0, 0, guide.width, guide.height)
    if not geometry or geometry.get('type') == 'Point':
        return full, None
    if geometry.get('type') not in ('Polygon', 'MultiPolygon'):
        raise ValueError('DSen2 needs a valid polygon or point AOI')
    projected = transform_geom('EPSG:4326', guide.crs, geometry)
    w = from_bounds(*bounds(projected), transform=guide.transform)
    left, top = floor(w.col_off), floor(w.row_off)
    right, bottom = ceil(w.col_off + w.width), ceil(w.row_off + w.height)
    try:
        return Window(left, top, right-left, bottom-top).intersection(full), projected
    except ValueError as exc:
        raise ValueError('DSen2 AOI is outside this image') from exc


def preflight(model, record, geometry, data=None):
    from contextlib import ExitStack
    import rasterio

    if checkpoint(model, data) is None:
        raise ValueError('Install the local DSen2 model pack and TensorFlow runtime')
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', PROJ_NETWORK='OFF'), ExitStack() as contexts:
        guide, _ = open_inputs(model, record, contexts)
        area(guide, geometry)


def execute(task, data, _reference, _image, record, persist):
    from contextlib import ExitStack
    from pathlib import Path

    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import geometry_mask
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window
    from scipy.ndimage import minimum_filter

    model = task['model']
    resolution = MODELS[model]
    _, checksum, channels, outputs = WEIGHTS[resolution]
    folder = Path(data) / 'segmentations' / task['id']
    destination = folder / 'enhanced.tif'
    temporary = folder / 'enhanced.partial'
    published = False

    def progress(value, stage):
        task.update(status='running', progress=value, stage=stage)
        persist(task)

    try:
        progress(5, 'Verifying native Sentinel-2 bands and pinned DSen2 weights')
        weights = checkpoint(model, data)
        if weights is None:
            raise ValueError('DSen2 offline weights or TensorFlow runtime missing')
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', PROJ_NETWORK='OFF',
                          GDAL_TIFF_INTERNAL_MASK=True), ExitStack() as contexts:
            guide, sources = open_inputs(model, record, contexts, verify=True)
            region, geometry = area(guide, task.get('geometry'))
            predict = load_predictor(weights, checksum, channels, outputs)
            vrt_options = dict(crs=guide.crs, transform=guide.transform, width=guide.width,
                               height=guide.height, dtype='float32', src_nodata=0, nodata=float('nan'),
                               resampling=Resampling.bilinear)
            inputs = [(contexts.enter_context(WarpedVRT(src, **vrt_options)), scale, offset)
                      for src, scale, offset, _ in sources.values()]
            # Use the physical L1C reflectance on a fixed 10000 DN scale to
            # remove the product's encoding offset before inference.
            names = BANDS[resolution]
            grid = dict(crs=guide.crs, transform=guide.window_transform(region),
                        width=int(region.width), height=int(region.height))
            nodata = -9999.
            size, halo = 144, 14
            rows, cols = grid['height'], grid['width']
            total = ((rows + size-1)//size)*((cols + size-1)//size)
            complete = valid_pixels = 0
            with rasterio.open(temporary, 'w', driver='GTiff', **grid, dtype='float32',
                               count=outputs, nodata=nodata, compress='deflate',
                               predictor=3, tiled=True, BIGTIFF='IF_SAFER') as dst:
                dst.update_tags(model=model, weights_sha256=checksum, derived='model_estimate',
                                reference_eligible='false', source_resolution_m=str(resolution),
                                output_grid_m='10', units='TOA reflectance',
                                uncertainty='Native 10 m accuracy unmeasured')
                for i, band in enumerate(names, 1):
                    dst.set_band_description(i, band)
                for row in range(0, rows, size):
                    for col in range(0, cols, size):
                        progress(12 + 82*complete/total, f'Enhancing native grid tile {complete+1} of {total}')
                        y, x = int(region.row_off)+row, int(region.col_off)+col
                        h, w = min(size, rows-row), min(size, cols-col)
                        top, left = max(0, y-halo), max(0, x-halo)
                        bottom, right = min(guide.height, y+h+halo), min(guide.width, x+w+halo)
                        source_window = Window(left, top, right-left, bottom-top)
                        patch = np.stack([(src.read(1, window=source_window)*scale+offset)*10000
                                          for src, scale, offset in inputs], axis=-1)
                        valid = minimum_filter(np.isfinite(patch).all(axis=-1), size=2*halo+1, mode='nearest')
                        core = np.s_[y-top:y-top+h, x-left:x-left+w]
                        estimate, device = predict(np.where(np.isfinite(patch), patch, 0))
                        values = estimate[core] / 10000
                        valid = valid[core] & np.isfinite(values).all(axis=-1)
                        window = Window(col, row, w, h)
                        if geometry:
                            valid &= geometry_mask([geometry], out_shape=(h, w),
                                                   transform=dst.window_transform(window), invert=True)
                        valid_pixels += int(valid.sum())
                        dst.write(np.moveaxis(np.where(valid[..., None], values, nodata), -1, 0), window=window)
                        complete += 1
            if not valid_pixels:
                raise ValueError('No DSen2 pixels have valid science inputs throughout the model neighbourhood')
        progress(97, 'Verifying the derived GeoTIFF')
        output_hash = digest(temporary)
        temporary.replace(destination)
        task.update(status='complete', progress=100, stage='Enhanced bands ready for export',
                    artifact=f'/api/segmentations/{task["id"]}/enhanced.tif', bands=list(names),
                    provenance={'weights_sha256': checksum, 'output_sha256': output_hash, 'device': device,
                                'source_resolution_m': resolution, 'output_grid_m': 10, 'valid_pixels': valid_pixels,
                                'calibration_sha256': next(iter(sources.values()))[3]['radiometry']['metadata_sha256'],
                                'units': 'TOA reflectance', 'source_product': next(iter(sources.values()))[3]['radiometry']['product_uri'],
                                'input_bands': {band: {'sha256': value[3]['sha256'], 'scale': value[1], 'offset': value[2]}
                                                for band, value in sources.items()},
                                'derived': True, 'reference_eligible': False},
                    validation={'status': 'unmeasured', 'reason': 'Native 10 m accuracy has not been measured for this product'})
        persist(task)
        published = True
    except Exception as exc:
        task.update(status='failed', stage='Super-resolution needs attention', error=str(exc))
        persist(task)
    finally:
        temporary.unlink(missing_ok=True)
        if not published:
            destination.unlink(missing_ok=True)
    return task
