"""Conservative grouping of local raster bands into revisioned image products.

No VRT, remote URI or generated code is used. Unknown layouts can declare their
band order in spectrascope-image.json or be explicitly reviewed by the user.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time

import rasterio
from sensor_metadata import for_band, band_name
from rasterio.warp import transform_bounds
from offline_policy import local_raster, raster_driver

SUFFIXES = {'.tif', '.tiff', '.jp2'}
BAND = r'(?:B(?:AND)?[_-]?\d{1,4}[A-Z]?|BLUE|GREEN|RED|NIR|SWIR[12]?)'
S2 = re.compile(r'^(T\d{2}[A-Z]{3}_\d{8}T\d{6})_(B(?:\d{2}|8A)|AOT|WVP|SCL|TCI)(?:_(\d+)m)?$', re.I)
LANDSAT = re.compile(r'^(L[COTEM]\d{2}_[A-Z0-9]+_\d{6}_\d{8}_\d{8}_\d{2}_[A-Z0-9]+)_(SR_B\d+|ST_B\d+|B\d+|QA_PIXEL|QA_RADSAT|SR_QA_AEROSOL)$', re.I)
HLS_BANDS=['BLUE','GREEN','RED','NIR','SWIR1','SWIR2']
HLS = re.compile(r'^(HLS\.(L30|S30)\.T\d{2}[A-Z]{3}\.(\d{7}T\d{6})\.v\d+\.\d+)\.(B(?:\d{2}|8A)|FMASK|QA|SAA|SZA|VAA|VZA)$', re.I)
CBERS_WPM = re.compile(r'^(CBERS_4A_WPM_(\d{8})_\d{3}_\d{3}_L[24])_(BAND[0-4])$', re.I)
HLS_SCIENCE = {'hls-l30': [f'B{i:02d}' for i in (*range(1,8),9,10,11)],
               'hls-s30': [f'B{i:02d}' for i in range(1,13)] + ['B8A']}
HLS_AUXILIARY = {'FMASK','QA','SAA','SZA','VAA','VZA'}


def sensor_family(value):
    if re.fullmatch(r'maxar|vantor|worldview[-_ ]?(?:[1-4]|legion)|wv0[1-4]|geoeye[-_ ]?1|ge01|quickbird(?:[-_ ]?2)?|qb02',
                    str(value or '').strip(),re.I):
        return 'maxar'
    return value or ''


def auxiliary_raster(relative):
    """Recognized browse/quality assets are not independent image acquisitions."""
    path=Path(relative)
    if any(part.lower() in ('qi_data','masks','mask','thumbnails','thumbnail','previews','preview','quicklooks') for part in path.parts[:-1]):return True
    stem=path.stem.upper()
    hls = HLS.fullmatch(stem)
    if hls and hls[4] in HLS_AUXILIARY:return True
    if re.search(r'_ST_(?:TRAD|URAD|DRAD|ATRAN|EMIS|EMSD|CDIST|QA)(?: \(\d+\))?$',stem):return True
    return bool(re.search(r'(?:^|[_\-.])(?:MSK(?:[_\-.]|$)|MASKS?(?:[_\-.]|$)|THUMB(?:NAIL)?(?:[_\-.]|$)|PREVIEW(?:[_\-.]|$)|QUICKLOOK(?:[_\-.]|$)|BROWSE(?:[_\-.]|$)|PVI(?:[_\-.]|$)|QA(?:_?(?:PIXEL|RADSAT|60|10|20))?(?:[_\-.]|$)|SCL(?:[_\-.]|$)|TCI(?:[_\-.]|$)|AOT(?:[_\-.]|$)|WVP(?:[_\-.]|$))',stem))


def standalone_auxiliary(relative):
    return auxiliary_raster(relative) and not any(pattern.fullmatch(Path(relative).stem) for pattern in (S2,LANDSAT,HLS))


def catalog_image_product(image):
    sources=image.get('sources',[])
    return not sources or any(not auxiliary_raster(s.get('relative_path') or s.get('original_path') or s.get('path','')) for s in sources)


def temporal_hls(level,names,dates):
    if str(level).upper()!='HLS' or names!=HLS_BANDS*3 or not isinstance(dates,list) or len(dates)!=3 or not all(isinstance(d,str) for d in dates) or len(set(dates))!=3:return False
    try:
        for d in dates:datetime.fromisoformat(d.replace('Z','+00:00'))
        return True
    except ValueError:return False


def hls_bundle(images):
    """Select six model channels per date from three compatible HLS granules."""
    from raster_pipeline import canonical_band

    if len(images)!=3 or any(i.get('sensor') not in HLS_SCIENCE or i.get('status')!='ready'
                             or i.get('processing_level')!='HLS' for i in images):
        raise ValueError('Select three ready HLS L30 or S30 acquisitions from Local imagery')
    ordered=sorted(images,key=lambda i:datetime.fromisoformat(i['datetime'].replace('Z','+00:00')))
    tiles=[re.fullmatch(r'HLS\.(?:L30|S30)\.(T\d{2}[A-Z]{3})\.\d{7}T\d{6}\.v\d+\.\d+',i['name'],re.I)
           for i in ordered]
    dates=[datetime.fromisoformat(i['datetime'].replace('Z','+00:00')).astimezone(timezone.utc).date().isoformat() for i in ordered]
    if len(set(dates))!=3 or not all(tiles) or len({tile[1].upper() for tile in tiles})!=1:
        raise ValueError('Use three distinct dates from the same named HLS tile')
    sources=[]
    grid=None
    for image,day in zip(ordered,dates):
        mapped=defaultdict(list)
        for source in image['sources']:
            if source['count']==1:
                mapped[canonical_band(source['band_names'][0],image['sensor'])].append(source)
        for band in HLS_BANDS:
            if len(mapped[band])!=1:raise ValueError(f'{image["name"]}: {band} needs one identified source band')
            source=mapped[band][0]
            with rasterio.open(local_raster(source['path']),driver=raster_driver(source['path'])) as raster:
                affine=raster.transform
                if (not raster.crs or not raster.crs.is_projected or raster.crs.linear_units_factor[0]!='metre'
                        or not math.isclose(affine.a,30,abs_tol=.01) or not math.isclose(affine.e,-30,abs_tol=.01)
                        or affine.b or affine.d):
                    raise ValueError('HLS bundle requires north-up 30 m projected rasters')
                current=(raster.crs,raster.width,raster.height,affine)
                if grid is not None and current!=grid:raise ValueError('HLS rasters have different grids or extents')
                grid=current
            sources.append({**source,'key':f'{day}:{band}','source_band_name':source['band_names'][0],
                            'band_names':[band],'acquisition_date':day})
    lineage=[{'image_id':i['id'],'source_revision':i.get('review_revision') or i['sha256']} for i in ordered]
    identity=hashlib.sha256(json.dumps(lineage,sort_keys=True).encode()).hexdigest()
    first=ordered[0]
    return {'id':'local-hls-'+identity[:24],'product_key':'hls-bundle:'+identity,'sha256':identity,
            'name':f'HLS {tiles[0][1].upper()} · {dates[0]} to {dates[-1]}','catalog':'local','collection':'local-imagery',
            'sensor':'hls-temporal','platform':'HLS temporal bundle','processing_level':'HLS','datetime':ordered[-1]['datetime'],
            'date_basis':'Three selected HLS acquisitions','temporal_dates':dates,'source_acquisitions':lineage,
            'sources':sources,'alternate_sources':[],'auxiliary_sources':[],'band_names':HLS_BANDS*3,
            'raster_bands':len(sources),'bytes':sum(s['bytes'] for s in sources),'bbox':first['bbox'],'geometry':first['geometry'],
            'completeness':{'state':'verified','expected_bands':HLS_BANDS*3,'found_bands':HLS_BANDS*3},
            'status':'ready','active':True,'warnings':[],'cloud_cover':None,'thumbnail':None,'product_url':'','assets':{}}


def now():
    return datetime.now(timezone.utc).isoformat()


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def band_sort(name):
    match = re.search(r'\d+', name)
    return (int(match[0]) if match else 100000, name.upper())


def wavelength_nm(tags):
    """Convert only a declared wavelength unit, never guess from magnitude."""
    tags={str(k).lower():v for k,v in tags.items()}
    value=tags.get('wavelength_nm')
    factor=1
    if value is None:
        value=tags.get('wavelength') or tags.get('center_wavelength')
        unit=str(tags.get('wavelength_units') or tags.get('wavelength_unit') or '').lower()
        if unit in ('um','µm','micrometer','micrometers','micrometres'):factor=1000
        elif unit not in ('nm','nanometer','nanometers','nanometres'):return None
    try:
        result=float(value)*factor
        return result if math.isfinite(result) and result>0 else None
    except (ValueError,TypeError):return None


def inspect(path, relative, cached=None):
    """Only open supported real local rasters; detect concurrent file writes."""
    path = Path(path)
    before = path.stat()
    stamp = [before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino, before.st_dev]
    if cached and cached.get('stamp') == stamp:
        return {**cached, 'path': str(path), 'relative_path': relative}
    with rasterio.open(local_raster(str(path)), driver=raster_driver(str(path))) as src:
        if src.driver not in ('GTiff', 'JP2OpenJPEG') or not src.crs:
            raise ValueError('A georeferenced GeoTIFF or JPEG2000 is required')
        if any('complex' in dtype for dtype in src.dtypes):
            raise ValueError('Complex-valued rasters require a dedicated adapter; imaginary values cannot be discarded')
        if not all(math.isfinite(float(v)) for v in [*src.scales,*src.offsets]):
            raise ValueError('Raster scale/offset metadata must contain finite numbers')
        bbox = list(transform_bounds(src.crs, 'EPSG:4326', *src.bounds, densify_pts=21))
        w, s, e, n = bbox
        if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
            raise ValueError('Unsupported bounds or antimeridian crossing')
        result = {'path': str(path), 'relative_path': relative, 'stamp': stamp,
                  'bytes': before.st_size, 'count': src.count, 'bbox': bbox,
                  'crs': str(src.crs), 'width': src.width, 'height': src.height,
                  'resolution': [abs(src.transform.a), abs(src.transform.e)],
                  'descriptions': list(src.descriptions), 'scales': list(src.scales), 'dtypes':list(src.dtypes),
                  'channel_metadata':[{str(k).lower():v for k,v in src.tags(i+1).items() if str(k).lower() in ('band_name','band_id','wavelength_nm','wavelength','center_wavelength','wavelength_units','wavelength_unit')} for i in range(src.count)],
                  'offsets': list(src.offsets), 'nodata':src.nodata if src.nodata is None or math.isfinite(src.nodata) else None,
                  'nodata_is_nan':src.nodata is not None and not math.isfinite(src.nodata),
                  'tags': {key.lower(): value for key, value in src.tags().items()
                           if key.lower() in ('product_id','scene_id','band_id','band_name','acquisition_datetime','sensor','satellite','processing_level','tifftag_datetime')}}
    result['sha256'] = checksum(path)
    result['verified_at'] = time.time()
    after = path.stat()
    if stamp != [after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino, after.st_dev]:
        raise ValueError('File changed while being read; waiting for a stable copy')
    return result


def inferred_key(file):
    p = Path(file['relative_path'])
    match = S2.fullmatch(p.stem)
    if match:
        # Preserve product version/level and granule identity, while collapsing
        # R10m/R20m/R60m alternative representations of the same spectral band.
        scope = next((part for part in p.parts if part.upper().endswith('.SAFE')), '')
        granule = ''
        if 'GRANULE' in p.parts:
            i = p.parts.index('GRANULE')
            granule = p.parts[i + 1] if i + 1 < len(p.parts) else ''
        level = 'L2A' if 'MSIL2A' in scope.upper() else 'L1C' if 'MSIL1C' in scope.upper() else 'unknown'
        parent = str(p.parent)
        if not scope:
            parent = re.sub(r'[/\\]R(?:10|20|60)m$', '', parent, flags=re.I)
            if re.fullmatch(BAND,Path(parent).name,re.I):parent=str(Path(parent).parent)
        return (f's2:{scope}:{granule}:{parent if not scope else ""}:{match[1]}:{level}', match[2].upper(), 'sentinel-2', False)
    match = LANDSAT.fullmatch(p.stem)
    if match:
        return ('landsat:' + match[1], match[2].upper(), 'landsat', False)
    match = HLS.fullmatch(p.stem)
    if match:
        return ('hls:' + match[1].upper(), match[4].upper(), 'hls-' + match[2].lower(), False)
    match = CBERS_WPM.fullmatch(p.stem)
    if match:
        return ('cbers-wpm:' + match[1].upper(), match[3].upper(), 'cbers-4a-wpm', False)
    tags=file.get('tags',{})
    scene=tags.get('product_id') or tags.get('scene_id')
    if scene:
        channel=(file.get('channel_metadata') or [{}])[0]
        band=(tags.get('band_id') or tags.get('band_name') or channel.get('band_id') or channel.get('band_name') or file['descriptions'][0]) if file['count']==1 else None
        identity=json.dumps([scene,tags.get('sensor'),tags.get('processing_level'),tags.get('acquisition_datetime')])
        return ('metadata:'+identity,band,tags.get('sensor',''),not bool(tags.get('acquisition_datetime')))
    if file['count'] > 1:
        return ('stack:' + p.as_posix(), None, '', False)
    match = re.fullmatch(rf'(.+?)[_-]({BAND})(?:[_-]\d+m)?', p.stem, re.I)
    if match:
        parent=p.parent.parent if re.fullmatch(BAND,p.parent.name,re.I) else p.parent
        return ('named:' + str(parent) + ':' + match[1], match[2].upper(), '', False)
    # Identically named image files placed in sibling B01/B02/... folders.
    if re.fullmatch(BAND, p.parent.name, re.I):
        return ('band-folders:' + str(p.parent.parent) + ':' + p.stem, p.parent.name.upper(), '', False)
    if re.fullmatch(BAND, p.stem, re.I):
        return ('bare:' + str(p.parent), p.stem.upper(), '', True)
    return ('stack:' + p.as_posix(), None, '', False)


def products(files, base, manifests=None):
    """Return products and explicit grouping warnings, without inventing bands."""
    base = Path(base)
    by_rel = {f['relative_path']: f for f in files}
    groups = defaultdict(list)
    options = {}
    claimed = set()
    issues = []
    for manifest_path, document in (manifests or []):
        key = 'manifest:' + str(Path(manifest_path).relative_to(base))
        try:
            entries = document['bands']
            if not isinstance(entries, list) or not entries:
                raise ValueError('bands must be a non-empty list')
            selected = []
            for entry in entries:
                if not isinstance(entry,dict) or not isinstance(entry.get('name'),str) or not entry['name'].strip():
                    raise ValueError('Each manifest band entry requires a name')
                relative = entry['path']
                if not isinstance(relative,str) or '\\' in relative or '://' in relative or ':' in relative:
                    raise ValueError('Manifest band paths must be local relative paths')
                target = (Path(manifest_path).parent / relative).resolve()
                if not target.is_relative_to(base.resolve()):
                    raise ValueError('Band manifest cannot refer outside this folder')
                rel = target.relative_to(base.resolve()).as_posix()
                if rel in claimed:
                    raise ValueError('A band is claimed by more than one manifest')
                if rel not in by_rel:
                    raise ValueError('Missing or unstable manifest band: ' + relative)
                names=entry.get('band_names')
                if names is not None and (not isinstance(names,list) or len(names)!=by_rel[rel]['count'] or not all(isinstance(n,str) and n.strip() for n in names)):
                    raise ValueError('band_names must name every channel in a multiband file')
                for field in ('scale','offset'):
                    if field in entry and (isinstance(entry[field],bool) or not isinstance(entry[field],(int,float)) or not math.isfinite(entry[field])):
                        raise ValueError(field+' must be a finite number')
                wave=entry.get('wavelength_nm')
                if wave is not None:
                    waves=wave if isinstance(wave,list) else [wave]
                    if len(waves)!=by_rel[rel]['count'] or any(v is not None and (isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0) for v in waves):
                        raise ValueError('wavelength_nm must give a positive nm value (or null) for every channel')
                selected.append((by_rel[rel], entry['name'], entry))
            if len({f['relative_path'] for f, _, _ in selected}) != len(selected):
                raise ValueError('Declare a multiband file once; its band_names give its channel order')
            groups[key] = selected
            claimed.update(f['relative_path'] for f, _, _ in selected)
            options[key] = {**document, 'explicit': True}
        except Exception as exc:
            issues.append({'path': str(manifest_path), 'error': str(exc)})
            # Files under an invalid manifest are held for review, not silently
            # reinterpreted as unrelated products.
            for rel in by_rel:
                if (base / rel).is_relative_to(Path(manifest_path).parent):
                    claimed.add(rel)
    for file in files:
        if file['relative_path'] in claimed:
            continue
        key, band, sensor, uncertain = inferred_key(file)
        if standalone_auxiliary(file['relative_path']):
            continue
        tags=file.get('tags',{})
        opt_name=None
        if tags.get('acquisition_datetime') and not key.startswith(('s2:','landsat:','hls:','cbers-wpm:','metadata:')):
            opt_name=key.split(':')[-1]
            key+='|acquisition:'+tags['acquisition_datetime']
        groups[key].append((file, band, {}))
        options[key] = {'sensor': sensor or tags.get('sensor') or tags.get('satellite',''), 'uncertain': uncertain}
        if key.startswith('s2:'):
            match=S2.fullmatch(Path(file['relative_path']).stem)
            safe=next((part for part in Path(file['relative_path']).parts if part.upper().endswith('.SAFE')),None)
            options[key]['name']=safe or match[1]
        if key.startswith('hls:'):
            match=HLS.fullmatch(Path(file['relative_path']).stem)
            options[key].update(name=match[1],processing_level='HLS',expected_bands=HLS_SCIENCE[sensor])
        if key.startswith('cbers-wpm:'):
            match=CBERS_WPM.fullmatch(Path(file['relative_path']).stem)
            options[key].update(name=match[1],processing_level=match[1].upper().split('_')[-1],expected_bands=[f'BAND{i}' for i in range(5)])
        if key.startswith('metadata:'):
            options[key].update(name=tags.get('product_id') or tags.get('scene_id'),sensor=tags.get('sensor',''),processing_level=tags.get('processing_level','unknown'),datetime=tags.get('acquisition_datetime'),datetime_source='Declared raster acquisition_datetime metadata')
        elif opt_name is not None:
            options[key].update(name=opt_name,sensor=tags.get('sensor') or sensor,processing_level=tags.get('processing_level','unknown'),datetime=tags['acquisition_datetime'],datetime_source='Declared raster acquisition_datetime metadata')
    result = []
    for key, members in sorted(groups.items()):
        opt = options[key]
        warnings = []
        selected = []
        auxiliary = []
        alternatives = []
        bands = defaultdict(list)
        for file, band, entry in members:
            if not opt.get('explicit') and band and (band in HLS_AUXILIARY or re.fullmatch(r'AOT|WVP|SCL|TCI|QA_PIXEL|QA_RADSAT|SR_QA_AEROSOL',band,re.I)):
                auxiliary.append({**file,'band':band,'excluded':'Quality, display or atmospheric retrieval product, not a measured spectral channel'})
                continue
            bands[band or file['relative_path']].append((file, band, entry))
        for band, choices in bands.items():
            if len(choices) > 1:
                # Choose finest S2 representation only where CRS and footprint
                # agree. Other duplicates must be reviewed explicitly.
                choices.sort(key=lambda item: abs(item[0]['resolution'][0] * item[0]['resolution'][1]))
                same = all(f['crs'] == choices[0][0]['crs'] and
                           max(abs(a-b) for a,b in zip(f['bbox'], choices[0][0]['bbox'])) < .002
                           for f, _, _ in choices)
                unique = len({tuple(f['resolution']) for f, _, _ in choices}) == len(choices)
                if not key.startswith('s2:') or not same or not unique:
                    warnings.append('Duplicate band ' + str(band) + '; review the file selection')
                alternatives.extend({**f, 'band': b} for f,b,_ in choices[1:])
            selected.append(choices[0])
        if not opt.get('explicit'):
            selected.sort(key=lambda item: band_sort(item[1] or item[0]['relative_path']))
        sources = []
        for file, band, entry in selected:
            names = entry.get('band_names') or ([band] if file['count'] == 1 and band else
                       [d or f'band_{i+1}' for i,d in enumerate(file['descriptions'])])
            if len(names) != file['count']:
                warnings.append('Band name count differs from raster channel count: ' + file['relative_path'])
                names = [d or f'band_{i+1}' for i,d in enumerate(file['descriptions'])]
            waves=[wavelength_nm(tags) for tags in file.get('channel_metadata',[])]
            wave=entry.get('wavelength_nm')
            if wave is None and any(v is not None for v in waves):wave=waves[0] if len(waves)==1 else waves
            sources.append({**file, 'key': band or file['relative_path'], 'band_names': names,
                            'wavelength_nm': wave,
                            'scale_override': entry.get('scale'), 'offset_override': entry.get('offset')})
        # SAFE JP2 values are encoded DN, including offsets since baseline 04.00.
        for source in sources:
            if Path(source['path']).suffix.lower() not in ('.jp2','.tif','.tiff'):continue
            try:
                metadata=for_band(source['path'],base)
                if metadata:
                    name=(source['band_names'][0].upper() if metadata.get('family')=='landsat' else band_name(source['band_names'][0])) if source['count']==1 else None
                    if name not in metadata['bands']:raise ValueError('Supplier metadata does not define calibration for '+str(name))
                    calibration=metadata['bands'][name]
                    if source.get('scale_override') is None and source.get('offset_override') is None and source['scales']==[1.] and source['offsets']==[0.]:
                        source.update(scale_override=calibration['scale'],offset_override=calibration['offset'],radiometry={**{k:v for k,v in metadata.items() if k!='bands'},'output_units':calibration.get('output_units',metadata['output_units'])})
                    opt['processing_level']=metadata['processing_level']
                elif any(p.suffix.upper()=='.SAFE' for p in Path(source['relative_path']).parents):
                    warnings.append('Missing SAFE product calibration XML; include MTD_MSIL1C.xml or MTD_MSIL2A.xml')
            except Exception as exc:
                warnings.append('Supplier calibration metadata: '+str(exc))
        if not sources:
            if auxiliary:issues.append({'path':key,'error':'Only quality/display/atmospheric assets were found; science bands are required'})
            continue
        if opt.get('uncertain'):
            warnings.append('Scene/acquisition identity is not fully declared; confirm that these files form one acquisition')
        if len(sources)==1 and sources[0]['count']==1 and not key.startswith('stack:') and not opt.get('explicit'):
            warnings.append('Only one separate band was found; select the complete product or confirm that this is intentionally a single-band image')
        if len({f['crs'] for f in sources}) > 1:
            warnings.append('Source CRS differs; review the band grouping before reprojection')
        # Reject non-overlapping bands, even when a filename suggests a match.
        west = max(f['bbox'][0] for f in sources); south = max(f['bbox'][1] for f in sources)
        east = min(f['bbox'][2] for f in sources); north = min(f['bbox'][3] for f in sources)
        if west >= east or south >= north:
            warnings.append('Band footprints do not overlap; this is not a usable single image')
        for f in sources[1:]:
            overlap = max(0, east-west)*max(0, north-south)
            area = (f['bbox'][2]-f['bbox'][0])*(f['bbox'][3]-f['bbox'][1])
            if area and overlap/area < .8:
                warnings.append('Band coverage differs substantially; verify scene identity')
                break
        raw_bands = [name for f in sources for name in f['band_names']]
        if len(set(raw_bands)) != len(raw_bands) and not temporal_hls(opt.get('processing_level'),raw_bands,opt.get('temporal_dates')):
            warnings.append('Repeated channel names; give each science channel a unique name')
        expected = opt.get('expected_bands', [])
        if not isinstance(expected,list) or not all(isinstance(n,str) for n in expected):
            warnings.append('expected_bands must be a list of names'); expected=[]
        if expected and set(expected)-set(raw_bands):
            warnings.append('Missing expected bands: ' + ', '.join(sorted(set(expected)-set(raw_bands))))
        processing = opt.get('processing_level', 'unknown')
        if key.startswith('s2:') and key.endswith(':L2A'): processing = 'L2A'
        if key.startswith('s2:') and key.endswith(':L1C'): processing = 'L1C'
        if key.startswith('landsat:') and '_L2' in key and processing=='unknown': processing = 'L2'
        sensor = sensor_family(opt.get('sensor', ''))
        if key.startswith('s2:') and processing != 'unknown':
            expected = {f'B{i:02d}' for i in range(1,13)} | {'B8A'}
            if processing == 'L2A': expected.discard('B10')
            missing = expected-set(raw_bands)
            if missing: warnings.append('Incomplete Sentinel-2 science bands: ' + ', '.join(sorted(missing)))
        elif key.startswith('s2:'):
            warnings.append('Sentinel-2 product level is unknown; confirm its expected bands and radiometric metadata')
        if key.startswith('landsat:') and '_L2' in key:
            expected = {f'SR_B{i}' for i in ([1,2,3,4,5,6,7] if re.search(r'L[COT]0[89]_', key) else [1,2,3,4,5,7])}
            missing = expected-set(raw_bands)
            if missing: warnings.append('Incomplete Landsat reflectance bands: ' + ', '.join(sorted(missing)))
        acquired = opt.get('datetime')
        date_basis = opt.get('datetime_source') or 'Acquisition time from explicit product manifest'
        if not acquired:
            match = re.search(r'\d{8}T\d{6}', key) if key.startswith('s2:') else None
            try:
                if match: acquired = datetime.strptime(match[0], '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc).isoformat()
                elif key.startswith('landsat:'):
                    acquired = datetime.strptime(key.split(':')[-1].split('_')[3], '%Y%m%d').replace(tzinfo=timezone.utc).isoformat()
                elif key.startswith('hls:'):
                    acquired = datetime.strptime(key.split('.')[-3], '%Y%jT%H%M%S').replace(tzinfo=timezone.utc).isoformat()
                elif key.startswith('cbers-wpm:'):
                    acquired = datetime.strptime(key.split('_')[3], '%Y%m%d').replace(tzinfo=timezone.utc).isoformat()
                date_basis = 'Acquisition date from product filename'
            except ValueError: acquired = None
        if not acquired:
            acquired = now(); date_basis = 'Catalog import time; acquisition date must be supplied for temporal inference'
        try:
            datetime.fromisoformat(acquired.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            warnings.append('Invalid acquisition datetime in manifest'); acquired = now()
        signature = [{k:f.get(k) for k in ('relative_path','sha256','band_names','wavelength_nm','scale_override','offset_override')} for f in sources]
        for item,source in zip(signature,sources):
            if source.get('radiometry'):
                item['radiometry']={k:v for k,v in source['radiometry'].items() if k!='metadata_path'}
        identity = hashlib.sha256(json.dumps([key,signature,sensor,processing,opt.get('datetime'),opt.get('temporal_dates')],sort_keys=True).encode()).hexdigest()
        bbox = [min(f['bbox'][0] for f in sources), min(f['bbox'][1] for f in sources),
                max(f['bbox'][2] for f in sources), max(f['bbox'][3] for f in sources)]
        w,s,e,n = bbox
        result.append({'id':'local-'+identity[:24], 'product_key':key,
            'name':opt.get('name') or (Path(sources[0]['relative_path']).name if len(sources)==1 else key.split(':')[-1]),
            'catalog':'local', 'collection':'local-imagery', 'platform':sensor or 'Local imagery',
            'sensor':sensor, 'processing_level':processing, 'datetime':acquired, 'date_basis':date_basis,
            'temporal_dates':opt.get('temporal_dates',[]),
            'bbox':bbox, 'geometry':{'type':'Polygon','coordinates':[[[w,s],[e,s],[e,n],[w,n],[w,s]]]},
            'band_names':raw_bands, 'raster_bands':len(raw_bands), 'sources':sources,
            'alternate_sources':alternatives, 'auxiliary_sources':auxiliary, 'bytes':sum(f['bytes'] for f in sources),
            'sha256':identity, 'warnings':list(dict.fromkeys(warnings)),
            'completeness':{'state':'incomplete' if expected and set(expected)-set(raw_bands) else 'verified' if expected else 'not-declared', 'expected_bands':sorted(expected), 'found_bands':raw_bands},
            'status':'needs-review' if warnings else 'ready', 'active':True,
            'cloud_cover':None, 'thumbnail':None, 'product_url':'', 'assets':{}})
    return result, issues
