"""Read supplier radiometry without inferring coefficients from pixel magnitude."""
import hashlib
import math
from pathlib import Path
import re
from xml.etree import ElementTree

MAX_METADATA_BYTES=2*1024*1024
SAFE_NAMES={'MTD_MSIL1C.XML','MTD_MSIL2A.XML'}


def is_metadata(name):
    name=Path(name).name.upper()
    return name in SAFE_NAMES or bool(re.fullmatch(r'.+_MTL\.(TXT|XML|JSON)',name))


def band_name(name):
    match=re.fullmatch(r'B(0?[1-9]|1[0-2]|8A)',str(name).upper())
    if not match:return None
    n=match[1]
    return 'B8A' if n=='8A' else f'B{int(n):02d}'


def read_safe(path):
    path=Path(path)
    if path.is_symlink() or path.stat().st_size>MAX_METADATA_BYTES:raise ValueError('Unsafe or oversized Sentinel calibration metadata')
    content=path.read_bytes()
    if b'<!DOCTYPE' in content.upper() or b'<!ENTITY' in content.upper():raise ValueError('XML entities are not accepted in calibration metadata')
    try:root=ElementTree.fromstring(content)
    except ElementTree.ParseError as exc:raise ValueError('Invalid supplier calibration XML') from exc
    nodes=list(root.iter());name=lambda node:node.tag.rsplit('}',1)[-1]
    def unique(tag):
        values={(node.text or '').strip() for node in nodes if name(node)==tag}
        if len(values)!=1:raise ValueError('Missing or conflicting '+tag+' in Sentinel product metadata')
        return values.pop()
    level='L1C' if path.name.upper()=='MTD_MSIL1C.XML' else 'L2A'
    quant=float(unique('QUANTIFICATION_VALUE' if level=='L1C' else 'BOA_QUANTIFICATION_VALUE'))
    if not math.isfinite(quant) or quant<=0:raise ValueError('Invalid Sentinel quantification value')
    baseline=unique('PROCESSING_BASELINE');product=unique('PRODUCT_URI')
    mapping={}
    for node in nodes:
        if name(node)=='Spectral_Information':
            band=band_name(node.attrib.get('physicalBand',''));ident=node.attrib.get('bandId')
            if not band or ident is None or ident in mapping:raise ValueError('Ambiguous Sentinel physical band mapping')
            mapping[ident]=band
    if not mapping:raise ValueError('Sentinel metadata does not declare its physical band mapping')
    offsets={};tag='RADIO_ADD_OFFSET' if level=='L1C' else 'BOA_ADD_OFFSET'
    for node in nodes:
        if name(node)!=tag:continue
        ident=node.attrib.get('band_id');value=float(node.text or '')
        if ident not in mapping or mapping[ident] in offsets or not math.isfinite(value):raise ValueError('Invalid or duplicate Sentinel band offset')
        offsets[mapping[ident]]=value
    if not offsets:
        if float(baseline)>=4:raise ValueError('Sentinel baseline 04.00+ requires declared per-band offsets')
        offsets={band:0. for band in mapping.values()}
    if set(offsets)!=set(mapping.values()):raise ValueError('Sentinel per-band offset list is incomplete')
    images=[(node.text or '').strip().removesuffix('.jp2') for node in nodes if name(node) in ('IMAGE_FILE','IMAGE_FILE_2A')]
    return {'processing_level':level,'processing_baseline':baseline,'product_uri':product,'image_files':images,
            'metadata_path':str(path),'metadata_sha256':hashlib.sha256(content).hexdigest(),
            'quantification_value':quant,'input_units':'supplier digital numbers','output_units':'reflectance',
            'formula':'reflectance = (DN + band_add_offset) / quantification_value; DN 0 is nodata',
            'bands':{band:{'scale':1/quant,'offset':value/quant,'add_offset_dn':value,'nodata_dn':0} for band,value in offsets.items()}}


def for_band(path, base):
    """Search only the selected product's ancestors, bounded by the import root."""
    path,base=Path(path).resolve(),Path(base).resolve()
    if path.suffix.lower() in ('.tif','.tiff'):
        return landsat_for_band(path,base)
    for parent in path.parents:
        if not parent.is_relative_to(base):break
        candidates=[file for file in parent.iterdir() if file.name.upper() in SAFE_NAMES and file.is_file()]
        if len(candidates)>1:raise ValueError('Conflicting L1C/L2A metadata in one product folder')
        if candidates:
            metadata=read_safe(candidates[0])
            if parent.suffix.upper()=='.SAFE' and parent.name!=metadata['product_uri']:raise ValueError('Metadata product identity differs from the SAFE folder')
            relative=path.relative_to(parent).with_suffix('').as_posix()
            if metadata['image_files'] and relative not in metadata['image_files']:raise ValueError('This band is not listed in the supplier metadata')
            return metadata
    return None


def calibrate_access(access, metadata, saved_path):
    """Create a new recipe; never mutate an earlier run's input declaration."""
    import copy
    record=copy.deepcopy(access);sources=record.get('local_sources',[])
    if not sources:raise ValueError('This repair requires a saved separate-band SAFE product')
    for source in sources:
        original=Path(source.get('original_path') or source['path'])
        if metadata['product_uri'] not in original.parts:raise ValueError('Calibration metadata belongs to a different Sentinel product')
        relative=Path(*original.parts[original.parts.index(metadata['product_uri'])+1:]).with_suffix('').as_posix()
        if metadata['image_files'] and relative not in metadata['image_files']:raise ValueError('A saved band is not listed in the supplier metadata')
        names=source.get('band_names',[])
        if len(names)!=1 or original.suffix.lower()!='.jp2':raise ValueError('Calibration repair accepts original single-band SAFE JP2 files only')
        band=band_name(names[0])
        if band not in metadata['bands']:raise ValueError('No declared calibration for '+str(band))
        values=metadata['bands'][band]
        for key,wanted in [('scale_override',values['scale']),('offset_override',values['offset'])]:
            if source.get(key) is not None and not math.isclose(float(source[key]),wanted,rel_tol=1e-9,abs_tol=1e-12):raise ValueError('Existing explicit calibration conflicts with supplier metadata; review it before changing it')
        source.update(scale_override=values['scale'],offset_override=values['offset'],radiometry={**{k:v for k,v in metadata.items() if k!='bands'},'metadata_path':str(saved_path)})
    record.update(processing_level=metadata['processing_level'],calibration_metadata={**{k:v for k,v in metadata.items() if k!='bands'},'metadata_path':str(saved_path)},derived_from_access_id=access['id'])
    return record


def read_landsat(path):
    """Collection-2 L2 coefficients from their exact MTL groups, never L1 values."""
    import json
    path=Path(path)
    if path.is_symlink() or path.stat().st_size>MAX_METADATA_BYTES:raise ValueError('Unsafe or oversized Landsat metadata')
    content=path.read_bytes();groups={}
    def put(group,key,value):
        target=groups.setdefault(group,{})
        if key in target and target[key]!=value:raise ValueError('Conflicting Landsat MTL field '+key)
        target[key]=value
    if path.suffix.lower()=='.txt':
        stack=[]
        for line in content.decode('utf-8-sig').splitlines():
            if '=' not in line:continue
            key,value=(part.strip() for part in line.split('=',1));value=value.strip('"')
            if key=='GROUP':stack.append(value)
            elif key=='END_GROUP':
                if not stack or stack.pop()!=value:raise ValueError('Unbalanced Landsat MTL groups')
            elif stack:put(stack[-1],key,value)
        if stack:raise ValueError('Incomplete Landsat MTL')
    elif path.suffix.lower()=='.xml':
        if b'<!DOCTYPE' in content.upper() or b'<!ENTITY' in content.upper():raise ValueError('XML entities are not accepted')
        root=ElementTree.fromstring(content)
        for group in root.iter():
            for child in group:
                if not len(child):put(group.tag.rsplit('}',1)[-1],child.tag.rsplit('}',1)[-1],(child.text or '').strip())
    elif path.suffix.lower()=='.json':
        def unique(pairs):
            obj={}
            for key,value in pairs:
                if key in obj:raise ValueError('Duplicate MTL JSON key')
                obj[key]=value
            return obj
        doc=json.loads(content,object_pairs_hook=unique).get('LANDSAT_METADATA_FILE',{})
        groups={key:value for key,value in doc.items() if isinstance(value,dict)}
    else:raise ValueError('Unsupported MTL metadata format')
    product=groups.get('PRODUCT_CONTENTS',{});attrs=groups.get('IMAGE_ATTRIBUTES',{})
    ident=str(product.get('LANDSAT_PRODUCT_ID',''));level=str(product.get('PROCESSING_LEVEL',''))
    if not re.fullmatch(r'LC0[89]_L2(?:SP|SR)_\d{6}_\d{8}_\d{8}_02_[A-Z0-9]+',ident):raise ValueError('This MTL adapter requires a Landsat 8/9 Collection-2 Level-2 product')
    if level!=ident.split('_')[1]:raise ValueError('MTL processing level differs from product identity')
    if attrs.get('SPACECRAFT_ID')!='LANDSAT_'+str(int(ident[2:4])):raise ValueError('Landsat spacecraft differs from product identity')
    def coefficient(group,key):
        try:value=float(groups[group][key])
        except (KeyError,TypeError,ValueError) as exc:raise ValueError('Missing Landsat L2 coefficient '+key) from exc
        if not math.isfinite(value):raise ValueError('Non-finite Landsat coefficient')
        return value
    bands={};files={}
    for n in range(1,8):
        band=f'SR_B{n}';files[band]=product.get(f'FILE_NAME_BAND_{n}')
        if files[band]!=ident+'_'+band+'.TIF':raise ValueError('MTL band file identity differs from product')
        bands[band]={'scale':coefficient('LEVEL2_SURFACE_REFLECTANCE_PARAMETERS',f'REFLECTANCE_MULT_BAND_{n}'),'offset':coefficient('LEVEL2_SURFACE_REFLECTANCE_PARAMETERS',f'REFLECTANCE_ADD_BAND_{n}'),'nodata_dn':0,'output_units':'reflectance'}
    if level=='L2SP':
        band='ST_B10';files[band]=product.get('FILE_NAME_BAND_ST_B10')
        if files[band]!=ident+'_'+band+'.TIF':raise ValueError('MTL thermal band identity differs from product')
        bands[band]={'scale':coefficient('LEVEL2_SURFACE_TEMPERATURE_PARAMETERS','TEMPERATURE_MULT_BAND_ST_B10'),'offset':coefficient('LEVEL2_SURFACE_TEMPERATURE_PARAMETERS','TEMPERATURE_ADD_BAND_ST_B10'),'nodata_dn':0,'output_units':'kelvin'}
    return {'family':'landsat','processing_level':level,'product_uri':ident,'spacecraft':attrs['SPACECRAFT_ID'],'metadata_path':str(path),'metadata_sha256':hashlib.sha256(content).hexdigest(),'input_units':'supplier digital numbers','output_units':'per-band physical units','formula':'physical value = DN × declared Level-2 multiplier + declared Level-2 additive coefficient; DN 0 is nodata','band_files':files,'bands':bands}


def landsat_for_band(path,base):
    path,base=Path(path).resolve(),Path(base).resolve()
    match=re.fullmatch(r'(LC0[89]_L2(?:SP|SR)_\d{6}_\d{8}_\d{8}_02_[A-Z0-9]+)_(SR_B[1-7]|ST_B10)',path.stem,re.I)
    if not match:return None
    for parent in path.parents:
        if not parent.is_relative_to(base):break
        candidates=[p for p in parent.iterdir() if p.is_file() and p.name.upper() in {match[1].upper()+'_MTL'+ext for ext in ('.TXT','.XML','.JSON')}]
        if candidates:
            candidates.sort(key=lambda p:('.txt','.xml','.json').index(p.suffix.lower()))
            metadata=read_landsat(candidates[0]);band=match[2].upper()
            if metadata['band_files'].get(band,'').upper()!=path.name.upper():raise ValueError('Landsat band does not match its MTL')
            return metadata
    return None


def calibrate_landsat_access(access,metadata,saved_path):
    import copy
    record=copy.deepcopy(access);sources=record.get('local_sources',[])
    if not sources:raise ValueError('MTL attachment requires saved separate-band Landsat input')
    for source in sources:
        original=Path(source.get('original_path') or source['path']);names=source.get('band_names',[])
        band=names[0].upper() if len(names)==1 else None
        if band not in metadata['bands'] or original.name.upper()!=metadata['band_files'][band].upper():raise ValueError('MTL does not match every saved Landsat science band')
        calibration=metadata['bands'][band]
        for key,wanted in [('scale_override',calibration['scale']),('offset_override',calibration['offset'])]:
            if source.get(key) is not None and not math.isclose(float(source[key]),wanted,rel_tol=1e-9,abs_tol=1e-12):raise ValueError('Explicit calibration conflicts with MTL; review before changing it')
        source.update(scale_override=calibration['scale'],offset_override=calibration['offset'],radiometry={**{k:v for k,v in metadata.items() if k!='bands'},'metadata_path':str(saved_path),'output_units':calibration['output_units']})
    record.update(processing_level=metadata['processing_level'],calibration_metadata={**{k:v for k,v in metadata.items() if k!='bands'},'metadata_path':str(saved_path)},derived_from_access_id=access['id'])
    return record
