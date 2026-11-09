"""Bounded parsers for RPB, GDAL/GeoEye RPC text and DIMAP RPC XML sidecars."""
import math
import re

_SCALARS = {
    'lineOffset':'line_off','sampOffset':'samp_off','latOffset':'lat_off','longOffset':'long_off',
    'heightOffset':'height_off','lineScale':'line_scale','sampScale':'samp_scale','latScale':'lat_scale',
    'longScale':'long_scale','heightScale':'height_scale','errBias':'err_bias','errRand':'err_rand',
}
_COEFFICIENTS = {
    'lineNumCoef':'line_num_coeff','lineDenCoef':'line_den_coeff',
    'sampNumCoef':'samp_num_coeff','sampDenCoef':'samp_den_coeff',
}
_NUMBER = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?')
_FIELDS = re.compile(r'\s*([A-Za-z][A-Za-z0-9]*)\s*=\s*([^;]+);')
_TEXT_FIELD = re.compile(r'[ \t]*([A-Z][A-Z0-9_]*)[ \t]*:[ \t]*([^ \t]+)[ \t]*')
_OPTIONAL = {'err_bias','err_rand'}


def _fields(text):
    result={}
    position=0
    while text[position:].strip():
        match=_FIELDS.match(text,position)
        if not match or match[1] in result:raise ValueError('Invalid or duplicate RPB field')
        result[match[1]]=match[2].strip()
        position=match.end()
    return result


def _number(value):
    if not _NUMBER.fullmatch(value):raise ValueError('RPC fields require finite decimal numbers')
    number=float(value)
    if not math.isfinite(number):raise ValueError('RPC fields require finite decimal numbers')
    return number


def parse_rpb(content):
    try:text=content.decode('ascii')
    except UnicodeError as exc:raise ValueError('Use an ASCII RPC00B RPB sidecar') from exc
    parts=re.fullmatch(r'(.*?)\bBEGIN_GROUP\s*=\s*IMAGE\s*;?\s+(.*?)\bEND_GROUP\s*=\s*IMAGE\s*;?\s+END\s*;?\s*',text,re.S)
    if not parts:raise ValueError('Expected one RPB IMAGE group')
    header=_fields(parts[1]);fields=_fields(parts[2])
    if header.get('SpecId')!='"RPC00B"' or set(header)-{'SpecId','satId','bandId'}:
        raise ValueError('Use a declared RPC00B RPB sidecar')
    required=(set(_SCALARS)-{'errBias','errRand'})|set(_COEFFICIENTS)
    if not required.issubset(fields) or set(fields)-set(_SCALARS)-set(_COEFFICIENTS):
        raise ValueError('RPB IMAGE group is missing required coefficients or has unsupported fields')
    parameters={key:_number(fields[name]) for name,key in _SCALARS.items() if name in fields}
    for name,key in _COEFFICIENTS.items():
        value=fields[name]
        if not (value.startswith('(') and value.endswith(')')):
            raise ValueError('RPB coefficient arrays must contain twenty numbers')
        items=[_number(item.strip()) for item in value[1:-1].split(',')]
        if len(items)!=20:raise ValueError('RPB coefficient arrays must contain twenty numbers')
        parameters[key]=items
    return _validated_rpc(parameters)


def parse_rpc_text(content):
    """The GDAL/GeoEye format declares scalar keys and one-based numbered coefficients."""
    try:lines=content.decode('ascii').splitlines()
    except UnicodeError as exc:raise ValueError('Use an ASCII GDAL/GeoEye RPC text sidecar') from exc
    fields={}
    for line in lines:
        if not line.strip():continue
        match=_TEXT_FIELD.fullmatch(line)
        if not match or match[1] in fields:raise ValueError('Invalid or duplicate RPC text field')
        fields[match[1]]=_number(match[2])
    scalars=set(_SCALARS.values())
    arrays={key:[f'{key.upper()}_{index}' for index in range(1,21)] for key in _COEFFICIENTS.values()}
    required={key.upper() for key in scalars-_OPTIONAL}|{name for names in arrays.values() for name in names}
    if required-fields.keys() or fields.keys()-required-{key.upper() for key in _OPTIONAL}:
        raise ValueError('RPC text requires all numbered coefficients and supported scalar fields only')
    parameters={key:fields[key.upper()] for key in scalars if key.upper() in fields}
    parameters.update({key:[fields[name] for name in names] for key,names in arrays.items()})
    return _validated_rpc(parameters)


def _dimap_xml(content):
    from xml.etree import ElementTree

    try:text=content.decode('utf-8-sig')
    except UnicodeError as exc:raise ValueError('Use a UTF-8 DIMAP RPC sidecar') from exc
    if re.search(r'<!\s*(?:DOCTYPE|ENTITY)\b',text,re.I):
        raise ValueError('DIMAP RPC XML cannot contain a DTD or entity declaration')
    try:root=ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:raise ValueError('Invalid DIMAP RPC XML') from exc
    if root.tag not in ('Dimap_Document','PHR_DIMAP_Document'):
        raise ValueError('Expected a DIMAP RPC document')
    return root


def _one(parent,name):
    elements=parent.findall(name)
    if len(elements)!=1:raise ValueError(f'Expected exactly one DIMAP {name}')
    return elements[0]


def _value(parent,name):
    element=_one(parent,name)
    if len(element) or not element.text or not element.text.strip():
        raise ValueError(f'DIMAP {name} must be a plain value')
    return element.text.strip()


def parse_dimap_rpc(content):
    """Read the full-scene ground-to-image model without resolving DIMAP links."""
    root=_dimap_xml(content)
    metadata=_one(root,'Metadata_Identification')
    if (_value(metadata,'METADATA_FORMAT')!='DIMAP'
            or _one(metadata,'METADATA_FORMAT').get('version') not in ('2.0','3.0')
            or _value(metadata,'METADATA_SUBPROFILE')!='RPC'):
        raise ValueError('Expected an explicit DIMAP 2.0 or 3.0 RPC subprofile')
    profile=_value(metadata,'METADATA_PROFILE')
    offsets={'PHR_SENSOR':1,'S6_SENSOR':1,'S7_SENSOR':1,'PNEO_SENSOR':0}
    if profile not in offsets:
        raise ValueError('DIMAP RPC sensor profile and pixel origin are unsupported')
    global_model=_one(_one(root,'Rational_Function_Model'),'Global_RFM')
    validity=_one(global_model,'RFM_Validity')
    if len(global_model.findall('Inverse_Model')+global_model.findall('GroundtoImage_Values'))!=1:
        raise ValueError('Expected one DIMAP ground-to-image model')
    model=global_model.find('Inverse_Model')
    if model is None:model=_one(global_model,'GroundtoImage_Values')
    parameters={key:_number(_value(validity,key.upper())) for key in set(_SCALARS.values())-_OPTIONAL}
    parameters.update({key:[_number(_value(model,f'{key.upper()}_{i}')) for i in range(1,21)]
                       for key in _COEFFICIENTS.values()})
    parameters['line_off']-=offsets[profile]
    parameters['samp_off']-=offsets[profile]
    return _validated_rpc(parameters),profile


def parse_dimap_tile(content, tile_name, profile):
    """Read a single tile index and derive its origin in the full scene."""
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,240}\.(?:tif|tiff|jp2)',tile_name,re.I):
        raise ValueError('Enter the tile filename only, including its TIFF or JP2 extension')
    root=_dimap_xml(content)
    metadata=_one(root,'Metadata_Identification')
    if (_value(metadata,'METADATA_FORMAT')!='DIMAP'
            or _one(metadata,'METADATA_FORMAT').get('version') not in ('2.0','3.0')
            or _value(metadata,'METADATA_PROFILE')!=profile
            or _value(metadata,'METADATA_SUBPROFILE')!='PRODUCT'):
        raise ValueError('The DIMAP product metadata must match the RPC sensor profile')

    def positive(raw):
        if not isinstance(raw,str) or not re.fullmatch(r'[1-9][0-9]{0,9}',raw) or int(raw)>2**31-1:
            raise ValueError('DIMAP tile indices and dimensions must be positive integers')
        return int(raw)

    raster=_one(root,'Raster_Data')
    dimensions=_one(raster,'Raster_Dimensions')
    width,height=(positive(_value(dimensions,name)) for name in ('NCOLS','NROWS'))
    grid=_one(_one(dimensions,'Tile_Set'),'Regular_Tiling')
    if any(_value(grid,name)!='0' for name in ('OVERLAP_COL','OVERLAP_ROW')):
        raise ValueError('Only non-overlapping regular DIMAP tiles are supported')
    size=_one(grid,'NTILES_SIZE')
    tile_width,tile_height=(positive(size.get(name)) for name in ('ncols','nrows'))
    counts=_one(grid,'NTILES_COUNT')
    columns,rows=(positive(counts.get(name)) for name in ('ntiles_C','ntiles_R'))
    if (columns,rows)!=((width+tile_width-1)//tile_width,(height+tile_height-1)//tile_height):
        raise ValueError('DIMAP grid counts disagree with its scene and tile dimensions')
    files=_one(_one(raster,'Data_Access'),'Data_Files').findall('Data_File')
    found=[]
    for file in files:
        # Treat the declared filename as an identifier. Never resolve or open it.
        href=_one(file,'DATA_FILE_PATH').get('href','')
        if href.casefold()==tile_name.casefold():found.append(file)
    if len(found)!=1:
        raise ValueError('Choose a tile filename listed exactly once in the DIMAP product metadata')
    col,row=(positive(found[0].get(name)) for name in ('tile_C','tile_R'))
    if col>columns or row>rows or sum(
            (file.get('tile_C'),file.get('tile_R'))==(str(col),str(row)) for file in files)!=1:
        raise ValueError('The DIMAP tile index is outside the grid or duplicated')
    x,y=(col-1)*tile_width,(row-1)*tile_height
    return {'filename':tile_name,'column':col,'row':row,'column_offset':x,'row_offset':y,
            'width':min(tile_width,width-x),'height':min(tile_height,height-y)}


def _validated_rpc(parameters):
    from rasterio.rpc import RPC
    if (any(value<=0 for key,value in parameters.items() if key.endswith('_scale'))
            or abs(parameters['lat_off'])>90 or abs(parameters['long_off'])>180
            or any(parameters[key][0]==0 for key in ('line_den_coeff','samp_den_coeff'))):
        raise ValueError('Invalid RPC scales, reference longitude/latitude or denominator')
    return RPC(**parameters)
