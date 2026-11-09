"""Read external shapefile polygons without extracting archive paths."""
from __future__ import annotations

import codecs
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import shapefile
from rasterio.crs import CRS
from rasterio.features import bounds, is_valid_geom
from rasterio.warp import transform_geom

MAX_BYTES = 64*1024*1024
MAX_EXPANDED_BYTES = 256*1024*1024
MAX_FEATURES = 10000
COMPONENTS = {'.shp','.shx','.dbf','.prj','.cpg'}


def read_polygons(files: list[tuple[str, bytes]], dataset: str = '', source_crs: str = '', encoding: str = '') -> tuple[dict, dict]:
    if not files or sum(len(data) for _,data in files)>MAX_BYTES:
        raise ValueError('Upload at most 64 MiB of shapefile components or a ZIP archive')
    pieces={}
    if len(files)==1 and files[0][0].lower().endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(files[0][1])) as archive:
            entries=[item for item in archive.infolist() if not item.is_dir()]
            if len(entries)>1000 or sum(item.file_size for item in entries)>MAX_EXPANDED_BYTES:
                raise ValueError('ZIP exceeds the 256 MiB expanded size or 1,000-file limit')
            for item in entries:
                path=PurePosixPath(item.filename.replace('\\','/'))
                if '__MACOSX' in path.parts or path.name.startswith('.') or path.suffix.lower() not in COMPONENTS:
                    continue
                key=(str(path.with_suffix('')).casefold(),path.suffix.lower())
                if key in pieces:raise ValueError('Archive has duplicate shapefile components')
                pieces[key]=archive.read(item)
    else:
        for name,data in files:
            path=PurePosixPath(name.replace('\\','/'))
            if path.suffix.lower() not in COMPONENTS:raise ValueError('Select .shp, .shx, .dbf, and optionally .prj/.cpg, or upload one ZIP')
            key=(str(path.with_suffix('')).casefold(),path.suffix.lower())
            if key in pieces:raise ValueError('Duplicate shapefile component')
            pieces[key]=data
    datasets=sorted({key for key,suffix in pieces if suffix=='.shp'})
    if not datasets:raise ValueError('No .shp file was found')
    if not dataset and len(datasets)>1:
        return {'datasets':datasets,'needs_dataset':True}, {}
    dataset=dataset or datasets[0]
    if dataset not in datasets:raise ValueError('Choose a dataset from the uploaded archive')
    missing=[suffix for suffix in ('.shp','.shx','.dbf') if (dataset,suffix) not in pieces]
    if missing:raise ValueError('Missing matching shapefile components: '+', '.join(missing))
    if source_crs:
        if not re.fullmatch(r'(?:EPSG:)?\d{3,6}',source_crs.strip(),re.IGNORECASE):raise ValueError('CRS override must be an EPSG code, such as EPSG:4326')
        crs=CRS.from_epsg(int(source_crs.split(':')[-1]))
    elif (dataset,'.prj') in pieces:
        crs=CRS.from_wkt(pieces[(dataset,'.prj')].decode('utf-8-sig'))
    else:
        raise ValueError('The .prj file is missing. Enter the actual source EPSG code; coordinates will not be guessed.')
    if not encoding:
        encoding=pieces.get((dataset,'.cpg'),b'utf-8').decode('utf-8-sig').strip()
    if encoding=='65001':encoding='utf-8'
    elif encoding.isdigit():encoding='cp'+encoding
    codecs.lookup(encoding)
    reader=shapefile.Reader(shp=io.BytesIO(pieces[(dataset,'.shp')]),shx=io.BytesIO(pieces[(dataset,'.shx')]),
                            dbf=io.BytesIO(pieces[(dataset,'.dbf')]),encoding=encoding)
    if reader.shapeType not in (5,15,25):raise ValueError('This import requires polygon or multipolygon shapefiles')
    if len(reader)>MAX_FEATURES:raise ValueError('Import at most 10,000 polygons at a time')
    fields=[field[0] for field in reader.fields if field[0]!='DeletionFlag']
    features=[];invalid=[];vertices=0
    try:
        for index,item in enumerate(reader.iterShapeRecords()):
            try:
                vertices+=len(item.shape.points)
                if vertices>2000000:raise OverflowError('Import at most 2 million polygon vertices at a time')
                geometry=item.shape.__geo_interface__
                if geometry.get('type') not in ('Polygon','MultiPolygon') or not is_valid_geom(geometry):raise ValueError('Empty or invalid polygon geometry')
                geometry=transform_geom(crs,'EPSG:4326',geometry)
                box=bounds(geometry)
                if not np.all(np.isfinite(box)) or box[0]<-180-1e-7 or box[2]>180+1e-7 or box[1]<-90-1e-7 or box[3]>90+1e-7:raise ValueError('Transformed coordinates are outside longitude/latitude bounds; check the source CRS')
                if box[0]<-180 or box[2]>180 or box[1]<-90 or box[3]>90:
                    def clamp(value):
                        if isinstance(value[0],(int,float)):return [max(-180,min(180,value[0])),max(-90,min(90,value[1]))]
                        return [clamp(part) for part in value]
                    geometry['coordinates']=clamp(geometry['coordinates'])
                properties=json.loads(json.dumps(item.record.as_dict(),default=str))
                features.append({'type':'Feature','id':str(index),'geometry':geometry,'properties':properties})
            except (OverflowError,UnicodeError):raise
            except Exception as exc:invalid.append({'feature':index,'reason':str(exc)})
    finally:reader.close()
    if not features:raise ValueError('No usable polygons were found. Check the geometry and source CRS.')
    boxes=[bounds(feature['geometry']) for feature in features]
    box=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]
    values={field:sorted({str(feature['properties'][field]) for feature in features if feature['properties'].get(field) not in (None,'')}) for field in fields}
    value_counts={field:len(items) for field,items in values.items()}
    values={field:items[:200] for field,items in values.items()}
    digest=hashlib.sha256()
    for name,data in sorted(files):digest.update(name.encode());digest.update(data)
    metadata={'dataset':dataset,'datasets':datasets,'feature_count':len(features),'invalid_count':len(invalid),
              'invalid_features':invalid[:20],'fields':fields,'field_values':values,'field_value_counts':value_counts,'source_crs':crs.to_string(),
              'output_crs':'EPSG:4326','encoding':encoding,'bbox':box,'sha256':digest.hexdigest(),
              'sample_features':[{'id':feature['id'],'properties':feature['properties']} for feature in features[:8]]}
    return metadata, {'type':'FeatureCollection','features':features}


def import_folder(data_dir: Path, import_id: str) -> Path:
    if not re.fullmatch(r'[0-9a-f-]{36}',import_id):raise ValueError('Invalid import identifier')
    return data_dir/'vector-imports'/import_id
