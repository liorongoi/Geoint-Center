"""Offline Shapefile delivery with class attributes, CRS, encoding and lineage."""
import json,tempfile,zipfile
from pathlib import Path
import shapefile
from rasterio.crs import CRS

def shapefile_zip(geojson,destination):
 metadata,features=stream_features(geojson)
 with tempfile.TemporaryDirectory() as tmp:
  stem=Path(tmp)/'classes'
  with shapefile.Writer(str(stem),shapeType=shapefile.POLYGON,encoding='utf-8') as out:
   out.field('class_id','N',size=10,decimal=0);out.field('class_name','C',size=254)
   for feature in features:
    if len(feature['properties']['class_name'].encode('utf-8'))>254:raise ValueError('Class name exceeds the Shapefile UTF-8 field width; use GeoJSON to preserve it')
    out.shape(feature['geometry']);out.record(feature['properties']['class_id'],feature['properties']['class_name'])
  stem.with_suffix('.prj').write_text(CRS.from_epsg(4326).to_wkt());stem.with_suffix('.cpg').write_text('UTF-8')
  (Path(tmp)/'metadata.json').write_text(json.dumps(metadata,indent=2))
  with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as archive:
   for path in sorted(Path(tmp).iterdir()):archive.write(path,path.name)
 return destination


def stream_features(path):
 stream=Path(path).open(encoding='utf-8');buffer='';marker='"features":['
 while marker not in buffer:
  block=stream.read(65536)
  if not block:stream.close();raise ValueError('Invalid GeoJSON feature collection')
  buffer+=block
  if len(buffer)>4*1024*1024:stream.close();raise ValueError('GeoJSON header is too large')
 head,buffer=buffer.split(marker,1);metadata=json.loads(head+'"features":[]}').get('spectrascope',{})
 def features():
  nonlocal buffer
  decoder=json.JSONDecoder()
  try:
   while True:
    buffer=buffer.lstrip(' \r\n\t,')
    if buffer.startswith(']'):break
    try:feature,end=decoder.raw_decode(buffer)
    except json.JSONDecodeError:
     block=stream.read(65536)
     if not block:raise ValueError('Truncated GeoJSON feature')
     buffer+=block
     if len(buffer)>64*1024*1024:raise ValueError('One polygon exceeds the safe export buffer; use GeoJSON directly')
     continue
    yield feature;buffer=buffer[end:]
  finally:stream.close()
 return metadata,features()
