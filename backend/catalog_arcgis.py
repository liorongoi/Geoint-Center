"""Explicit read contract for an internal ArcGIS Enterprise FeatureServer layer."""
import json
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

from internal_http import InternalHTTP

DRIVER = 'arcgis_features'
INVENTORY_DRIVER = 'arcgis_inventory'
TABLE = 'layer.imagery'
READ_CONTRACT = ('ArcGIS Enterprise 11.x query-only FeatureServer layer. Read layer metadata and one bounded '
                 'GeoJSON page using fixed GET operations. Links, edits, arbitrary filters and remote assets are excluded.')
FIELD_TYPES = {'esriFieldTypeOID', 'esriFieldTypeString', 'esriFieldTypeDate', 'esriFieldTypeDouble',
               'esriFieldTypeSingle', 'esriFieldTypeInteger', 'esriFieldTypeSmallInteger',
               'esriFieldTypeBigInteger', 'esriFieldTypeGUID', 'esriFieldTypeGlobalID'}
SERVICE_PATH = r'(?:/[A-Za-z0-9_.-]+)*/rest/services(?:/[A-Za-z0-9_.-]+)+/FeatureServer'
LAYER_ID = r'(?:0|[1-9][0-9]{0,8})'
FIELD = re.compile(r'[A-Za-z_][A-Za-z0-9_]{0,79}')
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
MILLISECOND = timedelta(milliseconds=1)


def utc_instant_field(data, columns):
    info = data.get('timeInfo')
    if not isinstance(info, dict) or data.get('datesInUnknownTimezone') not in (None, False):
        return None
    start = info.get('startTimeField')
    if not isinstance(start, str) or columns.get(start) != 'esriFieldTypeDate' or info.get('endTimeField') not in (None, '', start):
        return None
    reference = info.get('timeReference')
    if not isinstance(reference, dict) or reference.get('timeZone') not in ('UTC', 'Etc/UTC'):
        return None
    for ref in (reference, data.get('dateFieldsTimeReference')):
        if ref is not None and (not isinstance(ref, dict)
                or ref.get('timeZone') not in ('UTC', 'Etc/UTC')
                or ref.get('timeZoneIANA') not in (None, 'UTC', 'Etc/UTC')
                or ref.get('respectsDaylightSaving') not in (None, False)):
            return None
    return start


def object_id(value):
    if not re.fullmatch(r'(?:0|[1-9][0-9]{0,15})', value) or int(value) > 2**53 - 1:
        raise ValueError('Enter one ArcGIS object ID as a nonnegative decimal integer up to 9007199254740991')
    return int(value)


def client(endpoint, headers, ca_file, *, inventory=False):
    url = urlsplit(endpoint)
    path = url.path
    expected = SERVICE_PATH if inventory else SERVICE_PATH + '/' + LAYER_ID
    if (not re.fullmatch(expected, path)
            or any(part in ('.', '..') for part in path.split('/'))):
        if inventory:
            raise ValueError('Enter the full internal ArcGIS FeatureServer service URL ending in /FeatureServer')
        raise ValueError('Enter the full internal ArcGIS FeatureServer layer URL, including its numeric layer ID')
    if url.scheme != 'https':
        raise ValueError('ArcGIS catalog connections require verified HTTPS')
    return InternalHTTP(endpoint, headers, ca_file, timeout=10)


def query_only(data):
    version = data.get('currentVersion')
    if type(version) not in (int, float) or not 11 <= version < 12:
        raise ValueError('This contract requires ArcGIS Enterprise 11.x')
    capabilities = data.get('capabilities')
    if not isinstance(capabilities, str) or {v.strip().lower() for v in capabilities.split(',')} != {'query'}:
        raise ValueError('ArcGIS must expose Query only. Editing, sync and unreviewed capabilities are rejected.')
    return version


def inventory(http):
    """List feature-layer IDs from one metadata GET. A layer needs its own verification."""
    data = http.json('GET', '', query={'f': 'json'}, maximum=256 * 1024)
    if not isinstance(data, dict) or 'error' in data:
        raise ValueError('ArcGIS did not return FeatureServer service metadata')
    version = query_only(data)
    layers = data.get('layers')
    if not isinstance(layers, list) or not 1 <= len(layers) <= 100:
        raise ValueError('Choose a FeatureServer service with one to 100 layers')
    seen = set()
    candidates = []
    for layer in layers:
        if not isinstance(layer, dict):
            raise ValueError('Invalid ArcGIS layer inventory')
        ident, name = layer.get('id'), layer.get('name')
        if (type(ident) is not int or not 0 <= ident <= 999999999 or ident in seen
                or not isinstance(name, str) or not 1 <= len(name.strip()) <= 160
                or any(ord(c) < 32 or ord(c) == 127 for c in name)):
            raise ValueError('ArcGIS layer inventory has an invalid name or duplicate/invalid ID')
        seen.add(ident)
        if layer.get('type') in (None, 'Feature Layer'):
            candidates.append({'id': ident, 'name': name})
    if not candidates:
        raise ValueError('The service returned no supported feature layers')
    return {'version': version, 'layers': candidates}


def schema(http):
    layer_id = int(http.prefix.rsplit('/', 1)[1])
    data = http.json('GET', '', query={'f':'json'})
    if not isinstance(data, dict) or 'error' in data or data.get('id') != layer_id:
        raise ValueError('ArcGIS did not return the selected FeatureServer layer metadata')
    version = query_only(data)
    advanced = data.get('advancedQueryCapabilities')
    formats = data.get('supportedQueryFormats')
    limit = data.get('maxRecordCount')
    if (not isinstance(advanced,dict) or advanced.get('supportsPagination') is not True
            or not isinstance(formats,str) or 'geojson' not in {f.strip().lower() for f in formats.split(',')}
            or type(limit) is not int or limit < 1):
        raise ValueError('The layer must support bounded pagination and GeoJSON queries')
    fields = data.get('fields')
    if not isinstance(fields,list) or len(fields)>500:
        raise ValueError('The ArcGIS layer field inventory exceeds its inspection limit')
    columns = {}
    for field in fields:
        if not isinstance(field,dict):
            raise ValueError('Invalid ArcGIS field inventory')
        name, kind = field.get('name'),field.get('type')
        if isinstance(name,str) and FIELD.fullmatch(name) and isinstance(kind,str) and kind in FIELD_TYPES:
            if name in columns:
                raise ValueError('ArcGIS layer declares duplicate fields')
            columns[name]=kind
    oid = data.get('objectIdField')
    if not isinstance(oid,str) or columns.get(oid)!='esriFieldTypeOID':
        raise ValueError('A stable ArcGIS object identifier is required')
    geometry = data.get('type')=='Feature Layer' and data.get('geometryType') in (
        'esriGeometryPoint','esriGeometryMultipoint','esriGeometryPolyline','esriGeometryPolygon')
    if not geometry:
        raise ValueError('Choose a supported ArcGIS feature layer')
    columns['@geometry']='CRS84 geometry'
    result = {'layer_id':layer_id,'version':version,'max_records':min(limit,100),'oid':oid,
            'geometry':geometry,'columns':[{'name':name,'type':kind} for name,kind in sorted(columns.items())]}
    time_field = utc_instant_field(data, columns)
    if time_field:
        result['utc_instant_field'] = time_field
    return result


def preview(http, expected, table, mapping, limit, exact_id=None, bbox=None, window=None):
    from catalog_connections import geojson_snapshot, stac_mounted_path, verify_catalog_bounds
    columns = {column['name']:column['type'] for column in expected['columns']}
    if (table!=TABLE or set(mapping.values())-columns.keys()
            or columns.get(mapping.get('path'))!='esriFieldTypeString'
            or any(field=='@geometry' and role!='geometry' for role,field in mapping.items())
            or ('geometry' in mapping and mapping['geometry']!='@geometry')):
        raise ValueError('Map discovered fields, a text image path and optional native feature geometry')
    time_field = expected.get('utc_instant_field')
    if window and not time_field:
        raise ValueError('Reconnect a time-enabled ArcGIS layer that declares a single UTC date field')
    actual = schema(http)
    if 'utc_instant_field' not in expected:
        actual.pop('utc_instant_field', None)  # Preserve ordinary previews of older saved connections.
    if actual != expected:
        raise ValueError('ArcGIS layer permissions, version or fields changed. Reconnect before previewing.')
    fields = set(mapping.values())-{'@geometry'} | {expected['oid']}
    if window:
        fields.add(time_field)
        first,last = ((value-EPOCH)//MILLISECOND for value in window)
    count = 1 if exact_id is not None else min(limit,expected['max_records'])
    geometry = bbox is not None or mapping.get('geometry')=='@geometry'
    query = {'f':'geojson','where':'1=1','outFields':','.join(sorted(fields)),'orderByFields':expected['oid'],
             'resultOffset':0,'resultRecordCount':count,'returnGeometry':'true' if geometry else 'false',
             'returnZ':'false','returnM':'false','outSR':4326}
    if exact_id is not None:
        query['objectIds'] = exact_id
    if bbox is not None:
        query.update(geometry=','.join(format(value,'.17g') for value in bbox),
                     geometryType='esriGeometryEnvelope',inSR=4326,spatialRel='esriSpatialRelIntersects')
    if window:
        query['time'] = f'{first},{last}'
    page = http.json('GET','/query',query=query)
    if not isinstance(page,dict) or 'error' in page or not isinstance(page.get('features'),list) or len(page['features'])>count:
        raise ValueError('ArcGIS returned an invalid or oversized GeoJSON page')
    _, records = geojson_snapshot(json.dumps(page,ensure_ascii=False,separators=(',',':')).encode())
    if bbox is not None:
        verify_catalog_bounds([{'geometry':row['@geometry']} for row in records],bbox)
    if exact_id is not None:
        for row in records:
            returned_id = row.get(expected['oid'])
            feature_id = row['@id']
            if (type(returned_id) is not int or returned_id != exact_id
                    or feature_id is not None and str(feature_id) != str(exact_id)):
                raise ValueError('ArcGIS did not return the requested object ID')
    if any(mapping['path'] not in row for row in records):
        raise ValueError('ArcGIS response omitted the mapped image-path field')
    rows = []
    for row in records:
        mapped = {key:(stac_mounted_path(row[field]) if key=='path' else row.get(field))
                  for key,field in mapping.items()}
        if window:
            stamp = row.get(time_field)
            if type(stamp) is not int or not first <= stamp <= last:
                raise ValueError('ArcGIS returned a missing, invalid or out-of-window UTC timestamp')
            mapped['layer_time_utc'] = (EPOCH+stamp*MILLISECOND).isoformat(timespec='milliseconds').replace('+00:00','Z')
        rows.append(mapped)
    return rows
