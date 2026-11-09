"""Download only exact NASA HLS v2 products with a user-supplied Earthdata token."""
from contextlib import contextmanager
import re
import ssl
from urllib.parse import parse_qs, urlsplit

from imagery_products import HLS_AUXILIARY, HLS_SCIENCE
from offline_policy import require_online
from public_raster_transport import pinned_response, public_address, public_http_client

COLLECTIONS = {'hls-l30': ('L30', 'HLSL30'), 'hls-s30': ('S30', 'HLSS30')}
DATA_HOST = 'data.lpdaac.earthdatacloud.nasa.gov'
DELIVERY_HOST = 'd1nklfio7vscoe.cloudfront.net'
CMR_ROOT = 'https://cmr.earthdata.nasa.gov/stac/LPCLOUD/collections'
TOKEN_REQUIRED = 'Enter an Earthdata user token from your NASA profile in this download window.'
TOKEN_REJECTED = 'NASA did not authorize this download. Check your Earthdata token and approve LP DAAC Data Pool in NASA Earthdata if prompted.'


def product_route(catalog, item_id):
    if catalog not in COLLECTIONS or not isinstance(item_id, str):
        raise ValueError('Select a NASA HLS L30 or S30 v2 product.')
    sensor, collection = COLLECTIONS[catalog]
    if not re.fullmatch(rf'HLS\.{sensor}\.T\d{{2}}[A-Z]{{3}}\.\d{{7}}T\d{{6}}\.v2\.0', item_id, re.ASCII):
        raise ValueError('Select an exact NASA HLS v2 acquisition.')
    return collection, f'/lp-prod-protected/{collection}.020/{item_id}/'


async def exact_product(catalog, item_id):
    """Resolve only the selected public CMR identity. Never send the user token here."""
    require_online('NASA HLS download')
    collection, folder = product_route(catalog, item_id)
    endpoint = f'{CMR_ROOT}/{collection}_2.0/items/{item_id}'
    async with public_http_client([endpoint], timeout=20) as client:
        async with client.stream('GET', endpoint) as response:
            if response.status_code != 200:
                raise ValueError('NASA CMR could not resolve this exact HLS acquisition.')
            raw = bytearray()
            async for chunk in response.aiter_bytes(chunk_size=65536):
                raw.extend(chunk)
                if len(raw) > 256 * 1024:
                    raise ValueError('NASA HLS metadata exceeded its size limit.')
    import json
    data = json.loads(raw)
    if data.get('id') != item_id or data.get('collection') != collection + '_2.0':
        raise ValueError('NASA returned an unexpected HLS product identity.')
    bands = {}
    for band in [*HLS_SCIENCE[catalog], *sorted(HLS_AUXILIARY - {'QA'})]:
        suffix = 'Fmask' if band == 'FMASK' else band
        url = f'https://{DATA_HOST}{folder}{item_id}.{suffix}.tif'
        if data.get('assets', {}).get(suffix, {}).get('href') == url:
            bands[suffix] = url
        elif band in HLS_SCIENCE[catalog]:
            raise ValueError(f'NASA HLS catalog is missing the expected {band} raster.')
    return {'id': item_id, 'name': item_id, 'catalog': catalog, 'sensor': catalog,
            'collection': collection + '_2.0', 'processing_level': 'HLS',
            'datetime': data['properties']['datetime'], 'assets': bands, 'product_url': endpoint}


def delivery_path(location, asset_path, token=None):
    if not location or len(location) > 8192 or any(ord(ch) < 33 or ord(ch) > 126 for ch in location):
        raise ValueError('NASA returned an invalid HLS download redirect.')
    target = urlsplit(location)
    suffix = asset_path.removeprefix('/lp-prod-protected/')
    prefix = r'/s3-[0-9a-f]{32}/lp-prod-protected\.s3\.us-west-2\.amazonaws\.com/'
    if (target.scheme != 'https' or target.netloc != DELIVERY_HOST or target.fragment
            or not re.fullmatch(prefix + re.escape(suffix), target.path)):
        raise ValueError('NASA returned a download redirect outside its approved HLS delivery route.')
    try:
        fields = parse_qs(target.query, strict_parsing=True, max_num_fields=32)
    except ValueError:
        raise ValueError('NASA returned an invalid signed HLS delivery link.') from None
    if (not {'Signature', 'Key-Pair-Id'} <= fields.keys()
            or not ('Policy' in fields or fields.get('Expires', [''])[0].isdigit())
            or any(not re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', name) or len(values) != 1
                   or any(ord(c) < 32 or ord(c) == 127 for c in values[0])
                   or (token and token in values[0]) for name, values in fields.items())):
        raise ValueError('NASA returned an invalid signed HLS delivery link.')
    return target.path + '?' + target.query


@contextmanager
def authorized_response(catalog, url, token):
    require_online('NASA HLS download')
    source = urlsplit(url)
    parts = source.path.split('/')
    item_id = parts[-2] if len(parts) >= 2 else ''
    _, folder = product_route(catalog, item_id)
    allowed = [*HLS_SCIENCE[catalog], 'Fmask', 'SAA', 'SZA', 'VAA', 'VZA']
    if url not in {f'https://{DATA_HOST}{folder}{item_id}.{band}.tif' for band in allowed}:
        raise ValueError('NASA HLS download is outside the approved product routes.')
    if not token or not re.fullmatch(r'[A-Za-z0-9._~-]{16,2048}', token):
        raise ValueError(TOKEN_REQUIRED)
    tls = ssl.create_default_context()
    headers = {'Accept': 'image/tiff, application/octet-stream'}
    with pinned_response('GET', DATA_HOST, public_address(DATA_HOST), source.path, tls,
                         {**headers, 'Authorization': 'Bearer ' + token}) as response:
        if response.status in (401, 403):
            raise ValueError(TOKEN_REJECTED)
        if response.status not in (301, 302, 303, 307, 308):
            yield response
            return
        location = response.getheader('Location', '')
        if urlsplit(location).hostname == 'urs.earthdata.nasa.gov':
            raise ValueError(TOKEN_REJECTED)
        route = delivery_path(location, source.path, token)
    # The signed URL authorizes this specific file. Never forward the Earthdata token.
    with pinned_response('GET', DELIVERY_HOST, public_address(DELIVERY_HOST), route, tls, headers) as response:
        yield response
