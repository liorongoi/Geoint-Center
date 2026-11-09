"""Exact, read-only Copernicus Sentinel-2 SAFE downloads and checked extraction."""
import asyncio
from contextlib import contextmanager
import json
from pathlib import Path, PureWindowsPath
import re
import ssl
import stat
import time
from urllib.parse import urlsplit
from zipfile import ZipFile, BadZipFile

from imagery_products import SUFFIXES, inspect, products, standalone_auxiliary
from offline_policy import require_online
from public_raster_transport import TemporaryProviderError, pinned_response, public_address, public_http_client

CATALOGS = {'sentinel-2': 'L2A', 'sentinel-2-l1c': 'L1C'}
IDENTITY = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
STAC = 'https://stac.dataspace.copernicus.eu/v1/collections'
DOWNLOAD_HOST = 'download.dataspace.copernicus.eu'
ZIPPER_HOST = 'zipper.dataspace.copernicus.eu'
PRODUCT_PATH = r'/odata/v1/Products\([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\)/\$value'
MAX_ARCHIVE = 6 * 1024**3
MAX_EXPANDED = 12 * 1024**3
AUTH_ERROR = 'Copernicus did not accept the sign-in. Check the username, password and current two-factor code if enabled.'


async def fetch_json(url, *, form=None):
    require_online('Copernicus download')
    async with public_http_client([url], timeout=25, allow_search_post=form is not None) as client:
        async with client.stream('POST' if form else 'GET', url, data=form) as response:
            if response.status_code != 200:
                raise ValueError(AUTH_ERROR if form else 'Copernicus could not resolve the exact Sentinel-2 product.')
            raw = bytearray()
            async for chunk in response.aiter_bytes(chunk_size=65536):
                raw.extend(chunk)
                if len(raw) > 256 * 1024:
                    raise ValueError('Copernicus response exceeded its size limit.')
    try:
        data = json.loads(raw)
    except ValueError:
        raise ValueError('Copernicus returned an invalid response.') from None
    if not isinstance(data, dict):
        raise ValueError('Copernicus returned an invalid response.')
    return data


async def exact_product(catalog, ident):
    level = CATALOGS.get(catalog)
    if not level or not isinstance(ident, str) or not re.fullmatch(
        rf'S2[ABCD]_MSI{level}_\d{{8}}T\d{{6}}_N\d{{4}}_R\d{{3}}_T\d{{2}}[A-Z]{{3}}_\d{{8}}T\d{{6}}', ident, re.ASCII
    ):
        raise ValueError('Select an exact Sentinel-2 L1C or L2A product.')
    collection = 'sentinel-2-' + level.lower()
    source = f'{STAC}/{collection}/items/{ident}'
    data = await fetch_json(source)
    if data.get('id') != ident or data.get('collection') != collection:
        raise ValueError('Copernicus returned an unexpected product identity.')
    package = data.get('assets', {}).get('Product', {})
    path = urlsplit(package.get('href', '')).path
    if package.get('href') != f'https://{DOWNLOAD_HOST}{path}' or not re.fullmatch(PRODUCT_PATH, path):
        raise ValueError('Copernicus did not supply an approved SAFE package route.')
    return {'id': ident, 'name': ident, 'catalog': catalog, 'sensor': 'sentinel-2',
            'processing_level': level, 'collection': collection, 'product_url': source}, package['href']


class DownloadSession:
    """Private in-memory token pair. Tasks sharing it serialize token refresh."""
    def __init__(self, data):
        self.lock = asyncio.Lock()
        self.accept(data)

    def accept(self, data):
        access, refresh = data.get('access_token'), data.get('refresh_token')
        if not all(isinstance(token, str) and re.fullmatch(r'[A-Za-z0-9._~-]{16,8192}', token)
                   for token in (access, refresh)):
            raise ValueError('Copernicus did not return usable download credentials.')
        expiry = data.get('expires_in')
        if type(expiry) is not int or not 1 <= expiry <= 86400:
            raise ValueError('Copernicus returned an invalid session lifetime.')
        self.access = access
        self.refresh = refresh
        self.expiry = time.monotonic() + expiry

    async def token(self):
        async with self.lock:
            if self.expiry - time.monotonic() < 20:
                self.accept(await fetch_json(IDENTITY, form={'client_id': 'cdse-public',
                    'grant_type': 'refresh_token', 'refresh_token': self.refresh}))
            return self.access


async def sign_in(username, password, totp):
    if (not isinstance(username, str) or not 1 <= len(username.strip()) <= 256
            or not isinstance(password, str) or not 1 <= len(password) <= 1024
            or not isinstance(totp, str) or (totp and not re.fullmatch(r'[0-9]{6,8}', totp))):
        raise ValueError('Enter your Copernicus username and password, and a current code if two-factor authentication is enabled.')
    form = {'grant_type': 'password', 'client_id': 'cdse-public', 'username': username.strip(), 'password': password}
    if totp:
        form['totp'] = totp
    return DownloadSession(await fetch_json(IDENTITY, form=form))


@contextmanager
def archive_response(url, token):
    require_online('Copernicus download')
    path = urlsplit(url).path
    if url != f'https://{DOWNLOAD_HOST}{path}' or not re.fullmatch(PRODUCT_PATH, path):
        raise ValueError('Copernicus SAFE package route is outside the approved download service.')
    headers = {'Accept': 'application/zip, application/octet-stream', 'Authorization': 'Bearer ' + token}
    tls = ssl.create_default_context()
    with pinned_response('GET', DOWNLOAD_HOST, public_address(DOWNLOAD_HOST), path, tls, headers) as response:
        if response.status not in (301, 302, 303, 307, 308):
            yield checked_archive(response)
            return
        target = response.getheader('Location', '')
        if target != f'https://{ZIPPER_HOST}{path}':
            raise ValueError('Copernicus returned an unsupported SAFE package redirect.')
    # Copernicus documents the zipper origin as its authenticated product service.
    with pinned_response('GET', ZIPPER_HOST, public_address(ZIPPER_HOST), path, tls, headers) as response:
        yield checked_archive(response)


def checked_archive(response):
    if response.status in (429, 502, 503, 504):
        raise TemporaryProviderError(f'Copernicus download service returned HTTP {response.status}. Resume to retry.')
    if response.status in (401, 403):
        raise ValueError('Copernicus rejected the download session. Sign in again to resume.')
    if response.status != 200 or response.getheader('Content-Encoding', 'identity').lower() != 'identity':
        raise ValueError(f'Copernicus did not return a SAFE archive (HTTP {response.status}).')
    length = response.getheader('Content-Length')
    if length is not None and (not length.isdecimal() or not 0 < int(length) <= MAX_ARCHIVE):
        raise ValueError('Copernicus SAFE archive exceeds the allowed download size.')
    return response, int(length) if length is not None else None


def extract_safe(archive_path, destination, name, level, check_stop):
    """Extract only regular files beneath the expected SAFE root into an empty private folder."""
    try:
        with ZipFile(archive_path) as archive:
            members = archive.infolist()
            if not members or len(members) > 4096 or sum(info.file_size for info in members) > MAX_EXPANDED:
                raise ValueError('SAFE archive exceeds file-count or expanded-size limits.')
            seen = set()
            for info in members:
                parts = info.filename.rstrip('/').split('/')
                kind = stat.S_IFMT(info.external_attr >> 16)
                if (parts[0] != name or len(parts) > 16 or len(info.filename) > 1024
                        or any(not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', part)
                               or part.endswith('.') or PureWindowsPath(part).is_reserved() for part in parts)
                        or kind not in (0, stat.S_IFDIR if info.is_dir() else stat.S_IFREG)
                        or info.flag_bits & 1 or info.filename.rstrip('/').casefold() in seen):
                    raise ValueError('SAFE archive contains an unsafe, duplicate or unsupported entry.')
                seen.add(info.filename.rstrip('/').casefold())
                check_stop()
                path = destination.joinpath(*parts)
                if info.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, path.open('xb') as output:
                    remaining = info.file_size
                    while chunk := source.read(1024 * 1024):
                        check_stop()
                        remaining -= len(chunk)
                        if remaining < 0:
                            raise ValueError('SAFE archive has an invalid expanded length.')
                        output.write(chunk)
                    if remaining:
                        raise ValueError('SAFE archive was incomplete.')
    except BadZipFile:
        raise ValueError('The Copernicus download was not a complete SAFE ZIP archive.') from None
    if not (destination / name / f'MTD_MSI{level}.xml').is_file():
        raise ValueError('SAFE package is missing the original radiometric calibration XML.')
    return destination / name


def index_safe(folder, check_stop):
    base = folder.parent
    files = []
    for path in sorted(folder.rglob('*')):
        if path.is_symlink():
            raise ValueError('The local SAFE folder contains a symbolic link.')
        relative = path.relative_to(base).as_posix()
        if path.is_file() and path.suffix.lower() in SUFFIXES and not standalone_auxiliary(relative):
            check_stop()
            files.append(inspect(path, relative))
    found, issues = products(files, base)
    if not found or any(item['name'] != folder.name or item['status'] != 'ready'
                        or any(not source.get('radiometry') for source in item['sources']) for item in found):
        raise ValueError('SAFE bands or original calibration did not pass local import verification.')
    return found, issues
