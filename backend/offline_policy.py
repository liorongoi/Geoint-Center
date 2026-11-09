"""Strict desktop offline policy, shared by services and spawned model workers.

The desktop launcher enables this before importing ML/network libraries. Browser
requests have an independent allowlist in Electron. GDAL inputs are checked at
the raster boundary because its native HTTP client bypasses Python sockets.
"""
from __future__ import annotations
import ipaddress
import os
from pathlib import Path
import socket
import sys
from urllib.parse import urlsplit


def enabled() -> bool:
    if os.getenv('SPECTRASCOPE_OFFLINE', '').lower() in ('1', 'true', 'yes'):return True
    if os.getenv('SPECTRASCOPE_NETWORK_MODE')=='auto':
        from network_status import snapshot
        return not snapshot()['online']
    return False


def require_online(action: str) -> None:
    if enabled():
        raise ValueError(f'{action} is currently unavailable offline. Reconnect to use online services, or choose local files.')


def models_local_only() -> bool:
    return enabled() or os.getenv('SPECTRASCOPE_MODELS_LOCAL_ONLY')=='1'

def require_model_download(action: str) -> None:
    if models_local_only():raise ValueError(action+' is disabled: install the required local model pack from your media.')


def local_raster(path: str) -> str:
    if enabled():
        value = str(path)
        # Permit real disk files only, not GDAL virtual filesystems, VRTs,
        # network shares, connection strings, or remotely referenced datasets.
        if value.startswith(('\\\\', '//', '/vsi')) or '://' in value or Path(value).suffix.lower() not in ('.tif', '.tiff', '.jp2'):
            raise ValueError('Offline raster access requires a local GeoTIFF or JPEG2000 file; network, VRT and virtual filesystem sources are disabled.')
        if not Path(value).is_file():
            raise ValueError('The local raster file is missing. Select its location again.')
    return path


def raster_driver(path: str) -> str:
    """Identify a supported real file before GDAL can select a virtual driver."""
    with Path(local_raster(path)).open('rb') as stream:
        magic=stream.read(16)
    if magic.startswith((b'II*\x00',b'MM\x00*',b'II+\x00',b'MM\x00+')):return 'GTiff'
    if magic.startswith((b'\x00\x00\x00\x0cjP  \r\n\x87\n',b'\xff\x4f\xff\x51')):return 'JP2OpenJPEG'
    raise ValueError('A real GeoTIFF or JPEG2000 file is required; virtual raster definitions are not accepted')


def install() -> None:
    # Auto connectivity may probe public services at runtime. Only explicit
    # offline mode installs the irreversible process-wide socket guard.
    if os.getenv('SPECTRASCOPE_OFFLINE','').lower() not in ('1','true','yes') or getattr(socket, '_spectrascope_offline', False):
        return
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                      DO_NOT_TRACK='1', PROJ_NETWORK='OFF', OLLAMA_NO_CLOUD='1',
                      NO_PROXY='127.0.0.1,localhost,::1', no_proxy='127.0.0.1,localhost,::1')
    os.environ['GDAL_SKIP'] = 'VRT WMS WMTS HTTP DODS OGCAPI PLMOSAIC DAAS EEDAI STACTA'
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(key, None)
    original_resolve = socket.getaddrinfo
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def allowed(host):
        if isinstance(host, bytes):
            host = host.decode('ascii')
        if host == 'localhost':
            return True
        try:
            return ipaddress.ip_address(str(host)).is_loopback
        except ValueError:
            return False

    def resolve(host, port, *args, **kwargs):
        if host is not None and not allowed(host):
            raise OSError('Offline desktop policy blocked external DNS/network access')
        return original_resolve(host, port, *args, **kwargs)

    def connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6) and not allowed(address[0]):
            raise OSError('Offline desktop policy blocked external network access')
        return original_connect(self, address)

    def connect_ex(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6) and not allowed(address[0]):
            return 13
        return original_connect_ex(self, address)

    def audit(event, args):
        if event in ('socket.sendto', 'socket.sendmsg'):
            sock, address = args
            if sock.family in (socket.AF_INET, socket.AF_INET6) and address is not None and not allowed(address[0]):
                raise OSError('Offline desktop policy blocked external datagram access')
        elif event in ('socket.gethostbyname', 'socket.gethostbyaddr'):
            if not allowed(args[0]):
                raise OSError('Offline desktop policy blocked external DNS access')
        elif event == 'socket.getnameinfo' and not allowed(args[0][0]):
            raise OSError('Offline desktop policy blocked external DNS access')

    socket.getaddrinfo = resolve
    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    sys.addaudithook(audit)
    socket._spectrascope_offline = True


install()
