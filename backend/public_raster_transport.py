"""Expose approved supplier ranges to GDAL without giving GDAL internet access."""
from contextlib import asynccontextmanager, closing, contextmanager
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import ipaddress
import re
import secrets
import socket
import ssl
import threading
from urllib.parse import urlsplit

from data_policy import public_raster_url

MAX_RANGE = 32 * 1024**2
MAX_TRANSFER = 2 * 1024**3


class TemporaryProviderError(ValueError):
    """A provider returned a retryable status before sending any file bytes."""


def public_address(host):
    addresses = [ipaddress.ip_address(row[4][0]) for row in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)]
    if not addresses or any(not address.is_global or address.is_multicast for address in addresses):
        raise ValueError('A public raster provider resolved outside the public network')
    return str(addresses[0])


@contextmanager
def pinned_response(method, host, address, path, tls, headers):
    """Connect directly to the vetted address and authenticate the supplier hostname."""
    with closing(http.client.HTTPSConnection(host, timeout=20, context=tls)) as connection:
        with socket.create_connection((address, 443), timeout=20) as raw:
            connection.sock = tls.wrap_socket(raw, server_hostname=host)
        connection.request(method, path, headers={'Accept-Encoding':'identity', **headers})
        with connection.getresponse() as response:
            yield response


def upstream(method, host, address, path, byte_range, tls):
    """Fixed HTTPS origin and route. The caller receives no redirect target."""
    with pinned_response(method, host, address, path, tls, {'Range':byte_range} if byte_range else {}) as response:
        expected = 200 if method == 'HEAD' else 206
        if response.status != expected or response.getheader('Content-Encoding','identity').lower() != 'identity':
            raise ValueError('The provider did not return a direct raster byte range')
        length=response.getheader('Content-Length','')
        if not length.isdecimal():
            raise ValueError('Raster provider returned an unbounded response')
        headers={'Content-Length':length}
        if method=='HEAD':
            if response.getheader('Accept-Ranges','').lower()=='bytes':headers['Accept-Ranges']='bytes'
            return headers,b''
        actual = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.getheader('Content-Range',''))
        requested = re.fullmatch(r'bytes=(\d+)-(\d+)',byte_range)
        if (not actual or not requested or int(actual[1])!=int(requested[1])
                or int(actual[2])!=min(int(requested[2]),int(actual[3])-1)
                or int(length)!=int(actual[2])-int(actual[1])+1 or not 0<int(length)<=MAX_RANGE):
            raise ValueError('Raster provider returned an invalid byte range')
        headers['Content-Range']=response.getheader('Content-Range')
        body=response.read(int(length)+1)
        if len(body)!=int(length):raise ValueError('Raster range was incomplete')
        return headers,body


@contextmanager
def complete_response(catalog, url, max_bytes, *, earthdata_token=None):
    from offline_policy import require_online
    require_online('Downloading raster assets')
    if earthdata_token is not None:
        from earthdata_hls import authorized_response
        transfer = authorized_response(catalog, url, earthdata_token)
    else:
        parsed = urlsplit(public_raster_url(catalog, url))
        address = public_address(parsed.hostname)
        transfer = pinned_response('GET', parsed.hostname, address, parsed.path, ssl.create_default_context(),
                                   {'Accept': 'image/tiff, image/jp2, application/octet-stream'})
    with transfer as response:
        if 300<=response.status<400:
            raise ValueError('The provider redirected this raster. Import the saved file or use an approved direct asset.')
        if earthdata_token is not None and response.status in (429, 502, 503, 504):
            raise TemporaryProviderError(f'NASA download service returned HTTP {response.status}. Resume to retry this band.')
        if response.status!=200 or response.getheader('Content-Encoding','identity').lower()!='identity':
            raise ValueError(f'The provider did not return a complete raster (HTTP {response.status})')
        length=response.getheader('Content-Length')
        if length is not None and (not length.isdecimal() or not 0<int(length)<=max_bytes):
            raise ValueError('Raster length is invalid or exceeds the per-file download limit')
        yield response,int(length) if length is not None else None


def complete_download(catalog, url, max_bytes, *, earthdata_token=None):
    return bounded_download(lambda: complete_response(catalog, url, max_bytes, earthdata_token=earthdata_token),
                            max_bytes, retries=2 if earthdata_token else 0)


@asynccontextmanager
async def bounded_download(open_transfer, max_bytes, *, retries=0):
    import asyncio

    loop=asyncio.get_running_loop()
    deferred_close=False

    async def blocking(operation,*args):
        nonlocal deferred_close
        future=loop.run_in_executor(None,operation,*args)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # The socket may still be opening or reading. Close only after that
            # operation relinquishes the response.
            def close_after(done):
                try:done.result()
                except Exception:pass
                transfer.__exit__(None,None,None)
            deferred_close=True
            future.add_done_callback(close_after)
            raise

    for attempt in range(retries + 1):
        transfer = open_transfer()
        try:
            response, expected = await blocking(transfer.__enter__)
            break
        except TemporaryProviderError:
            if attempt == retries:
                raise
            await asyncio.sleep(2 ** attempt)
    try:
        async def chunks():
            written=0
            while chunk:=await blocking(response.read1,1024*1024):
                written+=len(chunk)
                if written>max_bytes:raise ValueError('Raster exceeds the per-file download limit')
                yield chunk
            if expected is not None and written!=expected:raise ValueError('Incomplete raster download')
        yield expected,chunks()
    finally:
        if not deferred_close:transfer.__exit__(None,None,None)


@contextmanager
def raster_range_gateway(catalog, urls):
    from offline_policy import require_online
    require_online('Raster streaming')
    tls=ssl.create_default_context()
    addresses={}
    routes={}
    token=secrets.token_urlsafe(24)
    for index,url in enumerate(urls):
        parsed=urlsplit(public_raster_url(catalog,url))
        host=parsed.hostname
        if host not in addresses:addresses[host]=public_address(host)
        routes[f'/{token}/{index}.tif']=(host,addresses[host],parsed.path)
    remaining=MAX_TRANSFER

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_HEAD(self):self.serve()
        def do_GET(self):self.serve()
        def serve(self):
            nonlocal remaining
            target=routes.get(self.path)
            if target is None:
                self.send_error(404)
                return
            byte_range=self.headers.get('Range','')
            if self.command=='GET':
                match=re.fullmatch(r'bytes=(\d{1,15})-(\d{1,15})',byte_range)
                if not match or not 0<int(match[2])-int(match[1])+1<=min(MAX_RANGE,remaining):
                    self.send_error(416)
                    return
            try:headers,body=upstream(self.command,*target,byte_range,tls)
            except (OSError,ValueError,http.client.HTTPException):
                self.send_error(502,'Provider requires a direct bounded range. Import a saved local raster.')
                return
            remaining-=len(body)
            self.send_response(200 if self.command=='HEAD' else 206)
            for name,value in headers.items():self.send_header(name,value)
            self.end_headers()
            if body:self.wfile.write(body)

    with HTTPServer(('127.0.0.1',0),Handler) as server:
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            yield [f'/vsicurl/http://127.0.0.1:{server.server_port}{route}' for route in routes]
        finally:
            server.shutdown()
            thread.join()


def public_http_client(origins, *, timeout, allow_search_post=False):
    """Pin catalog/probe traffic to configured HTTPS routes and public addresses."""
    import httpx
    from starlette.concurrency import run_in_threadpool

    routes = {(urlsplit(url).hostname, urlsplit(url).path.rstrip('/')) for url in origins}
    # Different TLS names may use the same CDN address. Never reuse an
    # authenticated connection for a different configured origin.
    pools = {host:httpx.AsyncHTTPTransport(trust_env=False) for host in {host for host,_ in routes}}

    class PublicTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            url = request.url
            if (url.scheme != 'https' or url.port not in (None, 443) or url.username or url.password
                    or url.fragment or (url.host, url.path.rstrip('/')) not in routes
                    or (request.method != 'GET' and not (allow_search_post and request.method == 'POST'))):
                raise httpx.ConnectError('Outside the configured public catalog or probe route', request=request)
            try:
                address = await run_in_threadpool(public_address, url.host)
            except (OSError, ValueError) as exc:
                raise httpx.ConnectError('Approved public origin did not resolve to a public address', request=request) from exc
            pinned = httpx.Request(request.method, url.copy_with(host=address), headers=request.headers,
                                   stream=request.stream, extensions={**request.extensions, 'sni_hostname': url.host})
            return await pools[url.host].handle_async_request(pinned)

        async def aclose(self):
            for pool in pools.values():
                await pool.aclose()

    return httpx.AsyncClient(transport=PublicTransport(), timeout=timeout,
                             trust_env=False, follow_redirects=False)
