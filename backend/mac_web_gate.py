"""Hold Mac web candidate traffic and autonomous work until durable cutover."""
import asyncio
import hmac
import os
from pathlib import Path


def pending():
    marker = os.environ.get('GEO_MAC_WEB_COMMIT')
    return bool(marker and not Path(marker).is_file())


async def wait_for_commit():
    while pending():
        await asyncio.sleep(.1)


def protect(app, version, token):
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    class CutoverGate(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            probe = request.method == 'GET' and hmac.compare_digest(
                request.headers.get('X-Geo-Cutover', '').encode(), token.encode())
            if probe and request.url.path == '/api/ready':
                return JSONResponse({'ok': True, 'app_version': version, 'cutover_pending': pending()})
            if pending() and not (probe and request.url.path == '/api/services/capabilities'):
                return JSONResponse({'detail': 'The workspace is being verified. Retry after the update.'},
                                    status_code=503, headers={'Retry-After': '2', 'Cache-Control': 'no-store'})
            return await call_next(request)

    app.add_middleware(CutoverGate)
