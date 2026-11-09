"""Entry point executed exclusively by the bundled, private Python runtime."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'mc2' / 'server'))
os.environ.setdefault('SPECTRASCOPE_OFFLINE','1')
from resource_admission import configure_native_host
configure_native_host()
if os.getenv('SPECTRASCOPE_OFFLINE','').lower() in ('1','true','yes'):
    from native_network_boundary import enforce_mac_loopback
    enforce_mac_loopback()
from offline_policy import install
install()

if __name__ == '__main__':
    import uvicorn
    service, port = sys.argv[1], int(sys.argv[2])
    if service not in ('app', 'map'):
        raise SystemExit('Unknown desktop service')
    from importlib import import_module
    from urllib.parse import urlsplit
    module=import_module('app' if service=='app' else 'mapcontrol_server.main')
    target=module.app
    origins = os.environ.get('SPECTRASCOPE_BROWSER_ORIGINS', '').split(',')
    origins = [origin for origin in origins if urlsplit(origin).hostname == '127.0.0.1'
               and urlsplit(origin).scheme == 'http' and urlsplit(origin).port]
    if origins:
        # An external browser does not inherit Electron's request filter. Restrict
        # BOTH application and iframe documents to the assigned local services.
        from starlette.middleware.base import BaseHTTPMiddleware
        local = ' '.join(origins)
        remote = 'https://tile.openstreetmap.org https://server.arcgisonline.com https://basemaps.cartocdn.com' if os.getenv('SPECTRASCOPE_NETWORK_MODE')=='auto' and os.getenv('SPECTRASCOPE_OFFLINE')!='1' else ''
        previews='https://datahub.creodias.eu https://landsatlook.usgs.gov https://data.inpe.br' if remote else ''
        sockets = ' '.join(origin.replace('http:', 'ws:', 1) for origin in origins)
        def browser_policy(extra=''):
            return (f"default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' blob:; "
                  f"style-src 'self' 'unsafe-inline'; img-src 'self' data: blob: {local} {remote} {previews} {extra}; "
                  f"font-src 'self' data:; connect-src 'self' blob: {local} {sockets} {remote} {extra}; "
                  f"frame-src {local}; frame-ancestors {local}; worker-src 'self' blob:; "
                  "object-src 'none'; base-uri 'self'; form-action 'self'; media-src 'self' blob:")
        class LocalBrowserPolicy(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                response = await call_next(request)
                extra=[]
                if remote and response.headers.get('content-type','').startswith('text/html'):
                    try:
                        import network_status
                        if network_status.snapshot()['online']:
                            if service=='app':
                                with module._database() as db:
                                    import json
                                    definitions=[json.loads(row['payload']).get('definition',{}) for row in db.execute("SELECT payload FROM settings WHERE key LIKE 'basemap:custom-%'")]
                            elif request.url.path.startswith('/map/'):
                                from mapcontrol_server.services import basemap_service
                                definitions=list((await basemap_service.list_basemaps(request.path_params.get('map_id') or request.url.path.split('/')[2])).values())
                            else:definitions=[]
                            for definition in definitions:
                                parsed=urlsplit(definition.get('url',''))
                                if parsed.scheme in ('http','https') and parsed.hostname and not parsed.username and not parsed.password:extra.append(parsed.scheme+'://'+parsed.netloc)
                    except (OSError,ValueError,KeyError):pass
                response.headers['Content-Security-Policy'] = browser_policy(' '.join(sorted(set(extra))))
                response.headers['Referrer-Policy'] = 'no-referrer'
                response.headers['X-Content-Type-Options'] = 'nosniff'
                return response
        target.add_middleware(LocalBrowserPolicy)
    if service == 'app' and os.getenv('GEO_MAC_WEB_COMMIT'):
        from mac_web_gate import protect
        protect(target, module.APP_VERSION, os.environ['GEO_MAC_WEB_PROBE'])
    uvicorn.run(target,
                host='127.0.0.1', port=port, access_log=False)
