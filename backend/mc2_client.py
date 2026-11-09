from __future__ import annotations

import asyncio
from functools import partial
import os
import sys
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from starlette.concurrency import run_in_threadpool

from internal_http import InternalHTTP
from managed_model_boundary import require_listener


def private_http_client(origin, headers=None, timeout=None, auth=None):
    origin_url = httpx.URL(origin)
    endpoint = None
    resolve_lock = asyncio.Lock()

    async def pin_destination(request):
        nonlocal endpoint
        requested = request.url
        if (requested.scheme, requested.host, requested.port) != (origin_url.scheme, origin_url.host, origin_url.port):
            raise httpx.ConnectError('Map request is outside its configured internal origin', request=request)
        if sys.platform in ('darwin','win32') and os.getenv('SPECTRASCOPE_BROWSER_ORIGINS'):
            try:
                await run_in_threadpool(require_listener,origin.rstrip('/'),'SPECTRASCOPE_MANAGED_MAP_PID','map')
            except ValueError as exc:
                raise httpx.ConnectError(str(exc),request=request) from exc
        if endpoint is None:
            async with resolve_lock:
                if endpoint is None:
                    try:
                        endpoint = await run_in_threadpool(partial(InternalHTTP, origin, local_map_bridge=True))
                    except (OSError, ValueError) as exc:
                        raise httpx.ConnectError('Map service needs a verified internal address', request=request) from exc
        # The bundled map listens on IPv4. Explicit IPv6 endpoints still work.
        address = '127.0.0.1' if endpoint.host == 'localhost' and '127.0.0.1' in endpoint.addresses else endpoint.address
        request.extensions['sni_hostname'] = endpoint.host
        request.url = requested.copy_with(host=address)

    return httpx.AsyncClient(headers=headers, timeout=timeout, auth=auth, trust_env=False,
                             follow_redirects=False, event_hooks={'request': [pin_destination]})


class MCPMapClient:
    """Small MCP client for driving the MC² server's native map tools."""

    def __init__(self, server_url: str):
        self.http_client_factory = partial(private_http_client, server_url)
        self.endpoint = server_url.rstrip("/") + "/mcp"

    async def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        async with streamablehttp_client(self.endpoint, httpx_client_factory=self.http_client_factory) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool, arguments)
                return result.model_dump(by_alias=True, exclude_none=True)

    async def close(self) -> None:
        return None
