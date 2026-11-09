"""Public STAC searches must stay on their declared read-only provider route."""
import unittest
from unittest.mock import patch

import httpx
from fastapi import HTTPException

from app import _catalog_request, _next_link
from network_status import _probe
from data_policy import public_raster_url


class CatalogEgress(unittest.IsolatedAsyncioTestCase):
    endpoint = "https://catalog.example/v1/search"

    def test_public_search_accepts_only_numeric_geometry(self):
        from data_policy import public_search_geometry
        self.assertEqual(public_search_geometry({'type':'Point','coordinates':[1,2],'notes':'PRIVATE'}),
                         {'type':'Point','coordinates':[1,2]})
        for coordinates in (['PRIVATE',2],[True,0],[0,float('nan')],[181,0],[0,0,3]):
            with self.subTest(coordinates=coordinates), self.assertRaises(ValueError):
                public_search_geometry({'type':'Point','coordinates':coordinates})

    async def test_pagination_rechecks_current_offline_policy(self):
        from app import MoreRequest, search_more
        with patch('offline_policy.enabled',return_value=True),patch('app.public_http_client') as transport:
            with self.assertRaises(HTTPException) as caught:
                await search_more(MoreRequest(search_id='isolated-saved-search'))
            self.assertEqual(caught.exception.status_code,409)
        transport.assert_not_called()

    def test_mcp_inventory_change_cannot_trigger_a_supplier_resource_read(self):
        from mcp_readonly import ReadOnlyMCP
        with patch('mcp_readonly.InternalHTTP'),patch.object(ReadOnlyMCP,'call') as call:
            client=ReadOnlyMCP('http://127.0.0.1:9876/mcp')
            call.return_value={'resources':[{'uri':'catalog://fixture/current'}]}
            with self.assertRaisesRegex(ValueError,'current bounded supplier inventory'):
                client.read_listed('catalog://fixture/removed')
            call.assert_called_once_with('resources/list')

    async def test_public_routes_pin_addresses_and_keep_tls_origins_separate(self):
        from public_raster_transport import public_http_client
        other = "https://second.example/status"
        sent = []

        async def transport(pool, request):
            sent.append((pool, request))
            return httpx.Response(200, json={"features":[]})

        with patch('public_raster_transport.public_address',return_value='93.184.215.14'), patch.object(httpx.AsyncHTTPTransport,'handle_async_request',transport):
            async with public_http_client([self.endpoint,other],timeout=1,allow_search_post=True) as client:
                self.assertEqual(await _catalog_request(client,self.endpoint,{'limit':1}),{'features':[]})
                self.assertTrue((await _probe(client,'other',other,'json'))[1])
                with self.assertRaises(httpx.ConnectError):
                    await client.post('https://catalog.example/writes',json={})
                with self.assertRaises(httpx.ConnectError):
                    await client.delete(self.endpoint)
            async with public_http_client([other],timeout=1) as client:
                with self.assertRaises(httpx.ConnectError):
                    await client.post(other,json={})
        self.assertEqual(len(sent),2)
        self.assertIsNot(sent[0][0],sent[1][0])
        self.assertEqual([r.url.host for _,r in sent],['93.184.215.14','93.184.215.14'])
        self.assertEqual([r.extensions['sni_hostname'] for _,r in sent],['catalog.example','second.example'])
        self.assertEqual([r.headers['Host'] for _,r in sent],['catalog.example','second.example'])

    def test_raster_download_routes_refuse_other_hosts_and_paths(self):
        good = "https://landsatlook.usgs.gov/data/fixture.tif"
        self.assertEqual(public_raster_url("landsat", good), good)
        for url in ("https://127.0.0.1/data/fixture.tif",
                    "https://landsatlook.usgs.gov/data/%2e%2e/fixture.tif",
                    "https://landsatlook.usgs.gov:8443/data/fixture.tif",
                    "https://other.example/data/fixture.tif",
                    good + "?credential=placeholder"):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, "outside the approved"):
                public_raster_url("landsat", url)

    async def test_redirects_never_replay_search_or_probe(self):
        seen = []

        def redirect(request):
            seen.append(str(request.url))
            return httpx.Response(307, headers={"Location": "https://outside.example/collect"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(redirect), follow_redirects=True) as client:
            with self.assertRaisesRegex(HTTPException, "redirects are disabled"):
                await _catalog_request(client, self.endpoint, {"intersects": {"type": "Point", "coordinates": [1, 2]}})
            _, available, _ = await _probe(client, "catalog", self.endpoint, "json")
            self.assertFalse(available)
        self.assertEqual(seen, [self.endpoint, self.endpoint])

    async def test_pagination_rejects_unreviewed_routes_and_verbs(self):
        seen = []

        def search(request):
            seen.append(request)
            return httpx.Response(200, json={"features": []})

        async with httpx.AsyncClient(transport=httpx.MockTransport(search)) as client:
            for href in ("http://catalog.example/v1/search", "https://other.example/v1/search",
                         "https://catalog.example:444/v1/search", "https://catalog.example:0/v1/search",
                         "https://u@catalog.example/v1/search",
                         "https://catalog.example/v1/collections/create"):
                with self.subTest(href=href):
                    self.assertIsNone(_next_link({"links": [{"rel": "next", "href": href}]}, self.endpoint))
                    with self.assertRaises(HTTPException):
                        await _catalog_request(client, self.endpoint, {}, {"href": href, "method": "POST"})
            with self.assertRaises(HTTPException):
                await _catalog_request(client, self.endpoint, {}, {"href": self.endpoint, "method": "PATCH"})
            link = _next_link({"links": [{"rel": "next", "href": "?page=2", "method": "POST",
                                          "headers": {"Authorization": "supplier-injected"}}]}, self.endpoint)
            await _catalog_request(client, self.endpoint, {"limit": 10}, link)

        self.assertEqual(len(seen), 1)
        self.assertEqual(str(seen[0].url), self.endpoint + "?page=2")
        self.assertNotIn("authorization", seen[0].headers)
