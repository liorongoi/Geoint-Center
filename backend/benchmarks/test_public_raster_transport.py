"""Test supplier range containment with disposable loopback data."""
from contextlib import contextmanager, nullcontext
import asyncio
import io
from threading import Event
import httpx
import re
import socket
import unittest
from unittest.mock import Mock, patch

import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from public_raster_transport import public_address, raster_range_gateway, upstream
from raster_pipeline import read_stack

URL = 'https://landsatlook.usgs.gov/data/fixture.tif'


class PublicRasterTransport(unittest.TestCase):
    def test_native_gdal_reads_only_validated_loopback_ranges(self):
        with MemoryFile() as image:
            with image.open(driver='GTiff',width=64,height=64,count=1,dtype='uint16',
                            crs='EPSG:32636',transform=from_origin(500000,3500000,30,30)) as raster:
                raster.write(np.full((64,64),12,dtype='uint16'),1)
            payload=image.read()
        calls=[]
        def fixture(method,host,address,path,byte_range,tls):
            calls.append((method,host,address,path))
            if method=='HEAD':return {'Content-Length':str(len(payload)),'Accept-Ranges':'bytes'},b''
            first,last=map(int,re.fullmatch(r'bytes=(\d+)-(\d+)',byte_range).groups())
            last=min(last,len(payload)-1)
            data=payload[first:last+1]
            return {'Content-Length':str(len(data)),'Content-Range':f'bytes {first}-{last}/{len(payload)}'},data
        with patch('offline_policy.enabled',return_value=False),patch('public_raster_transport.public_address',return_value='93.184.215.14'),patch('public_raster_transport.upstream',side_effect=fixture):
            stack=read_stack({'mode':'stream','asset_urls':[{'key':'B2','url':URL}]},{'catalog':'landsat'},64)
        self.assertTrue(np.all(stack.pixels==12))
        self.assertIn('GET',{call[0] for call in calls})
        self.assertEqual({call[1:] for call in calls},{('landsatlook.usgs.gov','93.184.215.14','/data/fixture.tif')})

    def test_gateway_rejects_redirect_and_unbounded_requests(self):
        with patch('offline_policy.enabled',return_value=False),patch('public_raster_transport.public_address',return_value='93.184.215.14'),patch('public_raster_transport.upstream',side_effect=ValueError('Redirect denied')) as origin:
            with raster_range_gateway('landsat',[URL]) as [wrapped]:
                address=wrapped.removeprefix('/vsicurl/')
                with httpx.Client(trust_env=False) as client:
                    self.assertEqual(client.get(address).status_code,416)
                    self.assertEqual(client.get(address,headers={'Range':'bytes=0-99999999999'}).status_code,416)
                    self.assertEqual(client.get(address+'/private').status_code,404)
                    self.assertEqual(client.get(address,headers={'Range':'bytes=0-99'}).status_code,502)
            self.assertEqual(origin.call_count,1)
        response=Mock(status=307)
        connection=Mock()
        connection.getresponse.return_value=nullcontext(response)
        tls=Mock()
        with patch('http.client.HTTPSConnection',return_value=connection),patch('socket.create_connection',return_value=nullcontext(Mock())):
            with self.assertRaisesRegex(ValueError,'direct raster'):
                upstream('GET','landsatlook.usgs.gov','93.184.215.14','/data/fixture.tif','bytes=0-99',tls)
        self.assertEqual(connection.request.call_count,1)

    def test_mixed_public_and_private_dns_answers_fail_closed(self):
        addresses=[(socket.AF_INET,socket.SOCK_STREAM,6,'',(ip,443)) for ip in ('93.184.215.14','127.0.0.1')]
        with patch('socket.getaddrinfo',return_value=addresses):
            with self.assertRaisesRegex(ValueError,'outside the public network'):public_address('landsatlook.usgs.gov')


class CompleteRasterTransport(unittest.IsolatedAsyncioTestCase):
    async def test_complete_transfer_enforces_bounds_and_declared_length(self):
        from public_raster_transport import complete_download
        for declared in (None,'2','8'):
            with self.subTest(declared=declared):
                response=Mock(status=200)
                response.getheader.side_effect=lambda key,default=None: declared if key=='Content-Length' else default
                response.read1=io.BytesIO(b'abc').read1
                with patch('offline_policy.enabled',return_value=False),patch('public_raster_transport.public_address',return_value='93.184.215.14'),patch('public_raster_transport.pinned_response',return_value=nullcontext(response)):
                    with self.assertRaisesRegex(ValueError,'limit|Incomplete'):
                        async with complete_download('landsat',URL,2 if declared is None else 5) as (_,chunks):
                            _=[chunk async for chunk in chunks]

    async def test_cancelled_connection_is_closed_after_opening_finishes(self):
        from public_raster_transport import complete_download
        started,release,closed=Event(),Event(),Event()

        @contextmanager
        def fixture(*args):
            started.set()
            release.wait(timeout=3)
            response=Mock(status=200)
            response.getheader.side_effect=lambda key,default=None: default
            try:yield response
            finally:closed.set()

        async def transfer():
            async with complete_download('landsat',URL,5):
                self.fail('Cancelled transfer should never receive the response')

        with patch('offline_policy.enabled',return_value=False),patch('public_raster_transport.public_address',return_value='93.184.215.14'),patch('public_raster_transport.pinned_response',side_effect=fixture):
            task=asyncio.create_task(transfer())
            try:
                self.assertTrue(await asyncio.to_thread(started.wait,3))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):await task
            finally:release.set()
            self.assertTrue(await asyncio.to_thread(closed.wait,3))
