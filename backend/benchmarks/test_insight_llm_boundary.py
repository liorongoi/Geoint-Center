"""Planner privacy checks with an isolated loopback provider and dummy data."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import socket
import threading
import unittest
from unittest.mock import Mock, patch

from insight_llm import Planner


class PlannerBoundary(unittest.TestCase):
    def test_chat_waits_for_local_analysis_and_rechecks_its_destination(self):
        from concurrent.futures import ThreadPoolExecutor
        release=threading.Event()
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(release.wait,3)
            self.planner.analysis_executor=executor
            async def run():
                task=asyncio.create_task(self.planner.request(self.config,'POST','/api/chat',{'model':'local'}))
                await asyncio.sleep(.05)
                self.assertFalse(task.done())
                self.assertEqual(self.requests,[])
                self.planner.config.return_value={**self.config,'base_url':'http://localhost:12345'}
                release.set()
                with self.assertRaisesRegex(ValueError,'connection changed'):await task
            asyncio.run(run())
        self.planner.key.assert_not_called()

    def setUp(self):
        self.requests = []
        requests = self.requests

        class Provider(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                requests.append(('GET', self.path, None))
                self.reply({'models': [{'name': 'local:latest'}, {'name': 'remote:latest', 'remote_host': 'remote.invalid'}]})

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(('POST', self.path, request))
                if request.get('redirect'):
                    self.send_response(307)
                    self.send_header('Location', 'http://8.8.8.8/private')
                    self.end_headers()
                else:
                    self.reply({'ok': True})

            def reply(self, payload):
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Provider)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.config = {'base_url': f'http://127.0.0.1:{self.server.server_port}', 'provider': 'ollama'}
        self.planner = Planner(None, '.')
        self.planner.config = Mock(return_value=self.config)
        self.planner.key = Mock(return_value='dummy-test-key')
        environment=patch.dict(os.environ, {'SPECTRASCOPE_OFFLINE':'0','SPECTRASCOPE_MODELS_LOCAL_ONLY':'0',
            'SPECTRASCOPE_LLM_URL':self.config['base_url'],'SPECTRASCOPE_MANAGED_LLM_PID':str(os.getpid())})
        environment.start()
        self.addCleanup(environment.stop)
        if __import__('sys').platform!='darwin':
            listener=patch('managed_model_boundary.owns_listener',return_value=True)
            listener.start()
            self.addCleanup(listener.stop)


    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, payload):
        return asyncio.run(self.planner.request(self.config, 'POST', '/api/chat', payload))

    def test_inventory_and_private_prompt_reuse_one_dns_resolution(self):
        address = (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', self.server.server_port))
        with patch('socket.getaddrinfo', side_effect=[[address]]):
            self.assertEqual(self.request({'model': 'local', 'messages': ['isolated sample']}), {'ok': True})
        self.assertEqual([r[:2] for r in self.requests], [('GET', '/api/tags'), ('POST', '/api/chat')])

    def test_cloud_names_and_remote_aliases_never_receive_a_prompt(self):
        for name in ('qwen:cloud', 'qwen:80b-cloud', 'remote:latest', 'missing:latest'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.request({'model': name, 'messages': ['isolated sample']})
        self.assertTrue(all(r[0] == 'GET' for r in self.requests))

    def test_public_address_from_builtin_bridge_alias_is_refused_before_socket(self):
        self.config['base_url'] = 'http://host.docker.internal:11434'
        public = (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('8.8.8.8', 11434))
        with patch('socket.getaddrinfo', return_value=[public]), patch('socket.socket.connect') as connect:
            with self.assertRaisesRegex(ValueError, 'managed local Ollama'):
                self.request({'model': 'local'})
        connect.assert_not_called()

    def test_unmanaged_loopback_never_receives_prompt_or_credentials(self):
        with patch.dict(os.environ, {'SPECTRASCOPE_MANAGED_LLM_PID':''}):
            with self.assertRaisesRegex(ValueError,'Independent endpoints are blocked'):
                self.request({'model':'local','messages':['PRIVATE_FORWARDING_CANARY']})
        self.assertEqual(self.requests, [])
        self.planner.key.assert_not_called()

    def test_wrong_owner_is_rejected_before_connecting(self):
        with patch('managed_model_boundary.owns_listener',return_value=False):
            with self.assertRaisesRegex(ValueError,'listener could not be verified'):
                self.request({'model':'local','messages':['PRIVATE_FORWARDING_CANARY']})
        self.assertEqual(self.requests, [])
        self.planner.key.assert_not_called()

    def test_connected_peer_owner_is_checked_before_sending_http(self):
        def owned(_pid,_port,peer_port=None):
            return peer_port is None
        with patch('managed_model_boundary.owns_listener',side_effect=owned):
            with self.assertRaisesRegex(ValueError,'listener could not be verified'):
                self.request({'model':'local','messages':['PRIVATE_FORWARDING_CANARY']})
        self.assertEqual(self.requests, [])

    def test_redirect_is_not_followed_even_if_proxy_environment_exists(self):
        with patch.dict(os.environ, {'HTTP_PROXY': 'http://8.8.8.8:8080'}):
            with self.assertRaisesRegex(ValueError, '307'):
                self.request({'model': 'local', 'redirect': True})
        self.assertEqual([r[1] for r in self.requests], ['/api/tags', '/api/chat'])

    def test_saved_provider_change_prevents_sending_credentials_to_old_base(self):
        self.planner.config.return_value = {**self.config, 'base_url': 'http://localhost:12345'}
        with self.assertRaisesRegex(ValueError, 'connection changed'):
            self.request({'model': 'local'})
        self.planner.key.assert_not_called()
        self.assertEqual(self.requests, [])
