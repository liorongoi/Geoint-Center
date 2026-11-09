"""Exercise worker claims against disposable TLS and redirect endpoints."""
import json
import os
from pathlib import Path
import ssl
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from pool_worker import Worker


class WorkerTransport(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.certificate = self.root / 'fixture.pem'
        key = self.root / 'fixture.key'
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1',
                        '-subj', '/CN=worker-fixture', '-addext', 'subjectAltName=IP:127.0.0.1',
                        '-keyout', str(key), '-out', str(self.certificate)],
                       check=True, capture_output=True, timeout=10)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(self.certificate, key)
        self.requests = []
        self.redirect_requests = []
        self.reply = {'task': None}
        self.redirect = False
        owner = self

        class Endpoint(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                owner.requests.append(self.path)
                if owner.redirect:
                    self.send_response(307)
                    self.send_header('Location', owner.trap + '/collect')
                    self.end_headers()
                    return
                data = json.dumps(owner.reply).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        class Trap(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                owner.redirect_requests.append(self.path)
                self.send_error(400)

        self.trap = self.start(Trap)
        self.origin = self.start(Endpoint, context)
        self.key = self.root / 'worker.key'
        self.key.write_text('1' * 64)

    def start(self, handler, tls=None):
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        if tls:
            server.socket = tls.wrap_socket(server.socket, server_side=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.shutdown)
        return f"{'https' if tls else 'http'}://127.0.0.1:{server.server_port}"

    def worker(self, origin=None, ca=None):
        return Worker(origin or self.origin, 'fixture', self.key, self.root / 'cache', ca)

    def test_trusted_endpoint_and_redirect(self):
        worker = self.worker(ca=self.certificate)
        self.assertFalse(worker.once())
        self.redirect = True
        with patch.dict(os.environ, {'HTTPS_PROXY': self.trap, 'HTTP_PROXY': self.trap}):
            with self.assertRaisesRegex(ValueError, '307'):
                worker.once()
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.redirect_requests, [])

    def test_untrusted_certificate_and_wrong_hostname(self):
        with self.assertRaises(ssl.SSLCertVerificationError):
            self.worker().once()
        with self.assertRaises(ssl.SSLCertVerificationError):
            self.worker(self.origin.replace('127.0.0.1', 'localhost'), self.certificate).once()
        self.assertEqual(self.requests, [])

    def test_invalid_task_paths_never_create_attempt_storage(self):
        worker = self.worker(ca=self.certificate)
        outside = self.root / 'outside-cache'
        for identifier, attempt in ((str(outside), 1), ('../traversal', 1),
                                    ('58ba1fe1-c342-4dc2-a02c-dc28d1f386f2', '../../escape'),
                                    ('58ba1fe1-c342-4dc2-a02c-dc28d1f386f2', True),
                                    ('58ba1fe1-c342-4dc2-a02c-dc28d1f386f2', 11)):
            self.reply = {'task': {'id': identifier, 'attempt': attempt, 'lease_token': 'dummy'}}
            with self.subTest(identifier=identifier, attempt=attempt):
                with self.assertRaisesRegex(ValueError, 'invalid worker task'):
                    worker.once()
        self.assertFalse(outside.exists())
        self.assertFalse((self.root / 'cache' / 'work').exists())
        self.assertEqual(len(self.requests), 5)
