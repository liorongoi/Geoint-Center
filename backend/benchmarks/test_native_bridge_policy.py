"""Local worker credentials and payloads stay on the pinned local transport."""
import json
import socket
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import native_gpu_bridge as bridge


class BridgePolicy(unittest.TestCase):
 def test_config_accepts_only_local_origin_without_url_credentials(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);folder=root/'native-gpu';folder.mkdir();(folder/'worker.token').write_text('TEST-TOKEN')
   for url in ('http://localhost:18767/','http://127.0.0.1:18767','http://[::1]:18767','http://host.docker.internal:18767'):
    with self.subTest(url=url):
     (folder/'connection.json').write_text(json.dumps({'url':url}))
     self.assertEqual(bridge.connection(root),(url.rstrip('/'),'TEST-TOKEN'))
   invalid=('http://@localhost:18767','http://user@localhost:18767','http://user:secret@localhost:18767',
            'http://localhost:18767?query=private','http://localhost:18767#fragment','http://localhost:18767?',
            'http://localhost:18767#','http://outside.example.invalid:18767','https://localhost:18767',
            'http://localhost:0','http://localhost:65536','http://localhost:abc','http://localhost:18767\\outside',
            'http://localhost:18767\n')
   for url in invalid:
    with self.subTest(url=url):
     (folder/'connection.json').write_text(json.dumps({'url':url}))
     with self.assertRaises(ValueError):bridge.connection(root)

 def test_invalid_host_path_and_public_bridge_resolution_are_refused(self):
  with patch('native_gpu_bridge.InternalHTTP') as endpoint:
   for url in ('http://outside.example.invalid','http://@localhost:18767','http://localhost:18767?x=1'):
    with self.subTest(url=url),self.assertRaises(ValueError):bridge.transport((url,'TEST-TOKEN'))
   endpoint.assert_not_called()
  with patch('socket.getaddrinfo',return_value=[(socket.AF_INET,socket.SOCK_STREAM,6,'',('8.8.8.8',18767))]),patch('socket.socket.connect') as connect:
   with self.assertRaises(ValueError):bridge.transport(('http://host.docker.internal:18767','TEST-TOKEN'))
   connect.assert_not_called()

 def test_real_loopback_checkpoint_requests_are_pinned_and_redirects_are_refused(self):
  seen=[]
  class Worker(BaseHTTPRequestHandler):
   def log_message(self,*args):pass
   def do_GET(self):
    self.reply(None)
   def do_POST(self):
    self.reply(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
   def reply(self,body):
    seen.append((self.path,self.headers['X-SpectraScope-Token'],body))
    if self.path=='/redirect':
     self.send_response(307);self.send_header('Location','http://8.8.8.8/leak');self.end_headers();return
    response=b'{"ok":true}'
    self.send_response(200);self.send_header('Content-Length',str(len(response)));self.end_headers();self.wfile.write(response)
  with ThreadingHTTPServer(('127.0.0.1',0),Worker) as server:
   thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
   try:
    address=(socket.AF_INET,socket.SOCK_STREAM,6,'',('127.0.0.1',server.server_port))
    with patch('socket.getaddrinfo',side_effect=[[address]]):
     client=bridge.transport((f'http://localhost:{server.server_port}','TEST-TOKEN'))
     self.assertEqual(bridge.request(client,'/health'),{'ok':True})
     self.assertEqual(bridge.request(client,'/work/fixed-id/ack',{'sequence':5}),{'ok':True})
     with self.assertRaisesRegex(ValueError,'307'):bridge.request(client,'/redirect',{'test':'dummy'})
     for path in ('//outside/health','/health?query=1','/health#fragment','/health\n'):
      with self.assertRaises(ValueError):bridge.request(client,path)
   finally:
    server.shutdown();thread.join()
  self.assertEqual(seen,[('/health','TEST-TOKEN',None),('/work/fixed-id/ack','TEST-TOKEN',{'sequence':5}),
                         ('/redirect','TEST-TOKEN',{'test':'dummy'})])


if __name__=='__main__':unittest.main()
