"""Real loopback TLS and redirects; private payloads never use external/proxy paths."""
import datetime,ipaddress,json,os,socket,ssl,tempfile,threading,unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from unittest.mock import patch
from internal_http import InternalHTTP

class InternalTransport(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.requests=[];requests=self.requests
  class Endpoint(BaseHTTPRequestHandler):
   def log_message(self,*args):pass
   def do_GET(self):
    requests.append(self.path)
    if self.path=='/redirect':self.send_response(302);self.send_header('Location','http://8.8.8.8/private');self.end_headers();return
    content=json.dumps({'ok':True}).encode();self.send_response(200);self.send_header('Content-Length',str(len(content)));self.end_headers();self.wfile.write(content)
  self.server=ThreadingHTTPServer(('127.0.0.1',0),Endpoint);self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start();self.url='http://127.0.0.1:'+str(self.server.server_port)
 def tearDown(self):self.server.shutdown();self.server.server_close();self.thread.join();self.tmp.cleanup()
 def test_dns_is_pinned_and_proxy_environment_is_not_used(self):
  with patch('socket.getaddrinfo',return_value=[(socket.AF_INET,socket.SOCK_STREAM,6,'',('127.0.0.1',self.server.server_port))]):client=InternalHTTP('http://internal.invalid:'+str(self.server.server_port),{'X-Private':'fixture-only'})
  with patch('socket.getaddrinfo',side_effect=AssertionError('Credential requests must not re-resolve DNS')),patch.dict(os.environ,{'HTTP_PROXY':'http://8.8.8.8:3128','HTTPS_PROXY':'http://8.8.8.8:3128'}):self.assertTrue(client.json('GET','/data')['ok'])
  self.assertEqual(self.requests,['/data'])
 def test_redirects_and_public_or_cleartext_lan_destinations_are_blocked(self):
  with self.assertRaisesRegex(ValueError,'302'):InternalHTTP(self.url).json('GET','/redirect')
  self.assertEqual(self.requests,['/redirect'])
  for url in ('http://8.8.8.8','https://169.254.169.254','http://10.0.0.5','https://user:secret@127.0.0.1','https://127.0.0.1?secret=x'):
   with self.subTest(url=url),self.assertRaises(ValueError):InternalHTTP(url)
 def test_real_tls_requires_verified_certificate(self):
  from cryptography import x509
  from cryptography.x509.oid import NameOID,ExtendedKeyUsageOID
  from cryptography.hazmat.primitives import hashes,serialization
  from cryptography.hazmat.primitives.asymmetric import rsa
  key=rsa.generate_private_key(public_exponent=65537,key_size=2048);name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'Isolated Geo Intelligence TLS test')]);now=datetime.datetime.now(datetime.timezone.utc)
  cert=x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key()).serial_number(x509.random_serial_number()).not_valid_before(now-datetime.timedelta(minutes=1)).not_valid_after(now+datetime.timedelta(days=1)).add_extension(x509.BasicConstraints(ca=True,path_length=0),critical=True).add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]),critical=False).add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),critical=False).sign(key,hashes.SHA256())
  ca=self.root/'ca.pem';private=self.root/'key.pem';ca.write_bytes(cert.public_bytes(serialization.Encoding.PEM));private.write_bytes(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
  tls=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);tls.load_cert_chain(ca,private);self.server.socket=tls.wrap_socket(self.server.socket,server_side=True);url=self.url.replace('http:','https:')
  with self.assertRaises(ssl.SSLCertVerificationError):InternalHTTP(url).json('GET','/data')
  self.assertTrue(InternalHTTP(url,{'X-Private':'fixture-only'},ca).json('GET','/data')['ok']);self.assertEqual(self.requests,['/data'])

if __name__=='__main__':unittest.main()
