"""Mandatory read-only supplier and local-compute boundaries; no network access."""
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock,patch

import data_policy


class DataPolicy(unittest.TestCase):
 def compute_then_http(self,url,http):
  data_policy.require_internal_compute(url)
  return http(url)

 def test_external_model_endpoints_fail_before_dns_or_http(self):
  http=Mock()
  with patch.dict('os.environ',{'SPECTRASCOPE_INTERNAL_COMPUTE_HOSTS':''}),patch('data_policy.socket.getaddrinfo',side_effect=AssertionError('External DNS must not occur')):
   for url in ('https://api.example.invalid/v1','https://api.openai.com/v1','http://8.8.8.8:11434'):
    with self.subTest(url=url),self.assertRaises(ValueError):self.compute_then_http(url,http)
  http.assert_not_called()

 def test_approved_name_cannot_resolve_to_public_or_link_local_network(self):
  for address in ('8.8.8.8','169.254.169.254','fe80::1'):
   with self.subTest(address=address),patch.dict('os.environ',{'SPECTRASCOPE_INTERNAL_COMPUTE_HOSTS':'inside.example.invalid'}),patch('data_policy.socket.getaddrinfo',return_value=[(socket.AF_INET,socket.SOCK_STREAM,6,'',(address,11434))]):
    http=Mock()
    with self.assertRaises(ValueError):self.compute_then_http('http://inside.example.invalid:11434',http)
    http.assert_not_called()

 def test_explicit_internal_compute_accepts_local_private_addresses(self):
  with patch.dict('os.environ',{'SPECTRASCOPE_INTERNAL_COMPUTE_HOSTS':'inside.example.invalid'}),patch('data_policy.socket.getaddrinfo',return_value=[(socket.AF_INET,socket.SOCK_STREAM,6,'',('10.0.0.7',11434))]):
   self.assertEqual(data_policy.require_internal_compute('http://inside.example.invalid:11434'),'http://inside.example.invalid:11434')
  with patch('data_policy.socket.getaddrinfo',return_value=[(socket.AF_INET,socket.SOCK_STREAM,6,'',('127.0.0.1',11434))]):
   self.assertEqual(data_policy.require_internal_compute('http://localhost:11434'),'http://localhost:11434')

 def test_computation_url_credentials_queries_and_fragments_are_rejected(self):
  with patch('data_policy.socket.getaddrinfo',return_value=[(socket.AF_INET,socket.SOCK_STREAM,6,'',('127.0.0.1',11434))]):
   for url in ('http://user:secret@localhost:11434','http://@localhost:11434','http://localhost:11434?secret=x','http://localhost:11434#fragment'):
    with self.subTest(url=url),self.assertRaises(ValueError):data_policy.require_internal_compute(url)

 def test_supplier_requires_a_verified_read_contract(self):
  for kind in ('read','query','inspect','list','download'):
   with self.subTest(kind=kind):
    with self.assertRaises(ValueError):data_policy.supplier_operation(kind)
    self.assertIsNone(data_policy.supplier_operation(kind,verified_read_only=True))
  for kind in ('insert','update','delete','publish','write','execute','unknown','POST','call_tool'):
   with self.subTest(kind=kind),self.assertRaises(ValueError):data_policy.supplier_operation(kind,verified_read_only=True)

 def test_outputs_cannot_enter_source_roots_even_through_aliases(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);source=root/'supplier';source.mkdir();original=source/'original.txt';original.write_text('UNMODIFIED')
   for path in (source,source/'new-result.tif',source/'nested'/'result.tif',root/'output'/ '..'/'supplier'/'result.tif'):
    with self.subTest(path=path),self.assertRaises(ValueError):data_policy.protect_output(path,[source])
   alias=root/'alias';alias.symlink_to(source,target_is_directory=True)
   with self.assertRaises(ValueError):data_policy.protect_output(alias/'result.tif',[source])
   self.assertEqual(data_policy.protect_output(root/'own-results'/'result.tif',[source]),(root/'own-results'/'result.tif').resolve())
   self.assertEqual(original.read_text(),'UNMODIFIED');self.assertEqual(list(source.iterdir()),[original])


if __name__=='__main__':unittest.main()
