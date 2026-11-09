"""Pinned private-address HTTP(S), no proxy, redirect or public-network fallback."""
import http.client,ipaddress,json,socket,ssl
from urllib.parse import urlencode,urlsplit

class InternalHTTP:
 def __init__(self,url,headers=None,ca_file=None,timeout=30,*,local_model_bridge=False,local_map_bridge=False,verify_peer=None):
  parsed=urlsplit(url)
  if parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment or '?' in url or '#' in url or '\\' in url:raise ValueError('Use an internal service URL without credentials, query or fragment')
  self.host=parsed.hostname;self.port=parsed.port or (443 if parsed.scheme=='https' else 80);self.scheme=parsed.scheme;self.prefix=parsed.path.rstrip('/');self.headers=headers or {};self.timeout=timeout
  if '..' in self.prefix.split('/') or '%' in self.prefix:raise ValueError('Use a plain internal service base path')
  addresses=list(dict.fromkeys(r[4][0] for r in socket.getaddrinfo(self.host,self.port,type=socket.SOCK_STREAM)))
  if not addresses:raise ValueError('Internal endpoint could not be resolved')
  bridge=(local_model_bridge and self.host in ('host.docker.internal','ollama')
          or local_map_bridge and self.host=='mapcontrol' and self.port==8000 and not self.prefix)
  bridge_ranges=tuple(ipaddress.ip_network(net) for net in ('10.0.0.0/8','172.16.0.0/12','192.168.0.0/16','fc00::/7'))
  for raw in addresses:
   address=ipaddress.ip_address(raw)
   if not (address.is_loopback or address.is_private) or address.is_link_local or address.is_multicast or address.is_unspecified:raise ValueError('Private data may be sent only inside the approved private/loopback system boundary')
   if bridge and not (address.is_loopback or any(address in network for network in bridge_ranges)):raise ValueError('The local model bridge must resolve to a loopback or private network address')
   if self.scheme=='http' and not address.is_loopback and not bridge:raise ValueError('Remote internal services require verified HTTPS; cleartext is loopback-only')
  if __import__('os').environ.get('SPECTRASCOPE_OFFLINE','').lower() in ('1','true','yes') and any(not ipaddress.ip_address(a).is_loopback for a in addresses):raise ValueError('Strict offline-only mode is loopback-only; LAN workers require an explicitly enabled internal network launch')
  self.addresses=addresses;self.address=addresses[0];self.context=ssl.create_default_context(cafile=str(ca_file) if ca_file else None);self.verify_peer=verify_peer
 def connection(self):
  # Connect to the already checked address, while validating the original TLS
  # hostname. DNS rebinding cannot redirect later credential-bearing requests.
  failure=None
  for address in self.addresses[:8]:
   conn=http.client.HTTPConnection(self.host,self.port,timeout=self.timeout)
   try:
    conn.sock=socket.socket(socket.AF_INET6 if ipaddress.ip_address(address).version==6 else socket.AF_INET,socket.SOCK_STREAM)
    conn.sock.settimeout(self.timeout);conn.sock.connect((address,self.port))
   except OSError as exc:
    conn.close();failure=exc;continue
   try:
    if self.verify_peer is not None:self.verify_peer(conn.sock)
    if self.scheme=='https':conn.sock=self.context.wrap_socket(conn.sock,server_hostname=self.host)
   except BaseException:conn.close();raise
   return conn
  raise failure
 def send(self,method,path,body=None,file=None,extra_headers=None,*,query=None):
  if (not path.startswith('/') and not (path=='' and self.prefix)) or path.startswith('//') or any(c in path for c in ('\\','?','#','\r','\n')):raise ValueError('Use a typed internal operation path')
  if query:
   if method!='GET':raise ValueError('Typed query parameters are only supported on GET reads')
   path+='?'+urlencode(query)
  conn=self.connection();headers={**self.headers,**(extra_headers or {})}
  if body is not None:body=json.dumps(body,allow_nan=False).encode();headers['Content-Type']='application/json'
  if file is not None:
   from pathlib import Path
   headers['Content-Length']=str(Path(file).stat().st_size)
  try:
   conn.putrequest(method,self.prefix+path)
   for key,value in headers.items():conn.putheader(key,value)
   if body is not None:conn.putheader('Content-Length',str(len(body)))
   conn.endheaders()
   if file is not None:
    with open(file,'rb') as stream:
     for chunk in iter(lambda:stream.read(1024*1024),b''):conn.send(chunk)
   elif body is not None:conn.send(body)
   response=conn.getresponse()
   if not 200<=response.status<300:
    # Do not echo private remote response bodies, tokens or DSNs into logs.
    status=response.status;response.read(4096);raise ValueError(f'Internal service request failed (HTTP {status}); no redirect or fallback was attempted')
   return conn,response
  except BaseException:conn.close();raise
 def json(self,method,path,body=None,file=None,*,maximum=2*1024*1024,query=None,content_crs=None):
  conn,response=self.send(method,path,body,file,query=query)
  try:
   if content_crs is not None and response.getheader('Content-Crs','').strip().strip('<>') not in ('',content_crs):
    raise ValueError('Supplier response declares a coordinate reference system other than CRS84')
   content=response.read(maximum+1)
   if len(content)>maximum:raise ValueError('Internal JSON response exceeded its limit')
   return json.loads(content)
  finally:conn.close()
 def download(self,path,destination,expected_hash,maximum=8*1024**3,checkpoint=None):
  from pathlib import Path
  import hashlib,os
  destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True);temp=destination.with_name(destination.name+'.part');conn,response=self.send('GET',path);h=hashlib.sha256();size=0
  try:
   with temp.open('wb') as output:
    while chunk:=response.read(1024*1024):
     if checkpoint:checkpoint()
     size+=len(chunk)
     if size>maximum:raise ValueError('Internal file exceeds its declared transfer budget')
     output.write(chunk);h.update(chunk)
    output.flush();os.fsync(output.fileno())
   if h.hexdigest()!=expected_hash:raise ValueError('Internal input file failed checksum verification')
   temp.replace(destination)
  finally:conn.close();temp.unlink(missing_ok=True)
