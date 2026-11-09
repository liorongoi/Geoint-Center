"""Known MCP discovery/resource reads. Tool annotations never authorize execution."""
import json
from internal_http import InternalHTTP

class ReadOnlyMCP:
 def __init__(self,endpoint,headers=None,ca_file=None,session=None,version=None):
  from urllib.parse import urlsplit
  if not urlsplit(endpoint).path.strip('/'):raise ValueError('Supply the complete internal MCP endpoint path')
  self.http=InternalHTTP(endpoint,headers,ca_file,timeout=10);self.session=session;self.version=version;self.serial=0
  self.http.prefix=urlsplit(endpoint).path
 def call(self,method,params=None,notification=False):
  if method not in ('initialize','notifications/initialized','resources/list','resources/read','tools/list'):raise ValueError('Supplier MCP tool execution and unknown methods are blocked; read-only hints are not authority')
  self.serial+=1;identifier=self.serial;body={'jsonrpc':'2.0','method':method,'params':params or {}}
  if not notification:body['id']=identifier
  headers={'Accept':'application/json, text/event-stream'}
  if self.session:headers['Mcp-Session-Id']=self.session
  if self.version:headers['MCP-Protocol-Version']=self.version
  conn,response=self.http.send('POST','',body,extra_headers=headers)
  try:
   session=response.getheader('Mcp-Session-Id')
   if session:
    if len(session)>4096 or any(c in session for c in ('\r','\n')):raise ValueError('Invalid MCP session header')
    self.session=session
   if notification:response.read(4096);return {}
   if 'text/event-stream' in (response.getheader('Content-Type') or ''):
    size=0;data=[];message=None
    while True:
     line=response.readline(2*1024*1024+1);size+=len(line)
     if size>2*1024**2:raise ValueError('MCP response exceeds its local inspection budget')
     if not line:raise ValueError('MCP stream ended without its requested response')
     if line.strip()==b'' and data:
      candidate=json.loads(b'\n'.join(data));data=[]
      if candidate.get('id')==identifier and ('result' in candidate or 'error' in candidate):message=candidate;break
      # Supplier-initiated sampling/tool/client requests are never executed.
     elif line.startswith(b'data:'):data.append(line[5:].strip())
   else:
    raw=response.read(2*1024*1024+1)
    if len(raw)>2*1024**2:raise ValueError('MCP response exceeds its local inspection budget')
    message=json.loads(raw)
   if not isinstance(message,dict) or message.get('id')!=identifier or 'error' in message or not isinstance(message.get('result'),dict):raise ValueError('MCP server did not accept this known read-only protocol operation')
   return message['result']
  finally:conn.close()
 def initialize(self):
  from mcp.types import LATEST_PROTOCOL_VERSION
  result=self.call('initialize',{'protocolVersion':LATEST_PROTOCOL_VERSION,'capabilities':{},'clientInfo':{'name':'Geo Intelligence read-only discovery','version':'0.2.21'}})
  version=result.get('protocolVersion')
  if not isinstance(version,str) or len(version)>40:raise ValueError('MCP initialization omitted its protocol version')
  self.version=version;self.call('notifications/initialized',notification=True);return result
 def discover(self):
  server=self.initialize();caps=server.get('capabilities',{});resources=self.call('resources/list').get('resources',[]) if 'resources' in caps else [];tools=self.call('tools/list').get('tools',[]) if 'tools' in caps else []
  if not isinstance(resources,list) or not isinstance(tools,list) or len(resources)>1000 or len(tools)>1000:raise ValueError('MCP inventory exceeds the bounded discovery limit')
  return {'protocol_version':self.version,'server':server.get('serverInfo',{}),'resources':[{'uri':r['uri'],'name':r.get('name',r['uri']),'mimeType':r.get('mimeType')} for r in resources if isinstance(r,dict) and isinstance(r.get('uri'),str)],'tools':[{'name':t['name'],'description':str(t.get('description',''))[:2000],'annotations':t.get('annotations',{}),'execution':'blocked; annotation alone is insufficient'} for t in tools if isinstance(t,dict) and isinstance(t.get('name'),str)],'pagination':'First bounded discovery page; no opaque supplier cursor followed automatically'}
 def read_listed(self,uri):
  # Recheck the current inventory. Never relay caller-supplied requests,
  # tool arguments, or URIs absent from this supplier's saved and live lists.
  resources=self.call('resources/list').get('resources',[])
  if (not isinstance(resources,list) or len(resources)>1000
      or not any(isinstance(item,dict) and item.get('uri')==uri for item in resources)):
   raise ValueError('The resource is absent from the current bounded supplier inventory. Refresh the connection.')
  return self.call('resources/read',{'uri':uri})
