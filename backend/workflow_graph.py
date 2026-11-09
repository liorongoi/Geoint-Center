"""Typed, versioned workflow DAGs; exact footprint intersection, no generated code."""
from __future__ import annotations
import json,math,re
from fastapi import HTTPException
from rasterio.features import is_valid_geom,bounds
from mission_workspace import now

TASK_TYPES={
 'lulc':{'id':'lulc','name':'Land cover (LULC)','output':'classification','available':True},
 'super_resolution':{'id':'super_resolution','name':'Super-resolution','output':'enhanced raster','available':True},
 'segmentation':{'id':'segmentation','name':'Object segmentation','output':'masks','available':True},
 'detection':{'id':'detection','name':'Object detection','output':'detections','available':True,'reason':'Oriented R-CNN requires its separate local pretrained component.'}}


def polygon(box):
 w,s,e,n=box;return {'type':'Polygon','coordinates':[[[w,s],[e,s],[e,n],[w,n],[w,s]]]}
def footprint(image):return image.get('geometry') or polygon(image['bbox'])
def geometries_intersect(a,b):
 """Planar GeoJSON ring intersection including holes and shared boundaries."""
 def polygons(g):
  if g['type']=='Polygon':return [g['coordinates']]
  if g['type']=='MultiPolygon':return g['coordinates']
  return []
 def edges(ring):return list(zip(ring,ring[1:]+ring[:1]))
 def orient(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
 def on(a,b,p):return abs(orient(a,b,p))<1e-10 and min(a[0],b[0])-1e-10<=p[0]<=max(a[0],b[0])+1e-10 and min(a[1],b[1])-1e-10<=p[1]<=max(a[1],b[1])+1e-10
 def crosses(a,b,c,d):return (orient(a,b,c)*orient(a,b,d)<0 and orient(c,d,a)*orient(c,d,b)<0) or any((on(a,b,c),on(a,b,d),on(c,d,a),on(c,d,b)))
 def ring_position(p,r):
  if any(on(x,y,p) for x,y in edges(r)):return 0
  inside=False
  for x,y in edges(r):
   if (x[1]>p[1])!=(y[1]>p[1]) and p[0]<(y[0]-x[0])*(p[1]-x[1])/(y[1]-x[1])+x[0]:inside=not inside
  return 1 if inside else -1
 def contained(p,poly):
  if ring_position(p,poly[0])==-1:return False
  for hole in poly[1:]:
   pos=ring_position(p,hole)
   if pos==1:return False
  return True
 if a['type']=='Point':return any(contained(a['coordinates'],p) for p in polygons(b))
 if b['type']=='Point':return geometries_intersect(b,a)
 box_a,box_b=bounds(a),bounds(b)
 if box_a[0]>box_b[2] or box_b[0]>box_a[2] or box_a[1]>box_b[3] or box_b[1]>box_a[3]:return False
 for pa in polygons(a):
  for pb in polygons(b):
   if any(crosses(x,y,u,v) for ra in pa for rb in pb for x,y in edges(ra) for u,v in edges(rb)):return True
   if contained(pa[0][0],pb) or contained(pb[0][0],pa):return True
 return False


def validate_graph(graph,store,require_qualified=False):
 nodes=graph.get('nodes',[]);links=graph.get('links',[])
 if not 2<=len(nodes)<=50 or len(links)>150:raise ValueError('Use 2–50 blocks and at most 150 connections')
 by_id={n['id']:n for n in nodes}
 if len(by_id)!=len(nodes) or any(not isinstance(k,str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}',k) for k in by_id):raise ValueError('Blocks need unique IDs')
 input_nodes=[n for n in nodes if n['type']=='input']
 if len(input_nodes)!=1:raise ValueError('A pipeline needs exactly one image input block')
 if not any(n['type']=='export' for n in nodes):raise ValueError('Add an export block')
 incoming={n['id']:[] for n in nodes};outgoing={n['id']:[] for n in nodes};seen=set()
 for link in links:
  source,target=link.get('from'),link.get('to')
  if source not in by_id or target not in by_id or source==target or (source,target) in seen:raise ValueError('Connections must join different existing blocks, without duplicates')
  seen.add((source,target));incoming[target].append(source);outgoing[source].append(target)
 degree={k:len(v) for k,v in incoming.items()};ready=[k for k in by_id if degree[k]==0];order=[]
 while ready:
  k=ready.pop(0);order.append(k)
  for target in outgoing[k]:
   degree[target]-=1
   if degree[target]==0:ready.append(target)
 if len(order)!=len(nodes):raise ValueError('Workflow contains a cycle; connect blocks in one direction')
 kinds={};schemas={};models={};blocked=[]
 for identifier in order:
  node=by_id[identifier];parents=incoming[identifier];kind=node['type']
  if kind=='input':
   if parents:raise ValueError('Image input cannot have an upstream block')
   kinds[identifier]='science';continue
  if not parents:raise ValueError('Every processing/output block must connect to the image input')
  if kind=='prepare':
   if len(parents)!=1 or kinds[parents[0]]!='science':raise ValueError('Band preparation requires science-image input')
   kinds[identifier]='science'
  elif kind=='algorithm':
   profile=store.model(node.get('profile_id',''))
   if profile['mission_id']!=graph['mission_id']:raise ValueError('All model configurations must belong to this mission')
   if profile['task']=='segmentation':
    if len([p for p in parents if kinds[p]=='science'])!=1 or any(kinds[p] not in ('science','prompt') for p in parents) or len(parents)>2:raise ValueError('SAM requires source science pixels and optionally one generated class-box prompt')
   elif len(parents)!=1 or kinds[parents[0]]!='science':raise ValueError('This algorithm needs original science pixels. Class codes cannot replace spectral bands; use a comparison or class-prompt block.')
   if profile['task'] not in graph['task_types']:raise ValueError('Algorithm task is not selected in this pipeline')
   if profile['state']!='qualified':blocked.append(profile['name'])
   if profile['task']=='segmentation' and not any(kinds[p]=='prompt' for p in parents) and not node.get('points') and not node.get('box_geometry'):raise ValueError('Automatic SAM needs explicitly configured foreground points or a box prompt')
   models[identifier]=profile;kinds[identifier]='masks' if profile['task']=='segmentation' else 'detections' if profile['task']=='detection' else 'enhanced' if profile['task']=='super_resolution' else 'classification';schemas[identifier]=profile.get('classes',[])
  elif kind=='class_prompt':
   if len(parents)!=1 or kinds[parents[0]]!='classification':raise ValueError('Class prompt requires one classification output')
   if not isinstance(node.get('class_value'),int) or not 1<=node['class_value']<=len(schemas[parents[0]]):raise ValueError('Select a real class value for the generated box prompt')
   kinds[identifier]='prompt'
  elif kind=='compare':
   if len(parents)!=2 or any(kinds[p]!='classification' for p in parents):raise ValueError('Compare requires exactly two classification outputs')
   if schemas[parents[0]]!=schemas[parents[1]]:raise ValueError('Class comparison requires identical class names and order. Different taxonomies cannot be averaged or compared as equivalent codes.')
   kinds[identifier]='binary';schemas[identifier]=['Same class','Different class']
  elif kind=='export':
   if any(kinds[p]=='science' for p in parents):raise ValueError('Export connects to algorithm/comparison outputs')
   if not node.get('formats') or any(f not in ('geotiff','geojson','shapefile') for f in node['formats']):raise ValueError('Choose GeoTIFF, GeoJSON and/or Shapefile exports')
   if any(kinds[p]=='detections' for p in parents) and 'geotiff' in node['formats']:raise ValueError('Oriented object boxes export as GeoJSON or Shapefile, not a semantic GeoTIFF. Choose vector formats.')
   if any(kinds[p]=='enhanced' for p in parents) and set(node['formats'])!={'geotiff'}:raise ValueError('Super-resolution exports as a continuous GeoTIFF. Use a separate export block for vector results.')
   kinds[identifier]='files'
  else:raise ValueError('Unknown block type; install a supported adapter rather than generated code')
 if any(not outgoing[k] and by_id[k]['type']!='export' for k in by_id):raise ValueError('Connect every branch to an export block')
 if not any(n['type']=='algorithm' for n in nodes):raise ValueError('Add at least one algorithm block')
 if blocked and require_qualified:raise ValueError('Waiting for qualification: '+', '.join(blocked))
 return {'valid':True,'order':order,'incoming':incoming,'kinds':kinds,'blocked_models':blocked,'models':models,'policy':'Science bands retain their declared calibration. Algorithms run on the selected AOI and analysis grid; Dynamic World stays at native 10 m. Compare uses A’s grid and nearest-neighbor B. Exports preserve original result pixels; polygon boundaries follow pixel edges.'}
