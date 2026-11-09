"""Per-class sampling evidence on the same bounded grid used for candidate training."""
import numpy as np
from raster_pipeline import label_mask
from algorithm_registry import MODEL_DETAILS


def measure(backend,item,max_size=384):
 store=backend.workflow_store;_,job=store.mission(item['mission_id'],item['job_id']);classes=item['classes']
 rows=[{'class':c,'training_pixels':0,'human_pixels':0,'teacher_pixels':0,'validation_pixels':0,'regions':0,'images':0} for c in classes];errors=[];valid_total=0
 with store.database() as db:labels=[__import__('json').loads(r['payload']) for r in db.execute('SELECT payload FROM ground_truth WHERE mission_id=? AND job_id=? ORDER BY created_at,rowid',(item['mission_id'],item['job_id']))]
 for image in job.get('images',[]):
  try:
   _,record=backend._reference_scope(item['mission_id'],item['job_id'],image['id']);stack=backend.read_stack(record,image,max_size=max_size,geometry=job.get('aoi'));valid_total+=int(stack.valid.sum())
   refs=[r for r in labels if not r.get('image_id') or r['image_id']==image['id']]
   from workflow_inputs import training_references
   train=training_references(refs,classes)
   val=[r for r in refs if r.get('role')=='validation' and r.get('benchmark_split')!='test' and r.get('origin')!='reviewed_pseudo_label']
   held=[{**r,'label':'__heldout__'} for r in refs if r.get('role')=='validation' or r.get('benchmark_split') in ('validation','test')]
   mask=label_mask(stack,train,classes);v=label_mask(stack,val,classes);excluded=label_mask(stack,held,['__heldout__'])>=0;mask[excluded]=-1
   if image.get('benchmark_split') in ('validation','test'):mask[:]=-1
   # Ownership removes duplicate/overlapping region contributions; human edits win.
   owners=np.full(mask.shape,-1,'int32');ordered=sorted(train,key=lambda r:r.get('origin')!='reviewed_pseudo_label')
   for index,r in enumerate(ordered):
    one=label_mask(stack,[r],classes);owners[(one>=0)&(one==mask)]=index
   for index,row in enumerate(rows):
    selected=mask==index;ids=np.unique(owners[selected]);ids=ids[ids>=0];row['training_pixels']+=int(selected.sum());row['validation_pixels']+=int((v==index).sum());row['images']+=int(selected.any());row['regions']+=len(ids)
    for owner in ids:
     count=int((selected&(owners==owner)).sum());row['teacher_pixels' if ordered[owner].get('origin')=='reviewed_pseudo_label' else 'human_pixels']+=count
  except Exception as exc:errors.append({'image_id':image['id'],'error':str(exc)})
 minimum=MODEL_DETAILS.get(item['algorithm'],{}).get('minimum',20);recommended=max(100,minimum*2)
 for row in rows:
  row['recommended_pixels']=recommended;row['coverage_percent']=round(100*row['training_pixels']/max(1,valid_total),2)
  row['status']='missing' if not row['training_pixels'] else 'add samples' if row['training_pixels']<recommended or row['regions']<3 or row['images']<2 else 'coverage target met'
 ready=not errors and len(classes)>=2 and all(r['training_pixels']>=5 for r in rows) and sum(r['training_pixels'] for r in rows)>=minimum
 return {'classes':rows,'fit_minimum_met':ready,'adapter_minimum_total_pixels':minimum,'minimum_pixels_each_class':5,'grid_max_size':max_size,'errors':errors,'scope':'Unique valid pixels on the selected training grid. Validation/test geometries are excluded from training. Latest human decisions own overlaps. Teacher-derived pixels are counted separately.','guidance':'Recommended coverage per class: at least '+str(recommended)+' pixels, 3 reviewed regions and 2 images. These are sampling guidelines, not measured accuracy or a guarantee of sufficient diversity. Qualification still uses independent validation goals.'}
