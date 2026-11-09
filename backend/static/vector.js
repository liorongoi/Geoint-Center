const vectorImportState={metadata:null,busy:false};

function updateVectorImportScope(mission,job){
  const target=$('vectorImportJob');
  target.innerHTML=(mission.jobs||[]).map(item=>`<option value="${escapeHtml(item.id)}">${escapeHtml(item.name)}</option>`).join('');
  if(job)target.value=job.id;
  const imageSelect=$('vectorImportImage'),previous=imageSelect.value;
  imageSelect.innerHTML='<option value="">All images in this job · shared reference</option>'+(job?.images||[]).map(image=>`<option value="${escapeHtml(image.id)}">${escapeHtml(image.name||image.id)}</option>`).join('');
  if([...imageSelect.options].some(option=>option.value===previous))imageSelect.value=previous;
}

function vectorStatus(message,error=false){$('vectorImportStatus').textContent=message;$('vectorImportStatus').classList.toggle('error',error);}

function renderVectorMapping(){
  const data=vectorImportState.metadata,field=$('vectorClassField').value;
  $('vectorFixedLabelWrap').classList.toggle('hidden',Boolean(field));
  const count=field?data.field_value_counts[field]:0;
  $('saveVectorImport').disabled=count>200;
  $('vectorClassMapping').innerHTML=!field?'':count>200?'<p class="error">This attribute has more than 200 values. Choose a categorical class field.</p>':`<h4>Map source values to class names</h4><div class="vector-mapping-rows">${data.field_values[field].map(value=>`<label><span>${escapeHtml(value)}</span><input class="input-field" data-vector-class="${escapeHtml(value)}" value="${escapeHtml(value)}" maxlength="80" aria-label="Class name for ${escapeHtml(value)}"></label>`).join('')}</div>`;
}

async function readVectorImport(){
  if(vectorImportState.busy)return;
  const files=[...$('vectorImportFiles').files];if(!files.length){vectorStatus('Select a ZIP or the shapefile components first.',true);return;}
  vectorImportState.busy=true;$('readVectorImport').disabled=true;$('vectorImportPreview').classList.add('hidden');
  vectorStatus('Reading polygons, attributes and coordinate system…');
  const form=new FormData();files.forEach(file=>form.append('files',file));
  form.append('dataset',$('vectorImportDataset').value);form.append('source_crs',$('vectorImportCrs').value.trim());form.append('encoding',$('vectorImportEncoding').value);
  try{
    const data=await api('/api/vector-imports/preview',{method:'POST',body:form});vectorImportState.metadata=data;
    $('vectorDatasetWrap').classList.toggle('hidden',data.datasets.length<2);
    $('vectorImportDataset').innerHTML=data.datasets.map(name=>`<option ${name===data.dataset?'selected':''}>${escapeHtml(name)}</option>`).join('');
    if(data.needs_dataset){vectorStatus('This ZIP contains several datasets. Choose one, then read again.');return;}
    $('vectorImportPreview').classList.remove('hidden');
    $('vectorImportSummary').innerHTML=`<b>${Number(data.feature_count).toLocaleString()} polygons</b> · ${escapeHtml(data.source_crs)} → WGS84${data.invalid_count?`<p>${data.invalid_count} invalid/empty features will be excluded. ${escapeHtml(data.invalid_features[0]?.reason||'')}</p>`:''}`;
    $('vectorClassField').innerHTML='<option value="">Use one class for all polygons</option>'+data.fields.map(field=>`<option value="${escapeHtml(field)}">${escapeHtml(field)} · ${data.field_value_counts[field]} values</option>`).join('');
    const recommended=data.fields.find(field=>/^(class|label|landcover|lc_type|category|classname|lulc)$/i.test(field));
    if(recommended)$('vectorClassField').value=recommended;
    renderVectorMapping();vectorStatus('Polygons are ready. Choose class names, image scope and reference role before importing.');
  }catch(error){vectorImportState.metadata=null;vectorStatus(error.message,true);}
  finally{vectorImportState.busy=false;$('readVectorImport').disabled=false;}
}

async function saveVectorImport(){
  const metadata=vectorImportState.metadata,mission=state.missions.find(item=>item.id===state.currentMissionId),job=currentTrainingJob();
  if(!metadata?.feature_count||!mission||!job)return;
  const classField=$('vectorClassField').value,fixed=$('vectorFixedLabel').value.trim(),mapping={};
  document.querySelectorAll('[data-vector-class]').forEach(input=>mapping[input.dataset.vectorClass]=input.value.trim());
  if(!classField&&!fixed){vectorStatus('Enter a class name or choose a class attribute.',true);return;}
  $('saveVectorImport').disabled=true;vectorStatus('Saving polygon references and checking the automatic training rule…');
  try{
    const saved=await api(`/api/vector-imports/${metadata.id}/commit`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mission_id:mission.id,job_id:job.id,image_id:$('vectorImportImage').value||null,class_field:classField||null,class_mapping:mapping,fixed_label:fixed,role:$('vectorImportRole').value,confidence:Number($('vectorImportConfidence').value),notes:$('vectorImportNotes').value.trim(),skip_unlabelled:$('vectorSkipUnlabelled').checked})});
    state.groundTruth.unshift(...saved.labels);renderTrainingPanel();
    vectorStatus(`Imported ${saved.summary.labels} polygons across ${Object.keys(saved.summary.class_counts).length} classes as ${saved.summary.role==='train'?'training':'independent validation'} references. ${saved.summary.skipped_unlabelled+saved.summary.skipped_invalid} skipped.`);
    $('saveVectorImport').textContent='Imported';acceptAutomaticTrainingRuns(saved.automatic_training);
  }catch(error){vectorStatus(error.message,true);$('saveVectorImport').disabled=false;}
}

document.addEventListener('DOMContentLoaded',()=>{
  $('readVectorImport').onclick=readVectorImport;$('saveVectorImport').onclick=saveVectorImport;$('vectorClassField').onchange=renderVectorMapping;
  $('vectorImportJob').onchange=()=>{$('groundTruthJob').value=$('vectorImportJob').value;renderTrainingPanel();};
  $('vectorImportFiles').onchange=()=>{vectorImportState.metadata=null;$('vectorImportPreview').classList.add('hidden');$('vectorImportDataset').innerHTML='';$('vectorDatasetWrap').classList.add('hidden');$('saveVectorImport').textContent='Import references into job';vectorStatus('');};
  $('previewVectorMap').onclick=async()=>{const data=vectorImportState.metadata;if(!data?.feature_count)return;try{await api(`/api/vector-imports/${data.id}/map`,{method:'POST'});if(!$('missionWorkspace').classList.contains('hidden'))toggleMissionWorkspace();setWorkspaceView('map');toast('External polygons are displayed on the map. Their classes are saved only when you import them.');}catch(error){vectorStatus(error.message,true);}};
});
