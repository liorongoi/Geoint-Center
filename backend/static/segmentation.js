/* SAM proposes boundaries. The user assigns a class and explicitly approves. */
function mountSegmentationPanel(ctx){
  if($('segmentationPanel'))$('segmentationPanel').remove();
  const panel=document.createElement('details');panel.id='segmentationPanel';panel.className='segmentation-panel';panel.open=true;
  panel.innerHTML=`<div class="eyebrow">ASSISTED BOUNDARIES</div><h3>SAM 3 interactive segmentation <span>PRETRAINED</span></h3><p>Choose a drawing tool, then click or drag directly on the source image. Run SAM to propose a boundary, then assign and approve its class.</p><label class="field-label">DISPLAY BANDS · RED, GREEN, BLUE BY DEFAULT<input id="segmentBands" class="input-field" value="RED,GREEN,BLUE" aria-label="Segmentation display bands"></label><div class="segment-prompt-actions"><button type="button" class="secondary-button" id="segmentPointMode">Draw include points</button><button type="button" class="secondary-button" id="segmentExcludeMode">Draw exclude points</button><button type="button" class="secondary-button" id="segmentDrawBox">Draw SAM box</button></div><div class="segment-prompt-actions"><button type="button" class="secondary-button" id="segmentInclude">＋ Include clicked point</button><button type="button" class="secondary-button" id="segmentExclude">− Exclude clicked point</button></div><div class="segment-prompt-summary" id="segmentPromptSummary">No prompts yet</div><div class="segment-prompt-actions"><button type="button" class="primary-button" id="segmentPoints">Segment points</button><button type="button" class="secondary-button" id="segmentBox">Run SAM for box</button><button type="button" class="text-button" id="segmentClear">Clear prompts</button></div><div id="segmentStatus" class="segment-status" aria-live="polite"></div><div class="pipeline-progress hidden" id="segmentProgress" role="progressbar" aria-label="Segmentation progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><div id="segmentProgressFill" class="pipeline-progress-fill"></div></div><div id="segmentCandidates"></div><details><summary>Saved proposals for this image</summary><div id="segmentHistory">Loading…</div></details><p class="segment-note">Uses a three-band display composite. Other spectral bands stay with the LULC classifiers. Proposals are unclassified and become training references only after you save your classification. SAM training is not enabled.</p>`;
  panel.innerHTML='<summary>SAM 3 · Segment a region</summary><div>'+panel.innerHTML+'</div>';
  panel.querySelector('h3').insertAdjacentHTML('afterend',satelliteGuidance('sam3_tracker'));
  $('manualReviewForm').before(panel);
  ctx.segmentPoints=[];ctx.segmentBoxGeometry=null;
  function pointMode(label){window.resultImageViewer?.setSamPoint(label,c=>{if(ctx.segmentPoints.length>=32){toast('Use at most 32 points.');return;}ctx.segmentPoints.push({longitude:c.longitude,latitude:c.latitude,label});updatePoints();});$('segmentStatus').textContent=label===1?'Click foreground points on the image, then Segment points.':'Click background points to exclude, then Segment points.';}
  $('segmentPointMode').onclick=()=>pointMode(1);$('segmentExcludeMode').onclick=()=>pointMode(0);
  $('segmentDrawBox').onclick=()=>{window.resultImageViewer?.selectBox(geometry=>{ctx.segmentBoxGeometry=geometry;$('segmentStatus').textContent='SAM box drawn. Select Run SAM for box to create proposals.';});$('segmentStatus').textContent='Drag a rectangle around the object on the image.';};
  function addPoint(label){
    if(!ctx.pixel){toast('Click a valid image pixel first.');return;}
    if(ctx.segmentPoints.length>=32){toast('Use at most 32 points per proposal.');return;}
    ctx.segmentPoints.push({longitude:ctx.pixel.longitude,latitude:ctx.pixel.latitude,label});
    updatePoints();
  }
  function updatePoints(){
    $('segmentPromptSummary').textContent=ctx.segmentPoints.length?`${ctx.segmentPoints.filter(p=>p.label===1).length} include / ${ctx.segmentPoints.filter(p=>p.label===0).length} exclude points`:'No prompts yet';
  }
  $('segmentInclude').onclick=()=>addPoint(1);$('segmentExclude').onclick=()=>addPoint(0);
  $('segmentClear').onclick=()=>{ctx.segmentPoints=[];ctx.segmentBoxGeometry=null;window.resultImageViewer?.clearSam();updatePoints();$('segmentStatus').textContent='Prompts cleared.';};
  $('segmentPoints').onclick=()=>{
    if(!ctx.segmentPoints.length&&ctx.pixel)addPoint(1);
    startSegmentation(ctx,false);
  };
  $('segmentBox').onclick=()=>startSegmentation(ctx,true);
  loadSegmentationHistory(ctx);
  api('/api/segmentation/models').then(data=>{const model=data.models?.[0];if(!model||state.resultReview!==ctx)return;if(!model.runtime_available||!model.weights_cached){$('segmentStatus').textContent='Install the separate offline SAM 3 component to enable segmentation. No account or download is needed in the app.';for(const id of ['segmentPoints','segmentBox'])$(id).disabled=true;}}).catch(error=>{if(state.resultReview===ctx)$('segmentStatus').textContent=error.message;});
}

async function loadSegmentationHistory(ctx){
  try{
    const tasks=await api(`/api/segmentations?run_id=${encodeURIComponent(ctx.runId)}&image_index=${ctx.index}`);
    if(state.resultReview!==ctx||!$('segmentHistory'))return;
    $('segmentHistory').innerHTML=tasks.length?tasks.map(task=>`<button type="button" class="text-button" data-segment-history="${escapeHtml(task.id)}">${escapeHtml(fmtDate(task.created_at))} · ${escapeHtml(task.provenance?.name||task.model||'Saved model')} · ${escapeHtml(task.status)} · ${task.candidates?.length||0} proposals</button>`).join(''):'No saved segmentations yet.';
    $('segmentHistory').querySelectorAll('[data-segment-history]').forEach(button=>button.onclick=async()=>{
      try{const task=await api(`/api/segmentations/${encodeURIComponent(button.dataset.segmentHistory)}`);if(state.resultReview===ctx)await pollSegmentation(ctx,task);}catch(error){toast(error.message);}
    });
  }catch(error){if(state.resultReview===ctx&&$('segmentHistory'))$('segmentHistory').textContent=error.message;}
}

async function startSegmentation(ctx,useBox){
  if(ctx.segmentBusy)return;
  ctx.segmentBusy=true;
  for(const id of ['segmentPoints','segmentBox'])$(id).disabled=true;
  try{
    const bands=$('segmentBands').value.split(',').map(s=>s.trim().toUpperCase()).filter(Boolean);
    if(bands.length!==3)throw new Error('Enter exactly three mapped band names.');
    let geometry=null;
    if(useBox){geometry=ctx.segmentBoxGeometry;if(!['Polygon','MultiPolygon'].includes(geometry?.type))throw new Error('Choose Draw SAM box and drag around the object on the image first.');}
    if(!useBox&&!ctx.segmentPoints.some(p=>p.label===1))throw new Error('Include a foreground point first.');
    $('segmentStatus').textContent='Submitting segmentation…';
    const task=await api(`/api/results/${encodeURIComponent(ctx.runId)}/${ctx.index}/segmentations`,{
      method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({points:ctx.segmentPoints,box_geometry:geometry,rgb_bands:bands})});
    if(state.resultReview!==ctx)return;
    ctx.segmentTaskId=task.id;
    await pollSegmentation(ctx,task);
  }catch(error){if(state.resultReview===ctx)$('segmentStatus').textContent=error.message;}
  finally{ctx.segmentBusy=false;if(state.resultReview===ctx)for(const id of ['segmentPoints','segmentBox'])if($(id))$(id).disabled=false;}
}

async function pollSegmentation(ctx,task){
  while(state.resultReview===ctx){
    showSegmentationTask(ctx,task);
    if(!['queued','running'].includes(task.status))break;
    await new Promise(resolve=>setTimeout(resolve,1500));
    if(state.resultReview!==ctx)return;
    task=await api(`/api/segmentations/${encodeURIComponent(task.id)}`);
  }
  if(state.resultReview===ctx)loadSegmentationHistory(ctx);
}

function showSegmentationTask(ctx,task){
  if(state.resultReview!==ctx||!$('segmentStatus'))return;
  $('segmentStatus').textContent=task.error||`${task.progress||0}% · ${task.stage}`;
  $('segmentProgress').classList.remove('hidden');
  $('segmentProgress').setAttribute('aria-valuenow',Number(task.progress)||0);
  $('segmentProgressFill').style.width=`${Number(task.progress)||0}%`;
  if(task.status!=='complete')return;
  $('segmentCandidates').innerHTML=`<p><b>${escapeHtml(task.provenance?.name||task.model||'Saved model')}</b></p><p class="segment-note">Quality is the model’s mask estimate, not measured accuracy. Compare boundaries with the source image.</p>${task.candidates.map(candidate=>`<article class="segment-candidate"><img src="${escapeHtml(candidate.preview)}" alt="SAM proposed region ${candidate.id+1} over source imagery"><div><b>Proposal ${candidate.id+1}</b><span>${candidate.pixels.toLocaleString()} source pixels · estimated quality ${(candidate.predicted_mask_quality*100).toFixed(1)}%</span></div><div class="segment-prompt-actions"><button type="button" class="secondary-button" data-segment-map="${candidate.id}">Preview on map</button><button type="button" class="primary-button" data-segment-use="${candidate.id}">Use this boundary</button></div><div class="segment-downloads"><a href="${escapeHtml(candidate.mask)}" download>GeoTIFF mask ↓</a><a href="${escapeHtml(candidate.geojson)}" download>GeoJSON ↓</a></div></article>`).join('')}<a class="text-button" href="/api/segmentations/${encodeURIComponent(task.id)}/report.json" download>Segmentation report ↓</a>`;
  $('segmentCandidates').querySelectorAll('[data-segment-map]').forEach(button=>button.onclick=async()=>{
    button.disabled=true;
    try{await api(`/api/segmentations/${encodeURIComponent(task.id)}/candidates/${button.dataset.segmentMap}/map`,{method:'POST'});toast('Proposal boundary added to the map.');}catch(error){toast(error.message);}finally{button.disabled=false;}
  });
  $('segmentCandidates').querySelectorAll('[data-segment-use]').forEach(button=>button.onclick=()=>{
    const candidate=task.candidates.find(c=>c.id===Number(button.dataset.segmentUse));
    ctx.scope='segment';ctx.geometry=candidate.geometry;ctx.sourceModel=null;
    ctx.segmentationId=task.id;ctx.segmentationCandidate=candidate.id;
    updateReviewGeometryStatus();toast('Boundary selected. Choose a class and save your classification to approve it.');
  });
}
