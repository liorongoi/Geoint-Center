const anomalyState={runs:[],views:new Map(),token:0};
const anomalyNames={rx:'Regularized RX',local_rx:'Local RX',isolation_forest:'Isolation Forest',autoencoder:'Spectral autoencoder'};

function renderAnomalyPanel(){
  const mission=state.missions.find(m=>m.id===state.currentMissionId);if(!mission)return;
  const token=++anomalyState.token,host=$('anomalyPanel');
  host.innerHTML=`<header class="anomaly-heading"><div><div class="eyebrow">FULL SPECTRUM · FOUR COMPLEMENTARY DETECTORS</div><h2>Spectral anomalies</h2><p>Explore unusual spectra across every connected science band. Inspect individual detectors, compare their votes, and save your evaluation.</p></div><span class="anomaly-orbit">◉</span></header><div class="anomaly-setup"><section class="anomaly-card"><h3>1 · Choose imagery</h3><label class="field-label">JOB<select id="anomalyJob" class="select-field">${(mission.jobs||[]).map(j=>`<option value="${escapeHtml(j.id)}">${escapeHtml(j.name)}</option>`).join('')}</select></label><div id="anomalyImages" class="anomaly-images"></div><p class="anomaly-note">Use Raster access to connect streaming, downloaded or local imagery. Benchmark validation/test images are protected from background fitting.</p></section><section class="anomaly-card"><h3>2 · Configure detection</h3><div class="anomaly-detector-options">${Object.entries(anomalyNames).map(([id,name])=>`<div class="anomaly-choice"><label><input type="checkbox" value="${id}" data-anomaly-detector checked><span>${name}<small>${id==='autoencoder'?'Learns background spectra; full-band reconstruction error':id==='local_rx'?'Local background rings with regularized covariance':id==='rx'?'Global background with Ledoit–Wolf covariance':'Nonlinear isolation using all spectral features'}</small>${satelliteGuidance(id,true)}</span></label>${satelliteDetails(id)}</div>`).join('')}</div><div class="anomaly-options"><label class="field-label">FLAG AT PERCENTILE<input id="anomalyThreshold" class="input-field" type="number" min="50" max="99.99" step="0.01" value="99.5"></label><label class="field-label">MINIMUM DETECTOR VOTES<select id="anomalyMinVotes" class="select-field"><option>1</option><option selected>2</option><option>3</option><option>4</option></select></label><label class="field-label">ANALYSIS GRID<select id="anomalyGrid" class="select-field"><option value="192">192 px · quick</option><option value="384">384 px · standard</option><option value="768">768 px · detailed</option><option value="1024">1024 px · detailed</option></select></label><label class="field-label">BACKGROUND SOURCE<select id="anomalyBackground" class="select-field"><option value="scene">Unreviewed scene background</option><option value="reviewed_normal">Approved normal regions · new candidate</option></select></label></div><details><summary>Local windows & autoencoder training</summary><div class="anomaly-options"><label class="field-label">OUTER WINDOW · ODD PIXELS<input id="anomalyOuter" class="input-field" type="number" value="63" step="2" min="19" max="255"></label><label class="field-label">GUARD WINDOW · ODD PIXELS<input id="anomalyGuard" class="input-field" type="number" value="31" step="2" min="17" max="127"></label><label class="field-label">AUTOENCODER MAXIMUM EPOCHS<input id="anomalyEpochs" class="input-field" type="number" value="15" min="1" max="200"></label></div><p class="anomaly-note">Local RX shares a covariance within each 16 × 16 score cell. The autoencoder trains from scratch on selected background and keeps its best reconstruction holdout checkpoint.</p></details><p class="anomaly-note">99.5 selects roughly the highest 0.5% of scores per detector, subject to ties. Percentiles are scene-relative ranks, not probabilities. No labeled anomalies are needed to begin.</p><button id="runAnomalies" class="primary-button full-width">Run spectral anomaly detection</button><div id="anomalySubmitStatus" role="status"></div></section></div><div id="anomalyTrainingRule" class="anomaly-training-rule"></div><div class="card-heading"><div><div class="eyebrow">SAVED RUNS & THRESHOLD VERSIONS</div><h2>Anomaly results</h2></div></div><div id="anomalyRuns"></div>`;
  if(!anomalyState.jobs)anomalyState.jobs={};
  const preferred=mission.jobs.find(j=>j.id===anomalyState.jobs[mission.id])||mission.jobs.find(j=>(j.images||[]).some(i=>!['validation','test'].includes(i.benchmark_split)&&state.rasterAccess.some(a=>a.mission_id===mission.id&&a.job_id===j.id&&a.image_id===i.id&&['local-ready','stream-ready','downloaded'].includes(a.status))));
  if(preferred)$('anomalyJob').value=preferred.id;
  $('anomalyJob').onchange=()=>{anomalyState.jobs[mission.id]=$('anomalyJob').value;renderAnomalyImages();loadAnomalyReadiness(mission.id,$('anomalyJob').value);};
  $('runAnomalies').onclick=submitAnomalies;
  renderAnomalyImages();loadAnomalyReadiness(mission.id,$('anomalyJob').value);
  pollAnomalyRuns(mission.id,token);
}

function renderAnomalyImages(){
  const mission=state.missions.find(m=>m.id===state.currentMissionId),job=mission?.jobs.find(j=>j.id===$('anomalyJob').value);
  $('anomalyImages').innerHTML=(job?.images||[]).map(image=>{
    const record=state.rasterAccess.find(a=>a.mission_id===mission.id&&a.job_id===job.id&&a.image_id===image.id);
    const protectedImage=['validation','test'].includes(image.benchmark_split),ready=['local-ready','stream-ready','downloaded'].includes(record?.status);
    return `<label class="anomaly-image ${protectedImage||!ready?'unavailable':''}"><input type="checkbox" data-anomaly-image value="${escapeHtml(image.id)}" ${protectedImage||!ready?'disabled':''}><span><b>${escapeHtml(image.id)}</b><small>${protectedImage?'Protected benchmark '+image.benchmark_split:ready?`${escapeHtml(record.mode)} · ${record.raster_metadata?.bands?.length||image.band_names?.length||'all'} connected bands`:'Raster access needed'}</small></span></label>`;
  }).join('')||'<p>No images in this job.</p>';
}

async function submitAnomalies(){
  const imageIds=[...document.querySelectorAll('[data-anomaly-image]:checked')].map(i=>i.value);
  const models=[...document.querySelectorAll('[data-anomaly-detector]:checked')].map(i=>i.value);
  if(!imageIds.length||!models.length){toast('Select images and at least one detector.');return;}
  const button=$('runAnomalies');button.disabled=true;
  try{
    const run=await api('/api/anomalies/runs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
      mission_id:state.currentMissionId,job_id:$('anomalyJob').value,image_ids:imageIds,models,
      max_size:Number($('anomalyGrid').value),percentile:Number($('anomalyThreshold').value),minimum_votes:Number($('anomalyMinVotes').value),
      outer_window:Number($('anomalyOuter').value),guard_window:Number($('anomalyGuard').value),epochs:Number($('anomalyEpochs').value),background_mode:$('anomalyBackground').value})});
    anomalyState.runs.unshift(run);renderAnomalyRuns();$('anomalySubmitStatus').textContent='Queued. Progress follows raster reads, detector fitting and map creation.';$('anomalyRuns').scrollIntoView({block:'start',behavior:'smooth'});
  }catch(error){$('anomalySubmitStatus').textContent=error.message;}finally{button.disabled=false;}
}

async function pollAnomalyRuns(missionId,token){
  let fingerprint='';
  while(anomalyState.token===token&&state.missionTab==='anomalies'&&state.currentMissionId===missionId){
    try{
      const runs=await api(`/api/anomalies/runs?mission_id=${encodeURIComponent(missionId)}`);
      if(anomalyState.token!==token||state.missionTab!=='anomalies')return;
      const next=JSON.stringify(runs.map(r=>[r.id,r.status,r.progress,r.stage,r.outputs?.length]));
      anomalyState.runs=runs;
      if(next!==fingerprint){renderAnomalyRuns();fingerprint=next;}
      for(const run of runs.filter(r=>r.trigger==='correction_threshold'&&['queued','running'].includes(r.status))){
        if(!pollAnomalyRuns.notified)pollAnomalyRuns.notified=new Set();
        if(!pollAnomalyRuns.notified.has(run.id)){toast(`Training is happening: ${anomalyNames[run.requested_models[0]]}.`);pollAnomalyRuns.notified.add(run.id);}
      }
    }catch(error){if($('anomalyRuns')&&!anomalyState.runs.length)$('anomalyRuns').textContent=error.message;}
    await new Promise(resolve=>setTimeout(resolve,2000));
  }
}

function anomalyMetricHtml(metric){
  const pct=v=>v==null?'—':`${(v*100).toFixed(1)}%`;
  if(metric?.status!=='measured')return '<p class="anomaly-note">Precision / Recall / F1: unmeasured · save independent validation references to measure them.</p>';
  return `<div class="validation-metrics"><span><b>${pct(metric.precision)}</b>Precision</span><span><b>${pct(metric.recall)}</b>Recall</span><span><b>${pct(metric.f1)}</b>F1</span></div><p class="anomaly-note">Anomaly is the positive class · ${metric.pixels.toLocaleString()} validation pixels. ${escapeHtml(metric.scope)}</p>`;
}

function renderAnomalyRuns(){
  const host=$('anomalyRuns');if(!host)return;
  host.innerHTML=anomalyState.runs.map(run=>`<article class="anomaly-run"><div class="anomaly-run-heading"><div><div class="eyebrow">${run.candidate_version?'REVIEWED BACKGROUND · SEPARATE CANDIDATE':'UNSUPERVISED BACKGROUND FIT'} · ${fmtDate(run.created_at)}</div><h3>${run.requested_models.map(id=>anomalyNames[id]).join(' + ')}</h3></div><span class="count-chip">${escapeHtml(run.status)} · ${run.progress||0}%</span></div><div class="pipeline-progress" role="progressbar" aria-label="Anomaly run progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${run.progress||0}"><div class="pipeline-progress-fill" style="width:${run.progress||0}%"></div></div><p class="anomaly-stage">${escapeHtml(run.stage)}</p>${run.candidate_version?'<p class="anomaly-note">Candidate saved independently. Original detector outputs and thresholds remain available.</p>':''}${(run.outputs||[]).filter(o=>o.default_view).map((output,index)=>anomalyOutputHtml(run,output,run.outputs.indexOf(output))).join('')}<a href="/api/anomalies/runs/${encodeURIComponent(run.id)}/files/report.json" download>Complete run report ↓</a></article>`).join('')||'<div class="review-empty"><h3>No anomaly runs yet</h3><p>Select imagery above. All connected science bands enter the four detectors.</p></div>';
  host.querySelectorAll('[data-anomaly-apply]').forEach(button=>button.onclick=async()=>{
    const card=button.closest('[data-anomaly-output]'),run=anomalyState.runs.find(r=>r.id===card.dataset.run),index=Number(card.dataset.index);
    button.disabled=true;
    try{const view=await api(`/api/anomalies/runs/${run.id}/views`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_index:index,percentile:Number(card.querySelector('[data-anomaly-percentile]').value),minimum_votes:Number(card.querySelector('[data-anomaly-votes]').value)})});anomalyState.views.set(`${run.id}:${index}`,view);renderAnomalyRuns();}catch(error){toast(error.message);}finally{button.disabled=false;}
  });
  host.querySelectorAll('[data-anomaly-map]').forEach(button=>button.onclick=()=>{
    const card=button.closest('[data-anomaly-output]'),run=anomalyState.runs.find(r=>r.id===card.dataset.run),index=Number(card.dataset.index);
    openAnomalyMap(run,index,anomalyState.views.get(`${run.id}:${index}`)||run.outputs[index].latest_view||run.outputs[index].default_view,button.dataset.anomalyMap);
  });
}

function anomalyOutputHtml(run,output,index){
  const view=anomalyState.views.get(`${run.id}:${index}`)||output.latest_view||output.default_view;
  return `<section class="anomaly-output" data-anomaly-output data-run="${run.id}" data-index="${index}"><div class="anomaly-output-heading"><div><h3>${escapeHtml(output.image_id)}</h3><p><b>${output.band_count} science bands</b> · ${output.shape[2]} × ${output.shape[1]} analysis pixels · ${output.valid_pixels.toLocaleString()} valid pixels</p></div><details><summary>Bands & fitting details</summary><p>${escapeHtml(output.bands.join(' · '))}</p><p>${output.background_pixels.toLocaleString()} background pixels · ${output.excluded_bands.length} quality channels excluded · ${output.band_inventory.filter(b=>b.constant_in_background).length} constant background channels retained</p><p>Local RX uses one covariance per 16 × 16 cell; every score uses all bands. Autoencoder reconstruction holdout loss is not detection accuracy.</p></details></div><div class="anomaly-detector-maps">${Object.entries(output.detectors).map(([id,d])=>`<article><div><b>${escapeHtml(d.name)}</b><span>${view.detector_counts[id].toLocaleString()} flagged</span></div><button class="anomaly-map-preview" data-anomaly-map="${id}"><img src="${escapeHtml(d.preview)}" alt="${escapeHtml(d.name)} spectral anomaly heatmap"></button><div class="anomaly-score-legend"><span>Typical</span><i></i><span>Unusual</span></div><a href="${escapeHtml(d.scores)}" download>Raw score GeoTIFF ↓</a></article>`).join('')}</div><div class="anomaly-consensus"><div><div class="eyebrow">DETECTOR VOTES</div><h3>${view.flagged_pixels.toLocaleString()} flagged pixels <small>${view.flagged_percent}%</small></h3><p>${view.agreement_percent}% unanimous detector decisions · agreement does not establish correctness.</p><button class="anomaly-map-preview" data-anomaly-map="votes"><img src="${escapeHtml(view.preview)}" alt="Number of anomaly detectors flagging each pixel"></button><div class="anomaly-vote-legend">${Object.entries(view.vote_distribution).map(([n,count])=>`<span><i style="background:${['#102f36','#f6b847','#fa7f3c','#e853a0','#fc3d53'][n]}"></i>${n} votes · ${count.toLocaleString()}</span>`).join('')}</div></div><div><h3>Adjust sensitivity</h3><label class="field-label">DETECTOR PERCENTILE<input class="input-field" data-anomaly-percentile type="number" value="${view.percentile}" min="50" max="99.99" step="0.01"></label><label class="field-label">MINIMUM VOTES<select data-anomaly-votes class="select-field">${Array.from({length:view.detector_count},(_,i)=>`<option ${i+1===view.minimum_votes?'selected':''}>${i+1}</option>`).join('')}</select></label><button class="secondary-button" data-anomaly-apply>Apply threshold · save new view</button><button class="primary-button" data-anomaly-map="votes">Inspect & review on map</button><div class="anomaly-downloads"><a href="${escapeHtml(view.mask)}" download>Anomaly mask ↓</a><a href="${escapeHtml(view.votes)}" download>Detector votes ↓</a></div>${anomalyMetricHtml(view.metrics)}${view.metrics?.detectors?`<details><summary>Per-detector validation</summary><table class="model-vote-table"><thead><tr><th>Detector</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>${Object.entries(view.metrics.detectors).map(([id,m])=>`<tr><th>${escapeHtml(anomalyNames[id])}</th>${['precision','recall','f1'].map(k=>`<td>${m[k]==null?'—':(m[k]*100).toFixed(1)+'%'}</td>`).join('')}</tr>`).join('')}</tbody></table></details>`:''}</div></div></section>`;
}

async function loadAnomalyReadiness(missionId,jobId){
  if(!jobId)return;
  try{
    const data=await api(`/api/anomalies/readiness?mission_id=${encodeURIComponent(missionId)}&job_id=${encodeURIComponent(jobId)}`);
    if(state.currentMissionId!==missionId||$('anomalyJob')?.value!==jobId)return;
    $('anomalyTrainingRule').innerHTML=`<details><summary>Saved learning rule · ${data.policy.minimum_regions} corrected regions / ${data.policy.minimum_corrected_pixels} pixels / ${data.policy.minimum_images} images</summary><p>Approved normal pixels define the background; confirmed anomalies and validation regions are excluded. Each relevant detector receives a separate candidate when the saved thresholds are met. Training starts are the only notifications.</p>${data.algorithms.map(a=>`<p><b>${escapeHtml(a.name)}</b> · ${a.regions} regions · ${a.corrected_pixels} pixels · ${a.images} images<br><small>${escapeHtml(a.waiting_for.join(' · ')||'Ready')}</small></p>`).join('')}</details>`;
  }catch(error){if($('anomalyTrainingRule'))$('anomalyTrainingRule').textContent=error.message;}
}

async function openAnomalyMap(run,index,view,layer='votes'){
  try{
    if(state.resultReview)await closeAnalysisInspector();
    if(!$('missionWorkspace').classList.contains('hidden'))toggleMissionWorkspace();
    setWorkspaceView('map');$('mapWorkspace').classList.add('inspect-analysis');
    if($('analysisInspector'))$('analysisInspector').classList.add('hidden');
    let panel=$('anomalyInspector');if(!panel){panel=document.createElement('section');panel.id='anomalyInspector';panel.className='analysis-inspector';$('mapWorkspace').append(panel);}
    panel.classList.remove('hidden');
    const ctx={run,index,view,geometry:null,scope:'pixel',layer};state.anomalyMap=ctx;
    panel.innerHTML=`<header><div class="eyebrow">SPECTRAL ANOMALY REVIEW</div><h2>Inspect unusual spectra</h2><p>${run.outputs[index].band_count} science bands · threshold ${view.percentile} · ${view.minimum_votes}/${view.detector_count} votes</p><button id="closeAnomalyInspector" class="text-button">← Return to anomaly results</button></header><label class="field-label">MAP LAYER<select id="anomalyMapLayer" class="select-field"><option value="votes">Detector votes</option><option value="mask">Flagged anomalies only</option>${Object.entries(run.outputs[index].detectors).map(([id,d])=>`<option value="${id}">${escapeHtml(d.name)} heatmap</option>`).join('')}</select></label><p class="anomaly-note">Click the map to see each detector's score and decision. Percentiles are relative ranks, not confidence.</p><form id="anomalyCoordinateForm" class="inspect-coordinate-form"><label>Longitude<input id="anomalyLongitude" type="number" step="any" required></label><label>Latitude<input id="anomalyLatitude" type="number" step="any" required></label><button class="secondary-button">Inspect location</button></form><div id="anomalyPixelVotes">Choose a pixel.</div><form id="anomalyReviewForm" class="manual-review-form"><h3>Your evaluation</h3><div id="anomalyGeometryStatus">No reference region selected</div><div class="segment-prompt-actions"><button type="button" class="secondary-button" id="anomalyUsePixel">Use clicked pixel</button><button type="button" class="secondary-button" id="anomalyUseArea">Use drawn AOI</button></div><label class="field-label">REVIEWED AS<select id="anomalyReviewLabel" class="select-field"><option value="anomaly">Confirmed anomaly</option><option value="normal">Normal background / false alarm</option></select></label><label class="field-label">REFERENCE ROLE<select id="anomalyReviewRole" class="select-field"><option value="train">Background adaptation</option><option value="validation">Independent validation only</option></select></label><label class="field-label">CONFIDENCE<select id="anomalyReviewConfidence" class="select-field"><option value="3">3 · likely</option><option value="4">4 · strong</option><option value="5">5 · certain</option></select></label><label class="field-label">REVIEW NOTE<input id="anomalyReviewNotes" class="input-field" maxlength="500"></label><button id="saveAnomalyReview" class="primary-button full-width" disabled>Save my anomaly evaluation</button></form><div id="anomalyValidationMetrics">${anomalyMetricHtml(view.metrics)}</div><div id="anomalyReviewHistory" class="anomaly-note"></div>`;
    $('anomalyMapLayer').value=layer;
    const mapLayer=async()=>{await api(`/api/anomalies/runs/${run.id}/views/${view.id}/map?layer=${encodeURIComponent($('anomalyMapLayer').value)}`,{method:'POST'});};
    $('anomalyMapLayer').onchange=()=>mapLayer().catch(e=>toast(e.message));
    $('closeAnomalyInspector').onclick=async()=>{await api('/api/map/review/clear',{method:'POST'});state.anomalyMap=null;panel.classList.add('hidden');$('mapWorkspace').classList.remove('inspect-analysis');toggleMissionWorkspace();openMission(run.mission_id);setMissionTab('anomalies');resizeEmbeddedMap();};
    $('anomalyCoordinateForm').onsubmit=e=>{e.preventDefault();inspectAnomalyPixel(Number($('anomalyLongitude').value),Number($('anomalyLatitude').value));};
    $('anomalyUsePixel').onclick=()=>{if(!ctx.pixel){toast('Inspect a pixel first.');return;}ctx.scope='pixel';ctx.geometry=ctx.pixel.geometry;anomalyGeometryStatus(ctx);};
    $('anomalyUseArea').onclick=async()=>{try{const area=await api('/api/aoi');ctx.scope='area';ctx.geometry=area.geometry;anomalyGeometryStatus(ctx);}catch(e){toast(e.message);}};
    $('anomalyReviewForm').onsubmit=saveAnomalyReview;
    resizeEmbeddedMap();await mapLayer();loadAnomalyReviewHistory(ctx);
  }catch(error){toast(error.message);}
}

function anomalyGeometryStatus(ctx){$('anomalyGeometryStatus').textContent=ctx.geometry?ctx.scope==='pixel'?`Pixel ${ctx.pixel.row}, ${ctx.pixel.column}`:`Drawn ${ctx.geometry.type} · clipped to valid image pixels`:'No reference region selected';$('saveAnomalyReview').disabled=!ctx.geometry;}

async function inspectAnomalyPixel(longitude,latitude){
  const ctx=state.anomalyMap;if(!ctx||ctx.busy)return;ctx.busy=true;
  try{
    const pixel=await api(`/api/anomalies/runs/${ctx.run.id}/views/${ctx.view.id}/inspect?longitude=${longitude}&latitude=${latitude}`);
    if(state.anomalyMap!==ctx)return;
    ctx.pixel=pixel;if(ctx.scope==='pixel')ctx.geometry=pixel.geometry;
    $('anomalyLongitude').value=longitude;$('anomalyLatitude').value=latitude;
    $('anomalyPixelVotes').innerHTML=`<h3>${pixel.flagged?'Flagged anomaly':'Below combined threshold'}</h3><table class="model-vote-table"><thead><tr><th>Detector</th><th>Percentile</th><th>Flag</th></tr></thead><tbody>${pixel.votes.map(v=>`<tr><th>${escapeHtml(v.name)}<small>Score ${v.score.toPrecision(4)}</small></th><td>${v.percentile.toFixed(2)}</td><td>${v.flagged?'Yes':'No'}</td></tr>`).join('')}</tbody></table>`;
    anomalyGeometryStatus(ctx);
  }catch(error){toast(error.message);}finally{ctx.busy=false;}
}

async function saveAnomalyReview(event){
  event.preventDefault();const ctx=state.anomalyMap;if(!ctx?.geometry)return;
  const button=$('saveAnomalyReview');button.disabled=true;
  try{
    const result=await api(`/api/anomalies/runs/${ctx.run.id}/reviews`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_index:ctx.index,view_id:ctx.view.id,label:$('anomalyReviewLabel').value,role:$('anomalyReviewRole').value,confidence:Number($('anomalyReviewConfidence').value),geometry:ctx.geometry,notes:$('anomalyReviewNotes').value})});
    ctx.view.metrics=result.metrics;anomalyState.views.set(`${ctx.run.id}:${ctx.index}`,ctx.view);$('anomalyValidationMetrics').innerHTML=anomalyMetricHtml(result.metrics);
    if(result.automatic_training.runs.length)toast('Training is happening in separate detector versions.');else toast('Your anomaly evaluation is saved locally.');
    loadAnomalyReviewHistory(ctx);
  }catch(error){toast(error.message);}finally{button.disabled=false;}
}

async function loadAnomalyReviewHistory(ctx){
  const reviews=await api(`/api/anomalies/runs/${ctx.run.id}/reviews`);
  if(state.anomalyMap===ctx)$('anomalyReviewHistory').textContent=`${reviews.filter(r=>r.image_index===ctx.index).length} saved evaluations for this image. Original scores remain unchanged.`;
}

window.addEventListener('message',event=>{
  const frame=$('mapFrame');if(!frame?.src||event.source!==frame.contentWindow||event.origin!==new URL(frame.src).origin)return;
  if(event.data?.type==='spectrascope_map_click'&&Number.isFinite(event.data.longitude)&&Number.isFinite(event.data.latitude)&&state.anomalyMap)inspectAnomalyPixel(event.data.longitude,event.data.latitude);
});
