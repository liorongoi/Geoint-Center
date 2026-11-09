/* Human decisions are stored separately from immutable model predictions. */
function modelInitialization(model){
  const known={openearthmap:'Released pretrained eight-class RGB pixel classifier',prithvi_crop:'Released pretrained 13-class temporal pixel classifier',dynamicworld:'Official pretrained encoder and LULC head · no fitting for inference',prithvi:'Official pretrained Prithvi encoder · frozen; classifier head trained locally',unet:'U-Net trained from scratch unless a compatible local checkpoint was continued',ensemble:'Derived classifier · fixed equal probability voting'};
  const source=model.training?.initialization||known[model.id]||(['random_forest','svm','xgboost'].includes(model.id)?'Fitted from training labels · no pretrained estimator':'See saved training provenance');
  const members=model.training?.members;
  return `<span class="model-initialization">${escapeHtml(source)}</span>${members?.length?`<span>Members: ${members.map(m=>`${escapeHtml(m.id)} ${(m.weight*100).toFixed(0)}%`).join(' · ')}</span>`:''}`;
}
function modelMetricDetails(metric,classes=[],sampleUnit='validation pixels'){
  const pct=value=>value==null?'—':`${(Number(value)*100).toFixed(1)}%`;
  const macro=metric.per_class?.['macro avg']||{};
  const names=classes.length?classes:Object.keys(metric.per_class||{}).filter(name=>!['accuracy','macro avg','weighted avg','micro avg','samples avg'].includes(name));
  return `<div class="validation-metrics"><span><b>${pct(metric.macro_precision??macro.precision)}</b> Precision</span><span><b>${pct(metric.macro_recall??macro.recall)}</b> Recall</span><span><b>${pct(metric.macro_f1)}</b> F1</span></div><span>Macro averages · mIoU ${pct(metric.mean_iou)} · ${Number(metric.pixels).toLocaleString()} ${escapeHtml(sampleUnit)}</span><details class="class-metrics"><summary>Per-class Precision, Recall & F1</summary><table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Samples</th></tr></thead><tbody>${names.map((name,index)=>{const row=metric.per_class?.[name];if(!row)return '';const predicted=metric.confusion_matrix?.reduce((sum,line)=>sum+(line[index]||0),0);return `<tr><th>${escapeHtml(name)}</th><td>${predicted===0?'—':pct(row.precision)}</td><td>${row.support===0?'—':pct(row.recall)}</td><td>${pct(row['f1-score'])}</td><td>${Number(row.support).toLocaleString()}</td></tr>`;}).join('')}</tbody></table><p>Precision: how often a predicted class is correct. Recall: how much of the reference class is detected. A dash means no predicted/reference samples; macro averages use zero for undefined class scores. ${escapeHtml(metric.macro_definition||'')}</p></details>`;
}

function benchmarkSummary(report){
  const b=report.benchmark;if(!b?.taxonomy_classes)return '';
  if(report.training_experiment){
    const e=report.training_experiment,counts=b.split_counts||{},base=e.baseline_test_metrics;
    const pct=v=>v==null?'—':`${(v*100).toFixed(1)}%`;
    const rows=[['Released pretrained model',base],['Fine-tuned candidate',report.test_metrics]];
    return `<section class="benchmark-summary"><div class="eyebrow">PRETRAINED MODEL · LOCAL FINE-TUNING</div><h3>${escapeHtml(b.dataset)}</h3><p>${counts.train||0} training / ${counts.validation||0} validation / ${counts.test||0} test patches · ${e.epochs} epochs · ${b.spectral_bands} input bands</p><p>${escapeHtml(e.selection)}. Original model preserved; this candidate is not automatically promoted.</p><table class="benchmark-score-table"><thead><tr><th>Same test images</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>${rows.map(([name,m])=>`<tr><th>${name}</th><td>${pct(m?.macro_precision)}</td><td>${pct(m?.macro_recall)}</td><td>${pct(m?.macro_f1)}</td></tr>`).join('')}</tbody></table>${report.test_metrics?modelMetricDetails(report.test_metrics,report.classes,'held-out reference pixels'):''}<details><summary>Training provenance & limits</summary><p>${escapeHtml(e.reference_origin)}</p><p>Native classes: ${escapeHtml((report.classes||[]).join(' · '))}</p><ul>${(b.limitations||[]).map(text=>`<li>${escapeHtml(text)}</li>`).join('')}</ul></details><a href="/api/results/${encodeURIComponent(report.id)}/report.json">Download fine-tuning report ↓</a> · <a href="?mission=${encodeURIComponent(report.mission_id)}&tab=reports#report-${encodeURIComponent(e.baseline_run_id)}">Original pretrained report ↗</a></section>`;
  }
  const selected=report.models?.find(m=>m.id===report.selected_model);
  const splits=b.split_counts||{};
  const global=report.global_validation;
  const percent=value=>value==null?'—':`${(value*100).toFixed(1)}%`;
  const globalHtml=global?`<section class="benchmark-global"><div class="eyebrow">INDEPENDENT TRANSFER CHECK</div><h3>Worldwide reference points</h3><p>${Number(global.points_evaluated).toLocaleString()} / ${global.points_attempted} attempted points · ${global.reference_classes} shared broad classes · ${Object.keys(global.continent_counts||{}).length} continents</p><p>${Object.entries(global.continent_counts||{}).map(([name,n])=>`${escapeHtml(name)} ${n}`).join(' · ')}</p><p>24 fine reference classes are available in the source archive. This regional model is evaluated only on the seven mapped broad classes.</p><table class="benchmark-score-table"><thead><tr><th>Algorithm</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>${(global.models||[]).map(m=>`<tr class="${m.id===global.selected_model?'benchmark-selected':''}"><th>${escapeHtml(m.name)}${m.id===global.selected_model?' · agent choice':''}</th><td>${percent(m.metrics?.macro_precision)}</td><td>${percent(m.metrics?.macro_recall)}</td><td>${percent(m.metrics?.macro_f1)}</td></tr>`).join('')}</tbody></table><details><summary>Class metrics, exclusions & methodology</summary>${(global.models||[]).map(m=>`<h4>${escapeHtml(m.name)}</h4>${modelMetricDetails(m.metrics,global.classes,'independent points')}`).join('')}<h4>Reference class coverage</h4><p>${Object.entries(global.class_counts||{}).map(([name,n])=>`${escapeHtml(name)} ${n}`).join(' · ')}</p><h4>Excluded attempts</h4>${Object.entries(global.excluded||{}).map(([reason,n])=>`<p>${n}: ${escapeHtml(reason)}</p>`).join('')||'<p>None</p>'}<ul>${(global.limitations||[]).map(s=>`<li>${escapeHtml(s)}</li>`).join('')}</ul></details><a href="/api/results/${encodeURIComponent(report.id)}/global-point-audit.json">Download point predictions & source audit ↓</a></section>`:b.pretrained?'<p>Pixel classifier loaded with its original pretrained head. Model agreement is unmeasured for a single classifier.</p>':b.source_run_id?`<p class="benchmark-pending">The separate worldwide transfer check is in the <a href="?mission=${encodeURIComponent(report.mission_id)}&tab=reports#report-${encodeURIComponent(b.source_run_id)}">original five-model report</a>. This ensemble report covers the regional test split.</p>`:'<p class="benchmark-pending">Worldwide point evaluation: awaiting the separate Landsat / GLC transfer report.</p>';
  return `<section class="benchmark-summary"><div class="eyebrow">${b.pretrained?"PRETRAINED LULC · NO LOCAL FITTING":"MEASURED BENCHMARK · REGIONAL PILOT"}</div><h3>${escapeHtml(b.dataset)} · ${b.covered_classes}/${b.taxonomy_classes} classes ${b.pretrained?"in native head":"covered"}</h3><p>${escapeHtml((b.countries||[]).join(', '))} · ${splits.train||0} training / ${splits.validation||0} validation / ${splits.test||0} test patches · ${b.spectral_bands||6} input bands</p><p>${escapeHtml(b.scope)}</p>${report.test_metrics?`<div class="benchmark-agent"><h4>${b.pretrained?"Released model on reference test windows":"Agent on untouched test images"}</h4><p>${b.pretrained?"Official pretrained classification head retained. No local fitting or model selection.":`Selected ${escapeHtml(selected?.name||report.selected_model||'ensemble')} using validation macro F1. The test set did not influence selection or fitting.`}</p>${modelMetricDetails(report.test_metrics,report.classes,b.pretrained?'reference pixels':'independent test pixels')}</div>`:''}<details><summary>Class coverage & benchmark limits</summary><p>${b.pretrained?"Native pretrained classes":"Fitted"}: ${escapeHtml((report.classes||[]).join(' · '))}</p><p>Not fitted: ${escapeHtml((b.untrained_classes||[]).join(' · ')||'None')}</p><ul>${(b.limitations||[]).map(s=>`<li>${escapeHtml(s)}</li>`).join('')}</ul>${b.pretrained?"":"<p>Prithvi uses a frozen pretrained encoder and a retrained classifier head. U-Net trains from scratch.</p>"}</details>${globalHtml}<div class="benchmark-report-links"><a href="/api/results/${encodeURIComponent(report.id)}/report.json">Download complete report ↓</a> · <a href="?mission=${encodeURIComponent(report.mission_id)}&tab=reports#report-${encodeURIComponent(report.id)}">Open this report directly ↗</a></div></section>`;
}

function renderInferenceActivity() {
  const active=state.reports.filter(run=>['queued','running','pausing','paused','stopping'].includes(run.status));
  if(active.length)renderInferenceActivity.lastRun=active[0].id;
  const recent=state.reports.find(run=>run.id===renderInferenceActivity.lastRun);
  const panel=$('inferenceActivity');
  if(!active.length&&!recent){panel.classList.add('hidden');return;}
  const runs=active.length?active:[recent];
  panel.classList.remove('hidden');
  panel.innerHTML=`<div class="activity-heading"><b>${active.length?'Analysis in progress':recent?.status==='failed'?'Analysis failed':recent?.status==='cancelled'?'Analysis stopped':recent?.status==='needs_input'?'Input needs attention':'Analysis finished'}</b><button class="text-button" id="dismissInferenceActivity">${active.length?(renderInferenceActivity.minimized?'Expand':'Minimize'):'Dismiss'}</button></div>${runs.map(run=>{
    const mission=state.missions.find(item=>item.id===run.mission_id),job=mission?.jobs.find(item=>item.id===run.job_id);
    const progress=Math.max(0,Math.min(100,Number(run.progress)||0));
    return `<div class="activity-run"><div><strong>${escapeHtml(run.run_kind==='training'?`Training · ${run.model_name||run.models?.[0]?.name||job?.name||run.job_id}`:job?.name||run.job_id)}</strong><b>${progress}%</b></div><div class="pipeline-progress" role="progressbar" aria-label="${escapeHtml(job?.name||'Inference')} progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${progress}"><div class="pipeline-progress-fill" style="width:${progress}%"></div></div><span>${escapeHtml(run.stage||run.status)}</span><small>${(run.models||[]).filter(model=>model.status==='complete').length}/${(run.requested_models||[]).length} models completed · ${escapeHtml(run.status)}</small></div>`;
  }).join('')}<small>Progress follows completed pipeline work; it is not an estimate of remaining time.</small>`;
  if(renderInferenceActivity.minimized)panel.classList.add('minimized');
  $('dismissInferenceActivity').onclick=()=>{if(active.length){renderInferenceActivity.minimized=!renderInferenceActivity.minimized;panel.classList.toggle('minimized',renderInferenceActivity.minimized);}else{renderInferenceActivity.lastRun=null;panel.classList.add('hidden');}};
  const current=active.find(run=>run.mission_id===state.currentMissionId)||recent;
  if(current){
    $('pipelineProgressFill').style.width=`${Number(current.progress)||0}%`;
    $('pipelineProgress').setAttribute('aria-valuenow',Number(current.progress)||0);
    $('pipelineProgressText').textContent=`${current.progress||0}% · ${current.stage||current.status}`;
    document.querySelectorAll('.pipeline-step').forEach((step,index)=>step.classList.toggle('active',index<=(current.progress<15?0:current.progress<85?1:2)));
  }
}

function mountAnalysisInspector(runId,index,output,classes,layer) {
  let ctx=state.resultReview;
  const same=ctx?.runId===runId&&ctx?.index===Number(index);
  if(!same)ctx={runId,index:Number(index),output,classes,pixel:null,geometry:null,scope:'pixel'};
  ctx.output=output;ctx.classes=classes;ctx.layer=layer;state.resultReview=ctx;
  let inspector=$('analysisInspector');
  if(!inspector){
    inspector=document.createElement('section');inspector.id='analysisInspector';inspector.className='analysis-inspector';
    $('mapWorkspace').append(inspector);
  }
  $('mapWorkspace').classList.add('inspect-analysis');
  inspector.classList.remove('hidden');
  if(!same||!$('pixelVotePanel')){
    inspector.innerHTML=`<header><div class="eyebrow">HUMAN REVIEW</div><h2>Inspect & classify</h2><p>Click any pixel on the image to compare model predictions.</p><button id="closeAnalysisInspector" class="text-button">Close image inspection</button></header><div id="analysisReviewSummary"></div><div id="pixelVotePanel" aria-live="polite"><p class="vote-empty">Choose a pixel on the image. Orange pixels have conflicting model predictions.</p></div><form id="manualReviewForm" class="manual-review-form"><h3>Your classification</h3><div id="reviewGeometryStatus">No pixel or drawn area selected</div><div class="review-scope-actions"><button type="button" class="secondary-button" id="reviewClickedPixel">Use clicked pixel</button><button type="button" class="secondary-button" id="reviewDrawnArea">Draw image region</button></div><label class="field-label">CORRECT CLASS<select id="manualReviewLabel" class="select-field">${classes.map(label=>`<option>${escapeHtml(label)}</option>`).join('')}</select></label><label class="field-label">USE THIS REFERENCE FOR<select id="manualReviewRole" class="select-field"><option value="train">Training / future fine-tuning</option><option value="validation">Independent validation only</option></select></label><label class="field-label">REVIEW NOTE<input class="input-field" id="manualReviewNotes" maxlength="500" placeholder="Reason or reference source"></label><button id="saveManualReview" class="primary-button full-width" type="submit">Save my classification</button><p>Creates a reviewed map and an image-specific reference in the local database. Original predictions and agreement remain unchanged.</p></form>`;
    $('closeAnalysisInspector').onclick=closeAnalysisInspector;
    $('manualReviewForm').onsubmit=saveManualClassification;
    $('reviewClickedPixel').onclick=()=>{if(!ctx.pixel){toast('Click a pixel on the image first.');return;}ctx.scope='pixel';ctx.geometry=ctx.pixel.geometry;updateReviewGeometryStatus();};
    $('reviewDrawnArea').onclick=()=>window.resultImageViewer?.selectBox();
  }
  $('analysisReviewSummary').append($('analysisMapReview'));
  if(!$('analysisReviewDetails')){
    const summary=$('analysisReviewSummary'),details=document.createElement('details'),heading=document.createElement('summary');
    details.id='analysisReviewDetails';details.open=true;heading.textContent=`${imageDisplayName(output.image_id)} · classes & display`;
    summary.replaceWith(details);details.append(heading,summary);
  }
  if(!$('inspectCoordinateForm')){
    const form=document.createElement('form');form.id='inspectCoordinateForm';form.className='inspect-coordinate-form';
    form.innerHTML='<label>Longitude<input id="inspectLongitude" type="number" step="any" min="-180" max="180" required placeholder="Longitude"></label><label>Latitude<input id="inspectLatitude" type="number" step="any" min="-90" max="90" required placeholder="Latitude"></label><button class="secondary-button" type="submit">Inspect location</button>';
    form.onsubmit=event=>{event.preventDefault();inspectAnalysisPixel(Number($('inspectLongitude').value),Number($('inspectLatitude').value));};
    $('pixelVotePanel').before(form);
  }
  if(output.has_manual_reviews&&!$('analysisResultLayer').querySelector('[value="reviewed"]'))$('analysisResultLayer').add(new Option('Human-reviewed land cover','reviewed'));
  $('analysisResultLayer').value=layer;
  if(!same||!$('segmentationPanel'))mountSegmentationPanel(ctx);
  window.mountPseudoLabelPanel?.(ctx);
  updateReviewGeometryStatus();
  resizeEmbeddedMap();
}

function updateReviewGeometryStatus(){
  const ctx=state.resultReview;
  if(!$('reviewGeometryStatus'))return;
  $('reviewGeometryStatus').textContent=ctx.geometry?ctx.scope==='pixel'?`Analysis pixel · row ${ctx.pixel.row}, column ${ctx.pixel.column}`:ctx.scope==='segment'?`SAM proposal ${ctx.segmentationCandidate+1} selected · assign a class and save to approve`:`Drawn ${ctx.geometry.type} · clipped to valid image pixels`:'No pixel or drawn area selected';
  $('saveManualReview').disabled=!ctx.geometry;
}

async function inspectAnalysisPixel(longitude,latitude){
  const ctx=state.resultReview;if(!ctx||ctx.inspectBusy)return;
  ctx.inspectBusy=true;
  $('pixelVotePanel').innerHTML='<p>Reading saved model predictions…</p>';
  try{
    const pixel=await api(`/api/results/${encodeURIComponent(ctx.runId)}/${ctx.index}/inspect?longitude=${encodeURIComponent(longitude)}&latitude=${encodeURIComponent(latitude)}`);
    if(state.resultReview!==ctx)return;
    ctx.pixel=pixel;ctx.sourceModel=null;
    $('inspectLongitude').value=pixel.longitude;$('inspectLatitude').value=pixel.latitude;
    $('analysisReviewDetails').open=false;
    if(ctx.scope==='pixel')ctx.geometry=pixel.geometry;
    $('pixelVotePanel').innerHTML=`<div class="vote-heading"><h3>Who predicted what?</h3><span class="${pixel.disagreement?'vote-conflict':'vote-agree'}">${pixel.disagreement?'Disagreement':'Unanimous agreement'}</span></div><p>${Number(pixel.longitude).toFixed(6)}, ${Number(pixel.latitude).toFixed(6)} · pixel ${pixel.row}, ${pixel.column}</p><table class="model-vote-table"><thead><tr><th>Model</th><th>Predicted class</th><th>Choose</th></tr></thead><tbody>${pixel.votes.map(vote=>`<tr><th>${escapeHtml(vote.name)}</th><td>${escapeHtml(vote.label||vote.status)}</td><td>${vote.label?`<button type="button" class="text-button" data-accept-vote="${escapeHtml(vote.id)}">Use class</button>`:'—'}</td></tr>`).join('')}</tbody></table><div class="vote-outcome"><b>Original final class:</b> ${escapeHtml(pixel.final_label)}<br><b>Your reviewed class:</b> ${escapeHtml(pixel.reviewed_label||'Not reviewed')}<br><b>Class votes:</b> ${Object.entries(pixel.tally).map(([label,count])=>`${escapeHtml(label)} ${count}`).join(' · ')}</div><p>Model class predictions are shown here. The final class follows the ${escapeHtml(pixel.combination)} policy. Per-model pixel probabilities were not saved.</p>`;
    $('manualReviewLabel').value=pixel.reviewed_label||pixel.final_label;
    $('pixelVotePanel').querySelectorAll('[data-accept-vote]').forEach(button=>button.onclick=()=>{const vote=pixel.votes.find(item=>item.id===button.dataset.acceptVote);$('manualReviewLabel').value=vote.label;ctx.sourceModel=vote.id;toast(`${vote.name} class selected. Save to confirm your decision.`);});
    $('manualReviewLabel').onchange=()=>ctx.sourceModel=null;
    updateReviewGeometryStatus();
  }catch(error){$('pixelVotePanel').innerHTML=`<p>${escapeHtml(error.message)}</p>`;}
  finally{ctx.inspectBusy=false;}
}

async function saveManualClassification(event){
  event.preventDefault();const ctx=state.resultReview;if(!ctx?.geometry)return;
  const button=$('saveManualReview');button.disabled=true;button.textContent='Saving reviewed classification…';
  try{
    const saved=await api(`/api/results/${encodeURIComponent(ctx.runId)}/${ctx.index}/reviews`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({label:$('manualReviewLabel').value,geometry:ctx.geometry,source_model:ctx.sourceModel||null,role:$('manualReviewRole').value,notes:$('manualReviewNotes').value.trim(),segmentation_id:ctx.scope==='segment'?ctx.segmentationId:null,segmentation_candidate:ctx.scope==='segment'?ctx.segmentationCandidate:null})});
    state.groundTruth.unshift(saved.ground_truth);
    const report=state.reports.find(run=>run.id===ctx.runId);if(report)report.outputs[ctx.index].has_manual_reviews=true;
    await reviewOutputOnMap(ctx.runId,ctx.index,'reviewed');
    if(ctx.pixel)await inspectAnalysisPixel(ctx.pixel.longitude,ctx.pixel.latitude);
    toast(`Saved ${saved.review.pixels.toLocaleString()} reviewed pixels. Reference available for ${saved.review.role==='train'?'fine-tuning':'validation'}.`);
    acceptAutomaticTrainingRuns(saved.automatic_training);
    window.refreshPseudoLabelPanel?.(ctx);
  }catch(error){toast(error.message);}
  finally{if($('saveManualReview')){$('saveManualReview').disabled=false;$('saveManualReview').textContent='Save my classification';}}
}

async function closeAnalysisInspector(){
  state.resultReview=null;$('mapWrap').append($('analysisMapReview'));$('analysisMapReview').classList.add('hidden');
  $('analysisInspector').classList.add('hidden');$('mapWorkspace').classList.remove('inspect-analysis');setMissionTab('reports');
}

window.addEventListener('message',event=>{
  const frames=['mapFrame','missionResultFrame'].map(id=>document.getElementById(id));
  const frame=frames.find(frame=>frame?.src&&event.source===frame.contentWindow&&event.origin===new URL(frame.src).origin);
  if(!frame)return;
  const visible=frame.id==='missionResultFrame'?!$('missionViewerPanel').classList.contains('hidden')&&!$('missionWorkspace').classList.contains('hidden'):!$('layout').classList.contains('hidden');
  if(event.data?.type==='spectrascope_map_click'&&Number.isFinite(event.data.longitude)&&Number.isFinite(event.data.latitude)&&state.resultReview&&visible)inspectAnalysisPixel(event.data.longitude,event.data.latitude);
});

function acceptAutomaticTrainingRuns(result){
  if(!result?.runs?.length)return;
  let seen;
  try{seen=new Set(JSON.parse(localStorage.getItem('spectrascope.training-notifications.v1')||'[]'));}catch{seen=new Set();}
  const fresh=[];
  for(const run of result.runs){
    if(!state.reports.some(item=>item.id===run.id))state.reports.unshift(run);
    if(!state.tuningRuns.some(item=>item.id===run.id))state.tuningRuns.unshift(run);
    if(!seen.has(run.id)){fresh.push(run.model_name||run.model);seen.add(run.id);}
  }
  localStorage.setItem('spectrascope.training-notifications.v1',JSON.stringify([...seen].slice(-200)));
  if(fresh.length)toast(`Automatic training is happening: ${fresh.join(', ')}. Progress is available in the app.`);
  renderInferenceActivity();
}

async function renderAutomaticTrainingPolicy(missionId,jobId){
  const panel=$('automaticTrainingPolicy');
  try{
    const data=await api(`/api/training/readiness?mission_id=${encodeURIComponent(missionId)}&job_id=${encodeURIComponent(jobId)}`);
    if(state.currentMissionId!==missionId||currentTrainingJob()?.id!==jobId)return;
    const policy=data.policy;
    const modelSelect=$('fineTuneModel'),selectedModel=modelSelect.value;
    modelSelect.innerHTML=data.algorithms.map(item=>`<option value="${escapeHtml(item.model_id)}">${escapeHtml(item.name)}</option>`).join('');
    if([...modelSelect.options].some(option=>option.value===selectedModel))modelSelect.value=selectedModel;
    panel.innerHTML=`<div class="auto-training-heading"><div><div class="eyebrow">SAVED RULE · ALL CURRENT & FUTURE ALGORITHMS</div><h3>Automatic training ${policy.enabled?'enabled':'disabled'}</h3></div><span>Notify when training starts</span></div><p>Each relevant algorithm trains independently after <b>${policy.minimum_regions} distinct regions</b>, <b>${policy.minimum_corrected_pixels} unique corrected analysis pixels</b>, and <b>${policy.minimum_images} images</b>. Only training references with confidence ${policy.minimum_confidence}/5 or higher count. Overlaps count once; validation areas are excluded.</p><div class="auto-training-algorithms">${data.algorithms.map(item=>`<div><b>${escapeHtml(item.name)}</b><span>${item.regions}/${policy.minimum_regions} regions · ${item.corrected_pixels}/${policy.minimum_corrected_pixels} pixels · ${item.images}/${policy.minimum_images} images</span><small>${escapeHtml(item.active_run_id?'Independent training is happening':item.ready?'Ready for automatic training':item.waiting_for.join(' · '))}</small></div>`).join('')}</div><p class="training-note">Corrections, trigger evidence, label IDs, model versions, and reports are saved locally. Original model predictions remain available.</p>`;
  }catch(error){panel.innerHTML=`<h3>Automatic training rule</h3><p>${escapeHtml(error.message)}</p>`;}
}
