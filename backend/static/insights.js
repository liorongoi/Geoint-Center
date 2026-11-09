(() => {
  'use strict';
  const el=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;};
  const dialog=el('dialog','insight-studio');dialog.id='insightStudio';dialog.setAttribute('aria-label','Insights and saved layer library');
  dialog.innerHTML=`<header class="insight-header"><div class="insight-identity"><img src="/static/branding/spectral-atlas.svg" alt="SpectraScope" width="38" height="38"><div><span class="eyebrow">SPECTRASCOPE · CROSS-MISSION ANALYSIS</span><h1>Insights <span>/ Layer studio</span></h1></div></div><div class="insight-header-actions"><a href="/static/user-manual.html#insights" target="_blank" rel="noopener">Help ↗</a><button type="button" data-refresh>Refresh library</button><button type="button" data-close aria-label="Close Insights">✕</button></div></header>
  <div class="insight-body"><aside class="insight-library"><div class="insight-section-head"><span class="eyebrow">01 · CHOOSE INPUTS</span><h2>Your layer library</h2><p>Reuse results from any mission. Select one or two saved layers, in A → B order.</p></div><nav class="insight-tabs" aria-label="Layer sources"><button type="button" data-library-tab="saved" class="active">Saved</button><button type="button" data-library-tab="jobs">Job results</button><button type="button" data-library-tab="catalog">Local catalog</button><button type="button" data-library-tab="upload">Import</button></nav><input data-search type="search" placeholder="Search name, mission, job…" aria-label="Search layers"><div class="insight-inputs" data-inputs></div><div data-list class="insight-layer-list"></div><small class="insight-storage-note">SQLite stores metadata and lineage. Immutable GeoTIFFs stay in the local data folder. Back up both together.</small></aside>
  <main class="insight-workbench" data-workbench-view="chat"><nav class="insight-view-switch" aria-label="Insights workspace"><button type="button" data-workbench="chat" aria-pressed="true">Chat & plans</button><button type="button" data-workbench="map" aria-pressed="false">Map</button><button type="button" data-workbench="split" aria-pressed="false">Side by side</button></nav><section class="insight-map"><div class="insight-map-heading"><span class="eyebrow">SPATIAL WORKSPACE</span><span data-map-status>No layer loaded · select “Map” in the library</span></div><div class="insight-map-wrap"><iframe data-map title="Insights map with opacity and swipe controls" allow="fullscreen"></iframe><div class="insight-map-empty" data-map-empty><span>⌖</span><b>Give your results a new perspective.</b><p>Load a layer, then use “Layers · Opacity & swipe” on the map.</p></div></div></section>
  <section class="insight-assistant"><div class="insight-chat-heading"><div><span class="eyebrow">02 · DESCRIBE → REVIEW → APPROVE</span><h2>What would you like to discover?</h2></div><div><button type="button" data-llm>LLM connection</button> <button type="button" data-history>Operation history</button></div></div><div class="insight-chat" data-chat aria-live="polite"><p class="insight-chat-note">Choose layers, describe a request, review the interpretation and approve it. Connect an LLM for conversational requests, or use the command examples locally.</p></div><details class="insight-example-drawer"><summary>Example operations</summary><div class="insight-examples"><button data-example="intersection of A and B">Intersect masks</button><button data-example="cross-classify A and B">Cross-classify</button><button data-example="subtract B from A">Subtract</button><button data-example="change map">Class change</button><button data-example="mask A with B">Mask A</button><button data-example="A >= 95">Threshold</button><button data-example="extract class 6 from A">Extract class</button><button data-example="mean of A and B">Mean</button><button data-example="union of A and B">Union</button></div></details><form data-form><div class="insight-compose"><textarea data-message rows="4" maxlength="2000" placeholder="e.g. Mask A with B" aria-label="Describe a spatial operation" required></textarea><button type="submit" class="insight-primary">Interpret request ↗</button></div><label class="insight-grid-choice">Processing grid<select data-grid aria-label="Processing grid"><option value="">Keep original grid · up to 4 million pixels</option><option value="4000000">Reduce if needed · up to 4 million pixels</option><option value="1000000">Reduce if needed · up to 1 million pixels</option><option value="250000">Quick exploration · up to 250,000 pixels</option></select><small>New result only. Review exact resolution before approval. Classes use nearest neighbor.</small></label><label class="insight-output-name">OUTPUT NAME<input data-output-name maxlength="120" value="Insight layer" required></label></form></section></main></div><div class="insight-notice" data-notice role="status"></div>`;
  document.body.append(dialog);
  const q=s=>dialog.querySelector(s),list=q('[data-list]'),chat=q('[data-chat]'),selected=new Set();
  let library=[],sources=[],catalog=[],tab='saved',search='',page=1,activePlan=null,poll=null,version=0,lastPlan=null;
  const message=t=>{q('[data-notice]').textContent=t;};
  async function api(path,options={}){const r=await fetch(path,options);let d=await r.json().catch(()=>({}));if(!r.ok)throw Error(typeof d.detail==='string'?d.detail:JSON.stringify(d.detail||`Request failed (${r.status})`));return d;}
  const post=(path,body)=>api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});
  function action(label,fn,cls=''){const b=el('button',cls,label);b.type='button';b.onclick=async()=>{b.disabled=true;try{await fn();}catch(e){message(e.message);}finally{b.disabled=false;}};return b;}
  function paragraph(container,text,cls){container.append(el('p',cls,text));}
  function renderInputs(){
    const box=q('[data-inputs]');box.replaceChildren();
    [...selected].forEach((id,i)=>{const l=library.find(x=>x.id===id);const chip=action(`${String.fromCharCode(65+i)} · ${l?.name||id} ×`,()=>{selected.delete(id);lastPlan=null;renderInputs();renderList();},'insight-input-chip');box.append(chip);});
    if(!selected.size)box.append(el('small','','No inputs selected yet'));
  }
  function matching(item){return JSON.stringify([item.name,item.provenance?.mission_id,item.provenance?.job_id,item.mission_id,item.job_id,item.mission_name,item.job_name,item.provenance?.mission_name,item.provenance?.job_name,item.run_id,item.filename]).toLowerCase().includes(search);}
  async function refresh(){
    const [l,s]=await Promise.all([api('/api/layers'),api('/api/layers/sources')]);
    library=l.layers;catalog=l.catalog;sources=s.sources;if(!library.length&&sources.length&&tab==='saved'){tab='jobs';setTabs();}renderInputs();renderList();
  }
  function details(item){
    const d=el('details','insight-layer-meta');d.append(el('summary','','Values & provenance'));
    const pre=el('pre','',JSON.stringify({kind:item.kind,grid:item.grid,legend:item.legend,provenance:item.provenance,sha256:item.sha256},null,2));d.append(pre);return d;
  }
  function renderList(){
    list.replaceChildren();
    if(tab==='upload'){
      paragraph(list,'Import one analytical band per georeferenced GeoTIFF. Browser upload limit: 128 MiB. Larger existing job outputs can be saved from Job results; originals remain at full resolution. Processing grids are chosen separately below the chat. Class rasters use integer codes; binary masks use 0/1.');
      const form=el('form','insight-import');const name=el('input');name.placeholder='Layer name';name.required=true;name.maxLength=120;name.setAttribute('aria-label','Imported layer name');
      const kind=el('select');kind.setAttribute('aria-label','Raster values');[['continuous','Continuous values'],['categorical','Class codes'],['binary','Binary 0/1 mask']].forEach(([v,t])=>kind.add(new Option(t,v)));
      const file=el('input');file.type='file';file.accept='.tif,.tiff';file.required=true;file.setAttribute('aria-label','GeoTIFF file');
      const legend=el('textarea');legend.rows=3;legend.placeholder='Optional class legend JSON: [{"value":1,"label":"Water","color":"#528aff"}]';legend.setAttribute('aria-label','Optional class legend JSON');legend.maxLength=250000;
      const submit=el('button','insight-primary','Import GeoTIFF');submit.type='submit';form.append(name,kind,file,legend,submit);
      form.onsubmit=async e=>{e.preventDefault();submit.disabled=true;try{const f=new FormData();f.append('file',file.files[0]);f.append('name',name.value);f.append('kind',kind.value);f.append('legend',legend.value||'[]');await api('/api/layers/upload',{method:'POST',body:f});tab='saved';setTabs();await refresh();message('Imported and saved to the local layer library.');}catch(err){message(err.message);}finally{submit.disabled=false;}};
      list.append(form);paragraph(list,'You can also copy GeoTIFFs into data/spectrascope/local-rasters/ on the host, or browse files already exported to the Local catalog.');return;
    }
    const records=(tab==='saved'?library:tab==='jobs'?sources:catalog).filter(matching);
    if(!records.length){paragraph(list,tab==='saved'?'No saved layers yet. Open Job results to save an existing output, or Import a GeoTIFF.':'No matching layers found. Refresh after jobs finish or export layers to the local catalog.','insight-empty');return;}
    paragraph(list,`Showing ${Math.min(records.length,page*50)} of ${records.length} layers`,'insight-layer-subtitle');
    records.slice(0,page*50).forEach(item=>{
      const card=el('article','insight-layer');const title=el('div','insight-layer-title');
      if(tab==='saved'){
        const check=el('input');check.type='checkbox';check.checked=selected.has(item.id);check.setAttribute('aria-label',`Use ${item.name} as input`);
        check.onchange=()=>{if(check.checked){if(selected.size>=2){check.checked=false;message('Choose up to two inputs. Remove a selected input to change their order.');return;}selected.add(item.id);}else selected.delete(item.id);lastPlan=null;renderInputs();};title.append(check);
      }
      const heading=el('strong','',item.name);heading.title=item.name;title.append(heading);card.append(title);
      paragraph(card,tab==='catalog'?item.path:[item.kind,item.provenance?.mission_name||item.mission_name||item.provenance?.mission_id||item.mission_id,item.provenance?.job_name||item.job_name||item.provenance?.job_id||item.job_id,item.run_id?.slice(0,8)].filter(Boolean).join(' · '),'insight-layer-subtitle');
      const actions=el('div','insight-layer-actions');
      if(tab==='saved'){
        actions.append(action('Map',()=>showMap(item)),action('Export…',()=>exportOptions(item)),action('Metadata ↓',async()=>downloadMetadata(item)));
        card.append(actions,details(item));
      } else if(tab==='jobs'){
        const alreadySaved=library.some(l=>l.provenance?.source_id===item.source_id);
        actions.append(action(alreadySaved?'Save another copy':'Save to library',async()=>{message('Copying and verifying the full-resolution layer. Large rasters may take a moment; no downsampling is applied.');await post('/api/layers/save',{source_id:item.source_id,name:item.name.slice(0,120),kind:item.kind});await refresh();message('Immutable copy saved. Find it under Saved.');}));card.append(actions);
      } else {
        const kind=el('select');kind.setAttribute('aria-label','Catalog layer value type');[['continuous','Continuous'],['categorical','Class codes'],['binary','0/1 mask']].forEach(([v,t])=>kind.add(new Option(t,v)));kind.value=item.kind||'continuous';
        actions.append(kind,action('Import to library',async()=>{await post('/api/layers/import-path',{path:item.path,name:item.name.slice(0,120),kind:kind.value});await refresh();message('Catalog layer copied into the library.');}));card.append(actions);
      }
      list.append(card);
    });
    if(records.length>page*50)list.append(action('Show 50 more layers',()=>{page++;renderList();}));
  }
  function downloadMetadata(item){const blob=new Blob([JSON.stringify(item,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=el('a');a.href=url;a.download=item.id+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  function exportOptions(item){
    const card=el('article','insight-plan');card.append(el('h3','',`Export · ${item.name}`));paragraph(card,'The GeoTIFF keeps its original pixel values, grid, CRS and nodata. Include the metadata file to retain its class legend and lineage.');
    const buttons=el('div','insight-plan-actions');
    const a=el('a','','Download GeoTIFF ↓');a.href=item.download_url;a.download=item.id+'.tif';buttons.append(a);
    buttons.append(action('Save in local catalog',async()=>{const result=await post(`/api/layers/${encodeURIComponent(item.id)}/export`);message('Saved in '+result.catalog_path);await refresh();}));
    if(window.showDirectoryPicker)buttons.append(action('Choose directory on this computer…',async()=>{
      // Call the picker before network awaits to retain browser user activation.
      const folder=await window.showDirectoryPicker({mode:'readwrite'});
      const response=await fetch(item.download_url);if(!response.ok)throw Error('Could not download saved layer');
      const filename=item.id+'.tif';
      // Avoid overwriting a user-owned filename silently.
      let exists=false;try{await folder.getFileHandle(filename);exists=true;}catch(e){if(e.name!=='NotFoundError')throw e;}
      try{await folder.getFileHandle(item.id+'.json');exists=true;}catch(e){if(e.name!=='NotFoundError')throw e;}
      if(exists)throw Error('This layer filename already exists in the directory. Choose a different directory or use the existing export.');
      const file=await folder.getFileHandle(filename,{create:true}),stream=await file.createWritable();
      try{await stream.write(await response.blob());await stream.close();}catch(e){await stream.abort().catch(()=>{});throw e;}
      const sidecar=await folder.getFileHandle(item.id+'.json',{create:true}),meta=await sidecar.createWritable();await meta.write(JSON.stringify(item,null,2));await meta.close();message('GeoTIFF and metadata exported to your chosen directory.');
    }));
    else paragraph(card,'To choose a directory, use your browser’s “Save as” option for the GeoTIFF download. Direct directory picking requires a supported browser.');
    card.append(buttons);chat.append(card);card.scrollIntoView({block:'nearest'});
  }
  async function showMap(item){
    setWorkbench('map');message('Loading georeferenced layer…');const result=await post(`/api/layers/${encodeURIComponent(item.id)}/map`);
    const frame=q('[data-map]');if(!frame.src)frame.src=result.map_url;
    q('[data-map-empty]').hidden=true;q('[data-map-status]').textContent=item.name;message(result.rendering);
    dialog.querySelector('.insight-map-legend')?.remove();
    if(result.display_legend?.length){const legend=el('div','insight-map-legend');result.display_legend.slice(0,24).forEach(c=>{const entry=el('span','',`${c.value} · ${c.label}`);const swatch=el('i');swatch.style.background=c.color;entry.prepend(swatch);legend.append(entry);});if(result.display_legend.length>24)legend.append(el('span','',`${result.display_legend.length-24} more classes in metadata`));q('.insight-map').append(legend);}
  }
  function setWorkbench(view){q('.insight-workbench').dataset.workbenchView=view;dialog.querySelectorAll('[data-workbench]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.workbench===view));}
  dialog.querySelectorAll('[data-workbench]').forEach(b=>b.onclick=()=>setWorkbench(b.dataset.workbench));
  function setTabs(){dialog.querySelectorAll('[data-library-tab]').forEach(b=>b.classList.toggle('active',b.dataset.libraryTab===tab));}
  dialog.querySelectorAll('[data-library-tab]').forEach(b=>b.onclick=()=>{tab=b.dataset.libraryTab;page=1;setTabs();renderList();});
  q('[data-search]').oninput=e=>{search=e.target.value.toLowerCase();page=1;renderList();};
  dialog.querySelectorAll('[data-example]').forEach(b=>b.onclick=()=>{q('[data-message]').value=b.dataset.example;q('[data-message]').focus();});
  function renderPlan(p){
    const card=el('article','insight-plan');card.dataset.plan=p.id;card.append(el('span','eyebrow','MY INTERPRETATION · APPROVAL REQUIRED'),el('h3','',p.name));
    if(p.planner?.interpretation)paragraph(card,p.planner.interpretation);
    paragraph(card,p.summary);paragraph(card,p.routing||'','insight-plan-limits');paragraph(card,`Interpreter: ${p.planner?.model||'Local command interpreter'}`,'insight-plan-limits');p.inputs.forEach(i=>paragraph(card,`${i.alias} = ${i.name} · ${i.kind} · units: ${i.units||'unspecified'} · scale: ${i.scale} · offset: ${i.offset}`));
    paragraph(card,p.grid_policy);paragraph(card,p.nodata_policy);paragraph(card,p.limitations,'insight-plan-limits');
    const controls=el('div','insight-plan-actions');controls.append(action('Approve & create layer',async()=>{if(activePlan && activePlan!==p.id)throw Error('Another operation is still being monitored.');const result=await post(`/api/layers/plans/${p.id}/approve`);controls.replaceChildren();activePlan=p.id;monitor(result,card);},'insight-primary'),action('Discard plan',async()=>{await post(`/api/layers/plans/${p.id}/cancel`);controls.replaceChildren(el('span','','Plan discarded.'));}));card.append(controls);chat.append(card);card.scrollIntoView({block:'nearest'});return card;
  }
  async function monitor(p,card){
    clearTimeout(poll);let progress=card.querySelector('progress');if(!progress){progress=el('progress');progress.max=100;progress.setAttribute('aria-label','Insight operation progress');card.append(progress);}
    progress.value=p.progress||0;
    let stage=card.querySelector('.insight-stage');if(!stage){stage=el('p','insight-stage');card.append(stage);}stage.textContent=`${p.status} · ${p.stage||''} · ${p.progress||0}%`;
    if(p.status==='failed'){paragraph(card,p.error,'insight-error');activePlan=null;return;}
    if(p.status==='complete'){
      activePlan=null;await refresh();const item=library.find(l=>l.id===p.output_layer_id);
      if(item){card.append(action('Show result on map',()=>showMap(item)),action('Export result…',()=>exportOptions(item)));message('Result saved. Choose Show result on map when you want to view it.');}return;
    }
    if(dialog.open)poll=setTimeout(async()=>{try{const next=await api(`/api/layers/plans/${p.id}`);await monitor(next,card);}catch(e){message(e.message);}},1500);
  }
  q('[data-form]').onsubmit=async e=>{
    e.preventDefault();const b=e.target.querySelector('[type=submit]');b.disabled=true;b.textContent='Interpreting…';message('Interpreting your request. No spatial operation runs before approval.');
    const text=q('[data-message]').value;chat.append(el('p','insight-user-message',text));
    try{const p=await post('/api/layers/chat/plan',{message:text,layer_ids:[...selected],name:q('[data-output-name]').value,previous_plan_id:lastPlan,grid_max_pixels:q('[data-grid]').value?Number(q('[data-grid]').value):null});if(p.kind==='reply'){message('Explanation / clarification only. No raster processing scheduled.');paragraph(chat,p.reply,'insight-assistant-reply');paragraph(chat,p.routing,'insight-plan-limits');}else{message('Plan ready. Review the inputs and processing policy, then approve or discard it.');lastPlan=p.id;renderPlan(p);}q('[data-message]').value='';}catch(err){paragraph(chat,err.message,'insight-error');message(err.message);}finally{b.disabled=false;b.textContent='Interpret request ↗';}
  };
  async function connection(){
    const config=await api('/api/insight-llm');
    const card=el('article','insight-plan');card.append(el('span','eyebrow','OPTIONAL · NATURAL LANGUAGE PLANNING'),el('h3','','LLM connection'));
    paragraph(card,'Private requests use the local Ollama service started by Geo Intelligence. Review and approve each operation before it runs.');
    const form=el('form','insight-connection-form');
    function field(text,input){const label=el('label','',text);label.append(input);form.append(label);return input;}
    const enabled=el('input');enabled.type='checkbox';enabled.checked=config.enabled;field('Enable LLM interpretation',enabled);
    const provider=el('select');provider.add(new Option('Ollama · local','ollama'));provider.add(new Option('Independent endpoint · private requests blocked','compatible'));provider.value=config.provider;field('Provider',provider);
    const base=el('input');base.type='url';base.value=config.base_url;base.required=true;field('API base URL',base);
    if(config.managed_url)form.append(action('Use managed Ollama',()=>{provider.value='ollama';base.value=config.managed_url;enabled.checked=true;modelMode();}));
    const model=el('input');model.value=config.model;model.maxLength=200;model.placeholder='Exact model ID';field('Model ID (choose an installed or accessible model)',model);
    const policy=el('select');[['auto','Automatic · detect this machine'],['profile','Choose a runtime profile'],['manual','Choose my own model']].forEach(([v,t])=>policy.add(new Option(t,v)));policy.value=config.model_policy||'auto';field('Model selection',policy);
    const memory=el('select');[['auto','Automatic'],['compact','Compact · 16 GB · Qwen3.5 2B'],['balanced','Balanced · 24–32 GB · Qwen3.5 4B'],['quality','Higher quality · 48 GB+ · Qwen3.5 9B']].forEach(([v,t])=>memory.add(new Option(t,v)));memory.value=({low:'compact',standard:'balanced'})[config.memory_profile]||config.memory_profile||'auto';field('Runtime profile',memory);
    paragraph(form,`Detected: ${config.hardware?.os||'Unknown OS'} · ${config.hardware?.ram_gb??'unknown'} GB · ${config.hardware?.source||''}. ${config.runtime?.execution_mode||'CPU fallback'}. Selected: ${config.model}.`,'insight-plan-limits');
    function modelMode(){model.disabled=provider.value==='ollama'&&policy.value!=='manual';memory.disabled=provider.value==='ollama'&&policy.value==='auto';}
    policy.onchange=modelMode;memory.onchange=()=>{policy.value='profile';modelMode();model.value=config.profiles?.[memory.value]?.model||config.model;};modelMode();

    const key=el('input');key.type='password';key.autocomplete='new-password';key.placeholder=config.has_key?'Saved key · leave blank to keep':'Optional for local Ollama';field('API key · stored on this server, never returned',key);
    const clear=el('input');clear.type='checkbox';field('Remove the saved API key',clear);
    const status=el('p','insight-plan-limits',config.enabled?'Configured model: '+config.model:'No LLM enabled. The local command interpreter is available.');
    const controls=el('div','insight-plan-actions'),save=el('button','insight-primary','Save connection');save.type='submit';controls.append(save);
    async function persist(){const result=await post('/api/insight-llm',{enabled:enabled.checked,provider:provider.value,base_url:base.value,model:model.value,api_key:key.value||null,clear_key:clear.checked,memory_profile:memory.value,model_policy:provider.value==='compatible'?'manual':policy.value});model.value=result.model;key.value='';clear.checked=false;key.placeholder=result.has_key?'Saved key · leave blank to keep':'No key stored';return result;}
    form.onsubmit=async e=>{e.preventDefault();save.disabled=true;try{const result=await persist();status.textContent=result.enabled?'Connection settings saved for '+result.model+'. Use List available models to check access.':'LLM disabled. Command examples run locally.';message(status.textContent);}catch(err){status.textContent=err.message;}finally{save.disabled=false;}};
    controls.append(action('List available models',async()=>{await persist();status.textContent='Connecting to provider…';const result=await api('/api/insight-llm/models');const select=el('select');select.setAttribute('aria-label','Available models');select.add(new Option('Select a model…',''));result.models.forEach(id=>select.add(new Option(id,id)));select.onchange=()=>{if(select.value){model.value=select.value;policy.value='manual';modelMode();status.textContent='Selected '+select.value+'; save to use it.';}};form.querySelector('[data-available-models]')?.remove();select.dataset.availableModels='true';form.append(select);status.textContent=result.models.length+' models available. '+(result.configured_model_available?'Configured model is available.':'Choose a model and save.');}));
    provider.onchange=()=>{base.value=provider.value==='ollama'?'http://host.docker.internal:11434':'';key.value='';modelMode();};
    form.append(controls,status);card.append(form);paragraph(card,'Choose from installed local models or a configured compatible service. Offline installations use their supplied models; this screen does not download models. In Docker, host.docker.internal reaches Ollama on your host. Do not put API keys in the chat.');
    paragraph(card,'The Compact profile uses a 4K context, unloads the local LLM after each request, and defers LLM interpretation while an EO job is running. Explicit commands still work. For more headroom, choose a smaller installed model such as qwen3.5:2b.');
    paragraph(card,'The managed local model receives request text and selected layer metadata. Raster pixels stay local.','insight-plan-limits');
    const settings=el('dialog','insight-settings');settings.setAttribute('aria-label','LLM connection settings');const heading=el('header');heading.append(el('h2','','LLM connection'),action('Close',()=>settings.close()));settings.append(heading,card);document.body.append(settings);settings.addEventListener('close',()=>settings.remove(),{once:true});settings.showModal();
  }
  async function history(){
    const result=await api('/api/layers/plans/history');
    const section=el('section','insight-history');section.append(el('h3','','Saved operation history'));
    if(!result.plans.length)paragraph(section,'No operations yet.');
    for(const p of result.plans){
      const row=el('article','insight-history-row');row.append(el('strong','',p.name),el('small','',`${p.status} · ${p.created_at.slice(0,19).replace('T',' ')} UTC`),el('p','',p.message));
      if(p.status==='awaiting_approval')row.append(action('Review plan',()=>renderPlan(p)));
      if(p.status==='complete'){const l=library.find(x=>x.id===p.output_layer_id);if(l)row.append(action('Map',()=>showMap(l)),action('Export…',()=>exportOptions(l)));}
      if(['running','queued'].includes(p.status))row.append(action('Follow progress',()=>{activePlan=p.id;monitor(p,row);}));
      if(p.error)paragraph(row,p.error,'insight-error');section.append(row);
    }
    chat.append(section);section.scrollIntoView({block:'nearest'});
  }
  q('[data-llm]').onclick=()=>connection().catch(e=>message(e.message));
  q('[data-history]').onclick=()=>history().catch(e=>message(e.message));
  q('[data-close]').onclick=()=>dialog.close();q('[data-refresh]').onclick=()=>refresh().catch(e=>message(e.message));
  dialog.addEventListener('close',()=>{clearTimeout(poll);version++;});
  async function open(){
    if(!dialog.open)dialog.showModal();message('Loading saved layers and job outcomes…');
    const v=++version;try{await refresh();if(v!==version)return;message('Ready. Inputs may come from different missions and jobs.');
      const h=await api('/api/layers/plans/history');const running=h.plans.find(p=>['queued','running'].includes(p.status));
      if(running){activePlan=running.id;const card=el('article','insight-plan');card.append(el('h3','',running.name));chat.append(card);monitor(running,card);}
    }catch(e){message(e.message);}
  }
  const button=document.getElementById('insightsButton');if(button)button.onclick=open;
  document.addEventListener('click',event=>{if(event.target.closest('[data-save-result-layers]')){tab='jobs';setTabs();open();}});
  document.addEventListener('spectrascope:basemap-change',e=>{const frame=q('[data-map]');if(frame.src){const u=new URL(e.detail.map_url||frame.src);u.searchParams.set('provider_reload',Date.now());frame.src=u.href;}});
  window.SpectraScopeInsights={open};
  if(new URLSearchParams(location.search).get('insights')==='1')open();
})();
