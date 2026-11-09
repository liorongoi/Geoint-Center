(() => {
  'use strict';
  let library=[],sequence=0,timer=null,reviewId=null,connectionError=false;
  const dialog=document.getElementById('localLibraryDialog');
  const statusName=w=>!w.enabled?'Paused':({waiting:'Waiting for first scan',scanning:'Scanning files…',settling:'Checking file stability…',ready:'Up to date','ready-with-issues':'Needs attention',attention:'Needs attention',empty:'No images found',error:'Scan failed'}[w.status]||w.status);
  function showSummary(watches,images){
    const enabled=watches.filter(w=>w.enabled).length;
    const issues=watches.filter(w=>w.error||w.issues?.length).length;
    const badge=document.getElementById('monitorSummaryBadge'),summary=document.getElementById('monitorSummary');
    if(badge)badge.textContent=enabled?`${enabled} active`:'Monitoring off';
    if(summary)summary.textContent=watches.length?`${images.length} image products · ${watches.length} folders${issues?` · ${issues} need attention`:''}. ${enabled?'Changes are checked automatically while services run.':'Check a folder to enable automatic updates.'}`:'Add a folder, then check “Monitor this folder” to keep your catalog updated.';
  }
  let catalogSignature=null;
  function syncCatalog(images,force=false){
    library=images;
    const ready=images.filter(i=>(i.status||'ready')==='ready');
    const signature=JSON.stringify(ready);
    if(!force&&signature===catalogSignature)return;
    catalogSignature=signature;
    updateLocalCatalogItems(ready,force);
  }
  async function summary(){try{const [w,i]=await Promise.all([api('/api/local-imagery/watches'),api('/api/local-imagery')]);showSummary(w.items,i.items);syncCatalog(i.items);}catch(exc){const node=document.getElementById('monitorSummary');if(node)node.textContent='Folder status unavailable. Open Manage folders for details.';}}
  const e=escapeHtml;
  const completeness=image=>({verified:'Matches expected bands',incomplete:'Missing expected bands','not-declared':'Expected bands not supplied'}[image.completeness?.state]||'Expected bands not supplied');
  const request=(url,method,body)=>api(url,{method,headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const error=message=>{document.getElementById('libraryError').textContent=message||'';};
  function renderImages(){
    document.getElementById('libraryProducts').innerHTML=library.length?`<div class="library-table-wrap"><table class="library-table"><thead><tr><th>Image product</th><th>Bands / files</th><th>Source</th><th>Grouping</th><th></th></tr></thead><tbody>${library.map(i=>`<tr><td><strong>${e(i.name)}</strong><small>${e(i.datetime?.slice(0,10))} · ${e(i.sensor||'Unspecified sensor')}</small><small>${e(i.date_basis||'')}</small>${i.monitor_error?`<small class="library-error">${e(i.monitor_error)}</small>`:''}</td><td>${i.raster_bands} bands<small>${i.sources?.length||1} source files</small><small>Band completeness: ${e(completeness(i))}</small></td><td>${i.watch_id?'Monitored folder':'Manual import'}</td><td><span class="library-pill ${i.status==='needs-review'?'review':''}">${i.status==='needs-review'?'Review required':'Ready'}</span></td><td><button class="text-button" data-review="${e(i.id)}">Inspect bands</button></td></tr>`).join('')}</tbody></table></div>`:'<p>No local products yet. Import a folder or files, or check a listed folder to enable monitoring.</p>';
    document.getElementById('libraryProducts').querySelectorAll('[data-review]').forEach(b=>b.onclick=()=>review(b.dataset.review));
  }
  async function refresh(){
    const ticket=++sequence;
    try{
      const [watches,images]=await Promise.all([api('/api/local-imagery/watches'),api('/api/local-imagery')]);
      if(ticket!==sequence||!dialog.open)return;
      if(connectionError){error('');connectionError=false;}
      syncCatalog(images.items);showSummary(watches.items,images.items);renderImages();
      const guidance=document.getElementById('watchPathGuidance');
      if(guidance)guidance.textContent=watches.allowed_roots?.length?`Allowed folders: ${watches.allowed_roots.join(', ')}. Enter the path visible to this service.`:'Choose a local folder on the computer running SpectraScope. Network shares are unavailable in the offline edition.';
      const pathInput=document.getElementById('watchPath');if(pathInput)pathInput.placeholder=watches.allowed_roots?.[0]||(watches.platform==='nt'?'C:\\Imagery\\Satellite products':'/Users/you/Imagery');
      document.getElementById('watchList').innerHTML=watches.items.map(w=>`<div class="watch-row"><div><label><input type="checkbox" data-watch="${e(w.id)}" ${w.enabled?'checked':''}>Monitor this folder</label><div class="watch-path">${e(w.path)}</div><small>${w.files_seen||0} files found · ${w.products||0} image product${w.products===1?'':'s'} · ${w.review_count||0} need review · checks every ${w.interval_seconds}s · ${w.recursive?'includes subfolders':'this folder only'}</small><small>Last scan: ${w.last_scan?e(new Date(w.last_scan).toLocaleString()):'Not scanned'}</small>${w.error?`<div class="library-error">${e(w.error)}</div>`:''}${w.issues?.length?`<p class="library-error">Some files need attention. Ready images remain available below.</p><details><summary>${w.issues.length} scan issue(s)</summary><pre>${e(w.issues.map(i=>`${i.path}: ${i.error}`).join('\n'))}</pre></details>`:''}</div><div class="watch-row-actions"><span class="watch-state ${w.issues?.length||w.error?'attention':!w.enabled?'paused':''}">${e(statusName(w))}</span><button class="secondary-button" data-scan-watch="${e(w.id)}" ${!w.enabled||w.status==='scanning'?'disabled':''}>Scan now</button><button class="text-button" data-remove-watch="${e(w.id)}">Remove</button></div></div>`).join('')||'<p>No folders configured. Monitoring is controlled only by each folder’s checkbox.</p>';
      document.getElementById('watchList').querySelectorAll('[data-scan-watch]').forEach(button=>button.onclick=async()=>{button.disabled=true;button.textContent='Scanning…';try{await request(`/api/local-imagery/watches/${button.dataset.scanWatch}/scan`,'POST',{});await refresh();}catch(exc){error(exc.message);}finally{button.disabled=false;button.textContent='Scan now';}});
      document.getElementById('watchList').querySelectorAll('[data-remove-watch]').forEach(button=>button.onclick=async()=>{if(!confirm('Remove this monitoring entry? Original files and existing job inputs are preserved.'))return;try{++sequence;await api(`/api/local-imagery/watches/${button.dataset.removeWatch}`,{method:'DELETE'});await refresh();}catch(exc){error(exc.message);}});
      document.getElementById('watchList').querySelectorAll('[data-watch]').forEach(input=>input.onchange=async()=>{
        ++sequence;input.disabled=true;
        try{await request(`/api/local-imagery/watches/${input.dataset.watch}`,'PATCH',{enabled:input.checked});error('');await refresh();}
        catch(exc){error(exc.message);await refresh();}
      });
    }catch(exc){connectionError=true;error(exc.message);}
  }
  function review(id){
    reviewId=id;
    const image=library.find(i=>i.id===id),box=document.getElementById('libraryReview');
    const choices=[...(image.sources||[]),...(image.alternate_sources||[])];
    box.hidden=false;
    box.innerHTML=`<h3>${e(image.name)} · bands and source files</h3><p>${image.warnings?.length?e(image.warnings.join(' · ')):'Each selected file belongs to this product. Band order follows the table.'}</p>${image.sensor?'':'<label>Sensor<select id="reviewedSensor" class="select-field"><option value="">Unspecified</option><option value="hls-l30">HLS L30</option><option value="hls-s30">HLS S30</option><option value="maxar">Maxar / Vantor</option><option value="cbers-4a-wpm">CBERS-4A WPM</option></select></label>'}<p>Band completeness: ${e(completeness(image))}. ${image.auxiliary_sources?.length||0} quality/display/atmospheric files excluded from science bands.</p>${image.auxiliary_sources?.length?`<details><summary>Excluded non-spectral assets</summary><pre>${e(image.auxiliary_sources.map(s=>`${s.relative_path}: ${s.excluded}`).join('\n'))}</pre></details>`:''}<div class="library-table-wrap"><table class="library-table"><thead><tr><th>Use</th><th>Source file</th><th>Channel names, in file order</th><th>Native grid</th></tr></thead><tbody>${choices.map((s,n)=>`<tr><td><input type="checkbox" data-source="${n}" ${n<(image.sources||[]).length?'checked':''}></td><td>${e(s.relative_path)}<small>${s.count} channel(s) · ${s.wavelength_nm?`${e(s.wavelength_nm)} nm`:'Wavelength not supplied'}</small></td><td><input class="input-field" data-names="${n}" value="${e((s.band_names||[s.band||s.descriptions?.[0]||'band_1']).join(', '))}" aria-label="Band names for ${e(s.relative_path)}"></td><td>${s.width} × ${s.height}<small>${e(s.crs)}</small></td></tr>`).join('')}</tbody></table></div><label style="display:block;margin-top:14px;font-size:12px"><input id="confirmBandGrouping" type="checkbox">I confirm these selected files and named bands form one image acquisition. I accept any listed incomplete-band warning.</label><div class="library-actions"><button class="primary-button" id="saveBandGrouping">Save band grouping</button><button class="text-button" id="closeBandGrouping">Close inspector</button></div>`;
    if(!choices.length)box.innerHTML='<p>This image uses the previous single-file catalog format. Its band names remain available in Raster access.</p>';
    else{
      document.getElementById('closeBandGrouping').onclick=()=>{reviewId=null;box.hidden=true;};
      document.getElementById('saveBandGrouping').onclick=async()=>{
        const button=document.getElementById('saveBandGrouping');button.disabled=true;
        try{
          const sources=[...box.querySelectorAll('[data-source]:checked')].map(input=>({relative_path:choices[Number(input.dataset.source)].relative_path,band_names:box.querySelector(`[data-names="${input.dataset.source}"]`).value.split(',').map(s=>s.trim())}));
          await request(`/api/local-imagery/groups/${id}/review`,'POST',{sources,sensor:document.getElementById('reviewedSensor')?.value||null,acknowledge_warnings:document.getElementById('confirmBandGrouping').checked});
          error('');box.hidden=true;reviewId=null;await refresh();toast('Band grouping saved for this exact image revision.');
        }catch(exc){error(exc.message);}finally{button.disabled=false;}
      };
    }
    box.scrollIntoView({block:'nearest'});
  }
  function open(issues=[]){
    if(dialog.open){refresh();return;}
    dialog.innerHTML=`<header><div><div class="eyebrow">LOCAL DATA</div><h2>Folder monitoring &amp; image library</h2><p>Choose folders to watch, see what was found, and inspect every band of an image.</p></div><button class="text-button" id="closeLocalLibrary" aria-label="Close library">✕</button></header><div class="local-library-body"><div id="libraryError" class="library-error" role="alert"></div><section class="library-section"><h3>Automatic folder monitoring</h3><form id="watchAddForm" class="watch-add-form"><label class="field-label">FOLDER PATH ON THIS COMPUTER<input class="input-field" id="watchPath" required placeholder="C:\\Imagery\\Satellite products"></label><button type="submit" class="secondary-button">Add folder to library</button><button type="button" id="chooseWatchFolder" class="secondary-button" hidden>Browse folders…</button><p id="watchPathGuidance" class="watch-path-guidance"></p><div class="watch-options"><label><input id="watchEnabled" type="checkbox">Monitor this folder automatically</label><label><input id="watchRecursive" type="checkbox" checked>Include subfolders</label></div></form><p>Checked: monitor while local services run and resume after restart. Unchecked: pause. Your checkbox is saved exactly as selected. In the browser edition, services continue in the notification area after a browser tab closes.</p><div id="watchList" class="watch-list"></div></section><section class="library-section"><h3>Image products</h3><div class="library-actions"><button class="secondary-button" id="useLocalLibrary">Browse all ready images →</button><button class="text-button" id="refreshLocalLibrary">Refresh library</button></div><p>This library shows all indexed images, without the search map or date filters. Images marked Review required must be checked before use.</p><div id="libraryProducts"></div></section><section id="libraryReview" class="library-review" hidden></section><div class="library-note">Monitoring indexes files in place. When you add an image to a mission/job, the app preserves copies of all selected bands. Folder changes update the catalog without changing existing job inputs. Raster formats: georeferenced GeoTIFF and JPEG2000.</div><details style="margin-top:16px"><summary>Custom multispectral / hyperspectral folder layouts</summary><p>Use filenames such as scene_B001.tif, scene_B002.tif, or sibling B001/B002 folders. Bare band filenames require review. For another layout, place a file named <code>spectrascope-image.json</code> beside the band folders. Paths stay inside the imported or monitored folder. Names and wavelengths are retained; no spectral band is silently discarded.</p><pre>{
  "name": "Survey scene 001",
  "datetime": "2026-10-18T10:00:00Z",
  "sensor": "hyperspectral",
  "bands": [
    {"path": "spectral/channel_a.tif", "name": "B001", "wavelength_nm": 420},
    {"path": "spectral/channel_b.tif", "name": "B002", "wavelength_nm": 430}
  ],
  "expected_bands": ["B001", "B002"]
}</pre><p>For a multiband file, declare it once with <code>band_names</code> listing every channel in file order. Scalar <code>scale</code> and <code>offset</code> entries override that file’s radiometric metadata; do not invent values. Online service URLs and VRT sources are not accepted.</p></details></div>`;
    dialog.showModal();
    document.getElementById('closeLocalLibrary').onclick=()=>dialog.close();
    document.getElementById('refreshLocalLibrary').onclick=refresh;
    const browse=document.getElementById('chooseWatchFolder');
    if(window.desktopSetup?.libraryDirectory){browse.hidden=false;browse.onclick=async()=>{try{const path=await window.desktopSetup.libraryDirectory();if(path)document.getElementById('watchPath').value=path;}catch(exc){error(exc.message);}};}
    document.getElementById('watchAddForm').onsubmit=async event=>{
      event.preventDefault();const button=event.submitter;button.disabled=true;
      try{await request('/api/local-imagery/watches','POST',{path:document.getElementById('watchPath').value,enabled:document.getElementById('watchEnabled').checked,recursive:document.getElementById('watchRecursive').checked});error('');document.getElementById('watchPath').value='';await refresh();}
      catch(exc){error(exc.message);}finally{button.disabled=false;}
    };
    document.getElementById('useLocalLibrary').onclick=()=>{
      selectCatalogTab('local');state.catalogTabs.local.revision++;
      syncCatalog(library,true);dialog.close();
    };
    error(issues?.map?.(i=>`${i.path}: ${i.error}`).join('\n')||'');
    refresh();timer=setInterval(refresh,5000);
  }
  dialog.addEventListener('close',()=>{clearInterval(timer);timer=null;reviewId=null;++sequence;});
  document.getElementById('monitorFoldersButton')?.addEventListener('click',()=>open());
  window.LocalImagery={open,refreshSummary:summary,showCatalog:async()=>{selectCatalogTab('local');state.catalogTabs.local.revision++;state.localCatalogFiltered=false;saveCatalogTab();catalogSignature=null;await summary();}};
  summary();setInterval(()=>{if(!dialog.open)summary();},5000);
})();
