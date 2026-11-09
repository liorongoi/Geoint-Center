const $ = (id) => document.getElementById(id);
const state = { mapId: null, items: [], selected: new Set(), aoi: null, focused: null, searchId: null, hasMore: false, missions: [], reports: [], groundTruth: [], tuningRuns: [], rasterAccess: [], localRasterRoot: "", currentMissionId: null, analysisTargets: new Set(), algorithms: new Set(), expandedJobs: new Set(), rasterTargets: new Set(), missionDialogWithScenes: true, missionTab: "inference" };
const catalogs = {
  local: {label:'Local imagery', source:'Saved local science rasters', url:'#'},
  "sentinel-2-l1c": { label: "Sentinel-2 L1C", source: "Copernicus Data Space · TOA", url: "https://dataspace.copernicus.eu/" },
  "sentinel-2": { label: "Sentinel-2 L2A", source: "Copernicus Data Space", url: "https://dataspace.copernicus.eu/" },
  landsat: { label: "Landsat Collection 2 L2", source: "USGS LandsatLook", url: "https://landsatlook.usgs.gov/" },
  "hls-l30": { label: "HLS L30", source: "NASA · Earthdata Login for files", url: "https://hls.gsfc.nasa.gov/data-access-and-tools/" },
  "hls-s30": { label: "HLS S30", source: "NASA · Earthdata Login for files", url: "https://hls.gsfc.nasa.gov/data-access-and-tools/" },
  "cbers-4a-wpm": { label: "CBERS-4A WPM Level 4", source: "INPE Brazil Data Cube", url: "https://data.inpe.br/dados/cbers-4a/" },
  "eo1-hyperion": { label: "EO-1 Hyperion L1T", source: "USGS LandsatLook", url: "https://landsatlook.usgs.gov/" },
};
const algorithmCatalog = [
  {id:"dynamicworld",name:"Dynamic World · pretrained LULC",detail:"Ready nine-class pixel model · Sentinel-2 L1C · nine bands at 10m · no training needed",tag:"PRETRAINED",color:"mint",mark:"DW"},
  {id:"prithvi_crop",name:"Prithvi crop/LULC · pretrained head",detail:"Ready 13-class pixel head · three genuine HLS dates · 18 bands at 30m",tag:"PRETRAINED",color:"blue",mark:"PC"},
  {id:"openearthmap",name:"OpenEarthMap · pretrained U-Net",detail:"Ready eight-class pixel head · RGB at 0.25–0.5m · unsuitable for Sentinel pixels",tag:"PRETRAINED",color:"violet",mark:"OEM"},
  {id:"random_forest",name:"Random Forest",detail:"Reliable spectral-band baseline for limited labels",tag:"SPECTRAL",color:"mint",mark:"RF"},
  {id:"svm",name:"Support Vector Machine",detail:"High-dimensional bands and smaller training sets",tag:"SPECTRAL",color:"blue",mark:"SVM"},
  {id:"xgboost",name:"XGBoost",detail:"Non-linear band and index relationships",tag:"SPECTRAL",color:"amber",mark:"XGB"},
  {id:"unet",name:"U-Net segmentation",detail:"Spatial context for boundaries and heterogeneous cover",tag:"SPATIAL",color:"violet",mark:"U"},
  {id:"prithvi",name:"Prithvi-EO-2.0 foundation model",detail:"Pretrained reflectance features + supervised classification head",tag:"FOUNDATION",color:"blue",mark:"EO"},
  {id:"ensemble",name:"LULC Ensemble",detail:"Runs compatible members and combines class probabilities; at least two members required",tag:"ENSEMBLE",color:"mint",mark:"Σ"},
];

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || data.message || `Request failed (${response.status})`);
  return data;
}
function localDate(date) { return `${date.getFullYear()}-${String(date.getMonth()+1).padStart(2,"0")}-${String(date.getDate()).padStart(2,"0")}`; }
function escapeHtml(value) { return String(value ?? "").replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch])); }
function toast(message) { const node = $("toast"); node.textContent = message; node.classList.add("visible"); clearTimeout(toast.timer); toast.timer = setTimeout(() => node.classList.remove("visible"), 3300); }
function setMessage(message, error = false) { const node = $("searchMessage"); node.textContent = message; node.classList.toggle("error", error); }
function setAoiStatus(message, active = false) { const node = $("aoiStatus"); node.lastElementChild.textContent = message; node.classList.toggle("active", active); }
function fmtDate(value) { if (!value) return "—"; const d = new Date(value); return Number.isNaN(d.valueOf()) ? value.slice(0,10) : d.toLocaleDateString(undefined,{year:"numeric",month:"short",day:"2-digit",timeZone:"UTC"}); }
function geomBounds(geometry) {
  const points = [];
  const visit = (value) => { if (!Array.isArray(value)) return; if (value.length >= 2 && typeof value[0] === "number" && typeof value[1] === "number") points.push(value); else value.forEach(visit); };
  visit(geometry?.coordinates);
  if (!points.length) return null;
  const xs = points.map(p=>p[0]), ys = points.map(p=>p[1]);
  let minX=Math.min(...xs), maxX=Math.max(...xs), minY=Math.min(...ys), maxY=Math.max(...ys);
  if (maxX-minX < .003) { minX-=.01; maxX+=.01; }
  if (maxY-minY < .003) { minY-=.01; maxY+=.01; }
  return [minX,minY,maxX,maxY];
}

function setupDates() {
  const today = new Date(); const from = new Date(today); from.setDate(today.getDate()-30);
  $("dateFrom").value = localDate(from); $("dateTo").value = localDate(today);
}
let connectivityBusy=false,lastMapNetwork=null;
async function checkConnectivity(){
  if(connectivityBusy)return;
  connectivityBusy=true;
  try{await applyConnectivity(await api('/api/connectivity'));}catch(error){const badge=$('connectivityStatus');if(badge){badge.textContent='Connection check unavailable';badge.title=error.message;}}
  finally{connectivityBusy=false;}
}
function startConnectivityPolling(){
  setInterval(checkConnectivity,30000);
  window.addEventListener('online',checkConnectivity);
  window.addEventListener('offline',checkConnectivity);
}
const catalogFields=['items','selected','focused','searchId','hasMore','localCatalogFiltered'];
const newCatalogTab=source=>({source,items:[],selected:new Set(),focused:null,searchId:null,hasMore:false,localCatalogFiltered:false,revision:0});
state.catalogTabs={online:newCatalogTab('sentinel-2'),local:newCatalogTab('local')};
state.catalogTab='online';
function saveCatalogTab(){
  const tab=state.catalogTabs[state.catalogTab];
  for(const field of catalogFields)tab[field]=state[field];
}
function restoreCatalogTab(){
  const tab=state.catalogTabs[state.catalogTab];
  for(const field of catalogFields)state[field]=tab[field];
  $('catalog').value=tab.source;
}
function selectedCatalogImages(){
  saveCatalogTab();
  return Object.values(state.catalogTabs).flatMap(tab=>tab.items.filter(item=>tab.selected.has(item.id)));
}
function updateCatalogTabs(){
  for(const kind of ['online','local']){
    const tab=state.catalogTabs[kind],button=$(kind+'CatalogTab');
    button.setAttribute('aria-selected',String(state.catalogTab===kind));button.tabIndex=state.catalogTab===kind?0:-1;
    $(kind+'CatalogCount').textContent=tab.items.length;
    button.classList.toggle('has-selection',tab.selected.size>0);
    button.title=`${tab.items.length} images · ${tab.selected.size} selected`;
  }
  $('catalogResultsPanel').setAttribute('aria-labelledby',state.catalogTab+'CatalogTab');
  $('catalogTabStatus').textContent=state.catalogTab==='local'
    ? (state.localCatalogFiltered?'Local search results · AOI/date filter applied. Browse all in Manage folders to remove it.':'Monitored and imported images · updates automatically. Online selections are kept.')
    : state.offline?'Offline · saved online results and selections are kept. Switch to Local imagery to work now.':'Search satellite catalogs · local monitoring continues in the other tab.';
  $('searchButton').disabled=Boolean(state.catalogSearching)||(state.catalogTab==='online'&&Boolean(state.offline));
  $('loadMoreButton').disabled=Boolean(state.catalogTabs[state.catalogTab].loading)||(state.catalogTab==='online'&&Boolean(state.offline));
}
function selectCatalogTab(kind,source=null){
  saveCatalogTab();
  const tab=state.catalogTabs[kind];
  if(source&&tab.source!==source){Object.assign(tab,newCatalogTab(source),{revision:tab.revision+1});}
  state.catalogTab=kind;restoreCatalogTab();
  const choice=catalogs[tab.source];$('sourceName').textContent=choice.source;$('sourceLink').href=choice.url;$('sourceLink').hidden=kind==='local';
  renderResults();
  $('resultTotal').textContent='· '+(kind==='local'?(state.localCatalogFiltered?'Local search':'Local library · updates automatically'):choice.label)+(state.hasMore?' · more available':'');
  if(kind==='local')window.LocalImagery?.refreshSummary();
}
function resetCatalogResults(){
  const tab=state.catalogTabs[state.catalogTab];tab.revision++;
  state.items=[];state.selected=new Set();state.focused=null;state.searchId=null;state.hasMore=false;state.localCatalogFiltered=false;
  renderResults();
}
function updateLocalCatalogItems(images,force=false){
  if(state.catalogTab==='local')saveCatalogTab();
  const tab=state.catalogTabs.local;
  if(!force&&tab.localCatalogFiltered)return;
  tab.items=images;tab.selected=new Set([...tab.selected].filter(id=>images.some(item=>item.id===id)));tab.hasMore=false;tab.searchId=null;
  if(force)tab.localCatalogFiltered=false;
  if(state.catalogTab==='local'){restoreCatalogTab();renderResults();$('resultTotal').textContent='· Local library · updates automatically';}
  else{updateCatalogTabs();updateSelectionCount();}
}
function clearCatalogSelections(){
  saveCatalogTab();for(const tab of Object.values(state.catalogTabs))tab.selected.clear();
  renderResults();
}
async function applyConnectivity(network,initial=false){
  const wasOffline=state.offline;
  state.offline=!network.online;state.network=network;
  const badge=$('connectivityStatus');
  if(badge){badge.innerHTML='<b></b> '+(network.online?(Object.keys(network.errors||{}).length?'ONLINE · SOME SERVICES UNAVAILABLE':'ONLINE · MAPS & CATALOGS'):network.mode==='offline'?'OFFLINE ONLY':'OFFLINE · LOCAL DATA');badge.title=network.online?'Online catalog search is available. '+Object.entries(network.errors||{}).map(([service,error])=>service+': '+error).join(' · '):network.reason||'Online services could not be reached. Local imagery and bundled maps remain available.';}
  for(const option of $('catalog').options)option.disabled=option.value!=='local'&&!network.online;
  const local=$('catalog').value==='local',choice=catalogs[$('catalog').value];
  $('sourceName').textContent=choice.source;$('sourceLink').hidden=local;$('sourceLink').href=choice.url;
  $('offlineCatalogHint').hidden=network.online;
  $('offlineCatalogHint').textContent='Offline · online services could not be reached. Local imagery and bundled world maps remain available. Connection checks resume automatically.';
  document.querySelector('.intro').textContent=network.online?'Search online satellite catalogs or use your saved local imagery.':'Use saved imagery, draw an area of interest, and build missions. Online services reconnect automatically.';
  for(const option of $('rasterAccessMode').options)option.disabled=!network.online&&option.value!=='local_path';
  if(!network.online)$('rasterAccessMode').value='local_path';
  if(initial||wasOffline!==state.offline)await window.SpectraScopeBasemaps?.refresh();
  const current=$('basemap').value,custom=$('basemap').querySelector('optgroup[data-custom]');
  const options=[];
  if(network.online){
    for(const [label,key] of [['Satellite / aerial imagery · Esri','satellite'],['Street map · OpenStreetMap','osm'],['Dark street map · Carto','carto_dark']]){
      const option=new Option(label+(network.maps?.[key]?'':' · unavailable'),key);option.disabled=!network.maps?.[key];option.title=network.errors?.[key]||'';options.push(option);
    }
  }
  options.push(new Option('World map · Dark (local)','offline'),new Option('World map · Light (local)','offline_light'),new Option('Your orthophoto · import local GeoTIFF…','import-orthophoto'));
  $('basemap').replaceChildren(...options);if(custom)$('basemap').append(custom);
  let desired=current;
  const available=value=>[...$('basemap').options].some(o=>o.value===value&&!o.disabled)&&value!=='import-orthophoto';
  if(initial||!available(desired)||(wasOffline&&network.online&&(!state.basemapUserChoice||state.preferredBasemap&&!state.preferredBasemap.startsWith('offline'))))desired=available(state.preferredBasemap)?state.preferredBasemap:network.maps?.satellite?'satellite':network.maps?.osm?'osm':'offline';
  $('basemap').value=available(desired)?desired:'offline';
  const signature=JSON.stringify({online:network.online,maps:network.maps||{}});
  if((initial||signature!==lastMapNetwork)&&state.mapId){
    if(state.mapDrawing){state.networkMapDeferred=true;return;}
    try{
      if($('basemap').value.startsWith('custom-'))await api('/api/map-providers/'+encodeURIComponent($('basemap').value)+'/activate',{method:'POST'});
      else await api('/api/basemap',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:$('basemap').value,automatic:true})});
      const url=new URL(state.mapUrl||$('mapFrame').src,location.href);url.searchParams.set('connection_revision',Date.now());$('mapFrame').src=url.href;
      lastMapNetwork=signature;state.networkMapDeferred=false;
    }catch(error){toast(error.message);}
  }
  updateCatalogTabs();
  if(initial){renderResults();await window.LocalImagery?.refreshSummary();}
}
async function init() {
  try{state.algorithmSuitability=await api('/static/algorithm-suitability.json?v=0.2.21');}catch(error){state.algorithmSuitability={};}
  bind();
  setupDates();
  await loadMissions();
  const link=new URLSearchParams(window.location.search),mission=link.get('mission');
  if(mission&&state.missions.some(item=>item.id===mission)){
    const tab=link.get('tab');
    toggleMissionWorkspace();
    openMission(mission,{tab:['automatic','qualification','inference','plan','rasters','reports','viewer','training','anomalies','insights'].includes(tab)?tab:'reports',jobId:link.get('job')});
    if(window.location.hash.startsWith('#report-'))document.getElementById(window.location.hash.slice(1))?.scrollIntoView();
  }
  try {
    const config = await api("/api/config");
    $('appVersion').textContent='v'+config.app_version;
    state.offline = config.offline;
    state.mapId = config.map_id;
    state.mapUrl = config.map_url;
    state.preferredBasemap=config.basemap||null;
    $('mapTheme').value=config.map_theme||'dark';
    selectCatalogTab(config.offline?'local':'online');
    await applyConnectivity(config.network||{mode:config.offline?'offline':'connected',online:!config.offline,maps:{osm:!config.offline,satellite:!config.offline}},true);
    await window.LocalImagery?.refreshSummary();
    state.mapId = config.map_id;
    if(config.memory_profile==='low') {
      $("analysisGrid").value='192';
      $("analysisGrid").title='16 GB profile: start with 192 px; larger grids and model training require additional memory.';
    }
    if(!$("mapFrame").getAttribute("src"))$("mapFrame").src = config.map_url;
    $("mapFrame").addEventListener("load", () => $("mapLoading").classList.add("hidden"));
    setTimeout(() => $("mapLoading").classList.add("hidden"), 14000);
  } catch (error) {
    $("mapLoading").innerHTML = `<strong>Map service unavailable</strong><span>${escapeHtml(error.message)}</span>`;
    setMessage("Could not connect to MC². Start the Docker services and refresh.", true);
  }
  startConnectivityPolling();
  startRunPolling();
  renderInferenceActivity();

}

function bind() {
  window.addEventListener('message',event=>{
    const frame=$('mapFrame');
    if(event.source!==frame.contentWindow||event.origin!==new URL(frame.src,location.href).origin||event.data?.source!=='mapcontrol')return;
    if(event.data.type==='drawing_status'){state.mapDrawing=Boolean(event.data.data?.active);if(!state.mapDrawing&&state.networkMapDeferred)applyConnectivity(state.network).catch(error=>toast(error.message));}
    if(event.data.type==='basemapchanged'&&[...$('basemap').options].some(o=>o.value===event.data.data?.basemap))$('basemap').value=event.data.data.basemap;
    if(event.data.type==='themechanged'&&['light','dark','auto'].includes(event.data.data?.theme))$('mapTheme').value=event.data.data.theme;
  });
  $('importLocalImagery').onclick = () => $('localImageryFiles').click();
  $('importImageryFolder').onclick = () => $('localImageryFolder').click();
  $('localLibraryButton').onclick = () => window.LocalImagery.open();
  const importProducts = async event => {
    const button = $('importLocalImagery'); button.disabled=true;
    try {
      const files=[...event.target.files].filter(f=>!f.name.startsWith('.')&&(/\.(tif|tiff|jp2)$/i.test(f.name)||/^MTD_MSIL(?:1C|2A)\.xml$/i.test(f.name)||/_MTL\.(txt|xml|json)$/i.test(f.name)||f.name==='spectrascope-image.json'));
      if(!files.length) throw Error('Select GeoTIFF/JP2 bands or a folder containing them.');
      button.textContent=`Importing ${files.length} files as image products…`;
      const form=new FormData();
      for(const file of files) form.append('files',file);
      form.append('relative_paths',JSON.stringify(files.map(f=>f.webkitRelativePath||f.name)));
      const imported=await api('/api/local-imagery/batch',{method:'POST',body:form});
      const ready=imported.items.filter(i=>i.status==='ready');
      selectCatalogTab('local');resetCatalogResults();
      state.items=ready;state.selected=new Set(ready.map(item=>item.id));state.hasMore=false;state.searchId=null;
      $('catalog').value='local';$('sourceName').textContent='Saved local science rasters';$('sourceLink').hidden=true;
      renderResults();$('resultTotal').textContent='· Local imagery';
      const bbox=ready[0]?.bbox;
      if(bbox){state.aoi={type:'Polygon',coordinates:[[[bbox[0],bbox[1]],[bbox[2],bbox[1]],[bbox[2],bbox[3]],[bbox[0],bbox[3]],[bbox[0],bbox[1]]]]};await api('/api/zoom',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({bbox})});}
      const review=imported.items.length-ready.length;
      toast(`${imported.items.length} image product(s), ${files.length} source files. ${review?`${review} need band review.`:'All bands are connected to each image.'}`);
      if(review||imported.issues?.length)window.LocalImagery.open(imported.issues);
    } catch(error) {toast(error.message);} finally{button.disabled=false;button.textContent='+ Import local imagery';event.target.value='';}
  };
  $('localImageryFiles').onchange=importProducts;
  $('localImageryFolder').onchange=importProducts;
  $('catalog').addEventListener('change',()=>selectCatalogTab($('catalog').value==='local'?'local':'online',$('catalog').value));
  for(const kind of ['online','local']){
    $(kind+'CatalogTab').onclick=()=>selectCatalogTab(kind);
    $(kind+'CatalogTab').onkeydown=event=>{
      if(['ArrowLeft','ArrowRight','Home','End'].includes(event.key)){
        event.preventDefault();const next=event.key==='Home'?'online':event.key==='End'?'local':kind==='online'?'local':'online';selectCatalogTab(next);$(next+'CatalogTab').focus();
      }
    };
  }
  $("cloudMax").addEventListener("input", e => {
    const value = Number(e.target.value); $("cloudValue").textContent = `${value}%`;
    e.target.style.background = `linear-gradient(90deg,#5ec4a1 ${value}%,#344244 ${value}%)`;
  });
  $("searchButton").addEventListener("click", doSearch);
  document.addEventListener("keydown", e => { if (e.key === "Enter" && document.activeElement.closest('.sidebar') && !["TEXTAREA","BUTTON"].includes(document.activeElement.tagName)) doSearch(); });
  $("setPoint").addEventListener("click", setPoint);
  $("refreshAoi").addEventListener("click", refreshAoi);
  $("selectAll").addEventListener("click", () => selectAll(true));
  $("clearSelection").addEventListener("click", clearCatalogSelections);
  $("loadMoreButton").addEventListener("click", loadMore);
  $("filtersTabButton").addEventListener("click", () => setWorkspaceView("filters"));
  const missionActionsToggle = $("missionActionsToggle");
  const closeMissionActions = () => missionActionsToggle.setAttribute("aria-expanded", "false");
  missionActionsToggle.addEventListener("click", () => {
    missionActionsToggle.setAttribute("aria-expanded", String(missionActionsToggle.getAttribute("aria-expanded") !== "true"));
  });
  document.addEventListener("click", event => {
    if (!event.target.closest("#missionActionsToggle, #missionActions") || event.target.closest("#missionActions button")) closeMissionActions();
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && missionActionsToggle.getAttribute("aria-expanded") === "true") {
      closeMissionActions();
      missionActionsToggle.focus();
    }
  });
  $("mapTabButton").addEventListener("click", () => setWorkspaceView("map"));
  $("resultsTabButton").addEventListener("click", () => setWorkspaceView("results"));
  $("continueButton").addEventListener("click", continueWithSelection);
  $("buildHLSBundle").addEventListener("click", buildHLSBundle);
  $("missionsButton").addEventListener("click", toggleMissionWorkspace);
  $("newMissionButton").addEventListener("click", () => openMissionDialog(false));
  $("emptyNewMissionButton").addEventListener("click", () => openMissionDialog(false));
  $("missionMode").addEventListener("change", updateMissionDialog);
  $("missionForm").addEventListener("submit", submitMissionDialog);
  $("missionDialogClose").addEventListener("click", () => $("missionDialog").close());
  $("backToMissions").addEventListener("click", showMissionHub);
  for (const [id,command] of [['pauseMissionButton','pause'],['resumeMissionButton','resume'],['stopMissionButton','stop'],['deleteMissionButton','delete']]) $(id).addEventListener('click',()=>controlCurrentMission(command));
  $("renameMissionButton").addEventListener("click", renameCurrentMission);
  $("addScenesButton").addEventListener("click", () => openMissionDialog(true, state.currentMissionId));
  $("savePlanButton").addEventListener("click", saveAnalysisPlan);
  $('missionNextAction').addEventListener('click',()=>{const target=$('missionNextAction').dataset.target||'plan';if(target==='catalog')toggleMissionWorkspace();else setMissionTab(target);});
  $('missionViewerReports').addEventListener('click',()=>setMissionTab('reports'));
  $("runInferenceButton").addEventListener("click", runInference);
  document.querySelectorAll("[data-mission-tab]").forEach(button=>button.addEventListener("click",()=>setMissionTab(button.dataset.missionTab)));
  $("rasterAccessJob").addEventListener("change", renderRasterPanel);
  $("rasterAccessMode").addEventListener("change", renderRasterPanel);
  $("selectAllRasterScenes").addEventListener("click", selectAllRasterScenes);
  $("configureRasterAccessButton").addEventListener("click", configureRasterAccess);
  $("captureGroundTruthButton").addEventListener("click", captureGroundTruth);
  $("fineTuneButton").addEventListener("click", prepareFineTuning);
  $("alignBandsButton").addEventListener("click", alignBands);
  $("groundTruthJob").addEventListener("change", renderTrainingPanel);
  $("basemap").addEventListener("change", changeBasemap);
  $('mapTheme').onchange=async()=>{const control=$('mapTheme');control.disabled=true;try{await api('/api/map-theme',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({theme:control.value})});}catch(error){toast(error.message);}finally{control.disabled=false;}};
  $("uploadButton").addEventListener("click", () => $("fileInput").click());
  $("fileInput").addEventListener("change", uploadFile);
  $("uploadMode").addEventListener("change", updateUploadOptions);
}

async function refreshAoi() {
  try {
    const data = await api("/api/aoi"); state.aoi = data.geometry;
    const kind = data.geometry.type.replace("Multi", "");
    setAoiStatus(`${data.name || "Map AOI"} · ${kind}`, true);
    toast("AOI read from the map. Set dates and search.");
    return data.geometry;
  } catch (error) { state.aoi = null; setAoiStatus("Draw a polygon or rectangle, or enter a point", false); toast(error.message); return null; }
}

async function setPoint() {
  const lon = Number($("longitude").value), lat = Number($("latitude").value);
  if (!Number.isFinite(lon) || !Number.isFinite(lat)) { toast("Enter longitude and latitude for the point AOI."); return; }
  $("setPoint").disabled = true;
  try {
    const data = await api("/api/point", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({longitude:lon, latitude:lat}) });
    state.aoi = data.geometry; setAoiStatus(`Point AOI · ${lon.toFixed(4)}, ${lat.toFixed(4)}`, true);
    toast("Point AOI placed on the map.");
  } catch (error) { toast(error.message); }
  finally { $("setPoint").disabled = false; }
}

async function doSearch() {
  if(state.catalogSearching)return;
  if(state.catalogTab==='online'&&state.offline){setMessage('Online search is unavailable. Your saved results are kept; Local imagery is available in its own tab.',true);return;}
  if (!state.mapId) { setMessage('Map service is not available.',true);return; }
  const start=$('dateFrom').value,end=$('dateTo').value;
  if(!start||!end||start>end){setMessage('Choose a valid date range.',true);return;}
  const geometry=await refreshAoi();if(!geometry)return;
  saveCatalogTab();
  const kind=state.catalogTab,tab=state.catalogTabs[kind],source=tab.source,revision=tab.revision;
  state.catalogSearching=true;updateCatalogTabs();$('searchButton').querySelector('span:nth-child(2)').textContent='Searching…';
  setMessage(kind==='local'?'Searching indexed local imagery…':'Querying the online catalog and updating scene footprints…');
  try{
    const result=await api('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({catalog:source,geometry,start,end,cloud_max:Number($('cloudMax').value),limit:50})});
    if(tab.source!==source||tab.revision!==revision)return;
    Object.assign(tab,{items:result.items,selected:new Set(),focused:null,searchId:result.search_id,hasMore:result.has_more,localCatalogFiltered:kind==='local'});
    if(state.catalogTab===kind){
      restoreCatalogTab();renderResults();$('resultTotal').textContent='· '+catalogs[source].label+(result.has_more?' · more available':'');setWorkspaceView('results');
    }else{updateCatalogTabs();updateSelectionCount();}
    setMessage(`${result.count} ${kind==='local'?'local':'online'} scenes found${result.has_more?'; more available.':'.'}`);
    if(!result.count)toast('No scenes matched this AOI, date range, and cloud limit.');
  }catch(error){setMessage(error.message,true);}
  finally{state.catalogSearching=false;updateCatalogTabs();$('searchButton').querySelector('span:nth-child(2)').textContent='Search catalog';}
}

function renderResults() {
  saveCatalogTab();updateCatalogTabs();
  $("resultCount").textContent = state.items.length;$("resultNoun").textContent=state.items.length===1?"image":"images";
  $("mobileResultCount").textContent = state.items.length;
  updateSelectionCount();
  $("loadMoreButton").classList.toggle("hidden", !state.hasMore);
  const list = $("sceneList");
  if (!state.items.length) { list.innerHTML = `<div class="empty-scenes">${$("catalog").value==='local'?(state.localCatalogFiltered?'No local images matched this AOI and date range. Use Browse all ready images in Manage folders to remove the filter.':'No ready local images yet. Open Monitor folders to check scan progress, paths and band-review issues.'):state.searchId?'No scenes matched this search. Try a longer date range or allow more cloud cover.':'Choose an area and date range, then search this online catalog.'}</div>`; return; }
  const rows = state.items.map((item,index) => {
    const id = escapeHtml(item.id); const checked = state.selected.has(item.id) ? "checked" : "";
    const date = fmtDate(item.datetime); const cloud = item.cloud_cover == null ? "—" : `${Math.round(item.cloud_cover)}%`;
    const platform = item.catalog==='local' ? escapeHtml(item.sensor||'Local imagery')+(item.raster_bands?` · ${item.raster_bands} bands`:'') : item.platform ? escapeHtml(item.platform.replaceAll("_"," ")) : (item.catalog === 'local' ? escapeHtml(item.sensor||'Local imagery') : item.catalog === "landsat" ? "Landsat" : item.catalog === "eo1-hyperion" ? "EO-1 Hyperion" : "Sentinel-2");
    const thumb = item.thumbnail&&!state.offline ? `<img class="thumb" src="${escapeHtml(item.thumbnail)}" loading="lazy" alt="" onerror="this.replaceWith(Object.assign(document.createElement('span'),{className:'thumb-placeholder',textContent:'▧'}))">` : `<span class="thumb-placeholder">▧</span>`;
    return `<tr class="scene-row ${state.focused===item.id?"focused":""}" data-index="${index}"><td><input class="scene-check" type="checkbox" data-index="${index}" ${checked} aria-label="Select ${id}"></td><td><div class="scene-name-cell">${thumb}<div><span class="scene-id" title="${id}">${escapeHtml(item.name||item.id)}</span><span class="scene-sub">${platform}</span></div></div></td><td>${date}</td><td class="scene-cloud">${cloud}</td><td>${item.product_url?`<a class="scene-link" href="${escapeHtml(item.product_url)}" target="_blank" rel="noreferrer" title="Open catalog record">↗</a>`:""}</td></tr>`;
  }).join("");
  list.innerHTML = `<table class="scene-table"><thead><tr><th></th><th>SCENE / SENSOR</th><th>DATE</th><th>CLOUD</th><th></th></tr></thead><tbody>${rows}</tbody></table>`;
  list.querySelectorAll(".scene-check").forEach(input => input.addEventListener("change", e => {
    const item = state.items[Number(e.target.dataset.index)];
    if (e.target.checked) state.selected.add(item.id); else state.selected.delete(item.id);
    updateSelectionCount();updateCatalogTabs();
    if(e.target.checked)focusItem(item);
  }));
  list.querySelectorAll(".scene-row").forEach(row => row.addEventListener("click", e => {
    if (e.target.closest("input, a")) return;
    const item = state.items[Number(row.dataset.index)]; focusItem(item);
  }));
}
function setWorkspaceView(view) {
  const showResults = view === "results";
  const showFilters = view === "filters";
  $("layout").classList.toggle("show-filters", showFilters);
  $("filtersTabButton").classList.toggle("active", showFilters);
  $("filtersTabButton").setAttribute("aria-selected", String(showFilters));
  const compact = window.matchMedia("(max-width: 1100px)").matches;
  const resultsPanel = $("resultsBlock");
  $("mapWorkspace").classList.toggle("show-results", compact && showResults);
  resultsPanel.classList.toggle("results-overlay", compact && showResults);
  $("mapTabButton").classList.toggle("active", !showResults && !showFilters);
  $("resultsTabButton").classList.toggle("active", showResults);
  $("mapTabButton").setAttribute("aria-selected", String(!showResults && !showFilters));
  $("resultsTabButton").setAttribute("aria-selected", String(showResults));
  if (!compact || !showResults) resizeEmbeddedMap();
}
function resizeEmbeddedMap() {
  const frame = $("mapFrame");
  try { frame.contentWindow.postMessage({type:"spectrascope:resize-map"}, new URL(frame.src).origin); }
  catch (error) { /* The map iframe may still be loading. */ }
  setTimeout(() => {
    try { frame.contentWindow.postMessage({type:"spectrascope:resize-map"}, new URL(frame.src).origin); }
    catch (error) { /* The map iframe may still be loading. */ }
  }, 120);
}
async function loadMore() {
  saveCatalogTab();const kind=state.catalogTab,tab=state.catalogTabs[kind],searchId=tab.searchId,revision=tab.revision;
  if(!searchId||!tab.hasMore||tab.loading||(kind==='online'&&state.offline))return;
  tab.loading=true;updateCatalogTabs();
  try{
    const result=await api('/api/search/more',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({search_id:searchId})});
    if(searchId!==tab.searchId||revision!==tab.revision)return;
    tab.items.push(...result.items);tab.hasMore=result.has_more;
    if(state.catalogTab===kind){restoreCatalogTab();renderResults();}else updateCatalogTabs();
  }catch(error){toast(error.message);}finally{tab.loading=false;updateCatalogTabs();}
}

function updateSelectionCount() {
  saveCatalogTab();const online=state.catalogTabs.online.selected.size,local=state.catalogTabs.local.selected.size;
  $('selectedCount').textContent=online+local;$('selectionSources').textContent=`${online} online · ${local} local`;$('continueButton').disabled=online+local===0;
  const isLocal=state.catalogTab==='local', readyHls=selectedLocalHLS().length;
  const selectedOnline=state.catalogTabs.online.items.filter(item=>state.catalogTabs.online.selected.has(item.id));
  const onlineHls=selectedOnline.length===3&&selectedOnline.every(item=>['hls-l30','hls-s30'].includes(item.catalog));
  const eligible=isLocal&&local===3&&readyHls===3, show=isLocal||onlineHls;
  const bundleButton=$('buildHLSBundle');
  bundleButton.classList.toggle('hidden',!show);
  bundleButton.disabled=!eligible||bundleButton.hasAttribute('aria-busy');
  const hint=$('hlsBundleHint');
  hint.classList.toggle('hidden',!show||eligible);
  hint.textContent=!isLocal?'Open Download online imagery and enter a NASA Earthdata token to save the three products and build the bundle.'
    :`Select exactly three ready HLS L30 or S30 products from different dates. ${readyHls} ready HLS selected.`;
}
function selectedLocalHLS(){
  const tab=state.catalogTabs.local;
  return tab.items.filter(image=>tab.selected.has(image.id)&&image.status==='ready'&&['hls-l30','hls-s30'].includes(image.sensor));
}
async function buildHLSBundle(){
  const selected=selectedLocalHLS();
  const button=$('buildHLSBundle');button.disabled=true;button.setAttribute('aria-busy','true');
  try{
    const result=await api('/api/local-imagery/hls-bundles',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_ids:selected.map(image=>image.id)})});
    const tab=state.catalogTabs.local;
    tab.items=[result.image,...tab.items.filter(item=>item.id!==result.image.id)];
    tab.selected=new Set([result.image.id]);
    if(state.catalogTab==='local'){restoreCatalogTab();renderResults();}
    else updateSelectionCount();
    toast('HLS bundle created. Add it to a mission for Prithvi Crop/LULC.');
  }catch(error){toast(error.message);}finally{button.removeAttribute('aria-busy');updateSelectionCount();}
}
function selectAll(select) { state.selected = select ? new Set(state.items.map(item=>item.id)) : new Set(); renderResults(); }
async function focusItem(item) {
  const bbox = item.bbox || geomBounds(item.geometry);
  if (!bbox) { toast("This catalog item has no mapped footprint."); return; }
  state.focused = item.id; renderResults();
  setWorkspaceView("map");
  try {
    const draw=()=>api('/api/map/footprint',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:item.id,geometry:item.geometry||null,bbox})});
    focusItem.pending=(focusItem.pending||Promise.resolve()).catch(()=>{}).then(draw);
    await focusItem.pending;
  }
  catch (error) { toast(`Could not focus map: ${error.message}`); }
}
function continueWithSelection() {
  if (!selectedCatalogImages().length) return;
  openMissionDialog(true);
}
async function loadMissions() {
  let legacy = [];
  try { legacy = JSON.parse(localStorage.getItem("spectrascope.missions.v1") || "[]"); }
  catch { legacy = []; }
  if (!Array.isArray(legacy)) legacy = [];
  try {
    const workspace = await api("/api/workspace");
    state.missions = workspace.workspace_exists ? (workspace.missions||[]) : legacy;
    state.reports = Array.isArray(workspace.reports) ? workspace.reports : [];
    state.groundTruth = Array.isArray(workspace.ground_truth) ? workspace.ground_truth : [];
    state.tuningRuns = Array.isArray(workspace.tuning_runs) ? workspace.tuning_runs : [];
    state.rasterAccess = Array.isArray(workspace.raster_access) ? workspace.raster_access : [];
    state.localRasterRoot = workspace.local_raster_root || "";
    if (!workspace.workspace_exists && legacy.length) await persistMissions();
  } catch (error) {
    state.missions = legacy;
    toast("Using this browser's saved missions while the local database is unavailable.");
  }
  updateMissionCount();
}
function persistMissions() {
  localStorage.setItem("spectrascope.missions.v1", JSON.stringify(state.missions));
  const snapshot=JSON.stringify({missions:state.missions});
  const task=(persistMissions.pending||Promise.resolve()).catch(()=>{}).then(()=>api("/api/workspace", {method:"PUT",headers:{"Content-Type":"application/json"},body:snapshot})).then(async result=>{if(state.missions.some(m=>m.jobs?.some(j=>j.images?.some(image=>image.catalog==='local')))){const workspace=await api("/api/workspace");state.rasterAccess=workspace.raster_access||[];}return result;});
  persistMissions.pending=task;
  updateMissionCount();
  return task;
}
function updateMissionCount() {
  $("missionCountBadge").textContent = state.missions.length;
}
function toggleMissionWorkspace() {
  const showing = !$("missionWorkspace").classList.contains("hidden");
  if (showing) {
    $("missionWorkspace").classList.add("hidden"); $("layout").classList.remove("hidden");
    $("missionsButton").innerHTML = `Mission workspace <span id="missionCountBadge">${state.missions.length}</span>`;
    resizeEmbeddedMap();
  } else {
    $("layout").classList.add("hidden"); $("missionWorkspace").classList.remove("hidden");
    $("missionsButton").innerHTML = `← Search imagery <span id="missionCountBadge">${state.missions.length}</span>`;
    showMissionHub();
  }
}
function openMissionDialog(withScenes = true, missionId = null) {
  state.missionDialogWithScenes = withScenes;
  const dialog = $("missionDialog");
  $("missionDialogTitle").textContent = withScenes ? "Add selected scenes" : "Create a mission";
  $("missionDialogLede").textContent = withScenes
    ? "Create a mission or add these scenes as a new job in an existing mission."
    : "Start a mission now. You can add imagery and jobs from the mission workspace.";
  $("dialogSceneCount").textContent = withScenes ? `${selectedCatalogImages().length} selected scenes from both catalogs will be added as one job.` : "";
  $("dialogSceneCount").classList.toggle("hidden", !withScenes);
  $("missionMode").value = missionId ? "existing" : "new";
  $("missionMode").querySelector('option[value="existing"]').hidden = state.missions.length === 0;
  $("existingMissionSelect").innerHTML = state.missions.map(m=>`<option value="${escapeHtml(m.id)}">${escapeHtml(m.name)}</option>`).join("");
  if (missionId) $("existingMissionSelect").value = missionId;
  const number = state.missions.length + 1;
  $("newMissionName").value = `Mission ${String(number).padStart(2,"0")}`;
  $("newJobName").value = `Analysis job ${number}`;
  $("missionMode").closest("label").classList.toggle("hidden", !withScenes);
  $("jobNameWrap").classList.toggle("hidden", !withScenes);
  updateMissionDialog();
  dialog.showModal();
}
function updateMissionDialog() {
  const existing = state.missionDialogWithScenes && $("missionMode").value === "existing";
  $("newMissionNameWrap").classList.toggle("hidden", existing);
  $("existingMissionWrap").classList.toggle("hidden", !existing);
  $("createJobButton").textContent = !state.missionDialogWithScenes ? "Create mission" : existing ? "Add job to mission" : "Create mission & job";
}
async function submitMissionDialog(event) {
  event.preventDefault();
  const submit=$('createJobButton');
  if(submit.disabled)return;
  const previousMissions=structuredClone(state.missions);
  const withScenes = state.missionDialogWithScenes;
  let mission = $("missionMode").value === "existing" && withScenes
    ? state.missions.find(m=>m.id === $("existingMissionSelect").value)
    : null;
  if (!mission) {
    const name = $("newMissionName").value.trim();
    if (!name) { toast("Enter a mission name."); return; }
    mission = {id:crypto.randomUUID(),name,createdAt:new Date().toISOString(),updatedAt:new Date().toISOString(),jobs:[]};
    state.missions.unshift(mission);
  }
  let newJobId=null;
  if (withScenes) {
    const images = selectedCatalogImages().map(item=>({...item,planAlgorithms:[]}));
    if (!images.length) { state.missions=previousMissions;toast("Select at least one catalog scene first."); return; }
    newJobId=crypto.randomUUID();
    mission.jobs.push({id:newJobId,name:$("newJobName").value.trim() || `Analysis job ${mission.jobs.length+1}`,createdAt:new Date().toISOString(),status:"draft",images,aoi:state.aoi,analysisPlan:null});
  }
  mission.updatedAt = new Date().toISOString();
  submit.disabled=true;
  submit.textContent=withScenes?'Preserving every selected band…':'Saving mission…';
  try { await persistMissions(); }
  catch(error){
    state.missions=previousMissions;
    localStorage.setItem('spectrascope.missions.v1',JSON.stringify(state.missions));
    updateMissionCount();toast(`Job was not saved: ${error.message}`);
    return;
  }
  finally {submit.disabled=false;updateMissionDialog();}
  if(withScenes)clearCatalogSelections();
  $("missionDialog").close();
  toggleMissionWorkspaceIfNeeded();
  openMission(mission.id,withScenes?{tab:'inference',jobId:newJobId}:{});
  toast(withScenes ? "Job added to the mission." : "Mission created.");
}
function toggleMissionWorkspaceIfNeeded() {
  if ($("missionWorkspace").classList.contains("hidden")) {
    $("layout").classList.add("hidden"); $("missionWorkspace").classList.remove("hidden");
    $("missionsButton").innerHTML = `← Search imagery <span id="missionCountBadge">${state.missions.length}</span>`;
  }
}
function showMissionHub() {
  state.currentMissionId = null;
  $("missionDetail").classList.add("hidden"); $("missionHub").classList.remove("hidden");
  renderMissionHub();
}
function renderMissionHub() {
  updateMissionCount();
  const grid = $("missionGrid");
  $("missionEmpty").classList.toggle("hidden", state.missions.length > 0);
  grid.classList.toggle("hidden", !state.missions.length);
  grid.innerHTML = state.missions.map(mission=>{
    const jobs = mission.jobs || [], images = jobs.reduce((sum,job)=>sum+(job.images||[]).length,0);
    const updated = mission.updatedAt ? fmtDate(mission.updatedAt) : "—";
    const active=state.reports.some(run=>run.mission_id===mission.id&&['queued','running','pausing','paused','stopping'].includes(run.status));
    const status=mission.control==='pause'?'PAUSED':mission.control==='stop'?'STOPPED':active?'RUNNING':state.reports.some(run=>run.mission_id===mission.id&&['complete','partial'].includes(run.status))?'RESULTS READY':jobs.some(j=>j.analysisPlan)?'PLANNED':'DRAFT';
    return `<article class="mission-card"><div class="mission-card-top"><span class="mission-card-icon">◎</span><span class="status-pill draft-pill">${status}</span></div><h2>${escapeHtml(mission.name)}</h2><p>Updated ${updated}</p><div class="mission-card-stats"><span><b>${jobs.length}</b> jobs</span><span><b>${images}</b> images</span></div><button class="text-button mission-delete-button" data-delete-mission="${escapeHtml(mission.id)}" aria-label="Delete ${escapeHtml(mission.name)}">Delete mission</button><button class="mission-open-button" data-open-mission="${escapeHtml(mission.id)}">Open mission <span>→</span></button></article>`;
  }).join("");
  grid.querySelectorAll("[data-delete-mission]").forEach(button=>button.addEventListener("click",()=>controlCurrentMission('delete',button.dataset.deleteMission)));
  grid.querySelectorAll("[data-open-mission]").forEach(button=>button.addEventListener("click",()=>openMission(button.dataset.openMission)));
}
function openMission(id,{tab='plan',jobId=null}={}) {
  const mission = state.missions.find(m=>m.id===id); if (!mission) return;
  state.currentMissionId = id; state.analysisTargets.clear(); state.algorithms.clear(); state.missionTab = tab;
  if(jobId && mission.jobs?.some(job=>job.id===jobId))window.MissionWorkspaces?.focusJob(id,jobId);
  state.expandedJobs = new Set((mission.jobs||[]).map(job=>job.id));
  $("missionHub").classList.add("hidden"); $("missionDetail").classList.remove("hidden");
  renderMissionDetail();
}
async function controlCurrentMission(command, missionId=state.currentMissionId) {
  const mission=state.missions.find(m=>m.id===missionId);if(!mission)return;
  if(command==='delete'&&!confirm(`Delete “${mission.name}” from the workspace and stop its active runs? Saved imagery, results, and corrections will be retained locally.`))return;
  if(command==='stop'&&!confirm('Stop all active runs in this mission? Completed artifacts are kept. Resuming the mission lets you start new runs; stopped runs do not restart.'))return;
  const ids=['pauseMissionButton','resumeMissionButton','stopMissionButton','deleteMissionButton'];
  ids.forEach(id=>$(id).disabled=true);
  try {
    await api(`/api/missions/${encodeURIComponent(mission.id)}/control/${command}`,{method:'POST'});
    mission.control=command;
    if(command==='delete') {
      state.missions=state.missions.filter(m=>m.id!==mission.id);
      localStorage.setItem('spectrascope.missions.v1',JSON.stringify(state.missions));
      showMissionHub();toast('Mission removed from the workspace. Saved data retained.');
    } else {
      renderMissionDetail();
      toast(command==='pause'?'Pause requested. The current operation will reach a safe checkpoint.':command==='stop'?'Stop requested. Completed outputs will be retained.':'Mission resumed. Stopped runs must be started again.');
    }
  }catch(error){toast(error.message);}
  finally{ids.forEach(id=>$(id).disabled=false);if(state.currentMissionId)renderMissionDetail();}
}
function renderMissionDetail() {
  const mission = state.missions.find(m=>m.id===state.currentMissionId); if (!mission) return;
  const jobs = mission.jobs || [], imageCount = jobs.reduce((sum,job)=>sum+(job.images||[]).length,0);
  $("detailMissionName").textContent = mission.name;
  $("detailMissionMeta").textContent = `Updated ${fmtDate(mission.updatedAt)} · ${jobs.length} job${jobs.length===1?"":"s"} · ${imageCount} image${imageCount===1?"":"s"}`;
  $("hierarchyCounts").textContent = `${jobs.length} jobs · ${imageCount} images`;
  const liveStatuses=jobs.map(job=>state.reports.find(run=>run.mission_id===mission.id&&run.job_id===job.id)?.status||job.status||'draft');
  $("missionStatus").textContent = liveStatuses.some(status=>['queued','running'].includes(status))?'RUNNING':liveStatuses.length&&liveStatuses.every(status=>status==='complete')?'COMPLETE':liveStatuses.some(status=>['partial','failed','needs_input'].includes(status))?'REVIEW NEEDED':jobs.some(job=>job.analysisPlan)?'PLANNED':'DRAFT';
  const control=mission.control||'resume', suspended=control!=='resume';
  $('pauseMissionButton').classList.toggle('hidden',suspended);
  $('resumeMissionButton').classList.toggle('hidden',!suspended);
  $('stopMissionButton').disabled=control==='stop';
  if(suspended)$('missionStatus').textContent=control==='pause'?'PAUSED / PAUSING':'STOPPED / STOPPING';
  $('missionControlMessage').textContent=control==='pause'?'Pause requested: processing pauses at its next safe checkpoint. Resume continues that run.':control==='stop'?'Stop requested: processing stops at its next safe checkpoint. Completed results are kept. Resume enables new runs.':'';
  $("addScenesButton").disabled = selectedCatalogImages().length === 0;
  const reports = state.reports.filter(run=>run.mission_id===mission.id);
  $("reportCountBadge").textContent = reports.length;
  updateMissionGuide(mission);

  $("rasterAccessBadge").textContent = state.rasterAccess.filter(record=>record.mission_id===mission.id).length;
  renderHierarchy(mission); renderAlgorithms(); renderTargetSummary(mission); setMissionTab(state.missionTab, false);
}
function updateMissionGuide(mission){
  if(!mission)return;
  const jobs=mission.jobs||[],imageCount=jobs.reduce((n,j)=>n+(j.images||[]).length,0),reports=state.reports.filter(r=>r.mission_id===mission.id);
  const readyImages=jobs.flatMap(job=>(job.images||[]).map(image=>({job,image}))).filter(({job,image})=>['stream-ready','downloaded','local-ready'].includes(state.rasterAccess.find(r=>r.mission_id===mission.id&&r.job_id===job.id&&r.image_id===image.id)?.status)).length;
  const completed=reports.some(run=>['complete','partial'].includes(run.status)&&run.outputs?.length),active=reports.some(run=>['queued','running','pausing','paused','stopping'].includes(run.status));
  let guide=completed?['Your results are ready','Open a result to compare its classification with the original image, inspect pixels, and save corrections.','View runs & results','reports']:active?['Analysis is in progress','Follow real progress in Runs & results. Pause and Stop are available above.','View progress','reports']:!imageCount?['Add imagery to this mission','Choose images in the catalog, then add the selected scenes as a job.','Choose imagery','catalog']:readyImages<imageCount?['Connect your source pixels',`${readyImages} of ${imageCount} images have readable source data. Configure local files, streaming, or downloaded rasters before running analysis.`,'Set up source data','rasters']:['Choose images and an analysis model','Select a job or individual images, choose a compatible model, then run analysis.','Choose images & models','plan'];
  $('missionGuideTitle').textContent=guide[0];$('missionGuideText').textContent=guide[1];$('missionNextAction').textContent=guide[2];$('missionNextAction').dataset.target=guide[3];

}
function renameCurrentMission() {
  const mission = state.missions.find(m=>m.id===state.currentMissionId); if (!mission) return;
  const next = prompt("Mission name",mission.name); if (!next?.trim()) return;
  mission.name = next.trim(); mission.updatedAt = new Date().toISOString(); persistMissions(); renderMissionDetail();
}
function imageDisplayName(imageOrId) {
  if(typeof imageOrId==='object')return imageOrId.name||imageOrId.original_name||imageOrId.id;
  for(const mission of state.missions)for(const job of mission.jobs||[]){const image=(job.images||[]).find(i=>i.id===imageOrId);if(image)return image.name||image.original_name||image.id;}
  return imageOrId;
}
function missionThumbnail(missionId,jobId,image){
  const fallback='/api/mission-thumbnail?'+new URLSearchParams({mission_id:missionId,job_id:jobId,image_id:image.id});
  const src=image.thumbnail&&(!state.offline||image.thumbnail.startsWith('/'))?image.thumbnail:fallback;
  return `<img src="${escapeHtml(src)}" data-source-fallback="${escapeHtml(fallback)}" alt="Source image preview" loading="lazy" onerror="if(!this.dataset.retried){this.dataset.retried='1';this.src=this.dataset.sourceFallback;}else{this.replaceWith(Object.assign(document.createElement('span'),{className:'thumbnail-unavailable',textContent:'No preview',title:'No thumbnail or readable RGB bands available'}));}">`;
}
function renderHierarchy(mission) {
  const tree = $("hierarchyTree"), jobs = mission.jobs || [];
  const missionImages = jobs.flatMap(job=>job.images||[]);
  const missionChecked = missionImages.length>0 && missionImages.every(image=>state.analysisTargets.has(`image:${jobs.find(job=>(job.images||[]).includes(image))?.id}:${image.id}`));
  let html = `<div class="tree-row tree-mission-row"><span class="tree-indent"></span><input type="checkbox" class="tree-check" data-node="mission:${escapeHtml(mission.id)}" ${missionChecked?"checked":""} ${missionImages.length?"":"disabled"}><span class="tree-mission-icon">◎</span><div class="tree-label"><b>${escapeHtml(mission.name)}</b><small>MISSION · ${jobs.length} jobs</small></div><span class="tree-status">${jobs.length?"ACTIVE":"EMPTY"}</span></div>`;
  if (!jobs.length) html += `<div class="tree-empty">No jobs yet. Select scenes in the search table and choose <b>Add selected scenes</b> to create one.</div>`;
  for (const job of jobs) {
    const images = job.images || [], expanded = state.expandedJobs.has(job.id), jobKey = `job:${job.id}`;
    const liveRun=state.reports.find(run=>run.mission_id===mission.id&&run.job_id===job.id);
    const jobChecked = images.length>0 && images.every(image=>state.analysisTargets.has(`image:${job.id}:${image.id}`));
    html += `<div class="tree-row tree-job-row"><button class="tree-expand" data-expand="${escapeHtml(job.id)}" aria-label="${expanded?"Collapse":"Expand"} job">${expanded?"⌄":"›"}</button><input type="checkbox" class="tree-check" data-node="${escapeHtml(jobKey)}" ${jobChecked?"checked":""} ${images.length?"":"disabled"}><span class="tree-job-icon">▤</span><div class="tree-label"><b>${escapeHtml(job.name)}</b><small>JOB · ${images.length} scenes${job.analysisPlan?` · ${job.analysisPlan.algorithms.length} algorithms planned`:""}</small></div><span class="status-pill draft-pill">${escapeHtml(liveRun?.status||job.status||"draft").toUpperCase().replaceAll('_',' ')}</span></div>`;
    if (expanded) for (const image of images) {
      const key = `image:${job.id}:${image.id}`;
      html += `<div class="tree-row tree-image-row"><span class="tree-branch">└</span><input type="checkbox" class="tree-check" data-node="${escapeHtml(key)}" ${state.analysisTargets.has(key)?"checked":""}><span class="tree-image-thumb">${missionThumbnail(mission.id,job.id,image)}</span><div class="tree-label"><b title="${escapeHtml(imageDisplayName(image))}">${escapeHtml(imageDisplayName(image))}</b><small>${escapeHtml(image.platform||image.catalog||"Satellite scene")} · ${fmtDate(image.datetime)}</small></div><button type="button" class="text-button" data-image-footprint="${escapeHtml(job.id)}:${escapeHtml(image.id)}">Show on map</button><span class="image-plan-count">${(image.planAlgorithms||[]).length?`${image.planAlgorithms.length} alg.`:""}</span></div>`;
    }
  }
  tree.innerHTML = html;
  tree.querySelectorAll("[data-expand]").forEach(button=>button.addEventListener("click",()=>{const id=button.dataset.expand;state.expandedJobs.has(id)?state.expandedJobs.delete(id):state.expandedJobs.add(id);renderMissionDetail();}));
  tree.querySelectorAll('[data-image-footprint]').forEach(button=>button.addEventListener('click',()=>{
    const image=jobs.flatMap(job=>(job.images||[]).map(image=>({job,image}))).find(entry=>`${entry.job.id}:${entry.image.id}`===button.dataset.imageFootprint)?.image;
    if(!image)return;
    if(!image.geometry&&!image.bbox){toast('This image has no georeferenced footprint. Check the imported product metadata.');return;}
    if(!$('missionWorkspace').classList.contains('hidden'))toggleMissionWorkspace();
    focusItem(image);
  }));
  tree.querySelectorAll("[data-node]").forEach(input=>input.addEventListener("change",()=>toggleAnalysisTarget(input.dataset.node,input.checked,mission)));
}
function toggleAnalysisTarget(key, checked, mission) {
  if (key.startsWith("mission:")) for (const job of mission.jobs||[]) for (const image of job.images||[]) {
    const imageKey=`image:${job.id}:${image.id}`; checked?state.analysisTargets.add(imageKey):state.analysisTargets.delete(imageKey);
  }
  else if (key.startsWith("job:")) { const job=mission.jobs.find(j=>`job:${j.id}`===key); for (const image of job?.images||[]) {
    const imageKey=`image:${job.id}:${image.id}`; checked?state.analysisTargets.add(imageKey):state.analysisTargets.delete(imageKey);
  }}
  else checked?state.analysisTargets.add(key):state.analysisTargets.delete(key);
  renderMissionDetail();
}
function getTargetImages(mission) {
  const output=[];
  for (const job of mission.jobs||[]) for (const image of job.images||[]) {
    if (state.analysisTargets.has(`image:${job.id}:${image.id}`)) output.push({job,image});
  }
  return output;
}
function renderTargetSummary(mission) {
  const targets=getTargetImages(mission), jobs=new Set(targets.map(x=>x.job.id));
  $("targetSummary").innerHTML = targets.length
    ? `<span class="target-icon">⌖</span><span><b>${targets.length} image${targets.length===1?"":"s"}</b> across ${jobs.size} job${jobs.size===1?"":"s"} selected</span>`
    : `<span class="target-icon">⌖</span><span>Select a job or image to configure</span>`;
  const canRun = targets.length && state.algorithms.size && (!mission.control||mission.control==='resume');
  $("savePlanButton").disabled = !canRun;
  $("runInferenceButton").disabled = !canRun;
}
function satelliteGuidance(id,compact=false){
  const info=state.algorithmSuitability?.[id];
  if(!info)return '<small class="satellite-guidance">Satellite suitability has not been loaded. Check Help before choosing inputs.</small>';
  if(compact)return `<span class="satellite-guidance satellite-quick" data-suitability-id="${escapeHtml(id)}"><small><strong>Satellites / sensors:</strong> ${escapeHtml(info.satellites)}</small></span>`;
  return `<span class="satellite-guidance" data-suitability-id="${escapeHtml(id)}"><small><strong>Satellites / sensors:</strong> ${escapeHtml(info.satellites)}</small><small><strong>Input:</strong> ${escapeHtml(info.requirements)}</small><small class="satellite-limit">${escapeHtml(info.limitations)}</small></span>`;
}
function satelliteDetails(id){
  return `<details class="satellite-details"><summary>Input requirements & limitations</summary>${satelliteGuidance(id)}</details>`;
}
function modelStartingPoint(id) {
  const text={dynamicworld:'Pretrained classifier · no labels needed for inference. Sentinel-2 L1C only.',prithvi_crop:'Pretrained classifier · no labels needed. Requires three real HLS acquisition dates.',openearthmap:'Pretrained classifier · no labels needed. Requires 0.25–0.5 m RGB, not Sentinel/Landsat.',prithvi:'Pretrained encoder only · you must supply labels for its classification head.',unet:'No built-in LULC checkpoint · train from your labels; compatible local checkpoints may be continued.',ensemble:'Combines locally fitted classifiers · requires your labeled examples.'};
  return escapeHtml(text[id]||'No built-in classifier weights · fit this model using your labeled examples.');
}
function openJobLabels(jobId) {
  setMissionTab('training');
  if(jobId){$('groundTruthJob').value=jobId;renderTrainingPanel();}
  $('referenceClass').focus();
}
function renderAlgorithms() {
  const container=$("algorithmList");
  container.innerHTML=algorithmCatalog.map(a=>`${a.id==='dynamicworld'?'<h3 class="model-group-heading">Pretrained classification · no labels to start</h3>':a.id==='random_forest'?'<h3 class="model-group-heading">Train with your labeled examples</h3><button type="button" class="secondary-button full-width" id="algorithmLabelShortcut">Create or import training labels</button>':''}<div class="algorithm-choice"><label class="algorithm-option ${state.algorithms.has(a.id)?"chosen":""}"><input type="checkbox" data-algorithm="${a.id}" ${state.algorithms.has(a.id)?"checked":""}><span class="algorithm-mark ${a.color}">${a.mark}</span><span class="algorithm-copy"><b>${a.name}</b><small>${a.detail}</small>${satelliteGuidance(a.id,true)}</span><span class="algorithm-tag">${a.tag}</span></label>${satelliteDetails(a.id)}<p class="model-starting-point">${modelStartingPoint(a.id)}</p></div>`).join("");
  if($('algorithmLabelShortcut'))$('algorithmLabelShortcut').onclick=()=>openJobLabels(getTargetImages(state.missions.find(m=>m.id===state.currentMissionId)||{} )[0]?.job.id);
  container.querySelectorAll("[data-algorithm]").forEach(input=>input.addEventListener("change",()=>{const pretrained=["dynamicworld","prithvi_crop","openearthmap"];if(input.checked){if(pretrained.includes(input.dataset.algorithm)){state.algorithms.clear();$("bandAlignment").checked=false;}else pretrained.forEach(id=>state.algorithms.delete(id));state.algorithms.add(input.dataset.algorithm);}else state.algorithms.delete(input.dataset.algorithm);renderAlgorithms();const mission=state.missions.find(m=>m.id===state.currentMissionId);if(mission)renderTargetSummary(mission);}));
}
function saveAnalysisPlan() {
  const mission=state.missions.find(m=>m.id===state.currentMissionId);if(!mission)return;
  const targets=getTargetImages(mission), ids=[...state.algorithms];
  for(const job of mission.jobs||[]){const picked=targets.filter(t=>t.job.id===job.id).map(t=>t.image.id);if(!picked.length)continue;
    for(const image of job.images||[]) if(picked.includes(image.id)) image.planAlgorithms=ids;
    job.analysisPlan={algorithms:ids,imageIds:picked,savedAt:new Date().toISOString()};job.status="planned";
  }
  mission.updatedAt=new Date().toISOString();persistMissions();renderMissionDetail();toast(`Saved ${ids.length} algorithm${ids.length===1?"":"s"} for ${targets.length} image${targets.length===1?"":"s"}.`);
}
function setMissionTab(tab, rerender = true) {
  if(tab==='plan')tab='qualification';
  for(const [key,id] of [['automatic','missionAutomation'],['qualification','missionQualification'],['inference','missionOperational']])$(id).classList.toggle('hidden',tab!==key);
  if(tab === "insights") { window.SpectraScopeInsights?.open(); return; }
  const tabChanged=state.missionTab!==tab;
  state.missionTab = tab;
  if(tabChanged)$('missionWorkspace').scrollTop=0;
  $('missionViewerPanel').classList.toggle('hidden',tab!=='viewer');
  $('missionDetail').classList.toggle('viewing-image',tab==='viewer');
  if(tab==='viewer')requestAnimationFrame(()=>window.resultImageViewer?.resize());
  $("analysisPlanPanel").classList.toggle("hidden", tab !== "plan");
  $("rasterPanel").classList.toggle("hidden", tab !== "rasters");
  $("reportsPanel").classList.toggle("hidden", tab !== "reports");
  $("trainingPanel").classList.toggle("hidden", tab !== "training");
  $("anomalyPanel").classList.toggle("hidden", tab !== "anomalies");
  document.querySelectorAll("[data-mission-tab]").forEach(button=>button.classList.toggle("active", button.dataset.missionTab === tab || button.dataset.missionTab==='qualification'&&tab==='training' || button.dataset.missionTab==='inference'&&['reports','viewer','rasters'].includes(tab)));
  window.MissionWorkspaces?.open(tab);
  $('missionGuideTitle').closest('.mission-guide').classList.toggle('hidden',['automatic','qualification','inference'].includes(tab));
  if (tab === "reports") renderReportsPanel();
  if (tab === "rasters") renderRasterPanel();
  if (tab === "training") renderTrainingPanel();
  if (tab === "anomalies") renderAnomalyPanel();
  if(tabChanged)requestAnimationFrame(()=>$('missionWorkspace').scrollTop=0);
}
function reportForCurrentJob() {
  return state.reports.filter(run=>run.mission_id===state.currentMissionId);
}
function currentRasterJob() {
  const mission=state.missions.find(item=>item.id===state.currentMissionId);
  return mission?.jobs?.find(job=>job.id===$("rasterAccessJob").value) || mission?.jobs?.[0] || null;
}
function selectAllRasterScenes() {
  const job=currentRasterJob(); if(!job)return;
  state.rasterTargets=new Set((job.images||[]).map(image=>image.id)); renderRasterPanel();
}
function renderRasterPanel() {
  const mission=state.missions.find(item=>item.id===state.currentMissionId);if(!mission)return;
  const jobSelect=$("rasterAccessJob"),previous=jobSelect.value;
  jobSelect.innerHTML=(mission.jobs||[]).map(job=>`<option value="${escapeHtml(job.id)}">${escapeHtml(job.name)} · ${(job.images||[]).length} images</option>`).join("");
  if([...jobSelect.options].some(option=>option.value===previous))jobSelect.value=previous;
  const job=currentRasterJob(), mode=$("rasterAccessMode").value;
  const explanations={stream:"Keep source rasters remote. The worker reads only required windows and bands through catalog URLs; no full local copy is made.",download:"Download all science-raster assets for the selected scenes into the local project cache. Use this when offline processing or repeatable local runs matter.",local_path:`Use your existing local GeoTIFF or JPEG 2000 files. Paths must be inside the mounted folder: ${state.localRasterRoot||"data/local-rasters"}.`};
  $("rasterModeExplainer").textContent=explanations[mode];
  const images=job?.images||[];
  state.rasterTargets=new Set([...state.rasterTargets].filter(id=>images.some(image=>image.id===id)));
  $("rasterSceneList").innerHTML=images.length?images.map(image=>{const checked=state.rasterTargets.has(image.id)?"checked":"";const existing=state.rasterAccess.find(record=>record.mission_id===mission.id&&record.job_id===job.id&&record.image_id===image.id);const localInput=mode==="local_path"?existing?.local_sources?.length?`<input class="raster-local-path" data-raster-path="${escapeHtml(image.id)}" disabled placeholder="${existing.local_sources.length} source files connected" title="All band files are connected. Metadata edits preserve the entire product.">`:`<input class="raster-local-path" data-raster-path="${escapeHtml(image.id)}" placeholder="relative/path/to/scene.tif" value="${escapeHtml(existing?.relative_path||"")}">`:"";return `<label class="raster-scene-row"><input type="checkbox" data-raster-image="${escapeHtml(image.id)}" ${checked}><span class="raster-scene-icon">▧</span><span><b>${escapeHtml(imageDisplayName(image))}</b><small>${escapeHtml(image.platform||image.catalog||"Satellite scene")} · ${fmtDate(image.datetime)}</small></span>${existing?`<em>${escapeHtml(existing.status)}</em>`:""}${localInput}</label>`;}).join(""):"<div class=\"raster-empty\">This job has no selected imagery yet.</div>";
  $("rasterSceneList").querySelectorAll("[data-raster-image]").forEach(input=>input.addEventListener("change",()=>{input.checked?state.rasterTargets.add(input.dataset.rasterImage):state.rasterTargets.delete(input.dataset.rasterImage);}));
  const seen=new Set(),records=state.rasterAccess.filter(record=>record.mission_id===mission.id&&record.job_id===job?.id).filter(record=>{if(seen.has(record.image_id))return false;seen.add(record.image_id);return true;});
  $('rasterAccessList').innerHTML=records.length?`<div class="raster-access-heading"><div><div class="eyebrow">Configured sources</div><h3>${escapeHtml(job.name)}</h3></div><span>${records.length} current inputs</span></div>${records.map(record=>{
    const calibration=record.calibration_metadata||record.local_sources?.find(source=>source.radiometry)?.radiometry;
    const isSentinel=record.sensor==='sentinel-2'&&['L1C','L2A'].includes(record.processing_level), isLandsat=record.sensor==='landsat';
    return `<div class="raster-access-row"><span class="raster-access-mode">${escapeHtml(record.mode.replace('_',' '))}</span><div><b>${escapeHtml(imageDisplayName(record.image_id))}</b><small>${escapeHtml(record.message||record.status)}</small><small>${calibration?`Supplier calibration attached · ${calibration.processing_baseline?'baseline '+escapeHtml(calibration.processing_baseline):escapeHtml(calibration.processing_level||'L2')} · ${escapeHtml(calibration.output_units)}`:(isSentinel?'SAFE calibration XML not attached':'No additional supplier metadata attached')}</small>${(isSentinel||isLandsat)&&record.local_sources?.length?`<label class="secondary-button">${isSentinel?'Attach SAFE metadata':'Attach Landsat MTL'}<input type="file" accept=".xml,.txt,.json" data-supplier-metadata="${escapeHtml(record.id)}" hidden></label>`:''}</div><span class="status-pill draft-pill">${escapeHtml(record.status)}</span></div>`;
  }).join('')}`:'<div class="raster-empty">No source data is configured for this job.</div>';
  $('rasterAccessList').querySelectorAll('[data-supplier-metadata]').forEach(input=>input.onchange=async()=>{
    if(!input.files.length)return;input.disabled=true;
    try{
      const form=new FormData();form.append('file',input.files[0]);
      const result=await api(`/api/raster-access/${encodeURIComponent(input.dataset.supplierMetadata)}/supplier-metadata`,{method:'POST',body:form});
      state.rasterAccess.unshift(result.record);renderRasterPanel();toast('Supplier calibration attached. Run analysis again to create new results; earlier runs are preserved.');
    }catch(error){toast(error.message);}finally{input.disabled=false;}
  });

}
async function configureRasterAccess() {
  const mission=state.missions.find(item=>item.id===state.currentMissionId),job=currentRasterJob(),mode=$("rasterAccessMode").value;
  const imageIds=[...state.rasterTargets];if(!mission||!job||!imageIds.length){toast("Select at least one job image.");return;}
  const localPaths={};document.querySelectorAll("[data-raster-path]").forEach(input=>{localPaths[input.dataset.rasterPath]=input.value.trim();});
  if(mode==="local_path"&&imageIds.some(id=>!localPaths[id])){toast("Provide a local raster path for every selected scene.");return;}
  const button=$("configureRasterAccessButton"), labels={stream:"Saving stream adapters…",download:"Downloading science rasters…",local_path:"Verifying local paths…"};button.disabled=true;button.textContent=labels[mode];
  try{await persistMissions();const bandNames=$("rasterBandNames").value.split(',').map(value=>value.trim()).filter(Boolean);const result=await api("/api/raster-access",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({mission_id:mission.id,job_id:job.id,image_ids:imageIds,mode,local_paths:localPaths,band_names:bandNames,processing_level:$("rasterProductLevel").value,temporal_dates:$("rasterTemporalDates").value.split(",").map(value=>value.trim()).filter(Boolean)})});state.rasterAccess.unshift(...result.records);state.localRasterRoot=result.local_raster_root||state.localRasterRoot;state.rasterTargets.clear();renderMissionDetail();setMissionTab("rasters");toast(`${result.records.length} raster access record${result.records.length===1?"":"s"} configured.`);}catch(error){toast(error.message);}finally{button.disabled=false;button.textContent="Configure selected scenes";}
}
function resultLegend(output, layer) {
  const swatch=(color,label)=>`<span><i style="background:${escapeHtml(color)}"></i>${escapeHtml(label)}</span>`;
  const items=['consensus','reviewed'].includes(layer)?(output.class_legend||[]).map(item=>swatch(item.color,item.label)).join(''):
    `${layer!=='disagreement'?swatch('#46cbaa','All models agree'):''}${layer!=='agreement'?swatch('#ff7e38','Models disagree'):''}`;
  return `<div class="result-legend">${items}<span><i class="nodata-swatch"></i>${layer==='agreement'?'Other pixels hidden':layer==='disagreement'?'Other pixels hidden':'No valid data'}</span></div>`;
}
function classBreakdown(output) {
  const classes=output.class_distribution||[];
  if(!classes.length)return '<p>Class coverage is unavailable for this result.</p>';
  const summary=output.classification_summary;
  return `<details class="class-breakdown" aria-label="Classified land-cover classes"><summary>Class coverage · ${summary.predicted_classes}/${summary.configured_classes} classes present</summary><p>Colors apply to the Land cover view. Coverage uses valid pixels.</p><table><thead><tr><th scope="col">Class</th><th scope="col">Pixels</th><th scope="col">Coverage</th></tr></thead><tbody>${classes.map(item=>`<tr class="${item.pixels?'':'class-absent'}"><th scope="row"><i style="background:${escapeHtml(item.color)}"></i>${escapeHtml(item.label)}${item.pixels?'':'<small>Not predicted</small>'}</th><td>${Number(item.pixels).toLocaleString()}</td><td>${item.percent==null?'—':`${item.percent}%`}</td></tr>`).join('')}</tbody></table></details>`;
}
function agreementBreakdown(output) {
  const classes=classBreakdown(output);
  const summary=output.agreement_summary;
  if(summary?.agreement_percent==null)return `${classes}<p>Agreement needs at least two successful models.</p>`;
  return `${classes}<div class="agreement-breakdown"><span><b>${summary.agreement_percent}%</b> agreement</span><span><b>${summary.disagreement_percent}%</b> disagreement</span></div><div class="agreement-distribution" role="img" aria-label="${summary.agreement_percent}% agreement, ${summary.disagreement_percent}% disagreement"><i style="width:${summary.agreement_percent}%"></i></div><p>${Number(summary.valid_pixels).toLocaleString()} valid pixels · ${summary.models_compared} models compared · nodata excluded</p>`;
}
function renderResultOutput(report, output, index) {
  const measured=output.agreement_summary?.agreement_percent!=null;
  const layer=measured?'review':'consensus';
  const mission=state.missions.find(m=>m.id===report.mission_id),job=mission?.jobs?.find(j=>j.id===report.job_id),image=job?.images?.find(i=>i.id===output.image_id);
  const source=image?`<div class="result-source-thumbnail"><span>Original image</span>${missionThumbnail(report.mission_id,report.job_id,image)}</div>`:'';
  return `<div class="result-output" data-output-index="${index}" data-output-run="${escapeHtml(report.id)}"><div>${source}<div class="result-preview"><img src="${escapeHtml(measured?output.review_preview:output.preview)}" alt="${measured?'Green agreement and orange disagreement':'Predicted land-cover classes'}"></div><div class="result-view-tabs" aria-label="Preview layer">${measured?'<button data-preview-layer="review" class="active">Agreement</button>':''}<button data-preview-layer="consensus" class="${measured?'':'active'}">Land cover</button></div><div class="preview-legend">${resultLegend(output,layer)}</div></div><div><b>${escapeHtml(imageDisplayName(output.image_id))}</b>${agreementBreakdown(output)}<p>Agreement means the models chose the same class. It does not prove the class is correct. Use Land cover to see the predicted classes, and validation F1 to assess accuracy.</p><div class="result-map-actions"><button class="secondary-button" data-review-result="${escapeHtml(report.id)}" data-review-index="${index}" data-map-layer="${layer}">${measured?'Open image & agreement':'Open image & classification'}</button>${measured?`<button class="secondary-button" data-review-result="${escapeHtml(report.id)}" data-review-index="${index}" data-map-layer="consensus">Compare land cover</button>`:''}</div><button class="secondary-button" data-export-class-vectors>Export class polygons · GeoJSON</button><div class="vector-export-status" role="status"></div><button class="secondary-button" data-save-result-layers>Save / reuse result layers ↗</button><details class="result-downloads"><summary>Download result rasters & report</summary><div class="result-links">${Object.entries(output.files||{}).map(([name,url])=>`<a href="${escapeHtml(url)}">${escapeHtml(name)} GeoTIFF ↓</a>`).join('')}<a href="/api/results/${encodeURIComponent(report.id)}/report.json">Full report ↓</a></div></details></div></div>`;
}
function renderReportsPanel() {
  const reports = reportForCurrentJob();
  $("reportPanelCount").textContent = `${reports.length} report${reports.length===1?"":"s"}`;
  const mission = state.missions.find(item=>item.id===state.currentMissionId);
  const jobsById = new Map((mission?.jobs||[]).map(job=>[job.id,job]));
  $("reportList").innerHTML = reports.length ? reports.map(report=>{
    const job = jobsById.get(report.job_id), issue = report.consensus?.needs_user_review;
    const models = (report.models||[]).map(model=>{const metric=model.metrics;return `<li><b>${escapeHtml(model.name)} · ${escapeHtml(model.status||"queued")}</b><span>${escapeHtml(model.error||model.why||"")}</span>${modelInitialization(model)}${metric?.status==="measured"?`<h4>Validation · used for selection</h4>${modelMetricDetails(metric,report.classes||[])}`:"<span>Validation accuracy unmeasured</span>"}${model.test_metrics?.status==="measured"?`<h4>Independent test · excluded from selection</h4>${modelMetricDetails(model.test_metrics,report.classes||[],"independent test pixels")}`:""}</li>`;}).join("");
    const alignment = report.band_alignment?.enabled ? `Band coregistration enabled · ${escapeHtml(report.band_alignment.method)}` : "Band coregistration not requested";
    const agreement=report.consensus?.agreement_percent;
    const title=["queued","running"].includes(report.status)?`${escapeHtml(report.stage||report.status)} · ${report.progress||0}%`:report.error?escapeHtml(report.error):issue?"Results ready · review model differences":escapeHtml(report.status||"Report");
    const manifest=new Map((report.benchmark?.manifest||[]).map(item=>[item.patch_id,item.split]));
    const outputs=report.benchmark?.taxonomy_classes?['test','validation','train'].map(split=>{const rows=(report.outputs||[]).map((output,index)=>({output,index})).filter(item=>manifest.get(item.output.image_id)===split);return rows.length?`<details class="benchmark-output-group"><summary>${escapeHtml(split)} maps · ${rows.length} images</summary>${rows.map(({output,index})=>renderResultOutput(report,output,index)).join('')}</details>`:'';}).join(''):(report.outputs||[]).map((output,index)=>renderResultOutput(report,output,index)).join('');
    return `<article id="report-${escapeHtml(report.id)}" class="report-card ${issue?"needs-review":""}"><div class="report-card-top"><div><div class="eyebrow">${escapeHtml(job?.name||"Analysis job")} · ${fmtDate(report.created_at)}</div><h3>${title}</h3></div><span class="agreement-score">${agreement==null?"Agreement unmeasured":`${agreement}% agreement`}</span></div><div class="pipeline-progress"><div class="pipeline-progress-fill" style="width:${Number(report.progress)||0}%"></div></div><p class="report-summary">${escapeHtml(report.consensus?.message||"")}</p><p class="report-summary">${escapeHtml(report.validation_protocol||"")}</p><div class="report-details"><div><b>Band handling</b><span>${alignment}</span></div><div><b>Combination</b><span>${escapeHtml(report.combination||"adaptive")}</span></div><div><b>Data status</b><span>${escapeHtml(report.data_status||"")}</span></div></div>${report.pretrained_example?`<section class="benchmark-summary"><div class="eyebrow">PRETRAINED PIXEL CLASSIFIER · RELEASED WEIGHTS</div><p>${escapeHtml(report.pretrained_example.note)}</p><a href="/api/results/${encodeURIComponent(report.id)}/report.json">Download complete report ↓</a></section>`:""}${benchmarkSummary(report)}<details class="benchmark-models"><summary>Individual algorithm results · ${(report.models||[]).length} models</summary><ul class="model-report-list">${models}</ul></details>${outputs}</article>`;
  }).join("") : `<div class="review-empty"><div>◈</div><h3>No inference reports yet</h3><p>Select images and model families in the analysis plan, then run the adaptive workflow. Each run will explain the model choice, band alignment policy, and whether the results need human review.</p></div>`;
  $("reportList").querySelectorAll('[data-review-result]').forEach(button=>button.addEventListener('click',()=>reviewOutputOnMap(button.dataset.reviewResult,button.dataset.reviewIndex,button.dataset.mapLayer)));
  $("reportList").querySelectorAll('[data-preview-layer]').forEach(button=>button.addEventListener('click',()=>{
    const card=button.closest('[data-output-index]'), output=state.reports.find(run=>run.id===card.dataset.outputRun)?.outputs[Number(card.dataset.outputIndex)];
    if(!output)return;
    const layer=button.dataset.previewLayer, image=card.querySelector('.result-preview img');
    image.src=layer==='consensus'?output.preview:output.review_preview;
    image.alt=layer==='consensus'?'Predicted land-cover classes':'Green agreement and orange disagreement';
    card.querySelector('.preview-legend').innerHTML=resultLegend(output,layer);
    card.querySelectorAll('[data-preview-layer]').forEach(item=>item.classList.toggle('active',item===button));
  }));
  $('reportList').querySelectorAll('[data-output-index]').forEach(card=>{
    const report=state.reports.find(run=>run.id===card.dataset.outputRun),index=Number(card.dataset.outputIndex);
    const vectorButton=card.querySelector('[data-export-class-vectors]');
    vectorButton.onclick=async()=>{
      const status=card.querySelector('.vector-export-status');vectorButton.disabled=true;
      status.textContent='Preparing original class polygons. Pixel-edge boundaries; no smoothing or class changes.';
      try{let task=await api(`/api/results/${encodeURIComponent(report.id)}/${index}/vector-export`,{method:'POST'});
        while(['queued','running'].includes(task.status)){status.textContent=`${task.progress}% · ${task.stage}`;await new Promise(resolve=>setTimeout(resolve,1000));task=await api(`/api/vector-exports/${task.id}`);}
        if(task.status!=='complete')throw new Error(task.error||'Vector export failed');
        status.innerHTML=`${Number(task.polygon_count).toLocaleString()} class polygons · <a href="${escapeHtml(task.download_url)}" download>Download GeoJSON ↓</a><p>WGS84 coordinates. Each polygon has class_id and class_name. This is a prediction, not ground truth.</p>`;
      }catch(e){status.textContent=e.message;}finally{vectorButton.disabled=false;}
    };
    if(report.requested_models?.length===1&&report.requested_models[0]==='dynamicworld'&&report.status==='complete'){
      const seed=document.createElement('button');seed.className='secondary-button';seed.textContent='Use as starting training labels';
      seed.onclick=async()=>{await reviewOutputOnMap(report.id,index,'consensus');window.beginPseudoLabels?.();};card.querySelector('.result-map-actions').append(seed);
    }
    const segmentButton=document.createElement('button');segmentButton.className='secondary-button';segmentButton.textContent='Segment regions';
    segmentButton.onclick=async()=>{await reviewOutputOnMap(report.id,index,'consensus');const panel=$('segmentationPanel');if(panel){panel.open=true;panel.scrollIntoView({block:'start',behavior:'smooth'});}};
    card.querySelector('.result-map-actions').append(segmentButton);
    if(!report?.outputs[index]?.has_manual_reviews)return;
    const button=document.createElement('button');button.className='secondary-button';button.textContent='Compare human-reviewed result';
    button.onclick=()=>reviewOutputOnMap(report.id,index,'reviewed');card.querySelector('.result-map-actions').append(button);
    if(report.outputs[index].review_history){const link=document.createElement('a');link.href=report.outputs[index].review_history;link.textContent='Review decisions JSON ↓';card.querySelector('.result-links').append(link);}
  });
  renderInferenceActivity();
}
async function reviewOutputOnMap(runId,index,layer='consensus') {
  if(reviewOutputOnMap.busy)return;
  state.anomalyMap=null;if($('anomalyInspector'))$('anomalyInspector').classList.add('hidden');
  reviewOutputOnMap.busy=true;
  document.querySelectorAll('[data-review-result],#analysisResultLayer').forEach(control=>control.disabled=true);
  try {
    toast('Loading analysis layer…');
    const report=state.reports.find(run=>run.id===runId);
    if(report&&state.currentMissionId!==report.mission_id)openMission(report.mission_id);
    if($('missionWorkspace').classList.contains('hidden'))toggleMissionWorkspace();
    setMissionTab('viewer');
    $('missionViewerStatus').textContent='Loading the saved source image and analysis overlay…';
    const result=await api(`/api/results/${encodeURIComponent(runId)}/${index}/viewer?layer=${encodeURIComponent(layer)}`);
    const output=result.output, measured=output.agreement_summary?.agreement_percent!=null;
    const panel=$("analysisMapReview");panel.classList.remove('hidden');
    panel.innerHTML=`<div class="eyebrow">ANALYSIS RESULT</div><b>${escapeHtml(imageDisplayName(output.image_id))}</b><label>Display <select id="analysisResultLayer">${(measured?[['review','Agreement + disagreement'],['consensus','Predicted land cover'],['agreement','Agreement only'],['disagreement','Disagreement only']]:[['consensus','Predicted land cover']]).map(([value,label])=>`<option value="${value}" ${layer===value?'selected':''}>${label}</option>`).join('')}</select></label>${resultLegend(output,layer)}${agreementBreakdown(output)}<small>Agreement ≠ accuracy. Land cover shows the assigned classes.</small>`;
    $("analysisResultLayer").addEventListener('change',event=>reviewOutputOnMap(runId,index,event.target.value));
    mountAnalysisInspector(runId,index,output,result.classes,layer);
    await window.showImageComparison(result.display,runId,index);
    $('missionViewerStatus').textContent=`${imageDisplayName(output.image_id)} — original image & analysis result`;
    $('missionViewerInspector').append($('analysisInspector'));
    toast(layer==='consensus'?'Land-cover classes displayed.':layer==='review'?'Green = all models agree. Orange = models disagree.':'Analysis layer updated.');
  } catch(error){$('missionViewerStatus').textContent='Could not open result: '+error.message;toast(error.message);}
  finally {reviewOutputOnMap.busy=false;document.querySelectorAll('[data-review-result],#analysisResultLayer').forEach(control=>control.disabled=false);}
}
function startRunPolling(){
  clearInterval(startRunPolling.timer);
  startRunPolling.timer=setInterval(async()=>{
    if(startRunPolling.busy)return;
    if(!state.reports.some(run=>['queued','running','pausing','paused','stopping'].includes(run.status))&&Date.now()-(startRunPolling.lastCheck||0)<5000)return;
    startRunPolling.lastCheck=Date.now();
    startRunPolling.busy=true;
    try{const result=await api('/api/inference/runs');
      const fingerprint=runs=>JSON.stringify(runs.map(run=>[run.id,run.status,run.progress,run.stage,(run.models||[]).map(model=>[model.id,model.status,model.progress]),(run.outputs||[]).map(output=>output.has_manual_reviews)]));
      const changed=fingerprint(state.reports)!==fingerprint(result.runs);state.reports=result.runs;
      acceptAutomaticTrainingRuns({runs:result.runs.filter(run=>run.trigger==='correction_threshold'&&['queued','running','pausing','paused','stopping'].includes(run.status))});
      renderInferenceActivity();
      if(changed&&state.currentMissionId){updateMissionGuide(state.missions.find(m=>m.id===state.currentMissionId));if(state.missionTab==='reports')renderReportsPanel();}
    }catch(error){console.warn('Progress polling failed',error);}finally{startRunPolling.busy=false;}
  },1000);
}
async function runInference() {
  const mission=state.missions.find(item=>item.id===state.currentMissionId); if(!mission)return;
  const targets=getTargetImages(mission), algorithms=[...state.algorithms];
  if (!targets.length || !algorithms.length) { toast("Choose at least one image and model family."); return; }
  if(mission.control&&mission.control!=='resume'){toast('Resume this mission before starting another run.');return;}
  const missing=targets.filter(({job,image})=>!['stream-ready','downloaded','local-ready'].includes(state.rasterAccess.find(r=>r.mission_id===mission.id&&r.job_id===job.id&&r.image_id===image.id)?.status));
  if(missing.length){setMissionTab('rasters');toast(`Configure source data for ${missing.length} selected image(s) before running.`);return;}

  if(algorithms.some(id=>!['dynamicworld','prithvi_crop','openearthmap'].includes(id))){
    const blocked=[...new Set(targets.map(t=>t.job.id))].find(id=>new Set(state.groundTruth.filter(l=>l.mission_id===mission.id&&l.job_id===id&&l.role!=='validation'&&l.confidence>=3&&(!l.image_id||targets.some(t=>t.job.id===id&&t.image.id===l.image_id))).map(l=>l.label)).size<2);
    if(blocked){openJobLabels(blocked);$('referenceStatus').textContent='This model needs training examples for at least two classes in the selected images. Draw them here or import external polygons, then return to Images & models.';return;}
  }
  saveAnalysisPlan();
  const button=$("runInferenceButton"); button.disabled=true; button.textContent="Preparing images…";
  const fill=$("pipelineProgressFill"); fill.style.width="0%";
  $("pipelineProgressText").textContent='Submitting jobs · 0%';
  document.querySelectorAll(".pipeline-step").forEach((step,index)=>step.classList.toggle("active",index===0));
  try {
    await persistMissions();
    const groups = new Map();
    targets.forEach(({job,image})=>{if(!groups.has(job.id))groups.set(job.id,[]);groups.get(job.id).push(image.id);});
    let index=0;
    for (const [jobId,imageIds] of groups) {
      index += 1;
      button.textContent=`Running job ${index} of ${groups.size}…`;
      document.querySelectorAll(".pipeline-step").forEach((step,stepIndex)=>step.classList.toggle("active",stepIndex<=1));
      const job=mission.jobs.find(job=>job.id===jobId);
      const run=await api("/api/inference/runs",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({mission_id:mission.id,job_id:jobId,image_ids:imageIds,algorithms,band_alignment:$("bandAlignment").checked,combination:$("combinationMode").value,max_size:Number($("analysisGrid").value),geometry:job?.aoi||null})});
      state.reports.unshift(run);
      renderInferenceActivity();
    }
    mission.updatedAt=new Date().toISOString(); persistMissions(); renderMissionDetail(); setMissionTab("reports");
    startRunPolling();toast("Jobs queued. Reports will update as actual model workers run.");
  } catch(error) { toast(error.message); fill.style.width="0"; }
  finally { button.disabled=false; button.textContent="Run selected analysis"; }
}
function currentTrainingJob() {
  const mission=state.missions.find(item=>item.id===state.currentMissionId);
  return mission?.jobs?.find(job=>job.id===$("groundTruthJob").value) || mission?.jobs?.[0] || null;
}
function renderTrainingPanel() {
  const mission=state.missions.find(item=>item.id===state.currentMissionId); if(!mission)return;
  const select=$("groundTruthJob"), previous=select.value;
  select.innerHTML=(mission.jobs||[]).map(job=>`<option value="${escapeHtml(job.id)}">${escapeHtml(job.name)} · ${(job.images||[]).length} images</option>`).join("");
  if ([...select.options].some(option=>option.value===previous)) select.value=previous;
  const job=currentTrainingJob(); const labels=state.groundTruth.filter(label=>label.mission_id===mission.id && label.job_id===job?.id);
  updateVectorImportScope(mission,job);
  window.referenceLabelEditor?.update(mission,job);
  const classes=[...new Set(labels.filter(label=>label.role!=='validation'&&label.confidence>=3).map(label=>label.label))];
  $("trainingReadiness").innerHTML=job ? `<b>${labels.length}</b> reviewed samples · <b>${classes.length}</b> class${classes.length===1?"":"es"}<span>Training classes only · validation is excluded. Pixel coverage and class balance are checked on the analysis grid when the worker starts.</span>` : "Create a job first.";
  $("groundTruthList").innerHTML=labels.length ? `<div class="ground-truth-heading"><div><div class="eyebrow">STORED REFERENCE LABELS</div><h3>${escapeHtml(job.name)}</h3></div><span>${labels.length} samples</span></div><div class="ground-truth-table">${labels.map(label=>`<div><b>${escapeHtml(label.label)}</b><span>${escapeHtml(label.image_id ? (job.images.find(i=>i.id===label.image_id)?.name||label.image_id) : "Shared across job")} · ${label.role==='validation'?'Validation only':label.origin==='reviewed_pseudo_label'?'Reviewed pseudo-label · training only':'Human training reference'} · ${label.geometry?.type||"No geometry"} · confidence ${label.confidence}/5</span><small>${escapeHtml(label.notes||"No note")}</small></div>`).join("")}</div>` : `<div class="ground-truth-empty">No labels yet. Open a source image above and draw examples, or import external polygons.</div>`;
  const history=state.tuningRuns.filter(run=>run.mission_id===mission.id&&run.job_id===job?.id);
  $("tuningHistory").innerHTML=history.length ? history.slice(0,4).map(run=>`<div class="tuning-row"><b>${escapeHtml(run.model_name)}</b><span>${escapeHtml(run.status)} · ${run.label_count} labels / ${run.class_count} classes</span></div>`).join("") : "";
  if(job)renderAutomaticTrainingPolicy(mission.id,job.id);
}
async function captureGroundTruth() {
  const mission=state.missions.find(item=>item.id===state.currentMissionId), job=currentTrainingJob();
  const label=$("groundTruthLabel").value.trim();
  if(!mission||!job||!label){toast("Choose a job and enter a land-cover class.");return;}
  const button=$("captureGroundTruthButton");button.disabled=true;button.textContent="Reading map AOI…";
  try{
    const area=await api("/api/aoi");
    const record=await api("/api/ground-truth",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({mission_id:mission.id,job_id:job.id,label,geometry:area.geometry,role:$("groundTruthRole").value,confidence:Number($("groundTruthConfidence").value),notes:$("groundTruthNotes").value.trim()})});
    state.groundTruth.unshift(record); $("groundTruthLabel").value=""; $("groundTruthNotes").value=""; renderTrainingPanel(); toast("Ground truth saved to the local project database.");
    acceptAutomaticTrainingRuns(record.automatic_training);
  }catch(error){toast(error.message);}finally{button.disabled=false;button.textContent="⌖ Save current map AOI as ground truth";}
}
async function prepareFineTuning() {
  const mission=state.missions.find(item=>item.id===state.currentMissionId),job=currentTrainingJob();if(!mission||!job)return;
  const button=$("fineTuneButton");button.disabled=true;button.textContent="Checking reference labels…";
  try{
    const run=await api("/api/fine-tuning/runs",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({mission_id:mission.id,job_id:job.id,model:$("fineTuneModel").value})});
    state.tuningRuns.unshift(run);state.reports.unshift(run);startRunPolling();setMissionTab('reports');toast(run.message);
  }catch(error){toast(error.message);}finally{button.disabled=false;button.textContent="Train a new model version";}
}
async function alignBands() {
  const file=$("alignmentFile").files?.[0];if(!file){toast("Choose a multiband GeoTIFF to align.");return;}
  const button=$("alignBandsButton"),result=$("alignmentResult");button.disabled=true;button.textContent="Estimating band shifts…";result.textContent="";
  const form=new FormData();form.append("file",file);form.append("reference_band",$("referenceBand").value);
  try{const alignment=await api("/api/band-alignment",{method:"POST",body:form});const shifts=alignment.shifts.filter(item=>!item.reference).map(item=>`B${item.band}: ${item.shift_x_px}px, ${item.shift_y_px}px`).join(" · ");result.innerHTML=`<b>Aligned to band ${alignment.reference_band}</b><span>${escapeHtml(shifts)}</span><a href="${escapeHtml(alignment.download_url)}">Download aligned GeoTIFF ↓</a>`;toast("Aligned GeoTIFF is ready to download.");}catch(error){result.textContent=error.message;result.classList.add("error");}finally{button.disabled=false;button.textContent="Align bands & save GeoTIFF";}
}
async function changeBasemap() {
  state.basemapUserChoice=true;state.preferredBasemap=$('basemap').value;
  if($("basemap").value==='import-orthophoto'){window.SpectraScopeBasemaps.openOrthophoto();$("basemap").value='offline';return;}
  if($("basemap").value.startsWith("custom-")){try{await window.SpectraScopeBasemaps.activate($("basemap").value);}catch(error){toast(error.message);}return;}
  try { await api("/api/basemap", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({key:$("basemap").value}) }); }
  catch (error) { toast(error.message); }
}
function updateUploadOptions() {
  const rgb = $("uploadMode").value === "rgb";
  $("bandsLabel").classList.toggle("hidden", !rgb); $("singleBandLabel").classList.toggle("hidden", rgb);
}
async function uploadFile() {
  const file = $("fileInput").files?.[0]; if (!file) return;
  const options = $("uploadOptions"); options.classList.add("visible");
  const message = $("uploadMessage"); message.textContent = `Uploading ${file.name}…`; message.classList.remove("error");
  const button = $("uploadButton"); button.disabled = true;
  const form = new FormData(); form.append("file", file); form.append("mode", $("uploadMode").value);
  form.append("bands", $("rgbBands").value); form.append("band", $("singleBand").value);
  try {
    const result = await api("/api/upload", {method:"POST", body:form});
    message.textContent = `${file.name} added to the map · ${result.width} × ${result.height}`;
    toast("GeoTIFF preview added to the MC² map.");
  } catch (error) { message.textContent = error.message; message.classList.add("error"); }
  finally { button.disabled = false; $("fileInput").value = ""; }
}

init();
