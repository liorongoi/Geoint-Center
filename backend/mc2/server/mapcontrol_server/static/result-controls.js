/* Display controls shared by every MC² map, including legacy result views. */
(() => {
  'use strict';
  function start(ctx) {
    const map=ctx.map, registry=ctx.registry;
    const style=document.createElement('style');
    style.textContent=`.result-controls{position:absolute;bottom:32px;left:12px;z-index:25;background:#14262ff2;color:#def1ea;border:1px solid #668d86;border-radius:10px;max-width:min(325px,calc(100% - 24px));font:12px/1.5 system-ui;box-shadow:0 8px 24px #0005}.result-controls summary{padding:10px 13px;cursor:pointer;font-weight:650}.result-controls-body{padding:0 12px 12px;display:grid;gap:8px}.result-controls label{display:grid;gap:4px}.result-controls select,.result-controls button{font:inherit;color:inherit;background:#213943;border:1px solid #52746f;border-radius:5px;padding:6px;max-width:100%;min-width:0}.result-controls input[type=range]{width:100%;accent-color:#82dcbb}.result-controls button{cursor:pointer}.result-controls small{color:#a9c9c1}.result-swipe-map{position:absolute;inset:0;pointer-events:none;z-index:4}.result-swipe-line{position:absolute;top:0;bottom:0;width:2px;background:#e8fff5;pointer-events:none;z-index:5;box-shadow:0 0 5px #000}.result-swipe-caption{position:absolute;top:75px;left:50%;transform:translateX(-50%);z-index:6;pointer-events:none;padding:6px 10px;background:#122d37e8;color:#ecfff9;font:11px system-ui;border-radius:6px;max-width:80%;text-align:center}`;
    document.head.append(style);
    const box=document.createElement('details');box.className='result-controls';
    box.innerHTML='<summary>Layers · Opacity & swipe</summary><div class="result-controls-body"><label>Result / layer (right side)<select data-target aria-label="Layer to control"></select></label><label>Opacity <output data-value>100%</output><input data-opacity type="range" min="0" max="100" value="100" aria-label="Selected layer opacity"></label><label>Compare left side with<select data-left aria-label="Left comparison layer"></select></label><button type="button" data-swipe>Start swipe comparison</button><label>Swipe divider<input data-divider type="range" min="0" max="100" value="50" aria-label="Swipe divider position"></label><small data-status>Controls affect display only; scientific raster values stay unchanged.</small></div>';
    map.getContainer().append(box);
    const q=s=>box.querySelector(s),target=q('[data-target]'),left=q('[data-left]'),opacity=q('[data-opacity]'),value=q('[data-value]'),status=q('[data-status]'),button=q('[data-swipe]'),divider=q('[data-divider]');
    let comparison=null,comparisonNode=null,line=null,caption=null,original=[],sync=null,signature='',pendingOpacity=null,leftChosen=false;
    const resultName=name=>/^(Analysis |Anomaly result · |Saved layer · |SAM proposal · )/.test(name||'');
    const setStatus=t=>status.textContent=t;
    function stop(){
      if(sync){map.off('move',sync);map.off('resize',sync);sync=null;}
      if(comparison){comparison.remove();comparison=null;}
      comparisonNode?.remove();line?.remove();caption?.remove();comparisonNode=null;line=null;caption=null;
      for(const [id,v] of original)if(map.getLayer(id))map.setLayoutProperty(id,'visibility',v);
      original=[];button.textContent='Start swipe comparison';
    }
    function split(){if(!comparisonNode)return;const n=Number(divider.value);comparisonNode.style.clipPath=`inset(0 ${100-n}% 0 0)`;line.style.left=n+'%';}
    function currentOpacity(){
      const reg=registry[target.value];if(!reg)return 1;
      for(const id of reg.layerIds||[]){const l=map.getLayer(id);if(l?.type==='raster')return Number(map.getPaintProperty(id,'raster-opacity')??1);}
      return ctx.assetInfo[target.value]?.style?.opacity??1;
    }
    function refresh(){
      const records=Object.entries(registry).filter(([,r])=>(r.layerIds||[]).some(id=>map.getLayer(id)));
      const next=records.map(([id,r])=>id+':'+r.name).join('|');
      if(next===signature)return;signature=next;stop();const selected=target.value,previous=left.value;
      target.replaceChildren();left.replaceChildren(new Option('Underlying map / basemap',''));
      for(const [id,r] of records.sort((a,b)=>Number(resultName(b[1].name))-Number(resultName(a[1].name)))){
        const text=r.name||id;target.add(new Option(text,id));left.add(new Option(text,id));
      }
      if(records.some(([id])=>id===selected))target.value=selected;
      if(records.some(([id])=>id===previous))left.value=previous;
      else if(!leftChosen){const source=records.find(([,r])=>(r.name||'').startsWith('Original image · '));if(source)left.value=source[0];}
      // Keep the user's open/closed choice when layers or map styles change.
      box.hidden=!records.length;updateOpacity();
    }
    function updateOpacity(){const n=Math.round(currentOpacity()*100);opacity.value=n;value.value=n+'%';}
    target.onchange=()=>{stop();updateOpacity();};left.onchange=()=>{leftChosen=true;stop();};
    opacity.oninput=()=>{
      const id=target.value;if(!registry[id])return;
      const n=Number(opacity.value)/100;value.value=opacity.value+'%';
      ctx.handlers.set_opacity({asset_id:id,opacity:n});
      if(comparison && left.value===id)for(const lid of registry[id].layerIds||[])if(comparison.getLayer(lid)?.type==='raster')comparison.setPaintProperty(lid,'raster-opacity',n);
    };
    opacity.onchange=async()=>{
      const id=target.value,n=Number(opacity.value)/100;
      const info=ctx.assetInfo[id]||{};
      let saved=info.style||{};if(typeof saved==='string'){try{saved=JSON.parse(saved);}catch{saved={};}}
      // update_style replaces a whole style: preserve its other properties.
      try{
        pendingOpacity=id;
        const response=await fetch(ctx.baseUrl+'/api/maps/'+encodeURIComponent(ctx.mapId)+'/events',{
          method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({type:'update_style',data:{asset_id:id,style:{...saved,opacity:n}},user_session_id:ctx.getUserSession()})});
        const result=await response.json();if(!response.ok||result.error)throw Error(result.error||'Could not save display opacity');
        setStatus('Opacity saved. Swipe is a temporary comparison view.');
      }catch(e){setStatus(e.message);}finally{pendingOpacity=null;}
    };
    divider.oninput=split;
    button.onclick=()=>{
      if(comparison){stop();return;}
      if(!target.value || target.value===left.value){setStatus('Choose two different layers, or compare with the underlying map.');return;}
      try{
        const snapshot=map.getStyle();const rightIds=new Set(registry[target.value]?.layerIds||[]),leftIds=new Set(registry[left.value]?.layerIds||[]);
        const outputIds=new Set(Object.values(registry).filter(r=>resultName(r.name)).flatMap(r=>r.layerIds||[]));
        for(const id of [...rightIds,...leftIds])outputIds.add(id);
        snapshot.layers=snapshot.layers.filter(l=>l.type!=='custom'&&(!outputIds.has(l.id)||leftIds.has(l.id))).map(l=>leftIds.has(l.id)?{...l,layout:{...l.layout,visibility:'visible'}}:l);
        for(const id of outputIds){if(!map.getLayer(id))continue;original.push([id,map.getLayoutProperty(id,'visibility')||'visible']);map.setLayoutProperty(id,'visibility',rightIds.has(id)?'visible':'none');}
        comparisonNode=document.createElement('div');comparisonNode.className='result-swipe-map';comparisonNode.setAttribute('aria-hidden','true');map.getContainer().append(comparisonNode);
        comparison=new maplibregl.Map({container:comparisonNode,style:snapshot,center:map.getCenter(),zoom:map.getZoom(),bearing:map.getBearing(),pitch:map.getPitch(),interactive:false,attributionControl:false,fadeDuration:0});
        comparison.on('error',()=>setStatus('A comparison source failed to load. Check the source service; stop swipe to return to the map.'));
        line=document.createElement('div');line.className='result-swipe-line';map.getContainer().append(line);
        caption=document.createElement('div');caption.className='result-swipe-caption';caption.textContent='← '+(left.selectedOptions[0]?.text||'Basemap')+' | '+(target.selectedOptions[0]?.text||'Result')+' →';map.getContainer().append(caption);
        sync=()=>{comparison?.resize();comparison?.jumpTo({center:map.getCenter(),zoom:map.getZoom(),bearing:map.getBearing(),pitch:map.getPitch()});};
        map.on('move',sync);map.on('resize',sync);split();button.textContent='Stop swipe comparison';setStatus('Drag the swipe slider. Pan and zoom remain synchronized. Stop swipe before editing map layers.');
      }catch(e){stop();setStatus(e.message);}
    };
    window.addEventListener('esip:assetschanged',e=>{
      // Close the temporary style clone on structural changes, not slider updates.
      if(!['update_style','set_opacity'].includes(e.detail?.type)){stop();signature='';}
      if(!pendingOpacity)refresh();
    });
    map.on('style.load',()=>{stop();signature='';refresh();});
    map.on('idle',refresh);window.addEventListener('pagehide',stop);refresh();
  }
  if(window.__esipInternals)start(window.__esipInternals);
  else window.addEventListener('esip:internals-ready',()=>start(window.__esipInternals),{once:true});
})();
