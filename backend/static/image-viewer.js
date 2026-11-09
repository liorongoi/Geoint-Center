/* Image coordinates and one shared transform keep both display layers registered. */
window.showImageComparison=async function(display,runId,index){
  index=Number(index);
  const host=document.getElementById('imageComparison');
  const previous=window.resultImageViewer;
  const same=previous?.runId===runId&&previous?.index===index;
  previous?.destroy();
  host.innerHTML=`<div class="comparison-tools"><div><button type="button" data-view="out" aria-label="Zoom out">−</button><button type="button" data-view="fit">Fit image</button><button type="button" data-view="in" aria-label="Zoom in">+</button><output id="comparisonZoom"></output></div><label>Result opacity <input id="comparisonOpacity" type="range" min="0" max="100" value="100"><output id="comparisonOpacityValue">100%</output></label><div class="sam-viewer-tools"><button type="button" data-sam-view="1">SAM + point</button><button type="button" data-sam-view="0">SAM − point</button><button type="button" data-sam-view="box">Draw SAM box</button><button type="button" data-sam-view="inspect">Inspect / pan</button></div><button type="button" data-view="fullscreen">Expand</button></div><div class="comparison-stage" tabindex="0" aria-label="Image comparison. Scroll to zoom, drag to pan, drag divider to compare."><div class="comparison-viewport"><div class="comparison-plane"><img class="comparison-source" alt="Original image RGB display"><img class="comparison-overlay" alt="Analysis result aligned with original"></div></div><span class="comparison-original-label">Original</span><span class="comparison-result-label">Result</span><div class="comparison-selection" hidden></div><div class="comparison-divider" role="slider" tabindex="0" aria-label="Swipe divider" aria-valuemin="0" aria-valuemax="100" aria-valuenow="50"><span>↔</span></div><div class="comparison-loading" role="status">Loading original & result…</div></div><footer class="comparison-footer"><span>Original RGB below · result above · display copies only</span><label>Swipe <input id="comparisonSwipe" type="range" min="0" max="100" value="50" aria-label="Swipe position"></label></footer>`;
  const stage=host.querySelector('.comparison-stage'),plane=host.querySelector('.comparison-plane'),overlay=host.querySelector('.comparison-overlay'),viewport=host.querySelector('.comparison-viewport'),divider=host.querySelector('.comparison-divider'),selection=host.querySelector('.comparison-selection');
  const body=document.createElement('div');body.className='comparison-body';
  stage.before(body);body.append(stage);
  const legend=document.createElement('aside');legend.className='comparison-expanded-legend';legend.setAttribute('aria-label','Classification legend');
  const review=state.resultReview;
  legend.innerHTML=`<h2>Classification legend</h2>${resultLegend(review?.output||{},'consensus')}${review?.layer&&!['consensus','reviewed'].includes(review.layer)?`<h3>Current display</h3>${resultLegend(review.output,review.layer)}`:''}<p>Class colors apply to the land-cover result. Opacity changes its appearance.</p>`;
  body.append(legend);
  const onFullscreen=()=>{host.querySelector('[data-view="fullscreen"]').textContent=document.fullscreenElement===host?'Exit expanded view':'Expand';};
  document.addEventListener('fullscreenchange',onFullscreen);
  const model={runId,index,scale:same?previous.scale:1,tx:same?previous.tx:0,ty:same?previous.ty:0,split:same?previous.split:50,opacity:same?previous.opacity:100,boxMode:false};
  let width=display.width,height=display.height,fitScale=1,gesture=null;
  const promptLayer=document.createElement('div');promptLayer.className='sam-image-prompts';
  model.samMarks=[];model.samBox=null;
  function renderPrompts(){promptLayer.innerHTML=model.samMarks.map(p=>`<span class="sam-point-mark ${p.label===0?'exclude':''}" style="left:${p.x*100}%;top:${p.y*100}%;transform:translate(-50%,-50%) scale(${1/model.scale})">${p.label===0?'−':'+'}</span>`).join('')+(model.samBox?`<span class="sam-box-mark" style="left:${Math.min(model.samBox[0].x,model.samBox[1].x)*100}%;top:${Math.min(model.samBox[0].y,model.samBox[1].y)*100}%;width:${Math.abs(model.samBox[1].x-model.samBox[0].x)*100}%;height:${Math.abs(model.samBox[1].y-model.samBox[0].y)*100}%"></span>`:'');}
  function render(){
    plane.style.width=width+'px';plane.style.height=height+'px';plane.style.transform=`translate(${model.tx}px,${model.ty}px) scale(${model.scale})`;
    viewport.style.clipPath=`inset(0 ${100-model.split}% 0 0)`;
    // The original occupies its own un-clipped plane with exactly the same transform.
    originalPlane.style.width=width+'px';originalPlane.style.height=height+'px';originalPlane.style.transform=plane.style.transform;
    promptLayer.style.width=width+'px';promptLayer.style.height=height+'px';promptLayer.style.transform=plane.style.transform;
    renderPrompts();
    overlay.style.opacity=model.opacity/100;divider.style.left=model.split+'%';divider.setAttribute('aria-valuenow',Math.round(model.split));
    host.querySelector('#comparisonSwipe').value=model.split;host.querySelector('#comparisonZoom').textContent=Math.round(model.scale/fitScale*100)+'%';
    host.querySelector('#comparisonOpacity').value=model.opacity;host.querySelector('#comparisonOpacityValue').textContent=Math.round(model.opacity)+'%';
  }
  const originalPlane=document.createElement('div');originalPlane.className='comparison-plane comparison-base';
  originalPlane.append(host.querySelector('.comparison-source'));stage.prepend(originalPlane);stage.append(promptLayer);
  function fit(){fitScale=Math.min(stage.clientWidth/width,stage.clientHeight/height)*.96;model.scale=fitScale;model.tx=(stage.clientWidth-width*model.scale)/2;model.ty=(stage.clientHeight-height*model.scale)/2;render();}
  function zoom(factor,cx=stage.clientWidth/2,cy=stage.clientHeight/2){const old=model.scale,next=Math.min(fitScale*32,Math.max(fitScale*.5,old*factor));model.tx=cx-(cx-model.tx)*next/old;model.ty=cy-(cy-model.ty)*next/old;model.scale=next;render();}
  function position(e){const r=stage.getBoundingClientRect();return {x:e.clientX-r.left,y:e.clientY-r.top};}
  function fraction(p){return {x:(p.x-model.tx)/model.scale/width,y:(p.y-model.ty)/model.scale/height};}
  async function location(p){const f=fraction(p);if(f.x<0||f.x>1||f.y<0||f.y>1)throw new Error('Choose a location inside the image.');return api(`/api/results/${encodeURIComponent(runId)}/${index}/viewer-location?x=${f.x}&y=${f.y}`);}
  stage.onwheel=e=>{e.preventDefault();const p=position(e);zoom(e.deltaY<0?1.15:1/1.15,p.x,p.y);};
  stage.onpointerdown=e=>{if(e.button!==0)return;const p=position(e);gesture={kind:e.target.closest('.comparison-divider')?'swipe':model.boxMode?'box':model.samPointCallback?'sam-point':'pan',start:p,tx:model.tx,ty:model.ty,moved:false};stage.setPointerCapture(e.pointerId);e.preventDefault();};
  stage.onpointermove=e=>{if(!gesture)return;const p=position(e),g=gesture;g.moved=g.moved||Math.hypot(p.x-g.start.x,p.y-g.start.y)>4;
    if(g.kind==='swipe'){model.split=Math.max(0,Math.min(100,p.x/stage.clientWidth*100));render();}
    else if(g.kind==='pan'){model.tx=g.tx+p.x-g.start.x;model.ty=g.ty+p.y-g.start.y;render();}
    else if(g.kind==='box'){selection.hidden=false;Object.assign(selection.style,{left:Math.min(g.start.x,p.x)+'px',top:Math.min(g.start.y,p.y)+'px',width:Math.abs(p.x-g.start.x)+'px',height:Math.abs(p.y-g.start.y)+'px'});}
  };
  stage.onpointerup=async e=>{const g=gesture;gesture=null;if(!g)return;stage.releasePointerCapture(e.pointerId);const p=position(e);
    try{if(g.kind==='box'&&g.moved){const pts=[g.start,{x:p.x,y:g.start.y},p,{x:g.start.x,y:p.y}];const coords=await Promise.all(pts.map(location));const ring=coords.map(c=>[c.longitude,c.latitude]);ring.push(ring[0]);const ctx=state.resultReview;if(model.samBoxCallback){model.samBox=[fraction(g.start),fraction(p)];model.samBoxCallback({type:'Polygon',coordinates:[ring]});model.samBoxCallback=null;render();}else if(ctx?.runId===runId&&ctx.index===index){ctx.scope='area';ctx.geometry={type:'Polygon',coordinates:[ring]};ctx.sourceModel=null;updateReviewGeometryStatus();toast('Image region selected. Choose a class and save.');}model.boxMode=false;stage.classList.remove('selecting');}
      else if(g.kind==='sam-point'&&!g.moved){const c=await location(p);model.samMarks.push({...fraction(p),label:model.samPointLabel});model.samPointCallback(c);render();}
      else if(g.kind==='pan'&&!g.moved){const c=await location(p);await inspectAnalysisPixel(c.longitude,c.latitude);}
    }catch(error){toast(error.message);}finally{selection.hidden=true;}
  };
  stage.onpointercancel=()=>{gesture=null;selection.hidden=true;};
  divider.onkeydown=e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();model.split=e.key==='Home'?0:e.key==='End'?100:Math.max(0,Math.min(100,model.split+(e.key==='ArrowRight'?2:-2)));render();};
  host.querySelector('#comparisonSwipe').oninput=e=>{model.split=Number(e.target.value);render();};
  host.querySelector('#comparisonOpacity').oninput=e=>{model.opacity=Number(e.target.value);render();};
  host.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>{const action=button.dataset.view;if(action==='fit')fit();else if(action==='in')zoom(1.3);else if(action==='out')zoom(1/1.3);else if(document.fullscreenElement)document.exitFullscreen();else host.requestFullscreen?.().catch(e=>toast(e.message));});
  const observer=new ResizeObserver(()=>{if(!same)fit();else render();});observer.observe(stage);
  model.resize=()=>render();model.destroy=()=>{observer.disconnect();document.removeEventListener('fullscreenchange',onFullscreen);};model.selectBox=(callback=null)=>{model.samPointCallback=null;model.samBoxCallback=callback;model.boxMode=true;stage.classList.add('selecting');toast('Drag a rectangle on the image.');};model.setSamPoint=(label,callback)=>{model.boxMode=false;model.samBoxCallback=null;model.samPointLabel=label;model.samPointCallback=callback;stage.classList.add('selecting');};model.clearSam=()=>{model.samMarks=[];model.samBox=null;model.samPointCallback=null;model.samBoxCallback=null;model.boxMode=false;stage.classList.remove('selecting');render();};window.resultImageViewer=model;
  host.querySelectorAll('[data-sam-view]').forEach(button=>button.onclick=()=>{const action=button.dataset.samView;if(action==='inspect'){model.samPointCallback=null;model.boxMode=false;stage.classList.remove('selecting');}else{const panel=$('segmentationPanel');if(panel)panel.open=true;if(action==='box')$('segmentDrawBox')?.click();else (action==='1'?$('segmentPointMode'):$('segmentExcludeMode'))?.click();}});
  const load=(img,url)=>new Promise((resolve,reject)=>{img.onload=resolve;img.onerror=()=>reject(new Error('Could not load the saved image comparison.'));img.src=url;});
  try{await Promise.all([load(originalPlane.querySelector('img'),display.source),load(overlay,display.overlay)]);if(same){fitScale=Math.min(stage.clientWidth/width,stage.clientHeight/height)*.96;render();}else fit();host.querySelector('.comparison-loading').hidden=true;}catch(e){host.querySelector('.comparison-loading').textContent=e.message;throw e;}
};
