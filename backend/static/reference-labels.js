/* Human references are available before the first inference run. */
window.referenceLabelEditor=(()=>{
  let objectLabels=[],preview=null,vertices=[],finished=false,tool='rectangle',scale=1,tx=0,ty=0,gesture=null,generation=0;
  const el=id=>document.getElementById(id),ns='http://www.w3.org/2000/svg';
  const status=text=>el('referenceStatus').textContent=text;
  function context(){const mission=state.missions.find(m=>m.id===state.currentMissionId),job=mission?.jobs?.find(j=>j.id===el('referenceJob').value);return {mission,job};}
  function invalidate(){generation++;preview=null;vertices=[];finished=false;el('referenceRaster').removeAttribute('href');el('referenceEmpty').hidden=false;el('openReferenceImage').disabled=false;draw();}
  function update(mission,job){
    const select=el('referenceJob'),old=select.value;select.innerHTML=(mission.jobs||[]).map(j=>`<option value="${escapeHtml(j.id)}">${escapeHtml(j.name)}</option>`).join('');select.value=job?.id||old;
    const images=el('referenceImage'),previous=images.value;images.innerHTML=(job?.images||[]).map(i=>`<option value="${escapeHtml(i.id)}">${escapeHtml(i.name||i.id)}</option>`).join('');
    if([...images.options].some(o=>o.value===previous))images.value=previous;
    if(preview&&(preview.mission_id!==mission.id||preview.job_id!==select.value||preview.image_id!==images.value))invalidate();
    el('referenceClassNames').innerHTML=[...new Set(state.groundTruth.filter(l=>l.mission_id===mission.id&&l.job_id===job?.id).map(l=>l.label))].map(name=>`<option value="${escapeHtml(name)}">`).join('');draw();
  }
  function fit(){if(!preview)return;const box=el('referenceCanvas');scale=Math.min(box.clientWidth/preview.width,box.clientHeight/preview.height)*.96;tx=(box.clientWidth-preview.width*scale)/2;ty=(box.clientHeight-preview.height*scale)/2;draw();}
  function draw(){
    el('referencePlane').setAttribute('transform',`translate(${tx} ${ty}) scale(${scale})`);
    const target=el('referenceDrawing');target.replaceChildren();
    if(preview&&vertices.length){const shape=document.createElementNS(ns,tool==='point'?'circle':'polygon');
      if(tool==='point'){shape.setAttribute('cx',vertices[0][0]*preview.width);shape.setAttribute('cy',vertices[0][1]*preview.height);shape.setAttribute('r',5/scale);}
      else shape.setAttribute('points',vertices.map(([x,y])=>`${x*preview.width},${y*preview.height}`).join(' '));
      shape.setAttribute('fill','#8ae3ce55');shape.setAttribute('stroke','#b4ffed');shape.setAttribute('stroke-width','2');target.append(shape);
    }
    const saved=el('referenceSaved');saved.replaceChildren();
    if(preview){for(const label of [...state.groundTruth,...objectLabels].filter(l=>l.source?.preview_id===preview.id)){
      const shape=document.createElementNS(ns,label.source.kind==='point'?'circle':'polygon'),pts=label.source.vertices;
      if(label.source.kind==='point'){shape.setAttribute('cx',pts[0][0]*preview.width);shape.setAttribute('cy',pts[0][1]*preview.height);shape.setAttribute('r',4/scale);}else shape.setAttribute('points',pts.map(([x,y])=>`${x*preview.width},${y*preview.height}`).join(' '));
      shape.setAttribute('fill',label.role==='validation'?'#e2b47144':'#7dbdec44');shape.setAttribute('stroke',label.role==='validation'?'#e2b471':'#7dbdec');shape.setAttribute('stroke-width','2');
      const title=document.createElementNS(ns,'title');title.textContent=`${label.label} · ${label.role}`;shape.append(title);saved.append(shape);
    }}
    el('saveImageReference').disabled=!preview||!finished||!vertices.length;
  }
  async function open(){const {mission,job}=context(),imageId=el('referenceImage').value;if(!mission||!job||!imageId){status('Add an image to this job first.');return;}
    invalidate();const request=++generation;status('Reading source pixels and preparing the image…');el('openReferenceImage').disabled=true;
    try{const data=await api(`/api/reference-images?${new URLSearchParams({mission_id:mission.id,job_id:job.id,image_id:imageId})}`);if(request!==generation)return;
      await new Promise((resolve,reject)=>{const img=new Image();img.onload=resolve;img.onerror=()=>reject(new Error('Could not load the source display'));img.src=data.source;});if(request!==generation)return;
      const loadedObjects=(await api('/api/object-labels?'+new URLSearchParams({mission_id:mission.id,job_id:job.id,image_id:imageId}))).labels||[];if(request!==generation)return;preview=data;objectLabels=loadedObjects;const image=el('referenceRaster');image.setAttribute('href',data.source);image.setAttribute('width',data.width);image.setAttribute('height',data.height);el('referenceEmpty').hidden=true;fit();status('Draw a rectangle, click polygon vertices then Finish polygon, or mark a point. Scroll to zoom; select Pan to move.');
    }catch(e){status(e.message);}finally{if(request===generation)el('openReferenceImage').disabled=false;}
  }
  function position(e){const r=el('referenceCanvas').getBoundingClientRect();return {x:e.clientX-r.left,y:e.clientY-r.top};}
  function fraction(p){return [(p.x-tx)/scale/preview.width,(p.y-ty)/scale/preview.height];}
  const inside=p=>p.every(v=>v>=0&&v<=1);
  document.addEventListener('DOMContentLoaded',()=>{
    el('referenceJob').onchange=()=>{el('groundTruthJob').value=el('referenceJob').value;renderTrainingPanel();};el('referenceImage').onchange=invalidate;el('openReferenceImage').onclick=open;
    el('referenceImportShortcut').onclick=()=>el('vectorImportFiles').scrollIntoView({behavior:'smooth',block:'center'});
    document.querySelectorAll('[data-label-tool]').forEach(b=>b.onclick=()=>{tool=b.dataset.labelTool;el('saveImageReference').textContent=tool==='rotated_box'?'Save rotated object label':'Save reference';el('referenceClass').placeholder=tool==='rotated_box'?'e.g. ship, plane, storage-tank':'e.g. Water, Forest, Built-up';vertices=[];finished=false;document.querySelectorAll('[data-label-tool]').forEach(x=>x.setAttribute('aria-pressed',x===b));draw();status(tool==='rotated_box'?'Click the first corner, the second corner along an edge, then a third point to set perpendicular height. Each box is one object instance.':tool==='polygon'?'Click each vertex, then Finish polygon.':'Draw a region, choose its class, then Save reference.');});
    el('fitReferenceImage').onclick=fit;el('clearReferenceRegion').onclick=()=>{vertices=[];finished=false;draw();status('Drawing cleared. Saved labels are unchanged.');};
    el('finishReferencePolygon').onclick=()=>{if(tool!=='polygon'||vertices.length<3){status('Select Polygon and draw at least three vertices.');return;}finished=true;draw();status('Polygon ready. Assign its class and save.');};
    const canvas=el('referenceCanvas');canvas.onwheel=e=>{if(!preview)return;e.preventDefault();const p=position(e),old=scale;scale=Math.min(64,Math.max(.05,scale*(e.deltaY<0?1.15:1/1.15)));tx=p.x-(p.x-tx)*scale/old;ty=p.y-(p.y-ty)*scale/old;draw();};
    canvas.onpointerdown=e=>{if(!preview||e.button!==0)return;const p=position(e),f=fraction(p);if(tool!=='pan'&&!inside(f)){status('Draw inside the source image.');return;}canvas.setPointerCapture(e.pointerId);e.preventDefault();gesture={start:p,first:f,tx,ty};
      if(tool==='rotated_box'){
        if(finished)vertices=[];
        if(vertices.length<2){vertices.push(f);finished=false;}
        else{const a=vertices[0].map((v,i)=>v*[preview.width,preview.height][i]),b=vertices[1].map((v,i)=>v*[preview.width,preview.height][i]),c=f.map((v,i)=>v*[preview.width,preview.height][i]);const dx=b[0]-a[0],dy=b[1]-a[1],size=Math.hypot(dx,dy);if(size<1){vertices=[];status('Edge is too short. Draw again.');draw();return;}const nx=-dy/size,ny=dx/size,height=(c[0]-a[0])*nx+(c[1]-a[1])*ny;const pts=[a,b,[b[0]+height*nx,b[1]+height*ny],[a[0]+height*nx,a[1]+height*ny]].map(p=>[p[0]/preview.width,p[1]/preview.height]);if(!pts.every(inside)){status('The rotated box extends outside the image. Choose a smaller height.');draw();return;}vertices=pts;finished=Math.abs(height)>=1;status('Rotated object box ready. Assign its class and save; it stays separate from LULC references.');}draw();
      }else if(tool==='point'){vertices=[f];finished=true;draw();}else if(tool==='polygon'){if(finished)vertices=[];vertices.push(f);finished=false;draw();}else if(tool==='rectangle'){vertices=[];finished=false;draw();}
    };
    canvas.onpointermove=e=>{if(!gesture||!preview)return;const p=position(e);if(tool==='pan'){tx=gesture.tx+p.x-gesture.start.x;ty=gesture.ty+p.y-gesture.start.y;draw();}else if(tool==='rectangle'){const end=fraction(p).map(v=>Math.max(0,Math.min(1,v))),a=gesture.first;vertices=[a,[end[0],a[1]],end,[a[0],end[1]]];draw();}};
    canvas.onpointerup=e=>{if(!gesture)return;if(tool==='rectangle'){finished=vertices.length===4&&Math.abs(vertices[0][0]-vertices[2][0])>.0001&&Math.abs(vertices[0][1]-vertices[2][1])>.0001;draw();}gesture=null;canvas.releasePointerCapture(e.pointerId);};canvas.onpointercancel=()=>{gesture=null;};
    canvas.onkeydown=e=>{if(e.key==='Escape'){vertices=[];finished=false;draw();}if(e.key==='Enter'&&tool==='polygon')el('finishReferencePolygon').click();};
    el('saveImageReference').onclick=async()=>{if(!preview||!finished)return;const label=el('referenceClass').value.trim();if(!label){status('Enter the land-cover class before saving.');el('referenceClass').focus();return;}const button=el('saveImageReference');button.disabled=true;status('Saving your reference…');
      try{const record=await api(`/api/reference-images/${preview.id}/${tool==='rotated_box'?'object-labels':'labels'}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({label,vertices,kind:tool,role:el('referenceRole').value,confidence:Number(el('referenceConfidence').value),notes:el('referenceNotes').value.trim()})});if(tool==='rotated_box')objectLabels.unshift(record);else state.groundTruth.unshift(record);vertices=[];finished=false;renderTrainingPanel();status(`Saved ${record.label} as ${record.role==='validation'?'validation only':'a training reference'}. Draw another class or return to Images & models.`);if(tool!=='rotated_box')acceptAutomaticTrainingRuns(record.automatic_training);}catch(e){status(e.message);draw();}
    };
    new ResizeObserver(()=>{if(preview)fit();}).observe(canvas);
  });
  return {update};
})();
