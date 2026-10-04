(()=>{
'use strict';
let catalog;
const format=n=>Number(n).toLocaleString();
const methods=['sequential_no_loop','sequential_loop','exhaustive'];
const labels={'sequential_no_loop':'Sequential · loop off','sequential_loop':'Sequential · loop on',exhaustive:'Exhaustive'};
const shared=document.getElementById('matching-gs-frame');
const panels=[];
function panel(id,method){
 const root=document.getElementById(id);
 root.innerHTML='<label>Method <select class="gs-method"></select></label><label>Component <select class="gs-component"></select></label><label>Rendered viewpoint <select class="gs-view"></select></label><p class="gs-result-status" role="status"></p><a class="gs-result-image-link" target="_blank" rel="noopener"><img class="gs-result-image" alt="Full trained gsplat prediction" hidden></a><p class="gs-result-stats"></p><a class="gs-local-link" target="_blank" rel="noopener">Open full 3D on this computer ↗</a>';
 const state={root,method:root.querySelector('.gs-method'),component:root.querySelector('.gs-component'),view:root.querySelector('.gs-view')};
 for(const key of methods)state.method.add(new Option(labels[key],key));
 state.method.value=method;
 state.method.onchange=()=>setComponents(state);
 state.component.onchange=()=>{setViews(state);synchronize();};
 state.view.onchange=()=>{if([...shared.options].some(o=>o.value===state.view.value)){shared.value=state.view.value;setSharedView();}else show(state);};
 panels.push(state);setComponents(state);
}
function setComponents(state,preferred){
 const entries=Object.entries(catalog.models).filter(([,m])=>m.method===state.method.value);
 state.component.replaceChildren(...entries.map(([key,m])=>new Option(`Component ${m.component} · ${m.registered_images} views${m.primary_component?' · largest':''}`,key)));
 state.component.value=entries.some(([key])=>key===preferred)?preferred:entries.find(([,m])=>m.primary_component)?.[0]||entries[0]?.[0];setViews(state);synchronize();
}
function setViews(state){
 const m=catalog.models[state.component.value];
 state.view.replaceChildren(...m.views.map(v=>new Option(v.frame,v.frame)));
 if(m.views.some(v=>v.frame===shared.value))state.view.value=shared.value;
 show(state);
}
function synchronize(){
 if(panels.length!==2)return;
 const a=catalog.models[panels[0].component.value],b=catalog.models[panels[1].component.value],previous=shared.value;
 const frames=a.views.map(v=>v.frame).filter(f=>b.views.some(v=>v.frame===f));
 shared.replaceChildren(...(frames.length?frames.map(f=>new Option(f,f)):[new Option('No shared rendered frame','')]));
 shared.disabled=!frames.length;
 if(frames.includes(previous))shared.value=previous;
 if(frames.length)setSharedView();
}
function setSharedView(){panels.forEach(state=>{if(catalog.models[state.component.value].views.some(v=>v.frame===shared.value))state.view.value=shared.value;show(state);});}
function show(state){
 const m=catalog.models[state.component.value],v=m.views.find(v=>v.frame===state.view.value),img=state.root.querySelector('img'),link=state.root.querySelector('.gs-result-image-link'),stats=state.root.querySelector('.gs-result-stats'),status=state.root.querySelector('.gs-result-status'),local=state.root.querySelector('.gs-local-link');
 img.hidden=!v;link.removeAttribute('href');local.hidden=m.state!=='complete';
 if(!v){status.textContent=m.state==='complete'?'No saved prediction for this component.':`Training ${m.state.replaceAll('_',' ')} · results will appear after the run finishes.`;stats.textContent=`${m.registered_images} registered views · ${m.training_images} training / ${m.validation_images} held out`;return;}
 img.src=v.render;img.alt=`${m.label}, component ${m.component}, full trained prediction at ${v.frame}`;link.href=v.render;
 status.textContent=`${v.frame} · ${v.width} × ${v.height} · full-model SH3 render · lossless WebP`;
 stats.textContent=`${format(m.gaussian_count)} valid exported Gaussians${m.excluded_invalid_gaussians?` (${format(m.excluded_invalid_gaussians)} non-finite rows omitted by gsplat)`:''} · ${m.training_images} training / ${m.validation_images} held-out images · ${m.seconds.toFixed(3)} s training`;
 local.href=`${catalog.local_viewer}?model=${encodeURIComponent(state.component.value)}`;
}
function table(){
 const body=document.getElementById('matching-gs-metrics');body.replaceChildren();
 for(const m of Object.values(catalog.models)){
  const row=document.createElement('tr');
  const metrics=m.metrics;
  const values=[`${m.label} · component ${m.component}${m.primary_component?' (largest)':''}`,m.state,m.registered_images,`${m.training_images} / ${m.validation_images}`,m.gaussian_count?`${format(m.gaussian_count)}${m.excluded_invalid_gaussians?` / ${format(m.trained_gaussian_count)} trained`:''}`:'—',metrics?metrics.psnr.toFixed(3):'—',metrics?metrics.ssim.toFixed(4):'—',metrics?metrics.lpips.toFixed(4):'—',m.seconds!==null?`${m.seconds.toFixed(3)} s`:'—'];
  for(const value of values){const td=document.createElement('td');td.textContent=value;row.append(td);}body.append(row);
 }
 const n=Object.values(catalog.models).filter(m=>m.state==='complete').length;
 document.getElementById('matching-gs-progress').textContent=`${n} / ${Object.keys(catalog.models).length} component training runs complete · RTX 4090 · 30,000 steps each`;
 const common=catalog.common_validation;
 if(common){
  document.getElementById('matching-gs-common').hidden=false;
  document.getElementById('matching-gs-common-description').textContent=`The same ${common.frame_count} held-out source frames in all three main models. Scores are recalculated on the saved 8-bit renders; each model uses its own camera undistortion. These measure appearance agreement, not independent geometric accuracy.`;
  const rows=document.getElementById('matching-gs-common-metrics');rows.replaceChildren();
  for(const method of methods){
   const row=document.createElement('tr'),key=Object.keys(catalog.models).find(k=>catalog.models[k].method===method&&catalog.models[k].primary_component),score=common.models[key];
   for(const value of [labels[method],score?score.frames:'Pending',score?score.mean.psnr.toFixed(3):'—',score?score.mean.ssim.toFixed(4):'—',score?score.mean.lpips.toFixed(4):'—']){const td=document.createElement('td');td.textContent=value;row.append(td);}rows.append(row);
  }
 }
}
shared.onchange=setSharedView;
document.getElementById('matching-gs-swap').onclick=()=>{const a=panels[0].method.value,b=panels[1].method.value,ak=panels[0].component.value,bk=panels[1].component.value;panels[0].method.value=b;panels[1].method.value=a;setComponents(panels[0],bk);setComponents(panels[1],ak);};
fetch('matching-gsplat/index.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('Gaussian comparison results are not available yet.');return r.json();}).then(data=>{
 catalog=data;shared.replaceChildren(...data.shared_frames.map(f=>new Option(f,f)));
 panel('matching-gs-a','sequential_no_loop');panel('matching-gs-b','sequential_loop');
 if(data.models.sequential_no_loop_component_2?.state==='complete')setComponents(panels[0],'sequential_no_loop_component_2');
 table();
}).catch(error=>{document.getElementById('matching-gs-progress').textContent=error.message;});
})();
