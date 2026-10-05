(()=>{
'use strict';
const $=(root,selector)=>root.querySelector(selector);
const fmt=value=>Number(value||0).toLocaleString();
const bufferCache=new Map();
const cameraLabel=key=>({
 '1x':'Full resolution','2x':'2× downsample','4x':'4× downsample','8x':'8× downsample'
}[key]||key);
async function json(path){const response=await fetch(path,{cache:'no-store'});if(!response.ok)throw Error(`Could not load ${path}`);return response.json()}
function cloudEntry(id,source,label,cloud,inputImages){
 if(!cloud||!cloud.url)return null;
 const dense=cloud.kind==='dense';
 return {id,kind:'cloud',source,label,cloud,inputImages,dense};
}
function collectClouds(base,experiments){
 const entries=[];
 for(const [key,dataset] of Object.entries(base.scales)){
  const resolution=cameraLabel(key),prefix=`500 frames · ${resolution}`;
  for(const component of dataset.components||[]){
   const item=cloudEntry(`base:${key}:sparse:${component.id}`,`COLMAP baseline · ${resolution}`,`${prefix} · sparse component ${component.id}`,component,dataset.input_images||500);
   if(item)entries.push(item);
  }
  const dense=cloudEntry(`base:${key}:dense`,`COLMAP baseline · ${resolution}`,`${prefix} · dense cloud preview`,dataset.dense,dataset.input_images||500);
  if(dense)entries.push(dense);
 }
 for(const [key,dataset] of Object.entries(experiments.scales)){
  if(key==='baseline_500')continue;
  for(const component of dataset.components||[]){
   const item=cloudEntry(`experiment:${key}:sparse:${component.id}`,dataset.label,`${dataset.label}${dataset.run_id?" · newest 500-frame run":""} · sparse component ${component.id}`,component,dataset.input_images||500);
   if(item)entries.push(item);
  }
  const dense=cloudEntry(`experiment:${key}:dense`,dataset.label,`${dataset.label} · dense cloud preview`,dataset.dense,dataset.input_images||500);
  if(dense)entries.push(dense);
 }
 return entries;
}
function collectGaussians(catalog){
 return Object.entries(catalog).map(([key,model])=>({
  id:`gsplat:${key}`,kind:'gsplat',modelKey:key,source:'gsplat · Attempt 01',
  label:`gsplat · ${cameraLabel(key==='1x'?'1x':key==='8x_component_2'?'8x':key)}${key==='8x_component_2'?' · separate component 2':''}`,
  model
 }));
}
function collectVoxelGaussians(catalog){
 return Object.entries(catalog.models).map(([key,model])=>({
  id:`voxel-gsplat:${key}`,kind:'gsplat',modelKey:key,source:'gsplat · Insta360 4× voxel study',
  label:`gsplat · Insta360 4× · ${model.label}`,
  model:{...model,count:model.gaussian_count,bytes:model.model_bytes,download_bytes:model.model_bytes},
  viewerHref:`voxel-gsplat.html?model=${encodeURIComponent(key)}#voxel-3d`,fullLocal:true
 }));
}
function makePanel(id,side){
 const panel=document.getElementById(id);
 panel.innerHTML=`<div class="compare-panel-head"><p class="compare-side">${side}</p><label>Result<select class="compare-choice"></select></label></div><p class="cloud-status" role="status">Choose a result.</p><canvas class="cloud-canvas" aria-label="Interactive COLMAP point cloud. Use pointer to orbit and scroll to zoom."></canvas><img class="compare-preview" alt="Saved rendered view from a gsplat model" hidden><label class="compare-view-control" hidden>Rendered viewpoint<select></select></label><div class="cloud-controls compare-cloud-controls"><button type="button" class="compare-reset">Reset view</button><label>Point size <input class="compare-size" type="range" min="1" max="5" value="2" step="0.5"></label><label><input class="compare-cameras" type="checkbox" checked>Camera centers</label></div><div class="compare-stats"></div>`;
 const canvas=$(panel,'.cloud-canvas');
 let renderer;
 try{renderer=new window.CloudRenderer(canvas)}catch(error){$(panel,'.cloud-status').textContent=error.message}
 $(panel,'.compare-reset').addEventListener('click',()=>renderer?.reset());
 $(panel,'.compare-size').addEventListener('input',event=>{if(renderer){renderer.pointSize=Number(event.target.value);renderer.draw()}});
 $(panel,'.compare-cameras').addEventListener('change',event=>{if(renderer){renderer.showCameras=event.target.checked;renderer.draw()}});
 return {panel,renderer,token:0};
}
function populateSelect(state,items){
 const select=$(state.panel,'.compare-choice');
 const groups=[['COLMAP point clouds',items.filter(item=>item.kind==='cloud')],['gsplat models',items.filter(item=>item.kind==='gsplat')]];
 select.replaceChildren();
 for(const [label,entries] of groups){if(!entries.length)continue;const group=document.createElement('optgroup');group.label=label;for(const item of entries){group.append(new Option(item.label,item.id))}select.append(group)}
 select.onchange=()=>loadSelection(state,items.find(item=>item.id===select.value));
}
function showCloud(state,item){
 const {panel,renderer}=state,canvas=$(panel,'.cloud-canvas'),preview=$(panel,'.compare-preview'),viewControl=$(panel,'.compare-view-control'),controls=$(panel,'.compare-cloud-controls');
 preview.hidden=true;viewControl.hidden=true;canvas.hidden=false;controls.hidden=false;
 const cloud=item.cloud,status=$(panel,'.cloud-status'),stats=$(panel,'.compare-stats');
 if(!renderer){status.textContent='Interactive point-cloud rendering is unavailable in this browser.';return}
 status.textContent='Loading derived point-cloud preview…';renderer.clear();
 const total=cloud.total_points??0,shown=cloud.displayed_points??total;
 const full=!item.dense&&cloud.render_all;
 const displayed=renderer.gl||full?shown:Math.min(shown,60000);
 const sampled=cloud.sampled||displayed<total;
 stats.innerHTML=`<strong>${fmt(displayed)} displayed points</strong> · ${fmt(total)} total${sampled?' · sampled preview':''}${item.dense?' · dense cloud':''}${!item.dense?`<br>${fmt(cloud.registered_images)} registered views in this sparse component${full?' · all exported points retained':''}`:''}`;
 if(item.dense)stats.innerHTML+=`<br>Preview sampling is intentional; full dense point clouds remain local.`;
 if(!item.dense&&cloud.render_all)stats.innerHTML+=`<br>All points in this published sparse component are rendered.`;
 const cacheKey=cloud.url;
 if(!bufferCache.has(cacheKey))bufferCache.set(cacheKey,fetch(cloud.url,{cache:'no-store'}).then(response=>{if(!response.ok)throw Error('Point-cloud preview could not be loaded.');return cloud.url.endsWith('.gz')?new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer():response.arrayBuffer()}));
 const ticket=state.token;
 bufferCache.get(cacheKey).then(buffer=>{if(ticket!==state.token)return;renderer.load(buffer,cloud.camera_positions||[],cloud.render_all||false);status.textContent=`${renderer.gl?'GPU':'Software'} point-cloud view · drag to orbit · scroll to zoom.`}).catch(error=>{if(ticket===state.token)status.textContent=error.message});
}
function showGaussian(state,item){
 const {panel}=state,canvas=$(panel,'.cloud-canvas'),preview=$(panel,'.compare-preview'),viewControl=$(panel,'.compare-view-control'),viewSelect=$('select',viewControl),controls=$(panel,'.compare-cloud-controls'),status=$(panel,'.cloud-status'),stats=$(panel,'.compare-stats');
 canvas.hidden=true;controls.hidden=true;preview.hidden=false;viewControl.hidden=false;
 const model=item.model,views=model.views||[];
 viewSelect.replaceChildren(...views.map((view,index)=>new Option(`${view.frame} · saved view ${index+1}`,String(index))));
 const updateView=()=>{const view=views[Number(viewSelect.value)||0];if(view){preview.src=view.render;preview.alt=`${item.label}, rendered from ${view.frame}`}};
 viewSelect.onchange=updateView;viewSelect.value='0';updateView();
 status.textContent='Saved gsplat render · use the interactive viewer link to orbit the trained model.';
 stats.innerHTML=`<strong>${fmt(model.count)} Gaussians</strong> · ${((model.download_bytes||model.bytes||0)/1e6).toFixed(1)} MB ${item.fullLocal?'complete local SH3 model':'model preview'}`;
 const link=document.createElement('a');link.className='compare-model-link';link.href=item.viewerHref||`gsplat.html?model=${encodeURIComponent(item.modelKey)}#gaussian-viewer`;link.textContent='Open this model in the interactive 3D viewer';stats.append(document.createElement('br'),link);
}
function loadSelection(state,item){
 if(!item)return;
 state.token++;$(state.panel,'.compare-choice').value=item.id;
 if(item.kind==='cloud')showCloud(state,item);else showGaussian(state,item);
}
(async()=>{
 const status=document.getElementById('compare-catalog-status');
 const left=makePanel('compare-a','LEFT RESULT'),right=makePanel('compare-b','RIGHT RESULT');
 try{
  const [base,experiments,gaussians,matching,voxels]=await Promise.all([json('clouds/index.json'),json('clouds/experiments.json'),json('gaussians/index.json'),json('clouds/matching.json'),json('voxel-gsplat-assets/index.json').catch(()=>({models:{}}))]);
  const combined={scales:{...experiments.scales,...matching.scales}};
  const items=[...collectClouds(base,combined),...collectGaussians(gaussians),...collectVoxelGaussians(voxels)];
  if(!items.length)throw Error('No comparison results are available.');
  populateSelect(left,items);populateSelect(right,items);
  const selectLeft=$(left.panel,'.compare-choice'),selectRight=$(right.panel,'.compare-choice');
  const baseline=items.find(item=>item.id==='base:4x:sparse:1')||items.find(item=>item.kind==='cloud');
  const iphone=items.find(item=>item.id==='experiment:iphone13pro_4x:sparse:0')||items.find(item=>item.kind==='gsplat');
  if(baseline)selectLeft.value=baseline.id;if(iphone)selectRight.value=iphone.id;
  loadSelection(left,baseline);loadSelection(right,iphone);
  document.getElementById('compare-swap').onclick=()=>{const a=selectLeft.value;selectLeft.value=selectRight.value;selectRight.value=a;loadSelection(left,items.find(item=>item.id===selectLeft.value));loadSelection(right,items.find(item=>item.id===selectRight.value))};
  status.textContent=`${items.length} selectable outputs · choose a result in either panel.`;
 }catch(error){status.textContent=error.message}
})();
})();
