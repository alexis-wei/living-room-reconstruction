import {OnlineGaussianViewer,onlineModel} from './online-gaussian-viewer.js';
const $=id=>document.getElementById(id),select=$('matching-full-model'),status=$('matching-full-state');
const group=document.body.dataset.fullCatalog?.includes('insta360')?'insta360':'matching';
let catalog,busy=false,asset;
const viewer=new OnlineGaussianViewer($('matching-full-canvas'),text=>status.textContent=text);
async function selected(){
  viewer.clear();asset=null;const key=select.value;status.textContent='Preparing hosted model…';
  try{
    const next=await onlineModel(group+'/'+key);if(select.value!==key)return;asset=next;
    $('matching-full-count').textContent=`${asset.count.toLocaleString()} exported Gaussians · complete SH3 PLY ${(asset.raw_bytes/1e6).toFixed(1)} MB · available online`;
    const link=$('matching-full-local');link.href=asset.raw_url+'?download=1';link.textContent='Download complete SH3 PLY';link.download='';
    status.textContent='Load progressive 3D to explore. Coarse geometry appears first and finer detail loads as you move. Each component has its own coordinate system.';
  }catch(error){status.textContent=error.message;}
}
async function load(file){
  if(busy||!asset)return;busy=true;$('matching-full-fetch').disabled=true;select.disabled=true;
  try{if(file&&(file.size!==asset.raw_bytes||Number((await file.slice(0,8192).text()).match(/element vertex (\d+)/)?.[1])!==asset.count))throw Error('This PLY does not match the selected model.');const m=catalog.models[select.value];await viewer.load(asset,m.views?.[0],m.scene_scale||1,{file});}
  catch(error){status.textContent=error.message;}
  finally{busy=false;$('matching-full-fetch').disabled=false;select.disabled=false;}
}
$('matching-full-fetch').textContent='Load progressive 3D';$('matching-full-fetch').onclick=()=>load();
$('matching-full-file').onchange=e=>{if(e.target.files[0])load(e.target.files[0]);};
$('matching-full-reset').onclick=()=>viewer.preset(catalog.models[select.value].views?.[0],catalog.models[select.value].scene_scale||1);
select.onchange=selected;
fetch(document.body.dataset.fullCatalog||'matching-gsplat/index.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('Model catalog unavailable.');return r.json();}).then(data=>{
  catalog=data;const entries=Object.entries(data.models).filter(([,m])=>m.state==='complete');
  select.replaceChildren(...entries.map(([key,m])=>new Option(`${m.label} · component ${m.component} · ${m.registered_images} cameras`,key)));
  const params=new URLSearchParams(location.search);if(params.has('model')||location.hash==='#matching-full-canvas')$('matching-full-canvas').closest('details').open=true;const requested=params.get('model')||(group==='matching'?'sequential_no_loop_component_1':'4x_component_0');if(entries.some(([key])=>key===requested))select.value=requested;selected();
}).catch(error=>status.textContent=error.message);
