import * as THREE from 'three';
import {SparkRenderer,SplatMesh} from './vendor/spark.module.js';
import {OrbitControls} from './vendor/OrbitControls.js';

const host=document.getElementById('matching-full-canvas'),status=document.getElementById('matching-full-state'),select=document.getElementById('matching-full-model');
let catalog,runtime,mesh,busy=false;
function init(){
 if(runtime)return;
 let renderer;
 try{renderer=new THREE.WebGLRenderer({antialias:false,alpha:false});}catch(error){throw Error('Full 3D requires WebGL2. Use Chrome or Edge with graphics acceleration. The full-model rendered comparisons above remain available.');}
 renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));host.replaceChildren(renderer.domElement);
 const scene=new THREE.Scene();scene.background=new THREE.Color('#18232c');scene.add(new SparkRenderer({renderer}));
 const camera=new THREE.PerspectiveCamera(60,1,.01,1000),controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
 new ResizeObserver(()=>{renderer.setSize(host.clientWidth,host.clientHeight);camera.aspect=host.clientWidth/host.clientHeight;camera.updateProjectionMatrix();}).observe(host);
 renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera);});runtime={renderer,scene,camera,controls};
}
function preset(){
 if(!runtime||!catalog)return;
 const m=catalog.models[select.value],v=m.views?.[0];if(!v)return;
 runtime.camera.position.fromArray(v.position);runtime.camera.up.fromArray(v.up);runtime.camera.fov=v.fov;runtime.camera.updateProjectionMatrix();
 runtime.controls.target.copy(runtime.camera.position).add(new THREE.Vector3(...v.forward).multiplyScalar(Math.max(.1,m.scene_scale*.3)));runtime.controls.update();
}
async function load(source,fileName){
 if(busy)return;busy=true;
 select.disabled=true;document.getElementById('matching-full-fetch').disabled=true;document.getElementById('matching-full-file').disabled=true;
 try{
  init();status.textContent='Loading the full local PLY; all trained Gaussians and SH3 appearance are retained.';
  if(mesh){runtime.scene.remove(mesh);mesh.dispose();mesh=null;}
  const options=typeof source==='string'?{url:source,maxSh:3}:{fileBytes:new Uint8Array(source),fileName,maxSh:3};
  const next=new SplatMesh(options);await next.initialized;mesh=next;runtime.scene.add(mesh);preset();
  status.textContent=`${catalog.models[select.value].label} · component ${catalog.models[select.value].component} · full local model loaded. Drag to orbit, right-drag to pan, scroll to zoom.`;
 }catch(error){status.textContent=error.message;}finally{busy=false;select.disabled=false;document.getElementById('matching-full-fetch').disabled=false;document.getElementById('matching-full-file').disabled=false;}
}
function selected(){
 const m=catalog.models[select.value],a=document.getElementById('matching-full-local');
 a.href=`${catalog.local_viewer}?model=${encodeURIComponent(select.value)}`;
 document.getElementById('matching-full-count').textContent=m.gaussian_count?`${m.gaussian_count.toLocaleString()} trained Gaussians · full model ${(m.model_bytes/1e6).toFixed(1)} MB · retained locally`:`${m.label} · ${m.state}`;
 status.textContent='Load the full model from this computer, open the local viewer, or choose its PLY file below. Files stay on this computer and are not uploaded.';
}
document.getElementById('matching-full-file').onchange=async event=>{
 const file=event.target.files[0];if(!file)return;
 const header=await file.slice(0,8192).text();const count=Number(header.match(/element vertex (\d+)/)?.[1]);
 if(count!==catalog.models[select.value].gaussian_count){status.textContent='This PLY point count does not match the selected trained component. Choose the matching component first.';return;}
 await load(await file.arrayBuffer(),file.name);
};
document.getElementById('matching-full-reset').onclick=preset;
document.getElementById('matching-full-fetch').onclick=()=>load(`${catalog.local_viewer}model/${encodeURIComponent(select.value)}.ply`);
select.onchange=()=>{if(mesh){runtime.scene.remove(mesh);mesh.dispose();mesh=null;}selected();};
const local=document.body.dataset.localFullViewer==='true';
fetch(local?'catalog.json':'matching-gsplat/index.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('Full-model catalog unavailable.');return r.json();}).then(data=>{
 catalog=data;const entries=Object.entries(data.models).filter(([,m])=>m.state==='complete');
 select.replaceChildren(...entries.map(([key,m])=>new Option(`${m.label} · component ${m.component} · ${m.registered_images} cameras`,key)));
 if(!entries.length){status.textContent='Training is still running. Full models will appear after completion.';return;}
 const requested=new URLSearchParams(location.search).get('model');if(entries.some(([key])=>key===requested))select.value=requested;selected();
 if(local){document.getElementById('matching-full-local').hidden=true;load(`model/${encodeURIComponent(select.value)}.ply`);select.onchange=()=>{selected();load(`model/${encodeURIComponent(select.value)}.ply`);};}
}).catch(error=>{status.textContent=error.message;});
