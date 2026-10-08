import {OnlineGaussianViewer,onlineModel} from './online-gaussian-viewer.js';
const $=id=>document.getElementById(id),labels={'1x':'Full resolution','2x':'2× downsample','4x':'4× downsample','8x':'8× downsample','8x_component_2':'8× · separate component 2'};
let catalog,asset,key,busy=false;
const viewer=new OnlineGaussianViewer($('gs-canvas'),text=>$('gs-state').textContent=text);
function preset(i){viewer.preset(catalog[key].views[i],catalog[key].radius);}
async function load(){if(busy||!asset)return;busy=true;for(const id of ['gs-load','gs-model','gs-renderer'])$(id).disabled=true;try{await viewer.load(asset,catalog[key].views[0],catalog[key].radius,{software:$('gs-renderer').value==='software'});}finally{busy=false;for(const id of ['gs-load','gs-model','gs-renderer'])$(id).disabled=false;}}
async function selected(){
  viewer.clear();const selection=$('gs-model').value;key=selection;asset=null;
  $('gs-load').disabled=true;$('gs-state').textContent='Preparing hosted model…';
  try{
    const next=await onlineModel('original/'+selection);if(key!==selection)return;asset=next;
    $('gs-size').textContent=`${asset.count.toLocaleString()} Gaussians in complete native export · SH3 progressive loading${asset.legacy_browser_count?` · previous browser buffer had ${asset.legacy_browser_count.toLocaleString()} rows`:''}`;
    const link=$('gs-download');link.href=asset.raw_url+'?download=1';link.textContent=`Download complete SH3 PLY · ${(asset.raw_bytes/1e6).toFixed(1)} MB`;link.download='';
    $('gs-images').replaceChildren(...catalog[key].views.map((v,i)=>{const f=document.createElement('figure'),a=document.createElement('a'),img=document.createElement('img'),c=document.createElement('figcaption'),b=document.createElement('button');a.href=v.render;a.target='_blank';a.rel='noopener';img.src=v.render;img.alt=`${labels[key]} full trained prediction of ${v.frame}`;img.loading='lazy';a.append(img);c.textContent=v.frame+' · full trained appearance';b.textContent='Go to this camera';b.onclick=async()=>{if(!viewer.loaded)await load();preset(i);};f.append(a,c,b);return f;}));
    $('gs-state').textContent='Choose a saved view, or load progressive 3D. The trained models and their source photographs are unchanged.';$('gs-load').disabled=false;
  }catch(error){$('gs-state').textContent=error.message;}
}
$('gs-load').onclick=load;$('gs-model').onchange=selected;$('gs-reset').onclick=()=>preset(0);$('gs-renderer').onchange=load;$('gs-fullscreen').onclick=()=>$('gs-canvas').requestFullscreen?.();
const link=document.createElement('a');link.id='gs-download';link.className='hint';$('gs-size').after(link);
fetch('gaussians/index.json',{cache:'no-store'}).then(r=>r.json()).then(data=>{catalog=data;const requested=new URLSearchParams(location.search).get('model');if(data[requested])$('gs-model').value=requested;selected();}).catch(error=>$('gs-state').textContent=error.message);
