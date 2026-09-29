(()=>{
'use strict';
const cache=new Map();
class CloudRenderer{
 constructor(canvas){
  this.canvas=canvas;this.gl=canvas.getContext('webgl',{antialias:true,alpha:false});this.ctx=this.gl?null:canvas.getContext('2d');if(!this.gl&&!this.ctx)throw Error('Canvas rendering is unavailable.');
  const gl=this.gl;
  if(gl){
  const vs=`attribute vec3 pos;attribute vec3 color;uniform vec2 angle;uniform vec2 pan;uniform float distance;uniform float aspect;uniform float size;varying vec3 rgb;void main(){float cy=cos(angle.x),sy=sin(angle.x),cx=cos(angle.y),sx=sin(angle.y);vec3 p=vec3(cy*pos.x+sy*pos.z,pos.y,-sy*pos.x+cy*pos.z);p=vec3(p.x,cx*p.y-sx*p.z,sx*p.y+cx*p.z);p.xy+=pan;float z=distance-p.z;gl_Position=vec4(p.x*1.8/aspect,-p.y*1.8,z*1.0002-.02,z);gl_PointSize=size;rgb=color;}`;
  const fs=`precision mediump float;varying vec3 rgb;void main(){vec2 d=gl_PointCoord-vec2(.5);if(dot(d,d)>.25)discard;gl_FragColor=vec4(rgb,1.);}`;
  const compile=(kind,src)=>{let sh=gl.createShader(kind);gl.shaderSource(sh,src);gl.compileShader(sh);if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(sh));return sh};
  this.program=gl.createProgram();gl.attachShader(this.program,compile(gl.VERTEX_SHADER,vs));gl.attachShader(this.program,compile(gl.FRAGMENT_SHADER,fs));gl.linkProgram(this.program);if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw Error('Could not initialize point renderer');
  this.pointBuffer=gl.createBuffer();this.cameraBuffer=gl.createBuffer();}
  this.count=0;this.cameraCount=0;this.showCameras=true;this.pointSize=2;this.reset();
  canvas.addEventListener('pointerdown',e=>{this.drag={x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId)});
  canvas.addEventListener('pointermove',e=>{if(!this.drag)return;let dx=e.clientX-this.drag.x,dy=e.clientY-this.drag.y;if(e.shiftKey){this.pan[0]+=dx*.003*this.distance;this.pan[1]+=dy*.003*this.distance}else{this.yaw+=dx*.008;this.pitch+=dy*.008}this.drag={x:e.clientX,y:e.clientY};this.draw()});
  canvas.addEventListener('pointerup',()=>this.drag=null);canvas.addEventListener('pointercancel',()=>this.drag=null);
  canvas.addEventListener('wheel',e=>{e.preventDefault();this.distance=Math.max(.15,Math.min(80,this.distance*Math.exp(e.deltaY*.001)));this.draw()},{passive:false});
  canvas.tabIndex=0;canvas.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')this.yaw-=.12;else if(e.key==='ArrowRight')this.yaw+=.12;else if(e.key==='ArrowUp')this.pitch-=.12;else if(e.key==='ArrowDown')this.pitch+=.12;else if(e.key==='+')this.distance*=.85;else if(e.key==='-')this.distance*=1.15;else return;e.preventDefault();this.draw()});
  new ResizeObserver(()=>this.draw()).observe(canvas);
 }
 reset(){this.yaw=.25;this.pitch=-.2;this.pan=[0,0];this.distance=2.8;if(this.canvas)this.draw()}
 load(buffer,cameras){const gl=this.gl;this.count=buffer.byteLength/24;if(!gl){this.softwarePoints=new Float32Array(buffer);this.softwareCameras=cameras;this.reset();this.draw();return}gl.bindBuffer(gl.ARRAY_BUFFER,this.pointBuffer);gl.bufferData(gl.ARRAY_BUFFER,buffer,gl.STATIC_DRAW);const arr=new Float32Array(cameras.length*6);cameras.forEach((p,i)=>arr.set([...p,1,.5,.14],i*6));this.cameraCount=cameras.length;gl.bindBuffer(gl.ARRAY_BUFFER,this.cameraBuffer);gl.bufferData(gl.ARRAY_BUFFER,arr,gl.STATIC_DRAW);this.reset()}
 clear(){this.count=0;this.cameraCount=0;this.draw()}
 draw(){const gl=this.gl;if(!gl){this.drawSoftware();return;}const c=this.canvas,dpr=Math.min(devicePixelRatio,2),w=Math.round(c.clientWidth*dpr),h=Math.round(c.clientHeight*dpr);if(c.width!==w||c.height!==h){c.width=w;c.height=h}gl.viewport(0,0,w,h);gl.clearColor(.035,.075,.115,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.useProgram(this.program);
 const uniform=(name)=>gl.getUniformLocation(this.program,name);gl.uniform2f(uniform('angle'),this.yaw,this.pitch);gl.uniform2fv(uniform('pan'),this.pan);gl.uniform1f(uniform('distance'),this.distance);gl.uniform1f(uniform('aspect'),w/h);
 const paint=(buffer,count,size)=>{gl.bindBuffer(gl.ARRAY_BUFFER,buffer);for(const [name,offset]of[['pos',0],['color',12]]){const a=gl.getAttribLocation(this.program,name);gl.enableVertexAttribArray(a);gl.vertexAttribPointer(a,3,gl.FLOAT,false,24,offset)}gl.uniform1f(uniform('size'),size*dpr);gl.drawArrays(gl.POINTS,0,count)};
 paint(this.pointBuffer,this.count,this.pointSize);if(this.showCameras)paint(this.cameraBuffer,this.cameraCount,5);
 }
 drawSoftware(){
  const ctx=this.ctx,c=this.canvas;if(!ctx)return;const ratio=Math.min(devicePixelRatio,2),w=Math.round(c.clientWidth*ratio),h=Math.round(c.clientHeight*ratio);if(!w||!h)return;if(c.width!==w||c.height!==h){c.width=w;c.height=h}ctx.fillStyle='#09131d';ctx.fillRect(0,0,w,h);if(!this.count||!this.softwarePoints)return;
  const cy=Math.cos(this.yaw),sy=Math.sin(this.yaw),cx=Math.cos(this.pitch),sx=Math.sin(this.pitch),project=(x,y,z)=>{const xx=cy*x+sy*z,zz=-sy*x+cy*z,yy=cx*y-sx*zz,depth=this.distance-(sx*y+cx*zz);if(depth<.02)return null;return[(xx+this.pan[0])*h*.9/depth+w/2,(yy+this.pan[1])*h*.9/depth+h/2,depth]};
  const points=this.softwarePoints,draw=[],n=Math.min(this.count,60000);
  for(let j=0;j<n;j++){const i=Math.floor(j*this.count/n)*6,p=project(points[i],points[i+1],points[i+2]);if(p&&p[0]>=0&&p[0]<w&&p[1]>=0&&p[1]<h)draw.push([p,i])}
  draw.sort((a,b)=>b[0][2]-a[0][2]);const size=this.pointSize*ratio;
  for(const [p,i] of draw){ctx.fillStyle=`rgb(${Math.round(points[i+3]*255)},${Math.round(points[i+4]*255)},${Math.round(points[i+5]*255)})`;ctx.fillRect(p[0],p[1],size,size)}
  if(this.showCameras){ctx.fillStyle='#ff8024';for(const c of this.softwareCameras||[]){const p=project(...c);if(p)ctx.fillRect(p[0]-2*ratio,p[1]-2*ratio,4*ratio,4*ratio)}}
 }

}
function setupPanel(id,index,preferred){
 const panel=document.getElementById(id);panel.innerHTML=`<div class="cloud-controls"><select aria-label="Resolution for ${id}"></select><select aria-label="Component for ${id}"></select></div><p class="cloud-status" role="status"></p><canvas class="cloud-canvas" aria-label="Interactive point cloud. Arrow keys orbit, plus and minus zoom."></canvas><div class="cloud-controls"><button type="button">Reset view</button><label>Point size <input type="range" min="1" max="5" value="2" step=".5" aria-label="Point size for ${id}"></label><label><input type="checkbox" checked>Camera centers</label></div><div class="cloud-stats"></div><div class="coverage-strip" aria-label="Source frame registration coverage"></div><div class="coverage-label"></div><details><summary>Components & remaining gaps</summary><p></p></details>`;
 const [res,comp]=panel.querySelectorAll('select'),status=panel.querySelector('.cloud-status'),stats=panel.querySelector('.cloud-stats'),strip=panel.querySelector('.coverage-strip'),label=panel.querySelector('.coverage-label');
 let renderer;try{renderer=new CloudRenderer(panel.querySelector('canvas'))}catch(e){status.textContent=e.message;return}
 for(const k of ['1x','2x','4x','8x']){const opt=new Option(k==='1x'?'Full resolution':`${k} downsample`,k);res.add(opt)}res.value=preferred;
 let token=0;
 const load=async()=>{
  const ticket=++token,d=index.scales[res.value],cloud=comp.value==='dense'?d.dense:d.components.find(c=>c.id===comp.value);renderer.clear();strip.replaceChildren();
  if(!cloud||!cloud.url){status.textContent='No completed point cloud yet for this selection.';stats.textContent='Reconstruction is still running or this component has no triangulated points.';label.textContent='';return}
  status.textContent='Loading derived point cloud…';
  const ids=new Set(cloud.frame_ids||[]);for(let i=1;i<=500;i++){const span=document.createElement('span');span.className=ids.has(i)?'present':'';span.title=`Frame ${String(i).padStart(4,'0')}: ${ids.has(i)?'registered':'not in this component'}`;strip.append(span)}
  label.textContent=cloud.kind==='dense'?'Dense fusion from the largest sparse component.':`Blue = registered in this component · ${ids.size}/500 frames · row order 1–500`;
  stats.innerHTML=`<strong>${cloud.total_points.toLocaleString()} ${cloud.kind==='dense'?'fused':'sparse'} points</strong> · ${Math.min(cloud.displayed_points,renderer.gl?Infinity:60000).toLocaleString()} displayed${(cloud.sampled||(!renderer.gl&&cloud.displayed_points>60000))?' (sampled preview)':' (all points)'}${cloud.kind==='sparse'?`<br>${cloud.registered_images} registered images · ${cloud.substantial?'substantial component':'small / potentially unstable component'}`:''}`;
  try{if(!cache.has(cloud.url))cache.set(cloud.url,fetch(cloud.url).then(r=>{if(!r.ok)throw Error('Cloud file unavailable');return r.arrayBuffer()}));const buf=await cache.get(cloud.url);if(ticket!==token)return;renderer.load(buf,cloud.camera_positions||[]);status.textContent=(renderer.gl?'GPU viewer':'Software viewer · up to 60,000 points')+' · Drag to explore. Orange dots show camera centers.'}catch(e){if(ticket===token)status.textContent=e.message}
 };
 const changeResolution=()=>{const d=index.scales[res.value];comp.replaceChildren();for(const c of d.components){const opt=new Option(`Component ${c.id} · ${c.registered_images} views${c.substantial?'':' · small'}${c.total_points?'':' · no points'}`,c.id);opt.disabled=!c.total_points;comp.add(opt)}if(d.dense)comp.add(new Option('Dense fused cloud · largest component','dense'));const first=d.components.find(c=>c.url);if(first)comp.value=first.id;else if(d.dense)comp.value='dense';
 panel.querySelector('details p').textContent=`${d.registered_union} unique frames belong to non-empty sparse models. ${d.missing_frame_ids.length} frames remain outside them. Missing frame IDs: ${d.missing_frame_ids.length?d.missing_frame_ids.join(', '):'none'}. Substantial means at least 10 registered images and 100 points. Smaller non-empty models are also inspectable; zero-point models are listed but disabled.`;load()};
 res.onchange=changeResolution;comp.onchange=load;panel.querySelector('button').onclick=()=>renderer.reset();panel.querySelector('input[type=range]').oninput=e=>{renderer.pointSize=Number(e.target.value);renderer.draw()};panel.querySelector('input[type=checkbox]').onchange=e=>{renderer.showCameras=e.target.checked;renderer.draw()};changeResolution();
}
fetch('clouds/index.json').then(r=>{if(!r.ok)throw Error('Point-cloud previews are not available yet.');return r.json()}).then(index=>{const ready=['1x','2x','4x','8x'].filter(k=>index.scales[k].components.some(c=>c.url));setupPanel('cloud-a',index,ready[0]||'1x');setupPanel('cloud-b',index,ready.includes('8x')?'8x':ready[1]||ready[0]||'8x')}).catch(e=>{document.getElementById('cloud-a').textContent=e.message});
})();
