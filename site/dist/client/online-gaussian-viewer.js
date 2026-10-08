import {SoftwareGaussianRenderer} from './gaussian-software.js';

let catalogPromise;
export function onlineModel(id) {
  catalogPromise ||= fetch('online-models.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('Hosted model catalog unavailable.');return r.json();});
  return catalogPromise.then(c=>{if(!c.models[id])throw Error('This completed model has not been published.');return c.models[id];});
}

// Progressive GPU rendering and an explicit, small software navigation fallback.
// Both read only authenticated same-origin URLs; no desktop server is required.
export class OnlineGaussianViewer {
  constructor(host,status) {this.host=host;this.status=status;this.revision=0;}
  async initialize(software=false) {
    if(this.runtime && this.runtime.forcedSoftware===software)return;
    this.disposeRuntime(false);
    const THREE=await import('three'),{OrbitControls}=await import('./vendor/OrbitControls.js');
    const canvas=document.createElement('canvas');let context;
    if(!software)try{context=canvas.getContext('webgl2',{alpha:false,antialias:false});}catch{}
    let renderer,SplatMesh,spark,softwareRenderer;
    if(context){
      const module=await import('./vendor/spark-streaming.module.js');SplatMesh=module.SplatMesh;
      renderer=new THREE.WebGLRenderer({canvas,context,antialias:false,alpha:false});renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));
      const mobile=matchMedia('(max-width:800px)').matches;
      spark=new module.SparkRenderer({renderer,pagedExtSplats:true,maxPagedSplats:mobile?2097152:4194304,numLodFetchers:3,lodSplatCount:mobile?750000:1500000});
    }else{softwareRenderer=new SoftwareGaussianRenderer(document.createElement('canvas'));renderer=softwareRenderer;}
    this.host.replaceChildren(renderer.domElement);renderer.domElement.tabIndex=0;
    renderer.domElement.setAttribute('aria-label','Interactive Gaussian reconstruction. Drag to orbit, right-drag to pan, scroll to zoom.');
    const scene=new THREE.Scene();scene.background=new THREE.Color('#18232c');if(spark)scene.add(spark);
    const camera=new THREE.PerspectiveCamera(60,1,.001,1000),controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
    const resize=()=>{renderer.setSize(Math.max(1,this.host.clientWidth),Math.max(1,this.host.clientHeight));camera.aspect=Math.max(1,this.host.clientWidth)/Math.max(1,this.host.clientHeight);camera.updateProjectionMatrix();};
    const observer=new ResizeObserver(resize);observer.observe(this.host);resize();
    renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera);});
    if(context)renderer.domElement.addEventListener('webglcontextlost',event=>{event.preventDefault();if(this.runtime?.renderer.domElement!==event.target||!this.current)return;const {asset,view,radius}=this.current;this.load(asset,view,radius,{software:true});});
    this.runtime={THREE,renderer,software:softwareRenderer,SplatMesh,spark,scene,camera,controls,observer,forcedSoftware:software};
  }
  preset(view,radius=1) {
    this.view=view;this.radius=radius;if(!this.runtime||!view)return;
    const {THREE,camera,controls}=this.runtime;camera.position.fromArray(view.position);camera.up.fromArray(view.up);camera.fov=view.fov;camera.updateProjectionMatrix();
    controls.target.copy(camera.position).add(new THREE.Vector3(...view.forward).multiplyScalar(Math.max(.1,radius*.3)));controls.update();
  }
  clear(){
    this.revision++;this.request?.abort();this.request=null;
    if(this.mesh){this.runtime?.scene.remove(this.mesh);this.mesh.dispose();this.mesh=null;}
    this.runtime?.software?.clear();this.loaded=false;this.current=null;
  }
  disposeRuntime(invalidate=true){
    if(!this.runtime)return;if(invalidate)this.clear();
    const r=this.runtime;r.observer.disconnect();r.controls.dispose();r.renderer.setAnimationLoop(null);r.renderer.dispose();r.spark?.dispose?.();this.runtime=null;
  }
  async load(asset,view,radius=1,{software=false,file}={}) {
    this.clear();const ticket=this.revision;this.current={asset,view,radius};this.status('Checking graphics support…');
    try {
      await this.initialize(software);if(ticket!==this.revision)return;this.preset(view,radius);
      if(this.runtime.software){
        if(file)throw Error('Local PLY loading requires WebGL2. Use the hosted navigation preview instead.');
        this.status(`Loading ${Math.round(asset.preview_bytes/1e6*10)/10} MB software navigation preview…`);
        const controller=new AbortController();this.request=controller;
        const response=await fetch(asset.preview_url,{signal:controller.signal});if(!response.ok)throw Error('Hosted navigation preview could not be loaded.');
        const raw=await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
        if(raw.byteLength!==asset.preview_count*32)throw Error('Navigation preview failed its size check.');
        if(ticket!==this.revision)return;
        const shown=await this.runtime.software.load(new Uint8Array(raw));if(ticket!==this.revision)return;
        this.loaded=true;this.status(`Software navigation preview · ${shown.toLocaleString()} sampled of ${asset.count.toLocaleString()} exported Gaussians · static color. Saved renders show trained appearance; download the complete SH3 PLY for full fidelity.`);
      }else{
        this.status(file?'Loading the selected complete SH3 PLY…':'Loading a coarse view; finer SH3 detail will stream as you explore…');
        const options=file?{fileBytes:new Uint8Array(await file.arrayBuffer()),fileName:file.name}:{url:asset.stream_url,paged:true};
        const next=new this.runtime.SplatMesh(options);next.maxSh=3;
        // Add the paged mesh immediately so Spark can request its root pages.
        this.runtime.scene.add(next);this.mesh=next;
        try{await next.initialized;}catch(error){if(this.mesh===next){this.runtime.scene.remove(next);next.dispose();this.mesh=null;}throw error;}
        if(ticket!==this.revision)return;
        this.loaded=true;this.preset(view,radius);
        this.status(file?`Complete SH3 PLY loaded · ${asset.count.toLocaleString()} exported Gaussians.`:`Progressive SH3 model ready · ${asset.count.toLocaleString()} source Gaussians. Detail loads as you move; adaptive level of detail limits what is drawn at once. Use the download link for the complete unchanged PLY.`);
      }
    }catch(error){
      if(ticket!==this.revision||error.name==='AbortError')return;
      if(!software&&!file){console.warn('Progressive renderer unavailable; using navigation fallback.',error.message);await this.load(asset,view,radius,{software:true});}
      else this.status(error.message);
    }
    finally{if(ticket===this.revision)this.request=null;}
  }
}
