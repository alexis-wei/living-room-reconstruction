(() => {
  const $ = id => document.getElementById(id);
  let catalog, runtime, mesh, busy = false;
  const duration = s => s == null ? 'Pending' : `${s} s`;
  function cell(value) { const td = document.createElement('td'); td.textContent = value; return td; }
  function fmt(value, digits=0) { return value == null ? 'Pending' : Number(value).toLocaleString(undefined,{maximumFractionDigits:digits,minimumFractionDigits:digits}); }
  function predictions() {
    $('fix-predictions').replaceChildren(...Object.entries(catalog.models).map(([key,m]) => {
      const figure = document.createElement('figure'), caption = document.createElement('figcaption');
      caption.textContent = m.label;
      const v = m.views.find(v => v.frame === $('fix-frame').value);
      if(v) { const a=document.createElement('a'),img=document.createElement('img'); a.href=v.render;a.target='_blank';a.rel='noopener';img.src=v.render;img.alt=`${m.label} prediction of ${v.frame}`;img.style.cssText='width:100%;height:auto';a.append(img);figure.append(a); }
      else { const p=document.createElement('p');p.textContent='The cleaned model’s prediction will appear after training and export finish.';figure.append(p); }
      figure.append(caption); return figure;
    }));
  }
  function modelSelected() {
    const m=catalog.models[$('fix-model').value];if(!m)return;
    $('fix-model-size').textContent=`${fmt(m.gaussian_count)} valid Gaussians · full local PLY ${fmt(m.model_bytes/1e6,1)} MB${m.excluded_invalid_gaussians?` · ${m.excluded_invalid_gaussians} non-finite rows excluded by the official exporter`:''}`;
    $('fix-local').href=`${catalog.local_viewer}?model=${encodeURIComponent($('fix-model').value)}`;
    if(mesh&&runtime){runtime.scene.remove(mesh);mesh.dispose();mesh=null;}
    $('fix-3d-state').textContent='Choose a completed model, then load it from this computer or select its PLY file.';
  }
  function preset() {
    if(!runtime)return;const m=catalog.models[$('fix-model').value],v=m.views.find(v=>v.frame===$('fix-frame').value)||m.views[0];if(!v)return;
    runtime.camera.position.fromArray(v.position);runtime.camera.up.fromArray(v.up);runtime.camera.fov=v.fov;runtime.camera.updateProjectionMatrix();
    runtime.controls.target.copy(runtime.camera.position).add(new runtime.THREE.Vector3(...v.forward).multiplyScalar(Math.max(.1,m.scene_scale*.3)));runtime.controls.update();
  }
  async function init() {
    if(runtime)return;
    const THREE=await import('three'),{SparkRenderer,SplatMesh}=await import('./vendor/spark.module.js'),{OrbitControls}=await import('./vendor/OrbitControls.js');
    let renderer;try{renderer=new THREE.WebGLRenderer({antialias:false});}catch(e){throw Error('Interactive 3D requires WebGL2. Open this page in Chrome or Edge with graphics acceleration. Saved image comparisons remain available.');}
    const host=$('fix-canvas');renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));host.replaceChildren(renderer.domElement);
    const scene=new THREE.Scene();scene.background=new THREE.Color('#18232c');scene.add(new SparkRenderer({renderer}));
    const camera=new THREE.PerspectiveCamera(60,1,.01,1000),controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
    const resize=()=>{renderer.setSize(host.clientWidth,host.clientHeight);camera.aspect=host.clientWidth/host.clientHeight;camera.updateProjectionMatrix();};new ResizeObserver(resize).observe(host);resize();renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera);});runtime={THREE,SplatMesh,renderer,scene,camera,controls};
  }
  async function load(file) {
    if(busy)return;busy=true;$('fix-load').disabled=true;$('fix-model').disabled=true;$('fix-file').disabled=true;
    try {
      await init();if(mesh){runtime.scene.remove(mesh);mesh.dispose();mesh=null;}
      const model=catalog.models[$('fix-model').value];$('fix-3d-state').textContent='Loading the full local model, including SH3 appearance…';
      let options;
      if(file){const header=await file.slice(0,8192).text();if(Number(header.match(/element vertex (\d+)/)?.[1])!==model.gaussian_count)throw Error('This PLY does not match the selected model’s point count.');options={fileBytes:new Uint8Array(await file.arrayBuffer()),fileName:file.name,maxSh:3};}
      else options={url:`${catalog.local_viewer}model/${encodeURIComponent($('fix-model').value)}.ply`,maxSh:3};
      const next=new runtime.SplatMesh(options);await next.initialized;mesh=next;runtime.scene.add(mesh);preset();$('fix-3d-state').textContent='Full model loaded. Drag to orbit, right-drag to pan, scroll to zoom.';
    }catch(e){$('fix-3d-state').textContent=e.message+' If the local server is unavailable, choose the matching PLY file.';}
    finally{busy=false;$('fix-load').disabled=false;$('fix-model').disabled=false;$('fix-file').disabled=false;}
  }
  fetch('fixing-gsplat-assets/index.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('Experiment record unavailable');return r.json();}).then(data=>{
    catalog=data;const before=data.models.original,after=data.models.full_resolution_cleaned;
    const descriptions={waiting_for_queue:'Prepared and queued behind the existing reconstruction. Training has not started.',waiting_for_gpu:'Waiting for the RTX 4090.',running:'The cleaned full-resolution model is training on the RTX 4090.',complete:'The cleaned full-resolution run is complete. Compare its held-out predictions and measurements below.',failed:'Training needs attention; completed results will be shown after recovery.'};
    $('fix-state').textContent=descriptions[after.state]||`Cleaned run: ${after.state}`;
    const audit=data.validation;$('fix-audit').textContent=audit?.passed?`Integrity validation passed ${audit.checks} checks: complete checkpoint and SH3 PLY, all ${audit.held_out_views} native held-out canvases and original source-frame identities. Original photographs and models are unchanged.`:'Final integrity validation has not yet been published.';
    const rows=[['Recovered views',m=>fmt(m.registered_images)],['Training photographs',m=>fmt(m.training_images)],['Held-out photographs',m=>fmt(m.validation_images)],['Sparse initialization points',m=>fmt(m===before?data.cleanup.original_sparse_points:data.cleanup.retained_sparse_points)],['Trainer scene scale',m=>fmt(m.trainer_scene_scale,3)],['Final valid Gaussians',m=>fmt(m.gaussian_count)],['PSNR ↑',m=>fmt(m.metrics?.psnr,6)],['SSIM ↑',m=>fmt(m.metrics?.ssim,6)],['LPIPS ↓',m=>fmt(m.metrics?.lpips,6)],['Run elapsed time, seconds',m=>duration(m.seconds)],['Full local PLY (MB)',m=>m.model_bytes==null?'Pending':fmt(m.model_bytes/1e6,1)]];
    $('fix-summary').replaceChildren(...rows.map(([label,fn])=>{const tr=document.createElement('tr');tr.append(cell(label),cell(fn(before)),cell(fn(after)));return tr;}));
    $('fix-removed').replaceChildren(...data.cleanup.removed_cameras.map(row=>{const tr=document.createElement('tr');tr.append(cell(row.frame),cell(row.point_observations),cell(row.reasons.join('; ')));return tr;}));
    $('fix-frame').replaceChildren(...data.shared_frames.map(f=>new Option(f,f)));$('fix-frame').onchange=()=>{predictions();if(mesh)preset();};predictions();
    const complete=Object.entries(data.models).filter(([,m])=>m.state==='complete');$('fix-model').replaceChildren(...complete.map(([key,m])=>new Option(m.label,key)));if(after.state==='complete')$('fix-model').value='full_resolution_cleaned';modelSelected();
    $('fix-model').onchange=modelSelected;$('fix-load').onclick=()=>load();$('fix-file').onchange=e=>{if(e.target.files[0])load(e.target.files[0]);};$('fix-reset').onclick=preset;
    if(location.hostname==='127.0.0.1'){const requested=new URLSearchParams(location.search).get('model');if(data.models[requested]?.state==='complete')$('fix-model').value=requested;modelSelected();load();}
  }).catch(e=>{$('fix-state').textContent=e.message;});
})();
