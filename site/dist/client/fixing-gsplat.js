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
  let viewer, onlinePromise;
  async function online() {
    onlinePromise ||= import('./online-gaussian-viewer.js');
    const module=await onlinePromise;
    viewer ||= new module.OnlineGaussianViewer($('fix-canvas'),text=>$('fix-3d-state').textContent=text);
    return module;
  }
  async function modelSelected() {
    const key=$('fix-model').value,m=catalog.models[key];if(!m)return;
    const module=await online();viewer.clear();
    try {
      const asset=await module.onlineModel('fix/'+key);if($('fix-model').value!==key)return;
      $('fix-model-size').textContent=`${fmt(asset.count)} exported Gaussians · complete SH3 PLY ${fmt(asset.raw_bytes/1e6,1)} MB · available online`;
      $('fix-local').href=asset.raw_url+'?download=1';$('fix-local').textContent='Download complete SH3 PLY';$('fix-local').download='';
      $('fix-3d-state').textContent='Load progressive 3D. The model appears at coarse detail first and refines as you explore.';
    }catch(error){$('fix-3d-state').textContent=error.message;}
  }
  function preset() {
    if(!viewer||!catalog)return;
    const m=catalog.models[$('fix-model').value],v=m.views.find(v=>v.frame===$('fix-frame').value)||m.views[0];
    viewer.preset(v,m.scene_scale||1);
  }
  async function load(file) {
    if(busy)return;busy=true;$('fix-load').disabled=true;$('fix-model').disabled=true;$('fix-file').disabled=true;
    try {
      const module=await online(),model=catalog.models[$('fix-model').value],asset=await module.onlineModel('fix/'+$('fix-model').value);
      if(file&&(file.size!==asset.raw_bytes||Number((await file.slice(0,8192).text()).match(/element vertex (\d+)/)?.[1])!==asset.count))throw Error('This PLY does not match the selected model.');
      const view=model.views.find(v=>v.frame===$('fix-frame').value)||model.views[0];
      await viewer.load(asset,view,model.scene_scale||1,{file});
    }catch(error){$('fix-3d-state').textContent=error.message;}
    finally{busy=false;$('fix-load').disabled=false;$('fix-model').disabled=false;$('fix-file').disabled=false;}
  }
  fetch('fixing-gsplat-assets/index.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('Experiment record unavailable');return r.json();}).then(data=>{
    catalog=data;const before=data.models.original,after=data.models.full_resolution_cleaned;
    const descriptions={waiting_for_queue:'Prepared and queued behind the existing reconstruction. Training has not started.',waiting_for_gpu:'Waiting for the RTX 4090.',running:'The cleaned full-resolution model is training on the RTX 4090.',complete:'The cleaned full-resolution run is complete. Compare its held-out predictions and measurements below.',failed:'Training needs attention; completed results will be shown after recovery.'};
    $('fix-state').textContent=descriptions[after.state]||`Cleaned run: ${after.state}`;
    const audit=data.validation;$('fix-audit').textContent=audit?.passed?`Integrity validation passed ${audit.checks} checks: complete checkpoint and SH3 PLY, all ${audit.held_out_views} native held-out canvases and original source-frame identities. Original photographs and models are unchanged.`:'Final integrity validation has not yet been published.';
    const rows=[['Recovered views',m=>fmt(m.registered_images)],['Training photographs',m=>fmt(m.training_images)],['Held-out photographs',m=>fmt(m.validation_images)],['Sparse initialization points',m=>fmt(m===before?data.cleanup.original_sparse_points:data.cleanup.retained_sparse_points)],['Trainer scene scale',m=>fmt(m.trainer_scene_scale,3)],['Final valid Gaussians',m=>fmt(m.gaussian_count)],['PSNR ↑',m=>fmt(m.metrics?.psnr,6)],['SSIM ↑',m=>fmt(m.metrics?.ssim,6)],['LPIPS ↓',m=>fmt(m.metrics?.lpips,6)],['Run elapsed time, seconds',m=>duration(m.seconds)],['Complete SH3 PLY (MB)',m=>m.model_bytes==null?'Pending':fmt(m.model_bytes/1e6,1)]];
    $('fix-summary').replaceChildren(...rows.map(([label,fn])=>{const tr=document.createElement('tr');tr.append(cell(label),cell(fn(before)),cell(fn(after)));return tr;}));
    $('fix-removed').replaceChildren(...data.cleanup.removed_cameras.map(row=>{const tr=document.createElement('tr');tr.append(cell(row.frame),cell(row.point_observations),cell(row.reasons.join('; ')));return tr;}));
    $('fix-frame').replaceChildren(...data.shared_frames.map(f=>new Option(f,f)));$('fix-frame').onchange=()=>{predictions();if(viewer?.loaded)preset();};predictions();
    const complete=Object.entries(data.models).filter(([,m])=>m.state==='complete');$('fix-model').replaceChildren(...complete.map(([key,m])=>new Option(m.label,key)));if(after.state==='complete')$('fix-model').value='full_resolution_cleaned';modelSelected();
    $('fix-model').onchange=modelSelected;$('fix-load').onclick=()=>load();$('fix-file').onchange=e=>{if(e.target.files[0])load(e.target.files[0]);};$('fix-reset').onclick=preset;
    if(location.hostname==='127.0.0.1'){const requested=new URLSearchParams(location.search).get('model');if(data.models[requested]?.state==='complete')$('fix-model').value=requested;modelSelected();load();}
  }).catch(e=>{$('fix-state').textContent=e.message;});
})();
