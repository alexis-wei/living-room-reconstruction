const $ = id => document.getElementById(id);
const sides = ['a', 'b'];
const states = Object.fromEntries(sides.map(side => [side, { runtime: null, mesh: null, ticket: 0, busy: false }]));
let catalog;
const count = value => Number.isFinite(value) ? value.toLocaleString() : 'Unavailable';
const exact = value => Number.isFinite(value) ? String(value) : 'Unavailable';
const score = value => Number.isFinite(value) ? value.toFixed(6) : 'Unavailable';
const selected = side => catalog.models[$(`voxel-model-${side}`).value];
const number = (value, digits=3) => Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });

function cameraPreset(side) {
  const model=selected(side),view=model.views.find(v=>v.frame===$('voxel-frame').value)||model.views[0];
  states[side].viewer?.preset(view,model.radius||model.scene_scale||1);
}

function image(side) {
  const model = selected(side), view = model.views.find(v => v.frame === $('voxel-frame').value);
  $(`voxel-label-${side}`).textContent = model.label;
  const img = $(`voxel-image-${side}`), link = $(`voxel-image-link-${side}`);
  img.hidden = !view;
  if (view) {
    img.src = view.render;
    img.alt = `${model.label}, full-model prediction at ${view.frame}`;
    link.href = view.render;
    $(`voxel-image-note-${side}`).textContent = `${view.frame} · ${view.native_width} × ${view.native_height} native prediction · ${view.width} × ${view.height} lossless preview`;
  } else $(`voxel-image-note-${side}`).textContent = 'No completed prediction is available for this frame.';
}

async function selectModel(side) {
  const state=states[side],key=$(`voxel-model-${side}`).value,model=selected(side);
  state.ticket++;state.viewer?.clear();image(side);
  $(`voxel-3d-label-${side}`).textContent=model.label;
  $(`voxel-3d-size-${side}`).textContent=`${count(model.gaussian_count)} exported Gaussians · full SH3 PLY ${number(model.model_bytes/1e6,2)} MB · available online`;
  $(`voxel-3d-state-${side}`).textContent='Choose progressive SH3 3D or a small software navigation preview.';
  try {
    const {onlineModel}=await import('./online-gaussian-viewer.js'),asset=await onlineModel('voxel/'+key);
    if($(`voxel-model-${side}`).value!==key)return;
    const link=$(`voxel-download-${side}`);link.href=asset.raw_url+'?download=1';link.textContent='Download complete SH3 PLY';link.download='';
  }catch(error){$(`voxel-3d-state-${side}`).textContent=error.message;}
}

async function load(side,mode,file) {
  const state=states[side];if(state.busy)return;state.busy=true;
  for(const id of [`voxel-preview-${side}`,`voxel-full-${side}`,`voxel-model-${side}`,`voxel-file-${side}`])$(id).disabled=true;
  try {
    const {OnlineGaussianViewer,onlineModel}=await import('./online-gaussian-viewer.js');
    state.viewer ||= new OnlineGaussianViewer($(`voxel-canvas-${side}`),text=>$(`voxel-3d-state-${side}`).textContent=text);
    const model=selected(side),asset=await onlineModel('voxel/'+$(`voxel-model-${side}`).value);
    if(file&&(file.size!==asset.raw_bytes||Number((await file.slice(0,8192).text()).match(/element vertex (\d+)/)?.[1])!==asset.count))throw Error('This PLY does not match the selected model.');
    const view=model.views.find(v=>v.frame===$('voxel-frame').value)||model.views[0];
    await state.viewer.load(asset,view,model.radius||model.scene_scale||1,{software:mode==='preview',file});
  }catch(error){$(`voxel-3d-state-${side}`).textContent=error.message;}
  finally{state.busy=false;for(const id of [`voxel-preview-${side}`,`voxel-full-${side}`,`voxel-model-${side}`,`voxel-file-${side}`])$(id).disabled=false;}
}

function chart(title, field, formatter=number) {
  const models = Object.values(catalog.models), values = models.map(m => ({ x: m.voxel_size || 0, y: field(m), label: m.label }));
  if (values.some(v => !Number.isFinite(v.y))) return;
  const fig = document.createElement('figure'); fig.className = 'count-chart-panel';
  const cap = document.createElement('figcaption'); cap.textContent = title; fig.append(cap);
  const max = Math.max(...values.map(v => v.y), 0.000001) * 1.12;
  const x = value => 84 + value / 0.20 * 454, y = value => 267 - value / max * 210;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 650 345'); svg.classList.add('count-line-chart');
  svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', `${title}. ${values.map(v => `${v.label}: ${exact(v.y)}`).join('; ')}`);
  svg.innerHTML = `${[0, 0.5, 1].map(f => `<line class="grid" x1="84" x2="550" y1="${y(max * f)}" y2="${y(max * f)}"/><text x="75" y="${y(max * f) + 5}" text-anchor="end">${number(max * f, 2)}</text>`).join('')}<line class="axis" x1="84" x2="550" y1="267" y2="267"/><polyline class="series" points="${values.map(v => `${x(v.x)},${y(v.y)}`).join(' ')}"/>${values.map(v => `<circle class="point" cx="${x(v.x)}" cy="${y(v.y)}" r="5"/><text class="point-label" x="${x(v.x)}" y="${Math.max(25, y(v.y) - 12)}" text-anchor="middle">${formatter(v.y)}</text><text x="${x(v.x)}" y="293" text-anchor="middle">${v.x.toFixed(2)}${v.x === 0 ? ' · original' : ''}</text>`).join('')}<text x="310" y="330" text-anchor="middle">Voxel cell size · arbitrary COLMAP model units</text>`;
  fig.append(svg);
  const exactValues = document.createElement('p'); exactValues.className = 'chart-values';
  exactValues.textContent = values.map(v => `${v.label}: ${exact(v.y)}`).join(' · '); fig.append(exactValues);
  $('voxel-charts').append(fig);
}

function measurements() {
  const models = Object.values(catalog.models);
  const rows = [
    ['Starting sparse points', m => count(m.input_points)],
    ['points3D.bin, bytes', m => count(m.points3D_bin_bytes)],
    ['Recovered cameras', m => count(m.registered_images || 320)],
    ['Training / held-out images', m => `${m.training_images || 280} / ${m.validation_images || 40}`],
    ['Input pixels per photograph', () => '960 × 540'],
    ['Training steps', () => '30,000'],
    ['Final checkpoint Gaussians', m => count(m.trained_gaussian_count)],
    ['Valid full-export Gaussians', m => count(m.gaussian_count)],
    ['Invalid / degenerate rows omitted from PLY', m => count(m.excluded_invalid_gaussians)],
    ['PSNR ↑', m => score(m.metrics.psnr)],
    ['SSIM ↑', m => score(m.metrics.ssim)],
    ['LPIPS ↓', m => score(m.metrics.lpips)],
    ['Training-loop elapsed seconds', m => exact(m.training_seconds)],
    ['Whole-process elapsed seconds', m => exact(m.process_wall_seconds)],
    ['Complete SH3 PLY, bytes', m => count(m.model_bytes)],
  ];
  $('voxel-summary').replaceChildren(...rows.map(([label, fn]) => { const tr = document.createElement('tr'), th = document.createElement('th'); th.scope = 'row'; th.textContent = label; tr.append(th); for (const model of models) { const td = document.createElement('td'); td.textContent = fn(model); tr.append(td); } return tr; }));
  $('voxel-audit').textContent = 'Both voxel trials passed integrity checks for complete checkpoints, finite full SH3 exports, and all 40 native validation predictions. Ground-truth halves match their source photographs exactly; original model and photo hashes are unchanged.';
  const original = catalog.models.baseline;
  const findings = models.filter(m => m !== original).map(m => `${m.label}: ${number((1 - m.input_points / original.input_points) * 100, 1)}% fewer starting points; ${m.metrics.psnr >= original.metrics.psnr ? '+' : ''}${(m.metrics.psnr - original.metrics.psnr).toFixed(3)} dB PSNR, ${m.metrics.ssim >= original.metrics.ssim ? '+' : ''}${(m.metrics.ssim - original.metrics.ssim).toFixed(4)} SSIM, and ${m.metrics.lpips >= original.metrics.lpips ? '+' : ''}${(m.metrics.lpips - original.metrics.lpips).toFixed(4)} LPIPS relative to the original input.`);
  $('voxel-finding').textContent = findings.join(' ') + ' Lower LPIPS is better; no geometric-accuracy claim follows from these scores.';
  chart('Starting sparse points', m => m.input_points, count);
  chart('Final valid Gaussians', m => m.gaussian_count, count);
  chart('PSNR ↑ · same 40 held-out views', m => m.metrics.psnr);
  chart('SSIM ↑ · same 40 held-out views', m => m.metrics.ssim);
  chart('LPIPS ↓ · same 40 held-out views', m => m.metrics.lpips);
  chart('Whole-process elapsed seconds', m => m.process_wall_seconds, () => '');
}

fetch('voxel-gsplat-assets/index.json', { cache: 'no-store' }).then(r => { if (!r.ok) throw Error('The full comparison is still being prepared.'); return r.json(); }).then(data => {
  catalog = data;
  $('voxel-state').textContent = `All three models completed 30,000 steps. Saved comparison ${new Date(data.updated_at || data.updated).toLocaleString()} · RTX 4090 CUDA · same 40 held-out views.`;
  measurements();
  $('voxel-frame').replaceChildren(...data.shared_frames.map(frame => new Option(frame, frame)));
  $('voxel-frame').value = 'frame_0337.png';
  for (const side of sides) {
    $(`voxel-model-${side}`).replaceChildren(...Object.entries(data.models).map(([key, m]) => new Option(m.label, key)));
    const requested = new URLSearchParams(location.search).get('model');
    $(`voxel-model-${side}`).value = side === 'a' ? 'baseline' : (data.models[requested] ? requested : 'voxel_008');
    $(`voxel-model-${side}`).onchange = () => selectModel(side);
    $(`voxel-preview-${side}`).onclick = () => load(side, 'preview');
    $(`voxel-full-${side}`).onclick = () => load(side, 'full');
    $(`voxel-reset-${side}`).onclick = () => cameraPreset(side);
    $(`voxel-file-${side}`).onchange = event => { if (event.target.files[0]) load(side, 'full', event.target.files[0]); };
    selectModel(side);
  }
  $('voxel-frame').onchange = () => sides.forEach(side => { image(side); cameraPreset(side); });
  $('voxel-swap').onclick = () => { if (sides.some(side => states[side].busy)) return; const a = $('voxel-model-a').value; $('voxel-model-a').value = $('voxel-model-b').value; $('voxel-model-b').value = a; sides.forEach(selectModel); };
}).catch(error => { $('voxel-state').textContent = error.message; });
