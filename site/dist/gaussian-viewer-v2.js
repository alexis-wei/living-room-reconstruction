import {SoftwareGaussianRenderer} from './gaussian-software.js';
const $ = id => document.getElementById(id);
const labels = {'1x':'Full resolution','2x':'2× downsample','4x':'4× downsample','8x':'8× downsample','8x_component_2':'8× · separate component 2'};
let catalog, current = '8x', runtime, mesh, token = 0, request;
function preset(i) {
  if (!runtime || !catalog) return;
  const v = catalog[current].views[i], {camera, controls, THREE} = runtime;
  camera.position.fromArray(v.position); camera.up.fromArray(v.up); camera.fov = v.fov; camera.updateProjectionMatrix();
  controls.target.copy(camera.position).add(new THREE.Vector3(...v.forward).multiplyScalar(Math.max(.1, catalog[current].radius * .3))); controls.update();
}
function clearModel() {
  if (mesh) {runtime.scene.remove(mesh); mesh.dispose(); mesh = null;}
  runtime?.software?.clear();
}
function select() {
  const wasLoaded = !!mesh || !!runtime?.software?.loaded;
  token++; request?.abort(); request = null; current = $('gs-model').value; clearModel();
  const m = catalog[current];
  $('gs-size').textContent = `${m.count.toLocaleString()} Gaussians · ${((m.download_bytes || m.bytes) / 1e6).toFixed(1)} MB download · complete browser model retained`;
  $('gs-load').textContent = 'Load interactive 3D'; $('gs-load').disabled = false;
  $('gs-state').textContent = 'Choose a rendered view below, or load the model to explore freely.';
  $('gs-images').replaceChildren(...m.views.map((v, i) => {
    const figure = document.createElement('figure'), link = document.createElement('a'), img = document.createElement('img'), caption = document.createElement('figcaption'), button = document.createElement('button');
    link.href = v.render; link.target = '_blank'; link.rel = 'noopener'; img.src = v.render; img.alt = `${labels[current]} reconstructed view from ${v.frame}`; img.loading = 'lazy'; link.append(img);
    caption.textContent = v.frame + ' · full trained appearance'; button.textContent = 'Go to this camera'; button.type = 'button';
    button.onclick = async () => {const ticket = token; if (!mesh && !runtime?.software?.loaded) await load(); if (ticket === token && (mesh || runtime?.software?.loaded)) preset(i);};
    figure.append(link, caption, button); return figure;
  }));
  if (wasLoaded) load();
}
async function initialize() {
  if (runtime) return;
  const THREE = await import('three'), {OrbitControls} = await import('./vendor/OrbitControls.js');
  const host = $('gs-canvas'), canvas = document.createElement('canvas');
  let renderer, software, SplatMesh, context;
  // A failed WebGL attempt must not lock the separate 2D fallback canvas.
  if ($('gs-renderer').value !== 'software') try {context = canvas.getContext('webgl2', {alpha:false, antialias:false});} catch {}
  if (context) {
    const spark = await import('./vendor/spark.module.js'); SplatMesh = spark.SplatMesh;
    renderer = new THREE.WebGLRenderer({canvas, context, antialias:false, alpha:false}); renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  } else {software = new SoftwareGaussianRenderer(document.createElement('canvas')); renderer = software;}
  host.replaceChildren(renderer.domElement); renderer.domElement.tabIndex = 0;
  renderer.domElement.setAttribute('aria-label', 'Interactive Gaussian model. Drag to orbit, right-drag or shift-drag to pan, scroll to zoom.');
  const scene = new THREE.Scene(); scene.background = new THREE.Color('#18232c');
  if (!software) {const {SparkRenderer} = await import('./vendor/spark.module.js'); scene.add(new SparkRenderer({renderer}));}
  const camera = new THREE.PerspectiveCamera(60, 1, .01, 1000), controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
  renderer.domElement.addEventListener('keydown', event => {
    const direction = new THREE.Vector3().subVectors(camera.position, controls.target);
    if (event.key === '+' || event.key === '=') direction.multiplyScalar(.85);
    else if (event.key === '-') direction.multiplyScalar(1.15);
    else if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') direction.applyAxisAngle(camera.up.clone().normalize(), event.key === 'ArrowLeft' ? .12 : -.12);
    else return;
    event.preventDefault(); camera.position.copy(controls.target).add(direction); controls.update();
  });
  const resize = () => {renderer.setSize(host.clientWidth, host.clientHeight); camera.aspect = host.clientWidth / host.clientHeight; camera.updateProjectionMatrix();};
  const observer = new ResizeObserver(resize); observer.observe(host); resize();
  renderer.setAnimationLoop(() => {controls.update(); renderer.render(scene, camera);});
  runtime = {THREE, SplatMesh, scene, camera, controls, renderer, software, observer};
}
function disposeRuntime() {
  clearModel(); if (!runtime) return;
  runtime.observer.disconnect(); runtime.controls.dispose(); runtime.renderer.setAnimationLoop(null); runtime.renderer.dispose(); runtime = null;
}
async function load() {
  if (!catalog || request) return;
  const ticket = token, m = catalog[current], controller = new AbortController(); request = controller; $('gs-load').disabled = true;
  try {
    $('gs-state').textContent = 'Checking graphics support…'; await initialize(); if (ticket !== token) return; clearModel();
    let offset = 0; const bytes = new Uint8Array(m.bytes);
    for (let i = 0; i < m.chunks.length; i++) {
      if (ticket !== token) return;
      $('gs-state').textContent = `Loading ${labels[current]}: part ${i + 1} of ${m.chunks.length}…${runtime.software ? ' Software preview selected.' : ''}`;
      const response = await fetch(m.chunks[i], {signal:controller.signal}); if (!response.ok) throw Error('Model download failed. Try loading again.');
      const raw = await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer(), chunk = unpack(raw, m.packed_chunks[i].count);
      bytes.set(chunk, offset); offset += chunk.length;
    }
    if (ticket !== token) return; if (offset !== m.bytes || offset !== m.count * 32) throw Error('Incomplete model download.');
    $('gs-state').textContent = 'Preparing Gaussian renderer…';
    if (runtime.software) {
      const shown = await runtime.software.load(bytes); if (ticket !== token) return; preset(0);
      $('gs-state').textContent = `${labels[current]} loaded · software preview: ${shown.toLocaleString()} of ${m.count.toLocaleString()} Gaussians (sampled). Drag to orbit, right-drag to pan, scroll to zoom. Saved views below show the full trained model.`;
    } else {
      const next = new runtime.SplatMesh({fileBytes:bytes, fileName:'model.splat'}); await next.initialized;
      if (ticket !== token) {next.dispose(); return;} mesh = next; runtime.scene.add(mesh); preset(0);
      $('gs-state').textContent = `${labels[current]} loaded · GPU rendering: all ${m.count.toLocaleString()} Gaussians. Drag to orbit, right-drag to pan, scroll/pinch to zoom.`;
    }
    $('gs-load').textContent = 'Reload model';
  } catch (error) {if (ticket === token && error.name !== 'AbortError') $('gs-state').textContent = error.message;}
  finally {if (request === controller) request = null; if (ticket === token) $('gs-load').disabled = false;}
}
$('gs-load').disabled = true; $('gs-load').onclick = load; $('gs-model').onchange = () => catalog && select();
$('gs-reset').onclick = () => preset(0); $('gs-fullscreen').onclick = () => $('gs-canvas').requestFullscreen?.().catch(() => {});
$('gs-renderer').onchange = () => {token++; request?.abort(); request = null; disposeRuntime(); load();};
fetch('gaussians/index.json', {cache:'no-store'}).then(response => {if (!response.ok) throw Error('Model catalog unavailable'); return response.json();}).then(data => {
  catalog = data; const requested = new URLSearchParams(location.search).get('model'); if (requested && catalog[requested]) $('gs-model').value = requested; select();
}).catch(error => {$('gs-state').textContent = error.message;});
const halves = new Float32Array(65536);
for (let i = 0; i < halves.length; i++) {const sign = i & 32768 ? -1 : 1, exp = i >> 10 & 31, mant = i & 1023; halves[i] = sign * (exp === 0 ? 2 ** -14 * (mant / 1024) : exp === 31 ? (mant ? NaN : Infinity) : 2 ** (exp - 15) * (1 + mant / 1024));}
function unpack(raw, n) {
  if (raw.byteLength !== 40 + n * 16) throw Error('Invalid Gaussian chunk');
  const h = new Float32Array(raw, 0, 10), p = new Uint16Array(raw, 40, n * 3), s = new Uint8Array(raw, 40 + n * 6, n * 3), rgba = new Uint8Array(raw, 40 + n * 9, n * 3), q = new Uint8Array(raw, 40 + n * 12, n * 4), out = new ArrayBuffer(n * 32), f = new Float32Array(out), b = new Uint8Array(out);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {f[i * 8 + k] = halves[p[i * 3 + k]] * h[3] + h[k]; const v = s[i * 3 + k]; f[i * 8 + 3 + k] = v ? Math.exp(h[4 + k] + (v - 1) / 254 * h[7 + k]) * h[3] : 0;}
    const color = rgba[i * 3] + rgba[i * 3 + 1] * 256;
    b[i * 32 + 24] = Math.round((color >> 11) * 255 / 31); b[i * 32 + 25] = Math.round(((color >> 5) & 63) * 255 / 63); b[i * 32 + 26] = Math.round((color & 31) * 255 / 31); b[i * 32 + 27] = rgba[i * 3 + 2]; b.set(q.subarray(i * 4, i * 4 + 4), i * 32 + 28);
  }
  return new Uint8Array(out);
}
