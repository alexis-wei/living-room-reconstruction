// Sampled Gaussian preview for browsers without a WebGL2 context.
export class SoftwareGaussianRenderer {
  constructor(canvas) {
    this.domElement = canvas; this.context = canvas.getContext('2d', {alpha:false});
    if (!this.context) throw Error('This browser cannot create a 2D canvas. Saved rendered views remain available below.');
    this.worker = new Worker(new URL('./gaussian-software-worker.js', import.meta.url), {type:'module'});
    this.revision = 0; this.loaded = false; this.pending = false;
    this.worker.onmessage = event => {
      const message = event.data;
      if (message.type === 'loaded') {
        if (message.revision !== this.revision) return;
        this.loaded = true; this.lastView = ''; this.loadResolve?.(message.count); this.loadResolve = this.loadReject = null;
      } else if (message.type === 'frame') {
        this.pending = false;
        if (message.revision !== this.revision || message.view !== this.lastView) return;
        this.context.putImageData(new ImageData(new Uint8ClampedArray(message.pixels), message.width, message.height), 0, 0);
      } else if (message.type === 'error') {this.pending = false; this.loadReject?.(Error(message.message)); this.loadResolve = this.loadReject = null;}
    };
    this.worker.onerror = event => {this.loadReject?.(Error(event.message || 'Software preview could not start.'));};
  }
  setPixelRatio() {}
  setSize(width, height) {const ratio = Math.min(1, 720 / width, 480 / height); this.domElement.width = Math.max(1, Math.round(width * ratio)); this.domElement.height = Math.max(1, Math.round(height * ratio)); this.lastView = ''; this.paintEmpty();}
  paintEmpty() {this.context.fillStyle = '#18232c'; this.context.fillRect(0, 0, this.domElement.width, this.domElement.height);}
  clear() {
    this.revision++; this.loaded = false; this.lastView = '';
    this.loadReject?.(new DOMException('Selection changed', 'AbortError')); this.loadResolve = this.loadReject = null;
    this.worker.postMessage({type:'clear', revision:this.revision}); this.paintEmpty();
  }
  load(bytes) {this.clear(); return new Promise((resolve, reject) => {this.loadResolve = resolve; this.loadReject = reject; this.worker.postMessage({type:'load', revision:this.revision, buffer:bytes.buffer}, [bytes.buffer]);});}
  render(scene, camera) {
    if (!this.loaded || this.pending) return;
    camera.updateMatrixWorld(); const matrix = Array.from(camera.matrixWorldInverse.elements), view = JSON.stringify([matrix, camera.fov, this.domElement.width, this.domElement.height]);
    if (view === this.lastView) return; this.lastView = view; this.pending = true;
    this.worker.postMessage({type:'render', revision:this.revision, view, matrix, fov:camera.fov, width:this.domElement.width, height:this.domElement.height});
  }
  setAnimationLoop(callback) {cancelAnimationFrame(this.animation); if (!callback) return; const tick = () => {callback(); this.animation = requestAnimationFrame(tick);}; tick();}
  dispose() {this.clear(); this.setAnimationLoop(null); this.worker.terminate();}
}
