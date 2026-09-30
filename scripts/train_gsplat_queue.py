"""Run one native-resolution gsplat job at a time after COLMAP releases CUDA."""
import argparse
import datetime
import fcntl
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
LOCAL = WORKSPACE / 'gsplat_local'
PYTHON = WORKSPACE / '.gsplat-env/bin/python'
EXAMPLES = WORKSPACE / '.gsplat-src/gsplat/examples'

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def save(path, data):
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(data, indent=2) + '\n'); temp.replace(path)

def environment():
    env = os.environ.copy(); env.pop('PYTHONPATH', None)
    env.update(CUDA_VISIBLE_DEVICES='0', CUDA_HOME=str(WORKSPACE / '.cuda-12.4'), TORCH_CUDA_ARCH_LIST='8.9', MAX_JOBS='4', TORCH_HOME=str(WORKSPACE / '.gsplat-cache/torch'), TORCH_EXTENSIONS_DIR=str(WORKSPACE / '.gsplat-cache/extensions'), TMPDIR=str(WORKSPACE / '.gsplat-cache/tmp'), MPLCONFIGDIR=str(WORKSPACE / '.gsplat-cache/matplotlib'), XDG_CACHE_HOME=str(WORKSPACE / '.gsplat-cache'), OMP_NUM_THREADS='8', PYTHONUNBUFFERED='1')
    env['PATH'] = str(WORKSPACE / '.cuda-12.4/bin') + ':' + str(PYTHON.parent) + ':' + env['PATH']
    # PyTorch's Ninja generator does not quote the compiler executable itself.
    env['PYTORCH_NVCC'] = shlex.quote(str(WORKSPACE / '.cuda-12.4/bin/nvcc'))
    return env

def gpu_busy():
    # Gating on actual compute processes also prevents overlap with a CUDA job
    # started outside this queue. Desktop graphics processes are not included.
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
    return bool(output.strip())

def colmap_cuda_active():
    for path in (WORKSPACE / 'reconstruction_local').glob('*/status.json'):
        stages = json.loads(path.read_text())['stages']
        if any(entry.get('state') == 'running' and entry.get('device') == 'CUDA' for entry in stages.values()): return True
    return False

def run_job(record, smoke=False):
    name = record['name'] + ('_smoke' if smoke else '')
    out = LOCAL / 'runs' / name; out.mkdir(parents=True, exist_ok=True)
    status_path = out / 'status.json'
    old = json.loads(status_path.read_text()) if status_path.exists() else {}
    if old.get('state') == 'complete': return
    steps = 100 if smoke else 30000
    eval_steps = ['100'] if smoke else ['7000', '30000']
    command = [str(PYTHON), str(Path(__file__).with_name('gsplat_entry.py')), 'default', '--data-dir', record['dataset'], '--data-factor', '1', '--result-dir', str(out), '--disable-viewer', '--disable-video', '--packed', '--max-steps', str(steps), '--eval-steps', *eval_steps, '--save-steps', *eval_steps, '--save-ply', '--ply-steps', str(steps)]
    status = {'name': name, 'state': 'waiting_for_gpu', 'requested_at': now(), 'data_factor': 1, 'max_steps': steps, 'dataset': record, 'command': command}
    save(status_path, status)
    while colmap_cuda_active() or gpu_busy(): time.sleep(30)
    status.update(state='running', started_at=now(), device='NVIDIA RTX 4090 CUDA'); save(status_path, status)
    print(name, 'started', flush=True)
    start = time.monotonic()
    with (out / 'training.log').open('w') as log:
        result = subprocess.run(command, cwd=EXAMPLES, env=environment(), stdout=log, stderr=subprocess.STDOUT)
    checkpoints = sorted(str(path.relative_to(out)) for path in (out / 'ckpts').glob('*.pt'))
    metrics = sorted(str(path.relative_to(out)) for path in (out / 'stats').glob('val*.json'))
    state = 'complete' if result.returncode == 0 and checkpoints and metrics else 'failed'
    status.update(state=state, finished_at=now(), seconds=round(time.monotonic()-start, 2), exit_code=result.returncode, checkpoints=checkpoints, validation_metrics=metrics)
    save(status_path, status); print(name, state, flush=True)
    if state != 'complete': raise RuntimeError(f'{name} failed; inspect {out / "training.log"}')

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--smoke-only', action='store_true'); args = parser.parse_args()
    LOCAL.mkdir(exist_ok=True)
    with (LOCAL / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = json.loads((LOCAL / 'datasets.json').read_text())
        indexed = {record['name']: record for record in records}
        run_job(indexed['8x'], smoke=True)
        if args.smoke_only: return
        for name in ['8x', '4x', '2x', '1x']:
            run_job(indexed[name])
        for record in records:
            if '_component_' in record['name']: run_job(record)
        save(LOCAL / 'queue_complete.json', {'completed_at': now(), 'jobs': [record['name'] for record in records]})

if __name__ == '__main__': main()
