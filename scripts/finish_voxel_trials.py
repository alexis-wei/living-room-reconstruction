"""Finish the two authorized 30,000-step voxel trials serially on the local GPU.

Original runs and short CUDA pilots are never overwritten. A completed trial is
reused; a failed/interrupted trial is retried in a fresh directory. The existing
launcher supplies the shared GPU lock, process checks and fixed hold-out views.
"""
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / 'gsplat_local/voxel_experiments/insta360_4x_20261005'


def write(path, value):
    temporary = path.with_suffix('.queue.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def complete_run(trial):
    for run in sorted((ROOT / 'runs').glob(f'{trial}_headless_*'), reverse=True):
        status_file = run / 'status.json'
        if not status_file.exists():
            continue
        status = json.loads(status_file.read_text())
        if (status.get('state') == 'completed' and status.get('max_steps') == 30000
                and (run / 'ckpts/ckpt_29999_rank0.pt').exists()
                and (run / 'stats/val_step29999.json').exists()):
            return run
    return None


def main():
    status_path = ROOT / 'full_queue_status.json'
    with (ROOT / 'full_queue.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('The full voxel queue is already running; no duplicate was launched.')
        status = {'state': 'running', 'queue_pid': os.getpid(),
                  'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'jobs': {}, 'device': 'NVIDIA RTX 4090 CUDA',
                  'scheduling': 'One GPU-heavy training process at a time; dense work remains canceled.'}
        for trial in ['voxel_008', 'voxel_020']:
            run = complete_run(trial)
            if run:
                status['jobs'][trial] = {'state': 'completed', 'output': str(run), 'reused': True}
                write(status_path, status)
                continue
            status['active_trial'] = trial
            status['jobs'][trial] = {'state': 'starting'}
            write(status_path, status)
            command = [sys.executable, str(Path(__file__).with_name('run_voxel_gsplat.py')),
                       '--trial', trial, '--mode', 'headless']
            result = subprocess.run(command, cwd=WORKSPACE)
            run = complete_run(trial)
            if result.returncode or run is None:
                status['state'] = 'failed'
                status['jobs'][trial] = {'state': 'failed', 'exit_code': result.returncode}
                write(status_path, status)
                raise SystemExit(1)
            status['jobs'][trial] = {'state': 'completed', 'output': str(run), 'reused': False}
            write(status_path, status)
        status.pop('active_trial', None)
        status.update(state='training_completed',
                      finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(status_path, status)
        print('Both full voxel trials have finished. Validate and export before publishing.', flush=True)


if __name__ == '__main__':
    main()
