"""Run the official gsplat trainer on separate, valid voxel-thinned inputs.

Interactive mode starts paused. Test mode is a short CUDA compatibility check,
not a quality comparison with the completed 30,000-step baseline.
"""
import argparse
import datetime
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from train_gsplat_queue import environment
from run_matching_gsplat import other_gpu_jobs

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / 'gsplat_local/voxel_experiments/insta360_4x_20261005'


def save(path, record):
    temporary = path.with_suffix('.launcher.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', choices=['voxel_008', 'voxel_020'], default='voxel_008')
    parser.add_argument('--mode', choices=['interactive', 'headless', 'test'], default='interactive')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--port', type=int, default=8894)
    parser.add_argument('--autostart', action='store_true')
    parser.add_argument('--stop-current', action='store_true', help='Stop only the interactive experiment launched by this script')
    args = parser.parse_args()
    if args.stop_current:
        latest = json.loads((ROOT / 'latest_interactive.json').read_text())
        record = json.loads(Path(latest['status']).read_text())
        pid = record['launcher_pid']
        cmdline = Path(f'/proc/{pid}/cmdline')
        if not cmdline.exists():
            print('The local interactive experiment is already stopped.')
            return
        arguments = [os.fsdecode(p) for p in cmdline.read_bytes().split(b'\0') if p]
        process_cwd = Path(f'/proc/{pid}/cwd').resolve()
        if len(arguments) < 2 or (process_cwd / arguments[1]).resolve() != Path(__file__).resolve():
            raise SystemExit('Saved PID now belongs to another process; it was left untouched.')
        os.kill(pid, signal.SIGTERM)
        print('Sent stop to the saved gsplat experiment only. Its files are preserved.')
        return
    if args.steps is not None and args.steps < 1:
        parser.error('--steps must be positive')
    dataset = ROOT / 'datasets' / args.trial
    if not any((dataset / path).is_file() for path in ['sparse/0/points3D.bin', 'sparse/points3D.bin']):
        raise FileNotFoundError(f'Prepare the voxel inputs first: {dataset}')
    steps = args.steps or (100 if args.mode == 'test' else 30000)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = ROOT / 'runs' / f'{args.trial}_{args.mode}_{timestamp}'
    out.mkdir(parents=True)
    status_path = out / 'status.json'
    entry = 'voxel_interface_entry.py' if args.mode == 'interactive' else 'gsplat_entry.py'
    evaluation_steps = ['7000', '30000'] if steps == 30000 else [str(steps)]
    command = [str(WORKSPACE / '.gsplat-env/bin/python'), str(Path(__file__).with_name(entry)),
               'default', '--data-dir', str(dataset), '--data-factor', '1', '--result-dir', str(out),
               '--disable-video', '--packed', '--max-steps', str(steps),
               '--eval-steps', *evaluation_steps, '--save-steps', *evaluation_steps,
               '--save-ply', '--ply-steps', str(steps), '--port', str(args.port)]
    if args.mode != 'interactive':
        command.append('--disable-viewer')
    env = environment()
    env.update(GSPLAT_HOLDOUT_FILE=str(WORKSPACE / 'insta360_local/recovery/gsplat/holdout_frames.json'),
               GSPLAT_START_PAUSED='0' if args.autostart else '1',
               GSPLAT_INTERACTIVE_STATUS=str(status_path))
    with (WORKSPACE / 'gsplat_local/queue.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another gsplat queue owns the GPU lock. Stop that job before launching another.')
        busy = other_gpu_jobs()
        if busy:
            raise SystemExit(f'Other GPU compute processes are active: {busy}. No training was started.')
        record = dict(state='starting', trial=args.trial, mode=args.mode, max_steps=steps,
                      dataset=str(dataset), output=str(out), command=command,
                      requested_at=timestamp, launcher_pid=os.getpid(),
                      device='NVIDIA RTX 4090 CUDA', viewer_url=f'http://127.0.0.1:{args.port}' if args.mode == 'interactive' else None,
                      comparison_note='Short test validates execution only; compare quality after identical 30000-step runs.')
        save(status_path, record)
        save(ROOT / f'latest_{args.mode}.json', dict(status=str(status_path), output=str(out), trial=args.trial))
        print(f'Output: {out}', flush=True)
        if record['viewer_url']:
            print(f"Open {record['viewer_url']} · starts paused · click Training → Resume", flush=True)
        start = time.perf_counter_ns()
        with (out / 'training.log').open('w') as log:
            child = subprocess.Popen(command, cwd=WORKSPACE / '.gsplat-src/gsplat/examples', env=env,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            def stop_child(signum, frame):
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            signal.signal(signal.SIGTERM, stop_child)
            signal.signal(signal.SIGINT, stop_child)
            result = child.wait()
        elapsed = time.perf_counter_ns() - start
        record = json.loads(status_path.read_text())
        complete = (out / f'ckpts/ckpt_{steps-1}_rank0.pt').exists() and (out / f'stats/val_step{steps-1:04d}.json').exists()
        record.update(state='completed' if complete else 'stopped' if result < 0 else 'failed',
                      exit_code=result, process_wall_seconds=elapsed / 1e9, elapsed_ns=elapsed,
                      viewer_closed=True)
        save(status_path, record)
        print(record['state'], f'{elapsed/1e9:.3f}s', flush=True)
        if result != 0 and not complete:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
