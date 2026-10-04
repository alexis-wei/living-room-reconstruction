"""Train the newest matching trial, with identical photo-supervised settings.

Run with the existing .gsplat-env. Full images, checkpoints, logs and PLY stay
outside the code repository. Components with fewer than ten cameras are kept
in COLMAP but excluded from the Gaussian training comparison.
"""
import argparse
import datetime
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

from train_gsplat_queue import environment

WORKSPACE = Path(__file__).resolve().parents[2]
LABELS = {'sequential_no_loop': 'Sequential · loop off',
          'sequential_loop': 'Sequential · loop on', 'exhaustive': 'Exhaustive'}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def other_gpu_jobs():
    result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                      '--format=csv,noheader,nounits'], text=True)
    active = []
    for line in result.splitlines():
        if not line.strip().isdigit():
            continue
        pid = int(line.strip())
        if pid == os.getpid():
            continue
        try:
            command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        except FileNotFoundError:
            continue
        # Open COLMAP desktop windows retain CUDA contexts after SIFT work;
        # these are not standalone reconstruction jobs. Never close the GUI.
        if len(command) > 1 and Path(os.fsdecode(command[0])).name == 'colmap' and command[1] == b'gui':
            continue
        active.append(pid)
    return active


def prepare(root, source):
    import sys
    import numpy as np
    sys.path.insert(0, str(WORKSPACE / '.gsplat-src/gsplat/examples'))
    from datasets.colmap import Parser
    report = json.loads((WORKSPACE / 'living-room-reconstruction/reports/matching_results.json').read_text())
    if report['run_id'] != source.name:
        raise ValueError('The matching report and COLMAP run do not agree')
    images = WORKSPACE / 'living_room_frames/downsample_4x_540x960'
    held_out = [f'frame_{i:04d}.png' for i in range(1, 501, 8)]
    save(root / 'holdout_frames.json', held_out)
    jobs, skipped = [], []
    for method, label in LABELS.items():
        for item in report['methods'][method]['components']:
            key = f'{method}_component_{item["id"]}'
            common = dict(name=key, method=method, label=label, component=item['id'],
                          registered_images=item['registered_images'], sparse_points=item['total_points'])
            if item['registered_images'] < 10 or item['total_points'] < 100:
                skipped.append({**common, 'reason': 'Fewer than 10 recovered cameras or 100 sparse points; sparse result retained, insufficient for a reliable gsplat comparison.'})
                continue
            dataset = root / 'datasets' / key
            dataset.mkdir(parents=True, exist_ok=True)
            for name, target in [('images', images), ('sparse', source / method / 'reconstruction' / item['id'])]:
                link = dataset / name
                if link.is_symlink():
                    assert link.resolve() == target.resolve(), link
                elif not link.exists():
                    link.symlink_to(target, target_is_directory=True)
                else:
                    raise FileExistsError(link)
            parsed = Parser(str(dataset), factor=1, normalize=True, test_every=8)
            assert len(parsed.image_names) == item['registered_images']
            assert len(parsed.points) == item['total_points']
            assert np.isfinite(parsed.camtoworlds).all() and np.isfinite(parsed.points).all()
            validation = [n for n in parsed.image_names if n in held_out]
            training = [n for n in parsed.image_names if n not in held_out]
            assert len(training) >= 2 and validation, key
            common.update(dataset=str(dataset), training_frames=training,
                          validation_frames=validation,
                          primary_component=item['id'] == report['methods'][method]['components'][0]['id'],
                          undistorted_sizes=[list(v) for v in parsed.imsize_dict.values()])
            jobs.append(common)
    # First deliver the main comparison, then every substantial disconnected model.
    jobs.sort(key=lambda job: (not job['primary_component'], list(LABELS).index(job['method']), -job['registered_images']))
    manifest = dict(source_run_id=source.name, prepared_at=now(), jobs=jobs, skipped_components=skipped,
                    settings=dict(gsplat='1.5.3', steps=30000, data_factor=1, seed=42,
                                  batch_size=1, sh_degree=3, packed=True, pose_optimization=False,
                                  appearance_optimization=False, ssim_lambda=0.2,
                                  input='Original 4x PNGs and refined OPENCV COLMAP cameras; gsplat parser undistorts without another downsample.',
                                  validation='Fixed source frames 1,9,17,...,497, excluded from all Gaussian training; cameras/SfM used all registered frames.'))
    save(root / 'datasets.json', manifest)
    return manifest


def train(root, job):
    out = root / 'runs' / job['name']
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / 'status.json'
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous.get('state') == 'complete':
            assert (out / 'ckpts/ckpt_29999_rank0.pt').is_file()
            return
        if previous.get('state') == 'failed':
            raise RuntimeError(f'Preserve failed outputs before retrying: {job["name"]}')
    command = [str(WORKSPACE / '.gsplat-env/bin/python'), str(Path(__file__).with_name('gsplat_entry.py')),
               'default', '--data-dir', job['dataset'], '--data-factor', '1', '--result-dir', str(out),
               '--disable-viewer', '--disable-video', '--packed', '--max-steps', '30000',
               '--eval-steps', '7000', '30000', '--save-steps', '7000', '30000',
               '--save-ply', '--ply-steps', '30000']
    status = dict(state='waiting_for_gpu', requested_at=now(), dataset=job, command=command)
    save(status_path, status)
    while other_gpu_jobs():
        time.sleep(15)
    env = environment()
    env['GSPLAT_HOLDOUT_FILE'] = str(root / 'holdout_frames.json')
    status.update(state='running', started_at=now(), device='NVIDIA RTX 4090 CUDA')
    save(status_path, status)
    print(job['name'], 'started', flush=True)
    start = time.perf_counter_ns()
    with (out / 'training.log').open('w') as log:
        result = subprocess.run(command, cwd=WORKSPACE / '.gsplat-src/gsplat/examples', env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter_ns() - start
    checkpoint = out / 'ckpts/ckpt_29999_rank0.pt'
    metrics = out / 'stats/val_step29999.json'
    state = 'complete' if result.returncode == 0 and checkpoint.is_file() and metrics.is_file() else 'failed'
    status.update(state=state, finished_at=now(), elapsed_ns=elapsed, seconds=elapsed / 1e9,
                  exit_code=result.returncode, metrics=json.loads(metrics.read_text()) if metrics.is_file() else None)
    save(status_path, status)
    print(job['name'], state, f'{elapsed / 1e9:.3f}s', flush=True)
    if state != 'complete':
        raise RuntimeError(f'{job["name"]} training failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=WORKSPACE / 'colmap_gui/iphone13pro_500_4x_comparison/timed_rerun_20261003_loop_enabled')
    parser.add_argument('--output', type=Path, default=WORKSPACE / 'gsplat_local/matching_20261003')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (WORKSPACE / 'gsplat_local/queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = prepare(args.output, args.source_run)
        if args.prepare_only:
            return
        save(args.output / 'queue_status.json', dict(state='running', started_at=now(), jobs=[j['name'] for j in manifest['jobs']]))
        for job in manifest['jobs']:
            train(args.output, job)
        save(args.output / 'queue_status.json', dict(state='complete', finished_at=now(), jobs=[j['name'] for j in manifest['jobs']]))


if __name__ == '__main__':
    main()
