"""Train the cleaned native-resolution model, preserving original Attempt 01."""
import fcntl
import json
import time
import subprocess
from pathlib import Path
from run_matching_gsplat import train, now, save

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / 'gsplat_local/full_resolution_fix_20261004'


def main():
    preparation = json.loads((ROOT / 'preparation.json').read_text())
    with (WORKSPACE / 'gsplat_local/queue.lock').open('a') as lock:
        save(ROOT / 'queue_status.json', dict(state='waiting_for_queue', requested_at=now()))
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(15)
        job = {k: preparation[k] for k in ('name', 'dataset', 'training_frames', 'validation_frames')}
        job.update(method='camera_cleanup', label='Full resolution · cleaned cameras',
                   component=0, registered_images=preparation['retained_images'],
                   sparse_points=preparation['retained_sparse_points'], primary_component=True)
        save(ROOT / 'queue_status.json', dict(state='running', started_at=now(), jobs=[job['name']]))
        train(ROOT, job)
        save(ROOT / 'queue_status.json', dict(state='complete', finished_at=now(), jobs=[job['name']]))
        subprocess.run([str(WORKSPACE / '.gsplat-env/bin/python'),
                        str(Path(__file__).with_name('export_gsplat_fix.py'))], check=True)


if __name__ == '__main__':
    main()
