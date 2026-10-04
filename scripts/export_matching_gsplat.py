"""Export lossless full-model predictions and aggregate matching-gsplat results.

No source photographs or Gaussian geometry are uploaded. The published images
are the prediction half of the official trainer's saved validation canvases.
Full SH3 PLYs and checkpoints stay local and can be opened in the local viewer.
"""
import datetime
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / 'gsplat_local/matching_20261003'
SITE = WORKSPACE / 'living-room-reconstruction/site/dist'
sys.path.insert(0, str(WORKSPACE / '.gsplat-src/gsplat/examples'))
from datasets.colmap import Parser


def main():
    manifest = json.loads((ROOT / 'datasets.json').read_text())
    out = SITE / 'matching-gsplat'
    out.mkdir(exist_ok=True)
    primary = [j for j in manifest['jobs'] if j['primary_component']]
    common = set(primary[0]['validation_frames'])
    for job in primary[1:]:
        common &= set(job['validation_frames'])
    shared = sorted(common)
    # Four chronological viewpoints present and held out in all three main models.
    choices = [shared[i] for i in np.linspace(0, len(shared) - 1, 4, dtype=int)]
    catalog = dict(source_run_id=manifest['source_run_id'], updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   settings=manifest['settings'], shared_frames=choices, models={},
                   skipped_components=manifest['skipped_components'], local_viewer='http://127.0.0.1:8790/')
    score_file = ROOT / 'common_validation_scores.json'
    if score_file.exists():
        scores = json.loads(score_file.read_text())
        assert scores['source_run_id'] == catalog['source_run_id']
        catalog['common_validation'] = {k: v for k, v in scores.items() if k != 'models'}
        catalog['common_validation']['models'] = {key: {k: v for k, v in row.items() if k != 'per_frame'}
                                                for key, row in scores['models'].items()}
    for job in manifest['jobs']:
        run = ROOT / 'runs' / job['name']
        status_file = run / 'status.json'
        status = json.loads(status_file.read_text()) if status_file.exists() else {'state': 'queued'}
        item = {k: v for k, v in job.items() if k not in ('dataset', 'training_frames', 'validation_frames')}
        item.update(state=status['state'], training_images=len(job['training_frames']),
                    validation_images=len(job['validation_frames']), views=[],
                    steps=30000, metrics=None, seconds=None)
        if status['state'] == 'complete':
            metrics = status['metrics']
            assert all(math.isfinite(metrics[k]) for k in ('psnr', 'ssim', 'lpips'))
            ply = run / 'ply/point_cloud_29999.ply'
            assert ply.is_file()
            item.update(metrics=metrics, seconds=status['seconds'], elapsed_ns=status['elapsed_ns'],
                        gaussian_count=metrics['num_GS'], model_bytes=ply.stat().st_size)
            parser = Parser(job['dataset'], factor=1, normalize=True, test_every=8)
            selected = choices if job['primary_component'] else [job['validation_frames'][i] for i in np.linspace(0, len(job['validation_frames']) - 1, 2, dtype=int)]
            for frame in dict.fromkeys(selected):
                if frame not in job['validation_frames']:
                    continue
                i = job['validation_frames'].index(frame)
                source = run / f'renders/val_step29999_{i:04d}.png'
                with Image.open(source) as im:
                    assert im.width % 2 == 0
                    prediction = im.crop((im.width // 2, 0, im.width, im.height))
                    name = f'{job["name"]}-{Path(frame).stem}.webp'
                    prediction.save(out / name, 'WEBP', lossless=True, method=6)
                    with Image.open(out / name) as decoded:
                        assert prediction.tobytes() == decoded.tobytes()
                    width, height = prediction.size
                ix = parser.image_names.index(frame)
                pose = parser.camtoworlds[ix]
                K = parser.Ks_dict[parser.camera_ids[ix]]
                item['views'].append(dict(frame=frame, render='matching-gsplat/' + name,
                                          width=width, height=height,
                                          position=pose[:3, 3].tolist(), forward=pose[:3, 2].tolist(),
                                          up=(-pose[:3, 1]).tolist(), fov=float(2 * np.arctan(height / (2 * K[1, 1])) * 180 / np.pi)))
            item['scene_scale'] = float(parser.scene_scale)
        catalog['models'][job['name']] = item
    (out / 'index.json').write_text(json.dumps(catalog, indent=2) + '\n')
    # Public report contains aggregate settings/numbers only, never camera geometry or images.
    aggregate = {k: v for k, v in catalog.items() if k not in ('models', 'local_viewer')}
    aggregate['models'] = {key: {k: v for k, v in item.items() if k not in ('views', 'scene_scale')}
                           for key, item in catalog['models'].items()}
    (WORKSPACE / 'living-room-reconstruction/reports/matching_gsplat_results.json').write_text(json.dumps(aggregate, indent=2) + '\n')
    print(json.dumps({key: {'state': d['state'], 'gaussians': d.get('gaussian_count'), 'views': len(d['views'])}
                      for key, d in catalog['models'].items()}, indent=2))


if __name__ == '__main__':
    main()
