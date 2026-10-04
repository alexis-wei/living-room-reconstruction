"""Export same-view predictions for Step 8; full photos and SH3 models stay local."""
import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / 'gsplat_local/full_resolution_fix_20261004'
sys.path.insert(0, str(WORKSPACE / '.gsplat-src/gsplat/examples'))
from datasets.colmap import Parser


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--site-dir', type=Path, default=WORKSPACE / 'living-room-reconstruction/site-step8/dist')
    args = args.parse_args()
    out = args.site_dir / 'fixing-gsplat-assets'
    out.mkdir(parents=True, exist_ok=True)
    preparation = json.loads((ROOT / 'preparation.json').read_text())
    validation = preparation['validation_frames']
    choices = [validation[i] for i in (0, 20, 40, 61)]
    catalog = dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   state=json.loads((ROOT / 'queue_status.json').read_text())['state'],
                   local_viewer='http://127.0.0.1:8792/', shared_frames=choices,
                   cleanup={k: preparation[k] for k in ('original_registered_images', 'retained_images', 'removed_cameras', 'original_sparse_points', 'retained_sparse_points', 'policy')},
                   settings=dict(steps=30000, gsplat='1.5.3', seed=42, data_factor=1, batch_size=1,
                                 packed=True, sh_degree=3, pose_optimization=False,
                                 appearance_optimization=False, ssim_lambda=0.2,
                                 refine_start=500, refine_stop=15000, refine_every=100,
                                 grow_grad2d=0.0002, grow_scale3d=0.01, prune_scale3d=0.1,
                                 prune_opacity=0.005, absgrad=False, gpu='RTX 4090 CUDA',
                                 comparison='Same native 2126×3781 undistorted photographs and the same 62 original held-out views. Training removed six unreliable cameras and 31 sparse points. Intrinsics, retained poses, seed and trainer configuration unchanged.'),
                   models={})
    audit_path = ROOT / 'validation_report_step29999.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        catalog['validation'] = dict(passed=not audit['errors'], checked_at=audit['checked_at'],
                                     checks=len(audit['checks']), errors=len(audit['errors']),
                                     warnings=len(audit['warnings']), held_out_views=62,
                                     native_prediction_dimensions=[2126, 3781],
                                     original_inputs_unchanged=True,
                                     scope='Checkpoint and full SH3 PLY integrity; every native held-out canvas and exact original source-frame identity. No surveyed geometric ground truth.')
    paths = {
        'original': (WORKSPACE / 'gsplat_local/runs/1x', WORKSPACE / 'gsplat_local/datasets/1x', 'Original full resolution', 493, 431),
        'full_resolution_cleaned': (ROOT / 'runs/full_resolution_cleaned', ROOT / 'datasets/full_resolution_cleaned', 'Full resolution · cleaned cameras', 487, 425),
    }
    # Make the original available through the same local-only model server.
    link = ROOT / 'runs/original'
    link.parent.mkdir(exist_ok=True)
    if not link.exists():
        link.symlink_to(paths['original'][0], target_is_directory=True)
    for key, (run, dataset, label, registered, training) in paths.items():
        status_path = run / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {'state': catalog['state']}
        item = dict(label=label, component=0, state=status['state'], registered_images=registered,
                    training_images=training, validation_images=len(validation), views=[], steps=30000,
                    seconds=status.get('seconds'), metrics=None)
        if status['state'] == 'complete':
            metrics = json.loads((run / 'stats/val_step29999.json').read_text())
            ply = run / 'ply/point_cloud_29999.ply'
            with ply.open('rb') as file:
                properties = []
                while line := file.readline():
                    if line.startswith(b'element vertex '):
                        count = int(line.split()[-1])
                    if line.startswith(b'property float '):
                        properties.append(line.split()[-1].decode())
                    if line.strip() == b'end_header':
                        offset = file.tell()
                        break
            values = np.memmap(ply, dtype=[(p, '<f4') for p in properties], offset=offset, mode='r', shape=(count,))
            assert len(properties) == 59 and all(np.isfinite(values[p]).all() for p in properties)
            assert 0 < count <= metrics['num_GS']
            parser = Parser(str(dataset), factor=1, normalize=True, test_every=8)
            item.update(metrics=metrics, gaussian_count=count, trained_gaussian_count=metrics['num_GS'],
                        excluded_invalid_gaussians=metrics['num_GS'] - count,
                        model_bytes=ply.stat().st_size, scene_scale=float(parser.scene_scale),
                        trainer_scene_scale=float(parser.scene_scale * 1.1))
            for frame in choices:
                index = validation.index(frame)
                with Image.open(run / f'renders/val_step29999_{index:04d}.png') as im:
                    predicted = im.crop((im.width // 2, 0, im.width, im.height))
                    native = predicted.size
                    predicted.thumbnail((800, 800), Image.Resampling.LANCZOS)
                    name = f'{key}-{Path(frame).stem}.webp'
                    predicted.save(out / name, 'WEBP', lossless=True, method=6)
                    with Image.open(out / name) as decoded:
                        assert decoded.tobytes() == predicted.tobytes()
                ix = parser.image_names.index(frame)
                pose = parser.camtoworlds[ix]
                K = parser.Ks_dict[parser.camera_ids[ix]]
                item['views'].append(dict(frame=frame, render='fixing-gsplat-assets/' + name,
                                          width=predicted.width, height=predicted.height,
                                          native_width=native[0], native_height=native[1],
                                          position=pose[:3, 3].tolist(), forward=pose[:3, 2].tolist(),
                                          up=(-pose[:3, 1]).tolist(),
                                          fov=float(2 * np.arctan(native[1] / (2 * K[1, 1])) * 180 / np.pi)))
        catalog['models'][key] = item
    (out / 'index.json').write_text(json.dumps(catalog, indent=2) + '\n')
    aggregate = {k: v for k, v in catalog.items() if k not in ('models', 'local_viewer')}
    # Camera positions are private geometry; only camera identifiers/support counts
    # and aggregate settings are suitable for the code-only GitHub repository.
    aggregate['cleanup'] = {**catalog['cleanup'], 'removed_cameras': [
        {k: v for k, v in row.items() if k not in ('radius', 'radius_over_median', 'image_id', 'frame_id')}
        for row in preparation['removed_cameras']]}
    aggregate['models'] = {key: {k: v for k, v in item.items() if k != 'views'} for key, item in catalog['models'].items()}
    (WORKSPACE / 'living-room-reconstruction/reports/gsplat_fix_results.json').write_text(json.dumps(aggregate, indent=2) + '\n')
    print(json.dumps({key: {'state': item['state'], 'gaussians': item.get('gaussian_count'), 'views': len(item['views'])}
                      for key, item in catalog['models'].items()}, indent=2))


if __name__ == '__main__':
    main()
