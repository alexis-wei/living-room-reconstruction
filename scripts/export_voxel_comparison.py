"""Stage bounded private voxel comparison previews, keeping full models local.

Run in .gsplat-env after validate_voxel_final.py. This exporter writes only to
gsplat_local staging: it never modifies or publishes the shared Site checkout.
The four held-out prediction halves remain 960x540 and lossless; source halves
are not uploaded. Small sampled SH0 SPLAT proxies are labeled explicitly.
"""
import datetime
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from finish_voxel_trials import complete_run

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / 'gsplat_local/voxel_experiments/insta360_4x_20261005'
STAGE = ROOT / 'staging'
ASSETS = STAGE / 'voxel-gsplat-assets'
BASELINE = WORKSPACE / 'insta360_local/recovery/gsplat/runs/4x_component_0'
FRAMES = ['frame_0081.png', 'frame_0217.png', 'frame_0337.png', 'frame_0497.png']
sys.path.insert(0, str(WORKSPACE / '.gsplat-src/gsplat/examples'))
from datasets.colmap import Parser


def write(path, data):
    temporary = path.with_suffix('.export.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def ply_rows(path):
    with path.open('rb') as stream:
        props = []
        while line := stream.readline():
            words = line.decode('ascii').strip().split()
            if words[:2] == ['element', 'vertex']:
                count = int(words[2])
            if words[:2] == ['property', 'float']:
                props.append((words[2], '<f4'))
            if words == ['end_header']:
                offset = stream.tell()
                break
    dtype = np.dtype(props)
    assert len(props) == 59 and path.stat().st_size == offset + count * dtype.itemsize
    return np.memmap(path, dtype=dtype, offset=offset, mode='r', shape=(count,))


def export_proxy(name, full_ply, sample_count):
    values = ply_rows(full_ply)
    count = min(sample_count, len(values))
    chosen = np.sort(np.random.default_rng(42).choice(len(values), size=count, replace=False))
    source = values[chosen]
    dtype = np.dtype([('pos', '<f4', (3,)), ('scale', '<f4', (3,)),
                      ('rgba', 'u1', (4,)), ('quat', 'u1', (4,))])
    assert dtype.itemsize == 32
    out = np.empty(count, dtype=dtype)
    out['pos'] = np.column_stack([source[k] for k in ['x', 'y', 'z']])
    out['scale'] = np.exp(np.column_stack([source[f'scale_{k}'] for k in range(3)]))
    colors = np.clip(.5 + .28209479177387814 * np.column_stack(
        [source[f'f_dc_{k}'] for k in range(3)]), 0, 1)
    opacity = 1 / (1 + np.exp(-np.clip(source['opacity'], -80, 80)))
    out['rgba'] = np.round(np.column_stack([colors, opacity]) * 255).astype('u1')
    quat = np.column_stack([source[f'rot_{k}'] for k in range(4)])
    quat /= np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-20)
    out['quat'] = np.clip(quat * 128 + 128, 0, 255).astype('u1')
    assert np.isfinite(out['pos']).all() and np.isfinite(out['scale']).all()
    raw = out.tobytes()
    file = ASSETS / f'{name}.splat.gz'
    file.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    assert gzip.decompress(file.read_bytes()) == raw
    positions = np.column_stack([values[k] for k in ['x', 'y', 'z']])
    center = np.median(positions, axis=0)
    radius = float(np.quantile(np.linalg.norm(positions - center, axis=1), .9))
    return ({'url': 'voxel-gsplat-assets/' + file.name, 'count': count,
             'total_points': len(values), 'bytes': len(raw),
             'download_bytes': file.stat().st_size, 'format': 'splat32-gzip',
             'sampled': count < len(values), 'sh_degree': 0,
             'sampling': 'Deterministic uniform sample without replacement, seed 42; no opacity or size pruning.',
             'quantization': 'Float32 centers/scales, uint8 DC color/alpha/quaternion; no higher-order SH.'},
            center.tolist(), radius)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'inputs_manifest.json').read_text())
    audit = json.loads((ROOT / 'validation_final.json').read_text())
    assert audit['state'].startswith('passed') and audit['source_hashes_unchanged']
    catalog = {'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'state': 'complete', 'title': 'Insta360 4x: voxel initialization comparison',
               'local_viewer': 'http://127.0.0.1:8794/',
               'common_frame_count': 40, 'shared_frames': FRAMES, 'models': {},
               'settings': {'gsplat': '1.5.3', 'steps': 30000, 'seed': 42,
                            'data_factor': 1, 'input_dimensions': [960, 540],
                            'camera_model': 'pinhole', 'source_camera_model': 'OPENCV_FISHEYE',
                            'images': 'Identical calibrated undistorted 4x images',
                            'registered_images': 320, 'training_images': 280, 'validation_images': 40,
                            'batch_size': 1, 'packed': True, 'sh_degree': 3, 'init_type': 'sfm',
                            'pose_opt': False, 'app_opt': False, 'ssim_lambda': .2,
                            'strategy': 'DefaultStrategy', 'refine_start': 500,
                            'refine_stop': 15000, 'refine_every': 100,
                            'grow_grad2d': .0002, 'grow_scale3d': .01,
                            'prune_scale3d': .1, 'prune_opacity': .005,
                            'gpu': 'NVIDIA RTX 4090 CUDA', 'execution': 'Sequential headless training'},
               'validation': {'summary': 'All three final step-29999 checkpoints and complete finite SH3 PLYs checked; all 40 native held-out source halves match the original photographs exactly.',
                              'source_hashes_unchanged': True, 'full_ply_all_finite': True,
                              'matched_heldout_frames': 40, 'settings_match_except_input_and_output': True,
                              'warnings': audit['warnings']},
               'limits': ['Only initialization points were voxel thinned; surviving coordinates, intrinsics, poses and input pixels were preserved.',
                          'Voxel sizes are arbitrary COLMAP model units, not measured meters.',
                          'Voxel thinning changes both point density and nearest-neighbor initial Gaussian scale.',
                          'The original inputs include coincident points and 714 zero initial Gaussian scales; both voxel inputs remove these degeneracies. This is not a pure point-count-only test.',
                          'Adaptive densification may produce more final Gaussians than the number of starting points.',
                          'The geometry-screened source recovers 320 of 500 views; omitted room coverage is shared by all three models.',
                          'Native held-out predictions and metrics use each complete trained SH3 model; online 3D previews are sampled SH0 proxies.',
                          'Quality measures appearance against withheld photographs, not surveyed 3D geometry.']}
    local_catalog = {'local_viewer': 'http://127.0.0.1:8794/', 'models': {}}
    original_validation = manifest['validation_frames']
    input_audit = json.loads((ROOT / 'validation_input.json').read_text())
    full_plys = {}
    labels = {'baseline': 'No voxel adjustment', 'voxel_008': 'Voxel 0.08 · moderate',
              'voxel_020': 'Voxel 0.20 · coarse'}
    for job in manifest['jobs']:
        name = job['name']
        run = BASELINE if name == 'baseline' else complete_run(name)
        assert run
        result = audit['models'][name]
        status = json.loads((run / 'status.json').read_text())
        parser = Parser(job['dataset'], factor=1, normalize=True, test_every=8)
        assert [frame for frame in parser.image_names if frame in set(original_validation)] == original_validation
        full_plys[name] = Path(result['full_ply'])
        item = {'label': labels[name], 'state': 'complete', 'component': 0,
                'voxel_size': job['voxel_size'], 'voxel_units': 'COLMAP model units (unscaled)',
                'input_points': job['points'],
                'points3D_bin_bytes': job['binary_files']['points3D.bin']['bytes'],
                'mean_input_reprojection_error_px': input_audit['modern_models'][name]['mean_reprojection_error_px'],
                'input_mean_track_length': input_audit['modern_models'][name]['mean_track_length'],
                'initial_zero_scale_gaussians': input_audit['legacy_parser'][name]['zero_initial_scales'],
                'registered_images': 320, 'cameras': 320, 'training_images': 280,
                'validation_images': 40, 'steps': 30000,
                'gaussian_count': result['full_ply_gaussians'],
                'trained_gaussian_count': result['checkpoint_gaussians'],
                'excluded_invalid_gaussians': result['excluded_nonfinite_rows'],
                'model_bytes': result['full_ply_bytes'],
                'checkpoint_all_finite': result['checkpoint_all_finite'],
                'training_seconds': result['train_stats']['ellipse_time'],
                'training_seconds_scope': 'Official cumulative trainer elapsed time through step 29999; includes any earlier in-loop evaluation/checkpoint overhead.',
                'process_wall_seconds': status.get('process_wall_seconds', status.get('seconds')),
                'process_wall_elapsed_ns': status.get('elapsed_ns'),
                'metrics': result['metrics'],
                'metrics_scope': 'Official full float-buffer SH3 validation averages over the same 40 native960x540 held-out views.',
                'views': [],
                'scene_scale': float(parser.scene_scale), 'world_transform': parser.transform.tolist(),
                'full_sh_degree': 3,
                'full_local_url': f'http://127.0.0.1:8794/model/{name}.ply'}
        for frame in FRAMES:
            ix = original_validation.index(frame)
            with Image.open(run / f'renders/val_step29999_{ix:04d}.png') as canvas:
                assert canvas.size == (1920, 540)
                prediction = canvas.crop((960, 0, 1920, 540)).convert('RGB')
                if name == 'baseline':
                    url = f'insta360-recovery-assets/4x_component_0-{Path(frame).stem}.webp'
                    # Reuse the existing private prediction only after checking
                    # that its native pixels are from this exact completed run.
                    existing = WORKSPACE / 'living-room-reconstruction/site/dist' / url
                    with Image.open(existing) as decoded:
                        assert decoded.size == prediction.size
                        assert decoded.convert('RGB').tobytes() == prediction.tobytes()
                else:
                    file = ASSETS / f'{name}-{Path(frame).stem}.webp'
                    prediction.save(file, 'WEBP', lossless=True, method=6)
                    with Image.open(file) as decoded:
                        assert decoded.tobytes() == prediction.tobytes()
                    url = 'voxel-gsplat-assets/' + file.name
            pi = parser.image_names.index(frame)
            pose = parser.camtoworlds[pi]
            K = parser.Ks_dict[parser.camera_ids[pi]]
            item['views'].append({'frame': frame, 'render': url, 'width': 960, 'height': 540,
                                  'native_width': 960, 'native_height': 540,
                                  'position': pose[:3, 3].tolist(), 'forward': pose[:3, 2].tolist(),
                                  'up': (-pose[:3, 1]).tolist(),
                                  'fov': float(2 * np.arctan(540 / (2 * K[1, 1])) * 180 / np.pi)})
        catalog['models'][name] = item
        local_catalog['models'][name] = {**item, 'ply_path': str(full_plys[name]),
                                         'checkpoint_path': result['checkpoint'], 'dataset_path': job['dataset']}
    # Keep the complete photographic/model artifacts outside the hosting archive.
    # Reduce proxy sample counts, never photograph resolution, if the new derived
    # preview budget requires it. Actual full-model metrics are unaffected.
    for sample_count in [15000, 12000, 10000, 7500, 6500, 5000]:
        for name, ply in full_plys.items():
            preview, center, radius = export_proxy(name, ply, sample_count)
            catalog['models'][name].update(preview=preview, center=center, radius=radius)
            local_catalog['models'][name].update(preview=preview, center=center, radius=radius)
        raw_asset_bytes = sum(f.stat().st_size for f in ASSETS.iterdir() if f.is_file())
        if raw_asset_bytes <= 2_000_000:
            break
    assert raw_asset_bytes <= 2_000_000, raw_asset_bytes
    catalog['preview_policy'] = {'predictions': 'Four identical held-out viewpoints from each complete SH3 model, native 960x540 pixels, lossless WebP, no resizing.',
                                 'interactive_3d': 'Explicitly sampled finite Gaussian proxies using SH0 colors; complete SH3 files remain available only locally.',
                                 'derived_asset_bytes': raw_asset_bytes}
    write(ASSETS / 'index.json', catalog)
    write(ROOT / 'local_catalog.json', local_catalog)
    aggregate = {k: v for k, v in catalog.items() if k not in ['models', 'local_viewer']}
    aggregate['models'] = {name: {k: v for k, v in item.items()
                                if k not in ['views', 'center', 'radius', 'scene_scale', 'world_transform',
                                             'full_local_url', 'preview']}
                           for name, item in catalog['models'].items()}
    write(STAGE / 'voxel_gsplat_results.json', aggregate)
    assets_bytes = sum(f.stat().st_size for f in ASSETS.iterdir() if f.is_file())
    write(ROOT / 'export_manifest.json', {'state': 'complete', 'assets_bytes': assets_bytes,
                                         'asset_files': [f.name for f in sorted(ASSETS.iterdir()) if f.is_file()],
                                         'staging': str(STAGE), 'local_catalog': str(ROOT / 'local_catalog.json')})
    queue_path = ROOT / 'full_queue_status.json'
    if queue_path.exists():
        queue = json.loads(queue_path.read_text())
        if queue['state'] == 'training_completed':
            queue.update(state='validated_and_exported',
                         validation=str(ROOT / 'validation_final.json'),
                         export_manifest=str(ROOT / 'export_manifest.json'))
            write(queue_path, queue)
    print(json.dumps({'asset_bytes': assets_bytes, 'models': {
        name: {'input': m['input_points'], 'gaussians': m['gaussian_count'],
               'proxy': m['preview']['count'], 'metrics': m['metrics']}
        for name, m in catalog['models'].items()}}, indent=2))


if __name__ == '__main__':
    main()
