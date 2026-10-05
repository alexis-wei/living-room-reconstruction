"""CPU integrity and matched-heldout audit for completed voxel gsplat trials.

This validates all rows of complete SH3 PLYs and all 40 native validation
canvases. It preserves and reports the original checkpoint's non-finite scales.
No validation assets are uploaded and no model or source photograph is edited.
"""
import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import yaml

from finish_voxel_trials import complete_run
from export_insta360_gsplat import validate_ply

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / 'gsplat_local/voxel_experiments/insta360_4x_20261005'
BASELINE = WORKSPACE / 'insta360_local/recovery/gsplat/runs/4x_component_0'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def configs_equal(a, b):
    # BaseLoader reads upstream's tagged YAML as data without instantiating
    # Python objects. Viewer ports and the intended dataset/output differ.
    left = yaml.load(a.read_text(), Loader=yaml.BaseLoader)
    right = yaml.load(b.read_text(), Loader=yaml.BaseLoader)
    for key in ['data_dir', 'result_dir', 'port']:
        left.pop(key, None)
        right.pop(key, None)
    return left == right


def main():
    torch.set_num_threads(4)
    manifest = json.loads((ROOT / 'inputs_manifest.json').read_text())
    images = Path(manifest['source_dataset']) / 'images'
    expected_frames = manifest['validation_frames']
    training = manifest['training_frames']
    assert len(expected_frames) == 40 and len(training) == 280
    assert not set(expected_frames) & set(training)
    report = {'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'cpu_only': True, 'models': {}, 'warnings': [],
              'source_hashes_unchanged': True,
              'common_validation_frames': expected_frames,
              'training_images': len(training), 'validation_images': len(expected_frames)}
    for name, record in manifest['source_binary_files'].items():
        path = Path(manifest['source_sparse']) / name
        assert path.stat().st_size == record['bytes'] and digest(path) == record['sha256'], name
    for name, record in manifest['source_image_files'].items():
        path = images / name
        assert path.stat().st_size == record['bytes'] and digest(path) == record['sha256'], name
    jobs = {job['name']: job for job in manifest['jobs']}
    for name in ['baseline', 'voxel_008', 'voxel_020']:
        run = BASELINE if name == 'baseline' else complete_run(name)
        if run is None:
            raise RuntimeError(f'{name}: a full 30,000-step completed run is required')
        assert configs_equal(BASELINE / 'cfg.yml', run / 'cfg.yml'), name
        job = jobs[name]
        dataset = Path(job['dataset'])
        sparse = dataset / 'sparse/0' if (dataset / 'sparse/0').exists() else dataset / 'sparse'
        for filename, record in job['binary_files'].items():
            path = sparse / filename
            assert path.stat().st_size == record['bytes'] and digest(path) == record['sha256'], (name, filename)
        checkpoint = run / 'ckpts/ckpt_29999_rank0.pt'
        data = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert data['step'] == 29999, (name, data['step'])
        splats = data['splats']
        n = len(splats['means'])
        assert splats['means'].shape == (n, 3)
        assert splats['scales'].shape == (n, 3)
        assert splats['quats'].shape == (n, 4)
        assert splats['sh0'].shape == (n, 1, 3)
        assert splats['shN'].shape == (n, 15, 3)
        invalid_values = {}
        invalid_types = {}
        row_finite = torch.ones(n, dtype=torch.bool)
        for key, tensor in splats.items():
            assert len(tensor) == n
            invalid_values[key] = int((~torch.isfinite(tensor)).sum())
            invalid_types[key] = {label: int(test(tensor).sum()) for label, test in
                                  [('nan', torch.isnan), ('negative_infinity', torch.isneginf),
                                   ('positive_infinity', torch.isposinf)]}
            row_finite &= torch.isfinite(tensor).reshape(n, -1).all(dim=1)
        ply = run / 'ply/point_cloud_29999.ply'
        vertices = validate_ply(ply)
        assert vertices == int(row_finite.sum()), (name, vertices, int(row_finite.sum()))
        metrics = json.loads((run / 'stats/val_step29999.json').read_text())
        train_stats = json.loads((run / 'stats/train_step29999_rank0.json').read_text())
        assert metrics['num_GS'] == train_stats['num_GS'] == n
        assert all(math.isfinite(metrics[k]) for k in ['psnr', 'ssim', 'lpips', 'ellipse_time'])
        renders = []
        for index, frame in enumerate(expected_frames):
            path = run / f'renders/val_step29999_{index:04d}.png'
            with Image.open(path) as canvas, Image.open(images / frame) as source:
                assert canvas.size == (1920, 540) and source.size == (960, 540)
                pixels = np.asarray(canvas.convert('RGB'))
                assert np.array_equal(pixels[:, :960], np.asarray(source.convert('RGB'))), (name, frame)
                # The fixed holdout source half must also match the original run,
                # not merely a different photograph at the same array index.
                with Image.open(BASELINE / f'renders/val_step29999_{index:04d}.png') as original:
                    assert np.array_equal(pixels[:, :960], np.asarray(original)[:, :960]), (name, frame)
                assert np.isfinite(pixels[:, 960:]).all()
            renders.append({'frame': frame, 'native_size': [960, 540],
                            'ground_truth_matches_original_source_exactly': True,
                            'bytes': path.stat().st_size})
        excluded = n - vertices
        if excluded:
            report['warnings'].append(f'{name}: original checkpoint retained with {excluded} non-finite rows; complete finite SH3 PLY excludes these rows explicitly.')
        item = {'run': str(run), 'checkpoint': str(checkpoint), 'full_ply': str(ply),
                'checkpoint_step': 29999, 'checkpoint_gaussians': n,
                'checkpoint_all_finite': not any(invalid_values.values()),
                'checkpoint_nonfinite_values': invalid_values,
                'checkpoint_nonfinite_types': invalid_types,
                'excluded_nonfinite_rows': excluded, 'full_ply_gaussians': vertices,
                'full_ply_bytes': ply.stat().st_size, 'all_ply_values_finite': True,
                'sh_degree': 3, 'trainer_settings_match_baseline_except_paths_port': True,
                'metrics': metrics, 'train_stats': train_stats,
                'native_validation_renders': renders}
        report['models'][name] = item
        print(name, n, vertices, '40 exact native holdouts', invalid_values, flush=True)
        del data, splats
    report['state'] = 'passed_with_preserved_baseline_warning' if report['warnings'] else 'passed'
    (ROOT / 'validation_final.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
