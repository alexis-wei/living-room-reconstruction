"""Copy the completed Step 05 loop-off Gaussian models into the local Final Zip.

Preserves original files, full SH3 PLY bytes and final checkpoints. No training,
resizing, alignment, or network operations. A manifest records SHA-256 checks.
"""
import hashlib
import json
import os
import shutil
import struct
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
SOURCE = WORKSPACE / 'gsplat_local/matching_20261003'
NAME = 'Alexis_Living_Room_iPhone13Pro_4x_Sequential_LoopOff_GSplat'
FINAL = WORKSPACE / 'Final Zip' / NAME
TITLE = 'Alexis Living Room — iPhone 13 Pro — 4× — Sequential, Loop Off — gsplat'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def validate_ply(path, expected):
    with path.open('rb') as stream:
        assert stream.readline() == b'ply\n'
        assert stream.readline() == b'format binary_little_endian 1.0\n'
        lines = []
        while (line := stream.readline()) != b'end_header\n':
            assert line, 'Truncated PLY header'
            lines.append(line.decode().strip())
        offset = stream.tell()
    count = int(next(line.split()[-1] for line in lines if line.startswith('element vertex ')))
    props = [line for line in lines if line.startswith('property ')]
    assert count == expected and len(props) == 59
    assert all(line.startswith('property float ') for line in props)
    assert path.stat().st_size == offset + count * 59 * 4
    values = np.memmap(path, dtype='<f4', offset=offset, shape=(count, 59), mode='r')
    for start in range(0, count, 100000):
        assert np.isfinite(values[start:start+100000]).all(), 'Non-finite Gaussian attributes'
    del values
    return {'gaussians': count, 'float_properties': 59, 'spherical_harmonic_degree': 3,
            'all_ply_values_finite': True, 'bytes': path.stat().st_size}


def camera_record(path):
    raw = path.read_bytes()
    assert struct.unpack_from('<Q', raw)[0] == 1
    camera_id, model_id, width, height = struct.unpack_from('<iiQQ', raw, 8)
    assert model_id == 4 and (width, height) == (540, 960)  # COLMAP OPENCV
    params = struct.unpack_from('<8d', raw, 32)
    return {'camera_id': camera_id, 'model': 'OPENCV', 'width': width, 'height': height,
            'parameters': dict(zip(['fx','fy','cx','cy','k1','k2','p1','p2'], params))}


def main():
    if FINAL.exists():
        raise FileExistsError(f'Existing package preserved: {FINAL}')
    manifest = json.loads((SOURCE / 'datasets.json').read_text())
    report = json.loads((WORKSPACE / 'living-room-reconstruction/reports/matching_gsplat_results.json').read_text())
    assert manifest['source_run_id'] == report['source_run_id'] == 'timed_rerun_20261003_loop_enabled'
    jobs = [job for job in manifest['jobs'] if job['method'] == 'sequential_no_loop']
    assert [job['component'] for job in jobs] == ['1','2','0','3']
    assert jobs[0]['primary_component']
    parent = FINAL.parent
    parent.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{NAME}_preparing_', dir=parent))
    copied = []

    def copy(source, relative):
        dest = staging / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        before = source.stat()
        shutil.copy2(source, dest)
        source_hash = digest(source)
        assert digest(dest) == source_hash
        after = source.stat()
        assert (before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns)
        copied.append({'path': dest.relative_to(staging).as_posix(), 'bytes': dest.stat().st_size,
                       'sha256': source_hash, 'source_byte_identical': True})

    models = []
    registered = set()
    for job in jobs:
        key = job['name']; run = SOURCE / 'runs' / key
        status = json.loads((run / 'status.json').read_text())
        assert status['state'] == 'complete' and status['exit_code'] == 0
        model_report = report['models'][key]
        primary = job['primary_component']
        folder = Path('01_PRIMARY_Component01' if primary else f'02_Disconnected_Components/Component{int(job["component"]):02d}_{job["registered_images"]}Views')
        stem = f'Alexis_Living_Room_iPhone13Pro_4x_Sequential_LoopOff_Component{int(job["component"]):02d}'
        ply_name = stem + ('_PRIMARY' if primary else '') + '_SH3_30k.ply'
        source_ply = run / 'ply/point_cloud_29999.ply'
        validation = validate_ply(source_ply, model_report['gaussian_count'])
        copy(source_ply, folder / ply_name)
        copy(run / 'ckpts/ckpt_29999_rank0.pt', folder / 'Training_Checkpoint' / (stem + '_30k.pt'))
        copy(run / 'cfg.yml', folder / 'Settings/original_training_cfg.yml')
        copy(run / 'status.json', folder / 'Settings/training_status.json')
        copy(run / 'stats/val_step29999.json', folder / 'Settings/validation_metrics.json')
        sparse = (SOURCE / 'datasets' / key / 'sparse').resolve()
        for filename in ['cameras.bin','images.bin','points3D.bin','frames.bin','rigs.bin','project.ini']:
            if (sparse / filename).is_file():
                copy(sparse / filename, folder / 'COLMAP_Cameras' / filename)
        frames = job['training_frames'] + job['validation_frames']
        assert len(frames) == job['registered_images'] and len(set(frames)) == len(frames)
        registered.update(frames)
        split = {'training_frames':job['training_frames'], 'validation_frames':job['validation_frames']}
        (staging / folder / 'Settings/frame_split.json').write_text(json.dumps(split,indent=2)+'\n')
        for index, frame in enumerate(job['validation_frames']):
            copy(run / f'renders/val_step29999_{index:04d}.png', folder / 'Native_Validation_Source_and_Prediction' / (Path(frame).stem + '_source_left_prediction_right.png'))
        item = {'name':stem, 'source_model_key':key, 'component':job['component'], 'primary':primary,
                'registered_views':job['registered_images'], 'training_views':len(job['training_frames']),
                'held_out_views':len(job['validation_frames']), 'sparse_initialization_points':job['sparse_points'],
                'full_quality_ply':(folder/ply_name).as_posix(), 'checkpoint':(folder/'Training_Checkpoint'/(stem+'_30k.pt')).as_posix(),
                'camera':camera_record(sparse/'cameras.bin'), 'undistorted_image_size':job['undistorted_sizes'],
                'metrics':status['metrics'], 'training_seconds':status['seconds'], 'device':status['device'],
                'ply_validation':validation}
        models.append(item)
        print(f'Copied and checked component{job["component"]}: {job["registered_images"]} views, {validation["gaussians"]:,} Gaussians',flush=True)
    assert len(registered) == 493
    src_colmap = (SOURCE / 'datasets' / jobs[0]['name'] / 'sparse').resolve().parents[1]
    copy(src_colmap / 'matching_project.ini', 'Settings/original_sequential_matching.ini')
    copy(src_colmap / 'mapper_project.ini', 'Settings/original_mapper.ini')
    copy(SOURCE / 'holdout_frames.json', 'Settings/source_holdout_frames.json')
    package = {'title':TITLE, 'created_at':datetime.now(timezone.utc).isoformat(),
               'source_site_page':'https://alexis-living-room-reconstruction.hello420892.chatgpt.site/matching#matching-gaussians',
               'source_run_id':manifest['source_run_id'], 'method':'sequential_no_loop',
               'input_images':500, 'input_size':[540,960], 'downsample_factor':4,
               'unique_registered_views':493, 'loop_detection':False, 'models':models,
               'settings':manifest['settings'], 'source_images_included':False,
               'validation_images_included':'All saved native-resolution source/prediction canvases, unmodified',
               'coordinate_warning':'Four separately normalized components are not aligned. Do not concatenate their PLYs as if they share coordinates.',
               'camera_warning':'iPhone-aware OPENCV estimate from nominal26mm initialization and refinement; not measured factory calibration.',
               'copied_files':copied, 'copied_bytes':sum(item['bytes'] for item in copied)}
    (staging/'manifest.json').write_text(json.dumps(package,indent=2)+'\n')
    (staging/'SHA256SUMS.txt').write_text(''.join(f'{item["sha256"]}  {item["path"]}\n' for item in copied))
    readme = f'''# {TITLE}

Start with `01_PRIMARY_Component01/`: its `PRIMARY_SH3_30k.ply` is the default
sequential, loop-off Gaussian model on the exhaustive-versus-sequential page.
It contains 3,430,406 Gaussians from 256 recovered cameras (224 training,
32 held-out views). This is the original completed model, not a new training run.

## Contents

- `01_PRIMARY_Component01/`: main full-quality Gaussian PLY, 30,000-step
  checkpoint, refined COLMAP cameras, exact settings, and all32 native validation
  canvases (source left, predicted reconstruction right).
- `02_Disconnected_Components/`: the three other loop-off models, with matching
  checkpoints/cameras/settings and all their native validation canvases.
- `Settings/`: original sequential-matching and mapping settings and holdout list.
- `manifest.json`: model identities, counts, refined intrinsics, timings, quality
  metrics, and byte-identical-copy checks.
- `SHA256SUMS.txt`: checksums for every copied original artifact.

## Model identity and limits

500 source frames were supplied at4x, 540x960pixels. The loop-off COLMAP result
splits into four components covering493 unique frames. Component1 has256 views,
component2 has150, component0 has77, and component3 has30; some frames overlap.
These models have separate normalized coordinate systems. They are not aligned
parts of one connected room and must not be concatenated directly.

The camera model was OPENCV, initialized from nominal iPhone13Pro1x26mm-equivalent
metadata (fx=fy=693.3333333333,cx=270,cy=480; initial distortion zero), then refined
by COLMAP. This is an estimated calibration. gsplat used the4x PNG photographs
and refined cameras, with parser undistortion to539x959 and data_factor1. Its
rendering camera is pinhole after undistortion. Sequential overlap10,
quadratic_overlap=true, loop_detection=false.

gsplat1.5.3:30,000steps, CUDA RTX4090, seed42, batch1, SHdegree3, packed
rasterization, no pose or appearance optimization. Main-model held-out metrics:
PSNR28.078964, SSIM0.918495, LPIPS0.146841. Training elapsed724.898140216seconds
(12minutes4.898seconds). Image agreement does not establish measured geometric
accuracy; holes/floaters and component gaps remain.

## Opening and reusing

Use a Gaussian-splat PLY viewer supporting the standard full SH3 layout; an
ordinary point-cloud viewer will not show the Gaussian shape/opacity/appearance.
The original full PLYs retain all59 float attributes per Gaussian without browser
sampling or quantization. Checkpoints are supplied for gsplat reuse, with the
original cfg.yml retained verbatim. Those configs contain the original machine's
paths; update paths before training elsewhere. Source training photographs are
kept in the original local dataset and are not duplicated in this delivery.

The validation canvases include source photographs and should be treated as
private. This package is local and has not been uploaded or sent anywhere.
'''
    (staging/'README.md').write_text(readme)
    os.rename(staging, FINAL)
    print(json.dumps({'package':str(FINAL),'models':len(models),'copied_bytes':package['copied_bytes'],
                      'all_copies_byte_identical':True,'unique_registered_views':len(registered)},indent=2),flush=True)


if __name__ == '__main__':
    main()
