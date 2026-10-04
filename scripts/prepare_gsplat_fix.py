"""Copy the original full-resolution model and remove poorly supported cameras.

Run with modern PyCOLMAP via the existing project-local .colmap-tools path.
Source photos and original reconstruction are never modified.
"""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pycolmap

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / 'gsplat_local/full_resolution_fix_20261004'
SOURCE = WORKSPACE / 'reconstruction_local/1x/dense/sparse'


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / 'preparation.json'
    if manifest_path.exists():
        print(manifest_path.read_text())
        return
    model = pycolmap.Reconstruction(SOURCE)
    audit = json.loads((WORKSPACE / 'gsplat_local/diagnostics/attempt01_full_resolution/camera_audit.json').read_text())['1x']
    centers = np.asarray(audit['centers'])
    radii = np.linalg.norm(centers - centers.mean(0), axis=1)
    median_radius = float(np.median(radii))
    by_name = dict(zip(audit['image_names'], radii))
    original_names = sorted(im.name for im in model.images.values() if im.has_pose)
    held_out = original_names[::8]
    originals = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in SOURCE.glob('*.bin')}
    original_points = model.num_points3D()
    original_cameras = {i: cam.params.copy() for i, cam in model.cameras.items()}
    removed = []
    # Count support before any deletion so that filtering cannot cascade.
    for im in list(model.images.values()):
        if not im.has_pose:
            continue
        support = im.num_points3D
        reasons = []
        if support < 20:
            reasons.append('fewer than 20 surviving sparse-point observations')
        if by_name[im.name] > 5 * median_radius:
            reasons.append('camera radius exceeds five times the median radius')
        if reasons:
            removed.append(dict(frame=im.name, image_id=im.image_id, frame_id=im.frame_id,
                                point_observations=support, radius=float(by_name[im.name]),
                                radius_over_median=float(by_name[im.name] / median_radius),
                                reasons=reasons))
    for row in removed:
        model.deregister_frame(row['frame_id'])
    dataset = ROOT / 'datasets/full_resolution_cleaned'
    sparse = dataset / 'sparse'
    sparse.mkdir(parents=True, exist_ok=True)
    model.write(sparse)
    images = dataset / 'images'
    target = WORKSPACE / 'reconstruction_local/1x/dense/images'
    if not images.exists():
        images.symlink_to(target, target_is_directory=True)
    retained = sorted(im.name for im in model.images.values() if im.has_pose)
    validation = [n for n in held_out if n in retained]
    assert validation == held_out, 'Cleanup changed the original held-out set'
    assert len(retained) == len(original_names) - len(removed)
    assert all(point.track.length() >= 2 for point in model.points3D.values())
    assert all(np.array_equal(original_cameras[i], cam.params)
               for i, cam in model.cameras.items())
    assert all(hashlib.sha256((SOURCE / n).read_bytes()).hexdigest() == digest
               for n, digest in originals.items())
    (ROOT / 'holdout_frames.json').write_text(json.dumps(held_out, indent=2) + '\n')
    manifest = dict(prepared_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    name='full_resolution_cleaned', dataset=str(dataset),
                    original_registered_images=len(original_names),
                    retained_images=len(retained), removed_cameras=removed,
                    original_sparse_points=original_points,
                    retained_sparse_points=model.num_points3D(),
                    training_frames=[n for n in retained if n not in held_out],
                    validation_frames=validation,
                    original_source_hashes=originals,
                    policy=dict(min_point_observations=20, max_radius_over_median=5,
                                support_counted_before_deletions=True,
                                point_handling='COLMAP removes deleted-image observations and points that lose sufficient track support',
                                intrinsics='Original undistorted PINHOLE intrinsics unchanged; estimated SIMPLE_RADIAL calibration upstream',
                                photos='Same native-resolution undistorted PNGs; data_factor=1',
                                comparison='Same 62 original validation views and seed 42. Only camera/track cleanup changes the inputs; default trainer settings retained.'))
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if not k.endswith('_frames')}, indent=2))


if __name__ == '__main__':
    main()
