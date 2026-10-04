"""Export every sparse component from the newest three-way matching trial."""
from pathlib import Path
import argparse
import gzip
import json
import re
import sqlite3

import numpy as np
import pycolmap


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-root', type=Path, default=project.parent / 'colmap_gui/iphone13pro_500_4x_comparison/timed_rerun_20261003_loop_enabled')
    args = parser.parse_args()
    out = project / 'site/dist/clouds'
    out.mkdir(parents=True, exist_ok=True)
    labels = {'sequential_no_loop': 'Sequential · loop off',
              'sequential_loop': 'Sequential · loop on', 'exhaustive': 'Exhaustive'}
    index = {'run_id': args.comparison_root.name, 'primary_axis_label': 'Matching method',
             'alignment': 'Each saved model has its own coordinate system and is inspected separately.', 'scales': {}}
    report = {'run_id': args.comparison_root.name, 'input_images': 500,
              'image_size': [540, 960], 'methods': {}}
    # Validate all three inputs before replacing any published cloud.
    for method in labels:
        root = args.comparison_root / method
        if not (root / 'database.db').is_file() or not any((root / 'reconstruction').glob('*/points3D.bin')):
            raise FileNotFoundError(f'Incomplete matching run: {method}')
    for method, label in labels.items():
        root = args.comparison_root / method
        with sqlite3.connect(f'file:{(root / "database.db").resolve()}?mode=ro', uri=True) as db:
            attempted, nonempty = db.execute('SELECT COUNT(*), SUM(rows > 0) FROM two_view_geometries').fetchone()
            keypoints = db.execute('SELECT SUM(rows) FROM keypoints').fetchone()[0]
            input_images = db.execute('SELECT COUNT(*) FROM images').fetchone()[0]
        assert input_images == 500, (method, input_images)
        components = []
        union = set()
        for path in sorted((root / 'reconstruction').iterdir()):
            if not (path / 'points3D.bin').is_file():
                continue
            model = pycolmap.Reconstruction(path)
            ids = sorted(int(re.search(r'(\d+)', im.name).group(1)) for im in model.images.values())
            union.update(ids)
            points = [model.points3D[key] for key in sorted(model.points3D)]
            item = dict(id=path.name, kind='sparse', registered_images=model.num_reg_images(), frame_ids=ids,
                        total_points=len(points), displayed_points=len(points), sampled=False, render_all=True,
                        substantial=model.num_reg_images() >= 10 and len(points) >= 100,
                        mean_reprojection_error=model.compute_mean_reprojection_error(),
                        mean_track_length=model.compute_mean_track_length())
            if points:
                xyz = np.asarray([pt.xyz for pt in points])
                rgb = np.asarray([pt.color for pt in points])
                center = np.median(xyz, axis=0)
                radius = max(float(np.percentile(np.linalg.norm(xyz - center, axis=1), 95)), 1e-6)
                packed = np.concatenate(((xyz - center) / radius, rgb / 255), axis=1).astype('<f4')
                run_slug = re.sub(r'[^a-zA-Z0-9_-]', '-', args.comparison_root.name)
                name = f'matching-{run_slug}-{method}-model-{path.name}.bin.gz'
                (out / name).write_bytes(gzip.compress(packed.tobytes(), compresslevel=6, mtime=0))
                item['url'] = 'clouds/' + name
                item['camera_positions'] = [((im.projection_center() - center) / radius).round(6).tolist() for im in model.images.values()]
            components.append(item)
        components.sort(key=lambda item: item['registered_images'], reverse=True)
        index['scales'][method] = dict(label=label, run_id=args.comparison_root.name, input_images=input_images, components=components, dense=None,
                                      registered_union=len(union), missing_frame_ids=sorted(set(range(1, 501)) - union), ready=bool(components))
        report['methods'][method] = dict(label=label, attempted_pairs=attempted, nonempty_geometries=nonempty, sift_keypoints=keypoints,
                                        registered_union=len(union), saved_models=len(components),
                                        largest_component_registered_images=max(item['registered_images'] for item in components),
                                        sparse_points_sum=sum(item['total_points'] for item in components),
                                        components=[{key: value for key, value in item.items() if key not in ('camera_positions', 'url')} for item in components])
    (out / 'matching.json').write_text(json.dumps(index, indent=2) + '\n')
    (project / 'reports/matching_results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({method: {key: value for key, value in data.items() if key != 'components'} for method, data in report['methods'].items()}, indent=2))


if __name__ == '__main__':
    main()
