"""Prepare calibrated, undistorted local inputs; run with the COLMAP environment."""
import json
from pathlib import Path
import pycolmap as p

WORKSPACE = Path(__file__).resolve().parents[2]
DATA = WORKSPACE / 'gsplat_local' / 'datasets'
SCALES = {'8x': ('downsample_8x_270x480', 480), '4x': ('downsample_4x_540x960', 960), '2x': ('downsample_2x_1080x1920', 1920), '1x': ('full_resolution', 3840)}

def main():
    DATA.mkdir(parents=True, exist_ok=True)
    records = []
    for scale, (folder, edge) in SCALES.items():
        run = WORKSPACE / 'reconstruction_local' / scale
        models = json.loads((run / 'models.json').read_text())
        source = run / 'dense'
        if not (source / 'sparse/cameras.bin').is_file():
            raise RuntimeError(f'{scale} undistortion not ready')
        target = DATA / scale
        target.mkdir(exist_ok=True)
        for name in ['images', 'sparse']:
            link = target / name
            if not link.exists(): link.symlink_to(source / name, target_is_directory=True)
        model = p.Reconstruction(source / 'sparse')
        records.append({'name': scale, 'scale': scale, 'component': models['selected_model'], 'registered_images': model.num_reg_images(), 'sparse_points': model.num_points3D(), 'dataset': str(target), 'data_factor': 1})
        # Preserve substantial disconnected captures as independent scenes.
        for component in models['models']:
            if component['id'] == models['selected_model'] or component['registered_images'] < 10 or component['points3D'] < 100: continue
            name = f'{scale}_component_{component["id"]}'
            target = DATA / name
            if not (target / 'sparse/cameras.bin').is_file():
                options = p.UndistortCameraOptions(); options.max_image_size = edge; options.max_scale = 1.0
                p.undistort_images(target, run / 'sparse' / str(component['id']), WORKSPACE / 'living_room_frames' / folder, num_patch_match_src_images=10, undistort_options=options, num_threads=8)
            records.append({'name': name, 'scale': scale, 'component': component['id'], 'registered_images': component['registered_images'], 'sparse_points': component['points3D'], 'dataset': str(target), 'data_factor': 1})
    (DATA.parent / 'datasets.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(records, indent=2))

if __name__ == '__main__': main()
