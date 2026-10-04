"""Compare the primary models on their common held-out frames, CPU only.

Scores use the trainer's saved 8-bit source/prediction canvases, not its float
buffers. Each method retains its own COLMAP undistortion. No images are copied
into the public aggregate report, and this process never consumes CUDA.
"""
import json
import os
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / 'gsplat_local/matching_20261003'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TORCH_HOME'] = str(WORKSPACE / '.gsplat-cache/torch')
os.environ['MPLCONFIGDIR'] = str(WORKSPACE / '.gsplat-cache/matplotlib')

import numpy as np
import torch
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def main():
    torch.set_num_threads(2)
    manifest = json.loads((ROOT / 'datasets.json').read_text())
    jobs = [j for j in manifest['jobs'] if j['primary_component']]
    common = sorted(set.intersection(*(set(j['validation_frames']) for j in jobs)))
    output = dict(source_run_id=manifest['source_run_id'], frame_count=len(common),
                  frames=common, source='Saved 8-bit validation canvases: source left, prediction right.',
                  caveat='Common source-frame set; camera undistortion differs per method. Appearance agreement is not independent geometric accuracy.',
                  models={})
    metrics = {'psnr': PeakSignalNoiseRatio(data_range=1.0),
               'ssim': StructuralSimilarityIndexMeasure(data_range=1.0),
               'lpips': LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True)}
    previous = ROOT / 'common_validation_scores.json'
    old = json.loads(previous.read_text())['models'] if previous.exists() else {}
    for job in jobs:
        run = ROOT / 'runs' / job['name']
        status_file = run / 'status.json'
        if not status_file.exists() or json.loads(status_file.read_text())['state'] != 'complete':
            continue
        if job['name'] in old:
            output['models'][job['name']] = old[job['name']]
            continue
        scores = []
        for frame in common:
            index = job['validation_frames'].index(frame)
            with Image.open(run / f'renders/val_step29999_{index:04d}.png') as image:
                canvas = np.asarray(image.convert('RGB'), dtype=np.float32).copy() / 255
            assert canvas.shape[1] % 2 == 0
            left, right = np.split(canvas, 2, axis=1)
            source = torch.from_numpy(left.copy()).permute(2, 0, 1)[None]
            prediction = torch.from_numpy(right.copy()).permute(2, 0, 1)[None]
            with torch.inference_mode():
                row = {'frame': frame}
                for name, metric in metrics.items():
                    metric.reset()
                    row[name] = float(metric(prediction, source))
            scores.append(row)
        output['models'][job['name']] = dict(method=job['method'],
            frames=len(scores), mean={name: float(np.mean([r[name] for r in scores])) for name in metrics},
            per_frame=scores)
        print(job['name'], output['models'][job['name']]['mean'], flush=True)
    previous.write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
