# Living room reconstruction

Reproducible video-frame preparation and independent COLMAP reconstructions at four resolutions. Source photographs, video, previews, databases, point clouds, meshes, and tool environments stay local. This repository contains code, configuration, image dimensions, timestamps, aggregate reconstruction metrics, and a static process report.

## Dataset

All scales contain the same 500 source frames. Identical filenames identify the same observation.

| Scale | Local image folder | Width × height |
| --- | --- | --- |
| 1× | full_resolution | 2160 × 3840 |
| 2× | downsample_2x_1080x1920 | 1080 × 1920 |
| 4× | downsample_4x_540x960 | 540 × 960 |
| 8× | downsample_8x_270x480 | 270 × 480 |

The source is an approximately 102-second HEVC video. All 2,446 source frames were analyzed; 500 were selected using local sharpness and motion-aware temporal bins. A blank-wall frame was replaced after feature checks. Master PNGs preserve decoded 8-bit RGB pixels after upright rotation. Reduced sets are independently resized from the masters using Lanczos. See `reports/dataset_manifest.csv` for frame identities and source timestamps. The 32-second blank-wall transition remains weak.

## Local layout

Place the repository beside `living_room_frames/` and `alexis_living_room.mov`. Reconstruction outputs default to the sibling directory `reconstruction_local/`. Alternatively pass explicit `--data` and `--output` paths.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/reconstruct.py --data ../living_room_frames --output ../reconstruction_local
```

A working NVIDIA driver is required. The CUDA 12 PyCOLMAP wheel provides the runtime dependencies. The observed machine has an NVIDIA GeForce RTX 4090, 24 GB VRAM, driver 580.178.04. CPU bundle adjustment is intentional; no unverified GPU bundle-adjustment claim is made.

## COLMAP stages

1. GPU SIFT feature extraction at each set's native image size; shared SIMPLE_RADIAL camera per set.
2. GPU descriptor matching with geometric verification: 15 adjacent frames, exponential offsets, and cross-sequence anchors every 20 frames; guided matching enabled.
3. CPU incremental mapping estimates cameras and sparse structure, with repeated bundle adjustment. All sparse components are retained.
4. CPU final bundle adjustment refines the largest component. Unregistered images and component sizes are recorded.
5. CPU image undistortion creates the dense workspace, capped at the input set's native long edge. Original PNGs are unchanged; undistortion inherently resamples the derived images.
6. CUDA PatchMatch estimates depth and normals with 10 source views, five iterations, and geometric consistency.
7. CPU stereo fusion combines consistent depth into `dense/fused.ply`.
8. CPU Poisson meshing at octree depth 10 creates `dense/mesh_poisson.ply`.

Dense reconstruction uses the largest registered sparse component for each scale. This is stated in the report; components are not silently combined. Each scale is solved independently, so coordinates and scale are arbitrary and models are not directly aligned.

The runner records stage start times, devices, durations, success/failure, and separate logs. Completed stages are skipped on rerun. A failed stage is retried when the runner is rerun; delete or archive its local outputs before changing parameters. Status files are completion evidence, not a promise of scene completeness.

## Report

[Open the private reconstruction report](https://alexis-living-room-reconstruction.hello420892.chatgpt.site).

`site/dist/` contains an image-free, clickable static report. `scripts/build_report.py` imports only an allowlist of aggregate local statistics. No photographs, video frames, thumbnails, point coordinates, camera poses, or raw logs are published. `scripts/audit_publish.py` checks publishable content before upload.

References: [COLMAP CLI](https://colmap.github.io/cli.html), [PyCOLMAP](https://colmap.github.io/pycolmap/index.html), [COLMAP tutorial](https://colmap.github.io/tutorial.html).
