# Living room reconstruction

Reproducible video-frame preparation and independent COLMAP reconstructions at four resolutions. Source photographs, video, previews, databases, full reconstruction outputs and tool environments stay local. Derived point-cloud previews are published only to the private Site at the user’s request. This repository contains code, configuration, image dimensions, timestamps, aggregate reconstruction metrics, and a static process report.

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

`site/dist/` contains the clickable static report, with private-only example and geometry assets excluded from GitHub. `scripts/build_report.py` imports only an allowlist of aggregate local statistics. No photographs, video frames, thumbnails, geometry, camera poses, or raw logs are uploaded to GitHub. The private Site separately includes derived point-cloud previews and camera centers for gap inspection. `scripts/audit_publish.py` checks publishable content before upload.

References: [COLMAP CLI](https://colmap.github.io/cli.html), [PyCOLMAP](https://colmap.github.io/pycolmap/index.html), [COLMAP tutorial](https://colmap.github.io/tutorial.html).


## Point-cloud viewer

Run `python scripts/export_viewer.py` using the COLMAP environment after completed reconstruction stages, then publish the Site. It exports every non-empty sparse component, labels substantial components (at least 10 registered images and 100 points), and includes completed dense fusion. Previews are capped at 200,000 deterministically sampled points; full counts and sampling labels remain visible. Each component has its own coordinate system and is viewed separately. Registration strips show which of the 500 source frames belong to each component, and the union across non-empty models highlights remaining gaps. Derived assets under `site/dist/clouds/` are excluded from GitHub uploads.


## Source examples on the private Site

At the user’s request, `scripts/export_examples.py` copies exactly frames0001,0130,0175,0405 at each of the four resolutions to `site/dist/examples/`. The16 PNGs are exact copies (verified by hashes), with timestamps and dimensions. They are included only in the private Site and excluded from GitHub. All other source photographs remain local. Run this exporter when regenerating the Site’s source examples.


## Attempt 01 — reconstruction baseline (2026-09-29)

This is the first experimental baseline, not a final reconstruction. Settings below were checked against `scripts/reconstruct.py` and the installed PyCOLMAP 4.2.1 options. Values marked **default** were inherited from that installed version, not explicitly set by the script. Preserve this baseline before making another attempt; do not change settings underneath running jobs.

### Cameras: what the overlays mean

COLMAP's native viewer draws registered cameras in red, usually as image planes/frustums showing position and orientation. Red is a display convention, not an error flag. Import a local model folder such as `../reconstruction_local/2x/sparse/1/` using **File > Import Model** to inspect it in the COLMAP GUI. Use its render options to adjust camera display size; this changes visibility, not the estimated reconstruction.

Our private website currently displays **orange camera-center dots**, not red frustums or camera orientations. Under either point-cloud panel, select a **sparse component** and enable **Camera centers**. Orbit and zoom to inspect their distribution. The Point size slider changes scene-point size; camera dots have a fixed display size. The dense preview currently exports no camera centers, so its checkbox cannot reveal cameras. The site does not edit camera poses or rerun COLMAP. Camera-center dots alone cannot show which direction the camera faced or prove that a pose is correct.

### Inputs and compute

- Same 500 upright PNG frames in each scale; dimensions are listed above. Each scale has its own database and independently estimated model.
- PyCOLMAP CUDA 12 package version 4.2.1; local NVIDIA RTX 4090, 24 GB VRAM, driver 580.178.04.
- Features, descriptor matching and PatchMatch use CUDA device 0. Mapping, final Ceres bundle adjustment, undistortion, fusion and meshing use CPU.
- Native maximum image long edges: 3840 / 1920 / 960 / 480 for 1x / 2x / 4x / 8x. Undistortion creates resampled derived images with maximum scale 1.0.

### Effective settings

| Stage | Attempt 01 settings |
| --- | --- |
| Camera model | One shared `SIMPLE_RADIAL` camera per resolution; no supplied calibration. Focal length and radial distortion refined; principal point held fixed (default). |
| SIFT extraction | 8,192 maximum features; native image-size ceiling; 12 threads; GPU 0. Default peak threshold 0.0066667, edge threshold 10, first octave -1, four octaves, three levels/octave, max two orientations, L1_ROOT normalization. Affine-shape estimation and domain-size pooling off (default). |
| Image pairs | Every frame matched with next 15 frames and offsets 20, 40, 80, 160, 320; also cross-sequence anchors every 20 frames. Deduplicated unordered pairs; imported-pair block size 100. |
| Descriptor matching | CUDA, 12 threads, guided matching on; geometric verification retained. Default ratio 0.8, distance 0.7, cross-check on, max 32,768 matches. |
| Incremental mapping | 12 threads; pipeline random seed 0; CPU BA; multiple models enabled (default). Default minimum 15 matches, up to 50 models, model-size threshold 10, automatic initial pair with 200 trials. Returned small/degenerate components are retained and explicitly reported. |
| Initialization / pose | Defaults: initial minimum 100 inliers, 4 px maximum error, 16 degree minimum triangulation angle; absolute pose minimum 30 inliers, 0.25 inlier ratio, 12 px maximum error. |
| Sparse point filtering | Default mapper maximum reprojection error 4 px and minimum triangulation angle 1.5 degrees; two-view-only tracks ignored by default. |
| Final bundle adjustment | Largest component only; Ceres CPU with 12 threads. Focal length, distortion, poses and points refined; principal point fixed. Default trivial loss, maximum 100 solver iterations. |
| Post-BA validation | Current runner reapplies 4 px / 1.5 degree observation filtering and recomputes point errors. Added during Attempt 01 after invalid 2x post-BA geometry was found; repaired 2x removed 36,093 observations, with pre-filter model preserved locally. Earlier 8x/4x results predate this additional final-filter step. |
| Dense view selection | Largest sparse component only; automatic selection of 10 source images per reference image. |
| PatchMatch | GPU 0, geometric consistency on, native image-size ceiling, 4 GB cache, 8 threads. Defaults: 5 iterations, window radius 5, window step 1, 15 samples, automatic depth range, filtering on, minimum NCC 0.1, minimum filter triangulation angle 3 degrees, minimum two consistent views. |
| Stereo fusion | Geometric depth input; CPU 8 threads, native ceiling, cache enabled at 4 GB; explicit PLY output. Defaults: minimum 5 pixels, max reprojection error 2 px, relative depth error 0.01, normal error 10 degrees, check 50 images. |
| Surface mesh | Poisson CPU, 12 threads, octree depth 10; default point weight 1, trim 10, color enabled. |

Pixel thresholds are measured in each resolution's pixels: the same 4 px threshold does not represent the same full-resolution tolerance across scales. A lower mean reprojection error at 8x alone therefore does not prove a better reconstruction.

### Baseline observations and limitations

At documentation time, the largest sparse components registered 343/500 images at 8x, 493/500 at 4x and 492/500 at 2x. Their reported mean errors were approximately 0.603, 0.704 and 0.905 px respectively; 2x is the repaired/filtered result. Full-resolution mapping and higher-resolution dense work were still running. Consult `reports/results.json` and local per-scale status files for subsequent results rather than treating these numbers as final.

The 8x reconstruction also has a substantial separate component (147 registered images, 6,944 points). Components can overlap in their registered images, so their image counts must not simply be added. Independent components are not spatially aligned. Dense processing currently covers only each scale's largest component. The website samples previews (up to 200,000 exported points, or 60,000 rendered in its software fallback), which can visually exaggerate gaps relative to full local PLY files. Poisson surfaces can bridge missing observations; a filled mesh is not evidence that the missing surface was measured.

### Candidate experiments for Attempt 02

These are proposed tests, not settings already applied. Change one family at a time and compare against Attempt 01.

1. **Improve coverage first.** Inspect unregistered frame IDs and the camera path around the weak blank-wall transition near 32 seconds. Additional captures with sideways movement, overlap, stable exposure/focus and textured objects can help triangulation. Turning the camera in place or changing a threshold cannot recover unseen surfaces.
2. **Increase feature and matching coverage.** Try 16,384 features and compare registered-image union and verified matches across weak transitions. For 500 images, exhaustive matching is 124,750 pairs; test it or targeted cross-component pairs if current pairing misses overlapping views. This increases runtime and may add ambiguous matches on repeated furniture/windows.
3. **Test camera calibration deliberately.** Keep SIMPLE_RADIAL as the baseline; test PINHOLE only if the video is already distortion-corrected, or a more flexible model only with evidence the simpler model is inadequate. Video stabilization, zoom or variable intrinsics may violate the shared-camera assumption. Do not freely refine principal point or add distortion parameters merely to reduce error.
4. **Evaluate sparse filtering/initialization.** Inspect match inliers and residuals before altering initial-pair or reprojection thresholds. Relaxing thresholds can connect models but also admit wrong poses; tightening filters trades coverage for cleaner geometry. Compare errors in a common resolution-normalized unit, track lengths and camera-path plausibility.
5. **Tune dense stereo only after poses are sound.** Try 20 source views versus 10, then test filtering sensitivity separately. Relaxing consistency or the fusion minimum pixel count may fill holes but also add floaters. Increasing Poisson depth changes mesh detail and memory use; it cannot supply missing measurements.

### Reproducible next-run procedure

Copy this baseline's code and record the exact commit, PyCOLMAP version, changed options and reason before running a new attempt. Use a new output directory, for example:

```sh
python scripts/reconstruct.py --data ../living_room_frames --output ../reconstruction_attempt02 --scales 4x --through bundle_adjustment
```

Edit the relevant option in a separate experiment copy/branch first: the current CLI selects scales, paths and stopping stage but does not expose individual tuning values. A rerun in the same output directory skips completed stages, so changing code alone will not regenerate their outputs. Do not overwrite the running Attempt 01 directory. Report/export scripts currently read `../reconstruction_local`; adapt them explicitly before reporting another output directory.

Record: registered-image union and largest-component coverage; component sizes and missing frame IDs; normalized reprojection error; track lengths; camera-path plausibility; sparse/dense counts; runtime and memory; visual comparisons of the same four frames and the same room regions. Prefer better coverage with credible geometry over point count alone. Current local logs and status files are the execution record; not every inherited default was serialized at job launch, so this table documents the inspected code and pinned-version defaults, including the noted mid-attempt repair.

Official references: [GUI and camera visualization](https://github.com/colmap/colmap/blob/main/doc/gui.rst), [camera models](https://colmap.github.io/cameras.html), [COLMAP FAQ](https://colmap.github.io/faq.html), [tutorial](https://colmap.github.io/tutorial.html).

## Local gsplat follow-on training

The requested Gaussian splatting pipeline uses the official gsplat `simple_trainer.py default` at release 1.5.3, source commit `937e29912570c372bed6747a5c9bf85fed877bae`, with PyTorch 2.6.0+cu124 and a workspace-local NVIDIA CUDA 12.4.1 compiler. System drivers and system CUDA are unchanged. The separate environment is required because the official example uses a legacy SceneManager parser named pycolmap, while reconstruction uses the modern CUDA-enabled PyCOLMAP package. Exact Python dependencies are recorded in `requirements-gsplat.lock.txt`.

`setup_gsplat_sources.py` downloads official source and verifies NVIDIA compiler archive checksums. `prepare_gsplat_data.py` prepares local calibrated, undistorted images and matching cameras with **data_factor=1** for each scale; it never resizes a second time. Undistortion slightly crops the native images: largest models use 265×472 (8×), 532×947 (4×), 1065×1894 (2×), and 2126×3781 (full). The substantial disconnected 8× component is an independent 147-view training scene (267×476). These scenes have separate coordinate systems.

`train_gsplat_queue.py` waits for active COLMAP CUDA work and other compute processes to release the GPU, runs a separate 100-step 8× smoke test, then trains 8×, 4×, 2×, full resolution, and the substantial secondary 8× component sequentially. The baseline is 30,000 steps, batch size 1, original DefaultStrategy densification, packed rasterization for memory efficiency, evaluation/checkpoints at steps 7,000 and 30,000, and final Gaussian PLY export. Camera and appearance optimization remain disabled. Videos and the live training viewer are disabled; held-out validation renders and PSNR/SSIM/LPIPS metrics are saved locally. The default sorted-image every-eighth holdout is per component; because registration differs, validation memberships are not identical across resolutions.

Run dataset preparation with the reconstruction Python environment, then run `python3 scripts/train_gsplat_queue.py` on the GPU host. Check `../gsplat_local/runs/*/status.json` and training logs for actual completion. Merely preparing this queue does not mean training has completed. A process lock prevents duplicate queue launches. All environments, training inputs, checkpoints, renders, and full splat models stay in the enclosing local project; only explicitly selected previews may be published to the private Site. None belong in this public repository.


## gsplat Attempt 01 — 2026-09-30

All four main datasets and the substantial secondary 8x component completed 30,000 steps on the RTX 4090. Official gsplat 1.5.3 source commit `937e29912570c372bed6747a5c9bf85fed877bae`, PyTorch 2.6.0+cu124 and project-local CUDA 12.4.1 compiler were used. `scripts/train_gsplat_queue.py` enforces one training job at a time and waits for COLMAP CUDA work. The 100-step smoke test passed. Training uses calibrated undistorted images at data factor 1 (no additional downsampling), packed rasterization, default strategy, held-out evaluation/checkpoints at 7,000 and 30,000 steps, and Gaussian PLY export. Undistortion may slightly crop image dimensions.

See `reports/gsplat_results.json` for timings and held-out metrics. Local checkpoints, PLY files and validation source/prediction pairs are in `../gsplat_local/runs/<dataset>/`. Do not publish these assets to public GitHub.

Quality is not approved: a sampled 2x prediction is severely washed out/occluded, despite successful process completion. Cross-resolution metrics are not directly comparable because image resolution and registered/held-out frame sets differ. Preserve these first-attempt outputs for diagnosis. Full-resolution COLMAP fusion remains separate from completed gsplat training.


### Final COLMAP completion and fusion recovery

All four COLMAP pipelines finished on 2026-09-30. Full-resolution fusion produced 5,486,454 points. Its original 4 GB cache repeatedly reread data without finishing the first reference; the failed attempt log is preserved locally. Retrying with `COLMAP_FUSION_CACHE_GB=12` completed fusion in 22.9 minutes, then meshing completed. No image size or geometry thresholds were changed. The runner now accepts that environment override while retaining a 4 GB default. Final counts are in `reports/results.json`. All five gsplat training runs are also complete; visual-quality caveats above remain unresolved baseline limitations, not execution failures.
