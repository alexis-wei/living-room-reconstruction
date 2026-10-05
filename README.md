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

### Step 01 final reprojection error comparison — verified 2026-10-04

The Step 01 table reports the largest component's mean reprojection error **after bundle adjustment and observation filtering**, from `sparse_metrics.json`, rather than the earlier mapping values in `models.json`. Reading each saved final model, updating its point errors in memory, and recalculating the COLMAP mean reproduced the saved values to within 1e-9 pixels. No model files were changed and no reconstruction was rerun.

| Input resolution | Registered / 500 | Mean error, input-image px | Full-resolution-equivalent px |
| --- | ---: | ---: | ---: |
| Full, 2160 × 3840 | 493 | 1.183460 | 1.183460 |
| 2× downsample, 1080 × 1920 | 492 | 0.904950 | 1.809900 |
| 4× downsample, 540 × 960 | 493 | 0.703813 | 2.815251 |
| 8× downsample, 270 × 480 | 343 | 0.603342 | 4.826738 |

Reprojection error describes the image-plane distance between observed features and projected reconstructed points. The equivalent value multiplies the native error by the downsample factor (1, 2, 4 or 8), putting residuals on the same pixel scale. It does not rescore all models on identical observations: registration, detected features and surviving tracks differ, particularly at 8×. Lower error indicates tighter image fit, not verified room geometry. These measurements do not provide dense-cloud, mesh or physical-dimension error; that would require reference geometry. The exporter retains full numerical precision in `reports/results.json` and records the metric stage and equivalent pixel value.


## Private Gaussian viewer

The private Site now includes an interactive Spark 2.3.0 / Three.js 0.180.0 Gaussian viewer for all five models, camera presets, and 20 GPU-rendered previews from the original checkpoints. `scripts/export_gsplat_site.py` preserves every trained Gaussian in 32-byte SPLAT chunks below 25 MB each: float32 centers/scales, 8-bit RGBA and normalized quaternion. Browser colors are SH0 only; view-dependent SH3 appearance remains in the original checkpoints/PLY and the rendered previews (up to 960-pixel long edge). Zero-extent Gaussians are retained. WebGL2 is required for interaction; previews work without it. Private assets under `site/dist/gaussians/` are excluded from public GitHub. Self-hosted vendor libraries are fetched from pinned official distributions, not image/model uploads to third-party viewers.

Browser hosting uses `scripts/compact_gsplat_site.py` after export: chunk-local float16 positions, 8-bit logarithmic scales, RGB565 colors, 8-bit opacity/quaternion, gzipped without removing Gaussians. Client-side decompression restores the standard SPLAT layout. This is a quantized visualization copy, not the original training output.

## Attempt 02 — frame-count and area experiments (2026-09-30)

The successful 4× baseline remains unchanged. Three independent datasets test frame count and coverage, all reconstructed at **540 × 960**:

| Experiment | Frames | Video coverage | Local dataset |
| --- | ---: | --- | --- |
| Whole room / fewer frames | 250 | Full video | `experiments_local/room_250/` |
| Whole room / more frames | 1,000 | Full video | `experiments_local/room_1000/` |
| Focused dining area | 500 | 34.5–37.0 and 61.0–81.0 seconds | `experiments_local/dining_500/` |

Each dataset has `full_resolution/` (2160 × 3840 RGB PNG), `downsample_4x_540x960/`, `manifest.csv`, exact selected source indices, `dataset.json`, local contact sheets, checksums, and PNG validation. Filenames are chronological within each experiment; the same filename across experiments does **not** necessarily identify the same video frame. Use source indices/timestamps to join datasets. Local hard links share identical source-frame files across sets; treat image inputs as immutable.

Whole-room selection uses 55% elapsed-time and 45% capped optical-flow weighting, one locally sharp exposure-aware frame per chronological bin, with a capacity constraint ensuring exactly the requested number of distinct recorded frames. No frame interpolation, cropping, sharpening or synthetic views. Upright native RGB frames are decoded from the original MOV; every reduced PNG is generated directly with Pillow LANCZOS, or reuses the identical original dataset output.

The dining area was selected after visually inspecting the video: the table appears from multiple sides, whereas the couch is predominantly front-facing. The focused intervals include nearby furniture/walls to help registration; this is not object segmentation. Many of its 500 frames are near-neighbors. The temporal discontinuity can cause separate components. No dataset supplies unseen backs, undersides or a guaranteed 360° reconstruction.

COLMAP settings match Attempt 01 at 4×: PyCOLMAP 4.2.1 CUDA; SINGLE SIMPLE_RADIAL camera per dataset; 8,192 feature limit; 15 sequential neighbors plus offsets 20/40/80/160/320 and every-20-image anchors; guided matching; CPU incremental mapping/global BA; post-BA filtering 4 px/1.5°; native-size undistortion; up to 10 source views; geometric CUDA PatchMatch with five iterations; CPU fusion. Indexed pair rules cover different time distances at different frame counts, so this is a practical comparison rather than a rigorously isolated image-count study. A former hardcoded `input_images=500` reporting field now counts the actual dataset.

`run_experiments.py` holds an exclusive lock, waits for validated extraction, reconstructs sparse cameras for all three sets, then processes their dense clouds sequentially through fusion. There is no new gsplat training or meshing in this request. CUDA feature extraction, matching and depth estimation use the local **RTX 4090**; mapping, BA and fusion use CPU. All meaningful sparse components are retained; dense processing uses the largest registered component, as in the baseline.

```bash
PYTHONPATH=.video-tools python3 living-room-reconstruction/scripts/prepare_experiments.py
python3 living-room-reconstruction/scripts/run_experiments.py
PYTHONPATH=.colmap-tools python3 living-room-reconstruction/scripts/export_experiments.py
```

Run from the enclosing local project directory, with the existing project-local dependencies. Queue progress is in `experiments_local/queue_status.json`; each dataset's `colmap/4x/status.json` and stage logs provide measured progress. Completed outputs are `colmap/4x/sparse_points.ply`, each camera model under `sparse/`, and `dense/fused.ply`. The private Site's `experiments.html` compares registration, fragmentation, reprojection error and point counts against the existing 500-frame baseline, with a coverage strip per component. The 1,000-frame sparse viewer exports and draws every point in every component (113,505 + 5,427 + 668 points); the camera-model experiment also shows all sparse points. Other new browser previews sample at most 40,000 points to fit hosting limits; complete PLYs remain local. Preview density is not reconstruction completeness. The original Site assets and models remain available.

All 1,750 native/reduced image pairs passed full PNG decoding, dimensions, uniqueness and checksum checks. Nine sampled source frames matched fresh video decoding and direct LANCZOS resizing pixel-for-pixel (`verify_experiment_pixels.py`; local `pixel_verification.json`).

`reports/experiments.json` records measured status; queued/running stages are not completed results. Source images and geometry remain excluded from GitHub. Only derived point-cloud previews are uploaded to the existing private Site for these experiments.


### Verified gsplat photograph inputs (2026-09-30)

The original 4× gsplat run **already used the source photographs** after COLMAP undistortion. It was not trained from a point cloud alone. The saved configuration uses `gsplat_local/datasets/4x`, whose `images/` link resolves to `reconstruction_local/4x/dense/images/`, and `data_factor=1` (no additional reduction). The parser successfully loads 493 registered photographs: 431 training images and 62 held-out validation images (`test_every=8`). An audited training sample is 532 × 947 RGB, reflecting undistortion of the 540 × 960 input.

The official trainer reads `data["image"]` and optimizes rendered pixels against those photograph pixels using 0.8 L1 + 0.2 SSIM loss. COLMAP sparse points initialize the Gaussians; camera poses associate images with viewpoints. The saved 30,000-step run, validation renders, and checkpoints belong to this photo-supervised workflow. There is no existing no-photo training baseline.

A source-photo-resolution comparison would instead reuse the 4× COLMAP camera poses and sparse initialization while changing photograph resolution, scaling camera intrinsics consistently and undistorting from native masters. That is distinct from adding previously absent images, and higher resolution alone does not guarantee better geometry. The user's choice between this comparison and retaining 4× photograph supervision is pending; no replacement baseline training has been launched. All five additional COLMAP experiments subsequently completed; the gsplat input-resolution decision remains pending.


## Attempt 03 — iPhone 13 Pro 1× camera comparison (2026-09-30)

The original model was explicitly **SIMPLE_RADIAL**, shared by all images. It estimated focal length and one radial distortion coefficient; it was not calibrated specifically for this phone. Original 4× initialization was f=1152 px, cx=270, cy=480, k=0, with no known-focal prior. The final baseline parameters were f=690.281615, cx=270, cy=480, k=0.019731457.

The MOV identifies `iPhone 13 Pro 26mm`, f/1.5, recorded with Blackmagic Camera. It supplies no calibrated intrinsic matrix or distortion coefficients; stabilization/cropping cannot be recovered from the phone name alone. The new shared **OPENCV** experiment uses the exact same 500 original 4× PNGs and initializes fx=fy=693.333333 px (960×26/36, a nominal full-frame-width approximation), cx=270, cy=480 and k1=k2=p1=p2=0. This is an estimated camera-aware variant, **not a measured or guaranteed correct calibration**. COLMAP refines focal lengths and distortion while keeping the principal point centered. The focal prior remains false.

`scripts/run_camera_model_experiment.py` copies the baseline feature/match database without modifying it, replaces the camera, discards old two-view geometry, verifies matches again, and runs fresh mapping and bundle adjustment. Both model complexity and initialization change, so this is not a single-variable experiment. Final OPENCV parameters (fx, fy, cx, cy, k1, k2, p1, p2):

```text
687.577687, 686.596229, 270, 480,
0.0650665682, -0.0757400330, -0.00148717409, 0.000813173908
```

| Largest sparse component | SIMPLE_RADIAL baseline | OPENCV variant |
| --- | ---: | ---: |
| Registered frames | 493 / 500 | 485 / 500 |
| Sparse points | 74,712 | 61,062 |
| Mean reprojection error | 0.703813 px | 0.456238 px |

The lower error comes with fewer registered views and points; it does not establish better geometric accuracy. No ground-truth calibration or dimensions were supplied. The private `camera-comparison.html` page presents both models side by side. Dense reconstruction completed after the three frame-selection experiments, using RTX 4090 CUDA depth estimation without overlapping GPU-heavy jobs. Fresh geometry verification, mapping and BA ran on CPU; original CUDA-extracted features were reused.

Local data/status: `../experiments_local/iphone13pro_4x/`. Reproduce from the enclosing workspace with `python3 living-room-reconstruction/scripts/run_camera_model_experiment.py`; its lock prevents duplicate runs. The original reconstruction stays intact. Camera-model background: https://colmap.github.io/cameras.html ; phone specifications: https://support.apple.com/en-us/111871 .


### Same OPENCV settings on the 1,000-frame dataset

`iphone13pro_1000_4x` reuses the exact existing `room_1000` native and 540×960 PNGs through local directory links. It repeats the 500-frame OPENCV experiment's shared camera, initial parameters `[693.333333,693.333333,270,480,0,0,0,0]`, no known-focal prior, focal/distortion refinement, fixed centered principal point, six CPU mapping threads, and post-BA filtering. Final calibrated values are independently estimated from the 1,000-image set; the 500-image fitted values are not frozen or copied. Existing SIMPLE_RADIAL models stay intact.

Run from the enclosing workspace:

```bash
PYTHONPATH=.colmap-tools python3 living-room-reconstruction/scripts/run_camera_model_experiment.py --dataset room_1000
```

The variant reuses the 1,000-image baseline's features/raw matches, re-verifies geometry, and runs fresh sparse mapping/BA. Dense processing waits for the three original experiments, then the 500-frame OPENCV variant, then an idle GPU. Local outputs are in `experiments_local/iphone13pro_1000_4x/colmap/4x/`. The private `camera-comparison-1000.html` compares the two 1,000-image models; all sparse points from every non-empty component are exported and drawn, with dense previews explicitly sampled. Live measured status is recorded in `reports/experiments.json`.


The 1,000-image OPENCV sparse run completed: largest component 968/1,000 registered images, 100,232 points, mean reprojection error 0.494405 px (SIMPLE_RADIAL: 985 images, 113,505 points, 0.803085 px). Secondary models contain 17 images/247 points and 15 images/zero points; zero-point models cannot be rendered. Lower fitting error comes with fewer views in the largest component, so it is not proof of improved geometric accuracy. Dense output is complete; see final counts below. Fitted OPENCV parameters `[fx,fy,cx,cy,k1,k2,p1,p2]`: `[687.563197,686.422498,270,480,0.0633152613,-0.0729851177,-0.00157287453,0.000789965114]`. All 100,232 + 247 sparse points are included in the private comparison viewer.


## Final additional COLMAP results — 2026-09-30

All five requested 4× experiments completed sparse reconstruction, bundle adjustment, native-size undistortion, CUDA depth maps and CPU fusion. Each final PLY vertex count was checked against the report. All CUDA stages used the local RTX 4090 sequentially.

| Dataset | Fused points |
| --- | ---: |
| room_250 | 751,310 |
| room_1000 | 2,616,892 |
| dining_500 | 869,092 |
| iphone13pro_4x | 1,748,593 |
| iphone13pro_1000_4x | 2,814,652 |

Private comparison pages include every requested sparse point and labeled dense previews (40,000 points each; the final 1,000-frame OPENCV preview uses 30,000 to fit the hosting archive limit). Full models, photographs and depth maps remain local; public GitHub contains only code and aggregate results. OPENCV intrinsics remain estimated, not measured factory calibration. Point count alone is not a geometric accuracy metric. Original four-scale COLMAP and five gsplat results are preserved. The source-photograph-resolution comparison remains pending; the later 4× matching-strategy gsplat experiment is documented under Attempt 05.


## Attempt 04 — sequential vocabulary-tree loop detection timing (2026-10-03)

A third 500-frame matching trial toggled COLMAP sequential loop detection on, and reran the no-loop sequential and exhaustive baselines under the same timing workflow. Each strategy used a fresh copy of the same SIFT feature database and the same OPENCV camera. Matching ran serially on the RTX 4090; incremental sparse mapping and bundle adjustment ran serially on the CPU with 32 logical threads.

| Strategy | Feature matching | Sparse mapping + bundle adjustment | Sum of measured stages |
| --- | ---: | ---: | ---: |
| Sequential, loop detection off | 13.143 s | 184.879 s | 198.022 s |
| Sequential, loop detection on | 96.788 s | 393.999 s | 490.787 s |
| Exhaustive | 52.144 s | 405.635 s | 457.779 s |

The timer uses Python `time.perf_counter_ns` around each COLMAP child process; the local audit record retains elapsed nanoseconds. Ten-second callbacks only report progress. Startup and exit are included. Image extraction, database copying/reset, feature-cache warmup, pauses between stages, and dense reconstruction are excluded. These are single runs, not averages.

| Strategy | Pair records | Geometries with inliers | Unique registered frames | Models | Largest component | Sparse points summed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sequential, loop off | 3,989 | 2,012 | 493 / 500 | 4 | 256 frames | 67,642 |
| Sequential, loop on | 5,986 | 2,870 | 496 / 500 | 1 | 496 frames | 69,847 |
| Exhaustive | 124,750 | 10,964 | 499 / 500 | 4 | 492 frames | 78,885 |

The loop-enabled COLMAP matcher used the cached Flickr vocabulary tree, querying every tenth frame for up to 50 similar views while retaining sequential overlap 10 and quadratic offsets. Relative to the no-loop run it tested 1,997 more pairs, raised the count of geometries with inliers from 2,012 to 2,870, and joined 496 registered images into a single model. The no-loop reconstruction split into four components. Exhaustive matching registered 499 unique frames but retained three small fragments beside its 492-frame component.

The exhaustive mapper exited successfully and saved its models, but its log contains CHOLMOD/Eigen linear-solver warnings during global bundle adjustment. Interpret that output cautiously. Point count and reprojection error alone do not establish geometric accuracy. Step 05 of the private Site now displays this newest timed run throughout: all nine components, every sparse point, unique registered-frame counts, component metrics, camera coverage, settings and the matching/mapping durations above. The overall comparison selector uses the same newest matching exports. Previous Site previews are archived locally. Source images, feature databases, logs, text exports and full sparse models remain local and are excluded from this public repository.

To reproduce on the local workspace, use the CUDA-enabled COLMAP launcher, existing extracted-feature database and frame folder; choose a new empty output directory:

```bash
python3 living-room-reconstruction/scripts/run_matching_strategy_timing.py \
  --workspace-root "/path/to/peripheral-project" \
  --output-dir "/path/to/peripheral-project/colmap_gui/iphone13pro_500_4x_comparison/new-timed-run"
python3 living-room-reconstruction/scripts/summarize_matching_strategy_timing.py \
  --run-dir "/path/to/peripheral-project/colmap_gui/iphone13pro_500_4x_comparison/new-timed-run" \
  --colmap-launcher "/path/to/peripheral-project/colmap_gui/run_cuda_colmap.sh"
```

The runner accepts explicit overrides for the workspace paths, matcher and mapper project files, vocabulary tree, COLMAP launcher, and callback interval. It refuses to overwrite a non-empty run and keeps copied databases and all run outputs in the selected local output directory. The public aggregate audit is `reports/matching_timing.json`; raw images, feature databases, process logs, and model files are not stored in GitHub.

## Attempt 05 — gsplat for the three newest matching strategies (2026-10-04)

Step 05 adds photo-supervised gsplat reconstructions for sequential matching with loop detection off, sequential matching with loop detection on, and exhaustive matching. These use the exact saved COLMAP components from `timed_rerun_20261003_loop_enabled`, the original 500 4× PNGs (540×960), and each component's recovered OPENCV cameras and sparse points. No COLMAP stage is rerun. The existing camera calibration remains estimated from the images, not measured factory calibration.

Each disconnected component has its own coordinate system and is trained independently. Eight components have at least ten cameras and 100 sparse points; all eight are included in the queue. The exhaustive two-camera/200-point fragment remains available in the COLMAP viewer but is too small for a reliable Gaussian comparison and is explicitly excluded. Main components are trained first, followed by smaller fragments.

The recipe is the pinned gsplat 1.5.3 official `examples/simple_trainer.py` default strategy: 30,000 steps, seed 42, batch size 1, packed CUDA rasterization, SH degree 3, 0.8 L1 + 0.2 SSIM loss, and no camera-pose or appearance optimization. Evaluation and checkpoints occur at 7,000 and 30,000 steps; a full SH3 PLY is saved at 30,000. `data_factor=1` means no additional image reduction. The parser undistorts the OPENCV input and removes one border pixel, giving 539×959 training views. GPU jobs run serially on the local RTX 4090, without closing desktop COLMAP windows. The settings are held constant across methods; they are not tuned separately to favor a result. This is a fixed-step comparison, not equal epochs: the three main training sets receive about 134, 69 and 70 image updates per view on average. Smaller components receive more updates per photograph, which also limits conclusions about geometric quality.

Source frames 1, 9, 17, …, 497 are excluded from Gaussian fitting in every component. The main no-loop, loop-on, and exhaustive components use 224/32, 435/61, and 430/62 training/validation images respectively. Thirty-two validation frames are shared by all three main components. COLMAP camera estimation and sparse initialization already used these images, so this evaluates held-out appearance, not independent geometric accuracy.

The private page provides paired method/component selectors and four same-frame predictions from the full SH3 models (frames 249, 329, 409 and 497). Smaller components provide at least two viewpoints each; their saved frame IDs are also rendered in the main models wherever those frames were recovered, allowing the couch and other fragments to be compared directly. The shared-frame selector updates for the chosen pair. Native prediction pixels are preserved in lossless WebP; the source-photo half of the trainer's validation canvas is not uploaded. Metrics are shown for every component with their different validation subsets disclosed. A second comparison recalculates PSNR, SSIM and AlexNet LPIPS on all 32 shared frames from the saved 8-bit canvases, CPU only. Each method uses its own camera undistortion, which still limits exact cross-method comparability. These metrics measure image agreement and do not establish correct room geometry.

Measured training durations use `time.perf_counter_ns` around each complete trainer process, including startup, evaluation and saving. Raw nanoseconds and unrounded seconds remain in the aggregate report; the page displays milliseconds. This is separate from the earlier matching/mapping timers. Saved checkpoints, configuration files, logs, validation canvases and full PLYs are retained locally in `gsplat_local/matching_20261003` and are excluded from public GitHub.

Full 3D viewing loads every valid exported Gaussian and SH3 coefficients directly from this computer through the loopback viewer, or from a local PLY selected in the browser. The private Site's “Explore a full trained model in 3D” control uses this local viewer; no full PLY is hosted or uploaded. WebGL2 and enough browser memory are required (the main models contain millions of Gaussians). If local-network loading is blocked, open the local-viewer link or choose the corresponding PLY. Start or restart the server with:

```bash
python3 living-room-reconstruction/scripts/serve_matching_gsplat.py
```

It listens only at `http://127.0.0.1:8790/`, serves a restricted completed-model allowlist, and permits cross-origin reads only from the existing private Site and local preview. Read-only model serving does not run a second training job. Reproduce and refresh from the enclosing workspace:

```bash
.gsplat-env/bin/python living-room-reconstruction/scripts/run_matching_gsplat.py
.gsplat-env/bin/python living-room-reconstruction/scripts/score_matching_gsplat.py
.gsplat-env/bin/python living-room-reconstruction/scripts/export_matching_gsplat.py
```

The queue lock prevents duplicate training; completed jobs are retained and skipped. Failed outputs must be preserved before a retry. `reports/matching_gsplat_results.json` contains the current aggregate results and settings. The original photograph-resolution comparison question remains separate from this authorized 4× matching experiment.

## Attempt 06 — Insta360 X5, native and 4× (2026-10-04)

The new upload is `insta360_living_room_footage.mp4`: inspected 3840 × 2160, 24 fps, HEVC, 92.958333 seconds, 8-bit full-range YUV420. Its exported dimensions are 4K regardless of the original recording setting. `prepare_insta360.py` excludes every frame before 10.0 seconds, takes one candidate from each of 500 equal-duration intervals, and favours preview sharpness (0.55), local spatial appearance novelty (0.30), bin-center proximity (0.15), with exposure penalties. It does not create views, interpolate video frames or upscale pixels.

Completed extraction: exactly 500 unique decoded RGB frames at 10.083333–92.875 seconds, saved as native RGB PNGs and aligned 960 × 540 PNGs resized directly with Lanczos. All 1,000 PNGs were decoded and checked pixel-for-pixel against their respective expected images; checksums and source frame indices/PTS/timestamps are retained locally. Median time gap is 0.166667 seconds, minimum 0.041667 and maximum 0.291667. This gives even temporal coverage and modest within-interval diversity, not proof of unique 3D angles or minimal geometric overlap.

Local source folders are `../insta360_frames_500/full_resolution/` and `../insta360_frames_500/downsample_4x_960x540/`; their names align from `frame_0001.png` to `frame_0500.png`. Storage at extraction is 2,782,855,203 and 337,607,788 bytes. Local `review/index.html`, ten contact sheets, manifest and checksums allow inspection. Native photographs, the video and review copies are excluded from this repository and the hosted Site.

### Camera research and assumptions

[Insta360's X5 explanation](https://www.insta360.com/blog/tips/understanding-8K-360-video.html) describes MegaView as 170°; [Studio FOV documentation](https://onlinemanual.insta360.com/studio/en-us/operation-guide/edit-function/adjust-the-perspective) describes reframing at export. Neither consulted source provides a factory or per-export intrinsic matrix. The uploaded video is a flat 16:9 wide-angle view, not an unprocessed dual-lens stream or a 2:1 equirectangular panorama. The original sensor calibration cannot simply be copied onto arbitrary reframed/stabilized pixels.

Use the explicitly requested `OPENCV_FISHEYE`, parameter order `fx, fy, cx, cy, k1, k2, k3, k4`. Initialization interprets nominal 170° as a **horizontal equidistant FOV assumption**, setting `fx=fy=width/radians(170)`, center at half width/height, and coefficients zero. This is an initial optimization prior, not measured calibration. Full initialization is approximately `1294.2105,1294.2105,1920,1080,0,0,0,0`; 4× approximately `323.5526,323.5526,480,270,0,0,0,0`. Use the precise values in the saved configuration. Shared focal lengths/distortion are refined by COLMAP; principal point stays fixed. MegaView warping or a changing zoom may limit this model; report estimated parameters and image fit without claiming certified physical accuracy. [COLMAP camera models](https://colmap.github.io/cameras.html) and [OpenCV fisheye equations](https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html) define the model.

### Reproducible pipeline

From the enclosing photo workspace, with the existing environments:

```sh
PYTHONPATH="$PWD/.video-tools" python3 living-room-reconstruction/scripts/prepare_insta360.py
python3 living-room-reconstruction/scripts/run_insta360_pipeline.py
PYTHONPATH="$PWD/.colmap-tools" python3 living-room-reconstruction/scripts/export_insta360_colmap.py
.gsplat-env/bin/python living-room-reconstruction/scripts/export_insta360_gsplat.py
python3 living-room-reconstruction/scripts/serve_insta360.py
```

The queue holds the existing gsplat GPU lock and preserves completed stages. It runs sparse stages for 4× and native, then 4× dense reconstruction plus gsplat, then native undistortion plus gsplat followed by native dense stereo/fusion/meshing. At most one CUDA stage runs at a time on the RTX 4090. Idle existing COLMAP GUI windows stay open. Failed outputs require diagnosis/preservation before retrying.

COLMAP settings reuse the documented original pipeline: native SIFT cap8,192; 15 sequential neighbors, offsets20/40/80/160/320 and anchors every20 images with guided matching; CPU incremental mapper seed0/12threads; Ceres global BA and filtering4px/1.5°; native-ceiling calibrated undistortion/10sourceviews; five CUDA geometric PatchMatch iterations; CPU fusion with12GBcache; Poisson depth10. Accurate monotonic nanosecond timings include process startup/output saving and exclude preceding GPU waits. All components are retained; dense uses the largest refined component.

gsplat1.5.3 uses the original-derived undistorted photographs **with their corresponding pinhole camera matrices**, data_factor1,30,000steps,batch1,packed,SH3,seed42,default densification,L1/SSIM0.8/0.2,no camera/appearance optimization. Fixed source frames1,9,...497 are held out; COLMAP still used their geometry. Each component with at least10 cameras/100points trains separately. All checkpoints/native predictions/full SH3 PLYs stay in `../insta360_local/gsplat/`; other small components remain in COLMAP. Full local viewer runs on127.0.0.1:8791 and requires WebGL2. The original matching experiment viewer remains on8790.

Chapter07 compares source dimensions/storage, feature/match counts, registration/components, sparse/dense/mesh counts, native/full-resolution-equivalent reprojection error, intrinsics, stage timings, Gaussians and PSNR/SSIM/LPIPS, with numeric factor1-versus4 charts. Two resolutions do not establish a general correlation; native appearance metrics have different pixel sizes/crops/registered sets and are not ground-truth geometry scores. Private browser clouds are explicitly sampled (sparse25,000/dense15,000 cap per model), and prediction-only previews use a maximum960px long edge followed by lossless WebP. This does not resize training inputs or full local outputs. Aggregate reports and code alone may be uploaded to this public repository; `site/dist/insta360-assets/` remains private.

To stay within the private Site's archive limit, twenty existing prediction renders and sixteen previously authorized source-example browser copies were converted from PNG to pixel-verified lossless WebP. Native PNG masters, image dimensions, all baseline Gaussian geometry and all COLMAP point-cloud assets remain unchanged. Only code and aggregate reports are published to public GitHub; all new rendered images stay on the owner-private Site. Official workflow: [gsplat COLMAP capture trainer](https://docs.gsplat.studio/main/examples/colmap.html).


### Completed matching-strategy gsplat runs

| Method / component | Recovered cameras | Exported / trained Gaussians | Validation PSNR | Complete process time |
| --- | ---: | ---: | ---: | ---: |
| Sequential · loop off / 1 | 256 | 3,430,406 | 28.079 | 724.898 s |
| Sequential · loop on / 0 | 496 | 3,277,924 | 19.588 | 677.864 s |
| Exhaustive / 1 | 492 | 3,357,036 | 20.545 | 688.142 s |
| Sequential · loop off / 2 | 150 | 4,779,697 | 28.030 | 994.045 s |
| Sequential · loop off / 0 | 77 | 6,181,200 | 22.981 | 1261.871 s |
| Sequential · loop off / 3 | 30 | 2,413,974 | 10.378 | 503.797 s |
| Exhaustive / 3 | 27 | 1,214,328 / 1,214,448 | 14.816 | 412.526 s |
| Exhaustive / 2 | 20 | 1,228,458 | 14.480 | 516.019 s |

All eight runs completed and retain their original checkpoints and full valid SH3 PLY exports locally, without sampling. The 27-camera exhaustive fragment had 120 non-finite scale rows in its checkpoint; gsplat’s standard exporter omitted those rows (1,214,328 valid exported vs 1,214,448 trained entries). The raw checkpoint remains intact. All eight PLYs were validated for exact vertex/file sizes and finite float values; all 41 hosted renders match the native prediction pixels exactly.

Four models are disconnected sequential loop-off components; three belong to exhaustive matching; loop-on has one. Scores above cover each component’s validation set. The common 32-frame main-model scores are recorded separately in the aggregate report and on Step 05. Tiny or fragmented models can have poor appearance agreement despite containing many Gaussians; they remain visible for inspecting gaps.


### Insta360 geometry failure and separate recovery

The original 500-view sparse baselines are preserved and visible with every sparse point: 260,369 native and 112,825 at 4×. Numeric registration and reprojection fit did not establish valid camera geometry. The 4× PatchMatch stage failed after 1628.022944567 seconds when collapsed local camera/point groups could not provide positive MVS depth ranges; 487 partial photometric depth maps remain local. Audits identified 180 collapsed 4× cameras and 34 native cameras. These are failed-quality baselines, not successful dense reconstructions.

A separate screened recovery uses copies of the final models, the same estimated OPENCV_FISHEYE calibration and original-resolution pixels. The scale-relative criterion excludes a camera when its median positive observed depth is below 10⁻⁵ times the 95th-percentile camera radius, or it has fewer than 100 positive-depth observations. Unsupported tracks and observations are filtered; only a global similarity normalization is applied, with no new bundle adjustment. Prepared 4× input retains 320 cameras / 67,152 points; native retains 457 / 242,918 (34 collapsed plus nine poorly supported cameras removed). Recovery started on the RTX 4090 at 2026-10-04T23:16:08Z. Every 320/320 4× and 457/457 native MVS depth bound is finite, positive and ordered; original intrinsics and camera poses (up to global similarity) were independently verified. Dense fusion, meshing and gsplat results remain pending. The private Insta360 page offers both original and screened sparse models, retaining every point (67,152 at 4× and 242,918 native) and showing excluded coverage. All original models, photographs and partial maps remain unchanged, and missing coverage must remain disclosed.

Recovery uses the original-derived undistorted photographs at 960×540 and 3840×2160, with data factor 1. Fixed holdout identities leave 280 training / 40 validation views at 4× and 402 / 55 at native resolution, with 39 shared held-out frames. Scores on 40 versus 55 views are not a direct same-view comparison. The optional common-set CPU comparison uses those 39 saved 8-bit native prediction canvases, is labeled separately from trainer float-render scores, and still compares different pixel resolutions.

`reports/insta360_recovery_results.json` records the separate aggregate recovery, while original reports remain unchanged. Reproduce preparation/queue/export with `prepare_insta360_recovery.py`, `run_insta360_recovery.py` and `export_insta360_recovery.py`; full checkpoints, SH3 PLYs and native prediction checks are covered by `validate_insta360_recovery_gsplat.py`. Recovery Gaussian models will use a separate local viewer on port 8793, preserving the original port 8791.

Hosting previews now use lossless float32 XYZ / byte RGB packing and, where it saves space, a reversible byte-plane transpose before gzip. Actual original/proposed JavaScript decoding agreed byte-for-byte on 40 clouds / 2,745,055 exported points; every count, order, camera and existing preview cap is preserved. See `tools/cloud_packing/README.md`. The screened sparse milestone archive is 264,837,120 expanded bytes, under the 256 MiB limit. Native photographs and full models are unchanged; geometry catalogs/buffers and prediction previews remain excluded from public GitHub.

## Step 08 — Full-resolution gsplat camera cleanup (2026-10-04)

The original full-resolution Gaussian run contained a recovered camera for frame 230 far outside the room trajectory. Only six sparse-point observations supported that pose. Because the pinned official gsplat trainer derives scene scale from the maximum camera radius, its scale was 21.649098214, versus about 2.48 at 4x. This scales position learning rates and Gaussian size thresholds. Native validation renders were already obscured before browser compression.

A separate copy removes frames 227, 228, 229, 230, 232 and 243. The reproducible rule excludes cameras with fewer than 20 surviving point observations or radius greater than five times the median; support is counted before deletion. Deregistration removes their observations and 31 points losing sufficient track support. The copy retains 487 cameras and 283,245 sparse points. Retained camera poses/intrinsics and original source hashes are verified unchanged. The normalized trainer scale is 2.476441577.

Training uses the same native 2126x3781 undistorted PNGs, data factor 1, gsplat 1.5.3, seed 42, 30,000 steps, packed rendering, SH3, batch size 1 and original default strategy. The same 62 original validation frames are explicitly held out; training has 425 frames. No bundle adjustment, intrinsic change or pose optimization is introduced. This tests camera/track cleanup together, not frame 230 alone.

Prepare with `PYTHONPATH=.colmap-tools python3 living-room-reconstruction/scripts/prepare_gsplat_fix.py`, then queue `.gsplat-env/bin/python living-room-reconstruction/scripts/run_gsplat_fix.py`. The runner waits on the shared GPU queue, saves checkpoints/validation renders/full PLY locally, and exports result previews after completion. Original Attempt 01 outputs are preserved. Step 8 uses the same four held-out viewpoints before/after; browser prediction previews are resized to 800px long edge then encoded losslessly. Full SH3 models remain local and can be opened through `serve_gsplat_fix.py` on localhost port 8792.

Completed on the RTX 4090 at 2026-10-04T22:31:42Z, preserving the original model. The timer uses monotonic nanoseconds around startup, training, native validation and saving, excluding queue waiting. Baseline time is available only at its saved precision; no extra precision is inferred.

| Measurement | Original full resolution | Cleaned-camera rerun |
| --- | ---: | ---: |
| Recovered / training / held-out views | 493 / 431 / 62 | 487 / 425 / 62 |
| Sparse initialization points | 283,276 | 283,245 |
| Valid / trained Gaussians | 343,663 / 343,663 | 1,355,457 / 1,355,457 |
| PSNR, same 62 native held-out views | 13.176762580871582 | 15.53478717803955 |
| SSIM | 0.7689108848571777 | 0.7694727182388306 |
| LPIPS | 0.6012338995933533 | 0.5254483222961426 |
| Run elapsed seconds | 1722.47 | 1701.8677089 |
| Full SH3 PLY bytes | 81,105,945 | 319,889,330 |

The cleaned model improves held-out appearance: PSNR +2.3580245971679688 dB, SSIM +0.000561833381652832, LPIPS −0.0757855772972107. This is a camera/track-cleanup experiment, not a measurement of physical room accuracy; uncertainty and poorly observed surfaces remain.

Final CPU integrity validation passed all 284 checks without warnings: checkpoint step and six tensor shapes, finite values and activated scales, complete 59-float SH3 PLY byte length/count, all 62 native 4252×3781 side-by-side canvases, exact ground-truth tile identities and unchanged original source hashes. Both PLYs retain every trained Gaussian; zero invalid rows were omitted. Run `.gsplat-env/bin/python living-room-reconstruction/scripts/validate_gsplat_fix.py` to reproduce the audit locally. Detailed audit files include private paths and remain local. `reports/gsplat_fix_results.json` contains aggregate validation and results only.

Step 8 publishes the same four before/after prediction previews, settings and measurements. Full models can be loaded through the local viewer; native photographs, camera geometry, checkpoints, full PLYs and private previews are excluded from public GitHub.


### Insta360 screened recovery: 4× dense and gsplat milestone (2026-10-05)

The separate 4× recovery completed CUDA geometric depth in **3083.15812182 seconds**, CPU fusion in **68.111270333 seconds**, and Poisson meshing in **6.695419913 seconds**. The fused cloud contains **587,559 finite points**; its mesh has **169,760 vertices and 320,861 valid triangles**. All 1,280 expected depth/normal maps have native 960×540 dimensions and valid byte counts. The private Site preserves every sparse point; its dense preview is explicitly sampled to 15,000 points, with the full dense cloud and mesh retained locally.

The 4× gsplat run completed 30,000 steps on the RTX 4090 in **338.643327543 seconds**, using 280 training and 40 held-out views. Held-out scores: **PSNR 17.227296829223633**, **SSIM 0.6609835028648376**, **LPIPS 0.5224205851554871**. The raw checkpoint contains 1,327,862 Gaussians, including 712 rows with non-finite log scales (2,136 values). The official finite-row SH3 export retains **1,327,150 Gaussians**, exactly **313,208,878 bytes**, with every exported field finite. Original checkpoints are preserved; omitted rows are reported rather than hidden. No other raw parameter tensor contains non-finite values.

This is **partial and inconsistent appearance recovery**. A kitchen/table validation view is recognizable, but several couch/window views become smooth occluding color fields. All 40 saved native validation canvases were decoded and their source halves matched prescribed photographs exactly. Independent projection checks agree before/after trainer normalization within approximately 8e-12 pixels on sampled 4× observations, ruling out an image-index, tile or normalization wiring error. Counts, low reprojection error and completed optimization do not establish faithful physical reconstruction.

Recovery previews and held-out renders are available in the private Insta360 chapter; the full unsampled SH3 model is served locally by `scripts/serve_insta360_recovery.py` on port 8793. Native-resolution Gaussian training remains queued/running in the same sequential queue, followed by native dense reconstruction. Aggregate scores use different 40/55 held-out subsets and image resolutions; 39 source-frame identities are shared and will be compared separately when both models finish. All photographs, full models, validation audit details and camera identity lists remain excluded from this code-only GitHub repository.


### Insta360 native gsplat and common-view comparison (2026-10-05)

Native-resolution Gaussian training completed 30,000 steps on the RTX 4090 in **1619.600885151 seconds**, with 402 training and 55 held-out source views. All **854,975** raw checkpoint Gaussians are finite; the complete SH3 PLY retains every row with **zero omitted Gaussians**, exactly **201,775,577 bytes**. All 55 saved native validation canvases were decoded, each containing a 3840×2160 source reference and prediction. Trainer float-render metrics are PSNR **14.379073143005371**, SSIM **0.692996621131897**, LPIPS **0.5352635979652405**.

The private report now compares native and 4× predictions at the same four source frames (81, 217, 337 and 497), and both complete local SH3 models are accessible through the independent recovery viewer. Both runs have partial, inconsistent appearance: the kitchen/table view is recognizable with blurred or ghosted detail, while several couch/window views are largely smooth, occluding color fields. A numerically finite native model is not a faithful whole-room reconstruction.

A separate reproducible CPU evaluation (`scripts/evaluate_insta360_recovery_common.py`) compares **39 identical held-out source frames**, using each model's saved 8-bit native-resolution PNG reference/prediction halves. It is separate from trainer float-render metrics and still confounded by image resolution and calibrated undistortion; it is not a controlled physical-accuracy comparison.

| Common 39-view mean | Native | 4× |
|---|---:|---:|
| PSNR ↑ | 14.814224512149126 | 17.36521323521932 |
| SSIM ↑ | 0.6918336844597107 | 0.6594010496941897 |
| LPIPS ↓ | 0.4991650138145838 | 0.5133265396341299 |

The mixed metric ranking does not support a general native-resolution quality advantage. Per-frame evaluations and photographs remain local; only aggregate means and reproducible code go to GitHub. Native dense stereo is still running in the existing single-GPU queue after passing its storage reserve guard.

The 4× non-finite scale diagnostic found **2,136 negative-infinite log-scale values**, zero NaNs or positive infinities, in 712 Gaussian rows. Initial float32 SfM positions contain 2,480 exact duplicate rows; 714 initial points are in duplicate groups of at least four. Upstream initializes scales from the log of mean three-neighbor distance, so zero-distance duplicate clusters are a strong explanation; this is not proven row-by-row attribution. Original models and checkpoints are preserved, and no automatic retraining or point deletion was performed.
