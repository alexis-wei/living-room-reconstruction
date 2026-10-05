"""Prepare reversible COLMAP voxel subsets, then validate with gsplat's Parser.

Preparation uses modern pycolmap, while --validate-parser uses the existing
gsplat environment and its legacy SceneManager. Neither mode initializes CUDA.
Only original measured representatives are retained; camera poses, calibration,
image pixels, retained point coordinates and tracks are never averaged or moved.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = WORKSPACE / "insta360_local/recovery/gsplat/datasets/4x_component_0"
DEFAULT_OUTPUT = WORKSPACE / "gsplat_local/voxel_experiments/insta360_4x_20261005"
DEFAULT_HOLDOUT = WORKSPACE / "insta360_local/recovery/gsplat/holdout_frames.json"
VOXELS = (("voxel_008", 0.08, 34199), ("voxel_020", 0.20, 17269))


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path):
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def binary_records(directory):
    return {path.name: file_record(path) for path in sorted(directory.glob("*.bin"))}


def link(target, destination):
    if destination.is_symlink():
        assert destination.resolve() == target.resolve(), destination
    elif destination.exists():
        raise FileExistsError(destination)
    else:
        destination.symlink_to(target.resolve(), target_is_directory=True)


def check_model(source, model, keep):
    """Check both sides of every retained point-to-image observation relation."""
    assert model.is_valid(), "Inconsistent COLMAP reconstruction"
    assert set(model.images) == set(source.images)
    assert set(model.cameras) == set(source.cameras)
    assert set(model.reg_image_ids()) == set(source.reg_image_ids())
    assert set(model.points3D) == keep
    camera_equal = pose_equal = image_xy_equal = True
    observations = 0
    for camera_id, camera in model.cameras.items():
        previous = source.cameras[camera_id]
        same = (camera.model == previous.model and camera.width == previous.width
                and camera.height == previous.height
                and np.array_equal(camera.params, previous.params))
        camera_equal &= same
        assert same, camera_id
    for image_id, image in model.images.items():
        previous = source.images[image_id]
        same_pose = np.array_equal(image.cam_from_world().matrix(), previous.cam_from_world().matrix())
        pose_equal &= same_pose
        assert same_pose and image.name == previous.name and image.camera_id == previous.camera_id
        assert len(image.points2D) == len(previous.points2D)
        for index, point2d in enumerate(image.points2D):
            old = previous.points2D[index]
            same_xy = np.array_equal(point2d.xy, old.xy)
            image_xy_equal &= same_xy
            assert same_xy
            if point2d.has_point3D():
                point_id = point2d.point3D_id
                assert point_id in keep and point_id == old.point3D_id
                observations += 1
            else:
                assert not old.has_point3D() or old.point3D_id not in keep
    min_track = min(point.track.length() for point in model.points3D.values())
    assert min_track >= 2
    for point_id, point in model.points3D.items():
        previous = source.points3D[point_id]
        assert np.array_equal(point.xyz, previous.xyz)
        assert np.array_equal(point.color, previous.color)
        assert point.error == previous.error
        actual = {(element.image_id, element.point2D_idx) for element in point.track.elements}
        before = {(element.image_id, element.point2D_idx) for element in previous.track.elements}
        assert actual == before
        assert np.isfinite(point.xyz).all() and np.isfinite(point.error)
        for image_id, index in actual:
            assert model.images[image_id].points2D[index].point3D_id == point_id
    assert observations == sum(point.track.length() for point in model.points3D.values())
    return {"is_valid": True, "registered_images": model.num_reg_images(),
            "retained_points": model.num_points3D(), "retained_observations": observations,
            "minimum_track_length": min_track, "camera_parameters_exact": bool(camera_equal),
            "camera_poses_exact": bool(pose_equal), "image_observation_coordinates_exact": bool(image_xy_equal),
            "retained_point_xyz_rgb_error_tracks_exact": True,
            "removed_observation_references_cleared": True,
            "mean_reprojection_error_px": model.compute_mean_reprojection_error(),
            "mean_track_length": model.compute_mean_track_length()}


def prepare(source_path, output, holdout_path):
    import pycolmap

    output.mkdir(parents=True, exist_ok=True)
    assert source_path.resolve() != output.resolve()
    source_sparse = source_path / "sparse"
    if (source_sparse / "0").is_dir():
        source_sparse /= "0"
    source = pycolmap.Reconstruction(source_sparse)
    assert source.is_valid() and source.num_reg_images() == 320 and source.num_points3D() == 67152
    before_hashes = binary_records(source_sparse)
    image_hashes = {path.name: file_record(path) for path in sorted((source_path / "images").glob("*.png"))}
    holdout = set(json.loads(holdout_path.read_text()))
    names = sorted(image.name for image in source.images.values())
    train = [name for name in names if name not in holdout]
    validation = [name for name in names if name in holdout]
    assert len(train) == 280 and len(validation) == 40
    xyz = np.asarray([source.points3D[point_id].xyz for point_id in sorted(source.points3D)])
    origin = np.median(xyz, axis=0)
    point_ids = sorted(source.points3D)
    points_by_priority = sorted(point_ids, key=lambda point_id: (
        -source.points3D[point_id].track.length(), source.points3D[point_id].error, point_id))
    datasets = output / "datasets"
    baseline_path = datasets / "baseline"
    baseline_path.mkdir(parents=True, exist_ok=True)
    link(source_path / "images", baseline_path / "images")
    link(source_sparse, baseline_path / "sparse")
    jobs = [{"name": "baseline", "voxel_size": None, "dataset": str(baseline_path),
             "points": source.num_points3D(), "registered_images": source.num_reg_images(),
             "source_untouched": True, "binary_files": before_hashes}]
    validation_records = {"baseline": check_model(source, source, set(point_ids))}
    for name, size, expected in VOXELS:
        selected = {}
        for point_id in points_by_priority:
            cell = tuple(np.floor((source.points3D[point_id].xyz - origin) / size).astype(np.int64))
            selected.setdefault(cell, point_id)
        keep = set(selected.values())
        assert len(keep) == expected, (name, len(keep), expected)
        dataset = datasets / name
        dataset.mkdir(parents=True, exist_ok=True)
        link(source_path / "images", dataset / "images")
        sparse = dataset / "sparse"
        existing = list(sparse.glob("*.bin")) if sparse.exists() else []
        if existing:
            model = pycolmap.Reconstruction(sparse)
        else:
            sparse.mkdir(parents=True, exist_ok=True)
            model = pycolmap.Reconstruction(source_sparse)
            for point_id in point_ids:
                if point_id not in keep:
                    model.delete_point3D(point_id)
            model.write_binary(sparse)
            model = pycolmap.Reconstruction(sparse)
        checked = check_model(source, model, keep)
        hashes = binary_records(sparse)
        assert hashes["cameras.bin"] == before_hashes["cameras.bin"]
        selected_ids = sorted(keep)
        selected_path = dataset / "retained_point_ids.json"
        save(selected_path, selected_ids)
        checked.update(voxel_size=size, voxel_grid_origin=origin.tolist(),
                       binary_files=hashes, retained_point_ids_sha256=sha256(selected_path),
                       images_symlink_exact=(dataset / "images").resolve() == (source_path / "images").resolve())
        validation_records[name] = checked
        jobs.append({"name": name, "voxel_size": size, "dataset": str(dataset), "points": len(keep),
                     "point_fraction": len(keep) / len(point_ids), "registered_images": model.num_reg_images(),
                     "binary_files": hashes, "selected_point_ids": str(selected_path)})
        print(f"{name}: {len(keep)} points; {model.num_reg_images()} unchanged cameras", flush=True)
    assert binary_records(source_sparse) == before_hashes
    assert {path.name: file_record(path) for path in sorted((source_path / "images").glob("*.png"))} == image_hashes
    manifest = {"created_at": now(), "source_dataset": str(source_path), "source_sparse": str(source_sparse),
                "source_binary_files": before_hashes, "source_image_files": image_hashes,
                "source_photographs_and_model_unchanged": True,
                "method": "One existing point per occupied voxel; longest track, lowest reprojection error, lowest point ID.",
                "voxel_grid_origin": origin.tolist(), "coordinate_units": "Arbitrary recovered model units; not meters.",
                "coordinate_quantization": False, "point_centroid_averaging": False,
                "camera_and_images_unchanged": True,
                "holdout_file": str(holdout_path), "holdout_sha256": sha256(holdout_path),
                "training_frames": train, "validation_frames": validation,
                "training_count": len(train), "validation_count": len(validation),
                "recommended_training_settings": {"data_factor": 1, "seed": 42, "max_steps": 30000,
                    "batch_size": 1, "sh_degree": 3, "packed": True, "init_type": "sfm",
                    "pose_opt": False, "app_opt": False, "ssim_lambda": 0.2, "strategy": "DefaultStrategy"},
                "comparison_limits": ["Lower initial point density is lower spatial detail, not improved calibration.",
                    "Voxel thinning changes nearest-neighbor initial Gaussian sizes.",
                    "Default densification can regrow points; fewer input points need not mean fewer final Gaussians.",
                    "Parser world normalization can choose a different equivalent PCA orientation per subset."],
                "jobs": jobs}
    save(output / "inputs_manifest.json", manifest)
    save(output / "validation_input.json", {"prepared_at": now(), "modern_pycolmap_version": pycolmap.__version__,
            "cpu_only": True, "original_model_and_images_hashes_unchanged": True,
            "modern_models": validation_records, "legacy_parser_state": "pending"})


def validate_parser(output):
    # Use the installed gsplat environment's legacy pycolmap/SceneManager.
    # Set CUDA_VISIBLE_DEVICES='' before launch for an independently CPU-only check.
    sys.path.insert(0, str(WORKSPACE / ".gsplat-src/gsplat/examples"))
    from datasets.colmap import Parser
    from utils import knn
    import torch

    manifest = json.loads((output / "inputs_manifest.json").read_text())
    heldout = set(json.loads(Path(manifest["holdout_file"]).read_text()))
    results = {}
    previous_parser = None
    for job in manifest["jobs"]:
        parser = Parser(job["dataset"], factor=1, normalize=True, test_every=8)
        assert len(parser.points) == job["points"] and len(parser.image_names) == 320
        assert all(np.isfinite(value).all() for value in [parser.points, parser.camtoworlds,
                                                        parser.transform, parser.points_err])
        assert all(np.isfinite(value).all() for value in parser.Ks_dict.values())
        assert set(parser.imsize_dict.values()) == {(960, 540)}
        train = [name for name in parser.image_names if name not in heldout]
        validation = [name for name in parser.image_names if name in heldout]
        assert train == manifest["training_frames"] and validation == manifest["validation_frames"]
        assert len(train) == 280 and len(validation) == 40
        assert all(Path(path).is_file() for path in parser.image_paths)
        assert all((indices >= 0).all() and (indices < len(parser.points)).all()
                   for indices in parser.point_indices.values())
        distances = knn(torch.from_numpy(parser.points).float(), 4)
        initial_scales = ((distances[:, 1:] ** 2).mean(dim=-1)).sqrt()
        assert initial_scales.device.type == "cpu"
        assert torch.isfinite(initial_scales).all()
        zero_scale_count = int((initial_scales <= 0).sum())
        # The preserved baseline can contain coincident recovered points. Keep
        # that original input intact and report its known initialization defect.
        # Voxel representatives must have strictly positive initial scales.
        if job["name"] != "baseline":
            assert zero_scale_count == 0, job["name"]
        if previous_parser is not None:
            assert parser.image_names == previous_parser.image_names
            assert [Path(path).resolve() for path in parser.image_paths] == [
                Path(path).resolve() for path in previous_parser.image_paths]
            for camera_id in parser.Ks_dict:
                assert np.array_equal(parser.Ks_dict[camera_id], previous_parser.Ks_dict[camera_id])
        else:
            previous_parser = parser
        results[job["name"]] = {"points": len(parser.points), "cameras": len(parser.image_names),
            "training_images": len(train), "validation_images": len(validation),
            "input_sizes": [list(value) for value in parser.imsize_dict.values()],
            "finite_points_poses_calibration_transform_errors": True,
            "observation_indices_in_range": True, "same_source_image_paths_and_intrinsics": True,
            "minimum_knn_initial_scale": float(initial_scales.min()),
            "maximum_knn_initial_scale": float(initial_scales.max()),
            "zero_initial_scales": zero_scale_count, "nonfinite_initial_scales": 0,
            "strictly_positive_initial_scales": zero_scale_count == 0,
            "baseline_original_coincident_points_preserved": job["name"] == "baseline" and zero_scale_count > 0,
            "parser_world_transform": parser.transform.tolist()}
        print(f"gsplat Parser {job['name']}: {len(parser.points)} points, 280 train / 40 validation, {zero_scale_count} zero CPU KNN scales", flush=True)
    report_path = output / "validation_input.json"
    report = json.loads(report_path.read_text())
    report.update(legacy_parser_state="passed_with_preserved_baseline_warning", legacy_validated_at=now(), legacy_parser=results,
                  pytorch_version=torch.__version__, cuda_initialized=torch.cuda.is_initialized())
    assert not report["cuda_initialized"]
    save(report_path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--validate-parser", action="store_true")
    args = parser.parse_args()
    if args.validate_parser:
        validate_parser(args.output.resolve())
    else:
        prepare(args.source.resolve(), args.output.resolve(), args.holdout.resolve())


if __name__ == "__main__":
    main()
