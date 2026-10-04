"""CPU validation of Step 08 checkpoints, SH3 exports and all held-out renders.

Run with the project-local .gsplat-env Python. This writes only a local audit
report; it never changes a checkpoint, model, photograph or trainer setting.
At step 6999 a PLY is not expected. Step 29999 requires the final SH3 PLY.
"""
import argparse
import datetime
import gc
import hashlib
import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / "gsplat_local/full_resolution_fix_20261004"
RUN = ROOT / "runs/full_resolution_cleaned"
CHUNK_ROWS = 32768
EXPECTED_SHAPES = {
    "means": (3,), "scales": (3,), "quats": (4,), "opacities": (),
    "sh0": (1, 3), "shN": (15, 3),
}
PLY_PROPERTIES = (
    ["x", "y", "z"] + [f"f_dc_{i}" for i in range(3)]
    + [f"f_rest_{i}" for i in range(45)] + ["opacity"]
    + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                sha256=sha256(path))


def json_safe(value):
    """Keep a failed finite-value audit writable as standards-compliant JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


class Audit:
    def __init__(self, step):
        self.report = dict(
            step=step, checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            device="CPU only", models_modified=False, photographs_modified=False,
            checks={}, errors=[], warnings=[], durations_seconds={},
            interpretation=("Artifact integrity and held-out image agreement do not measure "
                            "physical 3D accuracy; no surveyed ground truth is available."),
        )

    def check(self, name, passed, detail=None):
        self.report["checks"][name] = dict(passed=bool(passed), detail=detail)
        if not passed:
            self.report["errors"].append(name)
        return bool(passed)

    def warn(self, message):
        self.report["warnings"].append(message)

    def stage(self, name, fn):
        start = time.perf_counter_ns()
        try:
            return fn()
        except Exception as error:
            self.check(name + "_completed", False, f"{type(error).__name__}: {error}")
            return None
        finally:
            self.report["durations_seconds"][name] = (time.perf_counter_ns() - start) / 1e9


def validate_checkpoint(audit, step, metrics):
    path = RUN / f"ckpts/ckpt_{step}_rank0.pt"
    # mmap avoids a second full copy of a potentially large checkpoint. Every
    # tensor stays on CPU, and weights_only prevents arbitrary pickle execution.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    audit.report["checkpoint"] = dict(identity=identity(path),
                                      loader="torch.load(map_location='cpu', weights_only=True, mmap=True)")
    item = audit.report["checkpoint"]
    try:
        audit.check("checkpoint_step", checkpoint.get("step") == step,
                    dict(expected=step, actual=checkpoint.get("step")))
        splats = checkpoint["splats"]
        names = sorted(splats)
        audit.check("checkpoint_tensor_names", set(names) == set(EXPECTED_SHAPES), names)
        count = int(splats["means"].shape[0])
        item.update(gaussian_count=count, tensor_names=names, tensors={})
        audit.check("checkpoint_gaussian_count_positive", count > 0, count)
        audit.check("checkpoint_validation_metric_count", count == metrics["num_GS"],
                    dict(checkpoint=count, validation_metrics=metrics["num_GS"]))
        training_stats_path = RUN / f"stats/train_step{step:04d}_rank0.json"
        if training_stats_path.exists():
            training_stats = json.loads(training_stats_path.read_text())
            audit.check("checkpoint_training_metric_count", count == training_stats["num_GS"],
                        dict(checkpoint=count, training_metrics=training_stats["num_GS"]))
        invalid_rows = np.zeros(count, dtype=bool)
        overflow_rows = np.zeros(count, dtype=bool)
        underflow_rows = np.zeros(count, dtype=bool)
        for name, tail in EXPECTED_SHAPES.items():
            value = splats[name]
            shape_ok = isinstance(value, torch.Tensor) and tuple(value.shape) == (count, *tail)
            audit.check("checkpoint_shape_" + name, shape_ok,
                        dict(expected=[count, *tail], actual=list(value.shape)))
            if not shape_ok:
                continue
            audit.check("checkpoint_cpu_float32_" + name,
                        value.device.type == "cpu" and value.dtype == torch.float32,
                        dict(device=str(value.device), dtype=str(value.dtype)))
            finite_values = total_values = nonfinite_row_count = 0
            row_examples = []
            for start in range(0, count, CHUNK_ROWS):
                chunk = value[start:start + CHUNK_ROWS]
                finite = torch.isfinite(chunk).reshape(len(chunk), -1)
                row_bad = ~finite.all(dim=1)
                bad = row_bad.numpy()
                invalid_rows[start:start + len(chunk)] |= bad
                finite_values += int(finite.sum())
                total_values += finite.numel()
                nonfinite_row_count += int(row_bad.sum())
                if len(row_examples) < 20:
                    row_examples.extend((np.flatnonzero(bad) + start)[:20 - len(row_examples)].tolist())
                if name == "scales":
                    linear = torch.exp(chunk)
                    raw_finite = torch.isfinite(chunk).all(dim=1)
                    overflow_rows[start:start + len(chunk)] = (
                        raw_finite & ~torch.isfinite(linear).all(dim=1)
                    ).numpy()
                    underflow_rows[start:start + len(chunk)] = (
                        raw_finite & (linear == 0).any(dim=1)
                    ).numpy()
                del chunk, finite, row_bad
            item["tensors"][name] = dict(shape=list(value.shape), dtype=str(value.dtype),
                                          finite_values=finite_values, total_values=total_values,
                                          nonfinite_values=total_values - finite_values,
                                          nonfinite_rows=nonfinite_row_count,
                                          example_nonfinite_row_indices=row_examples)
        item.update(raw_nonfinite_gaussian_rows=int(invalid_rows.sum()),
                    float32_exp_scale_overflow_rows=int(overflow_rows.sum()),
                    float32_exp_scale_underflow_rows=int(underflow_rows.sum()),
                    example_exp_scale_overflow_row_indices=np.flatnonzero(overflow_rows)[:20].tolist(),
                    activation_check="CPU float32 exp(log_scales); distinct from raw checkpoint finiteness")
        if invalid_rows.any():
            audit.warn(f"Checkpoint retains {int(invalid_rows.sum())} Gaussian rows with non-finite raw parameters; raw checkpoint is preserved. Observed PLY omissions are reported separately.")
        if overflow_rows.any():
            audit.warn(f"{int(overflow_rows.sum())} Gaussian rows have finite log-scales but float32 exp(scales) overflows. Raw finite values alone do not establish renderability; do not attribute PLY omissions to overflow without evidence.")
        if underflow_rows.any():
            audit.warn(f"{int(underflow_rows.sum())} Gaussian rows have scales that underflow to zero under CPU float32 exp; no model parameters were changed.")
        return count
    finally:
        # Do not retain mapped checkpoint tensors during PLY and PNG decoding.
        del checkpoint
        gc.collect()


def validate_ply(audit, step, metrics):
    path = RUN / f"ply/point_cloud_{step}.ply"
    required = step == 29999
    if not path.exists() and not required:
        audit.report["ply"] = dict(state="not_expected_at_this_step", path=str(path),
                                   reason="Configured ply_steps is [30000]; no 7000-step PLY requested.")
        return
    audit.check("ply_exists", path.is_file(), str(path))
    if not path.is_file():
        return
    properties, elements, header = [], [], []
    with path.open("rb") as stream:
        while True:
            line = stream.readline(8192)
            if not line or sum(map(len, header)) + len(line) > 65536:
                raise ValueError("Missing or oversized PLY header")
            header.append(line)
            words = line.strip().split()
            if words[:1] == [b"element"]:
                elements.append([words[1].decode("ascii"), int(words[2])])
            if words[:1] == [b"property"]:
                properties.append([word.decode("ascii") for word in words[1:]])
            if words == [b"end_header"]:
                offset = stream.tell()
                break
    audit.check("ply_binary_little_endian", header[:2] == [b"ply\n", b"format binary_little_endian 1.0\n"])
    audit.check("ply_vertex_element_only", len(elements) == 1 and elements[0][0] == "vertex", elements)
    count = elements[0][1]
    expected_properties = [["float", name] for name in PLY_PROPERTIES]
    audit.check("ply_all_59_SH3_float_properties", properties == expected_properties, properties)
    if properties != expected_properties or len(elements) != 1:
        return
    expected_bytes = offset + count * len(PLY_PROPERTIES) * 4
    actual_bytes = path.stat().st_size
    audit.check("ply_exact_byte_count", actual_bytes == expected_bytes,
                dict(actual=actual_bytes, expected=expected_bytes, header=offset))
    audit.check("ply_count_within_validation_metric_count", 0 < count <= metrics["num_GS"],
                dict(exported=count, trained=metrics["num_GS"]))
    if actual_bytes != expected_bytes or count <= 0:
        return
    values = np.memmap(path, dtype="<f4", offset=offset, mode="r", shape=(count, 59))
    column_nonfinite = np.zeros(59, dtype=np.int64)
    nonfinite_rows = 0
    scale_overflow_rows = 0
    scale_start = PLY_PROPERTIES.index("scale_0")
    try:
        for start in range(0, count, CHUNK_ROWS):
            chunk = values[start:start + CHUNK_ROWS]
            finite = np.isfinite(chunk)
            column_nonfinite += (~finite).sum(axis=0)
            nonfinite_rows += int((~finite.all(axis=1)).sum())
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                scales = chunk[:, scale_start:scale_start + 3]
                scale_overflow_rows += int((np.isfinite(scales).all(axis=1)
                                           & ~np.isfinite(np.exp(scales)).all(axis=1)).sum())
            del chunk, finite
    finally:
        del values
        gc.collect()
    omitted = metrics["num_GS"] - count
    audit.report["ply"] = dict(identity=identity(path), state="checked", vertex_count=count,
                               trained_gaussian_count=metrics["num_GS"], omitted_gaussian_count=omitted,
                               float_properties=PLY_PROPERTIES, header_bytes=offset,
                               expected_file_bytes=expected_bytes, nonfinite_rows=nonfinite_rows,
                               nonfinite_values_by_property=dict(zip(PLY_PROPERTIES, map(int, column_nonfinite))),
                               float32_exp_scale_overflow_rows=scale_overflow_rows,
                               omission_interpretation="Observed exported count difference; no assumption that finite raw checkpoint rows must be exported or that scale overflow is the cause.")
    audit.check("ply_every_float_finite", not column_nonfinite.any(),
                dict(nonfinite_values=int(column_nonfinite.sum()), nonfinite_rows=nonfinite_rows))
    if omitted:
        audit.warn(f"Full SH3 PLY contains {count} valid exported rows versus {metrics['num_GS']} trained Gaussians ({omitted} omitted); raw checkpoint remains unchanged.")
    if scale_overflow_rows:
        audit.warn(f"Exported PLY retains {scale_overflow_rows} finite log-scale rows whose CPU float32 exp overflows. PLY field finiteness and count do not establish physical correctness or renderability.")


def validate_inputs_and_renders(audit, step):
    sys.path.insert(0, str(WORKSPACE / ".gsplat-src/gsplat/examples"))
    from datasets.colmap import Parser, Dataset
    preparation_path = ROOT / "preparation.json"
    preparation = json.loads(preparation_path.read_text())
    holdout_path = ROOT / "holdout_frames.json"
    held_out = json.loads(holdout_path.read_text())
    original = Parser(str(WORKSPACE / "gsplat_local/datasets/1x"), factor=1, normalize=True, test_every=8)
    cleaned = Parser(str(ROOT / "datasets/full_resolution_cleaned"), factor=1, normalize=True, test_every=8)
    original_validation = [original.image_names[i] for i in Dataset(original, split="val").indices]
    validation = [name for name in cleaned.image_names if name in set(held_out)]
    training = [name for name in cleaned.image_names if name not in set(held_out)]
    audit.check("exact_original_62_holdout_names_and_order",
                len(held_out) == len(set(held_out)) == 62
                and held_out == original_validation == validation == preparation["validation_frames"], held_out)
    audit.check("exact_disjoint_425_training_names", len(training) == 425
                and training == preparation["training_frames"] and not set(training).intersection(validation))
    audit.check("registered_camera_and_sparse_counts", len(cleaned.image_names) == 487
                and len(cleaned.points) == 283245,
                dict(registered_images=len(cleaned.image_names), sparse_points=len(cleaned.points)))
    audit.check("retained_intrinsics_equal_original", all(
        np.array_equal(original.Ks_dict[key], value) for key, value in cleaned.Ks_dict.items()))
    audit.check("input_pose_point_arrays_finite", np.isfinite(cleaned.camtoworlds).all()
                and np.isfinite(cleaned.points).all())
    source = WORKSPACE / "reconstruction_local/1x/dense/sparse"
    actual_hashes = {name: sha256(source / name) for name in preparation["original_source_hashes"]}
    audit.check("original_source_model_hashes_unchanged", actual_hashes == preparation["original_source_hashes"])
    audit.report["source_identities"] = dict(preparation=identity(preparation_path),
                                             holdout=identity(holdout_path),
                                             original_model_sha256=actual_hashes,
                                             validation_frame_order=validation,
                                             trainer_scene_scale_before=float(original.scene_scale * 1.1),
                                             trainer_scene_scale_after=float(cleaned.scene_scale * 1.1))
    names = {name: i for i, name in enumerate(cleaned.image_names)}
    render_dir = RUN / "renders"
    expected = {f"val_step{step}_{i:04d}.png" for i in range(62)}
    actual = {path.name for path in render_dir.glob(f"val_step{step}_*.png")}
    audit.check("exact_62_native_render_files", actual == expected,
                dict(count=len(actual), missing=sorted(expected - actual), extra=sorted(actual - expected)))
    rows = []
    for index, frame in enumerate(validation):
        path = render_dir / f"val_step{step}_{index:04d}.png"
        parser_index = names[frame]
        camera_id = cleaned.camera_ids[parser_index]
        width, height = map(int, cleaned.imsize_dict[camera_id])
        row = dict(index=index, frame=frame, path=str(path), expected_canvas_size=[width * 2, height])
        if not path.is_file():
            row.update(fully_decoded=False, error="Missing render")
            rows.append(row)
            continue
        try:
            with Image.open(path) as canvas:
                canvas.load()  # Decode the entire canvas, including its prediction tile.
                row.update(canvas_size=list(canvas.size), mode=canvas.mode, fully_decoded=True)
                size_ok = canvas.size == (width * 2, height)
                audit.check(f"render_{index:04d}_native_two_tile_dimensions", size_ok, row["canvas_size"])
                audit.check(f"render_{index:04d}_RGB", canvas.mode == "RGB", canvas.mode)
                if size_ok and canvas.mode == "RGB":
                    # The eval writer truncates (float pixels / 255 * 255) to
                    # uint8. Permit at most one LSB of that documented round trip.
                    with Image.open(cleaned.image_paths[parser_index]) as photo:
                        photo.load()
                        audit.check(f"render_{index:04d}_source_dimensions", photo.size == (width, height), list(photo.size))
                        truth = np.asarray(photo.convert("RGB"))
                        pixels = np.asarray(canvas)
                        max_difference = changed_values = 0
                        for start in range(0, height, 256):
                            difference = np.abs(pixels[start:start + 256, :width].astype(np.int16)
                                                - truth[start:start + 256].astype(np.int16))
                            max_difference = max(max_difference, int(difference.max()))
                            changed_values += int(np.count_nonzero(difference))
                        row.update(ground_truth_max_uint8_difference=max_difference,
                                   ground_truth_changed_channel_values=changed_values,
                                   source_frame_identity_verified=max_difference <= 1)
                        audit.check(f"render_{index:04d}_ground_truth_matches_holdout_frame",
                                    max_difference <= 1, dict(max_uint8_difference=max_difference,
                                                              changed_channel_values=changed_values))
                        del truth, pixels
        except Exception as error:
            row.update(fully_decoded=False, error=f"{type(error).__name__}: {error}")
            audit.check(f"render_{index:04d}_fully_decoded", False, row["error"])
        rows.append(row)
    audit.report["native_renders"] = dict(frame_count=len(rows),
                                         fully_decoded_count=sum(row.get("fully_decoded", False) for row in rows),
                                         tile_layout="Left: same held-out source photograph; right: saved prediction.",
                                         source_match_tolerance_uint8=1, files=rows)
    audit.check("all_62_native_render_canvases_decoded",
                len(rows) == 62 and all(row.get("fully_decoded") for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, choices=(6999, 29999), default=29999)
    args = parser.parse_args()
    torch.set_num_threads(2)
    audit = Audit(args.step)
    start = time.perf_counter_ns()
    audit.report["versions"] = dict(python=sys.version.split()[0], torch=torch.__version__,
                                     gsplat=importlib.metadata.version("gsplat"), numpy=np.__version__)
    metrics_path = RUN / f"stats/val_step{args.step}.json"
    metrics = None
    try:
        metrics = json.loads(metrics_path.read_text())
        audit.report["validation_metrics"] = dict(values=metrics, identity=identity(metrics_path),
                                                   domain="Mean held-out image appearance scores over the same 62 original views.")
        for key in ("psnr", "ssim", "lpips"):
            audit.check("validation_metric_finite_" + key,
                        isinstance(metrics.get(key), (int, float)) and math.isfinite(metrics[key]), metrics.get(key))
        audit.check("validation_metric_gaussian_count_integer", isinstance(metrics.get("num_GS"), int)
                    and metrics["num_GS"] > 0, metrics.get("num_GS"))
    except Exception as error:
        audit.check("validation_metrics_loaded", False, f"{type(error).__name__}: {error}")
    if metrics is not None:
        audit.stage("checkpoint", lambda: validate_checkpoint(audit, args.step, metrics))
        gc.collect()
        audit.stage("ply", lambda: validate_ply(audit, args.step, metrics))
    audit.stage("inputs_and_62_native_renders", lambda: validate_inputs_and_renders(audit, args.step))
    audit.report["durations_seconds"]["total"] = (time.perf_counter_ns() - start) / 1e9
    audit.report["state"] = "failed" if audit.report["errors"] else ("passed_with_warnings" if audit.report["warnings"] else "passed")
    report_path = ROOT / f"validation_report_step{args.step}.json"
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(json_safe(audit.report), indent=2, allow_nan=False) + "\n")
    temporary.replace(report_path)
    print(json.dumps(dict(state=audit.report["state"], report=str(report_path),
                          errors=audit.report["errors"], warnings=audit.report["warnings"],
                          seconds=audit.report["durations_seconds"]["total"]), indent=2))
    return 1 if audit.report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
