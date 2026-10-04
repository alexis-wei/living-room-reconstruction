#!/usr/bin/env python3
"""Time matched COLMAP 4.2.1 stages for three 500-frame pair strategies.

Runs sequential matching without loop detection, sequential matching with the
COLMAP vocabulary-tree loop detector, and exhaustive matching. Each starts
from an isolated copy of the same saved feature-only database; existing
comparison databases/models are never modified. All six timed COLMAP child
processes run serially, with configurable callback checkpoints and raw elapsed
nanoseconds saved for each stage. Images, databases, logs, and models stay in
the caller-selected local workspace.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import shutil
import socket
import sqlite3
import subprocess
import time
from pathlib import Path

# Paths default to the enclosing local project layout. Override them when the
# saved COLMAP files live elsewhere; no images, databases, logs, or models are
# written into this source repository by default.
RUN_DIR: Path
PROJECT_ROOT: Path
COMPARISON: Path
IMAGES: Path
SOURCE_DB: Path
MATCH_TEMPLATE: Path
_EXHAUSTIVE_TEMPLATE: Path
MAPPER_TEMPLATE: Path
COLMAP: Path
VOCAB_TREE: Path | None
CALLBACK_SECONDS = 10.0

METHODS = ("sequential_no_loop", "sequential_loop", "exhaustive")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def write_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def event(path: Path, value: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(value, separators=(",", ":")) + "\n")
        out.flush()
        os.fsync(out.fileno())


def validate_source() -> None:
    if not COLMAP.is_file() or not os.access(COLMAP, os.X_OK):
        raise RuntimeError(f"COLMAP launcher is missing or not executable: {COLMAP}")
    if not SOURCE_DB.is_file():
        raise RuntimeError(f"Source feature database is missing: {SOURCE_DB}")
    if not MATCH_TEMPLATE.is_file() or not _EXHAUSTIVE_TEMPLATE.is_file() or not MAPPER_TEMPLATE.is_file():
        raise RuntimeError("A saved COLMAP project template is missing")
    if not IMAGES.is_dir() or len(list(IMAGES.glob("*.png"))) != 500:
        raise RuntimeError(f"Expected exactly 500 PNG images under {IMAGES}")
    if VOCAB_TREE is None or not VOCAB_TREE.is_file() or VOCAB_TREE.stat().st_size < 10_000_000:
        raise RuntimeError("COLMAP's cached Flickr vocabulary tree is missing or incomplete")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()
    if not gpu or not any("RTX 4090" in name for name in gpu):
        raise RuntimeError(f"Expected the local RTX 4090; nvidia-smi reported {gpu}")
    existing = ("timings.json", "metadata.json", "events.jsonl", *METHODS)
    if any((RUN_DIR / name).exists() for name in existing):
        raise RuntimeError("This run directory already contains outputs; refusing to overwrite")


def prepare_database(method: str) -> tuple[Path, Path, Path]:
    work = RUN_DIR / method
    work.mkdir(parents=True)
    db = work / "database.db"
    with sqlite3.connect(SOURCE_DB) as source, sqlite3.connect(db) as target:
        source.backup(target)
        target.execute("PRAGMA journal_mode=DELETE")
        target.execute("DELETE FROM two_view_geometries")
        target.execute("DELETE FROM matches")
        target.commit()
        target.execute("VACUUM")
        health = target.execute("PRAGMA quick_check").fetchone()[0]
        counts = tuple(
            target.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("images", "keypoints", "descriptors", "matches", "two_view_geometries")
        )
        if health != "ok" or counts != (500, 500, 500, 0, 0):
            raise RuntimeError(f"Bad isolated database for {method}: {health}; counts={counts}")

    matching_path = work / "matching_project.ini"
    template = _EXHAUSTIVE_TEMPLATE if method == "exhaustive" else MATCH_TEMPLATE
    matching_text = template.read_text(encoding="utf-8")
    matching_text = matching_text.replace(
        "database_path=" + matching_text.split("database_path=", 1)[1].splitlines()[0],
        f"database_path={db}", 1,
    )
    matching_text = matching_text.replace(
        "image_path=" + matching_text.split("image_path=", 1)[1].splitlines()[0],
        f"image_path={IMAGES}", 1,
    )
    if method == "exhaustive":
        matching_text += "\n[ExhaustiveMatching]\nblock_size=50\n"
    else:
        loop = method == "sequential_loop"
        matching_text += (
            "\n[SequentialMatching]\n"
            "overlap=10\n"
            "quadratic_overlap=true\n"
            "expand_rig_images=true\n"
            f"loop_detection={'true' if loop else 'false'}\n"
            "loop_detection_period=10\n"
            "loop_detection_num_images=50\n"
            "loop_detection_min_index_distance=0\n"
            "loop_detection_num_nearest_neighbors=1\n"
            "loop_detection_num_checks=64\n"
            "loop_detection_num_images_after_verification=0\n"
            "loop_detection_max_num_features=-1\n"
        )
        if loop:
            matching_text += f"vocab_tree_path={VOCAB_TREE}\n"
        matching_text += "num_threads=-1\n"
    if "[FeatureMatching]" not in matching_text or not any(line.strip() in {"use_gpu=true", "use_gpu=1"} for line in matching_text.splitlines()):
        raise RuntimeError(f"GPU matching is not enabled in {template}")
    matching_path.write_text(matching_text, encoding="utf-8")

    output = work / "reconstruction"
    output.mkdir()
    mapper_path = work / "mapper_project.ini"
    mapper_text = MAPPER_TEMPLATE.read_text(encoding="utf-8")
    for key, value in (("database_path", db), ("image_path", IMAGES), ("output_path", output)):
        import re
        mapper_text, count = re.subn(rf"(?m)^{key}=.*$", f"{key}={value}", mapper_text, count=1)
        if count != 1 and key == "output_path":
            mapper_text, count = re.subn(
                r"(?m)^image_path=.*$",
                lambda match: match.group(0) + f"\noutput_path={value}",
                mapper_text, count=1,
            )
        if count != 1:
            raise RuntimeError(f"Could not set {key} in mapper template")
    mapper_path.write_text(mapper_text, encoding="utf-8")
    return db, matching_path, mapper_path


def warm_features(databases: dict[str, Path]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for method, db in databases.items():
        total = 0
        with sqlite3.connect(db) as con:
            for table in ("keypoints", "descriptors"):
                for (blob,) in con.execute(f"SELECT data FROM {table}"):
                    total += len(blob)
        sizes[method] = total
    if len(set(sizes.values())) != 1:
        raise RuntimeError(f"Feature payload differs across isolated databases: {sizes}")
    return sizes


def run_stage(
    name: str, command: list[str], log_path: Path,
    timings: dict[str, object], timing_path: Path, callbacks: Path,
) -> None:
    start_utc = utc_now()
    start_ns = time.perf_counter_ns()
    event(callbacks, {"event": "stage_started", "stage": name, "utc": start_utc})
    print(f"START {name} · {start_utc}", flush=True)
    with log_path.open("wb") as log:
        child = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
        next_tick = start_ns + int(CALLBACK_SECONDS * 1_000_000_000)
        try:
            while True:
                status = child.poll()
                now = time.perf_counter_ns()
                if status is not None:
                    break
                if now >= next_tick:
                    elapsed_ns = now - start_ns
                    event(callbacks, {
                        "event": "timer_checkpoint", "stage": name,
                        "elapsed_ns": elapsed_ns,
                        "elapsed_seconds": round(elapsed_ns / 1e9, 3), "utc": utc_now(),
                    })
                    print(f"CHECKPOINT {name} · {elapsed_ns / 1e9:.3f} s", flush=True)
                    while next_tick <= now:
                        next_tick += int(CALLBACK_SECONDS * 1_000_000_000)
                time.sleep(0.05)
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
    end_ns = time.perf_counter_ns()
    finished = {
        "event": "stage_finished", "stage": name, "return_code": status,
        "elapsed_ns": end_ns - start_ns,
        "elapsed_seconds": round((end_ns - start_ns) / 1e9, 3),
        "started_utc": start_utc, "ended_utc": utc_now(),
    }
    event(callbacks, finished)
    timings[name] = {**finished, "status": "completed" if status == 0 else "failed"}
    write_json(timing_path, timings)
    print(f"FINISH {name} · {finished['elapsed_seconds']:.3f} s · exit {status}", flush=True)
    if status != 0:
        raise subprocess.CalledProcessError(status, command)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd(),
                        help="Local workspace containing the images and COLMAP tools")
    parser.add_argument("--comparison-root", type=Path,
                        help="Folder with the saved sequential/exhaustive configs and feature database")
    parser.add_argument("--image-dir", type=Path, help="Folder containing the same 500 input PNGs")
    parser.add_argument("--source-database", type=Path, help="Feature database copied before each method")
    parser.add_argument("--sequential-config", type=Path, help="Saved sequential matching project INI")
    parser.add_argument("--exhaustive-config", type=Path, help="Saved exhaustive matching project INI")
    parser.add_argument("--mapper-config", type=Path, help="Saved CPU mapper project INI")
    parser.add_argument("--colmap-launcher", type=Path, help="CUDA-enabled COLMAP launcher")
    parser.add_argument("--vocab-tree", type=Path,
                        help="COLMAP SIFT vocabulary tree; defaults to its local cache")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New, empty local directory for copied databases, logs and models")
    parser.add_argument("--callback-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.callback_seconds <= 0:
        parser.error("--callback-seconds must be positive")

    global RUN_DIR, PROJECT_ROOT, COMPARISON, IMAGES, SOURCE_DB
    global MATCH_TEMPLATE, _EXHAUSTIVE_TEMPLATE, MAPPER_TEMPLATE, COLMAP
    global VOCAB_TREE, CALLBACK_SECONDS
    PROJECT_ROOT = args.workspace_root.expanduser().resolve()
    COMPARISON = (args.comparison_root or PROJECT_ROOT / "colmap_gui" / "iphone13pro_500_4x_comparison").expanduser().resolve()
    IMAGES = (args.image_dir or PROJECT_ROOT / "living_room_frames" / "downsample_4x_540x960").expanduser().resolve()
    SOURCE_DB = (args.source_database or COMPARISON / "sequential" / "database.db").expanduser().resolve()
    MATCH_TEMPLATE = (args.sequential_config or COMPARISON / "sequential" / "project.ini").expanduser().resolve()
    _EXHAUSTIVE_TEMPLATE = (args.exhaustive_config or COMPARISON / "exhaustive" / "project.ini").expanduser().resolve()
    MAPPER_TEMPLATE = (args.mapper_config or COMPARISON / "sequential" / "reconstruction" / "project.ini").expanduser().resolve()
    COLMAP = (args.colmap_launcher or PROJECT_ROOT / "colmap_gui" / "run_cuda_colmap.sh").expanduser().resolve()
    if args.vocab_tree:
        VOCAB_TREE = args.vocab_tree.expanduser().resolve()
    else:
        matches = sorted(Path.home().joinpath(".cache", "colmap").glob(
            "*-vocab_tree_faiss_flickr100K_words256K.bin"))
        VOCAB_TREE = matches[0] if matches else None
    RUN_DIR = args.output_dir.expanduser().resolve()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CALLBACK_SECONDS = args.callback_seconds
    validate_source()
    prepared = {method: prepare_database(method) for method in METHODS}
    databases = {method: paths[0] for method, paths in prepared.items()}
    feature_bytes = warm_features(databases)
    callbacks = RUN_DIR / "events.jsonl"
    timing_path = RUN_DIR / "timings.json"
    timings: dict[str, object] = {}
    metadata = {
        "title": "Three-way COLMAP sequential loop-detection timing study",
        "started_utc": utc_now(),
        "colmap_version": "COLMAP 4.2.1, CUDA build bd1fcf6",
        "machine": {
            "hostname": socket.gethostname(), "platform": platform.platform(),
            "logical_cpu_threads": os.cpu_count(),
        },
        "gpu": "NVIDIA GeForce RTX 4090, CUDA device 0 for SIFT matching",
        "inputs": {
            "images": 500, "dimensions": "540x960", "camera": "OPENCV",
            "features_reused": True, "feature_extraction_timed": False,
            "feature_payload_bytes_per_database": feature_bytes,
        },
        "methods": {
            "sequential_no_loop": {
                "pairing": "sequential", "overlap": 10,
                "quadratic_overlap": True, "loop_detection": False,
            },
            "sequential_loop": {
                "pairing": "sequential", "overlap": 10,
                "quadratic_overlap": True, "loop_detection": True,
                "vocabulary_tree": str(VOCAB_TREE),
                "loop_detection_period": 10, "retrieved_images": 50,
                "min_index_distance": 0, "nearest_neighbors": 1,
                "checks": 64, "images_after_verification": 0,
                "max_num_features": -1,
            },
            "exhaustive": {"pairing": "exhaustive", "block_size": 50},
        },
        "shared_matching": {
            "guided_matching": True, "use_gpu": True, "gpu_index": 0,
            "sift_max_ratio": 0.8, "sift_max_distance": 0.7,
            "sift_cross_check": True,
        },
        "shared_mapper": "Saved COLMAP mapper config; multiple models enabled, CPU bundle adjustment, all CPU threads.",
        "timing": "time.perf_counter_ns around each COLMAP child process; callback checkpoints every 10 seconds; process startup/shutdown included; database copies/reset, SIFT extraction, cache warmup, and between-stage gaps excluded.",
        "stage_order": [f"{m}_matching" for m in METHODS] + [f"{m}_sparse_mapping_bundle_adjustment" for m in METHODS],
        "existing_projects_preserved": True,
        "inputs_and_outputs_written_only_to_workspace": True,
    }
    write_json(RUN_DIR / "metadata.json", metadata)
    write_json(timing_path, timings)

    for method in METHODS:
        _, match_config, _ = prepared[method]
        matcher = "exhaustive_matcher" if method == "exhaustive" else "sequential_matcher"
        run_stage(
            f"{method}_matching", [str(COLMAP), matcher, "--project_path", str(match_config)],
            RUN_DIR / f"{method}_matching.log", timings, timing_path, callbacks,
        )
    for method in METHODS:
        _, _, mapper_config = prepared[method]
        run_stage(
            f"{method}_sparse_mapping_bundle_adjustment",
            [str(COLMAP), "mapper", "--project_path", str(mapper_config)],
            RUN_DIR / f"{method}_sparse_mapping.log", timings, timing_path, callbacks,
        )

    db_stats = {}
    for method, db in databases.items():
        with sqlite3.connect(db) as con:
            db_stats[method] = {
                "images": con.execute("SELECT COUNT(*) FROM images").fetchone()[0],
                "keypoint_rows": con.execute("SELECT COUNT(*) FROM keypoints").fetchone()[0],
                "descriptor_rows": con.execute("SELECT COUNT(*) FROM descriptors").fetchone()[0],
                "match_pair_records": con.execute("SELECT COUNT(*) FROM matches").fetchone()[0],
                "geometric_verification_records": con.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0],
                "geometries_with_inliers": con.execute("SELECT COUNT(*) FROM two_view_geometries WHERE rows>0 AND cols>0").fetchone()[0],
                "inlier_correspondences": con.execute("SELECT SUM(rows*cols) FROM two_view_geometries").fetchone()[0] or 0,
            }
    metadata["post_match_database_counts"] = db_stats
    metadata["output_models"] = {
        method: sorted(
            p.name for p in (RUN_DIR / method / "reconstruction").iterdir()
            if p.is_dir() and (p / "images.bin").exists()
        ) for method in METHODS
    }
    metadata["completed_utc"] = utc_now()
    write_json(RUN_DIR / "metadata.json", metadata)
    print(f"COMPLETE · raw timings saved to {timing_path}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
