#!/usr/bin/env python3
"""Summarize pair counts and sparse-model coverage from a local timed run."""
import argparse
import json
import sqlite3
import subprocess
from pathlib import Path

METHODS = ("sequential_no_loop", "sequential_loop", "exhaustive")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Local output directory created by run_matching_strategy_timing.py")
    parser.add_argument("--colmap-launcher", type=Path, required=True,
                        help="CUDA-enabled COLMAP launcher; model conversion itself is CPU-only")
    args = parser.parse_args()
    run = args.run_dir.expanduser().resolve()
    launcher = args.colmap_launcher.expanduser().resolve()
    exports = run / "model_exports"
    exports.mkdir(exist_ok=True)
    summary = {}
    for method in METHODS:
        source = run / method / "reconstruction"
        components, all_names = [], set()
        if source.is_dir():
            models = sorted(p for p in source.iterdir() if p.is_dir() and (p / "images.bin").is_file())
        else:
            models = []
        for model in models:
            export = exports / method / model.name
            export.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                str(launcher), "model_converter", "--input_path", str(model),
                "--output_path", str(export), "--output_type", "TXT",
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            records = [line for line in (export / "images.txt").read_text().splitlines()
                       if not line.startswith("#")]
            names = {" ".join(line.split()[9:]) for line in records[::2]
                     if len(line.split()) >= 10}
            points = sum(1 for line in (export / "points3D.txt").read_text().splitlines()
                         if line.strip() and not line.startswith("#"))
            components.append({"model": model.name, "registered_images": len(names),
                               "sparse_points": points})
            all_names.update(names)
        db = run / method / "database.db"
        with sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True) as con:
            pairs = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
            geometry_records, geometries_with_inliers, inlier_correspondences = con.execute(
                "SELECT COUNT(*), SUM(CASE WHEN rows>0 AND cols>0 THEN 1 ELSE 0 END), SUM(rows*cols) "
                "FROM two_view_geometries").fetchone()
        summary[method] = {
            "pair_records": pairs,
            "geometric_verification_records": geometry_records,
            "geometries_with_inliers": geometries_with_inliers or 0,
            "inlier_correspondences": inlier_correspondences or 0,
            "model_count": len(components),
            "unique_registered_source_images": len(all_names),
            "largest_component_registered_images": max((x["registered_images"] for x in components), default=0),
            "sparse_points_summed_across_components": sum(x["sparse_points"] for x in components),
            "components": components,
        }
    (run / "model_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
