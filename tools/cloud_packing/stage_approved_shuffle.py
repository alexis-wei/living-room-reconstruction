#!/usr/bin/env python3
"""Build local overlays for the publisher-approved lossless shuffled format.

Run after pack_lossless_clouds.py and measure_byte_planes.py. Source Site and
recovery publication-stage files are read only; exact originals are archived.
"""
import argparse
import copy
import gzip
import json
import math
from pathlib import Path
import numpy as np
from pack_lossless_clouds import COMPACT_FORMAT, archive_original, decode_compact, deterministic_gzip, pointers, safe_asset_path, sha256, write_overlay
from measure_byte_planes import PLANE_FORMAT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", required=True, type=Path)
    parser.add_argument("--recovery-dist", required=True, type=Path)
    args = parser.parse_args()
    staging = args.staging_root.resolve()
    baseline = json.loads((staging / "validation_manifest.json").read_text())
    measurement = json.loads((staging / "byte_plane_candidate/measurement_manifest.json").read_text())
    assert measurement["proposed_format"] == PLANE_FORMAT
    source = Path(baseline["source_dist"])
    base_overlay = staging / "overlay"
    combined = staging / "combined_overlay"
    archive = staging / "originals"
    candidate = staging / "byte_plane_candidate"
    shuffled_urls = {Path(item["url"]) for item in measurement["assets"] if item["selected_for_candidate"]}
    base_packed_urls = {Path(item["url"]) for item in baseline["assets"] if item["status"] == "packed_losslessly"}
    for filename in sorted(base_overlay.rglob("*")):
        if filename.is_file():
            write_overlay(combined, filename.relative_to(base_overlay), filename.read_bytes())
    for relative in sorted(shuffled_urls):
        archive_original(archive, relative, (source / relative).read_bytes())
        write_overlay(combined, relative, (candidate / "assets" / relative).read_bytes())
    for item in baseline["catalogs"]:
        relative = Path(item["path"])
        old_raw = (source / relative).read_bytes()
        old_document = json.loads(old_raw)
        document = copy.deepcopy(old_document)
        for pointer, cloud in pointers(document):
            url = safe_asset_path(cloud["url"])
            if url in shuffled_urls:
                cloud["format"] = PLANE_FORMAT
            elif url in base_packed_urls:
                cloud["format"] = COMPACT_FORMAT
        restored = copy.deepcopy(document)
        for (old_pointer, old), (new_pointer, new) in zip(pointers(old_document), pointers(restored)):
            assert old_pointer == new_pointer
            if "format" in old:
                new["format"] = old["format"]
            else:
                new.pop("format", None)
        assert restored == old_document
        updated = (json.dumps(document, indent=2, allow_nan=False) + "\n").encode()
        if document != old_document:
            archive_original(archive, relative, old_raw)
            write_overlay(combined, relative, updated)
    old_viewer = (source / "cloud-viewer.js").read_bytes()
    new_viewer = (candidate / "proposed_cloud-viewer.js").read_bytes()
    archive_original(archive, Path("cloud-viewer.js"), old_viewer)
    write_overlay(combined, Path("cloud-viewer.js"), new_viewer)
    asset_records = []
    for item in baseline["assets"]:
        relative = Path(item["url"])
        archive_original(archive, relative, (source / relative).read_bytes())
        effective = combined / relative if (combined / relative).exists() else source / relative
        fmt = PLANE_FORMAT if relative in shuffled_urls else COMPACT_FORMAT if relative in base_packed_urls or item["status"] == "already_compact_unchanged" else None
        asset_records.append({"url": item["url"], "count": item["displayed_points"],
                              "original_path": str(archive / relative),
                              "original_format": COMPACT_FORMAT if item["status"] == "already_compact_unchanged" else None,
                              "original_gzip_sha256": item["original_gzip_sha256"],
                              "effective_path": str(effective), "effective_format": fmt,
                              "effective_gzip_sha256": sha256(effective.read_bytes()),
                              "source_group": "existing_site"})
    recovery = args.recovery_dist.resolve()
    recovery_overlay = staging / "recovery_shuffle_overlay"
    recovery_archive = staging / "recovery_originals"
    recovery_records = []
    recovery_metadata_growth = 0
    recovery_asset_savings = 0
    recovery_original_bytes = sum(file.stat().st_size for file in recovery.rglob("*") if file.is_file())
    for catalog in sorted(recovery.rglob("*.json")):
        raw = catalog.read_bytes()
        document = json.loads(raw)
        before_document = copy.deepcopy(document)
        refs = list(pointers(document))
        if not refs:
            continue
        changed = False
        for pointer, cloud in refs:
            assert cloud.get("format") == COMPACT_FORMAT
            relative = safe_asset_path(cloud["url"])
            original = (recovery / relative).read_bytes()
            payload = gzip.decompress(original)
            count = cloud["displayed_points"]
            decoded = decode_compact(payload, count)
            assert np.isfinite(np.frombuffer(decoded, dtype="<f4")).all()
            shuffled = np.frombuffer(payload, dtype="u1").reshape(count, 15).T.copy().tobytes()
            encoded = deterministic_gzip(shuffled)
            assert np.frombuffer(gzip.decompress(encoded), dtype="u1").reshape(15, count).T.copy().tobytes() == payload
            improved = len(encoded) < len(original)
            effective = recovery / relative
            if improved:
                archive_original(recovery_archive, relative, original)
                write_overlay(recovery_overlay, relative, encoded)
                effective = recovery_overlay / relative
                cloud["format"] = PLANE_FORMAT
                changed = True
                recovery_asset_savings += len(original) - len(encoded)
            record = {"url": relative.as_posix(), "count": count,
                      "original_path": str(recovery / relative), "original_format": COMPACT_FORMAT,
                      "original_gzip_sha256": sha256(original), "original_gzip_bytes": len(original),
                      "effective_path": str(effective), "effective_format": PLANE_FORMAT if improved else COMPACT_FORMAT,
                      "effective_gzip_sha256": sha256(effective.read_bytes()), "effective_gzip_bytes": effective.stat().st_size,
                      "extra_savings_bytes": len(original) - effective.stat().st_size,
                      "decoded_float32_sha256": sha256(decoded), "source_group": "new_recovery"}
            recovery_records.append(record)
        restored = copy.deepcopy(document)
        for (old_pointer, old), (new_pointer, new) in zip(pointers(before_document), pointers(restored)):
            assert old_pointer == new_pointer
            new["format"] = old["format"]
        assert restored == before_document
        if changed:
            relative = catalog.relative_to(recovery)
            updated = (json.dumps(document, indent=2, allow_nan=False) + "\n").encode()
            archive_original(recovery_archive, relative, raw)
            write_overlay(recovery_overlay, relative, updated)
            recovery_metadata_growth += len(updated) - len(raw)
    for item in asset_records + recovery_records:
        assert sha256(Path(item["original_path"]).read_bytes()) == item["original_gzip_sha256"]
    replacements = {file.relative_to(combined): file.read_bytes() for file in combined.rglob("*") if file.is_file()}
    net_existing = sum((source / relative).stat().st_size - len(raw) for relative, raw in replacements.items())
    pad = lambda size: math.ceil(size / 512) * 512
    tar_existing = sum(pad((source / relative).stat().st_size) - pad(len(raw)) for relative, raw in replacements.items())
    report = {"schema_version": 1, "accepted_by_primary_publisher_for_staging": True,
              "format": PLANE_FORMAT, "combined_overlay": str(combined), "recovery_overlay": str(recovery_overlay),
              "source_dist": str(source), "source_viewer_path": str(archive / "cloud-viewer.js"), "staged_viewer_path": str(combined / "cloud-viewer.js"),
              "source_site_git_head": baseline["source_site_git_head"],
              "totals": {"existing_site_unique_clouds": len(asset_records),
                         "existing_site_exported_points": sum(item["count"] for item in asset_records),
                         "existing_site_shuffled_clouds": len(shuffled_urls),
                         "existing_site_net_file_savings_bytes": net_existing,
                         "existing_site_tar_512_padded_savings_bytes": tar_existing,
                         "new_recovery_unique_clouds": len(recovery_records),
                         "new_recovery_exported_points": sum(item["count"] for item in recovery_records),
                         "new_recovery_cloud_gzip_savings_bytes": recovery_asset_savings,
                         "new_recovery_catalog_growth_bytes": recovery_metadata_growth,
                         "new_recovery_net_file_savings_bytes": recovery_asset_savings - recovery_metadata_growth,
                         "new_recovery_source_file_bytes": recovery_original_bytes,
                         "new_recovery_effective_file_bytes": recovery_original_bytes - recovery_asset_savings + recovery_metadata_growth},
              "metadata_validation": {"all_catalog_changes_are_format_only": True,
                                      "all_source_cloud_bytes_unchanged": True,
                                      "no_sampling_dropping_reordering_or_cap_changes": True},
              "assets": asset_records + recovery_records,
              "overlay_files": [{"path": str(relative), "bytes": len(raw), "sha256": sha256(raw)} for relative, raw in sorted(replacements.items())]}
    (staging / "combined_validation_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["totals"], indent=2))


if __name__ == "__main__":
    main()
