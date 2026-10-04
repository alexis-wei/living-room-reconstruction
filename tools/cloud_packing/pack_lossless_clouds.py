#!/usr/bin/env python3
"""Stage exact, reversible Site cloud packing without changing the source Site.

Requires NumPy. Run with --source-dist PATH --staging-root PATH. Only referenced
legacy cloud assets whose RGB float32 bit patterns all equal np.float32(b/255)
for a byte b are converted. XYZ bytes, point order, counts and all catalog
metadata except the format marker remain exact. overlay/ contains replacements;
originals/ archives the exact old bytes. Unrepresentable assets remain untouched.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any
from urllib.parse import urlsplit

import numpy as np


COMPACT_FORMAT = "xyz-f32-rgb-u8"
COMPACT_DTYPE = np.dtype([("xyz", "<f4", (3,)), ("rgb", "u1", (3,))], align=False)
RGB_LUT = np.asarray([np.float32(byte / 255) for byte in range(256)], dtype="<f4")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pointers(value: Any, path: tuple[Any, ...] = ()):
    """Yield all cloud catalog objects, including repeated asset references."""
    if isinstance(value, dict):
        if isinstance(value.get("url"), str) and urlsplit(value["url"]).path.endswith(".bin.gz"):
            yield path, value
        for key, child in value.items():
            yield from pointers(child, path + (key,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from pointers(child, path + (index,))


def pointer_text(path: tuple[Any, ...]) -> str:
    return "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in path)


def safe_asset_path(url: str) -> Path:
    parsed = urlsplit(url)
    path = PurePosixPath(parsed.path)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsupported non-local cloud URL: {url!r}")
    return Path(*path.parts)


def deterministic_gzip(payload: bytes) -> bytes:
    stream = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=stream, compresslevel=9, mtime=0) as archive:
        archive.write(payload)
    return stream.getvalue()


def archive_original(originals: Path, relative: Path, source: bytes) -> None:
    target = originals / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != source:
        raise ValueError(f"Existing original-byte archive differs: {target}")
    target.write_bytes(source)
    assert target.read_bytes() == source


def write_overlay(overlay: Path, relative: Path, data: bytes) -> None:
    target = overlay / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    assert target.read_bytes() == data


def decode_compact(payload: bytes, count: int) -> bytes:
    if len(payload) != count * COMPACT_DTYPE.itemsize:
        raise ValueError("Compact payload length does not match displayed_points")
    records = np.frombuffer(payload, dtype=COMPACT_DTYPE, count=count)
    decoded = np.empty((count, 6), dtype="<f4")
    decoded[:, :3] = records["xyz"]
    # Matches JS raw.getUint8()/255 assigned into Float32Array.
    decoded[:, 3:] = RGB_LUT[records["rgb"]]
    return decoded.tobytes(order="C")


def validate_asset(relative: Path, source_bytes: bytes, count: int, original_format: str | None) -> tuple[dict, bytes | None]:
    payload = gzip.decompress(source_bytes)
    report = {
        "url": relative.as_posix(), "displayed_points": count,
        "original_format": original_format or "xyz-f32-rgb-f32 (implicit)",
        "original_gzip_bytes": len(source_bytes), "original_payload_bytes": len(payload),
        "original_gzip_sha256": sha256(source_bytes), "original_payload_sha256": sha256(payload),
        "point_order_preserved": True, "gzip_level": None,
    }
    if original_format == COMPACT_FORMAT:
        decoded = decode_compact(payload, count)
        records = np.frombuffer(payload, dtype=COMPACT_DTYPE, count=count)
        if not np.isfinite(records["xyz"]).all():
            raise ValueError(f"Nonfinite XYZ in {relative}")
        report.update(status="already_compact_unchanged", count_validated=True, all_values_finite=True,
                      decoded_float32_sha256=sha256(decoded), rgb_lossless=True,
                      staged_gzip_bytes=len(source_bytes), staged_payload_bytes=len(payload), saved_gzip_bytes=0)
        return report, None
    if original_format not in (None, "xyz-f32-rgb-f32"):
        raise ValueError(f"Unsupported legacy format {original_format!r}: {relative}")
    if len(payload) != count * 24:
        raise ValueError(f"Legacy payload length mismatch for {relative}: {len(payload)} vs {count * 24}")
    points = np.frombuffer(payload, dtype="<f4").reshape((count, 6))
    if not np.isfinite(points).all():
        raise ValueError(f"Nonfinite legacy float32 values in {relative}")
    report.update(count_validated=True, all_values_finite=True)
    rgb = points[:, 3:]
    # Candidate calculation is merely a lookup; bit equality decides eligibility.
    candidates = np.clip(np.rint(rgb.astype(np.float64) * 255), 0, 255).astype("u1")
    restored_rgb = RGB_LUT[candidates]
    exact = rgb.view("<u4") == restored_rgb.view("<u4")
    mismatches = int(np.count_nonzero(~exact))
    report["rgb_nonrepresentable_components"] = mismatches
    if mismatches:
        indices = np.argwhere(~exact)
        point, channel = (int(x) for x in indices[0])
        report.update(status="nonrepresentable_unchanged", rgb_lossless=False,
                      first_nonrepresentable={"point_index": point, "rgb_channel": channel,
                                              "original_float32_bits_hex": f"{int(rgb.view('<u4')[point, channel]):08x}"},
                      staged_gzip_bytes=len(source_bytes), staged_payload_bytes=len(payload), saved_gzip_bytes=0)
        return report, None
    compact = np.empty(count, dtype=COMPACT_DTYPE)
    # Copy uint32 views to explicitly preserve all XYZ float32 bytes, including signed zero.
    compact["xyz"].view("<u4")[:] = points[:, :3].view("<u4")
    compact["rgb"] = candidates
    packed_payload = compact.tobytes(order="C")
    decoded = decode_compact(packed_payload, count)
    if decoded != payload:
        raise AssertionError(f"Decoded point bytes differ from original: {relative}")
    xyz_original = points[:, :3].tobytes(order="C")
    xyz_packed = np.frombuffer(packed_payload, dtype=COMPACT_DTYPE)["xyz"].tobytes(order="C")
    assert xyz_original == xyz_packed
    packed_gzip = deterministic_gzip(packed_payload)
    assert gzip.decompress(packed_gzip) == packed_payload
    assert decode_compact(gzip.decompress(packed_gzip), count) == payload
    report.update(rgb_lossless=True, xyz_bytes_equal=True, decoded_buffer_bytes_equal=True,
                  decoded_float32_sha256=sha256(decoded), xyz_float32_sha256=sha256(xyz_original),
                  staged_gzip_bytes=len(packed_gzip), staged_payload_bytes=len(packed_payload),
                  staged_gzip_sha256=sha256(packed_gzip), staged_payload_sha256=sha256(packed_payload),
                  saved_gzip_bytes=len(source_bytes) - len(packed_gzip),
                  saved_payload_bytes=len(payload) - len(packed_payload), gzip_level=9)
    if len(packed_gzip) >= len(source_bytes):
        report.update(status="no_compressed_savings_unchanged", staged_gzip_bytes=len(source_bytes),
                      staged_payload_bytes=len(payload), saved_gzip_bytes=0, saved_payload_bytes=0)
        return report, None
    report["status"] = "packed_losslessly"
    return report, packed_gzip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dist", required=True, type=Path)
    parser.add_argument("--staging-root", required=True, type=Path)
    args = parser.parse_args()
    source = args.source_dist.resolve()
    staging = args.staging_root.resolve()
    if source == staging or source in staging.parents or staging in source.parents:
        raise ValueError("Staging and source directories must be disjoint")
    overlay, originals = staging / "overlay", staging / "originals"
    if overlay.exists() and any(overlay.rglob("*")):
        raise ValueError("Overlay already contains files; use a fresh staging root")
    staging.mkdir(parents=True, exist_ok=True)
    viewer = source / "cloud-viewer.js"
    viewer_bytes = viewer.read_bytes()
    viewer_text = viewer_bytes.decode()
    required_decoder_fragments = ["cloud.format==='xyz-f32-rgb-u8'", "raw.getFloat32(i*15+j*4,true)",
                                 "raw.getUint8(i*15+12+j)/255", "new Float32Array(cloud.displayed_points*6)"]
    if not all(fragment in viewer_text for fragment in required_decoder_fragments):
        raise ValueError("Viewer decoder needs inspection; expected compact decoder not found")
    catalogs = {}
    references = {}
    for filename in sorted(source.rglob("*.json")):
        raw = filename.read_bytes()
        document = json.loads(raw)
        refs = list(pointers(document))
        if not refs:
            continue
        relative = filename.relative_to(source)
        catalogs[relative] = {"raw": raw, "document": document, "refs": refs}
        for pointer, entry in refs:
            asset = safe_asset_path(entry["url"])
            count = entry.get("displayed_points")
            if type(count) is not int or count < 0:
                raise ValueError(f"Invalid displayed_points in {relative}:{pointer_text(pointer)}")
            references.setdefault(asset, []).append({"catalog": relative, "pointer": pointer,
                                                    "entry": entry, "count": count, "format": entry.get("format")})
    files_before = {path.relative_to(source): (path.stat().st_size, sha256(path.read_bytes()))
                    for path in sorted(source.rglob("*")) if path.is_file()}
    asset_reports = []
    converted = set()
    for relative, refs in sorted(references.items()):
        counts = {ref["count"] for ref in refs}
        formats = {ref["format"] for ref in refs}
        if len(counts) != 1 or len(formats) != 1:
            raise ValueError(f"Inconsistent catalog references: {relative}")
        original = (source / relative).read_bytes()
        report, packed = validate_asset(relative, original, next(iter(counts)), next(iter(formats)))
        report["references"] = [{"catalog": ref["catalog"].as_posix(), "pointer": pointer_text(ref["pointer"]),
                                 "total_points": ref["entry"].get("total_points"),
                                 "sampled": ref["entry"].get("sampled"), "render_all": ref["entry"].get("render_all")}
                                for ref in refs]
        if packed is not None:
            archive_original(originals, relative, original)
            write_overlay(overlay, relative, packed)
            converted.add(relative)
        asset_reports.append(report)
        print(f"{report['status']}: {relative} ({report['saved_gzip_bytes']:+,} bytes saved)", flush=True)
    catalog_reports = []
    for relative, catalog in catalogs.items():
        original_document = copy.deepcopy(catalog["document"])
        edits = []
        for pointer, entry in catalog["refs"]:
            if safe_asset_path(entry["url"]) in converted:
                entry["format"] = COMPACT_FORMAT
                edits.append(pointer)
        changed = bool(edits)
        if changed:
            updated = (json.dumps(catalog["document"], indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()
            archive_original(originals, relative, catalog["raw"])
            write_overlay(overlay, relative, updated)
            restored = copy.deepcopy(json.loads(updated))
            for pointer in edits:
                entry = restored
                old_entry = original_document
                for part in pointer:
                    entry, old_entry = entry[part], old_entry[part]
                if "format" in old_entry:
                    entry["format"] = old_entry["format"]
                else:
                    del entry["format"]
            assert restored == original_document, f"Metadata changed besides format in {relative}"
        else:
            updated = catalog["raw"]
        # Every effective reference resolves to a valid buffer with matching format/count.
        for pointer, entry in pointers(json.loads(updated)):
            asset = safe_asset_path(entry["url"])
            effective = overlay / asset if asset in converted else source / asset
            payload = gzip.decompress(effective.read_bytes())
            stride = 15 if entry.get("format") == COMPACT_FORMAT else 24
            assert len(payload) == entry["displayed_points"] * stride
        catalog_reports.append({"path": relative.as_posix(), "status": "format_markers_updated" if changed else "unchanged",
                                "original_sha256": sha256(catalog["raw"]), "staged_sha256": sha256(updated),
                                "original_bytes": len(catalog["raw"]), "staged_bytes": len(updated),
                                "modified_format_pointers": [pointer_text(pointer) for pointer in edits],
                                "all_other_metadata_equal": True, "all_cloud_references_validated": True})
    files_after = {path.relative_to(source): (path.stat().st_size, sha256(path.read_bytes()))
                   for path in sorted(source.rglob("*")) if path.is_file()}
    assert files_before == files_after, "Source Site changed during packing"
    replacements = {path.relative_to(overlay): path.stat().st_size for path in overlay.rglob("*") if path.is_file()}
    before_bytes = sum(size for size, digest in files_before.values())
    after_bytes = sum(replacements.get(relative, size) for relative, (size, digest) in files_before.items())
    round_tar = lambda value: math.ceil(value / 512) * 512
    tar_block_savings = sum(round_tar(files_before[relative][0]) - round_tar(size) for relative, size in replacements.items())
    manifest = {"schema_version": 1, "source_dist": str(source), "source_site_git_head": subprocess.check_output(
                    ["git", "-C", str(source.parent), "rev-parse", "HEAD"], text=True).strip(),
                "viewer_decoder_sha256": sha256(viewer_bytes), "python_version": sys.version.split()[0], "numpy_version": np.__version__,
                "overlay": str(overlay), "originals": str(originals),
                "rules": {"rgb": "Every original RGB float32 bit pattern must equal np.float32(byte/255).",
                          "xyz": "Copy original little-endian XYZ float32 bytes exactly.",
                          "points": "No sampling, dropping, reordering, renormalization, cap or count changes.",
                          "metadata": "Only format markers for successfully converted assets are changed.",
                          "gzip": "Level 9; empty filename and mtime 0 for reproducibility."},
                "validation": {"passed": True, "all_referenced_buffers_count_and_finite_validated": True,
                               "all_packed_buffers_decode_byte_equal_to_original": True,
                               "all_catalog_metadata_preserved_except_format": True,
                               "source_site_all_file_sizes_and_sha256_unchanged": True,
                               "viewer_unchanged": True, "source_files_count": len(files_before)},
                "totals": {"catalog_count": len(catalogs), "catalog_reference_count": sum(len(refs) for refs in references.values()),
                           "unique_referenced_clouds": len(references), "packed_cloud_count": len(converted),
                           "unchanged_cloud_count": len(references) - len(converted),
                           "exported_points_unique_assets": sum(item["displayed_points"] for item in asset_reports),
                           "packed_exported_points": sum(item["displayed_points"] for item in asset_reports if item["status"] == "packed_losslessly"),
                           "cloud_gzip_savings_bytes": sum(item["saved_gzip_bytes"] for item in asset_reports),
                           "cloud_uncompressed_payload_savings_bytes": sum(item.get("saved_payload_bytes", 0) for item in asset_reports),
                           "catalog_growth_bytes": sum(item["staged_bytes"] - item["original_bytes"] for item in catalog_reports),
                           "source_site_file_bytes": before_bytes, "overlay_site_file_bytes": after_bytes,
                           "net_site_file_savings_bytes": before_bytes - after_bytes,
                           "tar_512_padded_file_content_savings_bytes": tar_block_savings},
                "catalogs": catalog_reports, "assets": asset_reports,
                "overlay_files": [{"path": relative.as_posix(), "bytes": size,
                                   "sha256": sha256((overlay / relative).read_bytes())} for relative, size in sorted(replacements.items())]}
    (staging / "validation_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(manifest["totals"], indent=2), flush=True)


if __name__ == "__main__":
    main()
