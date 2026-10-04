#!/usr/bin/env python3
"""Measure a reversible byte-plane candidate; never edits the source or baseline overlay.

Output is a measured candidate asset directory and proposed viewer code, not an
accepted deployment overlay. The primary publisher must accept the new format.
"""
from __future__ import annotations
import argparse
import copy
import gzip
import json
from pathlib import Path
import numpy as np
from pack_lossless_clouds import COMPACT_DTYPE, COMPACT_FORMAT, RGB_LUT, decode_compact, deterministic_gzip, sha256, pointers, safe_asset_path

PLANE_FORMAT = "xyz-f32-rgb-u8-shuffled"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-staging", required=True, type=Path)
    args = parser.parse_args()
    baseline = args.baseline_staging.resolve()
    manifest = json.loads((baseline / "validation_manifest.json").read_text())
    source, overlay = Path(manifest["source_dist"]), baseline / "overlay"
    candidate = baseline / "byte_plane_candidate"
    assets_root = candidate / "assets"
    candidate.mkdir(parents=True, exist_ok=True)
    records = []
    best_urls = set()
    for item in manifest["assets"]:
        relative = Path(item["url"])
        original = (source / relative).read_bytes()
        assert sha256(original) == item["original_gzip_sha256"]
        original_payload = gzip.decompress(original)
        effective = overlay / relative if (overlay / relative).exists() else source / relative
        effective_bytes = effective.read_bytes()
        count = item["displayed_points"]
        if effective == overlay / relative or item["status"] == "already_compact_unchanged":
            compact_payload = gzip.decompress(effective_bytes)
        else:
            points = np.frombuffer(original_payload, dtype="<f4").reshape(count, 6)
            candidates = np.clip(np.rint(points[:, 3:].astype(np.float64) * 255), 0, 255).astype("u1")
            assert np.array_equal(points[:, 3:].view("<u4"), RGB_LUT[candidates].view("<u4"))
            compact = np.empty(count, dtype=COMPACT_DTYPE)
            compact["xyz"].view("<u4")[:] = points[:, :3].view("<u4")
            compact["rgb"] = candidates
            compact_payload = compact.tobytes()
        assert len(compact_payload) == count * 15
        planes = np.frombuffer(compact_payload, dtype="u1").reshape(count, 15).T.copy().tobytes()
        plane_gzip = deterministic_gzip(planes)
        assert gzip.decompress(plane_gzip) == planes
        recovered_compact = np.frombuffer(gzip.decompress(plane_gzip), dtype="u1").reshape(15, count).T.copy().tobytes()
        assert recovered_compact == compact_payload
        expected_decoded = decode_compact(original_payload, count) if item["status"] == "already_compact_unchanged" else original_payload
        assert decode_compact(recovered_compact, count) == expected_decoded
        improved = len(plane_gzip) < len(effective_bytes)
        if improved:
            target = assets_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != plane_gzip:
                raise ValueError(f"Existing measured bytes differ: {target}")
            target.write_bytes(plane_gzip)
            best_urls.add(relative)
        record = {"url": item["url"], "displayed_points": count, "baseline_gzip_bytes": len(effective_bytes),
                  "candidate_gzip_bytes": len(plane_gzip), "extra_savings_bytes": len(effective_bytes) - len(plane_gzip),
                  "selected_for_candidate": improved, "gzip_level": 9,
                  "original_gzip_sha256": sha256(original), "candidate_gzip_sha256": sha256(plane_gzip),
                  "compact_unshuffle_bytes_equal": True, "decoded_float32_bytes_equal": True,
                  "decoded_float32_sha256": sha256(expected_decoded), "xyz_rgb_and_order_preserved": True}
        records.append(record)
        print(f"{relative}: {record['extra_savings_bytes']:+,} additional bytes", flush=True)
    metadata = []
    catalog_growth = 0
    for item in manifest["catalogs"]:
        relative = Path(item["path"])
        effective = overlay / relative if (overlay / relative).exists() else source / relative
        raw = effective.read_bytes()
        document = json.loads(raw)
        edits = []
        for pointer, cloud in pointers(document):
            if safe_asset_path(cloud["url"]) in best_urls:
                old_format = cloud.get("format")
                cloud["format"] = PLANE_FORMAT
                edits.append({"pointer": "/" + "/".join(str(part) for part in pointer),
                              "url": cloud["url"], "old_format": old_format, "proposed_format": PLANE_FORMAT})
        if edits:
            serialized = (json.dumps(document, indent=2, allow_nan=False) + "\n").encode()
            catalog_growth += len(serialized) - len(raw)
            metadata.append({"catalog": item["path"], "proposed_format_edits": edits,
                             "baseline_bytes": len(raw), "proposed_bytes": len(serialized)})
    viewer = (source / "cloud-viewer.js").read_text()
    old = "if(cloud.format==='xyz-f32-rgb-u8'){"
    new = ("if(cloud.format==='xyz-f32-rgb-u8-shuffled'){"
           "const n=cloud.displayed_points;"
           "if(!Number.isSafeInteger(n)||n<0||!Number.isSafeInteger(n*15)||buf.byteLength!==n*15)throw Error('Point count mismatch');"
           "const planes=new Uint8Array(buf),interleaved=new Uint8Array(buf.byteLength);"
           "for(let lane=0;lane<15;lane++)for(let i=0;i<n;i++)"
           "interleaved[i*15+lane]=planes[lane*n+i];buf=interleaved.buffer;}"
           "if(cloud.format==='xyz-f32-rgb-u8'||cloud.format==='xyz-f32-rgb-u8-shuffled'){")
    assert viewer.count(old) == 1
    proposed = viewer.replace(old, new)
    (candidate / "proposed_cloud-viewer.js").write_text(proposed)
    (candidate / "proposed_catalog_format_edits.json").write_text(json.dumps(metadata, indent=2) + "\n")
    additional = sum(max(0, item["extra_savings_bytes"]) for item in records)
    report = {"accepted_for_publication": False, "proposed_format": PLANE_FORMAT,
              "source_dist": str(source), "baseline_overlay": str(overlay), "candidate_assets": str(assets_root),
              "proposed_viewer": str(candidate / "proposed_cloud-viewer.js"),
              "proposed_catalog_edits": str(candidate / "proposed_catalog_format_edits.json"),
              "rules": {"encoding": "Interleaved compact bytes shaped (N,15), transposed to (15,N) and flattened.",
                        "decoding": "byteplanes[lane*N+i] copied to interleaved[i*15+lane], then existing compact decoder.",
                        "preservation": "Exact compact bytes and decoded XYZ/RGB float32 bytes; all points/order preserved."},
              "totals": {"clouds_measured": len(records), "candidate_improved_clouds": len(best_urls),
                         "candidate_exported_points": sum(item["displayed_points"] for item in records if item["selected_for_candidate"]),
                         "baseline_net_site_savings_bytes": manifest["totals"]["net_site_file_savings_bytes"],
                         "extra_cloud_gzip_savings_bytes": additional,
                         "extra_catalog_growth_bytes": catalog_growth,
                         "extra_viewer_growth_bytes": len(proposed.encode()) - len(viewer.encode()),
                         "candidate_total_net_site_savings_bytes": manifest["totals"]["net_site_file_savings_bytes"] + additional - catalog_growth - (len(proposed.encode()) - len(viewer.encode()))},
              "assets": records, "validation_passed": True}
    (candidate / "measurement_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["totals"], indent=2), flush=True)


if __name__ == "__main__":
    main()
