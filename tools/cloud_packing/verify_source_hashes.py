#!/usr/bin/env python3
"""Read-only preflight: fail if any source file targeted by the overlay changed."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("staging_root", type=Path)
    args = parser.parse_args()
    root = args.staging_root.resolve()
    manifest = json.loads((root / "combined_validation_manifest.json").read_text())
    baseline = json.loads((root / "validation_manifest.json").read_text())
    source = Path(baseline["source_dist"])
    for item in manifest["overlay_files"]:
        relative = Path(item["path"])
        archived = root / "originals" / relative
        assert (source / relative).read_bytes() == archived.read_bytes(), f"Target source bytes changed: {relative}"
        assert digest((root / "combined_overlay" / relative).read_bytes()) == item["sha256"]
    for item in manifest["assets"]:
        if item["source_group"] == "existing_site":
            original = source / item["url"]
        else:
            original = Path(item["original_path"])
        assert digest(original.read_bytes()) == item["original_gzip_sha256"], f"Original cloud changed: {item['url']}"
        assert digest(Path(item["effective_path"]).read_bytes()) == item["effective_gzip_sha256"]
    print(json.dumps({"passed": True, "overlay_replacements_verified": len(manifest["overlay_files"]),
                      "original_clouds_verified": len(manifest["assets"]),
                      "unrelated_source_site_files_not_checked": True}, indent=2))


if __name__ == "__main__":
    main()
