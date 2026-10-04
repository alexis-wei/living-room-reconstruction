"""Re-encode existing private rendered PNGs as pixel-identical lossless WebP."""
import json
import hashlib
import shutil
from pathlib import Path
from PIL import Image


def main():
    workspace = Path(__file__).resolve().parents[2]
    site = workspace / 'living-room-reconstruction/site/dist'
    catalog_path = site / 'gaussians/index.json'
    catalog = json.loads(catalog_path.read_text())
    backup = workspace / 'gsplat_local/site_render_png_archive'
    backup.mkdir(exist_ok=True)
    old_files = []
    saved = 0
    for model in catalog.values():
        for view in model['views']:
            source = site / view['render']
            if source.suffix != '.png':
                continue
            target = source.with_suffix('.webp')
            with Image.open(source) as original:
                original.save(target, 'WEBP', lossless=True, method=6)
                with Image.open(target) as decoded:
                    assert original.mode == decoded.mode and original.size == decoded.size
                    assert original.tobytes() == decoded.tobytes()
            if target.stat().st_size >= source.stat().st_size:
                target.unlink()
                continue
            shutil.copy2(source, backup / source.name)
            saved += source.stat().st_size - target.stat().st_size
            view['render'] = target.relative_to(site).as_posix()
            old_files.append(source)
    catalog_path.write_text(json.dumps(catalog, indent=2) + '\n')
    for old in old_files:
        old.unlink()
    print(json.dumps({'renders_changed': len(old_files), 'bytes_saved': saved, 'pixels_unchanged': True}))
    # Preserve the same sixteen source examples, their dimensions and pixels.
    # Only the browser container format changes; local PNG masters stay intact.
    examples_path = site / 'examples/index.json'
    examples = json.loads(examples_path.read_text())
    examples['selection'] = 'The same four source frames at every resolution. Browser copies use pixel-verified lossless WebP; original PNGs remain local.'
    example_saved = 0
    old_files = []
    for scale, items in examples['scales'].items():
        for item in items:
            source = site / item['url']
            if source.suffix != '.png':
                continue
            target = source.with_suffix('.webp')
            with Image.open(source) as original:
                original.save(target, 'WEBP', lossless=True, method=6)
                with Image.open(target) as decoded:
                    assert original.mode == decoded.mode and original.size == decoded.size
                    assert original.tobytes() == decoded.tobytes()
            archive = workspace / 'gsplat_local/site_examples_png_archive' / scale
            archive.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, archive / source.name)
            example_saved += source.stat().st_size - target.stat().st_size
            item.update(url=target.relative_to(site).as_posix(), format='lossless WebP',
                        bytes=target.stat().st_size, sha256=hashlib.sha256(target.read_bytes()).hexdigest())
            old_files.append(source)
    examples_path.write_text(json.dumps(examples, indent=2) + '\n')
    for source in old_files:
        source.unlink()
    print(json.dumps({'example_copies_changed': len(old_files), 'bytes_saved': example_saved, 'pixels_and_resolutions_unchanged': True}))


if __name__ == '__main__':
    main()
