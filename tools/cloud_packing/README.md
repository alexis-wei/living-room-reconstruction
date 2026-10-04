# Lossless private point-cloud preview packing

These local helpers stage overlays for the private Site; they never publish or change the original input directory. Use a fresh staging directory and NumPy. Full source photographs and COLMAP/gsplat models remain local.

Run the scripts in this order, passing the actual input paths:

```text
pack_lossless_clouds.py --source-dist ORIGINAL_SITE_DIST --staging-root FRESH_OUTPUT
measure_byte_planes.py --baseline-staging FRESH_OUTPUT
stage_approved_shuffle.py --staging-root FRESH_OUTPUT --recovery-dist ORIGINAL_RECOVERY_PUBLICATION_STAGE
verify_js_decode.mjs FRESH_OUTPUT
verify_source_hashes.py FRESH_OUTPUT
```

Run Python helpers with a Python environment containing NumPy and the JavaScript helper with Node. The inputs should be the original legacy/compact snapshot, before applying this overlay. Exact original bytes and manifests remain in the private staging directory; do not upload them to the public repository.

Packing requires every RGB float32 bit pattern to equal `float32(byte / 255)` exactly. It preserves existing XYZ float32 bytes, point order, counts, camera metadata and preview caps. A reversible byte-plane transpose improves gzip compression; the viewer reverses it before the usual parser. Only buffers with positive savings are replaced. Full sparse previews remain complete, including software rendering.

The published October 4 overlay was checked with the actual original and proposed JavaScript decoder on 40 clouds and 2,745,055 exported points. Every decoded XYZ/RGB float32 buffer was byte-identical. Existing previews saved 7,627,485 file bytes; the new recovery clouds saved another 385,816 bytes. This packing does not change native model precision or training resolution.
