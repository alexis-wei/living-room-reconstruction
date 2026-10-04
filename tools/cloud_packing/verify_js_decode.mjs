// Verify the actual decoder copied into staged cloud-viewer.js, in the JS runtime.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import zlib from 'node:zlib';
import assert from 'node:assert/strict';

const root = path.resolve(process.argv[2]);
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'combined_validation_manifest.json'), 'utf8'));
const digest = value => crypto.createHash('sha256').update(value).digest('hex');
const arrayBuffer = bytes => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
function extractDecoder(filename, firstMarker) {
  const source = fs.readFileSync(filename, 'utf8');
  const start = source.indexOf(firstMarker);
  const end = source.indexOf('renderer.load(buf', start);
  assert.ok(start >= 0 && end > start, `Decoder not found in ${filename}`);
  return new Function('input', 'cloud', 'let buf=input;' + source.slice(start, end) + 'return buf;');
}
const oldDecode = extractDecoder(manifest.source_viewer_path, "if(cloud.format==='xyz-f32-rgb-u8'){");
const newDecode = extractDecoder(manifest.staged_viewer_path, "if(cloud.format==='xyz-f32-rgb-u8-shuffled'){");
const results = [];
for (const asset of manifest.assets) {
  const originalGzip = fs.readFileSync(asset.original_path);
  const stagedGzip = fs.readFileSync(asset.effective_path);
  assert.equal(digest(originalGzip), asset.original_gzip_sha256);
  assert.equal(digest(stagedGzip), asset.effective_gzip_sha256);
  const original = zlib.gunzipSync(originalGzip);
  const staged = zlib.gunzipSync(stagedGzip);
  const expected = Buffer.from(oldDecode(arrayBuffer(original), {format:asset.original_format, displayed_points:asset.count}));
  const actual = Buffer.from(newDecode(arrayBuffer(staged), {format:asset.effective_format, displayed_points:asset.count}));
  assert.equal(expected.byteLength, asset.count * 24);
  assert.equal(actual.byteLength, expected.byteLength);
  assert.ok(actual.equals(expected), `Decoded float32 bytes differ: ${asset.url}`);
  const points = new Float32Array(arrayBuffer(actual));
  for (const value of points) assert.ok(Number.isFinite(value), `Nonfinite decoded value: ${asset.url}`);
  results.push({url:asset.url, source_group:asset.source_group, count:asset.count, effective_format:asset.effective_format,
                decoded_float32_bytes:actual.byteLength, decoded_float32_sha256:digest(actual),
                decoded_xyz_rgb_all_float32_bytes_equal:true, count_and_order_equal:true, all_finite:true});
}
const invalidCounts = [-1, .5, Number.MAX_SAFE_INTEGER + 1, Number.MAX_SAFE_INTEGER];
for (const displayed_points of invalidCounts) {
  assert.throws(() => newDecode(new ArrayBuffer(0), {format:manifest.format, displayed_points}), /Point count mismatch/);
}
assert.throws(() => newDecode(new ArrayBuffer(14), {format:manifest.format, displayed_points:1}), /Point count mismatch/);
assert.throws(() => newDecode(new ArrayBuffer(16), {format:manifest.format, displayed_points:1}), /Point count mismatch/);
assert.equal(newDecode(new ArrayBuffer(0), {format:manifest.format, displayed_points:0}).byteLength, 0);
const report = {validation_passed:true, node_version:process.version, actual_staged_viewer_decoder_executed:true,
                actual_original_viewer_decoder_executed:true, staged_viewer_sha256:digest(fs.readFileSync(manifest.staged_viewer_path)),
                all_original_source_cloud_bytes_verified:true, backwards_compatible_legacy_and_compact_formats:true,
                shuffled_count_bounds_and_payload_length_validation_passed:true,
                unique_clouds:results.length, exported_points:results.reduce((total,item) => total + item.count, 0),
                assets:results};
fs.writeFileSync(path.join(root, 'js_decode_validation.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({validation_passed:report.validation_passed, unique_clouds:report.unique_clouds, exported_points:report.exported_points,
                           node_version:report.node_version, staged_viewer_sha256:report.staged_viewer_sha256}, null, 2));
