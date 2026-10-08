import { env } from 'cloudflare:workers';

const PREFIX = 'reconstruction-v1/';
const PART_LIMIT = 8 * 1024 * 1024;
const json = (data, status = 200) => Response.json(data, {status, headers:{'Cache-Control':'no-store'}});
function keyFor(path) {
  if (!/^[a-zA-Z0-9][a-zA-Z0-9_./-]{0,350}$/.test(path) || path.split('/').some(s => s === '..' || s === '.')) throw Error('Invalid asset key');
  return PREFIX + path;
}
const hex = buffer => Array.from(new Uint8Array(buffer), b => b.toString(16).padStart(2, '0')).join('');

// The Sites dispatcher authenticates every request, including model/chunk reads.
// The separate upload secret grants writes only to the temporary import client.
export async function handle(request, bindings) {
  const url = new URL(request.url);
  try {
    if (url.pathname === '/__model-import') {
      if (!bindings.MODEL_IMPORT_SECRET || request.headers.get('X-Model-Import-Secret') !== bindings.MODEL_IMPORT_SECRET) return json({error:'Forbidden'},403);
      if (!bindings.BUCKET) return json({error:'Model storage unavailable'},503);
      const key = keyFor(url.searchParams.get('key') || ''), action = url.searchParams.get('action');
      if (request.method === 'POST' && action === 'create') {
        const metadata = await request.json();
        if (!Number.isSafeInteger(metadata.size) || metadata.size < 1 || !/^[a-f0-9]{64}$/.test(metadata.sha256)) return json({error:'Invalid metadata'},400);
        const existing = await bindings.BUCKET.head(key);
        if (existing) return json({existing:true,size:existing.size,sha256:existing.customMetadata?.sha256});
        const upload = await bindings.BUCKET.createMultipartUpload(key, {
          httpMetadata:{contentType:metadata.content_type || 'application/octet-stream'},
          customMetadata:{sha256:metadata.sha256,size:String(metadata.size)}
        });
        return json({upload_id:upload.uploadId});
      }
      const uploadId = url.searchParams.get('upload_id');
      if (!uploadId) return json({error:'Missing upload identifier'},400);
      const upload = bindings.BUCKET.resumeMultipartUpload(key,uploadId);
      if (request.method === 'PUT' && action === 'part') {
        const partNumber = Number(url.searchParams.get('part'));
        const length = Number(request.headers.get('Content-Length'));
        if (!Number.isInteger(partNumber) || partNumber < 1 || partNumber > 10000 || !Number.isSafeInteger(length) || length < 1 || length > PART_LIMIT) return json({error:'Invalid part'},400);
        const bytes = await request.arrayBuffer();
        if (bytes.byteLength !== length) return json({error:'Incomplete part'},400);
        const sha256 = hex(await crypto.subtle.digest('SHA-256',bytes));
        if (sha256 !== request.headers.get('X-Part-SHA256')) return json({error:'Part checksum mismatch'},400);
        const part = await upload.uploadPart(partNumber,bytes);
        return json({...part,sha256});
      }
      if (request.method === 'POST' && action === 'complete') {
        const {parts} = await request.json();
        if (!Array.isArray(parts) || !parts.length || parts.length > 10000 || parts.some((p,i) => p.partNumber !== i+1 || typeof p.etag !== 'string')) return json({error:'Invalid completed parts'},400);
        const object = await upload.complete(parts);
        return json({size:object.size,sha256:object.customMetadata?.sha256});
      }
      return json({error:'Unsupported import operation'},405);
    }
    if (url.pathname.startsWith('/models/')) {
      if (!['GET','HEAD'].includes(request.method)) return json({error:'Read-only models'},405);
      if (!bindings.BUCKET) return json({error:'Model storage unavailable'},503);
      const key = keyFor(decodeURIComponent(url.pathname.slice('/models/'.length)));
      const head = await bindings.BUCKET.head(key);
      if (!head) return json({error:'Model asset not found'},404);
      const headers = new Headers({'Accept-Ranges':'bytes','Cache-Control':'private, max-age=31536000, immutable','ETag':head.httpEtag,'X-Content-SHA256':head.customMetadata?.sha256 || '', 'X-Content-Type-Options':'nosniff'});
      head.writeHttpMetadata(headers);
      if (url.searchParams.has('download')) {const parts=key.split('/');const name=parts.at(-1)==='model.ply'?parts.slice(-4,-2).join('_')+'_SH3.ply':parts.at(-1);headers.set('Content-Disposition',`attachment; filename="${name}"`);}
      if (request.headers.get('If-None-Match') === head.httpEtag) return new Response(null,{status:304,headers});
      let range;
      const requestedRange = request.headers.get('Range');
      if (requestedRange) {
        const match = /^bytes=(\d*)-(\d*)$/.exec(requestedRange);
        if (!match || (!match[1] && !match[2])) return new Response(null,{status:416,headers:{'Content-Range':`bytes */${head.size}`}});
        const start = match[1] ? Number(match[1]) : Math.max(0,head.size-Number(match[2]));
        const end = match[1] && match[2] ? Math.min(head.size-1,Number(match[2])) : head.size-1;
        if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start > end || start >= head.size) return new Response(null,{status:416,headers:{'Content-Range':`bytes */${head.size}`}});
        range = {offset:start,length:end-start+1};
        headers.set('Content-Range',`bytes ${start}-${end}/${head.size}`);
      }
      headers.set('Content-Length',String(range?.length || head.size));
      if (request.method === 'HEAD') return new Response(null,{status:range?206:200,headers});
      const object = await bindings.BUCKET.get(key,range?{range}:undefined);
      if (!object) return json({error:'Model asset not found'},404);
      return new Response(object.body,{status:range?206:200,headers});
    }
    if (url.pathname === '/__model-storage-status') return json({storage:bindings.BUCKET?'ready':'unavailable',assets:!!bindings.ASSETS});
    if (!bindings.ASSETS) return json({error:'Site assets unavailable'},503);
    const assetUrl = new URL(url);
    if (assetUrl.pathname === '/') assetUrl.pathname = '/index.html';
    else if (!assetUrl.pathname.split('/').pop().includes('.')) assetUrl.pathname += '.html';
    return bindings.ASSETS.fetch(new Request(assetUrl,request));
  } catch (error) {
    console.error('Reconstruction asset request failed',url.pathname,error?.message);
    return json({error:'The asset request failed. Please retry.'},500);
  }
}
export default {fetch(request) {return handle(request,env);}};
