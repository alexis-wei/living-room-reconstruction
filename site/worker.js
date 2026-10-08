import { env } from 'cloudflare:workers';

const PREFIX = 'reconstruction-v1/';
const json = (data, status = 200) => Response.json(data, {status, headers:{'Cache-Control':'no-store'}});
function keyFor(path) {
  if (!/^[a-zA-Z0-9][a-zA-Z0-9_./-]{0,350}$/.test(path) || path.split('/').some(s => s === '..' || s === '.')) throw Error('Invalid asset key');
  return PREFIX + path;
}

// The Sites dispatcher authenticates every request, including model/chunk reads.
// Uploads used a temporary import Worker; the final viewer has no write route.
export async function handle(request, bindings) {
  const url = new URL(request.url);
  try {
    // Production is read-only after verified imports. No runtime secret can reopen uploads.
    if (url.pathname === '/__model-import') return json({error:'Imports are disabled on the published viewer'},403);
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
