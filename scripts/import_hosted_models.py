"""Import an explicit, private asset manifest into the Site's authenticated R2 store.

Credentials arrive through hidden stdin, are never written to disk, and are sent
only to the selected Site origin. Every uploaded part is SHA-256 checked by the
Worker. Receipts contain file identities and checksums, never credentials.
"""
import argparse
import concurrent.futures
import hashlib
import json
import sys
import termios
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ORIGIN = 'https://alexis-living-room-reconstruction.hello420892.chatgpt.site'
PART_SIZE = 8 * 1024 * 1024

def hidden_input():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd) if sys.stdin.isatty() else None
    if old:
        new = list(old); new[3] &= ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSANOW, new)
    try:
        print('Ready for private model import credentials on stdin (hidden).', flush=True)
        return json.loads(sys.stdin.readline())
    finally:
        if old: termios.tcsetattr(fd, termios.TCSANOW, old)

def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(PART_SIZE), b''): h.update(block)
    return h.hexdigest()

class Client:
    def __init__(self, credentials):
        self.credentials = credentials
    def request(self, path, method='GET', body=None, headers=None, parse=True):
        if not path.startswith('/') or path.startswith('//'): raise ValueError('Site-relative path required')
        h = {'OAI-Sites-Authorization': 'Bearer ' + self.credentials['service_token']}
        h.update(headers or {})
        if isinstance(body, dict):
            body = json.dumps(body).encode(); h['Content-Type'] = 'application/json'
        for attempt in range(4):
            try:
                req = urllib.request.Request(ORIGIN + path, data=body, headers=h, method=method)
                with urllib.request.urlopen(req, timeout=120) as response:
                    data = response.read()
                    if parse: return json.loads(data)
                    return response.status, dict(response.headers), data
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise RuntimeError(f'Site request failed with HTTP {error.code}: {path.split("?")[0]}') from None
            except (urllib.error.URLError, TimeoutError):
                if attempt == 3: raise RuntimeError('Site connection failed after retries') from None
            time.sleep(min(2 ** attempt, 8))
    def upload(self, job):
        path = Path(job['path']); before = path.stat()
        if before.st_size != job['size'] or digest_file(path) != job['sha256']: raise RuntimeError(f'Input changed: {job["key"]}')
        params = {'key': job['key']}
        def endpoint(action, **extra):
            return '/__model-import?' + urllib.parse.urlencode(params | {'action':action} | extra)
        secret = {'X-Model-Import-Secret':self.credentials['import_secret']}
        created = self.request(endpoint('create'), 'POST', {'size':job['size'],'sha256':job['sha256'],'content_type':job.get('content_type','application/octet-stream')}, secret)
        if created.get('existing'):
            if created['size'] != job['size'] or created.get('sha256') != job['sha256']: raise RuntimeError('Existing immutable object differs')
            return {'key':job['key'],'size':job['size'],'sha256':job['sha256'],'state':'verified_existing'}
        upload_id = created['upload_id']; total = (job['size'] + PART_SIZE - 1) // PART_SIZE
        def upload_part(number):
            with path.open('rb') as f:
                f.seek((number-1)*PART_SIZE); data=f.read(PART_SIZE)
            sha = hashlib.sha256(data).hexdigest()
            result = self.request(endpoint('part',upload_id=upload_id,part=number), 'PUT', data, secret | {'X-Part-SHA256':sha,'Content-Type':'application/octet-stream','Content-Length':str(len(data))})
            if result['sha256'] != sha or result['partNumber'] != number: raise RuntimeError('Server part verification failed')
            return {'partNumber':number,'etag':result['etag']}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            parts = list(pool.map(upload_part, range(1,total+1)))
        result = self.request(endpoint('complete',upload_id=upload_id),'POST',{'parts':parts},secret)
        after = path.stat()
        if (after.st_size,after.st_mtime_ns) != (before.st_size,before.st_mtime_ns): raise RuntimeError('Source changed during upload')
        # Multipart completion can omit custom metadata in its response. The
        # independent HEAD read below is authoritative for stored metadata.
        if result['size'] != job['size'] or (result.get('sha256') is not None and result['sha256'] != job['sha256']): raise RuntimeError('Final object metadata verification failed')
        code, h, _ = self.request('/models/'+job['key'],'HEAD',parse=False)
        if code != 200 or int(h.get('Content-Length',0)) != job['size'] or h.get('X-Content-SHA256') != job['sha256']: raise RuntimeError('Hosted object readback verification failed')
        return {'key':job['key'],'size':job['size'],'sha256':job['sha256'],'state':'uploaded_verified','verified_parts':total}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('manifest',type=Path);parser.add_argument('--check',action='store_true');parser.add_argument('--workers',type=int,default=1,choices=[1,4]);args=parser.parse_args()
    client=Client(hidden_input())
    if args.check:
        print('storage',client.request('/__model-storage-status'),flush=True)
        code, headers, data = client.request('/gsplat',parse=False)
        print('page_check',{'status':code,'is_html':b'<!doctype html>' in data.lower(),'bytes':len(data)},flush=True)
        if args.manifest.exists():
            job=json.loads(args.manifest.read_text())['jobs'][0]
            code,h,_=client.request('/models/'+job['key'],'HEAD',parse=False)
            print('first_model',{'status':code,'bytes':h.get('Content-Length'),'sha256':h.get('X-Content-SHA256'),'expected_bytes':job['size'],'expected_sha256':job['sha256']},flush=True)
        return
    jobs=json.loads(args.manifest.read_text())['jobs']; receipt=args.manifest.with_name(args.manifest.stem+'_receipts.json')
    results=json.loads(receipt.read_text()) if receipt.exists() else []
    completed={r['key'] for r in results if r['state'] in ['uploaded_verified','verified_existing']}
    started=time.monotonic()
    pending=[job for job in jobs if job['key'] not in completed]
    if args.workers>1 and any(j['size']>PART_SIZE for j in pending):raise ValueError('Parallel objects are restricted to small streaming chunks; native models import one at a time.')
    def upload(job):
        print(f'Import: {job["key"]} · {job["size"]/1e6:.1f} MB',flush=True)
        return client.upload(job)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
      for result in pool.map(upload,pending):
        results.append(result);receipt.write_text(json.dumps(results,indent=2)+'\n')
        print(f'Verified {len(results)}/{len(jobs)} · elapsed {time.monotonic()-started:.1f} s',flush=True)
    print(json.dumps({'state':'complete','objects':len(results),'bytes':sum(r['size'] for r in results),'seconds':time.monotonic()-started}),flush=True)

if __name__=='__main__':main()
