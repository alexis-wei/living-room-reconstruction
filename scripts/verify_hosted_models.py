"""Read-only verification of private model objects and deployed viewer routes.

Use hidden stdin credentials and explicit local manifests. Do not publish the
manifests or the full report: they contain identities of private model assets.
"""
import argparse
import concurrent.futures
import json
import urllib.error
import urllib.request
from pathlib import Path
from import_hosted_models import Client, hidden_input, ORIGIN


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',type=Path);parser.add_argument('--pages',action='store_true');args=parser.parse_args()
    client=Client(hidden_input());jobs={}
    for name in ['raw_import.json','streaming_import.json']:
        for j in json.loads((args.stage/name).read_text())['jobs']:jobs[j['key']]=j
    def check(j):
        code,headers,_=client.request('/models/'+j['key'],'HEAD',parse=False)
        if code!=200 or int(headers.get('Content-Length',0))!=j['size'] or headers.get('X-Content-SHA256')!=j['sha256']:raise RuntimeError('Hosted asset identity differs: '+j['key'])
        return {'key':j['key'],'bytes':j['size'],'sha256':j['sha256'],'verified':True}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(check,jobs.values()))
    first=next(j for j in jobs.values() if j['key'].endswith('/model.ply'))
    code,headers,body=client.request('/models/'+first['key'],headers={'Range':'bytes=0-8191'},parse=False)
    assert code==206 and body==Path(first['path']).open('rb').read(8192) and headers['Content-Range']==f'bytes 0-8191/{first["size"]}'
    anonymous={}
    for suffix in ['model.ply','model-lod.rad','model-lod-0.radc']:
        key=first['key'].rsplit('/',1)[0]+'/'+suffix
        try:
            with urllib.request.urlopen(urllib.request.Request(ORIGIN+'/models/'+key,method='HEAD'),timeout=30) as response:status=response.status
        except urllib.error.HTTPError as e:status=e.code
        assert status in [401,403],f'Anonymous access was not denied: {suffix}'
        anonymous[suffix]=status
    pages={}
    if args.pages:
        expected={'/gsplat':'gaussian-viewer-progressive.js','/matching':'matching-online-viewer.js','/insta360':'matching-online-viewer.js','/fixing-gsplat':'fixing-gsplat.js','/voxel-gsplat':'voxel-gsplat.js','/':'progressive SH3 Gaussian models'}
        for route,marker in expected.items():
            code,_,body=client.request(route,parse=False);assert code==200 and marker.encode() in body;pages[route]=code
        code,_,body=client.request('/online-models.json',parse=False);catalog=json.loads(body);assert code==200 and len(catalog['models'])==20
        try:
            client.request('/__model-import?key=test&action=create',headers={'X-Model-Import-Secret':client.credentials.get('import_secret','')},parse=False)
        except RuntimeError as error:
            if 'HTTP 403:' not in str(error):raise
        else:raise RuntimeError('Temporary import access is still enabled')
    report={'objects':len(results),'stored_bytes':sum(r['bytes'] for r in results),'range_readback_verified':True,'anonymous_statuses':anonymous,'pages':pages,'temporary_import_disabled':bool(args.pages),'assets':results}
    (args.stage/('production_verification.json' if args.pages else 'storage_verification.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='assets'}),flush=True)

if __name__=='__main__':main()
