"""Read-only local full SH3 viewer for screened Insta360 recovery models."""
from http.server import ThreadingHTTPServer
import serve_matching_gsplat as base
class Handler(base.Handler):
    catalog_path=base.WORKSPACE/'insta360_local/recovery/publication_stage/insta360-recovery-assets/gsplat.json'
    model_root=base.WORKSPACE/'insta360_local/recovery/gsplat'
if __name__=='__main__':
    base.ALLOWED_ORIGINS.add('http://127.0.0.1:8793')
    base.PAGE=base.PAGE.replace('Matching-strategy gsplat models','Insta360 screened recovery gsplat models')
    print('Screened recovery full-model viewer: http://127.0.0.1:8793/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8793),Handler).serve_forever()
