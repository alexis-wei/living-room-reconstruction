"""Read-only localhost full-resolution Insta360 Gaussian viewer; no uploads."""
from http.server import ThreadingHTTPServer
import serve_matching_gsplat as base

class Handler(base.Handler):
    catalog_path=base.SITE/'insta360-assets/gsplat.json'
    model_root=base.WORKSPACE/'insta360_local/gsplat'

if __name__=='__main__':
    base.ALLOWED_ORIGINS.add('http://127.0.0.1:8791')
    base.PAGE=base.PAGE.replace('Matching-strategy gsplat models','Insta360 X5 gsplat models')
    print('Insta360 full-model viewer: http://127.0.0.1:8791/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8791),Handler).serve_forever()
