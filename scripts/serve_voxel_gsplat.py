"""Serve complete voxel-trial SH3 models on localhost without uploading them."""
import json
import mimetypes
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from serve_matching_gsplat import Handler as BaseHandler, WORKSPACE, SITE, ALLOWED_ORIGINS

ROOT = WORKSPACE / 'gsplat_local/voxel_experiments/insta360_4x_20261005'
ALLOWED_ORIGINS.update({'http://127.0.0.1:8794', 'http://127.0.0.1:8768'})


class Handler(BaseHandler):
    def target(self):
        route = unquote(urlparse(self.path).path)
        if route.startswith('/model/') and route.endswith('.ply'):
            key = route[len('/model/'):-4]
            catalog = json.loads((ROOT / 'local_catalog.json').read_text())
            if key not in catalog['models']:
                return None
            file = Path(catalog['models'][key]['ply_path']).resolve()
            allowed = [ROOT.resolve(), (WORKSPACE / 'insta360_local/recovery/gsplat').resolve()]
            return file if any(file.is_relative_to(base) for base in allowed) else None
        if route == '/catalog.json':
            return SITE / 'voxel-gsplat-assets/index.json'
        if route == '/':
            route = '/voxel-gsplat.html'
        target = (SITE / route.lstrip('/')).resolve()
        if not target.is_relative_to(SITE.resolve()) or target.suffix not in {'.html', '.js', '.css', '.json', '.webp', '.gz'}:
            return None
        return target

    def serve(self, head=False):
        origin = self.headers.get('Origin')
        if origin and origin not in ALLOWED_ORIGINS:
            self.send_error(403)
            return
        target = self.target()
        if target is None or not target.is_file():
            self.send_error(404)
            return
        mime = 'text/javascript' if target.suffix == '.js' else 'application/octet-stream' if target.suffix in {'.ply', '.gz'} else mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(target.stat().st_size))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        if head:
            return
        try:
            with target.open('rb') as stream:
                while chunk := stream.read(1 << 20):
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        self.serve()

    def do_HEAD(self):
        self.serve(head=True)


if __name__ == '__main__':
    print('Complete voxel comparison models: http://127.0.0.1:8794/', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8794), Handler).serve_forever()
