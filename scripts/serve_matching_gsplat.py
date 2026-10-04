"""Serve full matching-trial PLYs only on this computer, without uploading them."""
import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

WORKSPACE = Path(__file__).resolve().parents[2]
SITE = WORKSPACE / 'living-room-reconstruction/site/dist'
ROOT = WORKSPACE / 'gsplat_local/matching_20261003'
ALLOWED_ORIGINS = {'https://alexis-living-room-reconstruction.hello420892.chatgpt.site',
                   'http://127.0.0.1:8766', 'http://127.0.0.1:8790'}
PAGE = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Full local gsplat models</title><link rel="stylesheet" href="style.css"><script type="importmap">{"imports":{"three":"./vendor/three.module.js","three/addons/postprocessing/Pass.js":"./vendor/Pass.js"}}</script></head><body data-local-full-viewer="true"><main class="experiment-page"><p class="eyebrow">FULL MODELS · LOCAL RTX 4090 EXPERIMENT</p><h1>Matching-strategy gsplat models</h1><p>All valid exported Gaussians and degree-three spherical harmonics are retained in these local PLY files, without sampling. Each disconnected component has its own coordinate system. Rendering uses this browser’s graphics acceleration.</p><label>Model <select id="matching-full-model"></select></label><p id="matching-full-count"></p><p id="matching-full-state" role="status"></p><div id="matching-full-canvas"><p>The full model appears here after loading.</p></div><button id="matching-full-reset">Reset camera</button><button id="matching-full-fetch" hidden>Load from computer</button><a id="matching-full-local" hidden>Open locally</a><label>Or choose a local PLY <input id="matching-full-file" type="file" accept=".ply"></label><p>Drag to orbit · right-drag to pan · scroll to zoom. This server listens only on 127.0.0.1; no models or photographs leave this computer.</p></main><script type="module" src="matching-full-viewer.js"></script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def end_headers(self):
        origin = self.headers.get('Origin')
        if origin in ALLOWED_ORIGINS:
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Vary', 'Origin')
            self.send_header('Access-Control-Allow-Private-Network', 'true')
        super().end_headers()

    def do_OPTIONS(self):
        if self.headers.get('Origin') not in ALLOWED_ORIGINS:
            self.send_error(403)
            return
        self.send_response(204)
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Range')
        self.end_headers()

    def do_GET(self):
        origin = self.headers.get('Origin')
        if origin and origin not in ALLOWED_ORIGINS:
            self.send_error(403)
            return
        route = unquote(urlparse(self.path).path)
        if route == '/':
            data = PAGE.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if route == '/catalog.json':
            target = SITE / 'matching-gsplat/index.json'
        elif route.startswith('/model/') and route.endswith('.ply'):
            key = route[len('/model/'):-4]
            catalog = json.loads((SITE / 'matching-gsplat/index.json').read_text())
            if key not in catalog['models'] or catalog['models'][key]['state'] != 'complete':
                self.send_error(404, 'Completed model unavailable')
                return
            target = ROOT / 'runs' / key / 'ply/point_cloud_29999.ply'
        elif route in ('/style.css', '/matching-full-viewer.js') or route.startswith('/vendor/'):
            target = (SITE / route.lstrip('/')).resolve()
            allowed_base = SITE / 'vendor' if route.startswith('/vendor/') else SITE
            if not target.is_relative_to(allowed_base.resolve()):
                self.send_error(404)
                return
        else:
            self.send_error(404)
            return
        if not target.is_file():
            self.send_error(404)
            return
        length = target.stat().st_size
        if target.suffix == '.js':
            mime = 'text/javascript'
        elif target.suffix == '.ply':
            mime = 'application/octet-stream'
        else:
            mime = mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(length))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        try:
            with target.open('rb') as file:
                while chunk := file.read(1 << 20):
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8790)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Full gsplat viewer: http://127.0.0.1:{args.port}/', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
