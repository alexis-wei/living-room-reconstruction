"""Serve Step 8 and complete SH3 PLYs only on this computer."""
import argparse
import mimetypes
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse
import serve_matching_gsplat as base

WORKSPACE = Path(__file__).resolve().parents[2]
SITE = WORKSPACE / 'living-room-reconstruction/site-step8/dist'
ROOT = WORKSPACE / 'gsplat_local/full_resolution_fix_20261004'
base.ALLOWED_ORIGINS.add('http://127.0.0.1:8792')
base.PAGE = (SITE / 'fixing-gsplat.html').read_text()


class Handler(base.Handler):
    catalog_path = SITE / 'fixing-gsplat-assets/index.json'
    model_root = ROOT

    def do_GET(self):
        origin = self.headers.get('Origin')
        if origin and origin not in base.ALLOWED_ORIGINS:
            self.send_error(403)
            return
        route = unquote(urlparse(self.path).path)
        if route in ('/', '/catalog.json') or route.startswith('/model/'):
            return super().do_GET()
        if route in ('/style.css', '/chapter-nav.js', '/fixing-gsplat.js', '/fixing-gsplat.html') or route.startswith(('/vendor/', '/fixing-gsplat-assets/')):
            target = (SITE / route.lstrip('/')).resolve()
            if not target.is_relative_to(SITE.resolve()) or not target.is_file():
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/javascript' if target.suffix == '.js' else mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
            self.send_header('Content-Length', str(target.stat().st_size))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                with target.open('rb') as file:
                    while chunk := file.read(1 << 20):
                        self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self.send_error(404)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8792)
    args = parser.parse_args()
    print(f'Full-resolution cleanup viewer: http://127.0.0.1:{args.port}/', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
