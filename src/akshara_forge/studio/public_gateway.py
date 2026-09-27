"""Restricted loopback frontend for the public NetBird demo session."""
import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import httpx

ASSETS = re.compile(r'/studio/[A-Za-z0-9_./-]+\.(?:css|js|woff2?|ttf)$')
ENVIRONMENT_PAGE = re.compile(r'/\?environment=[a-f0-9]{32}$')
JOBS = re.compile(r'/api/generation/[a-f0-9]{32}(?:/(?:download|evidence|page/[1-9][0-9]*))?$')


def permitted(method, path):
    if method == 'POST':
        return path in ('/api/generate', '/api/reset', '/api/step', '/api/agent')
    return method == 'GET' and (path in ('/', '/workbench', '/api/state', '/drawing', '/drawing?task=bridge', '/drawing?task=car', '/drawing?task=turbine', '/api/studio', '/api/bridge-sample', '/renderer.js', '/viewer.js') or bool(ENVIRONMENT_PAGE.fullmatch(path)) or bool(ASSETS.fullmatch(path)) or bool(JOBS.fullmatch(path)))


def trusted_request(host, origin, public_origin, method, upstream_host=None):
    parsed = urlsplit(public_origin)
    return (parsed.scheme == 'https' and bool(parsed.netloc) and host in (parsed.netloc, upstream_host) and host is not None
            and (origin == public_origin if method == 'POST' else origin in (None, public_origin)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--port', type=int, default=8789)
    p.add_argument('--bind', default='127.0.0.1')
    a = p.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, content, content_type='application/json'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(content)

        def handle_request(self):
            try:
                public_origin = json.loads(a.session.read_text())['origin']
            except (OSError, ValueError, KeyError):
                return self.respond(503, b'{"error":"Demo session is starting"}')
            if not trusted_request(self.headers.get('Host'), self.headers.get('Origin'), public_origin, self.command, f'{a.bind}:{a.port}'):
                return self.respond(403, b'{"error":"Invalid origin"}')
            if not permitted(self.command, self.path):
                return self.respond(403, b'{"error":"Operator-only endpoint"}')
            body = None
            if self.command == 'POST':
                if self.headers.get('Content-Type') != 'application/json':
                    return self.respond(415, b'{"error":"JSON required"}')
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                except ValueError:
                    size = 0
                if not 0 < size <= 12*1024*1024:
                    return self.respond(413, b'{"error":"Invalid upload size"}')
                self.connection.settimeout(30)
                body = self.rfile.read(size)
            try:
                with httpx.Client(timeout=60, trust_env=False) as client:
                    r = client.request(self.command, 'http://127.0.0.1:8787'+self.path, content=body,
                                       headers={'Content-Type':'application/json', 'Origin':'http://127.0.0.1:8787'})
                content = r.content
                if self.path == '/api/studio' and r.status_code == 200:
                    data = r.json()
                    data['generation']['can_configure'] = False
                    content = json.dumps(data).encode()
                self.respond(r.status_code, content, r.headers.get('content-type', 'application/octet-stream'))
            except httpx.HTTPError:
                self.respond(502, b'{"error":"Application unavailable"}')

        do_GET = handle_request
        do_POST = handle_request

    ThreadingHTTPServer((a.bind, a.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
