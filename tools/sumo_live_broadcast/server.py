"""Stdlib-only HTTP/SSE server for the live SUMO broadcast.

Endpoints:
    GET  /              single-page frontend
    GET  /api/init      static payload (network geometry, phases, controllers)
    GET  /api/stream    Server-Sent Events; one JSON frame per sim step
    POST /api/control   JSON commands (pause/speed/wait/switch/phase/
                        release/reset)

No third-party web dependencies are used: SSE is a long-lived chunked
response and control commands arrive as plain POSTs, which keeps the whole
feature runnable inside the existing colight environment.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent / 'static'
_MIME = {
    '.js': 'application/javascript; charset=utf-8',
    '.mjs': 'application/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.map': 'application/json',
    '.png': 'image/png',
    '.svg': 'image/svg+xml',
    '.woff2': 'font/woff2',
}


class _Handler(BaseHTTPRequestHandler):
    server_version = 'SumoLiveBroadcast/1.0'
    protocol_version = 'HTTP/1.1'

    # -- helpers ------------------------------------------------------------

    @property
    def engine(self):
        return self.server.engine

    def _send_static(self, rel_path):
        path = (STATIC_DIR / rel_path).resolve()
        if not path.is_file() or STATIC_DIR not in path.parents:
            self._send_json({'error': 'not found'}, status=404)
            return
        body = path.read_bytes()
        mime = _MIME.get(path.suffix.lower(), 'application/octet-stream')
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        body = html.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- routes -------------------------------------------------------------

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path in ('/', '/index.html'):
            self._send_html(self.server.page_html)
            return
        if path.startswith('/static/'):
            self._send_static(path[len('/static/'):])
            return
        if path == '/api/init':
            self._send_json(self.engine.init_payload())
            return
        if self.path.startswith('/api/stream'):
            self._stream()
            return
        self._send_json({'error': 'not found'}, status=404)

    def do_POST(self):
        if not self.path.startswith('/api/control'):
            self._send_json({'error': 'not found'}, status=404)
            return
        try:
            length = int(self.headers.get('Content-Length') or 0)
            payload = json.loads(self.rfile.read(length) or b'{}')
            op = payload.get('op')
            if op == 'start':
                self.engine.start_session(payload.get('scene'))
            elif op == 'stop':
                self.engine.stop_session()
            elif op == 'cancel':
                self.engine.cancel_build()
            elif op == 'pause':
                self.engine.pause(payload.get('value', True))
            elif op == 'resume':
                self.engine.pause(False)
            elif op == 'speed':
                self.engine.set_speed(payload['value'])
            elif op == 'wait':
                self.engine.set_wait(payload.get('value', True))
            elif op == 'switch':
                self.engine.switch_controller(payload['controller'])
            elif op == 'scene':
                self.engine.set_scene(payload['scene'])
            elif op == 'phase':
                self.engine.manual_phase(
                    payload['phase'], payload.get('junction'),
                )
            elif op == 'release':
                self.engine.release_manual(payload.get('junction'))
            elif op == 'resolve':
                self.engine.command(
                    'resolve_decision', phase=payload.get('phase'),
                    junction=payload.get('junction'),
                )
            elif op == 'reset':
                self.engine.reset()
            else:
                self._send_json({'error': f'unknown op: {op}'}, status=400)
                return
        except Exception as exc:
            self._send_json({'error': f'{type(exc).__name__}: {exc}'},
                            status=400)
            return
        frame, _ = self.engine.latest_frame(timeout=0)
        self._send_json({'ok': True, 'state': frame})

    def _stream(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Connection', 'keep-alive')
        self.send_header('X-Accel-Buffering', 'no')
        self.end_headers()
        version = 0
        try:
            while True:
                frame, version = self.engine.latest_frame(
                    since_version=version, timeout=20,
                )
                if frame is None:
                    break
                chunk = f'data: {json.dumps(frame, ensure_ascii=False)}\n\n'
                self.wfile.write(chunk.encode('utf-8'))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return

    def log_message(self, fmt, *args):  # keep the console clean
        pass


class BroadcastServer:
    def __init__(self, engine, page_html, host='127.0.0.1', port=8010):
        self.engine = engine
        self.page_html = page_html
        handler = _Handler
        self._httpd = ThreadingHTTPServer((host, int(port)), handler)
        self._httpd.engine = engine
        self._httpd.page_html = page_html
        self._httpd.daemon_threads = True

    @property
    def address(self):
        host, port = self._httpd.server_address[:2]
        return f'http://{host}:{port}'

    def serve_forever(self):
        self._httpd.serve_forever()

    def start_background(self):
        thread = threading.Thread(
            target=self._httpd.serve_forever, name='sumo-live-http',
            daemon=True,
        )
        thread.start()
        return thread

    def shutdown(self):
        self._httpd.shutdown()
        self._httpd.server_close()
