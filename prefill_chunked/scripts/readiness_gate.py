#!/usr/bin/env python3
"""Reverse proxy that blocks /health until warmup completes."""

from __future__ import annotations

import http.client
import http.server
import socketserver
import threading
from typing import Callable

GATED_PATHS = frozenset({"/health", "/ping"})


class ReadinessState:
    def __init__(self) -> None:
        self._ready = threading.Event()

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def open(self) -> None:
        self._ready.set()


def start_readiness_gate(
    *,
    public_host: str,
    public_port: int,
    upstream_host: str,
    upstream_port: int,
    state: ReadinessState,
    log: Callable[[str], None] | None = None,
) -> socketserver.ThreadingTCPServer:
    state_ref = state
    upstream_host_ref = upstream_host
    upstream_port_ref = upstream_port
    log_fn = log or (lambda _message: None)

    class ProxyHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            return

        def _path_only(self) -> str:
            return self.path.split("?", 1)[0]

        def _gate_request(self) -> bool:
            if state_ref.ready:
                return False
            if self.command in {"GET", "HEAD"} and self._path_only() in GATED_PATHS:
                return True
            return not state_ref.ready

        def _send_not_ready(self) -> None:
            body = b"Service warming up"
            self.send_response(503, "Service Unavailable")
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _filtered_request_headers(self) -> dict[str, str]:
            headers: dict[str, str] = {}
            for key, value in self.headers.items():
                if key.lower() in {"host", "connection", "proxy-connection"}:
                    continue
                headers[key] = value
            return headers

        def _proxy(self) -> None:
            content_length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(content_length) if content_length > 0 else None
            connection = http.client.HTTPConnection(
                upstream_host_ref,
                upstream_port_ref,
                timeout=600,
            )
            try:
                connection.request(
                    self.command,
                    self.path,
                    body=body,
                    headers=self._filtered_request_headers(),
                )
                upstream = connection.getresponse()
                self.send_response(upstream.status, upstream.reason)
                for header, value in upstream.getheaders():
                    lowered = header.lower()
                    if lowered in {
                        "connection",
                        "proxy-connection",
                        "transfer-encoding",
                    }:
                        continue
                    self.send_header(header, value)
                self.send_header("Connection", "close")
                self.end_headers()
                if self.command != "HEAD":
                    while True:
                        chunk = upstream.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            finally:
                connection.close()

        def do_GET(self) -> None:
            if self._gate_request():
                self._send_not_ready()
                return
            self._proxy()

        def do_HEAD(self) -> None:
            if self._gate_request():
                self._send_not_ready()
                return
            self._proxy()

        def do_POST(self) -> None:
            if self._gate_request():
                self._send_not_ready()
                return
            self._proxy()

        def do_PUT(self) -> None:
            if self._gate_request():
                self._send_not_ready()
                return
            self._proxy()

        def do_DELETE(self) -> None:
            if self._gate_request():
                self._send_not_ready()
                return
            self._proxy()

        def do_OPTIONS(self) -> None:
            if self._gate_request():
                self._send_not_ready()
                return
            self._proxy()

        def do_PATCH(self) -> None:
            if self._gate_request():
                self._send_not_ready()
                return
            self._proxy()

    class ThreadingGateServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    server = ThreadingGateServer((public_host, public_port), ProxyHandler)
    thread = threading.Thread(
        target=server.serve_forever,
        name="readiness-gate",
        daemon=True,
    )
    thread.start()
    log_fn(
        f"Readiness gate listening on {public_host}:{public_port} "
        f"-> {upstream_host}:{upstream_port}"
    )
    server._gate_thread = thread  # type: ignore[attr-defined]
    return server


def stop_readiness_gate(server: socketserver.ThreadingTCPServer | None) -> None:
    if server is None:
        return
    server.shutdown()
    server.server_close()
