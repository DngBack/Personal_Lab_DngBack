#!/usr/bin/env python3
"""Readiness gate + length-based router for short/long vLLM pools."""

from __future__ import annotations

import http.client
import http.server
import json
import socketserver
import threading
import time
from typing import Callable

from readiness_gate import ReadinessState
from routing_utils import INFERENCE_PATHS, path_only, resolve_routing

GATED_PATHS = frozenset({"/health", "/ping"})


def start_length_router(
    *,
    public_host: str,
    public_port: int,
    short_host: str,
    short_port: int,
    long_host: str,
    long_port: int,
    split_prompt_tokens: int,
    model_name: str,
    state: ReadinessState,
    log: Callable[[str], None] | None = None,
) -> socketserver.ThreadingTCPServer:
    state_ref = state
    split_ref = split_prompt_tokens
    model_ref = model_name
    log_fn = log or (lambda _message: None)
    short_host_ref = short_host
    short_port_ref = short_port
    long_host_ref = long_host
    long_port_ref = long_port
    tokenize_url = f"http://{short_host}:{short_port}/tokenize"

    class RouterHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            return

        def _path(self) -> str:
            return path_only(self.path)

        def _gate_request(self) -> bool:
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

        def _backend_healthy(self, upstream_host: str, upstream_port: int) -> bool:
            connection = http.client.HTTPConnection(
                upstream_host,
                upstream_port,
                timeout=10,
            )
            try:
                connection.request("GET", "/health")
                upstream = connection.getresponse()
                upstream.read()
                return 200 <= upstream.status < 300
            except OSError:
                return False
            finally:
                connection.close()

        def _pick_upstream(
            self, body: bytes | None
        ) -> tuple[str, int, str, dict[str, object] | None]:
            if self._path() not in INFERENCE_PATHS or not body:
                return short_host_ref, short_port_ref, "short(default)", None

            try:
                payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return short_host_ref, short_port_ref, "short(parse-failed)", None

            if not isinstance(payload, dict):
                return short_host_ref, short_port_ref, "short(invalid-json)", None

            t0 = time.perf_counter()
            pool, prompt_tokens, method = resolve_routing(
                payload,
                split_prompt_tokens=split_ref,
                tokenize_url=tokenize_url if state_ref.ready else None,
                model_name=model_ref,
            )
            route_ms = (time.perf_counter() - t0) * 1000
            label = f"{pool}({prompt_tokens} tok,{method})"
            route_info = {
                "event": "route",
                "path": self._path(),
                "pool": pool,
                "prompt_tokens": prompt_tokens,
                "method": method,
                "request_bytes": len(body),
                "route_overhead_ms": round(route_ms, 3),
            }
            if pool == "long":
                return long_host_ref, long_port_ref, label, route_info
            return short_host_ref, short_port_ref, label, route_info

        def _proxy(
            self,
            body: bytes | None,
            upstream_host: str,
            upstream_port: int,
        ) -> int:
            connection = http.client.HTTPConnection(
                upstream_host,
                upstream_port,
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
                status = upstream.status
                self.send_response(status, upstream.reason)
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
                return status
            finally:
                connection.close()
            return 502

        def _handle(self) -> None:
            if self._gate_request():
                self._send_not_ready()
                return

            if (
                self.command in {"GET", "HEAD"}
                and self._path() in GATED_PATHS
            ):
                short_ok = self._backend_healthy(short_host_ref, short_port_ref)
                long_ok = self._backend_healthy(long_host_ref, long_port_ref)
                if short_ok and long_ok:
                    self.send_response(200, "OK")
                    body = b"OK"
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    if self.command != "HEAD":
                        self.wfile.write(body)
                else:
                    self._send_not_ready()
                return

            content_length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(content_length) if content_length > 0 else None
            upstream_host, upstream_port, route_label, route_info = self._pick_upstream(body)
            if route_info is not None:
                log_fn(json.dumps(route_info))
                log_fn(f"Route {self._path()} -> {route_label}")
            status = self._proxy(body, upstream_host, upstream_port)
            if route_info is not None:
                log_fn(
                    json.dumps(
                        {
                            "event": "route_done",
                            "path": route_info["path"],
                            "pool": route_info["pool"],
                            "worker": f"{upstream_host}:{upstream_port}",
                            "status": status,
                        }
                    )
                )

        def do_GET(self) -> None:
            self._handle()

        def do_HEAD(self) -> None:
            self._handle()

        def do_POST(self) -> None:
            self._handle()

        def do_PUT(self) -> None:
            self._handle()

        def do_DELETE(self) -> None:
            self._handle()

        def do_OPTIONS(self) -> None:
            self._handle()

        def do_PATCH(self) -> None:
            self._handle()

    class ThreadingRouterServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    server = ThreadingRouterServer((public_host, public_port), RouterHandler)
    thread = threading.Thread(
        target=server.serve_forever,
        name="length-router",
        daemon=True,
    )
    thread.start()
    log_fn(
        f"Length router on {public_host}:{public_port} "
        f"(split>={split_prompt_tokens} tok -> long {long_host}:{long_port}, "
        f"else short {short_host}:{short_port}; fast-path routing)"
    )
    server._router_thread = thread  # type: ignore[attr-defined]
    return server


def stop_length_router(server: socketserver.ThreadingTCPServer | None) -> None:
    if server is None:
        return
    server.shutdown()
    server.server_close()
