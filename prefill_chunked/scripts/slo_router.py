#!/usr/bin/env python3
"""SLO-aware multi-replica router for short/long vLLM workers."""

from __future__ import annotations

import http.client
import http.server
import json
import socketserver
import threading
import time
from typing import Callable

from readiness_gate import ReadinessState
from routing_utils import (
    INFERENCE_PATHS,
    WorkerEndpoint,
    WorkerState,
    path_only,
    select_slo_worker,
    track_request_end,
    track_request_start,
)

GATED_PATHS = frozenset({"/health", "/ping"})


def start_slo_router(
    *,
    public_host: str,
    public_port: int,
    workers: list[WorkerEndpoint],
    split_prompt_tokens: int,
    model_name: str,
    state: ReadinessState,
    prefix_locality: bool = True,
    log: Callable[[str], None] | None = None,
) -> socketserver.ThreadingTCPServer:
    state_ref = state
    split_ref = split_prompt_tokens
    model_ref = model_name
    log_fn = log or (lambda _message: None)
    workers_ref = workers
    worker_by_id = {worker.worker_id: worker for worker in workers_ref}
    worker_states: dict[str, WorkerState] = {
        worker.worker_id: WorkerState() for worker in workers_ref
    }
    states_lock = threading.Lock()
    short_worker = next((worker for worker in workers_ref if worker.pool == "short"), workers_ref[0])
    tokenize_url = f"http://{short_worker.host}:{short_worker.port}/tokenize"

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

        def _default_worker(self) -> WorkerEndpoint:
            return short_worker

        def _pick_upstream(
            self, body: bytes | None
        ) -> tuple[WorkerEndpoint, str, dict[str, object] | None]:
            if self._path() not in INFERENCE_PATHS or not body:
                worker = self._default_worker()
                return worker, f"{worker.worker_id}(default)", None

            try:
                payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                worker = self._default_worker()
                return worker, f"{worker.worker_id}(parse-failed)", None

            if not isinstance(payload, dict):
                worker = self._default_worker()
                return worker, f"{worker.worker_id}(invalid-json)", None

            t0 = time.perf_counter()
            with states_lock:
                worker, pool, prompt_tokens, method, prefix_hash = select_slo_worker(
                    payload,
                    workers_ref,
                    worker_states,
                    split_prompt_tokens=split_ref,
                    tokenize_url=tokenize_url if state_ref.ready else None,
                    model_name=model_ref,
                    prefix_locality=prefix_locality,
                )
                worker_state = worker_states[worker.worker_id]
                prefix_hit_candidate = bool(
                    prefix_hash and prefix_hash in worker_state.recent_prefix_hashes
                )
                track_request_start(worker_state, prompt_tokens, prefix_hash)
            route_ms = (time.perf_counter() - t0) * 1000
            label = f"{worker.worker_id}/{pool}({prompt_tokens} tok,{method})"
            route_info: dict[str, object] = {
                "event": "route",
                "path": self._path(),
                "pool": pool,
                "worker": worker.worker_id,
                "prompt_tokens": prompt_tokens,
                "method": method,
                "prefix_hash": prefix_hash or None,
                "prefix_hit_candidate": prefix_hit_candidate,
                "request_bytes": len(body),
                "route_overhead_ms": round(route_ms, 3),
                "queue_prefill_tokens": worker_states[worker.worker_id].queue_prefill_tokens,
                "inflight": worker_states[worker.worker_id].inflight,
            }
            return worker, label, route_info

        def _proxy(
            self,
            body: bytes | None,
            worker: WorkerEndpoint,
            route_info: dict[str, object] | None,
        ) -> int:
            connection = http.client.HTTPConnection(
                worker.host,
                worker.port,
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
                if route_info is not None:
                    prompt_tokens = int(route_info["prompt_tokens"])
                    worker_id = str(route_info["worker"])
                    with states_lock:
                        track_request_end(worker_states[worker_id], prompt_tokens)
            return 502

        def _handle(self) -> None:
            if self._gate_request():
                self._send_not_ready()
                return

            if self.command in {"GET", "HEAD"} and self._path() in GATED_PATHS:
                healthy = all(
                    self._backend_healthy(worker.host, worker.port)
                    for worker in workers_ref
                )
                if healthy:
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
            worker, route_label, route_info = self._pick_upstream(body)
            if route_info is not None:
                log_fn(json.dumps(route_info))
                log_fn(f"Route {self._path()} -> {route_label}")
            status = self._proxy(body, worker, route_info)
            if route_info is not None:
                log_fn(
                    json.dumps(
                        {
                            "event": "route_done",
                            "path": route_info["path"],
                            "pool": route_info["pool"],
                            "worker": route_info["worker"],
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
        name="slo-router",
        daemon=True,
    )
    thread.start()
    worker_summary = ", ".join(
        f"{worker.worker_id}@{worker.host}:{worker.port}" for worker in workers_ref
    )
    log_fn(
        f"SLO router on {public_host}:{public_port} "
        f"(split>={split_prompt_tokens} tok; workers: {worker_summary}; "
        f"prefix_locality={prefix_locality})"
    )
    server._router_thread = thread  # type: ignore[attr-defined]
    server._worker_by_id = worker_by_id  # type: ignore[attr-defined]
    return server


def stop_slo_router(server: socketserver.ThreadingTCPServer | None) -> None:
    if server is None:
        return
    server.shutdown()
    server.server_close()
