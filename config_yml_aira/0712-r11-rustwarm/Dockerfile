# r11-rustwarm-v1: FROM r11-engpace-rust-v1, only sitecustomize.py changes.
#
# Fixes AIR_WARMUP / AIR_PRIME_PACK / AIR_GC_FREEZE being silently inert under
# VLLM_USE_RUST_FRONTEND=1. Root cause: those stages were only ever armed by
# hooking vllm.entrypoints.launcher.serve_http, which the Rust-frontend
# dispatch path (run_multi_api_server in vllm/entrypoints/cli/serve.py) never
# imports. This build adds a second, symmetric hook on run_multi_api_server
# itself. See sitecustomize.py:_patch_cli_serve for the full writeup.
#
# Known gap NOT fixed here: AIR_HEALTH_GATE (/health 503-until-warm) needs a
# Python ASGI app to wrap; under the Rust frontend, vllm-rs owns the socket
# and answers /health itself, so warmup now runs but isn't reflected in
# /health. Logged clearly at boot when this applies.
FROM dngback/vllm-v0.24.0-ubuntu2404-opt-2:r11-engpace-rust-v1

COPY sitecustomize.py /etc/python3.12/sitecustomize.py
COPY sitecustomize.py /usr/lib/python3.12/sitecustomize.py
COPY sitecustomize.py /usr/local/lib/python3.12/dist-packages/sitecustomize.py
COPY sitecustomize.py /app/sitecustomize.py
