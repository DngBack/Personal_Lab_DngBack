ARG BASE_IMAGE=vllm/vllm-openai:latest
FROM ${BASE_IMAGE}

WORKDIR /app

RUN python3 -m pip install --no-cache-dir pyyaml

COPY scripts/readiness_gate.py /app/scripts/readiness_gate.py
COPY scripts/routing_utils.py /app/scripts/routing_utils.py
COPY scripts/length_router.py /app/scripts/length_router.py
COPY scripts/slo_router.py /app/scripts/slo_router.py
COPY scripts/run_baseline.py /app/scripts/run_baseline.py
COPY scripts/patch_vllm_concurrent_prefill.py /app/scripts/patch_vllm_concurrent_prefill.py
COPY scripts/scheduler_concurrent_prefill_v023.py /app/scripts/scheduler_concurrent_prefill_v023.py
COPY scripts/probe_attn_backend.py /app/scripts/probe_attn_backend.py

COPY configs/config_dp3_short_dp1_long.yaml /app/configs/config_dp3_short_dp1_long.yaml
COPY configs/config_tp2_dp2_short_long.yaml /app/configs/config_tp2_dp2_short_long.yaml
COPY configs/config_tp2_dp2_native.yaml /app/configs/config_tp2_dp2_native.yaml
COPY configs/config_tp2_triton_attn.yaml /app/configs/config_tp2_triton_attn.yaml
COPY configs/config_5.yaml /app/configs/config_5.yaml
COPY configs/config_v1b_2short_2long.yaml /app/configs/config_v1b_2short_2long.yaml
# Default: 3×DP short (GPU 0-2) + 1×DP long (GPU 3).
COPY configs/config_v1b_2short_2long.yaml /app/configs/baseline.yaml

ENV BASELINE_CONFIG_PATH=/app/configs/baseline.yaml
ENV MODEL_PATH=/models/gpt-oss-20b
# Optional A/B: BASELINE_CONFIG_PROFILE=dp3_short|v1b|jit|triton_attn|native|tp4

# Bake scheduler patch at build time; runtime patch remains as a fallback.
RUN python3 /app/scripts/patch_vllm_concurrent_prefill.py \
    && python3 /app/scripts/patch_vllm_concurrent_prefill.py --check

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=1800s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

ENTRYPOINT ["python3", "/app/scripts/run_baseline.py"]
