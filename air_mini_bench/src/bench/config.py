from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[3]
AIR_DATA_ROOT = LAB_ROOT / "air_data"
DEFAULT_HF_ROOT = AIR_DATA_ROOT / "data" / "hf"
BENCH_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = BENCH_ROOT / "output"

CONTEST_REF = "LLM Inference Optimization Challenge V2 — Section 6–7"


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    total_requests: int
    conversation: int
    tool_agent: int
    long_context: int
    warmup_ratio: float = 0.05  # 5% — đủ cho test nhỏ; đề gốc ~10%
    probe_slot_ratio: float = 0.08
    # Contest SLO hints (ms) — calibrate on your stack
    slo_ttft_ms: int = 4000
    slo_tbt_ms: int = 80
    request_timeout_s: int = 60


PHASE1 = PhaseSpec(
    name="phase1",
    total_requests=250,
    conversation=150,
    tool_agent=100,
    long_context=0,
    slo_ttft_ms=4000,
    slo_tbt_ms=80,
    request_timeout_s=120,
)

PHASE2 = PhaseSpec(
    name="phase2",
    total_requests=500,
    conversation=155,
    tool_agent=295,
    long_context=50,
    slo_ttft_ms=10000,
    slo_tbt_ms=200,
    request_timeout_s=300,
)

PHASES = {PHASE1.name: PHASE1, PHASE2.name: PHASE2}

# Length targets (tokens) from contest Section 7 — median / p95
LENGTH_PROFILES = {
    "conversation": {"input_median": 320, "input_p95": 1200, "output_median": 180, "output_p95": 850},
    "tool_agent": {"input_median": 8600, "input_p95": 18400, "output_median": 60, "output_p95": 220},
    "long_context": {"input_median": 15000, "input_p95": 78000, "output_median": 450, "output_p95": 1800},
}

# Figure 5 / Section 7: median + tail thực tế (long-context ~15k median, mẫu ~24–25k).
# Dùng với vLLM --max-model-len 32768 (xem scripts/start_vllm.sh).
REALISTIC_INPUT_CAP = {
    "conversation": 1200,
    "tool_agent": 18400,
    "long_context": 25000,
}
REALISTIC_OUTPUT_CAP = {
    "conversation": 850,
    "tool_agent": 220,
    "long_context": 1800,
}

# Đẩy input lên gần cap / tail contest — dùng khi muốn stress VRAM & prefill
HEAVY_INPUT_CAP = REALISTIC_INPUT_CAP
HEAVY_OUTPUT_CAP = REALISTIC_OUTPUT_CAP

# Chỉ cho model/context nhỏ (8k) — smoke test nhanh.
COMPACT_INPUT_CAP = {
    "conversation": 1200,
    "tool_agent": 5500,
    "long_context": 6500,
}
COMPACT_OUTPUT_CAP = {
    "conversation": 850,
    "tool_agent": 256,
    "long_context": 512,
}

# Backward-compatible alias
LOCAL_INPUT_CAP = COMPACT_INPUT_CAP
LOCAL_OUTPUT_CAP = COMPACT_OUTPUT_CAP

# 25k input + ~1.8k output + safety → 28672+ ; làm tròn 32k cho vLLM
DEFAULT_MAX_CONTEXT_TOKENS = 32768
DEFAULT_CONTEXT_SAFETY_TOKENS = 256

# Khuyến nghị khớp DEFAULT_MAX_CONTEXT_TOKENS khi chạy vLLM local
VLLM_RECOMMENDED_MAX_MODEL_LEN = 32768
