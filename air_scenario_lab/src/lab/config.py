from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[2]
AIR_DATA_ROOT = LAB_ROOT.parent / "air_data"
DEFAULT_HF_ROOT = AIR_DATA_ROOT / "data" / "hf"
DEFAULT_OUTPUT = LAB_ROOT / "output"

CONTEST_REF = "LLM Inference Optimization Challenge V2 — Section 6–7"

# LEval subsets for conversation (instructions-only — short real user turns)
CONVERSATION_LEVAL_SOURCES: tuple[tuple[str, str], ...] = (
    ("Exam", "gsm100"),
    ("Exam", "quality"),
    ("Exam", "tpo"),
    ("Generation", "natural_question"),
    ("Generation", "financial_qa"),
    ("Generation", "scientific_qa"),
)


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    slo_ttft_ms: int
    slo_tbt_ms: int
    request_timeout_s: int = 300


PHASE1 = PhaseSpec(name="phase1", slo_ttft_ms=4000, slo_tbt_ms=80, request_timeout_s=120)
PHASE2 = PhaseSpec(name="phase2", slo_ttft_ms=10000, slo_tbt_ms=200, request_timeout_s=300)
PHASES = {PHASE1.name: PHASE1, PHASE2.name: PHASE2}

LENGTH_PROFILES = {
    "conversation": {"input_median": 320, "input_p95": 1200, "output_median": 180, "output_p95": 850},
    "tool_agent": {"input_median": 8600, "input_p95": 18400, "output_median": 60, "output_p95": 220},
    "long_context": {"input_median": 15000, "input_p95": 78000, "output_median": 450, "output_p95": 1800},
}

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

DEFAULT_MAX_CONTEXT_TOKENS = 32768
DEFAULT_CONTEXT_SAFETY_TOKENS = 256
