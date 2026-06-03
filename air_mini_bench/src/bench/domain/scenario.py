"""Scenario catalog: workload mix, arrival pattern, and cache behavior per suite."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScenarioSpec:
    """
    Declarative definition of one benchmark suite.

    Attributes:
        name: Directory name under ``output/<phase>/``.
        phase: ``phase1`` or ``phase2``.
        mix: Fractions for conversation / tool_agent / long_context (sum = 1).
        arrival: Arrival algorithm key (see ``bench.arrivals``).
        cache_mode: ``hot`` | ``cold`` | None for tool prefix reuse experiments.
    """

    name: str
    phase: str
    mix: dict[str, float]
    arrival: str
    arrival_params: dict[str, Any] = field(default_factory=dict)
    output_bias: str = "default"
    cache_mode: str | None = None
    description: str = ""
    priority: int = 1


PHASE2_PRIORITY = (
    ScenarioSpec(
        name="official_like",
        phase="phase2",
        mix={"conversation": 0.31, "tool_agent": 0.59, "long_context": 0.10},
        arrival="official_window",
        arrival_params={"mean_ms": 50.0, "window_jitter": 2000},
        description="Gần đề P2: mix chuẩn, cửa sổ timestamp liên tục từ Poisson dài.",
        priority=1,
    ),
    ScenarioSpec(
        name="steady_poisson",
        phase="phase2",
        mix={"conversation": 0.31, "tool_agent": 0.59, "long_context": 0.10},
        arrival="steady_poisson",
        arrival_params={"mean_ms": 50.0},
        description="Baseline exponential mean 50ms.",
        priority=1,
    ),
    ScenarioSpec(
        name="microburst",
        phase="phase2",
        mix={"conversation": 0.31, "tool_agent": 0.59, "long_context": 0.10},
        arrival="microburst",
        arrival_params={"wave_min": 6, "wave_max": 14, "gap_ms": (80, 400)},
        description="Sóng 6–14 request cùng timestamp.",
        priority=1,
    ),
    ScenarioSpec(
        name="tool_cache_hot",
        phase="phase2",
        mix={"conversation": 0.10, "tool_agent": 0.85, "long_context": 0.05},
        arrival="session_cluster",
        arrival_params={"intra_gap_ms": (5, 40), "inter_session_ms": (400, 1200)},
        cache_mode="hot",
        description="Tool 85%, prefix text thật dùng chung theo session.",
        priority=1,
    ),
    ScenarioSpec(
        name="decode_pressure",
        phase="phase2",
        mix={"conversation": 0.80, "tool_agent": 0.20, "long_context": 0.0},
        arrival="steady_poisson",
        arrival_params={"mean_ms": 35.0},
        output_bias="long_decode",
        description="Conv-heavy, output dài, arrival nhanh — stress decode/TBT.",
        priority=1,
    ),
    ScenarioSpec(
        name="long_context_pressure",
        phase="phase2",
        mix={"conversation": 0.10, "tool_agent": 0.40, "long_context": 0.50},
        arrival="lc_spread_poisson",
        arrival_params={"mean_ms": 50.0},
        description="50% long-context, Poisson, không gom LC cùng timestamp.",
        priority=1,
    ),
)

PHASE2_EXTENDED = (
    ScenarioSpec(
        name="tool_cache_cold",
        phase="phase2",
        mix={"conversation": 0.10, "tool_agent": 0.85, "long_context": 0.05},
        arrival="session_cluster",
        arrival_params={"intra_gap_ms": (5, 40), "inter_session_ms": (400, 1200)},
        cache_mode="cold",
        description="Tool 85% nhưng mỗi request prefix khác — so sánh cache hit.",
        priority=2,
    ),
    ScenarioSpec(
        name="fast_queue",
        phase="phase2",
        mix={"conversation": 0.31, "tool_agent": 0.59, "long_context": 0.10},
        arrival="steady_poisson",
        arrival_params={"mean_ms": 16.0},
        description="Queue delay: exponential mean ~16ms.",
        priority=2,
    ),
    ScenarioSpec(
        name="large_burst",
        phase="phase2",
        mix={"conversation": 0.20, "tool_agent": 0.70, "long_context": 0.10},
        arrival="large_burst",
        arrival_params={"batch_min": 10, "batch_max": 25, "gap_ms": (300, 1500)},
        description="Tool 70%, burst 10–25 — stress prefill scheduler.",
        priority=2,
    ),
    ScenarioSpec(
        name="flood_admission",
        phase="phase2",
        mix={"conversation": 0.40, "tool_agent": 0.55, "long_context": 0.05},
        arrival="flood",
        arrival_params={},
        description="Admission/OOM: toàn bộ request t=0, ít LC.",
        priority=2,
    ),
)

PHASE1_SUITES = (
    ScenarioSpec(
        name="p1_official_like",
        phase="phase1",
        mix={"conversation": 0.60, "tool_agent": 0.40, "long_context": 0.0},
        arrival="official_window",
        arrival_params={"mean_ms": 50.0, "window_jitter": 1500},
        description="Gần P1: 60/40, cửa sổ timestamp liên tục.",
        priority=1,
    ),
    ScenarioSpec(
        name="p1_steady",
        phase="phase1",
        mix={"conversation": 0.60, "tool_agent": 0.40, "long_context": 0.0},
        arrival="steady_poisson",
        arrival_params={"mean_ms": 50.0},
        description="P1 baseline Poisson 50ms.",
        priority=1,
    ),
    ScenarioSpec(
        name="p1_burst",
        phase="phase1",
        mix={"conversation": 0.50, "tool_agent": 0.50, "long_context": 0.0},
        arrival="large_burst",
        arrival_params={"batch_min": 10, "batch_max": 25, "gap_ms": (300, 1500)},
        description="Burst 10–25 cùng timestamp.",
        priority=1,
    ),
    ScenarioSpec(
        name="p1_tool_cache_hot",
        phase="phase1",
        mix={"conversation": 0.20, "tool_agent": 0.80, "long_context": 0.0},
        arrival="session_cluster",
        arrival_params={"intra_gap_ms": (5, 40), "inter_session_ms": (400, 1200)},
        cache_mode="hot",
        description="P1 prefix cache hot — shared tool prefix thật.",
        priority=1,
    ),
    ScenarioSpec(
        name="p1_tool_cache_cold",
        phase="phase1",
        mix={"conversation": 0.20, "tool_agent": 0.80, "long_context": 0.0},
        arrival="session_cluster",
        arrival_params={"intra_gap_ms": (5, 40), "inter_session_ms": (400, 1200)},
        cache_mode="cold",
        description="P1 prefix cache cold — prefix không reuse.",
        priority=1,
    ),
    ScenarioSpec(
        name="p1_decode_pressure",
        phase="phase1",
        mix={"conversation": 0.85, "tool_agent": 0.15, "long_context": 0.0},
        arrival="steady_poisson",
        arrival_params={"mean_ms": 30.0},
        output_bias="long_decode",
        description="P1 decode/TBT: conv 85%, output dài.",
        priority=1,
    ),
)

PHASE2_SUITES = PHASE2_PRIORITY + PHASE2_EXTENDED

SUITES_BY_PHASE: dict[str, tuple[ScenarioSpec, ...]] = {
    "phase1": PHASE1_SUITES,
    "phase2": PHASE2_SUITES,
}

PRIORITY_SUITES_BY_PHASE: dict[str, tuple[ScenarioSpec, ...]] = {
    "phase1": tuple(s for s in PHASE1_SUITES if s.priority == 1),
    "phase2": tuple(s for s in PHASE2_SUITES if s.priority == 1),
}


def get_suite(phase: str, name: str) -> ScenarioSpec:
    """Resolve a suite by phase and name; raises ``KeyError`` if unknown."""
    for spec in SUITES_BY_PHASE[phase]:
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown suite {name} for {phase}")


def list_suite_names(phase: str, *, priority_only: bool = False) -> list[str]:
    """List suite directory names for a phase."""
    suites = PRIORITY_SUITES_BY_PHASE[phase] if priority_only else SUITES_BY_PHASE[phase]
    return [s.name for s in suites]
