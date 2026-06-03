"""Benchmark run results and contest-style score aggregation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..config import PhaseSpec
from ..metrics import (
    aggregate_probe_scores,
    compute_score,
    latency_summary,
)


@dataclass
class RequestResult:
    """Per-request metrics collected during replay."""

    request_id: str
    workload_type: str
    is_warmup: bool
    is_probe_slot: bool
    input_length: int = 0
    scheduled_timestamp_ms: int = 0
    ttft_ms: float | None = None
    tbt_ms: float | None = None
    output_tokens: int = 0
    latency_ms: float | None = None
    effective: bool = False
    error: str | None = None
    completion: str = ""


@dataclass
class BenchReport:
    """Aggregated outcome of replaying one suite against an inference endpoint."""

    phase: str
    scenario: str = "contest"
    results: list[RequestResult] = field(default_factory=list)
    probe_scores: list[dict[str, float]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    wall_time_s: float = 0.0

    def to_dict(self, spec: PhaseSpec, baseline_mean: float = 0.831) -> dict[str, Any]:
        """Serialize to contest-style JSON (ERC, latency, accuracy gate, score)."""
        scored = [r for r in self.results if not r.is_warmup]
        effective = [r for r in scored if r.effective]
        erc = len(effective) / max(len(scored), 1)
        sub_mean = aggregate_probe_scores(self.probe_scores)
        accuracy_drop = max(0.0, baseline_mean - sub_mean)
        score = compute_score(erc, accuracy_drop)

        by_wl: dict[str, list[RequestResult]] = defaultdict(list)
        for r in scored:
            by_wl[r.workload_type].append(r)

        def _erc_wl(rs: list[RequestResult]) -> dict[str, Any]:
            eff = [x for x in rs if x.effective]
            return {
                "n": len(rs),
                "n_effective": len(eff),
                "erc": round(len(eff) / max(len(rs), 1), 4),
            }

        ttfts = [r.ttft_ms for r in scored if r.ttft_ms is not None]
        tbts = [r.tbt_ms for r in scored if r.tbt_ms is not None]

        return {
            "phase": self.phase,
            "scenario": self.scenario,
            "wall_time_s": round(self.wall_time_s, 2),
            "slo_ttft_ms": spec.slo_ttft_ms,
            "slo_tbt_ms": spec.slo_tbt_ms,
            "erc": {
                "n_total": len(scored),
                "n_effective": len(effective),
                "erc": round(erc, 4),
                "erc_percent": round(erc * 100, 2),
                "by_workload": {wl: _erc_wl(rs) for wl, rs in sorted(by_wl.items())},
            },
            "latency": {
                "ttft_ms": latency_summary(ttfts),
                "tbt_ms": latency_summary(tbts),
                "slo_fail_ttft": sum(
                    1 for r in scored if r.ttft_ms is not None and r.ttft_ms > spec.slo_ttft_ms
                ),
                "slo_fail_tbt": sum(
                    1 for r in scored if r.tbt_ms is not None and r.tbt_ms > spec.slo_tbt_ms
                ),
            },
            "accuracy": {
                "accuracy_drop": round(accuracy_drop, 4),
                "baseline_mean": baseline_mean,
                "submission_mean": round(sub_mean, 4),
                "n_probes": len(self.probe_scores),
                "pass_gate": accuracy_drop < 0.02,
            },
            "score": score,
            "errors": self.errors,
        }
