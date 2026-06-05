from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from ..config import PHASES, PhaseSpec


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    phase: str
    total_requests: int
    mix: dict[str, float]
    arrival: str
    arrival_params: dict[str, Any] = field(default_factory=dict)
    output_bias: str = "default"
    cache_mode: str | None = None
    warmup_ratio: float = 0.10
    probe_slot_ratio: float = 0.08
    length_profile: str = "realistic"
    slo_ttft_ms: int | None = None
    slo_tbt_ms: int | None = None
    seed: int = 42
    description: str = ""

    def phase_spec(self) -> PhaseSpec:
        base = PHASES[self.phase]
        ttft = self.slo_ttft_ms if self.slo_ttft_ms is not None else base.slo_ttft_ms
        tbt = self.slo_tbt_ms if self.slo_tbt_ms is not None else base.slo_tbt_ms
        return PhaseSpec(
            name=base.name,
            slo_ttft_ms=ttft,
            slo_tbt_ms=tbt,
            request_timeout_s=base.request_timeout_s,
        )

    def workload_counts(self) -> dict[str, int]:
        keys = [k for k in ("conversation", "tool_agent", "long_context") if self.mix.get(k, 0) > 0]
        counts = {k: int(round(self.total_requests * self.mix[k])) for k in keys}
        delta = self.total_requests - sum(counts.values())
        order_keys = sorted(keys, key=lambda k: self.mix[k], reverse=True)
        i = 0
        while delta != 0 and order_keys:
            k = order_keys[i % len(order_keys)]
            counts[k] += 1 if delta > 0 else -1
            delta += -1 if delta > 0 else 1
            i += 1
        return counts


def load_scenario_yaml(path: Path) -> ScenarioSpec:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid scenario YAML: {path}")
    return ScenarioSpec(
        name=str(data["name"]),
        phase=str(data.get("phase", "phase2")),
        total_requests=int(data["total_requests"]),
        mix={k: float(v) for k, v in data["mix"].items()},
        arrival=str(data.get("arrival", "steady_poisson")),
        arrival_params=dict(data.get("arrival_params") or {}),
        output_bias=str(data.get("output_bias", "default")),
        cache_mode=data.get("cache_mode"),
        warmup_ratio=float(data.get("warmup_ratio", 0.10)),
        probe_slot_ratio=float(data.get("probe_slot_ratio", 0.08)),
        length_profile=str(data.get("length_profile", "realistic")),
        slo_ttft_ms=data.get("slo_ttft_ms"),
        slo_tbt_ms=data.get("slo_tbt_ms"),
        seed=int(data.get("seed", 42)),
        description=str(data.get("description", "")),
    )


def apply_overrides(
    spec: ScenarioSpec,
    *,
    requests: int | None = None,
    mean_ms: float | None = None,
    mix: dict[str, float] | None = None,
    seed: int | None = None,
) -> ScenarioSpec:
    arrival_params = dict(spec.arrival_params)
    if mean_ms is not None:
        arrival_params["mean_ms"] = mean_ms
    return replace(
        spec,
        total_requests=requests if requests is not None else spec.total_requests,
        mix=mix if mix is not None else spec.mix,
        arrival_params=arrival_params,
        seed=seed if seed is not None else spec.seed,
    )
