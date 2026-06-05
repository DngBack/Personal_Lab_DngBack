from __future__ import annotations

import random
from typing import Any

from .config import CONTEST_REF
from .domain.scenario import ScenarioSpec


def pick_probe_slots(
    rows: list[dict[str, Any]], ratio: float, rng: random.Random
) -> set[str]:
    scored = [r for r in rows if not r["is_warmup"]]
    n_slots = max(1, int(round(len(scored) * ratio)))
    eligible = [
        r["request_id"]
        for r in scored
        if r["workload_type"] in ("tool_agent", "long_context")
    ]
    rng.shuffle(eligible)
    return set(eligible[: min(n_slots, len(eligible))])


def probe_distribution_spec(spec: ScenarioSpec, probe_count: int) -> dict[str, Any]:
    if spec.phase == "phase1":
        by_wl = {"tool_agent": probe_count, "long_context": 0}
    else:
        ta = max(1, int(probe_count * 0.75))
        lc = probe_count - ta
        by_wl = {"tool_agent": ta, "long_context": lc}
    return {
        "phase": spec.phase,
        "probe_slot_ratio": spec.probe_slot_ratio,
        "total_probe_slots": probe_count,
        "by_workload": by_wl,
        "note": CONTEST_REF,
    }
