from __future__ import annotations

import random
from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .domain.scenario import ScenarioSpec


def _exp_gaps(n: int, rng: random.Random, scale_ms: float) -> list[int]:
    ts = [0]
    for _ in range(1, n):
        gap = max(1, int(rng.expovariate(1.0 / scale_ms)))
        ts.append(ts[-1] + gap)
    return ts


def _microburst(
    n: int, rng: random.Random, wave_min: int, wave_max: int, gap_range: tuple[int, int]
) -> list[int]:
    ts: list[int] = [0]
    i = 1
    while i < n:
        wave = rng.randint(wave_min, wave_max)
        gap = rng.randint(*gap_range)
        t = ts[-1] + gap
        for _ in range(min(wave, n - i)):
            ts.append(t)
            i += 1
    return ts


def _official_window(n: int, rng: random.Random, mean_ms: float, window_jitter: int) -> list[int]:
    offset = rng.randint(0, max(window_jitter, 1))
    long_n = n + offset + rng.randint(0, max(1, int(mean_ms * 10)))
    long_ts = _exp_gaps(long_n, rng, mean_ms)
    return long_ts[offset : offset + n]


def _lc_spread_poisson(
    n: int, lc_positions: set[int], rng: random.Random, mean_ms: float
) -> list[int]:
    ts = _exp_gaps(n, rng, mean_ms)
    for i in sorted(lc_positions):
        if i > 0 and ts[i] <= ts[i - 1]:
            ts[i] = ts[i - 1] + 1
        if i < n - 1 and ts[i] >= ts[i + 1]:
            ts[i + 1] = ts[i] + max(1, int(rng.expovariate(1.0 / mean_ms)))
    return ts


def build_arrival_timestamps_for_spec(
    n: int,
    spec: ScenarioSpec,
    rng: random.Random,
    *,
    lc_indices: set[int] | None = None,
) -> list[int]:
    p = spec.arrival_params
    kind = spec.arrival

    if kind == "steady_poisson":
        return _exp_gaps(n, rng, float(p.get("mean_ms", 50.0)))
    if kind == "official_window":
        return _official_window(
            n, rng, float(p.get("mean_ms", 50.0)), int(p.get("window_jitter", 2000))
        )
    if kind == "microburst":
        g = p.get("gap_ms", (80, 400))
        return _microburst(
            n, rng, int(p.get("wave_min", 6)), int(p.get("wave_max", 14)), (g[0], g[1])
        )
    if kind == "large_burst":
        g = p.get("gap_ms", (300, 1500))
        return _microburst(
            n, rng, int(p.get("batch_min", 10)), int(p.get("batch_max", 25)), (g[0], g[1])
        )
    if kind == "flood":
        return [0] * n
    if kind == "lc_spread_poisson":
        return _lc_spread_poisson(
            n, lc_indices or set(), rng, float(p.get("mean_ms", 50.0))
        )
    if kind == "session_cluster":
        return _exp_gaps(n, rng, 50.0)
    raise ValueError(f"Unknown arrival kind: {kind}")


def build_session_clustered_timestamps(
    order: list[str],
    session_by_rid: dict[str, str | None],
    rng: random.Random,
    intra_gap: tuple[int, int],
    inter_session: tuple[int, int],
) -> dict[str, int]:
    ts_map: dict[str, int] = {}
    t = 0
    prev_sid: str | None = None

    for rid in order:
        sid = session_by_rid.get(rid)
        if prev_sid is not None:
            if sid is not None and sid == prev_sid:
                t += rng.randint(*intra_gap)
            elif sid is not None and sid != prev_sid:
                t += rng.randint(*inter_session)
            else:
                t += rng.randint(20, 80)
        ts_map[rid] = t
        prev_sid = sid

    return ts_map


def analyze_timestamps(timestamps: list[int]) -> dict[str, Any]:
    if not timestamps:
        return {}
    gaps = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    bucket = Counter(timestamps)
    max_concurrent_start = max(bucket.values()) if bucket else 0
    n_unique = len(bucket)
    return {
        "n_requests": len(timestamps),
        "trace_span_ms": timestamps[-1] - timestamps[0],
        "gap_ms": {
            "min": min(gaps) if gaps else 0,
            "mean": round(sum(gaps) / len(gaps), 2) if gaps else 0,
            "p50": _pct(gaps, 50),
            "p95": _pct(gaps, 95),
        },
        "concurrent_starts": {
            "max_at_one_timestamp": max_concurrent_start,
            "n_unique_timestamps": n_unique,
            "pct_batched": round(
                100.0 * (len(timestamps) - n_unique) / max(len(timestamps), 1),
                2,
            ),
        },
    }


def _pct(vals: list[int | float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))
