from __future__ import annotations

import re
from typing import Any


def exact_match(pred: str, ref: str) -> float:
    return float(_normalize(pred) == _normalize(ref))


def token_f1(pred: str, ref: str) -> float:
    p = _normalize(pred).split()
    r = _normalize(ref).split()
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    common: dict[str, int] = {}
    for t in p:
        common[t] = common.get(t, 0) + 1
    hits = 0
    for t in r:
        if common.get(t, 0) > 0:
            hits += 1
            common[t] -= 1
    if hits == 0:
        return 0.0
    prec = hits / len(p)
    rec = hits / len(r)
    return 2 * prec * rec / (prec + rec)


def _normalize(text: str) -> str:
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)


def median_tbt_ms(gaps_ms: list[float]) -> float | None:
    if len(gaps_ms) < 2:
        return None
    n = len(gaps_ms)
    trim = max(1, int(n * 0.05))
    middle = sorted(gaps_ms)[trim : n - trim]
    if not middle:
        middle = gaps_ms
    mid = len(middle) // 2
    if len(middle) % 2:
        return middle[mid]
    return (middle[mid - 1] + middle[mid]) / 2


def request_effective(
    ttft_ms: float | None,
    tbt_ms: float | None,
    output_tokens: int,
    slo_ttft_ms: float,
    slo_tbt_ms: float,
) -> bool:
    if output_tokens < 1:
        return False
    if ttft_ms is None or tbt_ms is None:
        return False
    return ttft_ms <= slo_ttft_ms and tbt_ms <= slo_tbt_ms


def f_accuracy_drop(drop: float) -> float:
    if drop >= 0.02:
        return 0.0
    if drop <= 0.005:
        return 1.0
    if drop <= 0.015:
        return 1.0 - (drop - 0.005) / 0.01 * 0.5
    return 0.5 * (0.02 - drop) / 0.005


def compute_score(erc: float, accuracy_drop: float) -> dict[str, float]:
    f_delta = f_accuracy_drop(accuracy_drop)
    return {
        "erc": erc,
        "accuracy_drop": accuracy_drop,
        "f_delta": f_delta,
        "score": round(100.0 * erc * f_delta, 2),
    }


def evaluate_probe(pred: str, ref: str, method: str) -> dict[str, float]:
    if method == "f1_em":
        return {"f1": token_f1(pred, ref), "em": exact_match(pred, ref)}
    if method in ("f1_rouge_l", "f1_rouge"):
        f1 = token_f1(pred, ref)
        return {"f1": f1, "rouge_l": f1}
    return {"f1": token_f1(pred, ref)}


def percentile(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def latency_summary(vals: list[float]) -> dict[str, float | int | None]:
    if not vals:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p95": None, "max": None}
    return {
        "count": len(vals),
        "min": round(min(vals), 2),
        "p50": round(percentile(vals, 50) or 0, 2),
        "p90": round(percentile(vals, 90) or 0, 2),
        "p95": round(percentile(vals, 95) or 0, 2),
        "max": round(max(vals), 2),
    }


def aggregate_probe_scores(scores: list[dict[str, float]]) -> float:
    if not scores:
        return 0.0
    vals = [s.get("f1", 0.0) for s in scores]
    return sum(vals) / len(vals)
