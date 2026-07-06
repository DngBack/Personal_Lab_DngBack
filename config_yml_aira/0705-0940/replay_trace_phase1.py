#!/usr/bin/env python3
"""Replay trace-round1.jsonl against an OpenAI-compatible vLLM server.

Sends each request at its trace timestamp offset with stream=True, measures
client-side TTFT (first content delta) and TBT (gaps between content deltas),
then prints aggregate stats in the same shape as the contest result files.

Usage:
  python3 replay_trace_phase1.py --base-url http://localhost:8000 \
      --trace ../phase1_info/trace-round1.jsonl --tag baseline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

TTFT_SLO_MS = 1500.0
TBT_SLO_MS = 45.0


def pctl(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))
    return xs[idx]


async def replay_one(
    client: httpx.AsyncClient,
    base_url: str,
    trace_req: dict,
    t0: float,
    results: list[dict],
) -> None:
    offset_s = trace_req["timestamp_ms"] / 1000.0
    delay = t0 + offset_s - time.perf_counter()
    if delay > 0:
        await asyncio.sleep(delay)

    body = dict(trace_req["body"])
    body["stream"] = True

    record: dict = {
        "request_id": trace_req["request_id"],
        "offset_ms": trace_req["timestamp_ms"],
        "error": None,
        "ttft_ms": None,
        "tbt_ms": [],
        "num_chunks": 0,
        "finish_reason": None,
        "e2e_ms": None,
    }
    t_send = time.perf_counter()
    try:
        async with client.stream(
            "POST", f"{base_url}/v1/chat/completions", json=body
        ) as resp:
            if resp.status_code != 200:
                text = await resp.aread()
                record["error"] = f"HTTP {resp.status_code}: {text[:200]!r}"
                results.append(record)
                return
            t_prev = None
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason"):
                    record["finish_reason"] = choice["finish_reason"]
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if not content:
                    continue
                now = time.perf_counter()
                if t_prev is None:
                    record["ttft_ms"] = (now - t_send) * 1000.0
                else:
                    record["tbt_ms"].append((now - t_prev) * 1000.0)
                t_prev = now
                record["num_chunks"] += 1
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["e2e_ms"] = (time.perf_counter() - t_send) * 1000.0
    results.append(record)


async def run(args: argparse.Namespace) -> dict:
    trace = [json.loads(l) for l in Path(args.trace).read_text().splitlines() if l]
    if args.limit:
        trace = trace[: args.limit]

    limits = httpx.Limits(max_connections=256, max_keepalive_connections=256)
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=600.0)
    results: list[dict] = []
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        # readiness ping (not counted)
        ping = {
            "model": trace[0]["body"]["model"],
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }
        r = await client.post(f"{args.base_url}/v1/chat/completions", json=ping)
        r.raise_for_status()

        t0 = time.perf_counter() + 0.25
        tasks = [
            asyncio.create_task(replay_one(client, args.base_url, tr, t0, results))
            for tr in trace
        ]
        await asyncio.gather(*tasks)
        wall_s = time.perf_counter() - t0

    ok = [r for r in results if r["error"] is None and r["ttft_ms"] is not None]
    errs = [r for r in results if r["error"] is not None]
    ttfts = [r["ttft_ms"] for r in ok]
    tbt_means = [statistics.fmean(r["tbt_ms"]) for r in ok if r["tbt_ms"]]
    tbt_pooled = [g for r in ok for g in r["tbt_ms"]]
    passed = [
        r
        for r in ok
        if r["ttft_ms"] <= TTFT_SLO_MS
        and (not r["tbt_ms"] or statistics.fmean(r["tbt_ms"]) <= TBT_SLO_MS)
    ]
    summary = {
        "tag": args.tag,
        "total_count": len(results),
        "failed_count": len(errs),
        "passed_slo": len(passed),
        "erc": round(len(passed) / max(1, len(results)), 4),
        "ttft_p50_ms": round(pctl(ttfts, 0.50)),
        "ttft_p95_ms": round(pctl(ttfts, 0.95)),
        "ttft_max_ms": round(max(ttfts)) if ttfts else None,
        "tbt_median_ms": round(pctl(tbt_pooled, 0.50), 1),
        "tbt_p95_ms": round(pctl(tbt_pooled, 0.95), 1),
        "tbt_req_mean_p95_ms": round(pctl(tbt_means, 0.95), 1),
        "wall_time_s": round(wall_s, 1),
        "slo": {"ttft_ms": TTFT_SLO_MS, "tbt_mean_ms": TBT_SLO_MS},
    }
    if errs:
        summary["sample_errors"] = [e["error"] for e in errs[:3]]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"phase1_replay_metrics.{args.tag}.jsonl"
    with metrics_path.open("w") as f:
        for r in sorted(results, key=lambda x: x["request_id"]):
            f.write(json.dumps(r) + "\n")
    summary_path = out_dir / f"phase1_replay_summary.{args.tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--trace",
        default=str(Path(__file__).resolve().parent.parent / "phase1_info" / "trace-round1.jsonl"),
    )
    parser.add_argument("--tag", default="local")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    summary = asyncio.run(run(args))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
