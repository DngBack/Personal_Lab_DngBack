#!/usr/bin/env python3
"""
Attention backend profiler for GptOss vLLM serving.

Đo TTFT/throughput theo context length và phát hiện Triton JIT compile spikes.

Cách dùng:
  # Probe server đang chạy trên port 8000
  python3 probe_attn_backend.py --url http://127.0.0.1:8000

  # A/B: probe cả hai backend (cần 2 server riêng biệt)
  python3 probe_attn_backend.py --url http://127.0.0.1:8001 --label flash_attn
  python3 probe_attn_backend.py --url http://127.0.0.1:8002 --label triton_attn
  python3 probe_attn_backend.py --compare flash_attn.json triton_attn.json

  # Chỉ test JIT warm-up coverage
  python3 probe_attn_backend.py --url http://127.0.0.1:8000 --mode jit_check
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────────────
# Request helpers
# ──────────────────────────────────────────────────────

def _make_request(
    server_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    stream: bool,
    timeout: int,
    reasoning_effort: str = "low",
) -> dict[str, Any]:
    """Send a single chat completion and return timing metrics."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": stream,
        "chat_template_kwargs": {"reasoning_effort": reasoning_effort},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{server_url.rstrip('/')}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    ttft: float | None = None
    output_tokens = 0
    status = 0

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            if stream:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        chunk = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    delta = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if delta and ttft is None:
                        ttft = time.monotonic() - t0
                    if delta:
                        output_tokens += 1
            else:
                body = json.loads(resp.read().decode())
                ttft = time.monotonic() - t0
                output_tokens = (
                    body.get("usage", {}).get("completion_tokens", 0)
                )
    except (OSError, urllib.error.URLError) as exc:
        return {
            "ok": False,
            "error": str(exc),
            "elapsed_ms": (time.monotonic() - t0) * 1000,
        }

    elapsed = time.monotonic() - t0
    return {
        "ok": status == 200,
        "status": status,
        "ttft_ms": (ttft or elapsed) * 1000,
        "elapsed_ms": elapsed * 1000,
        "output_tokens": output_tokens,
        "tok_per_sec": output_tokens / elapsed if elapsed > 0 else 0,
    }


def _get_model(server_url: str, timeout: int = 10) -> str:
    req = urllib.request.Request(
        f"{server_url.rstrip('/')}/v1/models",
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            models = data.get("data", [])
            if models:
                return str(models[0].get("id", "/models/gpt-oss-20b"))
    except Exception:
        pass
    return "/models/gpt-oss-20b"


def _make_prompt(target_chars: int, max_tokens: int) -> str:
    seed = (
        "Produce a steady answer after reading this prompt. "
        f"Write close to {max(1, max_tokens)} output tokens. "
        "Use the requested budget so the decode path is exercised. "
    )
    reps = (target_chars // len(seed)) + 1
    return (seed * reps)[:max(1, target_chars)]


# ──────────────────────────────────────────────────────
# Probe modes
# ──────────────────────────────────────────────────────

@dataclass
class ProbeResult:
    label: str
    mode: str
    server_url: str
    model: str
    measurements: list[dict[str, Any]] = field(default_factory=list)

    def ttft_ms_list(self) -> list[float]:
        return [m["ttft_ms"] for m in self.measurements if m.get("ok") and "ttft_ms" in m]

    def summary(self) -> dict[str, Any]:
        ttft = self.ttft_ms_list()
        if not ttft:
            return {"ok_count": 0, "error_count": len(self.measurements)}
        ttft_s = sorted(ttft)
        n = len(ttft_s)
        return {
            "label": self.label,
            "ok_count": n,
            "error_count": len(self.measurements) - n,
            "ttft_p50_ms": ttft_s[n // 2],
            "ttft_p90_ms": ttft_s[min(int(n * 0.9), n - 1)],
            "ttft_p99_ms": ttft_s[min(int(n * 0.99), n - 1)],
            "ttft_min_ms": ttft_s[0],
            "ttft_max_ms": ttft_s[-1],
            "ttft_mean_ms": statistics.mean(ttft_s),
            "ttft_stdev_ms": statistics.stdev(ttft_s) if n >= 2 else 0,
            "jit_spike_count": sum(
                1 for t in ttft_s if t > statistics.mean(ttft_s) * 3
            ),
        }


def probe_latency_sweep(
    server_url: str,
    model: str,
    label: str,
    prompt_lengths: list[int],
    max_tokens: int,
    concurrency: int,
    rounds: int,
    timeout: int,
    stream: bool,
) -> ProbeResult:
    """Test TTFT across different prompt token lengths."""
    result = ProbeResult(label=label, mode="latency_sweep", server_url=server_url, model=model)

    all_tasks: list[tuple[str, int, int]] = []
    for length in prompt_lengths:
        prompt = _make_prompt(length * 4, max_tokens)
        for _ in range(rounds):
            all_tasks.append((prompt, length, max_tokens))

    print(f"\n{'='*60}", flush=True)
    print(f"[{label}] Latency sweep: {len(prompt_lengths)} lengths × {rounds} rounds", flush=True)
    print(f"  url={server_url}  concurrency={concurrency}  stream={stream}", flush=True)

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {
            ex.submit(_make_request, server_url, model, prompt, mt, stream, timeout): length
            for prompt, length, mt in all_tasks
        }
        for fut, length in futs.items():
            r = fut.result()
            r["prompt_tokens_target"] = length
            result.measurements.append(r)
            completed += 1
            if completed % max(1, len(all_tasks) // 5) == 0:
                pct = 100 * completed // len(all_tasks)
                print(f"  [{label}] {pct}% done ({completed}/{len(all_tasks)})", flush=True)

    return result


def probe_jit_check(
    server_url: str,
    model: str,
    label: str,
    timeout: int,
    stream: bool,
) -> ProbeResult:
    """
    Probe JIT compilation coverage.

    Gửi requests đơn lẻ ở các sequence length khác nhau để phát hiện
    latency spike bất thường (dấu hiệu Triton JIT compilation in-flight).

    Pattern: gửi request 1 (JIT chưa có cache) → request 2 cùng shape
    (JIT đã cache). Nếu request 1 >> request 2, đó là JIT spike.
    """
    result = ProbeResult(label=label, mode="jit_check", server_url=server_url, model=model)

    # Shapes likely to trigger different Triton kernel compilations:
    # - Small decode: batch_size=1, context=32
    # - Medium prefill: batch_size=1, context=512
    # - Large prefill: batch_size=1, context=4096
    # - Mixed: concurrent requests of different lengths
    shapes = [
        (32, 16, "tiny_single"),
        (128, 32, "short_single"),
        (512, 32, "medium_single"),
        (2048, 64, "large_single"),
        (4256, 96, "bulk_single"),
        (8192, 128, "long_single"),
        # Repeat to detect JIT cache effect
        (32, 16, "tiny_repeat"),
        (512, 32, "medium_repeat"),
        (4256, 96, "bulk_repeat"),
    ]

    print(f"\n{'='*60}", flush=True)
    print(f"[{label}] JIT check: {len(shapes)} shapes (sequential)", flush=True)

    for prompt_chars, max_tokens, name in shapes:
        prompt = _make_prompt(prompt_chars * 4, max_tokens)
        r = _make_request(server_url, model, prompt, max_tokens, stream, timeout)
        r["shape_name"] = name
        r["prompt_chars"] = prompt_chars
        result.measurements.append(r)

        ttft = r.get("ttft_ms", 0)
        ok = "OK" if r.get("ok") else f"FAIL({r.get('status', '?')})"
        print(
            f"  {name:25s}  ttft={ttft:8.1f}ms  {ok}",
            flush=True,
        )

    # Detect JIT spikes: if first occurrence >> second occurrence of same shape
    ttft_by_prefix: dict[str, float] = {}
    spikes = []
    for r in result.measurements:
        base = r.get("shape_name", "").replace("_repeat", "").replace("_single", "")
        if base not in ttft_by_prefix:
            ttft_by_prefix[base] = r.get("ttft_ms", 0)
        else:
            first = ttft_by_prefix[base]
            second = r.get("ttft_ms", 0)
            if first > 0 and second > 0 and first > second * 2.5:
                spikes.append((base, first, second))

    if spikes:
        print(f"\n  ⚠  JIT spikes detected ({len(spikes)}):", flush=True)
        for name, first, second in spikes:
            print(
                f"    {name}: first={first:.0f}ms vs repeat={second:.0f}ms "
                f"(ratio={first/second:.1f}x)",
                flush=True,
            )
        print(
            "  → Extend warmup to cover these shapes before real traffic arrives.",
            flush=True,
        )
    else:
        print("\n  ✓ No significant JIT spikes detected.", flush=True)

    return result


def probe_decode_throughput(
    server_url: str,
    model: str,
    label: str,
    batch_sizes: list[int],
    prompt_tokens: int,
    max_tokens: int,
    timeout: int,
) -> ProbeResult:
    """
    Measure decode throughput (tok/s) at different batch sizes.
    Helps identify if bottleneck is compute (scales with batch) or
    memory bandwidth / KV cache (plateaus with batch).
    """
    result = ProbeResult(label=label, mode="decode_throughput", server_url=server_url, model=model)
    prompt = _make_prompt(prompt_tokens * 4, max_tokens)

    print(f"\n{'='*60}", flush=True)
    print(
        f"[{label}] Decode throughput sweep: "
        f"prompt_tokens≈{prompt_tokens}, max_tokens={max_tokens}",
        flush=True,
    )

    for batch in batch_sizes:
        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch) as ex:
            futs = [
                ex.submit(_make_request, server_url, model, prompt, max_tokens, False, timeout)
                for _ in range(batch)
            ]
            results = [f.result() for f in futs]
        wall = time.monotonic() - t0
        ok_results = [r for r in results if r.get("ok")]
        total_out = sum(r.get("output_tokens", 0) for r in ok_results)
        tps = total_out / wall if wall > 0 else 0
        avg_ttft = (
            statistics.mean(r["ttft_ms"] for r in ok_results) if ok_results else 0
        )
        print(
            f"  batch={batch:3d}  wall={wall*1000:6.0f}ms  "
            f"tok/s={tps:6.1f}  avg_ttft={avg_ttft:.0f}ms  "
            f"ok={len(ok_results)}/{batch}",
            flush=True,
        )
        for r in results:
            r["batch_size"] = batch
        result.measurements.extend(results)

    return result


# ──────────────────────────────────────────────────────
# Compare two result files
# ──────────────────────────────────────────────────────

def compare_results(path_a: str, path_b: str) -> None:
    with open(path_a) as f:
        data_a = json.load(f)
    with open(path_b) as f:
        data_b = json.load(f)

    label_a = data_a.get("label", path_a)
    label_b = data_b.get("label", path_b)

    # Per-prompt-length comparison
    def by_length(data: dict) -> dict[int, list[float]]:
        out: dict[int, list[float]] = {}
        for m in data.get("measurements", []):
            if not m.get("ok"):
                continue
            length = m.get("prompt_tokens_target", 0)
            if length:
                out.setdefault(length, []).append(m["ttft_ms"])
        return out

    rows_a = by_length(data_a)
    rows_b = by_length(data_b)
    all_lengths = sorted(set(rows_a) | set(rows_b))

    print(f"\n{'='*70}", flush=True)
    print(f"A/B Comparison: {label_a} vs {label_b}", flush=True)
    print(f"{'tokens':>10}  {'A p50':>10}  {'B p50':>10}  {'delta':>10}  {'winner':>8}", flush=True)
    print("-" * 55, flush=True)
    for length in all_lengths:
        a_vals = sorted(rows_a.get(length, []))
        b_vals = sorted(rows_b.get(length, []))
        a_p50 = a_vals[len(a_vals) // 2] if a_vals else float("nan")
        b_p50 = b_vals[len(b_vals) // 2] if b_vals else float("nan")
        if a_p50 == a_p50 and b_p50 == b_p50:  # nan check
            delta_pct = 100 * (b_p50 - a_p50) / a_p50
            winner = label_b if b_p50 < a_p50 else label_a
            print(
                f"{length:>10}  {a_p50:>10.1f}  {b_p50:>10.1f}  "
                f"{delta_pct:>+9.1f}%  {winner:>8}",
                flush=True,
            )
    print("=" * 70, flush=True)


# ──────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────

def _wait_health(url: str, timeout: int = 120) -> bool:
    deadline = time.monotonic() + timeout
    health = f"{url.rstrip('/')}/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="vLLM server URL")
    parser.add_argument("--label", default="server", help="Label for output file")
    parser.add_argument(
        "--mode",
        choices=["latency", "jit_check", "decode_throughput", "all"],
        default="all",
        help="Probe mode",
    )
    parser.add_argument("--compare", nargs=2, metavar=("FILE_A", "FILE_B"))
    parser.add_argument("--output", help="Save results to JSON file (e.g. flash_attn.json)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3, help="Rounds per scenario")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args(argv)

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return 0

    stream = not args.no_stream

    print(f"Probing {args.url} (label={args.label})", flush=True)
    print("Checking health...", flush=True)
    if not _wait_health(args.url):
        print(f"Server not healthy at {args.url}", file=sys.stderr)
        return 1
    print("  OK", flush=True)

    model = _get_model(args.url)
    print(f"  model={model}", flush=True)

    all_results: list[ProbeResult] = []
    mode = args.mode

    if mode in ("jit_check", "all"):
        r = probe_jit_check(args.url, model, args.label, args.timeout, stream)
        all_results.append(r)
        s = r.summary()
        print(f"\n  JIT check summary: {s}", flush=True)

    if mode in ("latency", "all"):
        # Prompt lengths covering all realistic traffic buckets
        lengths = [32, 64, 128, 256, 512, 1024, 2048, 4096, 4256, 5504, 8192, 16384, 24256]
        r = probe_latency_sweep(
            args.url, model, args.label,
            prompt_lengths=lengths,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            rounds=args.rounds,
            timeout=args.timeout,
            stream=stream,
        )
        all_results.append(r)
        s = r.summary()
        print(f"\n  Latency sweep summary:", flush=True)
        print(f"    p50 TTFT: {s.get('ttft_p50_ms', 0):.1f} ms", flush=True)
        print(f"    p90 TTFT: {s.get('ttft_p90_ms', 0):.1f} ms", flush=True)
        print(f"    p99 TTFT: {s.get('ttft_p99_ms', 0):.1f} ms", flush=True)
        print(f"    JIT spikes: {s.get('jit_spike_count', 0)}", flush=True)

    if mode in ("decode_throughput", "all"):
        r = probe_decode_throughput(
            args.url, model, args.label,
            batch_sizes=[1, 2, 4, 8, 16, 32],
            prompt_tokens=256,
            max_tokens=128,
            timeout=args.timeout,
        )
        all_results.append(r)

    if args.output:
        # Serialise all results to JSON
        combined = {
            "label": args.label,
            "url": args.url,
            "model": model,
            "measurements": [
                {**m, "mode": r.mode}
                for r in all_results
                for m in r.measurements
            ],
            "summaries": [r.summary() for r in all_results],
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
        print(f"\nResults saved to {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
