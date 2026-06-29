"""
Async HTTP benchmark runner.

Sends concurrent streaming chat-completions to a vLLM OpenAI-compatible
server and records per-request TTFT (time-to-first-token) and TBT
(median time-between-tokens).

Usage
-----
    import asyncio
    from dng_opt.bench.config import BenchConfig
    from dng_opt.bench.runner import BenchRunner

    cfg = BenchConfig(run_tag="fused")
    runner = BenchRunner(cfg)
    results = asyncio.run(runner.run_all())
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

import httpx

from .config import BenchConfig


@dataclass
class RequestResult:
    prompt_tokens: int
    ttft_ms: float          # time to first token (ms)
    tbt_median_ms: float    # median inter-token gap (ms)
    output_tokens: int
    total_ms: float
    success: bool
    error: str = ""

    @property
    def throughput_tok_s(self) -> float:
        if self.total_ms <= 0:
            return 0.0
        return self.output_tokens / (self.total_ms / 1000.0)


@dataclass
class BatchResult:
    batch_size: int
    requests: list[RequestResult] = field(default_factory=list)
    wall_time_ms: float = 0.0

    def ttft_p50(self) -> float:
        vals = sorted(r.ttft_ms for r in self.requests if r.success)
        if not vals:
            return float("nan")
        return vals[len(vals) // 2]

    def ttft_p95(self) -> float:
        vals = sorted(r.ttft_ms for r in self.requests if r.success)
        if not vals:
            return float("nan")
        idx = int(len(vals) * 0.95)
        return vals[min(idx, len(vals) - 1)]

    def tbt_p50(self) -> float:
        vals = sorted(r.tbt_median_ms for r in self.requests if r.success)
        if not vals:
            return float("nan")
        return vals[len(vals) // 2]

    def tbt_p95(self) -> float:
        vals = sorted(r.tbt_median_ms for r in self.requests if r.success)
        if not vals:
            return float("nan")
        idx = int(len(vals) * 0.95)
        return vals[min(idx, len(vals) - 1)]

    def throughput_tok_s(self) -> float:
        total_toks = sum(r.output_tokens for r in self.requests if r.success)
        return total_toks / (self.wall_time_ms / 1000.0) if self.wall_time_ms > 0 else 0.0

    def success_rate(self) -> float:
        if not self.requests:
            return 0.0
        return sum(r.success for r in self.requests) / len(self.requests)


class BenchRunner:
    def __init__(self, cfg: BenchConfig) -> None:
        self.cfg = cfg

    def _build_prompt(self, length_hint: int) -> str:
        import random
        topic = random.choice(self.cfg.topics)
        base = self.cfg.prompt_template.format(topic=topic)
        # Pad to approximate length_hint tokens (~4 chars/token)
        filler = " The answer should be comprehensive and well-structured."
        while len(base) < length_hint * 4:
            base += filler
        return base[:length_hint * 4]

    async def _stream_request(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        max_tokens: int,
    ) -> RequestResult:
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": 0.0,
        }
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }

        t_start = time.perf_counter()
        ttft_ms = float("nan")
        inter_token_gaps: list[float] = []
        output_tokens = 0
        t_last_token = t_start

        try:
            async with client.stream(
                "POST",
                f"{self.cfg.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=120.0,
            ) as resp:
                resp.raise_for_status()
                async for raw_line in resp.aiter_lines():
                    if not raw_line.startswith("data: "):
                        continue
                    chunk = raw_line[6:]
                    if chunk.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue

                    choice = data.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")

                    if content:
                        t_now = time.perf_counter()
                        if output_tokens == 0:
                            ttft_ms = (t_now - t_start) * 1000.0
                            t_last_token = t_now
                        else:
                            inter_token_gaps.append((t_now - t_last_token) * 1000.0)
                            t_last_token = t_now
                        output_tokens += 1

        except Exception as exc:  # noqa: BLE001
            total_ms = (time.perf_counter() - t_start) * 1000.0
            return RequestResult(
                prompt_tokens=len(prompt) // 4,
                ttft_ms=float("nan"),
                tbt_median_ms=float("nan"),
                output_tokens=0,
                total_ms=total_ms,
                success=False,
                error=str(exc),
            )

        total_ms = (time.perf_counter() - t_start) * 1000.0
        tbt_median = (
            sorted(inter_token_gaps)[len(inter_token_gaps) // 2]
            if inter_token_gaps
            else float("nan")
        )

        return RequestResult(
            prompt_tokens=len(prompt) // 4,
            ttft_ms=ttft_ms,
            tbt_median_ms=tbt_median,
            output_tokens=output_tokens,
            total_ms=total_ms,
            success=output_tokens > 0,
        )

    async def _run_batch(
        self,
        batch_size: int,
        n_iters: int,
    ) -> BatchResult:
        sem = asyncio.Semaphore(self.cfg.max_concurrent)
        batch = BatchResult(batch_size=batch_size)
        prompts = [
            self._build_prompt(self.cfg.input_tokens) for _ in range(batch_size * n_iters)
        ]

        async with httpx.AsyncClient() as client:
            # Warmup
            warmup_tasks = [
                self._stream_request(client, self._build_prompt(32), 16)
                for _ in range(self.cfg.warmup_iters)
            ]
            await asyncio.gather(*warmup_tasks)

            # Benchmark
            t_wall_start = time.perf_counter()

            async def _bounded(p: str) -> RequestResult:
                async with sem:
                    return await self._stream_request(client, p, self.cfg.output_tokens)

            results = await asyncio.gather(*[_bounded(p) for p in prompts])

            batch.wall_time_ms = (time.perf_counter() - t_wall_start) * 1000.0
            batch.requests = list(results)

        return batch

    async def run_all(self) -> dict[int, BatchResult]:
        """Run benchmark for every batch_size in cfg and return results dict."""
        all_results: dict[int, BatchResult] = {}
        for bs in self.cfg.batch_sizes:
            print(f"  batch_size={bs} ...", end=" ", flush=True)
            result = await self._run_batch(bs, self.cfg.bench_iters)
            all_results[bs] = result
            sr = result.success_rate()
            print(
                f"ttft_p50={result.ttft_p50():.1f}ms  "
                f"tbt_p50={result.tbt_p50():.1f}ms  "
                f"tok/s={result.throughput_tok_s():.1f}  "
                f"success={sr:.0%}"
            )
        return all_results
