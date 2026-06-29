"""Benchmark configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchConfig:
    # vLLM server endpoint
    base_url: str = "http://127.0.0.1:8000"
    api_key: str = "EMPTY"
    model: str = "Qwen/Qwen3.5-30B-A3B"

    # Workload
    batch_sizes: list[int] = field(default_factory=lambda: [1, 4, 8, 16])
    output_tokens: int = 256   # max_tokens per request
    input_tokens: int = 128    # approximate prompt length

    # Timing
    warmup_iters: int = 3
    bench_iters: int = 20

    # Concurrency (number of simultaneous HTTP requests for batch emulation)
    max_concurrent: int = 32

    # Output
    results_dir: str = "results"
    run_tag: str = "baseline"   # "baseline" | "instrumented" | "fused"

    # SLO thresholds for pass/fail in the report
    ttft_slo_ms: float = 4000.0
    tbt_slo_ms: float = 80.0

    # Prompt template filled to approximate input_tokens length
    prompt_template: str = (
        "You are a helpful assistant. "
        "Explain the following topic in detail: {topic}"
    )
    topics: list[str] = field(default_factory=lambda: [
        "quantum entanglement",
        "transformer attention mechanism",
        "the delta rule in neural networks",
        "GPU memory bandwidth",
        "autoregressive language model decoding",
    ])
