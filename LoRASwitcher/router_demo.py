import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

"""Simple one-LoRA routing baseline with vLLM.

This module demonstrates a practical baseline for adapter routing:
1) load one base model once,
2) choose one adapter from request metadata (`doc_type`),
3) run generation with exactly one LoRA request attached.

It supports both single-request inference and grouped inference
for requests sharing the same adapter to reduce switching overhead.
"""


@dataclass
class InferenceRequest:
    """Input payload for routed generation.

    Attributes:
        doc_type: Logical adapter key. Must match one key from `adapters.json`.
        prompt: User prompt to generate from.
    """

    doc_type: str
    prompt: str


class OneLoRARouter:
    """
    Router for one-LoRA-per-request generation.

    The class reads a config file containing:
    - `base_model`: model ID for vLLM base model loading
    - `adapters`: mapping entries with adapter `name` and HF `repo_id`

    Design goals:
    - Keep implementation minimal for pipeline verification.
    - Enforce one LoRA adapter per generation call.
    - Reuse downloaded adapters through a local path cache.
    """

    def __init__(
        self,
        config_path: Path,
        cache_dir: Path,
        gpu_memory_utilization: float,
        max_model_len: int,
    ) -> None:
        """Initialize vLLM and adapter metadata from JSON config.

        Args:
            config_path: Path to `adapters.json`.
            cache_dir: Writable directory for Hugging Face/vLLM cache.
            gpu_memory_utilization: Fraction of GPU memory vLLM is allowed to use.
            max_model_len: Maximum context length to allocate for KV cache.
        """
        raw = json.loads(config_path.read_text())
        self.base_model = raw["base_model"]
        self.adapters = {item["name"]: item["repo_id"] for item in raw["adapters"]}
        self._adapter_local_paths: Dict[str, str] = {}
        self._adapter_counter = 1
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_switch_sec = 0.0
        self.last_generate_sec = 0.0

        # Force writable local cache path to avoid permission issues.
        os.environ["HF_HOME"] = str(self.cache_dir)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(self.cache_dir / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(self.cache_dir / "transformers")

        try:
            self.llm = LLM(
                model=self.base_model,
                enable_lora=True,
                max_loras=1,  # Enforce single active LoRA in one generation call.
                trust_remote_code=True,
                download_dir=str(self.cache_dir / "vllm"),
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize vLLM base model. "
                f"Check that model '{self.base_model}' exists and is accessible. "
                "Also check cache permissions and GPU free memory. "
                "If the repo is private/gated, run `hf auth login` first."
            ) from exc

    def _ensure_adapter_downloaded(self, adapter_name: str) -> str:
        """Resolve adapter local path, downloading once if needed.

        Args:
            adapter_name: Adapter key from routing input (for example, `finance`).

        Returns:
            Local directory path containing the adapter files.

        Raises:
            ValueError: If `adapter_name` is not present in configured adapters.
        """
        if adapter_name not in self.adapters:
            supported = ", ".join(sorted(self.adapters.keys()))
            raise ValueError(f"Unknown adapter '{adapter_name}'. Supported: {supported}")

        if adapter_name in self._adapter_local_paths:
            return self._adapter_local_paths[adapter_name]

        repo_id = self.adapters[adapter_name]
        local_dir = snapshot_download(repo_id=repo_id, cache_dir=str(self.cache_dir / "hub"))
        self._adapter_local_paths[adapter_name] = local_dir
        return local_dir

    def generate_single(self, request: InferenceRequest, max_tokens: int = 128) -> str:
        """Generate text for one request using exactly one routed LoRA.

        Args:
            request: Routed request containing `doc_type` and `prompt`.
            max_tokens: Maximum number of generated tokens.

        Returns:
            Generated text string from the model output.
        """
        switch_started = time.time()
        adapter_path = self._ensure_adapter_downloaded(request.doc_type)
        lora_request = LoRARequest(request.doc_type, self._adapter_counter, adapter_path)
        self._adapter_counter += 1
        self.last_switch_sec = time.time() - switch_started

        params = SamplingParams(max_tokens=max_tokens, temperature=0.2, top_p=0.9)
        generate_started = time.time()
        outputs = self.llm.generate([request.prompt], params, lora_request=lora_request)
        self.last_generate_sec = time.time() - generate_started
        return outputs[0].outputs[0].text

    def generate_grouped(self, requests: List[InferenceRequest], max_tokens: int = 128) -> List[str]:
        """
        Generate for many requests by grouping them by adapter key.

        This method preserves one-LoRA semantics by applying one adapter per
        grouped sub-batch. It returns outputs in the same order as inputs.

        Args:
            requests: List of routed requests.
            max_tokens: Maximum number of generated tokens for each request.

        Returns:
            List of generated texts ordered by original request indices.
        """
        grouped: Dict[str, List[tuple[int, InferenceRequest]]] = {}
        for idx, req in enumerate(requests):
            grouped.setdefault(req.doc_type, []).append((idx, req))

        responses = [""] * len(requests)
        params = SamplingParams(max_tokens=max_tokens, temperature=0.2, top_p=0.9)

        for adapter_name, items in grouped.items():
            switch_started = time.time()
            adapter_path = self._ensure_adapter_downloaded(adapter_name)
            lora_request = LoRARequest(adapter_name, self._adapter_counter, adapter_path)
            self._adapter_counter += 1
            self.last_switch_sec = time.time() - switch_started

            prompts = [req.prompt for _, req in items]
            generate_started = time.time()
            outputs = self.llm.generate(prompts, params, lora_request=lora_request)
            self.last_generate_sec = time.time() - generate_started
            for (index, _), output in zip(items, outputs):
                responses[index] = output.outputs[0].text

        return responses


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the routing demo script."""
    parser = argparse.ArgumentParser(description="One-LoRA-per-request routing demo with vLLM.")
    parser.add_argument("--config", default="adapters.json", help="Path to adapter config JSON.")
    parser.add_argument("--doc-type", required=True, help="Adapter key to route to (e.g. 'finance').")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max generated tokens.")
    parser.add_argument(
        "--bench-grouped",
        action="store_true",
        help="Run a tiny grouped benchmark to validate one-LoRA routing pipeline.",
    )
    parser.add_argument(
        "--cache-dir",
        default=".hf_cache",
        help="Writable cache directory for HF and vLLM artifacts.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.15,
        help="Fraction of total GPU memory vLLM can reserve (use low value when GPU is busy).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Reduce context length to lower KV cache memory usage.",
    )
    parser.add_argument(
        "--validate-all-adapters",
        action="store_true",
        help="Run one inference for each configured adapter in one process.",
    )
    parser.add_argument(
        "--bench-requests",
        type=int,
        default=0,
        help="Run batched benchmark with this many synthetic requests (example: 50).",
    )
    parser.add_argument(
        "--bench-batch-size",
        type=int,
        default=10,
        help="Batch size used for --bench-requests benchmark.",
    )
    return parser.parse_args()


def run_grouped_benchmark(router: OneLoRARouter, max_tokens: int) -> None:
    """Run a tiny grouped benchmark to validate end-to-end routing flow.

    The benchmark mixes several `doc_type` values, then runs grouped inference
    and prints short output snippets plus total elapsed time.
    """
    sample_requests = [
        InferenceRequest("tldr", "Summarize this quarterly report in 3 bullets."),
        InferenceRequest("tldr", "Extract risks from this budget statement."),
        InferenceRequest("tldr", "Write a one-line neutral summary for this report."),
    ]

    started = time.time()
    outputs = router.generate_grouped(sample_requests, max_tokens=max_tokens)
    elapsed = time.time() - started

    print(f"[bench] processed={len(sample_requests)} elapsed_sec={elapsed:.2f}")
    for idx, text in enumerate(outputs):
        snippet = text[:120].replace("\n", " ")
        print(f"[bench][{idx}] {snippet}...")


def _build_even_requests(router: OneLoRARouter, total_requests: int) -> List[InferenceRequest]:
    """Build requests distributed as evenly as possible across configured adapters."""
    adapter_names = sorted(router.adapters.keys())
    if not adapter_names:
        raise ValueError("No adapters configured in adapters.json.")

    per_adapter = total_requests // len(adapter_names)
    remainder = total_requests % len(adapter_names)

    requests: List[InferenceRequest] = []
    for idx, adapter_name in enumerate(adapter_names):
        count = per_adapter + (1 if idx < remainder else 0)
        for sample_id in range(count):
            requests.append(
                InferenceRequest(
                    doc_type=adapter_name,
                    prompt=(
                        f"Adapter={adapter_name}; sample={sample_id}; "
                        "return one concise sentence."
                    ),
                )
            )
    return requests


def run_batch_benchmark(
    router: OneLoRARouter,
    max_tokens: int,
    total_requests: int,
    batch_size: int,
) -> None:
    """Run batched benchmark with even adapter distribution.

    Adapter groups are processed sequentially in this baseline.
    """
    requests = _build_even_requests(router, total_requests)
    started = time.time()
    consumed = 0
    while consumed < len(requests):
        batch = requests[consumed : consumed + batch_size]
        _ = router.generate_grouped(batch, max_tokens=max_tokens)
        consumed += len(batch)

    elapsed = time.time() - started
    req_per_sec = total_requests / elapsed if elapsed > 0 else 0.0
    print(
        f"[bench50] total={total_requests} batch_size={batch_size} "
        f"elapsed_sec={elapsed:.2f} req_per_sec={req_per_sec:.2f}"
    )
    print("[bench50] adapter_parallel=false (sequential per adapter group).")


def run_adapter_validation(router: OneLoRARouter, max_tokens: int) -> None:
    """Validate that each configured adapter can run one inference end-to-end."""
    print(f"[validate] adapters={len(router.adapters)}")
    ok = 0
    fail = 0
    for adapter_name in sorted(router.adapters.keys()):
        request = InferenceRequest(
            doc_type=adapter_name,
            prompt=f"Short test for adapter {adapter_name}. Return one sentence.",
        )
        try:
            started = time.time()
            output = router.generate_single(request, max_tokens=max_tokens)
            elapsed = time.time() - started
            switch_ms = router.last_switch_sec * 1000
            infer_ms = router.last_generate_sec * 1000
            print(
                f"[validate][ok] adapter={adapter_name} elapsed_sec={elapsed:.2f} "
                f"switch_ms={switch_ms:.1f} infer_ms={infer_ms:.1f} out_chars={len(output)}"
            )
            ok += 1
        except Exception as exc:
            print(f"[validate][fail] adapter={adapter_name} error={type(exc).__name__}: {exc}")
            fail += 1
    print(f"[validate][summary] ok={ok} fail={fail}")


def main() -> None:
    """CLI entrypoint for single generation and optional grouped benchmark."""
    args = parse_args()
    router = OneLoRARouter(
        Path(args.config),
        Path(args.cache_dir).resolve(),
        args.gpu_memory_utilization,
        args.max_model_len,
    )

    req = InferenceRequest(doc_type=args.doc_type, prompt=args.prompt)
    started = time.time()
    text = router.generate_single(req, max_tokens=args.max_tokens)
    elapsed = time.time() - started

    print(f"[single] doc_type={args.doc_type} latency_sec={elapsed:.2f}")
    print(
        f"[single] switch_ms={router.last_switch_sec * 1000:.1f} "
        f"infer_ms={router.last_generate_sec * 1000:.1f}"
    )
    print(text)

    if args.bench_grouped:
        run_grouped_benchmark(router, max_tokens=args.max_tokens)
    if args.validate_all_adapters:
        run_adapter_validation(router, max_tokens=args.max_tokens)
    if args.bench_requests > 0:
        run_batch_benchmark(
            router,
            max_tokens=args.max_tokens,
            total_requests=args.bench_requests,
            batch_size=args.bench_batch_size,
        )


if __name__ == "__main__":
    main()
