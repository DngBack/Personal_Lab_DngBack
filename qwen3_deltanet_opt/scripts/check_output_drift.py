#!/usr/bin/env python3
"""
Compare baseline and fused vLLM outputs on deterministic prompts.

This is a lightweight smoke test for end-to-end generation drift.  It does not
prove semantic equivalence, but it catches obvious issues such as empty output,
format breakage, repeated garbage text, or large text divergence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


DEFAULT_PROMPTS = [
    "Answer in one short sentence: What is the capital of France?",
    "Translate to Vietnamese: The model should return stable results.",
    "Solve step by step but keep it brief: If x + 7 = 19, what is x?",
    "Return valid JSON only with keys city and country for Hanoi.",
    "Write a Python function named add_one that returns x + 1.",
    "Summarize in two bullet points: DeltaNet uses recurrent state during decoding.",
    "Hoan thanh cau sau bang tieng Viet: Tri tue nhan tao co the giup lap trinh vien",
    "Classify the sentiment as positive, negative, or neutral: The latency is lower now.",
    "Write exactly five numbered steps for checking a model-serving benchmark.",
    "Return CSV only with header name,value and three rows for alpha=1, beta=2, gamma=3.",
    "Explain in Vietnamese in three sentences: vi sao can so sanh output baseline va fused.",
    "Write a short story of about 120 words about a developer debugging GPU memory usage.",
]


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _post_chat(base_url: str, model: str, prompt: str, max_tokens: int) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer EMPTY",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"]


def _check_ready(base_url: str) -> str:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/v1/models", timeout=10) as resp:
        data = json.load(resp)
    return data["data"][0]["id"]


def _looks_bad(text: str) -> list[str]:
    issues: list[str] = []
    stripped = text.strip()
    if not stripped:
        issues.append("empty")
    if len(stripped) < 3:
        issues.append("too_short")
    words = stripped.split()
    if len(words) >= 16:
        tail = words[-16:]
        if len(set(tail)) <= 3:
            issues.append("repeated_tail")
    if "\ufffd" in stripped:
        issues.append("replacement_char")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-url", default="http://127.0.0.1:8000")
    parser.add_argument("--fused-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--out", default="results/output_drift.json")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.70,
        help="Flag cases below this text-similarity ratio.",
    )
    args = parser.parse_args()

    try:
        baseline_model = _check_ready(args.baseline_url)
        fused_model = _check_ready(args.fused_url)
    except (urllib.error.URLError, TimeoutError, KeyError, IndexError) as exc:
        print(f"ERROR: server is not ready: {exc}", file=sys.stderr)
        return 2

    print(f"baseline ready: {baseline_model}")
    print(f"fused ready   : {fused_model}")
    print("")

    rows: list[dict[str, Any]] = []
    exact = 0
    flagged = 0

    for i, prompt in enumerate(DEFAULT_PROMPTS, 1):
        t0 = time.perf_counter()
        baseline = _post_chat(args.baseline_url, args.model, prompt, args.max_tokens)
        fused = _post_chat(args.fused_url, args.model, prompt, args.max_tokens)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        same = baseline == fused
        sim = _similarity(baseline, fused)
        issues = {
            "baseline": _looks_bad(baseline),
            "fused": _looks_bad(fused),
        }
        is_flagged = bool(issues["baseline"] or issues["fused"] or sim < args.similarity_threshold)
        exact += int(same)
        flagged += int(is_flagged)

        row = {
            "index": i,
            "prompt": prompt,
            "exact_match": same,
            "similarity": sim,
            "flagged": is_flagged,
            "issues": issues,
            "elapsed_ms": elapsed_ms,
            "baseline": baseline,
            "fused": fused,
        }
        rows.append(row)

        status = "OK" if not is_flagged else "CHECK"
        print(
            f"[{status}] #{i}: exact={same} sim={sim:.3f} "
            f"baseline_len={len(baseline.strip())} fused_len={len(fused.strip())}"
        )
        if is_flagged or not same:
            print(f"  prompt  : {prompt}")
            print(f"  baseline: {baseline.strip()[:240]!r}")
            print(f"  fused   : {fused.strip()[:240]!r}")

    summary = {
        "baseline_url": args.baseline_url,
        "fused_url": args.fused_url,
        "model": args.model,
        "num_prompts": len(DEFAULT_PROMPTS),
        "exact_matches": exact,
        "flagged": flagged,
        "mean_similarity": sum(r["similarity"] for r in rows) / len(rows),
        "results": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("")
    print(
        "summary: "
        f"exact={exact}/{len(DEFAULT_PROMPTS)} "
        f"flagged={flagged}/{len(DEFAULT_PROMPTS)} "
        f"mean_similarity={summary['mean_similarity']:.3f}"
    )
    print(f"wrote: {out_path}")
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
