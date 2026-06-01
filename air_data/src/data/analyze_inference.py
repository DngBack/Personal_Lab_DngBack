"""Researcher-oriented analysis for long-context inference optimization.

Focus: context length distributions (mean / percentiles), document & prefix
reuse (KV-cache / prefix-caching opportunities), and static vs dynamic prompt
split for prefill vs decode budgeting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_MODULE_DIR = Path(__file__).resolve().parent
_AIR_DATA_ROOT = _MODULE_DIR.parents[1]
_DEFAULT_HF_ROOT = _AIR_DATA_ROOT / "data" / "hf"

# Rough English-centric estimate; Vietnamese (ShareChat) often ~2–3 chars/token.
CHARS_PER_TOKEN = 4.0


def _percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = int(len(ordered) * p / 100)
    return ordered[min(idx, len(ordered) - 1)]


def _length_dist(values: list[int]) -> dict[str, Any]:
    if not values:
        return {}
    n = len(values)
    mean = sum(values) / n
    return {
        "n": n,
        "chars_mean": round(mean, 1),
        "chars_p50": _percentile(values, 50),
        "chars_p90": _percentile(values, 90),
        "chars_p99": _percentile(values, 99),
        "chars_max": max(values),
        "tokens_est_mean": round(mean / CHARS_PER_TOKEN, 0),
        "tokens_est_p99": round(_percentile(values, 99) / CHARS_PER_TOKEN, 0),
    }


def _shared_prefix_len(texts: list[str]) -> int:
    if not texts:
        return 0
    prefix = texts[0]
    for text in texts[1:]:
        i = 0
        limit = min(len(prefix), len(text))
        while i < limit and prefix[i] == text[i]:
            i += 1
        prefix = prefix[:i]
    return len(prefix)


def _exact_reuse_stats(texts: list[str]) -> dict[str, Any]:
    hashes = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
    counts = Counter(hashes)
    reused_groups = sum(1 for c in counts.values() if c > 1)
    rows_in_reused = sum(c for c in counts.values() if c > 1)
    return {
        "unique_exact": len(counts),
        "rows_with_duplicate_context": rows_in_reused - reused_groups,
        "groups_with_2plus_rows": reused_groups,
    }


def analyze_loogle_file(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    ctx_lens = [len(str(r.get("context") or "")) for r in rows]
    q_lens = [len(str(r.get("question") or "")) for r in rows]
    ans_lens = [len(str(r.get("answer") or "")) for r in rows]

    by_doc: dict[str, list[int]] = defaultdict(list)
    doc_to_ctx_hash: dict[str, str] = {}
    for r in rows:
        doc_id = str(r.get("doc_id") or "")
        ctx = str(r.get("context") or "")
        by_doc[doc_id].append(len(ctx))
        doc_to_ctx_hash[doc_id] = hashlib.sha256(ctx.encode()).hexdigest()

    total_ctx_if_naive = sum(ctx_lens)
    unique_doc_ctx = 0
    seen_docs: set[str] = set()
    for r in rows:
        doc_id = str(r.get("doc_id") or "")
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        unique_doc_ctx += len(str(r.get("context") or ""))

    q_per_doc = [len(v) for v in by_doc.values()]
    uniform_ctx_docs = sum(1 for lens in by_doc.values() if len(set(lens)) == 1)

    return {
        "path": str(path),
        "rows": len(rows),
        "context": _length_dist(ctx_lens),
        "question": _length_dist(q_lens),
        "answer": _length_dist(ans_lens),
        "exact_context_reuse": _exact_reuse_stats([str(r.get("context") or "") for r in rows]),
        "document_reuse": {
            "unique_documents": len(by_doc),
            "questions_per_doc": _length_dist(q_per_doc),
            "all_docs_same_context_len": uniform_ctx_docs == len(by_doc),
            "uniform_context_docs": uniform_ctx_docs,
        },
        "prefill_optimization": {
            "strategy": "Group by doc_id; prefill context once, append question per query (KV cache / prefix cache).",
            "naive_total_context_chars": total_ctx_if_naive,
            "unique_document_context_chars": unique_doc_ctx,
            "prefill_char_savings_ratio": round(1 - unique_doc_ctx / max(total_ctx_if_naive, 1), 4),
            "effective_queries_per_prefill_mean": round(len(rows) / max(len(by_doc), 1), 2),
        },
        "decode_optimization": {
            "note": "Answers are short (mean << context). Budget GPU for prefill; decode is cheap.",
            "answer_to_context_ratio_mean": round(
                (sum(ans_lens) / len(ans_lens)) / max(sum(ctx_lens) / len(ctx_lens), 1), 5
            ),
        },
    }


def analyze_leval_file(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    inputs = [str(r.get("input") or "") for r in rows]
    instructions = [
        "\n".join(r["instructions"])
        if isinstance(r.get("instructions"), list)
        else str(r.get("instructions") or "")
        for r in rows
    ]
    outputs = [
        "\n".join(r["outputs"]) if isinstance(r.get("outputs"), list) else str(r.get("outputs") or "")
        for r in rows
    ]

    prefix_len = _shared_prefix_len(inputs)
    suffix_lens = [len(inp) - prefix_len for inp in inputs]
    static_chars = prefix_len
    dynamic_inst_mean = sum(len(i) for i in instructions) / max(len(instructions), 1)

    naive_prefill = sum(len(inp) + len(inst) for inp, inst in zip(inputs, instructions))
    exact = _exact_reuse_stats(inputs)
    has_full_prefix_reuse = prefix_len > 0 and max(suffix_lens) == 0
    has_partial_prefix = prefix_len > 500 and not has_full_prefix_reuse

    if has_full_prefix_reuse:
        optimized_prefill = static_chars + dynamic_inst_mean * len(rows)
        savings_ratio = round(1 - optimized_prefill / max(naive_prefill, 1), 4)
        recommended = "Prefix-cache entire `input`; only `instructions` vary per query."
    elif has_partial_prefix:
        optimized_prefill = static_chars + sum(suffix_lens) + sum(len(i) for i in instructions)
        savings_ratio = round(1 - optimized_prefill / max(naive_prefill, 1), 4)
        recommended = "Prefix-cache shared few-shot block; full suffix still unique per row."
    else:
        optimized_prefill = naive_prefill
        savings_ratio = 0.0
        recommended = "No cross-row reuse — full prefill per example; prioritize long-context kernels."

    return {
        "path": str(path),
        "rows": len(rows),
        "input_total": _length_dist([len(x) for x in inputs]),
        "input_unique_suffix": _length_dist(suffix_lens),
        "instructions": _length_dist([len(x) for x in instructions]),
        "outputs": _length_dist([len(x) for x in outputs]),
        "exact_input_reuse": _exact_reuse_stats(inputs),
        "prompt_structure": {
            "shared_fewshot_prefix_chars": prefix_len,
            "shared_fewshot_prefix_tokens_est": round(prefix_len / CHARS_PER_TOKEN, 0),
            "per_row_unique_input_suffix_mean_chars": round(
                sum(suffix_lens) / max(len(suffix_lens), 1), 1
            ),
            "static_vs_dynamic": (
                "HIGH_REUSE: entire `input` identical — cache one prefill, swap `instructions` only."
                if prefix_len > 0 and max(suffix_lens) == 0
                else (
                    "PARTIAL_REUSE: shared few-shot prefix — use prefix caching on `input`."
                    if prefix_len > 100
                    else "LOW_REUSE: each row has distinct long `input` — full prefill per example."
                )
            ),
        },
        "prefill_optimization": {
            "naive_prefill_chars": int(naive_prefill),
            "optimized_prefill_chars_est": int(optimized_prefill),
            "prefill_char_savings_ratio": savings_ratio,
            "recommended": recommended,
        },
        "decode_optimization": {
            "output_to_input_ratio_mean": round(
                (sum(len(o) for o in outputs) / len(outputs))
                / max(sum(len(i) for i in inputs) / len(inputs), 1),
                5,
            ),
            "note": "Many Exam tasks have tiny `outputs` (e.g. gsm100) — optimize prefill latency, not max_tokens.",
        },
    }


def build_inference_report(hf_root: Path | None = None) -> dict[str, Any]:
    root = Path(hf_root or _DEFAULT_HF_ROOT).expanduser().resolve()
    loogle_base = root / "bigai-nlco__LooGLE"
    leval_base = root / "L4NLP__LEval"

    loogle: dict[str, Any] = {}
    if loogle_base.exists():
        for path in sorted(loogle_base.rglob("*.jsonl")):
            loogle[path.parent.name] = analyze_loogle_file(path)

    leval: dict[str, Any] = {}
    if leval_base.exists():
        for path in sorted(leval_base.rglob("*.jsonl")):
            key = str(path.relative_to(leval_base))
            leval[key] = analyze_leval_file(path)

    # Cross-dataset researcher recommendations
    loogle_savings = [
        v["prefill_optimization"]["prefill_char_savings_ratio"]
        for v in loogle.values()
        if "prefill_optimization" in v
    ]
    leval_high_reuse = [
        k
        for k, v in leval.items()
        if v.get("prompt_structure", {}).get("shared_fewshot_prefix_chars", 0) > 1000
        and v.get("input_unique_suffix", {}).get("chars_max", 1) == 0
    ]

    return {
        "hf_data_root": str(root),
        "assumptions": {
            "chars_per_token_est": CHARS_PER_TOKEN,
            "note": "Token estimates are approximate; measure with your tokenizer for deployment.",
        },
        "loogle": loogle,
        "leval": leval,
        "researcher_summary": {
            "long_context": {
                "loogle_context_tokens_est_p99_range": [
                    min(
                        v["context"]["tokens_est_p99"]
                        for v in loogle.values()
                        if v.get("context")
                    ),
                    max(
                        v["context"]["tokens_est_p99"]
                        for v in loogle.values()
                        if v.get("context")
                    ),
                ],
                "leval_heaviest_files_by_mean_input_tokens": sorted(
                    [
                        (
                            k,
                            v["input_total"]["tokens_est_mean"],
                        )
                        for k, v in leval.items()
                        if v.get("input_total")
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                )[:5],
            },
            "reuse_for_inference": {
                "loogle_doc_grouping": (
                    "Same doc_id → identical context. "
                    f"Prefill savings {min(loogle_savings):.0%}–{max(loogle_savings):.0%} "
                    "if you cache KV per document and run multiple questions."
                ),
                "leval_fewshot_prefix_cache": (
                    f"Full shared `input` block in: {leval_high_reuse or ['(none in current export)']}. "
                    "Use prefix caching / continuous batching with static system+few-shot."
                ),
                "leval_unique_context_files": [
                    k
                    for k, v in leval.items()
                    if v.get("exact_input_reuse", {}).get("unique_exact") == v.get("rows")
                ],
            },
            "inference_budget_priority": [
                "1. Prefill dominates (context >> answer). Optimize: FlashAttention, chunked prefill, prefix/KV reuse.",
                "2. LooGLE: schedule by doc_id (≈8–19 questions per doc) not random shuffle.",
                "3. LEval Exam gsm100: one 4.2k-token prefill serves 100 queries (instructions only change).",
                "4. LEval codeU/narrative_qa/legal: plan for 25k–55k token contexts; little cross-row reuse.",
                "5. Decode: cap max_new_tokens low on Exam; Generation subsets need higher caps (summaries).",
            ],
        },
    }


def _print_researcher_report(report: dict[str, Any]) -> None:
    print("=" * 72)
    print("INFERENCE OPTIMIZATION RESEARCH REPORT")
    print("=" * 72)
    print(f"Data root: {report['hf_data_root']}")
    print(f"Token estimate: ~{report['assumptions']['chars_per_token_est']} chars/token\n")

    print("--- LooGLE: long context + document reuse ---\n")
    for name, data in report.get("loogle", {}).items():
        ctx = data["context"]
        doc = data["document_reuse"]
        opt = data["prefill_optimization"]
        print(f"[{name}] {data['rows']} rows, {doc['unique_documents']} unique docs")
        print(
            f"  Context: mean {ctx['chars_mean']:,.0f} chars (~{ctx['tokens_est_mean']:,.0f} tok) "
            f"| p99 ~{ctx['tokens_est_p99']:,.0f} tok | max ~{ctx['chars_max']/CHARS_PER_TOKEN:,.0f} tok"
        )
        print(
            f"  Questions/doc: mean {doc['questions_per_doc']['chars_mean']:.1f} "
            f"| max {doc['questions_per_doc']['chars_max']}"
        )
        print(
            f"  REUSE: prefill savings {opt['prefill_char_savings_ratio']:.1%} "
            f"({opt['effective_queries_per_prefill_mean']:.1f} Q per doc prefill)"
        )
        print(f"  → {opt['strategy']}\n")

    print("--- LEval: few-shot prefix vs unique context ---\n")
    for key, data in report.get("leval", {}).items():
        ps = data["prompt_structure"]
        opt = data["prefill_optimization"]
        print(f"[{key}] {data['rows']} rows")
        print(
            f"  Input: mean {data['input_total']['chars_mean']:,.0f} chars "
            f"(p99 {data['input_total']['chars_p99']:,.0f})"
        )
        print(f"  Shared few-shot prefix: {ps['shared_fewshot_prefix_chars']:,} chars (~{ps['shared_fewshot_prefix_tokens_est']:,.0f} tok)")
        print(f"  Structure: {ps['static_vs_dynamic']}")
        if opt["prefill_char_savings_ratio"] > 0.05:
            print(f"  REUSE: prefill savings {opt['prefill_char_savings_ratio']:.1%} — {opt['recommended']}")
        else:
            print(f"  REUSE: none — {opt['recommended']}")
        print()

    print("--- Researcher action list ---\n")
    for item in report["researcher_summary"]["inference_budget_priority"]:
        print(f"  {item}")
    print()
    ru = report["researcher_summary"]["reuse_for_inference"]
    print(f"  LooGLE: {ru['loogle_doc_grouping']}")
    print(f"  LEval:  {ru['leval_fewshot_prefix_cache']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Long-context inference optimization analysis (reuse, means, percentiles)."
    )
    parser.add_argument("--hf-root", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    report = build_inference_report(args.hf_root)
    _print_researcher_report(report)

    if args.json_out:
        out = Path(args.json_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
