from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator

from .tokens import tokens_to_chars


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def leval_gsm100_rows(hf_root: Path) -> list[dict[str, Any]]:
    path = hf_root / "L4NLP__LEval" / "LEval" / "Exam" / "gsm100.jsonl"
    return _read_jsonl(path)


def leval_multidoc_rows(hf_root: Path) -> list[dict[str, Any]]:
    path = hf_root / "L4NLP__LEval" / "LEval" / "Generation" / "multidoc_qa.jsonl"
    return _read_jsonl(path)


def loogle_shortdep_rows(hf_root: Path) -> list[dict[str, Any]]:
    path = hf_root / "bigai-nlco__LooGLE" / "shortdep_qa" / "test.jsonl"
    return _read_jsonl(path)


def loogle_longdep_rows(hf_root: Path) -> list[dict[str, Any]]:
    path = hf_root / "bigai-nlco__LooGLE" / "longdep_qa" / "test.jsonl"
    return _read_jsonl(path)


def hf_available(hf_root: Path) -> bool:
    return bool(leval_gsm100_rows(hf_root) or loogle_shortdep_rows(hf_root))


def _fmt_instructions(raw: Any) -> str:
    if isinstance(raw, list):
        return "\n".join(str(x) for x in raw)
    return str(raw or "")


def _fmt_outputs(raw: Any) -> str:
    if isinstance(raw, list):
        return "\n".join(str(x) for x in raw)
    return str(raw or "")


def build_tool_agent_from_leval(row: dict[str, Any]) -> tuple[str, str, str]:
    inp = str(row.get("input") or "")
    inst = _fmt_instructions(row.get("instructions"))
    gold = _fmt_outputs(row.get("outputs"))
    prompt = f"{inp}\n\n{inst}".strip()
    return prompt, gold, "single_doc_qa"


def build_long_context_from_loogle(row: dict[str, Any]) -> tuple[str, str, str]:
    ctx = str(row.get("context") or "")
    q = str(row.get("question") or "")
    gold = str(row.get("answer") or "")
    prompt = f"{ctx}\n\nQuestion: {q}".strip()
    task = "longdep_qa" if len(ctx) > 80_000 else "shortdep_qa"
    return prompt, gold, task


def iter_tool_agent_prompts(hf_root: Path, rng: random.Random, n: int) -> Iterator[tuple[str, str | None, str]]:
    rows = leval_gsm100_rows(hf_root)
    if rows:
        for i in range(n):
            row = rows[i % len(rows)]
            prompt, gold, task = build_tool_agent_from_leval(row)
            yield prompt, gold, task
        return

    shared = (
        "You are a helpful assistant. Follow the examples below.\n"
        "Example 1: 2+2=4\nExample 2: 3+5=8\n"
    )
    for i in range(n):
        repeats = 35 + (i % 5)
        prefix = shared * repeats
        q = f"\n\nQuestion {i}: What is {rng.randint(10, 99)} + {rng.randint(1, 49)}?"
        yield prefix + q, str(rng.randint(0, 9)), "single_doc_qa"


def iter_long_context_prompts(hf_root: Path, rng: random.Random, n: int) -> Iterator[tuple[str, str | None, str]]:
    short_rows = loogle_shortdep_rows(hf_root)
    long_rows = loogle_longdep_rows(hf_root)
    pool: list[tuple[dict[str, Any], str]] = []
    for r in short_rows:
        pool.append((r, "shortdep_qa"))
    for r in long_rows:
        pool.append((r, "longdep_qa"))

    if pool:
        by_doc: dict[str, list[tuple[dict[str, Any], str]]] = {}
        for row, task in pool:
            doc_id = str(row.get("doc_id") or id(row))
            by_doc.setdefault(doc_id, []).append((row, task))
        doc_ids = list(by_doc.keys())
        rng.shuffle(doc_ids)
        idx = 0
        for _ in range(n):
            doc_id = doc_ids[idx % len(doc_ids)]
            row, task = by_doc[doc_id][idx % len(by_doc[doc_id])]
            prompt, gold, task = build_long_context_from_loogle(row)
            yield prompt, gold, task
            idx += 1
        return

    for i in range(n):
        doc_tokens = rng.randint(12_000, 40_000)
        filler = "The document states that key fact number {}. "
        ctx = (filler * (doc_tokens // 12))[: tokens_to_chars(doc_tokens)]
        q = f"\n\nQuestion: Summarize fact related to index {i}."
        yield ctx + q, "summary", "shortdep_qa"


def iter_conversation_prompts(
    rng: random.Random, n: int
) -> Iterator[tuple[str, None, str]]:
    vocab = ["hello", "thanks", "help", "explain", "summarize", "please"]
    for turn in range(n):
        words = [rng.choice(vocab) for _ in range(rng.randint(80, 220))]
        yield f"User turn {turn}: " + " ".join(words), None, "chat"
