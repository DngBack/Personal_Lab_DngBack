from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator

from ..config import CONVERSATION_LEVAL_SOURCES, REALISTIC_INPUT_CAP
from ..tokens import estimate_token_count, truncate_prompt_natural

_DOWNLOAD_HINT = (
    "Download required datasets via air_data:\n"
    "  cd air_data\n"
    "  python3 src/data/down_data.py L4NLP/LEval --config gsm100\n"
    "  python3 src/data/down_data.py L4NLP/LEval --config quality\n"
    "  python3 src/data/down_data.py bigai-nlco/LooGLE --config shortdep_qa\n"
    "  python3 src/data/down_data.py bigai-nlco/LooGLE --config longdep_qa"
)


class HfDataError(RuntimeError):
    pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _fmt_instructions(raw: Any) -> str:
    if isinstance(raw, list):
        return "\n".join(str(x) for x in raw)
    return str(raw or "")


def _fmt_outputs(raw: Any) -> str:
    if isinstance(raw, list):
        return "\n".join(str(x) for x in raw)
    return str(raw or "")


def leval_rows(hf_root: Path, subset: str, config_name: str) -> list[dict[str, Any]]:
    path = hf_root / "L4NLP__LEval" / "LEval" / subset / f"{config_name}.jsonl"
    return _read_jsonl(path)


def leval_gsm100_rows(hf_root: Path) -> list[dict[str, Any]]:
    return leval_rows(hf_root, "Exam", "gsm100")


def loogle_shortdep_rows(hf_root: Path) -> list[dict[str, Any]]:
    path = hf_root / "bigai-nlco__LooGLE" / "shortdep_qa" / "test.jsonl"
    return _read_jsonl(path)


def loogle_longdep_rows(hf_root: Path) -> list[dict[str, Any]]:
    path = hf_root / "bigai-nlco__LooGLE" / "longdep_qa" / "test.jsonl"
    return _read_jsonl(path)


def build_tool_agent_from_leval(row: dict[str, Any]) -> tuple[str, str, str]:
    inp = str(row.get("input") or "")
    inst = _fmt_instructions(row.get("instructions"))
    gold = _fmt_outputs(row.get("outputs"))
    prompt = f"{inp}\n\n{inst}".strip()
    return prompt, gold, "single_doc_qa"


def build_conversation_from_leval(row: dict[str, Any]) -> tuple[str, str, str]:
    """Conversation = short real user turn (instructions field), not padded few-shot block."""
    inst = _fmt_instructions(row.get("instructions"))
    gold = _fmt_outputs(row.get("outputs"))
    prompt = inst.strip()
    return prompt, gold, "chat"


def build_long_context_from_loogle(row: dict[str, Any]) -> tuple[str, str, str]:
    ctx = str(row.get("context") or "")
    q = str(row.get("question") or "")
    gold = str(row.get("answer") or "")
    prompt = f"{ctx}\n\nQuestion: {q}".strip()
    task = "longdep_qa" if len(ctx) > 80_000 else "shortdep_qa"
    return prompt, gold, task


def _conversation_pool(hf_root: Path) -> list[dict[str, Any]]:
    cap = REALISTIC_INPUT_CAP["conversation"]
    pool: list[dict[str, Any]] = []
    for subset, cfg in CONVERSATION_LEVAL_SOURCES:
        for row in leval_rows(hf_root, subset, cfg):
            prompt, _, _ = build_conversation_from_leval(row)
            n = estimate_token_count(prompt)
            if 16 <= n <= cap:
                pool.append(row)
    return pool


def check_hf_data(hf_root: Path) -> dict[str, Any]:
    gsm = leval_gsm100_rows(hf_root)
    short = loogle_shortdep_rows(hf_root)
    long_ = loogle_longdep_rows(hf_root)
    conv_rows = _conversation_pool(hf_root)
    conv_sources = [
        f"{subset}/{cfg}"
        for subset, cfg in CONVERSATION_LEVAL_SOURCES
        if leval_rows(hf_root, subset, cfg)
    ]

    ok = bool(gsm) and bool(short or long_) and bool(conv_rows)
    return {
        "ok": ok,
        "tool_agent_gsm100": len(gsm),
        "loogle_shortdep": len(short),
        "loogle_longdep": len(long_),
        "conversation_pool": len(conv_rows),
        "conversation_sources": conv_sources,
        "hf_root": str(hf_root),
    }


def require_hf_data(hf_root: Path) -> None:
    status = check_hf_data(hf_root)
    missing: list[str] = []
    if status["tool_agent_gsm100"] == 0:
        missing.append("LEval gsm100 (tool_agent)")
    if status["loogle_shortdep"] == 0 and status["loogle_longdep"] == 0:
        missing.append("LooGLE shortdep_qa or longdep_qa (long_context)")
    if status["conversation_pool"] == 0:
        missing.append("LEval short instructions for conversation (e.g. gsm100)")
    if missing:
        raise HfDataError(
            "Missing HF datasets for real prompts:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + f"\n\n{_DOWNLOAD_HINT}"
        )


def _prepare_prompt(
    prompt: str,
    gold: str,
    task: str,
    max_input_tokens: int,
) -> tuple[str, str, str, int]:
    prompt, actual_in = truncate_prompt_natural(prompt, max_input_tokens)
    return prompt, gold, task, actual_in


def iter_conversation_prompts(
    hf_root: Path,
    rng: random.Random,
    n: int,
    *,
    max_input_tokens: int,
) -> Iterator[tuple[str, str, str, int]]:
    pool = _conversation_pool(hf_root)
    if not pool:
        raise HfDataError(
            "No short LEval instruction rows for conversation. "
            f"Download gsm100 at minimum.\n\n{_DOWNLOAD_HINT}"
        )

    rng.shuffle(pool)
    for i in range(n):
        row = pool[i % len(pool)]
        prompt, gold, task = build_conversation_from_leval(row)
        prompt, gold, task, actual_in = _prepare_prompt(prompt, gold, task, max_input_tokens)
        yield prompt, gold, task, actual_in


def iter_long_context_prompts(
    hf_root: Path,
    rng: random.Random,
    n: int,
    *,
    max_input_tokens: int,
) -> Iterator[tuple[str, str, str, int]]:
    short_rows = loogle_shortdep_rows(hf_root)
    long_rows = loogle_longdep_rows(hf_root)
    pool: list[tuple[dict[str, Any], str]] = []
    for r in short_rows:
        pool.append((r, "shortdep_qa"))
    for r in long_rows:
        pool.append((r, "longdep_qa"))

    if not pool:
        raise HfDataError(f"No LooGLE rows at {hf_root}. {_DOWNLOAD_HINT}")

    by_doc: dict[str, list[tuple[dict[str, Any], str]]] = {}
    for row, task in pool:
        doc_id = str(row.get("doc_id") or id(row))
        by_doc.setdefault(doc_id, []).append((row, task))

    doc_ids = list(by_doc.keys())
    rng.shuffle(doc_ids)
    idx = 0
    for _ in range(n):
        doc_id = doc_ids[idx % len(doc_ids)]
        row, _ = by_doc[doc_id][idx % len(by_doc[doc_id])]
        prompt, gold, task = build_long_context_from_loogle(row)
        prompt, gold, task, actual_in = _prepare_prompt(prompt, gold, task, max_input_tokens)
        yield prompt, gold, task, actual_in
        idx += 1
