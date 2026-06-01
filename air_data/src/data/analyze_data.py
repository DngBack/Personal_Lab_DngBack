"""Summarize downloaded Hugging Face datasets for contest / benchmark prep."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

_MODULE_DIR = Path(__file__).resolve().parent
_AIR_DATA_ROOT = _MODULE_DIR.parents[1]
_DEFAULT_HF_ROOT = _AIR_DATA_ROOT / "data" / "hf"


def _char_stats(text: str) -> dict[str, float]:
    words = text.split()
    return {
        "chars": len(text),
        "words": len(words),
        "chars_per_word": round(len(text) / max(len(words), 1), 2),
    }


def _analyze_leval_row(row: dict[str, Any]) -> dict[str, Any]:
    instructions = row.get("instructions") or []
    outputs = row.get("outputs") or []
    inp = row.get("input") or ""
    inst_text = "\n".join(instructions) if isinstance(instructions, list) else str(instructions)
    out_text = "\n".join(outputs) if isinstance(outputs, list) else str(outputs)
    return {
        "input": _char_stats(str(inp)),
        "instructions": _char_stats(inst_text),
        "outputs": _char_stats(out_text),
        "evaluation": row.get("evaluation"),
        "source": row.get("source"),
    }


def _analyze_loogle_row(row: dict[str, Any]) -> dict[str, Any]:
    ctx = str(row.get("context") or "")
    q = str(row.get("question") or "")
    ans = str(row.get("answer") or "")
    ev = row.get("evidence")
    ev_text = json.dumps(ev, ensure_ascii=False) if ev is not None else ""
    return {
        "task": row.get("task"),
        "context": _char_stats(ctx),
        "question": _char_stats(q),
        "answer": _char_stats(ans),
        "evidence": _char_stats(ev_text),
        "title": row.get("title"),
    }


def _mean_field(rows: list[dict[str, Any]], *keys: str) -> dict[str, float]:
    totals: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        node = row
        for key in keys[:-1]:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        leaf = keys[-1]
        if isinstance(node, dict) and leaf in node:
            totals[leaf].append(float(node[leaf]))
    return {k: round(sum(v) / len(v), 1) for k, v in totals.items() if v}


def _load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def analyze_leval(hf_root: Path) -> dict[str, Any]:
    base = hf_root / "L4NLP__LEval"
    if not base.exists():
        return {"status": "missing", "path": str(base)}

    subsets: dict[str, Any] = {}
    for jsonl in sorted(base.rglob("*.jsonl")):
        rel = jsonl.relative_to(base)
        rows = _load_jsonl(jsonl)
        per_row = [_analyze_leval_row(r) for r in rows]
        subsets[str(rel)] = {
            "num_rows": len(rows),
            "columns": list(rows[0].keys()) if rows else [],
            "evaluation_types": sorted({r.get("evaluation") for r in rows if r.get("evaluation")}),
            "avg_lengths": {
                "input_chars": _mean_field(per_row, "input", "chars").get("chars", 0),
                "instruction_chars": _mean_field(per_row, "instructions", "chars").get("chars", 0),
                "output_chars": _mean_field(per_row, "outputs", "chars").get("chars", 0),
            },
            "task_hint": _leval_task_hint(rel.stem, rows[0] if rows else {}),
        }

    exam_rows = sum(v["num_rows"] for k, v in subsets.items() if "Exam" in k)
    gen_rows = sum(v["num_rows"] for k, v in subsets.items() if "Generation" in k)
    return {
        "status": "ok",
        "repo": "L4NLP/LEval",
        "benchmark": "Long-context evaluation (Exam + Generation)",
        "total_rows": exam_rows + gen_rows,
        "exam_rows": exam_rows,
        "generation_rows": gen_rows,
        "subsets": subsets,
        "contest_notes": [
            "Exam: QA / reasoning over very long `input` with few-shot chain-of-thought in prompt.",
            "Generation: summarization & long-form QA; judge with task-specific metrics (ROUGE, etc.).",
            "Typical fields: instructions (question), input (context), outputs (gold), evaluation tag.",
        ],
    }


def _leval_task_hint(name: str, sample: dict[str, Any]) -> str:
    hints = {
        "gsm100": "Math word problems (GSM-style); numeric answer in outputs.",
        "codeU": "Code understanding over long repos.",
        "quality": "Multiple-choice reading comprehension.",
        "coursera": "Course transcript QA.",
        "topic_retrieval_longchat": "Topic retrieval in long chats.",
        "narrative_qa": "Long narrative QA.",
        "gov_report_summ": "Government report summarization.",
        "legal_contract_qa": "Legal contract QA.",
    }
    if name in hints:
        return hints[name]
    ev = sample.get("evaluation")
    if ev == "exam":
        return "Exam-style long-context QA"
    if ev:
        return f"Generation task ({ev})"
    return "Long-context benchmark subset"


def analyze_loogle(hf_root: Path) -> dict[str, Any]:
    base = hf_root / "bigai-nlco__LooGLE"
    if not base.exists():
        return {"status": "missing", "path": str(base)}

    configs: dict[str, Any] = {}
    for jsonl in sorted(base.rglob("*.jsonl")):
        rel = jsonl.relative_to(base)
        cfg = rel.parts[0] if rel.parts else "default"
        rows = _load_jsonl(jsonl)
        per_row = [_analyze_loogle_row(r) for r in rows]
        configs[str(rel)] = {
            "num_rows": len(rows),
            "task": rows[0].get("task") if rows else None,
            "avg_context_chars": _mean_field(per_row, "context", "chars").get("chars", 0),
            "avg_question_chars": _mean_field(per_row, "question", "chars").get("chars", 0),
            "avg_answer_chars": _mean_field(per_row, "answer", "chars").get("chars", 0),
            "unique_docs": len({r.get("doc_id") for r in rows if r.get("doc_id")}),
        }

    total = sum(v["num_rows"] for v in configs.values())
    return {
        "status": "ok",
        "repo": "bigai-nlco/LooGLE",
        "benchmark": "Long dependency benchmark (QA, cloze, summarization)",
        "total_rows": total,
        "configs": configs,
        "contest_notes": [
            "longdep_qa: answer needs info far from question in context.",
            "shortdep_qa: answer near relevant span (easier retrieval).",
            "shortdep_cloze: fill-in-the-blank style.",
            "summarization: long doc -> short summary (gold in answer).",
            "All splits are `test`; metrics often use token-level F1 / ROUGE.",
        ],
    }


def analyze_sharechat(hf_root: Path) -> dict[str, Any]:
    base = hf_root / "tucnguyen__ShareChat"
    if not base.exists():
        return {
            "status": "missing",
            "repo": "tucnguyen/ShareChat",
            "contest_notes": [
                "Gated on Hugging Face — accept license, then: huggingface-cli login",
                "Download: python3 src/data/down_data.py tucnguyen/ShareChat",
            ],
        }
    info_path = base / "dataset_info.json"
    return {
        "status": "ok",
        "repo": "tucnguyen/ShareChat",
        "dataset_info": json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else [],
    }


def build_report(hf_root: Path | None = None) -> dict[str, Any]:
    root = Path(hf_root or _DEFAULT_HF_ROOT).expanduser().resolve()
    return {
        "hf_data_root": str(root),
        "datasets": {
            "LEval": analyze_leval(root),
            "LooGLE": analyze_loogle(root),
            "ShareChat": analyze_sharechat(root),
        },
        "cross_benchmark": {
            "shared_theme": "Long-context understanding (Vietnamese + English benchmarks in repo list)",
            "suggested_prep_order": [
                "1. LEval gsm100 + quality (smaller Exam sets) — format & prompting",
                "2. LooGLE shortdep_qa — baseline retrieval difficulty",
                "3. LooGLE longdep_qa — hard long-range dependency",
                "4. LEval Generation subsets — summarization / long QA",
                "5. ShareChat — after HF access (Vietnamese long-context)",
            ],
        },
    }


def _print_report(report: dict[str, Any]) -> None:
    print(f"HF data root: {report['hf_data_root']}\n")
    for name, data in report["datasets"].items():
        print(f"=== {name} ({data.get('repo', 'n/a')}) ===")
        print(f"Status: {data.get('status')}")
        if data.get("status") == "ok":
            print(f"Benchmark: {data.get('benchmark', 'n/a')}")
            print(f"Total rows: {data.get('total_rows', 'n/a')}")
            if "exam_rows" in data:
                print(f"  Exam: {data['exam_rows']}, Generation: {data['generation_rows']}")
            for note in data.get("contest_notes", []):
                print(f"  • {note}")
            block = data.get("subsets") or data.get("configs") or {}
            for key, meta in sorted(block.items()):
                if isinstance(meta, dict) and "num_rows" in meta:
                    extra = meta.get("avg_context_chars") or meta.get("avg_lengths", {})
                    print(f"  - {key}: {meta['num_rows']} rows {extra}")
        elif data.get("status") == "missing":
            print(f"  Path: {data.get('path', 'n/a')}")
            for note in data.get("contest_notes", []):
                print(f"  • {note}")
        print()

    print("=== Suggested prep order ===")
    for step in report["cross_benchmark"]["suggested_prep_order"]:
        print(f"  {step}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze downloaded HF datasets for contest prep.")
    parser.add_argument(
        "--hf-root",
        default=None,
        help="Root folder with downloaded data (default: air_data/data/hf)",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Write full report JSON to this path",
    )
    args = parser.parse_args()

    report = build_report(args.hf_root)
    _print_report(report)

    if args.json_out:
        out = Path(args.json_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
