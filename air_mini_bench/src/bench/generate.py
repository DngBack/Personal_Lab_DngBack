from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_CONTEXT_SAFETY_TOKENS,
    DEFAULT_HF_ROOT,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_OUTPUT,
    PHASES,
)
from .scenarios import get_suite
from .storage import write_json
from .suite_generate import generate_phase_suites, generate_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate workload-aware scenario suites (per-suite trace + payloads)"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hf-root", type=Path, default=DEFAULT_HF_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", choices=["phase1", "phase2", "all"], default="all")
    parser.add_argument("--suite", default=None)
    parser.add_argument("--all-suites", action="store_true")
    parser.add_argument(
        "--length-profile",
        choices=["heavy", "realistic", "compact", "local"],
        default="heavy",
    )
    parser.add_argument("--max-context-tokens", type=int, default=DEFAULT_MAX_CONTEXT_TOKENS)
    parser.add_argument("--context-safety-tokens", type=int, default=DEFAULT_CONTEXT_SAFETY_TOKENS)
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=None,
        help="Fraction of requests marked warmup (default: 0.05 from config; đề gốc ~0.10)",
    )
    args = parser.parse_args(argv)

    gen_kw = {
        "length_profile": args.length_profile,
        "max_context_tokens": args.max_context_tokens,
        "safety_tokens": args.context_safety_tokens,
        "warmup_ratio": args.warmup_ratio,
    }

    phases = ["phase1", "phase2"] if args.phase == "all" else [args.phase]

    if args.suite:
        for phase_name in phases:
            try:
                spec = get_suite(phase_name, args.suite)
            except KeyError:
                continue
            info = generate_suite(
                spec,
                PHASES[phase_name],
                args.output_dir / phase_name / spec.name,
                args.hf_root,
                args.seed,
                **gen_kw,
            )
            print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0

    manifest: dict[str, Any] = {}
    for phase_name in phases:
        manifest[phase_name] = generate_phase_suites(
            phase_name,
            args.output_dir,
            args.hf_root,
            args.seed,
            priority_only=not args.all_suites,
            **gen_kw,
        )
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
