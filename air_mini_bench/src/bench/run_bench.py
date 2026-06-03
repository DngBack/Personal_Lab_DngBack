from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from .config import BENCH_ROOT, DEFAULT_OUTPUT, PHASES
from .replay import replay_phase, save_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay mini trace: decode payloads from index.json and call LLM API"
    )
    parser.add_argument("--phase", choices=["phase1", "phase2"], required=True)
    parser.add_argument(
        "--suite",
        default=None,
        help="Suite dir under output/<phase>/ (default: p1_steady / steady_poisson)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("AIR_MINI_BENCH_BASE_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("AIR_MINI_BENCH_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("AIR_MINI_BENCH_MODEL", "gpt-oss-20b"),
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip HTTP; simulate metrics")
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--no-realtime", action="store_true", help="Fire requests back-to-back")
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=None,
        help="Limit concurrent requests when timestamps overlap",
    )
    parser.add_argument("--baseline-mean", type=float, default=0.831)
    args = parser.parse_args(argv)

    default_suite = "p1_steady" if args.phase == "phase1" else "steady_poisson"
    suite = args.suite or default_suite
    phase_dir = args.output_dir / args.phase / suite
    index_path = phase_dir / "index.json"
    if not index_path.exists():
        raise SystemExit(
            f"Missing {index_path}. Run: python -m bench.generate --phase {args.phase}"
        )

    if args.phase == "phase2" and args.model == "gpt-oss-20b":
        print("Note: Phase 2 contest uses gpt-oss-120b; override with --model if needed.")

    report = asyncio.run(
        replay_phase(
            phase_dir,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            dry_run=args.dry_run,
            max_requests=args.max_requests,
            realtime=not args.no_realtime,
            scenario=suite,
            baseline_probe_mean=args.baseline_mean,
            max_inflight=args.max_inflight,
        )
    )
    out = save_report(report, phase_dir, args.baseline_mean)
    print(json.dumps(report.to_dict(PHASES[args.phase], args.baseline_mean), indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
