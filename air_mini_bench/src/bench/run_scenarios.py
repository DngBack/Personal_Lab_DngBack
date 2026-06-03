from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from .config import DEFAULT_OUTPUT, PHASES
from .replay import replay_phase, save_report
from .scenarios import PRIORITY_SUITES_BY_PHASE, SUITES_BY_PHASE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run all scenario suites for a phase (each suite = own trace + arrivals)"
    )
    parser.add_argument("--phase", choices=["phase1", "phase2"], required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--all-suites",
        action="store_true",
        help="Include extended suites beyond priority 6/6",
    )
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
        default=os.environ.get("AIR_MINI_BENCH_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
    )
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-inflight", type=int, default=None)
    parser.add_argument("--baseline-mean", type=float, default=0.831)
    args = parser.parse_args(argv)

    specs = (
        SUITES_BY_PHASE[args.phase]
        if args.all_suites
        else PRIORITY_SUITES_BY_PHASE[args.phase]
    )
    phase_root = args.output_dir / args.phase
    summary: dict = {"phase": args.phase, "suites": {}}

    for spec in specs:
        suite_dir = phase_root / spec.name
        if not (suite_dir / "index.json").exists():
            print(f"Skip {spec.name}: not generated")
            continue
        print(f"\n=== {args.phase}/{spec.name} ===")
        report = asyncio.run(
            replay_phase(
                suite_dir,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                dry_run=args.dry_run,
                max_requests=args.max_requests,
                realtime=True,
                scenario=spec.name,
                baseline_probe_mean=args.baseline_mean,
                max_inflight=args.max_inflight,
            )
        )
        out = save_report(report, suite_dir, args.baseline_mean)
        body = report.to_dict(PHASES[args.phase], args.baseline_mean)
        summary["suites"][spec.name] = {
            "erc_percent": body["erc"]["erc_percent"],
            "erc_by_workload": body["erc"]["by_workload"],
            "wall_time_s": body["wall_time_s"],
            "ttft_p50": body["latency"]["ttft_ms"].get("p50"),
            "report": str(out),
        }
        print(json.dumps(body["erc"], indent=2))

    path = phase_root / "runs" / "scenario_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
