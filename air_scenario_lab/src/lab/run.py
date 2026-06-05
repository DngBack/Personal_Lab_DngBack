from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .config import DEFAULT_OUTPUT
from .services.api_replay import ApiReplayEngine, save_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay scenario suite against vLLM API and compute ERC.")
    parser.add_argument("--suite", type=str, required=True, help="Suite directory name under output/")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--max-inflight", type=int, default=None)
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--baseline-probe-mean", type=float, default=0.831)
    parser.add_argument("--quiet", action="store_true", help="Only print report path")
    args = parser.parse_args(argv)

    suite_dir = args.output_dir / args.suite
    if not (suite_dir / "index.json").exists():
        print(f"Suite not found: {suite_dir}", file=sys.stderr)
        return 1

    engine = ApiReplayEngine(
        suite_dir,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        max_requests=args.max_requests,
        realtime=not args.no_realtime,
        temperature=args.temperature,
        baseline_probe_mean=args.baseline_probe_mean,
        max_inflight=args.max_inflight,
    )

    report = asyncio.run(engine.run())
    out = save_report(report, suite_dir, engine.phase_spec, args.baseline_probe_mean)

    if args.quiet:
        print(out)
    else:
        print(f"Bench report: {out}")
        print(f"Request metrics: {suite_dir / 'request_metrics.jsonl'}")
        if report.probe_details:
            print(f"Probe details: {suite_dir / 'probe_details.jsonl'}")
        report.print_summary(engine.phase_spec, args.baseline_probe_mean)
    return 0


if __name__ == "__main__":
    sys.exit(main())
