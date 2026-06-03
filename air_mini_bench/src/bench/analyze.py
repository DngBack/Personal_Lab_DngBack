"""CLI: analyze generated scenario suites."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_OUTPUT, PHASES
from .services.dataset_analyzer import DatasetAnalyzer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze scenario suites")
    parser.add_argument("--phase", choices=["phase1", "phase2", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--suite", default=None, help="Single suite; default all priority suites")
    parser.add_argument("--all-suites", action="store_true")
    parser.add_argument("--run-metrics", type=Path, default=None)
    args = parser.parse_args(argv)

    phases = list(PHASES.keys()) if args.phase == "all" else [args.phase]
    analyzer = DatasetAnalyzer()
    report = analyzer.analyze_phases(
        args.output_dir,
        phases,
        suite_filter=args.suite,
        all_suites=args.all_suites,
        run_metrics=args.run_metrics,
    )

    for phase_name, suites in report.get("phases", {}).items():
        for name in suites:
            out = args.output_dir / phase_name / name / "dataset_analysis.json"
            print(f"Wrote {out}")

    print(f"Wrote {args.output_dir / 'analysis_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
