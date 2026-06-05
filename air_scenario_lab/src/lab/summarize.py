from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DEFAULT_OUTPUT, PHASES
from .domain.report import BenchReport, RequestResult
from .services.api_replay import save_report
from .storage import read_json, read_jsonl


def _load_report_from_artifacts(suite_dir: Path) -> BenchReport | None:
    metrics_path = suite_dir / "request_metrics.jsonl"
    if not metrics_path.exists():
        return None
    index = read_json(suite_dir / "index.json")
    report = BenchReport(
        phase=str(index.get("phase", "phase2")),
        scenario=str(index.get("suite", suite_dir.name)),
    )
    bench = read_json(suite_dir / "bench_report.json") if (suite_dir / "bench_report.json").exists() else {}
    report.wall_time_s = float(bench.get("wall_time_s", 0))
    report.errors = list(bench.get("errors", []))

    probe_details = read_jsonl(suite_dir / "probe_details.jsonl")
    probe_by_rid = {p["request_id"]: p for p in probe_details}

    for row in read_jsonl(metrics_path):
        r = RequestResult(
            request_id=row["request_id"],
            workload_type=row["workload_type"],
            is_warmup=row["is_warmup"],
            is_probe_slot=row.get("is_probe_slot", False),
            input_length=int(row.get("input_length", 0)),
            scheduled_timestamp_ms=int(row.get("scheduled_timestamp_ms", 0)),
            ttft_ms=row.get("ttft_ms"),
            tbt_ms=row.get("tbt_ms"),
            output_tokens=int(row.get("output_tokens", 0)),
            latency_ms=row.get("latency_ms"),
            effective=bool(row.get("effective", False)),
            error=row.get("error"),
        )
        report.results.append(r)
        if r.request_id in probe_by_rid:
            pd = probe_by_rid[r.request_id]
            report.probe_details.append(pd)
            if pd.get("scores"):
                report.probe_scores.append(pd["scores"])
        elif row.get("probe_f1") is not None:
            scores = {"f1": row.get("probe_f1", 0), "em": row.get("probe_em", 0)}
            report.probe_scores.append(scores)
            report.probe_details.append(
                {
                    "request_id": r.request_id,
                    "workload_type": r.workload_type,
                    "reference_preview": row.get("reference_preview", ""),
                    "completion_preview": row.get("completion_preview", ""),
                    "scores": scores,
                }
            )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize existing bench artifacts (no vLLM needed).")
    parser.add_argument("--suite", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-probe-mean", type=float, default=0.831)
    parser.add_argument("--rewrite-report", action="store_true", help="Rewrite bench_report.json with debug section")
    args = parser.parse_args(argv)

    suite_dir = args.output_dir / args.suite
    report = _load_report_from_artifacts(suite_dir)
    if report is None:
        print(f"No request_metrics.jsonl in {suite_dir}", file=sys.stderr)
        return 1

    index = read_json(suite_dir / "index.json")
    phase_name = str(index.get("phase", "phase2"))
    spec = PHASES.get(phase_name, PHASES["phase2"])
    if index.get("slo_ttft_ms"):
        from .config import PhaseSpec

        spec = PhaseSpec(
            name=phase_name,
            slo_ttft_ms=int(index["slo_ttft_ms"]),
            slo_tbt_ms=int(index["slo_tbt_ms"]),
            request_timeout_s=int(index.get("request_timeout_s", 300)),
        )

    if not report.probe_details and (suite_dir / "bench_report.json").exists():
        old = read_json(suite_dir / "bench_report.json")
        acc = old.get("accuracy", {})
        if acc.get("n_probes"):
            print("\n(note: probe_details.jsonl chưa có — accuracy lấy từ bench_report cũ)")
            print(f"  Probes: {acc.get('n_probes')}  submission_f1={acc.get('submission_mean')}  "
                  f"baseline={acc.get('baseline_mean')}")
            print(f"  accuracy_drop: {acc.get('accuracy_drop')}  pass_gate={acc.get('pass_gate')}")
            score = old.get("score", {})
            print(f"  Score: {score.get('score')}  Reason: accuracy_gate_fail (re-run lab.run để có probe_details)")

    report.print_summary(spec, args.baseline_probe_mean)
    if args.rewrite_report and report.probe_details:
        out = save_report(report, suite_dir, spec, args.baseline_probe_mean)
        print(f"\nRewrote: {out}")
    elif args.rewrite_report:
        print("\n(skip rewrite: thiếu probe_details — chạy lại lab.run để cập nhật bench_report đầy đủ)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
