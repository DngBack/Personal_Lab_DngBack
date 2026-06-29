"""
Load, compare, and pretty-print benchmark results.

Typical usage
-------------
    from dng_opt.bench.report import BenchReport
    report = BenchReport.from_json_files("results/baseline.json", "results/fused.json")
    report.print()
    report.save_csv("results/comparison.csv")
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runner import BatchResult

try:
    from rich.console import Console
    from rich.table import Table
    _RICH = True
except ImportError:
    _RICH = False


def _fmt(val: float, unit: str = "", decimals: int = 1) -> str:
    if math.isnan(val):
        return "—"
    return f"{val:.{decimals}f}{unit}"


def _delta(a: float, b: float, lower_is_better: bool = True) -> str:
    """Return coloured delta string: green = improvement."""
    if math.isnan(a) or math.isnan(b) or b == 0:
        return ""
    pct = (a - b) / b * 100
    if lower_is_better:
        sign = "-" if a < b else "+"
        improved = a < b
    else:
        sign = "+" if a > b else "-"
        improved = a > b
    colour = "green" if improved else "red"
    return f"[{colour}]{sign}{abs(pct):.1f}%[/{colour}]" if _RICH else f"{sign}{abs(pct):.1f}%"


class BenchReport:
    """Holds results from one or two runs and prints a comparison table."""

    def __init__(
        self,
        baseline: dict[int, "BatchResult"],
        optimized: dict[int, "BatchResult"] | None = None,
        baseline_tag: str = "baseline",
        optimized_tag: str = "fused",
        ttft_slo: float = 4000.0,
        tbt_slo: float = 80.0,
    ) -> None:
        self.baseline = baseline
        self.optimized = optimized
        self.baseline_tag = baseline_tag
        self.optimized_tag = optimized_tag
        self.ttft_slo = ttft_slo
        self.tbt_slo = tbt_slo

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data: dict = {
            "baseline_tag": self.baseline_tag,
            "optimized_tag": self.optimized_tag,
            "baseline": {
                str(bs): {
                    "ttft_p50": r.ttft_p50(),
                    "ttft_p95": r.ttft_p95(),
                    "tbt_p50": r.tbt_p50(),
                    "tbt_p95": r.tbt_p95(),
                    "throughput_tok_s": r.throughput_tok_s(),
                    "success_rate": r.success_rate(),
                    "wall_time_ms": r.wall_time_ms,
                }
                for bs, r in self.baseline.items()
            },
        }
        if self.optimized:
            data["optimized"] = {
                str(bs): {
                    "ttft_p50": r.ttft_p50(),
                    "ttft_p95": r.ttft_p95(),
                    "tbt_p50": r.tbt_p50(),
                    "tbt_p95": r.tbt_p95(),
                    "throughput_tok_s": r.throughput_tok_s(),
                    "success_rate": r.success_rate(),
                    "wall_time_ms": r.wall_time_ms,
                }
                for bs, r in self.optimized.items()
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved report → {path}")

    @classmethod
    def from_json_files(
        cls,
        baseline_path: str,
        optimized_path: str | None = None,
    ) -> "BenchReport":
        """Load pre-saved JSON reports and return a BenchReport for printing."""

        def _load(path: str) -> dict[int, dict]:
            with open(path) as f:
                raw = json.load(f)
            return {int(bs): v for bs, v in raw.items()}

        baseline_data = _load(baseline_path)
        opt_data = _load(optimized_path) if optimized_path else None
        # Wrap in a simple namespace that mimics BatchResult
        return cls(
            baseline=baseline_data,  # type: ignore[arg-type]
            optimized=opt_data,  # type: ignore[arg-type]
        )

    def save_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        rows: list[dict] = []
        for bs, r in self.baseline.items():
            row = {
                "batch_size": bs,
                "run": self.baseline_tag,
                "ttft_p50_ms": r.ttft_p50() if hasattr(r, "ttft_p50") else r.get("ttft_p50", float("nan")),
                "ttft_p95_ms": r.ttft_p95() if hasattr(r, "ttft_p95") else r.get("ttft_p95", float("nan")),
                "tbt_p50_ms": r.tbt_p50() if hasattr(r, "tbt_p50") else r.get("tbt_p50", float("nan")),
                "tbt_p95_ms": r.tbt_p95() if hasattr(r, "tbt_p95") else r.get("tbt_p95", float("nan")),
                "throughput_tok_s": r.throughput_tok_s() if hasattr(r, "throughput_tok_s") else r.get("throughput_tok_s", 0),
            }
            rows.append(row)
        if self.optimized:
            for bs, r in self.optimized.items():
                row = {
                    "batch_size": bs,
                    "run": self.optimized_tag,
                    "ttft_p50_ms": r.ttft_p50() if hasattr(r, "ttft_p50") else r.get("ttft_p50", float("nan")),
                    "ttft_p95_ms": r.ttft_p95() if hasattr(r, "ttft_p95") else r.get("ttft_p95", float("nan")),
                    "tbt_p50_ms": r.tbt_p50() if hasattr(r, "tbt_p50") else r.get("tbt_p50", float("nan")),
                    "tbt_p95_ms": r.tbt_p95() if hasattr(r, "tbt_p95") else r.get("tbt_p95", float("nan")),
                    "throughput_tok_s": r.throughput_tok_s() if hasattr(r, "throughput_tok_s") else r.get("throughput_tok_s", 0),
                }
                rows.append(row)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV → {path}")

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _get(self, result, attr: str) -> float:
        if hasattr(result, attr):
            v = getattr(result, attr)
            return v() if callable(v) else v
        if isinstance(result, dict):
            return result.get(attr, float("nan"))
        return float("nan")

    def print(self) -> None:  # noqa: A003
        if _RICH:
            self._print_rich()
        else:
            self._print_plain()

    def _print_rich(self) -> None:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        has_opt = self.optimized is not None
        title = (
            f"[bold]{self.baseline_tag}[/bold] vs [bold]{self.optimized_tag}[/bold]"
            if has_opt
            else f"[bold]{self.baseline_tag}[/bold]"
        )
        table = Table(title=title, show_lines=True)
        table.add_column("batch", justify="right")
        table.add_column(f"TTFT-p50 ({self.baseline_tag})", justify="right")
        table.add_column(f"TTFT-p50 ({self.optimized_tag})", justify="right") if has_opt else None
        table.add_column("Δ TTFT-p50", justify="right") if has_opt else None
        table.add_column(f"TBT-p50 ({self.baseline_tag})", justify="right")
        table.add_column(f"TBT-p50 ({self.optimized_tag})", justify="right") if has_opt else None
        table.add_column("Δ TBT-p50", justify="right") if has_opt else None
        table.add_column(f"tok/s ({self.baseline_tag})", justify="right")
        table.add_column(f"tok/s ({self.optimized_tag})", justify="right") if has_opt else None
        table.add_column("Δ tok/s", justify="right") if has_opt else None

        for bs in sorted(self.baseline.keys()):
            br = self.baseline[bs]
            b_ttft = self._get(br, "ttft_p50")
            b_tbt = self._get(br, "tbt_p50")
            b_thr = self._get(br, "throughput_tok_s")

            row = [str(bs), _fmt(b_ttft, "ms"), _fmt(b_tbt, "ms"), _fmt(b_thr, " t/s")]

            if has_opt and bs in self.optimized:
                or_ = self.optimized[bs]
                o_ttft = self._get(or_, "ttft_p50")
                o_tbt = self._get(or_, "tbt_p50")
                o_thr = self._get(or_, "throughput_tok_s")
                row = [
                    str(bs),
                    _fmt(b_ttft, "ms"), _fmt(o_ttft, "ms"), _delta(o_ttft, b_ttft),
                    _fmt(b_tbt, "ms"),  _fmt(o_tbt, "ms"),  _delta(o_tbt, b_tbt),
                    _fmt(b_thr, " t/s"), _fmt(o_thr, " t/s"), _delta(o_thr, b_thr, lower_is_better=False),
                ]
            table.add_row(*row)

        console.print(table)

    def _print_plain(self) -> None:
        has_opt = self.optimized is not None
        header = (
            f"{'bs':>4}  {'TTFT-p50':>10}  {'TBT-p50':>10}  {'tok/s':>8}"
        )
        if has_opt:
            header += (
                f"  | {'TTFT-p50-opt':>12}  {'TBT-p50-opt':>11}  {'tok/s-opt':>10}"
                f"  {'Δ-TTFT':>8}  {'Δ-TBT':>8}  {'Δ-tok/s':>8}"
            )
        print(header)
        print("-" * len(header))
        for bs in sorted(self.baseline.keys()):
            br = self.baseline[bs]
            b_ttft = self._get(br, "ttft_p50")
            b_tbt = self._get(br, "tbt_p50")
            b_thr = self._get(br, "throughput_tok_s")
            line = f"{bs:>4}  {_fmt(b_ttft, 'ms'):>10}  {_fmt(b_tbt, 'ms'):>10}  {_fmt(b_thr):>8}"
            if has_opt and bs in self.optimized:
                or_ = self.optimized[bs]
                o_ttft = self._get(or_, "ttft_p50")
                o_tbt = self._get(or_, "tbt_p50")
                o_thr = self._get(or_, "throughput_tok_s")
                dtt = _delta(o_ttft, b_ttft)
                dtb = _delta(o_tbt, b_tbt)
                dth = _delta(o_thr, b_thr, lower_is_better=False)
                line += f"  | {_fmt(o_ttft, 'ms'):>12}  {_fmt(o_tbt, 'ms'):>11}  {_fmt(o_thr):>10}  {dtt:>8}  {dtb:>8}  {dth:>8}"
            print(line)
