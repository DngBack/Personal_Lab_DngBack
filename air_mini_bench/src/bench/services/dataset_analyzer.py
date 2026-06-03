"""Offline analysis of generated suites and optional run metrics."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ..arrivals import analyze_timestamps
from ..config import DEFAULT_OUTPUT, PHASES
from ..domain.scenario import PRIORITY_SUITES_BY_PHASE, SUITES_BY_PHASE, ScenarioSpec
from ..domain.suite import SuitePaths
from ..storage import decode_prompt, read_json, read_jsonl, write_json


class DatasetAnalyzer:
    """Computes token/arrival/cache statistics for generated scenario suites."""

    @staticmethod
    def percentile(vals: list[float | int], p: float) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        k = (len(s) - 1) * p / 100.0
        f = int(k)
        c = min(f + 1, len(s) - 1)
        if f == c:
            return float(s[f])
        return float(s[f] + (s[c] - s[f]) * (k - f))

    @classmethod
    def token_stats(cls, vals: list[int | float]) -> dict[str, float]:
        if not vals:
            return {"count": 0, "min": 0, "max": 0, "mean": 0, "p50": 0, "p90": 0, "p95": 0}
        return {
            "count": len(vals),
            "min": float(min(vals)),
            "max": float(max(vals)),
            "mean": round(sum(vals) / len(vals), 2),
            "p50": round(cls.percentile(vals, 50), 2),
            "p90": round(cls.percentile(vals, 90), 2),
            "p95": round(cls.percentile(vals, 95), 2),
        }

    @staticmethod
    def histogram(vals: list[int], edges: list[int]) -> dict[str, int]:
        out: dict[str, int] = {}
        prev = 0
        for hi in edges:
            label = f"{prev}-{hi}"
            out[label] = sum(1 for v in vals if prev < v <= hi)
            prev = hi
        out[f">{edges[-1]}"] = sum(1 for v in vals if v > edges[-1])
        return out

    def analyze_suite(self, suite_dir: Path) -> dict[str, Any]:
        """Analyze one suite directory; writes ``dataset_analysis.json`` if called from CLI."""
        paths = SuitePaths(suite_dir)
        meta = read_json(paths.trace_meta)
        index = read_json(paths.index)
        rows = list(read_jsonl(paths.trace))
        scored = [r for r in rows if not r.get("is_warmup")]

        by_wl: dict[str, list[int]] = defaultdict(list)
        for r in scored:
            by_wl[r["workload_type"]].append(r["input_length"])

        all_in = [r["input_length"] for r in scored]
        ts = [index["entries"][r["request_id"]]["timestamp"] for r in rows]

        return {
            "suite": meta.get("suite", suite_dir.name),
            "phase": meta.get("phase"),
            "scenario_spec": meta.get("scenario_spec"),
            "totals": {
                "requests": len(rows),
                "scored": len(scored),
                "warmup": len(rows) - len(scored),
            },
            "input_tokens": {
                "overall": self.token_stats(all_in),
                "histogram": self.histogram(
                    all_in, [500, 2000, 8000, 12000, 18000, 25000, 100000]
                ),
                "by_workload": {wl: self.token_stats(v) for wl, v in sorted(by_wl.items())},
            },
            "arrival": analyze_timestamps(ts),
            "prefix_cache": self._prefix_cache_analysis(rows, suite_dir, index),
            "meta": meta,
        }

    @staticmethod
    def _prefix_cache_analysis(
        rows: list[dict],
        suite_dir: Path,
        index: dict,
    ) -> dict[str, Any]:
        by_session: dict[str, list[str]] = defaultdict(list)
        for r in rows:
            sid = r.get("cache_session_id")
            if sid:
                by_session[sid].append(r["request_id"])

        multi = {k: v for k, v in by_session.items() if len(v) >= 2}
        verified = 0
        for _sid, rids in multi.items():
            texts = []
            for rid in rids[:3]:
                pl = read_json(suite_dir / index["entries"][rid]["payload"])
                texts.append(decode_prompt(pl))
            if texts and all(t[:500] == texts[0][:500] for t in texts[1:]):
                verified += 1

        return {
            "sessions": len(by_session),
            "sessions_2plus": len(multi),
            "text_prefix_verified_sessions": verified,
            "note": "vLLM prefix cache follows real token prefix, not hash_ids alone.",
        }

    @classmethod
    def analyze_run_metrics(
        cls,
        metrics_path: Path,
        spec_slo: tuple[float, float],
    ) -> dict[str, Any]:
        rows = list(read_jsonl(metrics_path))
        scored = [r for r in rows if not r.get("is_warmup")]
        slo_ttft, slo_tbt = spec_slo

        def _erc(subset: list[dict]) -> dict[str, Any]:
            eff = [r for r in subset if r.get("effective")]
            return {
                "n": len(subset),
                "n_effective": len(eff),
                "erc": round(len(eff) / max(len(subset), 1), 4),
            }

        by_wl: dict[str, list[dict]] = defaultdict(list)
        for r in scored:
            by_wl[r["workload_type"]].append(r)

        ttfts = [r["ttft_ms"] for r in scored if r.get("ttft_ms") is not None]
        tbts = [r["tbt_ms"] for r in scored if r.get("tbt_ms") is not None]

        return {
            "n_scored": len(scored),
            "erc_overall": _erc(scored),
            "erc_by_workload": {wl: _erc(v) for wl, v in sorted(by_wl.items())},
            "ttft_ms": cls.token_stats(ttfts),
            "tbt_ms": cls.token_stats(tbts),
            "errors": sum(1 for r in scored if r.get("error")),
        }

    def analyze_phases(
        self,
        output_dir: Path,
        phases: list[str],
        *,
        suite_filter: str | None = None,
        all_suites: bool = False,
        run_metrics: Path | None = None,
    ) -> dict[str, Any]:
        """Analyze multiple suites and write ``analysis_summary.json``."""
        catalog = {
            s.name: s.description
            for phase_name in phases
            for s in SUITES_BY_PHASE[phase_name]
        }
        report: dict[str, Any] = {"catalog": catalog, "phases": {}}

        for phase_name in phases:
            phase_root = output_dir / phase_name
            phase_specs: tuple[ScenarioSpec, ...] = (
                SUITES_BY_PHASE[phase_name]
                if all_suites
                else PRIORITY_SUITES_BY_PHASE[phase_name]
            )
            if suite_filter:
                phase_specs = tuple(s for s in phase_specs if s.name == suite_filter)
            phase_report: dict[str, Any] = {}
            for spec in phase_specs:
                suite_dir = phase_root / spec.name
                if not SuitePaths(suite_dir).exists():
                    continue
                analysis = self.analyze_suite(suite_dir)
                if run_metrics and run_metrics.exists():
                    ps = PHASES[phase_name]
                    analysis["run"] = self.analyze_run_metrics(
                        run_metrics, (ps.slo_ttft_ms, ps.slo_tbt_ms)
                    )
                out = suite_dir / "dataset_analysis.json"
                write_json(out, analysis)
                phase_report[spec.name] = analysis
            report["phases"][phase_name] = phase_report

        write_json(output_dir / "analysis_summary.json", report)
        return report


def analyze_suite(suite_dir: Path) -> dict[str, Any]:
    """Functional wrapper for backward compatibility."""
    return DatasetAnalyzer().analyze_suite(suite_dir)


def analyze_run_metrics(metrics_path: Path, spec_slo: tuple[float, float]) -> dict[str, Any]:
    return DatasetAnalyzer.analyze_run_metrics(metrics_path, spec_slo)
