from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..config import PhaseSpec
from ..metrics import aggregate_probe_scores, compute_score, latency_summary, percentile


@dataclass
class RequestResult:
    request_id: str
    workload_type: str
    is_warmup: bool
    is_probe_slot: bool
    input_length: int = 0
    scheduled_timestamp_ms: int = 0
    ttft_ms: float | None = None
    tbt_ms: float | None = None
    output_tokens: int = 0
    latency_ms: float | None = None
    effective: bool = False
    error: str | None = None
    completion: str = ""


@dataclass
class BenchReport:
    phase: str
    scenario: str = "contest"
    results: list[RequestResult] = field(default_factory=list)
    probe_scores: list[dict[str, float]] = field(default_factory=list)
    probe_details: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    wall_time_s: float = 0.0

    def _input_length_summary(self, vals: list[int]) -> dict[str, float | int | None]:
        if not vals:
            return {"count": 0, "min": None, "p50": None, "p95": None, "max": None, "mean": None}
        s = sorted(vals)
        return {
            "count": len(vals),
            "min": min(vals),
            "p50": round(percentile([float(v) for v in vals], 50) or 0, 1),
            "p95": round(percentile([float(v) for v in vals], 95) or 0, 1),
            "max": max(vals),
            "mean": round(sum(vals) / len(vals), 1),
        }

    def _build_debug(self, spec: PhaseSpec, baseline_mean: float, score: dict[str, float]) -> dict[str, Any]:
        warmup = [r for r in self.results if r.is_warmup]
        scored = [r for r in self.results if not r.is_warmup]
        by_wl: dict[str, list[RequestResult]] = defaultdict(list)
        for r in self.results:
            by_wl[r.workload_type].append(r)

        wl_debug: dict[str, Any] = {}
        for wl, rs in sorted(by_wl.items()):
            scored_wl = [r for r in rs if not r.is_warmup]
            ttfts = [r.ttft_ms for r in scored_wl if r.ttft_ms is not None]
            tbts = [r.tbt_ms for r in scored_wl if r.tbt_ms is not None]
            outs = [r.output_tokens for r in scored_wl]
            ins = [r.input_length for r in rs]
            eff = [r for r in scored_wl if r.effective]
            wl_debug[wl] = {
                "n_total": len(rs),
                "n_warmup": sum(1 for r in rs if r.is_warmup),
                "n_scored": len(scored_wl),
                "n_effective": len(eff),
                "erc_percent": round(100.0 * len(eff) / max(len(scored_wl), 1), 2),
                "input_length": self._input_length_summary(ins),
                "output_tokens": self._input_length_summary(outs),
                "ttft_ms": latency_summary(ttfts),
                "tbt_ms": latency_summary(tbts),
                "slo_fail_ttft": sum(
                    1 for r in scored_wl if r.ttft_ms is not None and r.ttft_ms > spec.slo_ttft_ms
                ),
                "slo_fail_tbt": sum(
                    1 for r in scored_wl if r.tbt_ms is not None and r.tbt_ms > spec.slo_tbt_ms
                ),
                "errors": sum(1 for r in rs if r.error),
                "zero_output": sum(1 for r in scored_wl if r.output_tokens < 1),
            }

        probe_by_wl: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for pd in self.probe_details:
            probe_by_wl[str(pd.get("workload_type", "unknown"))].append(pd)

        probe_summary: dict[str, Any] = {}
        for wl, items in sorted(probe_by_wl.items()):
            f1s = [float(x.get("scores", {}).get("f1", 0)) for x in items]
            ems = [float(x.get("scores", {}).get("em", 0)) for x in items]
            probe_summary[wl] = {
                "n": len(items),
                "f1_mean": round(sum(f1s) / max(len(f1s), 1), 4),
                "em_mean": round(sum(ems) / max(len(ems), 1), 4),
            }

        sub_mean = aggregate_probe_scores(self.probe_scores)
        accuracy_drop = max(0.0, baseline_mean - sub_mean)
        score_reason = "pass"
        if score.get("f_delta", 1.0) == 0.0:
            if accuracy_drop >= 0.02:
                score_reason = f"accuracy_gate_fail: drop {accuracy_drop:.2%} >= 2%"
            else:
                score_reason = "f_delta_zero"
        elif score.get("erc", 1.0) < 1.0:
            score_reason = f"erc_below_100%: {score.get('erc', 0):.2%}"

        return {
            "requests": {
                "n_total": len(self.results),
                "n_warmup": len(warmup),
                "n_scored": len(scored),
                "n_errors": len(self.errors),
                "throughput_rps": round(len(self.results) / max(self.wall_time_s, 0.01), 2),
            },
            "by_workload": wl_debug,
            "probes": {
                "n_total": len(self.probe_details),
                "by_workload": probe_summary,
                "worst_5": sorted(
                    self.probe_details,
                    key=lambda x: float(x.get("scores", {}).get("f1", 0)),
                )[:5],
            },
            "score_explanation": {
                "formula": "Score = 100 * ERC * f(accuracy_drop)",
                "erc": score.get("erc"),
                "f_delta": score.get("f_delta"),
                "accuracy_drop": round(accuracy_drop, 4),
                "accuracy_gate_pass": accuracy_drop < 0.02,
                "reason": score_reason,
            },
        }

    def to_dict(self, spec: PhaseSpec, baseline_mean: float = 0.831) -> dict[str, Any]:
        scored = [r for r in self.results if not r.is_warmup]
        effective = [r for r in scored if r.effective]
        erc = len(effective) / max(len(scored), 1)
        sub_mean = aggregate_probe_scores(self.probe_scores)
        accuracy_drop = max(0.0, baseline_mean - sub_mean)
        score = compute_score(erc, accuracy_drop)

        by_wl: dict[str, list[RequestResult]] = defaultdict(list)
        for r in scored:
            by_wl[r.workload_type].append(r)

        def _erc_wl(rs: list[RequestResult]) -> dict[str, Any]:
            eff = [x for x in rs if x.effective]
            return {
                "n": len(rs),
                "n_effective": len(eff),
                "erc": round(len(eff) / max(len(rs), 1), 4),
            }

        ttfts = [r.ttft_ms for r in scored if r.ttft_ms is not None]
        tbts = [r.tbt_ms for r in scored if r.tbt_ms is not None]

        return {
            "phase": self.phase,
            "scenario": self.scenario,
            "wall_time_s": round(self.wall_time_s, 2),
            "slo_ttft_ms": spec.slo_ttft_ms,
            "slo_tbt_ms": spec.slo_tbt_ms,
            "erc": {
                "n_total": len(scored),
                "n_effective": len(effective),
                "erc": round(erc, 4),
                "erc_percent": round(erc * 100, 2),
                "by_workload": {wl: _erc_wl(rs) for wl, rs in sorted(by_wl.items())},
            },
            "latency": {
                "ttft_ms": latency_summary(ttfts),
                "tbt_ms": latency_summary(tbts),
                "slo_fail_ttft": sum(
                    1 for r in scored if r.ttft_ms is not None and r.ttft_ms > spec.slo_ttft_ms
                ),
                "slo_fail_tbt": sum(
                    1 for r in scored if r.tbt_ms is not None and r.tbt_ms > spec.slo_tbt_ms
                ),
            },
            "accuracy": {
                "accuracy_drop": round(accuracy_drop, 4),
                "baseline_mean": baseline_mean,
                "submission_mean": round(sub_mean, 4),
                "n_probes": len(self.probe_scores),
                "pass_gate": accuracy_drop < 0.02,
            },
            "score": score,
            "errors": self.errors[:20],
            "debug": self._build_debug(spec, baseline_mean, score),
        }

    def print_summary(self, spec: PhaseSpec, baseline_mean: float = 0.831) -> None:
        body = self.to_dict(spec, baseline_mean)
        erc = body["erc"]
        lat = body["latency"]
        acc = body["accuracy"]
        score = body["score"]
        dbg = body["debug"]

        print("\n=== Bench Summary ===")
        print(f"Scenario: {body['scenario']}  phase: {body['phase']}  wall: {body['wall_time_s']}s")
        print(f"Requests: total={dbg['requests']['n_total']} warmup={dbg['requests']['n_warmup']} "
              f"scored={dbg['requests']['n_scored']} throughput={dbg['requests']['throughput_rps']} req/s")
        print(f"SLO: TTFT<={body['slo_ttft_ms']}ms  TBT<={body['slo_tbt_ms']}ms")

        print(f"\n--- ERC ---")
        print(f"  Overall: {erc['erc_percent']}% ({erc['n_effective']}/{erc['n_total']})")
        for wl, w in sorted(erc["by_workload"].items()):
            print(f"  {wl}: {round(w['erc']*100,1)}% ({w['n_effective']}/{w['n']})")

        print(f"\n--- Latency (scored) ---")
        ttft = lat["ttft_ms"]
        tbt = lat["tbt_ms"]
        print(f"  TTFT ms: p50={ttft.get('p50')} p90={ttft.get('p90')} p95={ttft.get('p95')} max={ttft.get('max')}")
        print(f"  TBT  ms: p50={tbt.get('p50')} p90={tbt.get('p90')} p95={tbt.get('p95')} max={tbt.get('max')}")
        print(f"  SLO fails: TTFT={lat['slo_fail_ttft']}  TBT={lat['slo_fail_tbt']}")

        print(f"\n--- By workload (debug) ---")
        for wl, w in sorted(dbg["by_workload"].items()):
            inp = w["input_length"]
            out = w["output_tokens"]
            print(
                f"  {wl}: n={w['n_total']} scored={w['n_scored']} erc={w['erc_percent']}% "
                f"in_p50={inp.get('p50')} out_p50={out.get('p50')} "
                f"ttft_p50={w['ttft_ms'].get('p50')} errs={w['errors']}"
            )

        print(f"\n--- Accuracy / Score ---")
        print(f"  Probes: {acc['n_probes']}  submission_f1={acc['submission_mean']}  baseline={acc['baseline_mean']}")
        print(f"  accuracy_drop: {acc['accuracy_drop']:.2%}  pass_gate={acc['pass_gate']}")
        print(f"  Score: {score['score']}  (ERC={score['erc']}, f_delta={score['f_delta']})")
        print(f"  Reason: {dbg['score_explanation']['reason']}")

        if dbg["probes"]["n_total"]:
            print(f"\n--- Probe breakdown ---")
            for wl, ps in sorted(dbg["probes"]["by_workload"].items()):
                print(f"  {wl}: n={ps['n']} f1_mean={ps['f1_mean']} em_mean={ps['em_mean']}")
            print("  Worst probes (preview):")
            for p in dbg["probes"]["worst_5"][:3]:
                print(f"    {p['request_id']} f1={p['scores'].get('f1')} gold={p.get('reference_preview')!r}")
                print(f"      pred={p.get('completion_preview')!r}")

        if body["errors"]:
            print(f"\n--- Errors ({len(body['errors'])}) ---")
            for e in body["errors"][:5]:
                print(f"  {e}")
