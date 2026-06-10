#!/usr/bin/env python3
"""Tổng hợp bench round1 + round2 + optimized → báo cáo cải tiến."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results"

ROUNDS = [
    ("round1_plain", _RESULTS / "bench_report.json", "vLLM plain, max_tokens=4096"),
    ("round2_plain", _RESULTS / "round2_plain" / "bench_report.json", "vLLM plain, max_tokens=8192 + CCU sweep"),
    ("round2_optimized", _RESULTS / "round2_optimized" / "bench_report.json", "vLLM max-num-seqs=4, mem=0.85"),
]


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _find_scenario(report: dict, *ids: str) -> dict | None:
    by_id = {s["scenario_id"]: s for s in report.get("scenarios", [])}
    for i in ids:
        if i in by_id:
            return by_id[i]
    return None


def _delta(new: float | None, old: float | None, *, lower_better: bool = True) -> str:
    if new is None or old is None or old == 0:
        return "—"
    pct = (new - old) / old * 100
    better = (pct < 0) if lower_better else (pct > 0)
    sign = "+" if pct > 0 else ""
    tag = "↓ tốt hơn" if better else "↑ tệ hơn"
    return f"{sign}{pct:.1f}% ({tag})"


def build_report() -> dict[str, Any]:
    loaded = [(name, desc, _load(path)) for name, path, desc in ROUNDS]
    r1 = next((r for n, _, r in loaded if n == "round1_plain" and r), None)
    r2 = next((r for n, _, r in loaded if n == "round2_plain" and r), None)
    ro = next((r for n, _, r in loaded if n == "round2_optimized" and r), None)

    techniques: list[dict[str, Any]] = []

    # 1. max_new_tokens fix
    if r1 and r2:
        s1 = _find_scenario(r1, "s04_eight_parallel")
        s2 = _find_scenario(r2, "r2_ccu8_parallel")
        if s1 and s2:
            techniques.append({
                "technique": "Tăng max_new_tokens (4096 → 8192)",
                "layer": "application",
                "scenario_compare": "s04_ccu8 vs r2_ccu8",
                "before": {"ok_rate": s1["ok_count"] / s1["n_requests"], "lat_mean_s": s1["latency_s"]["mean"]},
                "after": {"ok_rate": s2.get("ok_rate"), "lat_mean_s": s2["latency_s"]["mean"]},
                "impact": "Fix JSON truncate khi CCU cao; reliability tăng, latency có thể tăng nhẹ do decode dài hơn.",
            })

    # 2. compact prompt
    if r2:
        base = _find_scenario(r2, "r2_serial_baseline")
        compact = _find_scenario(r2, "r2_compact_prompt")
        if base and compact:
            techniques.append({
                "technique": "Giảm input tokens (compact text_max 320→120)",
                "layer": "application",
                "before": {"prompt_tokens": 4103, "lat_mean_s": None},
                "after": {
                    "prompt_tokens": compact["prompt_tokens"]["mean"],
                    "lat_mean_s": compact["latency_s"]["mean"],
                },
                "delta_prompt_tokens": _delta(
                    compact["prompt_tokens"]["mean"], 4103, lower_better=True,
                ),
                "delta_latency": _delta(
                    compact["latency_s"]["mean"],
                    _find_scenario(r1, "s01_single_baseline")["latency_s"]["mean"] if r1 else None,
                    lower_better=True,
                ),
                "impact": "Prefill ngắn hơn → latency giảm nếu decode không đổi.",
            })

    # 3. CCU sweep round2
    if r2:
        serial = _find_scenario(r2, "r2_serial_baseline")
        ccu_rows = []
        for ccu in (2, 4, 8):
            sc = _find_scenario(r2, f"r2_ccu{ccu}_parallel")
            if sc and serial:
                ccu_rows.append({
                    "ccu": ccu,
                    "ok_rate": sc.get("ok_rate"),
                    "wall_s": sc["wall_time_s"],
                    "lat_mean_s": sc["latency_s"]["mean"],
                    "throughput_doc_s": sc["throughput_docs_per_s"],
                    "vs_serial_wall": _delta(sc["wall_time_s"], serial["wall_time_s"], lower_better=True),
                })
        techniques.append({
            "technique": "Tăng CCU (continuous batching)",
            "layer": "vLLM scheduling",
            "ccu_sweep": ccu_rows,
            "impact": "Throughput tăng, latency/request tăng; sweet spot thường CCU=2–4.",
        })

    # 4. vLLM max-num-seqs
    if r2 and ro:
        for ccu in (4, 8):
            plain = _find_scenario(r2, f"r2_ccu{ccu}_parallel")
            opt = _find_scenario(ro, f"opt_ccu{ccu}_parallel")
            if plain and opt:
                techniques.append({
                    "technique": f"vLLM max-num-seqs=4 (CCU={ccu})",
                    "layer": "vLLM config",
                    "before_plain": {
                        "ok_rate": plain.get("ok_rate"),
                        "lat_mean_s": plain["latency_s"]["mean"],
                        "wall_s": plain["wall_time_s"],
                    },
                    "after_optimized": {
                        "ok_rate": opt.get("ok_rate"),
                        "lat_mean_s": opt["latency_s"]["mean"],
                        "wall_s": opt["wall_time_s"],
                    },
                    "delta_latency": _delta(opt["latency_s"]["mean"], plain["latency_s"]["mean"], lower_better=True),
                    "delta_wall": _delta(opt["wall_time_s"], plain["wall_time_s"], lower_better=True),
                    "impact": "Giới hạn batch size → latency ổn định hơn dưới burst.",
                })

    # Summary table all scenarios
    all_scenarios: list[dict] = []
    for round_name, desc, report in loaded:
        if not report:
            continue
        for s in report.get("scenarios", []):
            all_scenarios.append({
                "round": round_name,
                "vllm_profile": report.get("vllm_profile", round_name),
                "description": desc,
                **s,
            })

    out = {
        "techniques_analysis": techniques,
        "all_scenarios": all_scenarios,
        "recommendations": [
            {
                "priority": 1,
                "action": "max_new_tokens >= 8192 hoặc structured JSON output",
                "reason": "Tránh truncate 41-field JSON khi CCU > 1",
            },
            {
                "priority": 2,
                "action": "CCU=4 + max-num-seqs=4",
                "reason": "Cân bằng throughput (~2x serial) và latency ổn định",
            },
            {
                "priority": 3,
                "action": "compact_blocks text_max=120",
                "reason": "Giảm ~5–15% prompt tokens mà không đổi logic",
            },
            {
                "priority": 4,
                "action": "Prefix cache: giữ system+fewshots+schema cố định",
                "reason": "8 request burst cùng prefix — đo r2_prefix_burst vs r2_ccu8",
            },
            {
                "priority": 5,
                "action": "Hybrid match thay LLM cho field dễ (rule-based)",
                "reason": "Bỏ hoàn toàn bước 14s/doc nếu accuracy đủ",
            },
        ],
        "rounds_available": [n for n, _, r in loaded if r is not None],
    }
    return out


def main() -> None:
    report = build_report()
    out_path = _RESULTS / "improvement_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out_path}")
    print(f"Rounds: {', '.join(report['rounds_available'])}")

    # Markdown summary to stdout
    print("\n## CCU sweep (round2)")
    for t in report["techniques_analysis"]:
        if t.get("technique", "").startswith("Tăng CCU"):
            for row in t.get("ccu_sweep", []):
                print(
                    f"  CCU={row['ccu']}: ok={row['ok_rate']:.0%}  "
                    f"wall={row['wall_s']}s  lat={row['lat_mean_s']}s  "
                    f"thr={row['throughput_doc_s']:.3f} doc/s"
                )


if __name__ == "__main__":
    main()
