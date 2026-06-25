from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

from .config import DEFAULT_HF_ROOT, DEFAULT_OUTPUT
from .domain.scenario import ScenarioSpec, apply_overrides, load_scenario_yaml
from .services.trace_generator import TraceGenerator
from .storage import write_json


def _parse_mix(raw: str | None) -> dict[str, float] | None:
    if not raw:
        return None
    mix: dict[str, float] = {}
    for part in raw.split(","):
        key, val = part.split("=")
        mix[key.strip()] = float(val.strip())
    return mix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate scenario suite with real HF prompts.")
    parser.add_argument("--config", type=Path, help="YAML scenario config")
    parser.add_argument("--name", type=str, help="Suite name (required without --config)")
    parser.add_argument("--phase", type=str, default="phase2", choices=["phase1", "phase2"])
    parser.add_argument("--requests", type=int, help="Override total_requests")
    parser.add_argument("--mix", type=str, help="Override mix, e.g. conversation=0.31,tool_agent=0.59,long_context=0.10")
    parser.add_argument("--arrival", type=str, default="steady_poisson")
    parser.add_argument("--mean-ms", type=float, help="Override arrival mean_ms")
    parser.add_argument("--cache-mode", type=str, choices=["hot", "cold"])
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--probe-slot-ratio", type=float, default=0.08)
    parser.add_argument("--length-profile", type=str, default="realistic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-root", type=Path, default=DEFAULT_HF_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if args.config:
        spec = load_scenario_yaml(args.config)
        spec = apply_overrides(
            spec,
            requests=args.requests,
            mean_ms=args.mean_ms,
            mix=_parse_mix(args.mix),
            seed=args.seed if args.seed != 42 else None,
        )
    else:
        if not args.name:
            parser.error("--name is required when --config is not provided")
        mix = _parse_mix(args.mix) or {
            "conversation": 0.31,
            "tool_agent": 0.59,
            "long_context": 0.10,
        }
        arrival_params: dict = {}
        if args.mean_ms is not None:
            arrival_params["mean_ms"] = args.mean_ms
        elif args.arrival == "steady_poisson":
            arrival_params["mean_ms"] = 50.0

        spec = ScenarioSpec(
            name=args.name,
            phase=args.phase,
            total_requests=args.requests or 500,
            mix=mix,
            arrival=args.arrival,
            arrival_params=arrival_params,
            cache_mode=args.cache_mode,
            warmup_ratio=args.warmup_ratio,
            probe_slot_ratio=args.probe_slot_ratio,
            length_profile=args.length_profile,
            seed=args.seed,
        )

    suite_dir = args.output_dir / spec.name
    gen = TraceGenerator(args.hf_root)
    meta = gen.generate_suite(spec, suite_dir)

    # Write scenario.yaml into suite dir (mirrors admission_crunch structure)
    if args.config:
        shutil.copy2(args.config, suite_dir / "scenario.yaml")
    else:
        _write_scenario_yaml(suite_dir, spec)

    _write_bench_suite(suite_dir, spec, meta)

    print(f"Generated suite: {suite_dir}")
    print(f"  requests: {meta['total_requests']}")
    print(f"  workloads: {meta['workload_counts']}")
    print(f"  hf_data_used: {meta['hf_data_used']}")
    print(f"  arrival: {meta['arrival_analysis'].get('gap_ms', {})}")
    concurrent = meta["arrival_analysis"].get("concurrent_starts", {})
    print(f"  ccu_max: {concurrent.get('max_at_one_timestamp')}  pct_batched: {concurrent.get('pct_batched')}%")
    return 0


def _write_scenario_yaml(suite_dir: Path, spec: ScenarioSpec) -> None:
    import yaml  # type: ignore[import]

    data: dict[str, Any] = {
        "name": spec.name,
        "phase": spec.phase,
        "total_requests": spec.total_requests,
        "mix": spec.mix,
        "arrival": spec.arrival,
        "arrival_params": spec.arrival_params,
        "warmup_ratio": spec.warmup_ratio,
        "probe_slot_ratio": spec.probe_slot_ratio,
        "length_profile": spec.length_profile,
        "slo_ttft_ms": spec.slo_ttft_ms,
        "slo_tbt_ms": spec.slo_tbt_ms,
        "seed": spec.seed,
        "description": spec.description,
    }
    if spec.cache_mode is not None:
        data["cache_mode"] = spec.cache_mode
    (suite_dir / "scenario.yaml").write_text(
        yaml.dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def _write_bench_suite(suite_dir: Path, spec: ScenarioSpec, meta: dict[str, Any]) -> None:
    n_warmup = meta["warmup_requests"]
    body: dict[str, Any] = {
        "name": spec.name,
        "phase": spec.phase,
        "format": "mooncake_trace",
        "description": (spec.description or "").strip(),
        "slo": {
            "ttft_ms": spec.phase_spec().slo_ttft_ms,
            "tbt_ms": spec.phase_spec().slo_tbt_ms,
        },
        "requests": {
            "total": spec.total_requests,
            "warmup": n_warmup,
            "scored": spec.total_requests - n_warmup,
        },
        "workload_mix": spec.mix,
        "arrival": {
            "kind": spec.arrival,
            "params": spec.arrival_params,
            "analysis": meta.get("arrival_analysis", {}),
        },
        "aiperf": {
            "input_file": "trace.jsonl",
            "custom_dataset_type": "mooncake_trace",
            "fixed_schedule": True,
        },
        "scoring": {
            "erc": "N_effective / N_scored (warmup excluded)",
            "score": "100 * ERC * f(accuracy_drop)",
            "note": meta.get("contest_ref", ""),
        },
    }
    write_json(suite_dir / "bench_suite.json", body)


if __name__ == "__main__":
    sys.exit(main())
