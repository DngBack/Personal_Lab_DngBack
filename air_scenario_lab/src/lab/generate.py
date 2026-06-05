from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DEFAULT_HF_ROOT, DEFAULT_OUTPUT
from .domain.scenario import apply_overrides, load_scenario_yaml
from .services.trace_generator import TraceGenerator


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
        from .domain.scenario import ScenarioSpec

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

    print(f"Generated suite: {suite_dir}")
    print(f"  requests: {meta['total_requests']}")
    print(f"  workloads: {meta['workload_counts']}")
    print(f"  hf_data_used: {meta['hf_data_used']}")
    print(f"  arrival: {meta['arrival_analysis'].get('gap_ms', {})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
