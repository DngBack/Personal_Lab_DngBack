#!/usr/bin/env python3
"""Smoke test: HF check → generate → verify prompts → ERC metrics on mock replay."""
from __future__ import annotations

import asyncio
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab.config import DEFAULT_HF_ROOT, DEFAULT_OUTPUT
from lab.domain.scenario import load_scenario_yaml
from lab.metrics import request_effective
from lab.services.api_replay import ApiReplayEngine, save_report
from lab.services.trace_generator import TraceGenerator
from lab.sources.hf_prompts import check_hf_data


def _verify_prompts(suite_dir: Path) -> None:
    for payload_path in sorted((suite_dir / "payloads").glob("r-*.json"))[:5]:
        p = json.loads(payload_path.read_text())
        prompt = base64.b64decode(p["prompt_b64"]).decode()
        if " word word" in prompt:
            raise RuntimeError(f"Filler padding found in {payload_path.name}")
    meta = json.loads((suite_dir / "trace_meta.json").read_text())
    if not meta.get("hf_data_used"):
        raise RuntimeError("hf_data_used must be true")


async def _mock_replay(suite_dir: Path) -> None:
    engine = ApiReplayEngine(
        suite_dir,
        base_url="http://mock",
        api_key="",
        model="mock",
        max_requests=10,
        realtime=False,
    )

    async def fake_stream(_client, prompt, max_tokens):
        await asyncio.sleep(0.01)
        n = min(max_tokens, 8)
        return " ".join(f"tok{i}" for i in range(n)), 50.0, 30.0, n, 200.0

    engine._stream_completion = fake_stream  # type: ignore[method-assign]
    report = await engine.run()
    out = save_report(report, suite_dir, engine.phase_spec)
    body = json.loads(out.read_text())
    assert body["erc"]["n_total"] > 0
    assert body["erc"]["erc_percent"] >= 0
    scored = [r for r in report.results if not r.is_warmup]
    for r in scored:
        expected = request_effective(
            r.ttft_ms, r.tbt_ms, r.output_tokens,
            engine.phase_spec.slo_ttft_ms, engine.phase_spec.slo_tbt_ms,
        )
        assert r.effective == expected, r.request_id
    print(f"  ERC mock replay: {body['erc']['erc_percent']}% -> {out}")


def main() -> int:
    hf_root = DEFAULT_HF_ROOT
    status = check_hf_data(hf_root)
    print("HF check:", "OK" if status["ok"] else "FAIL", status)
    if not status["ok"]:
        return 1

    spec = load_scenario_yaml(ROOT / "configs/scenarios/steady_baseline.yaml")
    from lab.domain.scenario import apply_overrides

    spec = apply_overrides(spec, requests=30)
    suite_dir = DEFAULT_OUTPUT / spec.name
    TraceGenerator(hf_root).generate_suite(spec, suite_dir)
    print(f"Generated: {suite_dir}")

    _verify_prompts(suite_dir)
    print("Prompt verification: OK (no word padding, hf_data_used=true)")

    asyncio.run(_mock_replay(suite_dir))
    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
