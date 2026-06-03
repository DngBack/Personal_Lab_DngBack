"""Replay suite traces against an OpenAI-compatible streaming API."""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from ..config import PHASES, PhaseSpec
from ..domain.report import BenchReport, RequestResult
from ..domain.suite import SuitePaths
from ..metrics import (
    evaluate_probe,
    median_tbt_ms,
    request_effective,
)
from ..storage import decode_prompt, load_index, load_payload, read_json, read_jsonl, write_json, write_jsonl


def load_scenario_timestamps(phase_dir: Path, scenario: str, index: dict) -> dict[str, int]:
    """Load per-request schedule; supports legacy ``scenarios/<name>.json``."""
    path = phase_dir / "scenarios" / f"{scenario}.json"
    if path.exists():
        data = read_json(path)
        return {k: int(v) for k, v in data["timestamps_by_request_id"].items()}
    return {rid: int(index["entries"][rid]["timestamp"]) for rid in index["order"]}


class ReplayEngine:
    """
    Replays ``index.json`` order against ``/v1/chat/completions`` with streaming.

    Realtime mode sleeps between timestamp groups and fires concurrent requests
    within the same millisecond bucket.
    """

    def __init__(
        self,
        suite_dir: Path,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dry_run: bool = False,
        max_requests: int | None = None,
        realtime: bool = True,
        scenario: str = "contest",
        temperature: float = 0.0,
        baseline_probe_mean: float = 0.831,
        max_inflight: int | None = None,
    ) -> None:
        self.paths = SuitePaths(suite_dir)
        self._index = load_index(suite_dir)
        phase_name = self._index.get("phase") or suite_dir.parent.name
        self.phase_spec = PHASES[phase_name]
        self.phase_name = phase_name
        self.base_url = base_url.rstrip("/") + "/v1/chat/completions"
        self.api_key = api_key
        self.model = model
        self.dry_run = dry_run
        self.max_requests = max_requests
        self.realtime = realtime
        self.scenario = scenario
        self.temperature = temperature
        self.baseline_probe_mean = baseline_probe_mean
        self.max_inflight = max_inflight
        self._ts_map = load_scenario_timestamps(suite_dir, scenario, self._index)
        self._probes = self._load_probes()
        self._temp = self._resolve_temperature()

    def _resolve_temperature(self) -> float:
        decoder_path = self.paths.root.parent.parent / "decoder_config.json"
        if decoder_path.is_file():
            return float(read_json(decoder_path).get("temperature", self.temperature))
        return self.temperature

    def _load_probes(self) -> dict[str, dict[str, Any]]:
        probes: dict[str, dict[str, Any]] = {}
        for pr in read_jsonl(self.paths.probes):
            probes[pr["substitute_for_request_id"]] = pr
        return probes

    async def run(self) -> BenchReport:
        """Execute replay and return aggregated report."""
        report = BenchReport(phase=self.phase_name, scenario=self.scenario)
        order = self._index["order"]
        if self.max_requests is not None:
            order = order[: self.max_requests]

        sem = asyncio.Semaphore(self.max_inflight) if self.max_inflight else None

        async with httpx.AsyncClient() as client:
            t_wall0 = time.perf_counter()
            if not self.realtime:
                await self._run_sequential(client, order, report, sem)
            else:
                await self._run_realtime(client, order, report, sem)
            report.wall_time_s = time.perf_counter() - t_wall0

        order_index = {rid: i for i, rid in enumerate(order)}
        report.results.sort(key=lambda r: order_index.get(r.request_id, 10**9))
        return report

    async def _run_sequential(
        self,
        client: httpx.AsyncClient,
        order: list[str],
        report: BenchReport,
        sem: asyncio.Semaphore | None,
    ) -> None:
        for rid in order:
            if sem:
                async with sem:
                    res, ps, err = await self._run_one(client, rid)
            else:
                res, ps, err = await self._run_one(client, rid)
            self._collect(report, res, ps, err)

    async def _run_realtime(
        self,
        client: httpx.AsyncClient,
        order: list[str],
        report: BenchReport,
        sem: asyncio.Semaphore | None,
    ) -> None:
        groups: dict[int, list[str]] = defaultdict(list)
        for rid in order:
            ts = self._ts_map.get(rid, self._index["entries"][rid]["timestamp"])
            groups[int(ts)].append(rid)

        prev_t = 0
        for t in sorted(groups.keys()):
            wait_ms = max(0, t - prev_t)
            if wait_ms > 0:
                await asyncio.sleep(wait_ms / 1000.0)
            prev_t = t

            async def _bounded(rid: str) -> tuple[RequestResult, dict[str, float] | None, str | None]:
                if sem:
                    async with sem:
                        return await self._run_one(client, rid)
                return await self._run_one(client, rid)

            outcomes = await asyncio.gather(*[_bounded(rid) for rid in groups[t]])
            for res, ps, err in outcomes:
                self._collect(report, res, ps, err)

    @staticmethod
    def _collect(
        report: BenchReport,
        res: RequestResult,
        probe_score: dict[str, float] | None,
        err: str | None,
    ) -> None:
        if err:
            report.errors.append(err)
        if probe_score:
            report.probe_scores.append(probe_score)
        report.results.append(res)

    async def _run_one(
        self,
        client: httpx.AsyncClient,
        rid: str,
    ) -> tuple[RequestResult, dict[str, float] | None, str | None]:
        ent = self._index["entries"][rid]
        payload = load_payload(self.paths.root, ent["payload"])
        prompt = decode_prompt(payload)
        max_tokens = int(payload.get("output_length") or payload.get("max_tokens") or 128)

        result = RequestResult(
            request_id=rid,
            workload_type=ent["workload_type"],
            is_warmup=ent["is_warmup"],
            is_probe_slot=ent.get("is_probe_slot", False),
            input_length=int(ent.get("input_length") or payload.get("input_length") or 0),
            scheduled_timestamp_ms=int(self._ts_map.get(rid, ent["timestamp"])),
        )

        err_msg: str | None = None
        probe_score: dict[str, float] | None = None

        if self.dry_run:
            result.output_tokens = max_tokens // 2
            result.ttft_ms = 100.0
            result.tbt_ms = 50.0
            result.latency_ms = 500.0
            result.completion = "[dry-run]"
        else:
            try:
                text, ttft, tbt, ntok, lat = await self._stream_completion(
                    client, prompt, max_tokens
                )
                result.completion = text
                result.ttft_ms = ttft
                result.tbt_ms = tbt
                result.output_tokens = ntok
                result.latency_ms = lat
            except Exception as exc:  # noqa: BLE001
                result.error = str(exc)
                err_msg = f"{rid}: {exc}"

        spec = self.phase_spec
        result.effective = request_effective(
            result.ttft_ms,
            result.tbt_ms,
            result.output_tokens,
            spec.slo_ttft_ms,
            spec.slo_tbt_ms,
        )

        if rid in self._probes:
            if self.dry_run:
                probe_score = {"f1": self.baseline_probe_mean, "em": 1.0}
            else:
                pr = self._probes[rid]
                probe_score = evaluate_probe(
                    result.completion,
                    str(pr.get("reference_answer") or ""),
                    str(pr.get("evaluation_method") or "f1_em"),
                )

        return result, probe_score, err_msg

    async def _stream_completion(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        max_tokens: int,
    ) -> tuple[str, float | None, float | None, int, float]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": self._temp,
            "stream": True,
        }
        t0 = time.perf_counter()
        ttft_ms: float | None = None
        gaps: list[float] = []
        last_t = t0
        chunks: list[str] = []
        token_count = 0
        timeout_s = float(self.phase_spec.request_timeout_s)

        async with client.stream(
            "POST",
            self.base_url,
            headers=headers,
            json=body,
            timeout=timeout_s,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    evt = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = (
                    evt.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content")
                )
                if not delta:
                    continue
                now = time.perf_counter()
                if ttft_ms is None:
                    ttft_ms = (now - t0) * 1000
                else:
                    gaps.append((now - last_t) * 1000)
                last_t = now
                chunks.append(delta)
                token_count += 1

        latency_ms = (time.perf_counter() - t0) * 1000
        return "".join(chunks), ttft_ms, median_tbt_ms(gaps), token_count, latency_ms


def save_report(
    report: BenchReport,
    phase_dir: Path,
    baseline_mean: float = 0.831,
    *,
    run_subdir: str | None = None,
) -> Path:
    """Write ``bench_report.json`` and ``request_metrics.jsonl``."""
    index = load_index(phase_dir)
    phase_name = index.get("phase") or phase_dir.parent.name
    spec = PHASES[phase_name]
    body = report.to_dict(spec, baseline_mean)

    if run_subdir:
        out_dir = phase_dir / "runs" / run_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "bench_report.json"
        metrics_path = out_dir / "request_metrics.jsonl"
    else:
        out = phase_dir / "bench_report.json"
        metrics_path = phase_dir / "request_metrics.jsonl"

    write_json(out, body)
    rows = []
    for r in report.results:
        row = asdict(r)
        row.pop("completion", None)
        rows.append(row)
    write_jsonl(metrics_path, rows)
    return out


async def replay_phase(
    phase_dir: Path,
    *,
    base_url: str,
    api_key: str,
    model: str,
    dry_run: bool = False,
    max_requests: int | None = None,
    realtime: bool = True,
    scenario: str = "contest",
    temperature: float = 0.0,
    baseline_probe_mean: float = 0.831,
    max_inflight: int | None = None,
) -> BenchReport:
    """Backward-compatible functional entry point."""
    engine = ReplayEngine(
        phase_dir,
        base_url=base_url,
        api_key=api_key,
        model=model,
        dry_run=dry_run,
        max_requests=max_requests,
        realtime=realtime,
        scenario=scenario,
        temperature=temperature,
        baseline_probe_mean=baseline_probe_mean,
        max_inflight=max_inflight,
    )
    return await engine.run()
