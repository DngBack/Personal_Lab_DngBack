from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from ..config import PhaseSpec
from ..domain.report import BenchReport, RequestResult
from ..metrics import evaluate_probe, median_tbt_ms, request_effective
from ..storage import decode_prompt, load_index, load_payload, read_json, read_jsonl, write_json, write_jsonl


class ApiReplayEngine:
    def __init__(
        self,
        suite_dir: Path,
        *,
        base_url: str,
        api_key: str,
        model: str,
        max_requests: int | None = None,
        realtime: bool = True,
        temperature: float = 0.0,
        baseline_probe_mean: float = 0.831,
        max_inflight: int | None = None,
    ) -> None:
        self.suite_dir = suite_dir
        self._index = load_index(suite_dir)
        meta_path = suite_dir / "trace_meta.json"
        if meta_path.exists():
            meta = read_json(meta_path)
            if not meta.get("hf_data_used"):
                raise RuntimeError(
                    f"Suite {suite_dir.name} was not generated with real HF data. "
                    "Regenerate with lab.generate after downloading datasets."
                )

        self.phase_spec = PhaseSpec(
            name=str(self._index.get("phase", "phase2")),
            slo_ttft_ms=int(self._index.get("slo_ttft_ms", 10000)),
            slo_tbt_ms=int(self._index.get("slo_tbt_ms", 200)),
            request_timeout_s=int(self._index.get("request_timeout_s", 300)),
        )
        self.base_url = base_url.rstrip("/") + "/v1/chat/completions"
        self.api_key = api_key
        self.model = model
        self.max_requests = max_requests
        self.realtime = realtime
        self.temperature = temperature
        self.baseline_probe_mean = baseline_probe_mean
        self.max_inflight = max_inflight
        self._probes = self._load_probes()

    def _load_probes(self) -> dict[str, dict[str, Any]]:
        probes: dict[str, dict[str, Any]] = {}
        for pr in read_jsonl(self.suite_dir / "probes.jsonl"):
            probes[pr["substitute_for_request_id"]] = pr
        return probes

    async def run(self) -> BenchReport:
        report = BenchReport(
            phase=self.phase_spec.name,
            scenario=str(self._index.get("suite", self.suite_dir.name)),
        )
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
                    res, ps, pd, err = await self._run_one(client, rid)
            else:
                res, ps, pd, err = await self._run_one(client, rid)
            self._collect(report, res, ps, pd, err)

    async def _run_realtime(
        self,
        client: httpx.AsyncClient,
        order: list[str],
        report: BenchReport,
        sem: asyncio.Semaphore | None,
    ) -> None:
        groups: dict[int, list[str]] = defaultdict(list)
        for rid in order:
            ts = int(self._index["entries"][rid]["timestamp"])
            groups[ts].append(rid)

        prev_t = 0
        for t in sorted(groups.keys()):
            wait_ms = max(0, t - prev_t)
            if wait_ms > 0:
                await asyncio.sleep(wait_ms / 1000.0)
            prev_t = t

            async def _bounded(rid: str) -> tuple[
                RequestResult, dict[str, float] | None, dict[str, Any] | None, str | None
            ]:
                if sem:
                    async with sem:
                        return await self._run_one(client, rid)
                return await self._run_one(client, rid)

            outcomes = await asyncio.gather(*[_bounded(rid) for rid in groups[t]])
            for res, ps, pd, err in outcomes:
                self._collect(report, res, ps, pd, err)

    @staticmethod
    def _collect(
        report: BenchReport,
        res: RequestResult,
        probe_score: dict[str, float] | None,
        probe_detail: dict[str, Any] | None,
        err: str | None,
    ) -> None:
        if err:
            report.errors.append(err)
        if probe_score:
            report.probe_scores.append(probe_score)
        if probe_detail:
            report.probe_details.append(probe_detail)
        report.results.append(res)

    async def _run_one(
        self,
        client: httpx.AsyncClient,
        rid: str,
    ) -> tuple[RequestResult, dict[str, float] | None, dict[str, Any] | None, str | None]:
        ent = self._index["entries"][rid]
        payload = load_payload(self.suite_dir, ent["payload"])
        prompt = decode_prompt(payload)
        max_tokens = int(payload.get("output_length") or payload.get("max_tokens") or 128)

        result = RequestResult(
            request_id=rid,
            workload_type=ent["workload_type"],
            is_warmup=ent["is_warmup"],
            is_probe_slot=ent.get("is_probe_slot", False),
            input_length=int(ent.get("input_length") or payload.get("input_length") or 0),
            scheduled_timestamp_ms=int(ent["timestamp"]),
        )

        err_msg: str | None = None
        probe_score: dict[str, float] | None = None
        probe_detail: dict[str, Any] | None = None

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
            pr = self._probes[rid]
            ref = str(pr.get("reference_answer") or "")
            probe_score = evaluate_probe(
                result.completion,
                ref,
                str(pr.get("evaluation_method") or "f1_em"),
            )
            probe_detail = {
                "request_id": rid,
                "workload_type": result.workload_type,
                "evaluation_method": pr.get("evaluation_method"),
                "reference_preview": ref[:120],
                "completion_preview": result.completion[:200],
                "scores": probe_score,
            }

        return result, probe_score, probe_detail, err_msg

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
            "temperature": self.temperature,
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
    suite_dir: Path,
    spec: PhaseSpec,
    baseline_mean: float = 0.831,
) -> Path:
    body = report.to_dict(spec, baseline_mean)
    out = suite_dir / "bench_report.json"
    write_json(out, body)
    rows = []
    probe_by_rid = {p["request_id"]: p for p in report.probe_details}
    for r in report.results:
        row = asdict(r)
        row.pop("completion", None)
        if r.request_id in probe_by_rid:
            pd = probe_by_rid[r.request_id]
            row["probe_f1"] = pd.get("scores", {}).get("f1")
            row["probe_em"] = pd.get("scores", {}).get("em")
            row["reference_preview"] = pd.get("reference_preview")
            row["completion_preview"] = pd.get("completion_preview")
        rows.append(row)
    write_jsonl(suite_dir / "request_metrics.jsonl", rows)
    if report.probe_details:
        write_jsonl(suite_dir / "probe_details.jsonl", report.probe_details)
    return out
