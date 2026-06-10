#!/usr/bin/env python3
"""Chạy các kịch bản extraction (schema_align) qua vLLM OpenAI API.

Yêu cầu: vLLM đang serve (mặc định http://localhost:8000).

    vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000

Chuẩn bị dữ liệu trước:

    python chandra4layout/bench/prepare_data.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from schema_align_llm import (  # noqa: E402
    build_user_prompt,
    compact_blocks,
    extract_json_obj,
    load_fewshots,
    reconcile_keys,
    render_fewshots_from_file,
)

_DATA_DIR = _HERE / "data"
_BLOCKS_DIR = _DATA_DIR / "blocks"
_LAYOUT_JSON = _ROOT / "data/samples/GIAY_GUI_TIEN_TIET_KIEM/layout _GIAY_GUI_TIEN_TIET_KIEM.json"
_SYSTEM_PROMPT = _ROOT / "prompts/schema_align_llm_system.txt"
_FEWSHOTS = _ROOT / "data/samples/schema_align_fewshots.json"
_LARGE_CHANDRA_PROMPT = _ROOT / "prompts/giay_gui_tien_tiet_kiem.txt"
_SCENARIOS = _HERE / "scenarios.yaml"
_DEFAULT_TEXT_MAX = 320


def _collect_schema_names(layout_root: dict | list) -> list[str]:
    import unicodedata

    def _fold(s: str) -> str:
        _vn = str.maketrans("đĐơƠưƯ", "dDoOuU")
        s = s.translate(_vn)
        s = unicodedata.normalize("NFKD", s.lower())
        return "".join(c for c in s if ("a" <= c <= "z") or c.isdigit())

    out: list[str] = []

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            n = o.get("name")
            if isinstance(n, str) and n.strip():
                out.append(n.strip())
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    walk(layout_root)
    seen: set[str] = set()
    uniq = []
    for n in out:
        k = _fold(n)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(n)
    return uniq


def _load_block_files(pages: list[str] | str) -> list[dict]:
    if pages == "all":
        files = sorted(_BLOCKS_DIR.glob("test_*_blocks.json"))
    else:
        files = [_BLOCKS_DIR / f"{p}_blocks.json" for p in pages]
    out = []
    for f in files:
        if not f.is_file():
            raise FileNotFoundError(f"Thiếu blocks cache: {f} — chạy prepare_data.py trước.")
        out.append(json.loads(f.read_text(encoding="utf-8")))
    return out


def _build_messages(
  *,
  schema_names: list[str],
  system_text: str,
  few_txt: str,
  blocks_list: list[list[dict]],
  mode: str,
  large_prompt: bool,
  compact_text_max: int = _DEFAULT_TEXT_MAX,
) -> list[dict[str, str]]:
    if mode == "mega_concat":
        merged = []
        for i, blks in enumerate(blocks_list):
            for b in blks:
                bb = dict(b)
                bb["_page"] = i
                merged.append(bb)
        user = build_user_prompt(
            schema_names, compact_blocks(merged, text_max=compact_text_max), few_txt,
        )
    elif mode == "single_html" or large_prompt:
        page_data = _load_block_files(["test_7"])[0]
        html_path = _ROOT / "results/giay_gui_tien_tiet_kiem_direct/test_7_page01_raw.html"
        chandra_html = html_path.read_text(encoding="utf-8") if html_path.is_file() else ""
        chandra_prompt = _LARGE_CHANDRA_PROMPT.read_text(encoding="utf-8")
        schema_blob = json.dumps(schema_names, ensure_ascii=False)
        user = (
            f"{few_txt.strip()}\n\n"
            f"## Prompt OCR gốc (tham khảo layout):\n{chandra_prompt}\n\n"
            f"## schema_fields:\n```json\n{schema_blob}\n```\n\n"
            f"## chandra_html (đầu ra OCR đầy đủ):\n```html\n{chandra_html}\n```\n\n"
            "Trả về **một** JSON object map schema_field → bbox [x0,y0,x1,y1] hoặc null. "
            "Bắt đầu ngay JSON:"
        )
    else:
        user = build_user_prompt(
            schema_names,
            compact_blocks(blocks_list[0], text_max=compact_text_max),
            few_txt,
        )

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user},
    ]


@dataclass
class RequestMetric:
    scenario_id: str
    request_idx: int
    page: str
    ok: bool
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    n_schema_hit: int = 0
    n_schema_total: int = 0
    error: str = ""


@dataclass
class ScenarioReport:
    scenario_id: str
    title: str
    n_requests: int
    ok_count: int = 0
    latency_s: list[float] = field(default_factory=list)
    prompt_tokens: list[int] = field(default_factory=list)
    completion_tokens: list[int] = field(default_factory=list)
    request_metrics: list[RequestMetric] = field(default_factory=list)
    wall_time_s: float = 0.0

    def summarize(self, *, technique: str = "", concurrency: int = 1) -> dict[str, Any]:
        lat = self.latency_s
        ok_rate = self.ok_count / self.n_requests if self.n_requests else 0.0
        throughput = self.n_requests / self.wall_time_s if self.wall_time_s > 0 else 0.0
        return {
            "scenario_id": self.scenario_id,
            "title": self.title,
            "technique": technique,
            "concurrency": concurrency,
            "n_requests": self.n_requests,
            "ok_count": self.ok_count,
            "ok_rate": round(ok_rate, 3),
            "wall_time_s": round(self.wall_time_s, 3),
            "throughput_docs_per_s": round(throughput, 4),
            "latency_per_doc_amortized_s": round(self.wall_time_s / self.n_requests, 3) if self.n_requests else None,
            "latency_s": {
                "mean": round(statistics.mean(lat), 3) if lat else None,
                "p50": round(statistics.median(lat), 3) if lat else None,
                "max": round(max(lat), 3) if lat else None,
            },
            "prompt_tokens": {
                "mean": round(statistics.mean(self.prompt_tokens), 1) if self.prompt_tokens else None,
                "max": max(self.prompt_tokens) if self.prompt_tokens else None,
            },
            "completion_tokens": {
                "mean": round(statistics.mean(self.completion_tokens), 1) if self.completion_tokens else None,
            },
        }


async def _one_request(
    client: httpx.AsyncClient,
    *,
    url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    scenario_id: str,
    request_idx: int,
    page: str,
    schema_names: list[str],
) -> RequestMetric:
    t0 = time.perf_counter()
    metric = RequestMetric(
        scenario_id=scenario_id,
        request_idx=request_idx,
        page=page,
        ok=False,
        latency_s=0.0,
        n_schema_total=len(schema_names),
    )
    try:
        resp = await client.post(
            url,
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
            timeout=600.0,
        )
        metric.latency_s = time.perf_counter() - t0
        if resp.status_code != 200:
            metric.error = f"HTTP {resp.status_code}: {resp.text[:500]}"
            return metric

        data = resp.json()
        usage = data.get("usage") or {}
        metric.prompt_tokens = int(usage.get("prompt_tokens", 0))
        metric.completion_tokens = int(usage.get("completion_tokens", 0))
        metric.total_tokens = int(usage.get("total_tokens", 0))

        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        obj = extract_json_obj(content)
        if obj is not None:
            rec = reconcile_keys(obj, schema_names)
            metric.n_schema_hit = sum(1 for v in rec.values() if v is not None)
            metric.ok = True
        else:
            metric.error = "JSON parse failed"
    except Exception as exc:
        metric.latency_s = time.perf_counter() - t0
        metric.error = str(exc)
    return metric


async def run_scenario(
    sc: dict,
    *,
    base_url: str,
    api_key: str,
    schema_names: list[str],
    system_text: str,
    few_txt: str,
    default_max_tokens: int,
    default_temperature: float,
    default_model: str,
    default_text_max: int,
) -> ScenarioReport:
    sc_id = sc["id"]
    pages_spec = sc.get("pages", "all")
    mode = sc.get("mode", "single")
    large = bool(sc.get("large_prompt", False))
    concurrency = int(sc.get("concurrency", 1))
    compact_text_max = int(sc.get("compact_text_max", default_text_max))

    report = ScenarioReport(
        scenario_id=sc_id,
        title=sc.get("title", sc_id),
        n_requests=1,
    )

    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    model = sc.get("model") or default_model
    max_tokens = int(sc.get("max_new_tokens") or default_max_tokens)
    temperature = float(sc.get("temperature", default_temperature))

    t_wall0 = time.perf_counter()

    async with httpx.AsyncClient(headers=headers) as client:
        if mode == "mega_concat":
            all_data = _load_block_files("all")
            blocks_list = [d["blocks"] for d in all_data]
            messages = _build_messages(
                schema_names=schema_names,
                system_text=system_text,
                few_txt=few_txt,
                blocks_list=blocks_list,
                mode=mode,
                large_prompt=large,
                compact_text_max=compact_text_max,
            )
            report.n_requests = 1
            m = await _one_request(
                client, url=url, model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
                scenario_id=sc_id, request_idx=0, page="all_8",
                schema_names=schema_names,
            )
            report.request_metrics.append(m)
            if m.ok:
                report.ok_count += 1
            report.latency_s.append(m.latency_s)
            report.prompt_tokens.append(m.prompt_tokens)
            report.completion_tokens.append(m.completion_tokens)

        elif pages_spec == "all":
            all_data = _load_block_files("all")
            report.n_requests = len(all_data)
            sem = asyncio.Semaphore(concurrency)

            async def _run_one(idx: int, page_data: dict) -> RequestMetric:
                async with sem:
                    stem = Path(page_data["pdf"]).stem
                    messages = _build_messages(
                        schema_names=schema_names,
                        system_text=system_text,
                        few_txt=few_txt,
                        blocks_list=[page_data["blocks"]],
                        mode="single",
                        large_prompt=large,
                        compact_text_max=compact_text_max,
                    )
                    return await _one_request(
                        client, url=url, model=model, messages=messages,
                        max_tokens=max_tokens, temperature=temperature,
                        scenario_id=sc_id, request_idx=idx, page=stem,
                        schema_names=schema_names,
                    )

            tasks = [_run_one(i, d) for i, d in enumerate(all_data)]
            results = await asyncio.gather(*tasks)
            for m in results:
                report.request_metrics.append(m)
                if m.ok:
                    report.ok_count += 1
                report.latency_s.append(m.latency_s)
                report.prompt_tokens.append(m.prompt_tokens)
                report.completion_tokens.append(m.completion_tokens)

        else:
            page_data = _load_block_files(pages_spec)[0]
            stem = Path(page_data["pdf"]).stem
            messages = _build_messages(
                schema_names=schema_names,
                system_text=system_text,
                few_txt=few_txt,
                blocks_list=[page_data["blocks"]],
                mode=mode,
                large_prompt=large,
                compact_text_max=compact_text_max,
            )
            report.n_requests = 1
            m = await _one_request(
                client, url=url, model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
                scenario_id=sc_id, request_idx=0, page=stem,
                schema_names=schema_names,
            )
            report.request_metrics.append(m)
            if m.ok:
                report.ok_count += 1
            report.latency_s.append(m.latency_s)
            report.prompt_tokens.append(m.prompt_tokens)
            report.completion_tokens.append(m.completion_tokens)

    report.wall_time_s = time.perf_counter() - t_wall0
    return report


async def main_async(args: argparse.Namespace) -> int:
    if not (_BLOCKS_DIR / "test_7_blocks.json").is_file():
        print("Chưa có dữ liệu bench. Chạy: python chandra4layout/bench/prepare_data.py", file=sys.stderr)
        return 1

    # Health check
    health_url = args.base_url.rstrip("/") + "/health"
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(health_url, timeout=5.0)
            if r.status_code != 200:
                print(f"vLLM health check failed: {r.status_code}", file=sys.stderr)
                return 1
    except httpx.HTTPError as exc:
        print(f"Không kết nối được vLLM tại {args.base_url}: {exc}", file=sys.stderr)
        print("Khởi động: vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000", file=sys.stderr)
        return 1

    scenarios_path = Path(args.scenarios_file)
    cfg = yaml.safe_load(scenarios_path.read_text(encoding="utf-8"))
    scenarios = cfg["scenarios"]
    if args.only:
        scenarios = [s for s in scenarios if s["id"] in args.only]

    layout = json.loads(_LAYOUT_JSON.read_text(encoding="utf-8"))
    schema_names = _collect_schema_names(layout)
    system_text = _SYSTEM_PROMPT.read_text(encoding="utf-8")
    few_txt = render_fewshots_from_file(load_fewshots(_FEWSHOTS))

    out_dir = Path(args.output_dir)
    if args.tag:
        out_dir = out_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    all_reports: list[dict] = []
    print(
        f"Schema fields: {len(schema_names)}  |  scenarios: {len(scenarios)}  |  tag: {args.tag or 'default'}",
        flush=True,
    )

    for sc in scenarios:
        print(f"\n=== {sc['id']}: {sc.get('title', '')} ===", flush=True)
        report = await run_scenario(
            sc,
            base_url=args.base_url,
            api_key=args.api_key,
            schema_names=schema_names,
            system_text=system_text,
            few_txt=few_txt,
            default_max_tokens=int(cfg.get("max_new_tokens", 4096)),
            default_temperature=float(cfg.get("temperature", 0.0)),
            default_model=cfg.get("model", "Qwen/Qwen2.5-3B-Instruct"),
            default_text_max=int(cfg.get("compact_text_max", _DEFAULT_TEXT_MAX)),
        )
        summary = report.summarize(
            technique=sc.get("technique", ""),
            concurrency=int(sc.get("concurrency", 1)),
        )
        all_reports.append(summary)
        print(
            f"  ok={summary['ok_count']}/{summary['n_requests']} ({summary['ok_rate']:.0%})  "
            f"wall={summary['wall_time_s']}s  "
            f"thr={summary['throughput_docs_per_s']:.3f} doc/s  "
            f"lat_mean={summary['latency_s']['mean']}s  "
            f"prompt_tok_mean={summary['prompt_tokens']['mean']}",
            flush=True,
        )
        detail_path = out_dir / f"{sc['id']}_metrics.json"
        detail_path.write_text(
            json.dumps({
                "summary": summary,
                "requests": [asdict(m) for m in report.request_metrics],
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    bench_report = {
        "tag": args.tag or "default",
        "vllm_profile": args.vllm_profile,
        "scenarios_file": str(scenarios_path),
        "base_url": args.base_url,
        "model": cfg.get("model"),
        "max_new_tokens": cfg.get("max_new_tokens"),
        "n_schema_fields": len(schema_names),
        "scenarios": all_reports,
    }
    report_path = out_dir / "bench_report.json"
    report_path.write_text(json.dumps(bench_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {report_path}", flush=True)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark extraction schema_align via vLLM.")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--output-dir", type=Path, default=_HERE / "results")
    p.add_argument("--scenarios-file", type=Path, default=_SCENARIOS)
    p.add_argument("--tag", default="", help="Subfolder kết quả (vd. round2_plain)")
    p.add_argument("--vllm-profile", default="plain", help="Ghi nhận profile vLLM: plain | optimized")
    p.add_argument("--only", nargs="*", help="Chỉ chạy scenario id (vd. s01_single_baseline)")
    args = p.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
