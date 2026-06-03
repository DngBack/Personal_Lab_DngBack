"""Build suite artifacts: workloads, prompts, arrivals, probes, and index."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..arrivals import (
    analyze_timestamps,
    build_arrival_timestamps_for_spec,
    build_session_clustered_timestamps,
)
from ..config import CONTEST_REF, LENGTH_PROFILES, PHASES, PhaseSpec
from ..domain.scenario import PRIORITY_SUITES_BY_PHASE, SUITES_BY_PHASE, ScenarioSpec
from ..domain.suite import GenerationConfig, SuitePaths
from ..hf_sources import hf_available, iter_long_context_prompts, iter_tool_agent_prompts
from ..prompt_builder import (
    build_conversation_prompt,
    build_long_context_prompt,
    build_tool_agent_prompt,
)
from ..probes_util import pick_probe_slots, probe_distribution_spec
from ..storage import decode_prompt, encode_prompt, write_json, write_jsonl
from ..tokens import compute_hash_ids
from ..trace_lengths import (
    enforce_prompt_tokens,
    fit_lengths,
    sample_length,
    validate_trace_lengths,
)


class TraceGenerator:
    """
    Creates one scenario suite on disk (trace.jsonl, payloads/, probes, index).

    Prefix-cache hot/cold suites use real shared prompt text per session, not
    hash_ids alone — matching vLLM KV prefix behavior.
    """

    def __init__(self, hf_root: Path, config: GenerationConfig | None = None) -> None:
        self.hf_root = hf_root
        self.config = config or GenerationConfig()

    def generate_suite(
        self,
        spec: ScenarioSpec,
        phase: PhaseSpec,
        suite_dir: Path,
        seed: int,
    ) -> dict[str, Any]:
        """Generate a single suite directory and return trace metadata."""
        rng = random.Random(seed)
        paths = SuitePaths(suite_dir)
        suite_dir.mkdir(parents=True, exist_ok=True)
        paths.payloads_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.config
        n = phase.total_requests
        workloads = self._allocate_workloads(n, spec.mix, rng)
        warmup_ratio = self.config.warmup_ratio if self.config.warmup_ratio is not None else phase.warmup_ratio
        n_warmup = max(1, int(round(n * warmup_ratio)))

        tool_idx = [i for i, w in enumerate(workloads) if w == "tool_agent"]
        session_by_index: dict[int, str] = {}
        if spec.cache_mode in ("hot", "cold"):
            session_by_index = self._tool_sessions(
                tool_idx, rng, hot=(spec.cache_mode == "hot")
            )

        lc_indices = {i for i, w in enumerate(workloads) if w == "long_context"}
        timestamps = build_arrival_timestamps_for_spec(
            n, spec, rng, lc_indices=lc_indices
        )

        hf_ta = list(iter_tool_agent_prompts(self.hf_root, rng, phase.tool_agent))
        hf_lc = list(iter_long_context_prompts(self.hf_root, rng, phase.long_context))
        ta_hf_i = lc_hf_i = 0

        trace_rows: list[dict[str, Any]] = []
        index_entries: dict[str, Any] = {}
        order: list[str] = []
        session_by_rid: dict[str, str | None] = {}
        prefix_groups: dict[str, list[str]] = {}

        for i, workload in enumerate(workloads):
            request_id = f"r-{i+1:05d}"
            target_in, target_out = self._sample_io_lengths(
                rng, workload, spec, cfg.max_context_tokens, cfg.safety_tokens
            )

            gold: str | None = None
            task = "chat"
            cache_meta: dict[str, Any] = {}

            if workload == "tool_agent" and spec.cache_mode in ("hot", "cold"):
                sid = session_by_index[i]
                prefix_tok = self._prefix_token_budget(target_in)
                q_idx = sum(
                    1 for j in tool_idx if j < i and session_by_index.get(j) == sid
                )
                prompt, gold, cache_meta = build_tool_agent_prompt(
                    session_id=sid,
                    prefix_tokens=prefix_tok,
                    total_input_tokens=target_in,
                    rng=rng,
                    question_idx=q_idx,
                )
                task = "single_doc_qa"
                session_by_rid[request_id] = sid
                prefix_groups.setdefault(sid, []).append(request_id)
            elif workload == "tool_agent" and hf_ta and ta_hf_i < len(hf_ta):
                prompt, gold, task = hf_ta[ta_hf_i]
                ta_hf_i += 1
                prompt, _ = enforce_prompt_tokens(prompt, target_in)
            elif workload == "tool_agent":
                sid = f"generic-{i:05d}"
                prefix_tok = self._prefix_token_budget(target_in)
                prompt, gold, cache_meta = build_tool_agent_prompt(
                    session_id=sid,
                    prefix_tokens=prefix_tok,
                    total_input_tokens=target_in,
                    rng=rng,
                    question_idx=i,
                )
                task = "single_doc_qa"
            elif workload == "long_context" and hf_lc and lc_hf_i < len(hf_lc):
                prompt, gold, task = hf_lc[lc_hf_i]
                lc_hf_i += 1
                prompt, target_in = enforce_prompt_tokens(prompt, target_in)
            elif workload == "long_context":
                prompt, gold = build_long_context_prompt(rng, i, target_in)
                task = "shortdep_qa"
                prompt, target_in = enforce_prompt_tokens(prompt, target_in)
            else:
                prompt = build_conversation_prompt(rng, i, target_in)
                prompt, target_in = enforce_prompt_tokens(prompt, target_in)

            if workload == "tool_agent" and "cache_session_id" not in cache_meta:
                prompt, target_in = enforce_prompt_tokens(prompt, target_in)

            hash_ids = compute_hash_ids(prompt)
            is_warmup = i < n_warmup
            rel_payload = f"payloads/{request_id}.json"

            payload = {
                "request_id": request_id,
                "workload_type": workload,
                "encoding": "utf-8+b64",
                "hash_ids": hash_ids,
                "prompt_b64": encode_prompt(prompt),
                "max_tokens": target_out,
                "input_length": target_in,
                "output_length": target_out,
                "reference_answer": gold,
                "task": task,
                "hf_root": str(self.hf_root) if hf_available(self.hf_root) else None,
                **cache_meta,
            }
            write_json(suite_dir / rel_payload, payload)

            trace_row: dict[str, Any] = {
                "request_id": request_id,
                "workload_type": workload,
                "timestamp": timestamps[i],
                "input_length": target_in,
                "output_length": target_out,
                "hash_ids": hash_ids,
                "is_warmup": is_warmup,
                "is_probe_slot": False,
                "cache_session_id": cache_meta.get("cache_session_id"),
                "prefix_tokens_actual": cache_meta.get("prefix_tokens_actual"),
            }
            trace_rows.append(trace_row)
            order.append(request_id)
            index_entries[request_id] = {
                "payload": rel_payload,
                "timestamp": timestamps[i],
                "workload_type": workload,
                "is_warmup": is_warmup,
                "is_probe_slot": False,
                "input_length": target_in,
                "output_length": target_out,
                "hash_ids": hash_ids,
                "cache_session_id": cache_meta.get("cache_session_id"),
            }

        if spec.cache_mode in ("hot", "cold"):
            order = self._reorder_for_cache_sessions(order, index_entries, rng)
            trace_rows.sort(key=lambda r: order.index(r["request_id"]))

        ts_map = self._finalize_timestamps(
            spec, order, index_entries, session_by_rid, rng
        )
        if ts_map:
            for rid, t in ts_map.items():
                index_entries[rid]["timestamp"] = t
                for row in trace_rows:
                    if row["request_id"] == rid:
                        row["timestamp"] = t

        self._validate_prefix_groups(prefix_groups, suite_dir, index_entries)
        probe_rows = self._attach_probes(
            trace_rows, index_entries, suite_dir, phase, rng
        )
        validate_trace_lengths(
            trace_rows, cfg.max_context_tokens, cfg.safety_tokens
        )

        write_jsonl(paths.trace, trace_rows)
        self._write_prompts_export(trace_rows, suite_dir, index_entries)
        write_jsonl(paths.probes, probe_rows)
        write_json(
            suite_dir / "probe_distribution_spec.json",
            probe_distribution_spec(phase, len(probe_rows)),
        )

        ts_list = [index_entries[r]["timestamp"] for r in order]
        cache_stats = self._cache_reuse_stats(prefix_groups, suite_dir, index_entries)

        meta = {
            "suite": spec.name,
            "phase": phase.name,
            "scenario_spec": {
                "mix": spec.mix,
                "arrival": spec.arrival,
                "arrival_params": spec.arrival_params,
                "output_bias": spec.output_bias,
                "cache_mode": spec.cache_mode,
                "description": spec.description,
            },
            "total_requests": n,
            "workload_counts": {
                "conversation": workloads.count("conversation"),
                "tool_agent": workloads.count("tool_agent"),
                "long_context": workloads.count("long_context"),
            },
            "warmup_requests": n_warmup,
            "warmup_ratio": warmup_ratio,
            "scored_requests": n - n_warmup,
            "probe_slots": len(probe_rows),
            "hf_data_used": hf_available(self.hf_root),
            "length_profile": cfg.length_profile,
            "max_context_tokens": cfg.max_context_tokens,
            "arrival_analysis": analyze_timestamps(ts_list),
            "prefix_cache": cache_stats,
            "contest_ref": CONTEST_REF,
            "slo_ttft_ms": phase.slo_ttft_ms,
            "slo_tbt_ms": phase.slo_tbt_ms,
        }
        write_json(paths.trace_meta, meta)
        write_json(
            paths.index,
            {
                "phase": phase.name,
                "suite": spec.name,
                "version": 2,
                "seed": seed,
                "length_profile": cfg.length_profile,
                "max_context_tokens": cfg.max_context_tokens,
                "order": order,
                "entries": index_entries,
            },
        )
        return meta

    def generate_phase(
        self,
        phase_name: str,
        out_dir: Path,
        seed: int,
        *,
        suite_names: list[str] | None = None,
        priority_only: bool = True,
    ) -> dict[str, Any]:
        """Generate multiple suites under ``out_dir/<phase_name>/``."""
        phase = PHASES[phase_name]
        phase_root = out_dir / phase_name
        phase_root.mkdir(parents=True, exist_ok=True)

        specs = SUITES_BY_PHASE[phase_name]
        if priority_only and suite_names is None:
            specs = PRIORITY_SUITES_BY_PHASE[phase_name]
        if suite_names:
            specs = tuple(s for s in SUITES_BY_PHASE[phase_name] if s.name in suite_names)

        manifest: dict[str, Any] = {"phase": phase_name, "suites": {}}
        for j, spec in enumerate(specs):
            info = self.generate_suite(
                spec, phase, phase_root / spec.name, seed + j * 17
            )
            manifest["suites"][spec.name] = info

        write_json(phase_root / "suites_manifest.json", manifest)
        return manifest

    def _sample_io_lengths(
        self,
        rng: random.Random,
        workload: str,
        spec: ScenarioSpec,
        max_context: int,
        safety: int,
    ) -> tuple[int, int]:
        cfg = self.config
        target_in = sample_length(
            rng,
            workload,
            "input",
            length_profile=cfg.length_profile,
            max_context_tokens=max_context,
            safety_tokens=safety,
        )
        if spec.output_bias == "long_decode" and workload == "conversation":
            lo = int(LENGTH_PROFILES["conversation"]["output_median"] * 2)
            hi = min(850, max_context - safety - target_in)
            target_out = rng.randint(max(400, lo), max(lo + 1, hi))
        else:
            target_out = sample_length(
                rng,
                workload,
                "output",
                length_profile=cfg.length_profile,
                max_context_tokens=max_context,
                safety_tokens=safety,
                target_in_hint=target_in,
            )
        return fit_lengths(
            workload,
            target_in,
            target_out,
            length_profile=cfg.length_profile,
            max_context_tokens=max_context,
            safety_tokens=safety,
        )

    @staticmethod
    def _allocate_workloads(n: int, mix: dict[str, float], rng: random.Random) -> list[str]:
        keys = [k for k in ("conversation", "tool_agent", "long_context") if mix.get(k, 0) > 0]
        counts = {k: int(round(n * mix[k])) for k in keys}
        delta = n - sum(counts.values())
        order_keys = sorted(keys, key=lambda k: mix[k], reverse=True)
        i = 0
        while delta != 0 and order_keys:
            k = order_keys[i % len(order_keys)]
            counts[k] += 1 if delta > 0 else -1
            delta += -1 if delta > 0 else 1
            i += 1
        wl: list[str] = []
        for k in keys:
            wl.extend([k] * counts[k])
        rng.shuffle(wl)
        return wl

    @staticmethod
    def _tool_sessions(
        tool_indices: list[int],
        rng: random.Random,
        *,
        hot: bool,
        requests_per_session: tuple[int, int] = (4, 8),
    ) -> dict[int, str]:
        mapping: dict[int, str] = {}
        if hot:
            pos = sess = 0
            while pos < len(tool_indices):
                sid = f"sess-{sess:04d}"
                batch = rng.randint(*requests_per_session)
                for j in range(batch):
                    if pos + j >= len(tool_indices):
                        break
                    mapping[tool_indices[pos + j]] = sid
                pos += batch
                sess += 1
        else:
            for idx in tool_indices:
                mapping[idx] = f"cold-{idx:05d}"
        return mapping

    @staticmethod
    def _prefix_token_budget(total_in: int) -> int:
        return max(4000, int(total_in * 0.88))

    @staticmethod
    def _reorder_for_cache_sessions(
        order: list[str],
        index_entries: dict[str, Any],
        rng: random.Random,
    ) -> list[str]:
        by_session: dict[str, list[str]] = defaultdict(list)
        other: list[str] = []
        for rid in order:
            sid = index_entries[rid].get("cache_session_id")
            if sid:
                by_session[sid].append(rid)
            else:
                other.append(rid)
        sessions = list(by_session.keys())
        rng.shuffle(sessions)
        new_order: list[str] = []
        oi = 0
        for sid in sessions:
            new_order.extend(by_session[sid])
            if oi < len(other):
                new_order.append(other[oi])
                oi += 1
        new_order.extend(other[oi:])
        return new_order

    @staticmethod
    def _same_timestamp_within_session(
        order: list[str],
        index_entries: dict[str, Any],
    ) -> dict[str, int]:
        t = 0
        ts_map: dict[str, int] = {}
        prev_sid = None
        for rid in order:
            sid = index_entries[rid].get("cache_session_id")
            if sid is not None and sid != prev_sid:
                t += 50
            elif sid is None:
                t += 30
            ts_map[rid] = t if sid else t
            if sid:
                prev_sid = sid
            else:
                prev_sid = None
        by_session: dict[str, list[str]] = {}
        for rid in order:
            sid = index_entries[rid].get("cache_session_id")
            if sid:
                by_session.setdefault(sid, []).append(rid)
        for _sid, rids in by_session.items():
            base = ts_map[rids[0]]
            for rid in rids:
                ts_map[rid] = base
        return ts_map

    def _finalize_timestamps(
        self,
        spec: ScenarioSpec,
        order: list[str],
        index_entries: dict[str, Any],
        session_by_rid: dict[str, str | None],
        rng: random.Random,
    ) -> dict[str, int] | None:
        if spec.cache_mode == "hot":
            return self._same_timestamp_within_session(order, index_entries)
        if spec.arrival == "session_cluster" and session_by_rid:
            return build_session_clustered_timestamps(
                order,
                session_by_rid,
                rng,
                tuple(spec.arrival_params.get("intra_gap_ms", (5, 40))),
                tuple(spec.arrival_params.get("inter_session_ms", (400, 1200))),
            )
        return None

    def _attach_probes(
        self,
        trace_rows: list[dict[str, Any]],
        index_entries: dict[str, Any],
        suite_dir: Path,
        phase: PhaseSpec,
        rng: random.Random,
    ) -> list[dict[str, Any]]:
        probe_slots = pick_probe_slots(trace_rows, phase.probe_slot_ratio, rng)
        probe_rows: list[dict[str, Any]] = []
        ta_probe = lc_probe = 0
        for row in trace_rows:
            rid = row["request_id"]
            if rid not in probe_slots:
                continue
            row["is_probe_slot"] = True
            index_entries[rid]["is_probe_slot"] = True
            payload = json.loads(
                (suite_dir / index_entries[rid]["payload"]).read_text(encoding="utf-8")
            )
            gold = payload.get("reference_answer") or ""
            wl = row["workload_type"]
            if wl == "long_context":
                method, pid = "f1_rouge_l", f"probe-lc-{lc_probe:04d}"
                lc_probe += 1
            else:
                method, pid = "f1_em", f"probe-ta-{ta_probe:04d}"
                ta_probe += 1
            probe_rows.append(
                {
                    "probe_id": pid,
                    "workload_type": wl,
                    "prompt_b64": payload["prompt_b64"],
                    "reference_answer": gold,
                    "evaluation_method": method,
                    "substitute_for_request_id": rid,
                    "task": payload.get("task"),
                }
            )
        return probe_rows

    @staticmethod
    def _write_prompts_export(
        trace_rows: list[dict[str, Any]],
        suite_dir: Path,
        index_entries: dict[str, Any],
    ) -> None:
        write_jsonl(
            suite_dir / "prompts.jsonl",
            [
                {
                    "request_id": r["request_id"],
                    "workload_type": r["workload_type"],
                    "prompt": decode_prompt(
                        json.loads(
                            (suite_dir / index_entries[r["request_id"]]["payload"]).read_text(
                                encoding="utf-8"
                            )
                        )
                    ),
                    "input_length": r["input_length"],
                    "output_length": r["output_length"],
                    "cache_session_id": r.get("cache_session_id"),
                }
                for r in trace_rows
            ],
        )

    @staticmethod
    def _validate_prefix_groups(
        prefix_groups: dict[str, list[str]],
        suite_dir: Path,
        index_entries: dict[str, Any],
    ) -> None:
        for sid, rids in prefix_groups.items():
            if len(rids) < 2:
                continue
            texts: list[str] = []
            for rid in rids:
                pl = json.loads(
                    (suite_dir / index_entries[rid]["payload"]).read_text(encoding="utf-8")
                )
                texts.append(decode_prompt(pl))
            prefix_len = min(len(texts[0]), 2000)
            p0 = texts[0][:prefix_len]
            if not all(t[:prefix_len] == p0 for t in texts[1:]):
                raise RuntimeError(
                    f"Session {sid}: prompts do not share real text prefix (cache hot invalid)"
                )

    @staticmethod
    def _cache_reuse_stats(
        prefix_groups: dict[str, list[str]],
        suite_dir: Path,
        index_entries: dict[str, Any],
    ) -> dict[str, Any]:
        if not prefix_groups:
            return {"sessions": 0, "note": "no tool cache sessions"}
        multi = {k: v for k, v in prefix_groups.items() if len(v) >= 2}
        shared_blocks = 0
        for _sid, rids in multi.items():
            texts = []
            for rid in rids[:2]:
                pl = json.loads(
                    (suite_dir / index_entries[rid]["payload"]).read_text(encoding="utf-8")
                )
                texts.append(decode_prompt(pl))
            h0 = compute_hash_ids(texts[0])
            h1 = compute_hash_ids(texts[1])
            shared_blocks += sum(1 for a, b in zip(h0, h1) if a == b)
        return {
            "sessions": len(prefix_groups),
            "sessions_with_2plus_requests": len(multi),
            "sample_shared_hash_blocks_first_pair": shared_blocks,
            "note": "hash_ids match when text prefix matches; vLLM uses real tokens.",
        }


def generate_suite(
    spec: ScenarioSpec,
    phase: PhaseSpec,
    suite_dir: Path,
    hf_root: Path,
    seed: int,
    *,
    length_profile: str = "heavy",
    max_context_tokens: int = 32768,
    safety_tokens: int = 256,
    warmup_ratio: float | None = None,
) -> dict[str, Any]:
    """Functional API used by CLI and legacy imports."""
    gen = TraceGenerator(
        hf_root,
        GenerationConfig(
            length_profile=length_profile,
            max_context_tokens=max_context_tokens,
            safety_tokens=safety_tokens,
            warmup_ratio=warmup_ratio,
        ),
    )
    return gen.generate_suite(spec, phase, suite_dir, seed)


def generate_phase_suites(
    phase_name: str,
    out_dir: Path,
    hf_root: Path,
    seed: int,
    *,
    suite_names: list[str] | None = None,
    priority_only: bool = True,
    **gen_kw: Any,
) -> dict[str, Any]:
    """Generate all suites for a phase (backward-compatible wrapper)."""
    cfg = GenerationConfig(
        length_profile=gen_kw.get("length_profile", "heavy"),
        max_context_tokens=gen_kw.get("max_context_tokens", 32768),
        safety_tokens=gen_kw.get("safety_tokens", 256),
        warmup_ratio=gen_kw.get("warmup_ratio"),
    )
    return TraceGenerator(hf_root, cfg).generate_phase(
        phase_name, out_dir, seed, suite_names=suite_names, priority_only=priority_only
    )
