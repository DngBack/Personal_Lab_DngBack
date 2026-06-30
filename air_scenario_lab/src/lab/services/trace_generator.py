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
from ..config import (
    CONTEST_REF,
    DEFAULT_CONTEXT_SAFETY_TOKENS,
    DEFAULT_MAX_CONTEXT_TOKENS,
    LENGTH_PROFILES,
    REALISTIC_INPUT_CAP,
)
from ..domain.scenario import ScenarioSpec
from ..probes_util import pick_probe_slots, probe_distribution_spec
from ..sources.hf_prompts import (
    build_tool_agent_from_leval,
    iter_conversation_prompts,
    iter_long_context_prompts,
    leval_gsm100_rows,
    require_hf_data,
)
from ..storage import encode_prompt, write_json, write_jsonl
from ..tokens import compute_hash_ids
from ..trace_lengths import fit_lengths, sample_length, validate_trace_lengths


class TraceGenerator:
    def __init__(self, hf_root: Path) -> None:
        self.hf_root = hf_root

    def generate_suite(self, spec: ScenarioSpec, suite_dir: Path) -> dict[str, Any]:
        require_hf_data(self.hf_root)
        rng = random.Random(spec.seed)
        suite_dir.mkdir(parents=True, exist_ok=True)
        (suite_dir / "payloads").mkdir(parents=True, exist_ok=True)

        n = spec.total_requests
        workloads = self._allocate_workloads(n, spec.mix, rng)
        n_warmup = max(1, int(round(n * spec.warmup_ratio)))

        tool_idx = [i for i, w in enumerate(workloads) if w == "tool_agent"]
        session_by_index = self._tool_sessions(tool_idx, rng, cache_mode=spec.cache_mode)

        lc_indices = {i for i, w in enumerate(workloads) if w == "long_context"}
        timestamps = build_arrival_timestamps_for_spec(
            n, spec, rng, lc_indices=lc_indices
        )

        counts = spec.workload_counts()
        conv_pool = list(
            iter_conversation_prompts(
                self.hf_root,
                rng,
                counts.get("conversation", 0),
                max_input_tokens=REALISTIC_INPUT_CAP["conversation"],
            )
        )
        lc_pool = list(
            iter_long_context_prompts(
                self.hf_root,
                rng,
                counts.get("long_context", 0),
                max_input_tokens=REALISTIC_INPUT_CAP["long_context"],
            )
        )
        gsm_rows = leval_gsm100_rows(self.hf_root)
        gsm_indices = list(range(len(gsm_rows)))
        rng.shuffle(gsm_indices)

        conv_i = lc_i = 0
        trace_rows: list[dict[str, Any]] = []
        index_entries: dict[str, Any] = {}
        order: list[str] = []
        session_by_rid: dict[str, str | None] = {}
        prefix_groups: dict[str, list[str]] = {}

        for i, workload in enumerate(workloads):
            request_id = f"r-{i+1:05d}"
            target_in_hint, target_out = self._sample_io_lengths(rng, workload, spec)

            gold: str | None = None
            task = "chat"
            cache_meta: dict[str, Any] = {}
            prompt = ""

            if workload == "conversation":
                prompt, gold, task, actual_in = conv_pool[conv_i]
                conv_i += 1
                target_in = actual_in
            elif workload == "long_context":
                prompt, gold, task, actual_in = lc_pool[lc_i]
                lc_i += 1
                target_in = actual_in
                target_out = min(target_out, DEFAULT_MAX_CONTEXT_TOKENS - DEFAULT_CONTEXT_SAFETY_TOKENS - target_in)
            elif workload == "tool_agent":
                sid = session_by_index.get(i)
                if spec.cache_mode == "hot" and sid:
                    row = gsm_rows[gsm_indices[i % len(gsm_indices)]]
                    prompt, gold, task = build_tool_agent_from_leval(row)
                    cache_meta = {"cache_session_id": sid}
                    session_by_rid[request_id] = sid
                    prefix_groups.setdefault(sid, []).append(request_id)
                elif spec.cache_mode == "cold":
                    row = gsm_rows[gsm_indices[i % len(gsm_indices)]]
                    prompt, gold, task = build_tool_agent_from_leval(row)
                    cache_meta = {"cache_session_id": f"cold-{i:05d}"}
                else:
                    row = gsm_rows[gsm_indices[i % len(gsm_indices)]]
                    prompt, gold, task = build_tool_agent_from_leval(row)

                cap = REALISTIC_INPUT_CAP["tool_agent"]
                from ..tokens import truncate_prompt_natural

                prompt, target_in = truncate_prompt_natural(prompt, min(target_in_hint, cap))
                target_out = min(
                    target_out,
                    DEFAULT_MAX_CONTEXT_TOKENS - DEFAULT_CONTEXT_SAFETY_TOKENS - target_in,
                )
            else:
                raise RuntimeError(f"Unknown workload: {workload}")

            if workload in ("conversation", "long_context"):
                target_out = min(
                    target_out,
                    DEFAULT_MAX_CONTEXT_TOKENS - DEFAULT_CONTEXT_SAFETY_TOKENS - target_in,
                )

            hash_ids = compute_hash_ids(prompt, target_in)
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
                "hf_root": str(self.hf_root),
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

        if spec.cache_mode == "hot":
            order = self._reorder_for_cache_sessions(order, index_entries, rng)
            trace_rows.sort(key=lambda r: order.index(r["request_id"]))

        ts_map = self._finalize_timestamps(spec, order, index_entries, session_by_rid, rng)
        if ts_map:
            for rid, t in ts_map.items():
                index_entries[rid]["timestamp"] = t
                for row in trace_rows:
                    if row["request_id"] == rid:
                        row["timestamp"] = t

        self._validate_prefix_groups(prefix_groups, suite_dir, index_entries)
        probe_rows = self._attach_probes(trace_rows, index_entries, suite_dir, spec, rng)
        validate_trace_lengths(
            trace_rows, DEFAULT_MAX_CONTEXT_TOKENS, DEFAULT_CONTEXT_SAFETY_TOKENS
        )

        write_jsonl(suite_dir / "trace.jsonl", trace_rows)
        write_jsonl(suite_dir / "probes.jsonl", probe_rows)
        write_json(
            suite_dir / "probe_distribution_spec.json",
            probe_distribution_spec(spec, len(probe_rows)),
        )

        ts_list = [index_entries[r]["timestamp"] for r in order]
        meta = {
            "suite": spec.name,
            "phase": spec.phase,
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
            "warmup_ratio": spec.warmup_ratio,
            "scored_requests": n - n_warmup,
            "probe_slots": len(probe_rows),
            "hf_data_used": True,
            "length_profile": spec.length_profile,
            "max_context_tokens": DEFAULT_MAX_CONTEXT_TOKENS,
            "arrival_analysis": analyze_timestamps(ts_list),
            "prefix_cache": self._cache_reuse_stats(prefix_groups, suite_dir, index_entries),
            "contest_ref": CONTEST_REF,
            "slo_ttft_ms": spec.phase_spec().slo_ttft_ms,
            "slo_tbt_ms": spec.phase_spec().slo_tbt_ms,
        }
        write_json(suite_dir / "trace_meta.json", meta)
        write_json(
            suite_dir / "index.json",
            {
                "phase": spec.phase,
                "suite": spec.name,
                "version": 1,
                "seed": spec.seed,
                "length_profile": spec.length_profile,
                "max_context_tokens": DEFAULT_MAX_CONTEXT_TOKENS,
                "slo_ttft_ms": spec.phase_spec().slo_ttft_ms,
                "slo_tbt_ms": spec.phase_spec().slo_tbt_ms,
                "request_timeout_s": spec.phase_spec().request_timeout_s,
                "order": order,
                "entries": index_entries,
            },
        )
        return meta

    def _sample_io_lengths(
        self, rng: random.Random, workload: str, spec: ScenarioSpec
    ) -> tuple[int, int]:
        target_in = sample_length(
            rng,
            workload,
            "input",
            length_profile=spec.length_profile,
            max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
            safety_tokens=DEFAULT_CONTEXT_SAFETY_TOKENS,
        )
        if spec.output_bias == "long_decode" and workload == "conversation":
            lo = int(LENGTH_PROFILES["conversation"]["output_median"] * 2)
            hi = min(850, DEFAULT_MAX_CONTEXT_TOKENS - DEFAULT_CONTEXT_SAFETY_TOKENS - target_in)
            target_out = rng.randint(max(400, lo), max(lo + 1, hi))
        else:
            target_out = sample_length(
                rng,
                workload,
                "output",
                length_profile=spec.length_profile,
                max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
                safety_tokens=DEFAULT_CONTEXT_SAFETY_TOKENS,
                target_in_hint=target_in,
            )
        return fit_lengths(
            workload,
            target_in,
            target_out,
            length_profile=spec.length_profile,
            max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS,
            safety_tokens=DEFAULT_CONTEXT_SAFETY_TOKENS,
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
        cache_mode: str | None,
        requests_per_session: tuple[int, int] = (4, 8),
    ) -> dict[int, str]:
        mapping: dict[int, str] = {}
        if cache_mode == "hot":
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
        elif cache_mode == "cold":
            for idx in tool_indices:
                mapping[idx] = f"cold-{idx:05d}"
        return mapping

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
            if sid and not sid.startswith("cold-"):
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
            prev_sid = sid if sid else None

        by_session: dict[str, list[str]] = {}
        for rid in order:
            sid = index_entries[rid].get("cache_session_id")
            if sid:
                by_session.setdefault(sid, []).append(rid)
        for rids in by_session.values():
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
        spec: ScenarioSpec,
        rng: random.Random,
    ) -> list[dict[str, Any]]:
        probe_slots = pick_probe_slots(trace_rows, spec.probe_slot_ratio, rng)
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
    def _validate_prefix_groups(
        prefix_groups: dict[str, list[str]],
        suite_dir: Path,
        index_entries: dict[str, Any],
    ) -> None:
        from ..storage import decode_prompt

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
        from ..storage import decode_prompt

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
            "note": "gsm100 shared input prefix; vLLM caches real token prefix.",
        }
