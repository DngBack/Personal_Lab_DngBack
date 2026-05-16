"""Map Chandra layout blocks → layout schema fields + bbox qua một LLM (Qwen/Instruct-ready).

Đầu vào: danh sách block (bbox 0–1000, nhãn Chandra, text).
Đầu ra: JSON các key trùng tên schema (layout JSON); value = bbox 4 số hoặc null."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import unicodedata


def _fold(s: str) -> str:
    _vn = str.maketrans("đĐơƠưƯ", "dDoOuU")
    s = s.translate(_vn)
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if ("a" <= c <= "z") or c.isdigit())


def compact_blocks(blocks: list[dict[str, Any]], text_max: int = 320) -> list[dict[str, Any]]:
    out = []
    for i, b in enumerate(blocks):
        t = (b.get("text") or "").strip().replace("\n", " ")
        if len(t) > text_max:
            t = t[: text_max - 3] + "..."
        out.append({
            "i": i,
            "label": b.get("label", ""),
            "bbox": b.get("bbox"),
            "text": t,
        })
    return out


def load_fewshots(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    shots = raw.get("shots") if isinstance(raw, dict) else None
    return shots if isinstance(shots, list) else []


def _fence_json_strip(s: str) -> str:
    s = s.strip()
    fence = re.search(r"```(?:json)?\s*\n([\s\S]*?)```", s, re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return s


def _balanced_object(s: str) -> str | None:
    start = s.find("{")
    if start < 0:
        return None
    depth, i = 0, start
    in_str = False
    esc = False
    qc = ""
    while i < len(s):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == qc:
                in_str = False
        else:
            if c in "\"'":
                in_str = True
                qc = c
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
        i += 1
    return None


def extract_json_obj(s: str) -> dict[str, Any] | None:
    stripped = _fence_json_strip(s)
    cand = _balanced_object(stripped)
    if cand is None:
        return None
    try:
        obj = json.loads(cand)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _canon_bbox(v: Any) -> list[float] | None:
    if v is None:
        return None
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
    except (TypeError, ValueError):
        return None
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    return [max(0, x0), max(0, y0), min(1000, x1), min(1000, y1)]


def reconcile_keys(
    obj: dict[str, Any],
    schema_names_ordered: list[str],
) -> dict[str, list[float] | None]:
    """Chuẩn hóa key LLM → tên trong layout (fold-match), bbox hợp lệ."""
    fold2canonical: dict[str, str] = {}
    for n in schema_names_ordered:
        fold2canonical.setdefault(_fold(n), n)

    out: dict[str, list[float] | None] = {n: None for n in schema_names_ordered}

    for k_raw, val in obj.items():
        if not isinstance(k_raw, str):
            continue
        kk = fold2canonical.get(_fold(k_raw.strip()))
        if kk is None:
            continue
        bbox = _canon_bbox(val)
        out[kk] = bbox
    return out


def build_user_prompt(
    schema_fields: list[str],
    blocks_compact: list[dict[str, Any]],
    fewshots_rendered: str,
) -> str:
    schema_blob = json.dumps(schema_fields, ensure_ascii=False)
    blocks_blob = json.dumps(blocks_compact, ensure_ascii=False)
    parts = []
    if fewshots_rendered.strip():
        parts.append(fewshots_rendered.strip())
        parts.append("")
    parts.append(f"Đây là `schema_fields` (thứ tự tham khảo):\n```json\n{schema_blob}\n```")
    parts.append("")
    parts.append("`chandra_blocks` (độc nhất theo chỉ mục i; bbox [x0,y0,x1,y1] normalized 0–1000):\n```json")
    parts.append(blocks_blob)
    parts.append("```")
    parts.append("")
    parts.append(
        "Trả ra **đúng một** object JSON là map từ **mỗi** tên trong `schema_fields` "
        "(string chính xác, UTF-8) → giá trị là either null hoặc mảng 4 float [x0,y0,x1,y1].\n"
        "Quy tắc:\n"
        "- Chọn một block sao cho văn bản/visual khớp trường schema; sao chép bbox của block đó (không tự làm bbox mới).\n"
        "- Một schema_field chỉ được gán tối đa một block; nếu không chắc → null.\n"
        "- Bảng lớn: dùng bbox của khối Table tương ứng (nếu có).\n"
        "- Logo: ô logo ngân hàng góc trên trái.\n"
        "- Đầu ra **chỉ** là một JSON object thuần: ký tự đầu là `{`, ký tự cuối là `}`."
        " Không markdown, không tiếng Anh giải thích.\n\n"
        "Bắt đầu ngay JSON:"
    )
    return "\n".join(parts)


def render_fewshots_from_file(shots: list[dict[str, Any]]) -> str:
    chunks = []
    for si, shot in enumerate(shots):
        if not isinstance(shot, dict):
            continue
        title = shot.get("title") or f"Example {si + 1}"
        sf = shot.get("schema_fields")
        bk = shot.get("chandra_blocks_compact")
        ex = shot.get("expected_bbox_map")
        if not isinstance(sf, list) or not isinstance(bk, list) or not isinstance(ex, dict):
            continue
        chunks.append(f"### {title}")
        chunks.append("Input `schema_fields`:")
        chunks.append(json.dumps(sf, ensure_ascii=False))
        chunks.append("`chandra_blocks`:")
        chunks.append(json.dumps(bk, ensure_ascii=False))
        chunks.append("Output JSON:")
        chunks.append(json.dumps(ex, ensure_ascii=False))
        chunks.append("")
    return "\n".join(chunks)



class CachedSchemaCausalLM:
    """Nạp Qwen/transformers một lần; `predict_schema_map` tái dùng trên nhiều trang."""

    def __init__(self, model_id: str, device: str, dtype_s: str) -> None:
        import gc
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._gc = gc

        tm = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dt = tm.get(dtype_s)

        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dt if dt is not None else "auto",
            device_map=device,
            trust_remote_code=True,
        )
        mdl.eval()
        self.tokenizer = tok
        self.model = mdl
        self._idev = next(mdl.parameters()).device

    def cleanup(self) -> None:
        self.model = None
        self.tokenizer = None
        self._gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def predict_schema_map(
        self,
        *,
        system_prompt_text: str,
        fewshots_path: Path | None,
        schema_names_ordered: list[str],
        blocks: list[dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[dict[str, list[float] | None], str]:
        """Trả reconcile map + decoded assistant text."""

        cmp_blocks = compact_blocks(blocks)
        few_txt = render_fewshots_from_file(load_fewshots(fewshots_path))

        extra_system = ""
        if fewshots_path and fewshots_path.is_file():
            try:
                raw_ex = json.loads(fewshots_path.read_text(encoding="utf-8"))
                es = raw_ex.get("system_addon") if isinstance(raw_ex, dict) else None
                if isinstance(es, str) and es.strip():
                    extra_system = "\n\n" + es.strip()
            except (json.JSONDecodeError, OSError):
                pass

        system = system_prompt_text.strip() + extra_system
        user = build_user_prompt(schema_names_ordered, cmp_blocks, few_txt)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            templ = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            templ = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        enc = self.tokenizer(templ, return_tensors="pt")
        enc = {k: v.to(self._idev) for k, v in enc.items()}

        do_sample = temperature > 0
        gen_kw: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": getattr(self.tokenizer, "eos_token_id", None),
            "repetition_penalty": 1.15,
        }
        if do_sample:
            gen_kw["temperature"] = max(0.05, temperature)
        else:
            gen_kw["temperature"] = 1.0
        with self._torch.inference_mode():
            ids = self.model.generate(**enc, **gen_kw)
        pref = enc["input_ids"].shape[-1]
        decoded = self.tokenizer.decode(ids[0, pref:], skip_special_tokens=True)

        obj = extract_json_obj(decoded)
        if obj is None:
            return {n: None for n in schema_names_ordered}, decoded
        rec = reconcile_keys(obj, schema_names_ordered)
        return rec, decoded
