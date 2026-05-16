"""Chỉ parse HTML do LLM in ra (div + data-schema); ghép layout bằng khớp chuỗi đúng, không fuzzy."""
from __future__ import annotations

import html as html_module
import json
import re
from typing import Any

_DIV_RE = re.compile(
    r"<div\b(?P<attrs>[^>]*)>(?P<inner>.*?)</div>",
    re.IGNORECASE | re.DOTALL,
)
_BBOX_RE = re.compile(r'data(?:-bbox)?\s*=\s*"([^"]+)"', re.IGNORECASE)
_LABEL_RE = re.compile(r'data-label\s*=\s*"([^"]+)"', re.IGNORECASE)
_SCHEMA_RE = re.compile(r'data-schema\s*=\s*"([^"]*)"', re.IGNORECASE)
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def merge_duplicate_schema_divs(html: str) -> str:
    """
    Gộp mọi ``<div>`` trùng ``data-schema`` (cùng chuỗi UTF-8, khác rỗng) thành một div:
    bbox = union các bbox, nội dung nối bằng ``<br/>``.

    VL/LLM đôi khi xuất nhiều ô cho cùng một trường; bước này chuẩn hoá trước khi ghi HTML và vẽ bbox.
    """
    matches = list(_DIV_RE.finditer(html))
    if len(matches) < 2:
        return html

    items: list[dict[str, Any]] = []
    for m in matches:
        attrs = m.group("attrs") or ""
        inner = m.group("inner") or ""
        sm = _SCHEMA_RE.search(attrs)
        schema = (sm.group(1).strip() if sm else "") or ""
        bbox: list[float] | None = None
        bm = _BBOX_RE.search(attrs)
        if bm:
            parts = bm.group(1).replace(",", " ").split()
            if len(parts) == 4:
                try:
                    x0, y0, x1, y1 = map(float, parts)
                    x0, x1 = min(x0, x1), max(x0, x1)
                    y0, y1 = min(y0, y1), max(y0, y1)
                    bbox = [x0, y0, x1, y1]
                except ValueError:
                    pass
        items.append({"span": m.span(), "schema": schema, "inner": inner, "bbox": bbox})

    by_schema: dict[str, list[int]] = {}
    for i, it in enumerate(items):
        sk = it["schema"]
        if not sk:
            continue
        by_schema.setdefault(sk, []).append(i)

    merge_groups = {sk: idxs for sk, idxs in by_schema.items() if len(idxs) > 1}
    if not merge_groups:
        return html

    skip_spans: set[tuple[int, int]] = set()
    first_replace: dict[int, str] = {}
    for idxs in merge_groups.values():
        group = [items[i] for i in idxs]
        bbs = [g["bbox"] for g in group if g["bbox"] is not None]
        if not bbs:
            continue
        sk = group[0]["schema"]
        x0 = int(min(b[0] for b in bbs))
        y0 = int(min(b[1] for b in bbs))
        x1 = int(max(b[2] for b in bbs))
        y1 = int(max(b[3] for b in bbs))
        inners = [g["inner"].strip() for g in group if g["inner"].strip()]
        inner_html = "<br/>".join(inners)
        esc = html_module.escape(sk, quote=True)
        new_div = (
            f'<div data-bbox="{x0} {y0} {x1} {y1}" data-label="{esc}" data-schema="{esc}">'
            f"{inner_html}</div>"
        )
        first_i = idxs[0]
        first_replace[items[first_i]["span"][0]] = new_div
        for j in idxs[1:]:
            skip_spans.add(items[j]["span"])

    pos = 0
    out: list[str] = []
    for m in matches:
        s, e = m.span()
        out.append(html[pos:s])
        if m.span() in skip_spans:
            pos = e
            continue
        if s in first_replace:
            out.append(first_replace[s])
        else:
            out.append(m.group(0))
        pos = e
    out.append(html[pos:])
    return "".join(out)


def parse_schema_divs(html: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for m in _DIV_RE.finditer(html):
        attrs = m.group("attrs") or ""
        inner = m.group("inner") or ""
        bm = _BBOX_RE.search(attrs)
        lm = _LABEL_RE.search(attrs)
        sm = _SCHEMA_RE.search(attrs)
        if not bm:
            continue
        parts = bm.group(1).replace(",", " ").split()
        if len(parts) != 4:
            continue
        try:
            x0, y0, x1, y1 = map(float, parts)
        except ValueError:
            continue
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        lines_raw = _BR_RE.split(inner)
        lines = [re.sub(r"\s+", " ", _TAG_RE.sub("", ln)).strip() for ln in lines_raw]
        lines = [ln for ln in lines if ln]
        text = " ".join(lines)
        blocks.append({
            "bbox": [x0, y0, x1, y1],
            "label": lm.group(1).strip() if lm else "",
            "schema": (sm.group(1).strip() if sm else "") or None,
            "text": text,
            "lines": lines,
        })
    return blocks


def blocks_to_flat_map(blocks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Last occurrence wins for duplicate schema names."""
    out: dict[str, dict[str, Any]] = {}
    for b in blocks:
        sk = b.get("schema")
        if not sk:
            continue
        out[sk] = {"text": b.get("text", ""), "bbox_1000": b.get("bbox"), "label": b.get("label")}
    return out


def attach_values_to_layout(layout: dict[str, Any], flat: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Gắn `extracted` khi `name` trong layout trùng khít (UTF-8) với key từ `data-schema` của LLM."""
    def enrich(o: Any) -> Any:
        if isinstance(o, dict):
            cp = {kk: enrich(vv) for kk, vv in o.items()}
            name = cp.get("name")
            if isinstance(name, str):
                nm = name.strip()
                if nm and nm in flat:
                    cp["extracted"] = flat[nm]
            return cp
        if isinstance(o, list):
            return [enrich(x) for x in o]
        return o

    return enrich(layout)


def layout_with_extractions_to_json(layout_path: str, schema_html: str) -> dict[str, Any]:
    from pathlib import Path

    blocks = parse_schema_divs(schema_html)
    flat = blocks_to_flat_map(blocks)
    raw = json.loads(Path(layout_path).read_text(encoding="utf-8"))
    root = dict(raw) if isinstance(raw, dict) else {"sections": raw}
    enriched = attach_values_to_layout(root, flat)
    return {
        "layout_with_values": enriched,
        "flat_by_schema": flat,
        "blocks": blocks,
    }


def extract_chandra_html_from_log(log_text: str) -> str:
    """Return HTML segment starting at first ``<div`` (Chandra / merged output)."""
    i = log_text.find("<div")
    if i >= 0:
        return log_text[i:].strip()
    return log_text.strip()
