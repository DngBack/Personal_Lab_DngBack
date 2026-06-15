"""Normalize heterogeneous model JSON into canonical per-page field dict."""

from __future__ import annotations

from typing import Any

from stitch.config import FIELD_ALIASES, canonical_field_name

_CHANDRA_LABEL_MAP: dict[str, str] = {
    "Page-Footer": "THÔNG TIN CHÂN TRANG",
    "Table": "BẢNG KÊ DỊCH VỤ",
}


def _bbox_to_str(bbox) -> str:
    if isinstance(bbox, str):
        return bbox
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return " ".join(str(x) for x in bbox[:4])
    return ""


def from_chandra_blocks(blocks_doc: dict[str, Any]) -> dict[str, dict[str, str]]:
    """
    Convert chandra_noitru *_blocks.json to page field dict.

    Picks largest Table block as BẢNG KÊ DỊCH VỤ; Page-Footer → THÔNG TIN CHÂN TRANG.
    """
    blocks = blocks_doc.get("blocks", [])
    out: dict[str, dict[str, str]] = {}
    tables: list[tuple[int, dict[str, Any]]] = []

    for b in blocks:
        label = b.get("label", "")
        text = b.get("text", "")
        if isinstance(text, list):
            text = " ".join(str(x) for x in text)
        if "lines" in b and isinstance(b["lines"], list):
            text = " ".join(str(x) for x in b["lines"])
        bbox = _bbox_to_str(b.get("bbox", ""))
        field_name = _CHANDRA_LABEL_MAP.get(label)
        if label == "Table":
            tables.append((len(text), b))
            continue
        if field_name:
            out[field_name] = {"content": text, "bbox": bbox, "type": label or "Text"}

    if tables:
        _, best = max(tables, key=lambda t: t[0])
        text = best.get("text", "")
        if isinstance(text, list):
            text = " ".join(str(x) for x in text)
        out["BẢNG KÊ DỊCH VỤ"] = {
            "content": text,
            "bbox": _bbox_to_str(best.get("bbox", "")),
            "type": "Table",
        }

    return out


def normalize_field_name(name: str) -> str:
    return FIELD_ALIASES.get(name, canonical_field_name(name))


def normalize_page_dict(page: dict[str, Any]) -> dict[str, dict[str, str]]:
    """
    Accept either:
    - canonical: {field: {content, bbox, type}}
    - wrapped: {"fields": {...}} or chandra blocks {"blocks": [...]}
    """
    if "blocks" in page:
        return from_chandra_blocks(page)
    raw = page.get("fields", page)
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, str]] = {}
    for raw_name, field in raw.items():
        if not isinstance(field, dict):
            continue
        name = normalize_field_name(raw_name)
        out[name] = {
            "content": field.get("content", ""),
            "bbox": field.get("bbox", ""),
            "type": field.get("type", "Text"),
        }
    return out


def normalize_pages(pages: list[dict[str, Any]]) -> list[dict[str, dict[str, str]]]:
    return [normalize_page_dict(p) for p in pages]
