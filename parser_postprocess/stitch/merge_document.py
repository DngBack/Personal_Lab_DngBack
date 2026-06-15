"""Orchestrate cross-page field merging into one document."""

from __future__ import annotations

from typing import Any

from stitch.config import MergeStrategy, canonical_field_name, resolve_strategy
from stitch.io import SamplePages
from stitch.page_layout import infer_page_height
from stitch.table_merge import merge_tables
from stitch.text_merge import merge_text_pages


def _field_occurrences(
    sample: SamplePages,
) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    """Map canonical field name → [(page_num, field_dict), ...] in page order."""
    occ: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for page in sample.pages:
        for raw_name, field in page.data.items():
            name = canonical_field_name(raw_name)
            occ.setdefault(name, []).append((page.page_num, field))
    return occ


def _merge_footer(
    entries: list[tuple[int, dict[str, Any]]],
) -> tuple[str, list[str]]:
    lines: list[str] = []
    for page_num, field in entries:
        content = field.get("content", "").strip()
        if content:
            lines.append(f"Trang {page_num}: {content}")
    note = ["per-page footer preserved"]
    return "\n".join(lines), note


def _pick_first(
    entries: list[tuple[int, dict[str, Any]]],
) -> tuple[dict[str, Any], list[int], list[str]]:
    page_num, field = entries[0]
    note = "single occurrence" if len(entries) == 1 else "first page"
    return field, [page_num], [note]


def _pick_last(
    entries: list[tuple[int, dict[str, Any]]],
) -> tuple[dict[str, Any], list[int], list[str]]:
    page_num, field = entries[-1]
    note = "single occurrence" if len(entries) == 1 else "last page"
    return field, [page_num], [note]


def _merge_field(
    field_name: str,
    entries: list[tuple[int, dict[str, Any]]],
    strategy: MergeStrategy,
    page_heights: dict[int, float] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    source_pages = [p for p, _ in entries]
    base_field = entries[0][1]
    out_type = base_field.get("type", "Text")
    bbox = base_field.get("bbox", "")
    notes: list[str] = []

    if strategy == MergeStrategy.TABLE_CONCAT:
        htmls = [e.get("content", "") for _, e in entries]
        bboxes = [e.get("bbox", "") for _, e in entries]
        heights = None
        if page_heights:
            heights = [page_heights.get(p) for p, _ in entries]
        content, notes = merge_tables(htmls, bboxes=bboxes, page_heights=heights)
    elif strategy == MergeStrategy.FOOTER_PER_PAGE:
        content, notes = _merge_footer(entries)
    elif strategy == MergeStrategy.TEXT_STITCH:
        texts = [e.get("content", "") for _, e in entries]
        content, notes = merge_text_pages(texts)
    elif strategy == MergeStrategy.LAST_PAGE:
        field, pages, notes = _pick_last(entries)
        content = field.get("content", "")
        source_pages = pages
        out_type = field.get("type", out_type)
        bbox = field.get("bbox", bbox)
    elif strategy in (MergeStrategy.FIRST_PAGE, MergeStrategy.AS_IS):
        field, pages, notes = _pick_first(entries)
        content = field.get("content", "")
        source_pages = pages
        out_type = field.get("type", out_type)
        bbox = field.get("bbox", bbox)
    else:
        field, pages, notes = _pick_first(entries)
        content = field.get("content", "")
        source_pages = pages

    merged = {
        "content": content,
        "type": out_type,
        "source_pages": source_pages,
        "bbox": bbox,
        "merge_notes": notes,
    }
    return merged, notes


def _field_order(sample: SamplePages, field_names: list[str]) -> list[str]:
    """Preserve first-seen field order across pages."""
    order: list[str] = []
    seen: set[str] = set()
    for page in sample.pages:
        for raw in page.data:
            name = canonical_field_name(raw)
            if name not in seen:
                order.append(name)
                seen.add(name)
    for name in field_names:
        if name not in seen:
            order.append(name)
    return order


def merge_sample(sample: SamplePages) -> dict[str, Any]:
    occurrences = _field_occurrences(sample)
    field_names = _field_order(sample, list(occurrences.keys()))
    page_heights = {p.page_num: infer_page_height(p.data) for p in sample.pages}
    fields: dict[str, Any] = {}
    merge_log: list[dict[str, Any]] = []

    for name in field_names:
        entries = occurrences.get(name, [])
        if not entries:
            continue
        field_type = entries[0][1].get("type", "Text")
        strategy = resolve_strategy(name, len(entries), field_type=field_type)
        merged, notes = _merge_field(name, entries, strategy, page_heights)
        fields[name] = merged
        merge_log.append(
            {
                "field": name,
                "strategy": strategy.value,
                "source_pages": merged["source_pages"],
                "notes": notes,
            }
        )

    return {
        "sample_id": sample.sample_id,
        "n_pages": sample.n_pages,
        "page_numbers": sample.page_numbers,
        "fields": fields,
        "merge_log": merge_log,
    }
