"""Page layout helpers for detecting cross-page continuations."""

from __future__ import annotations

from typing import Any

DEFAULT_PAGE_HEIGHT = 1200.0
BOTTOM_CUT_RATIO = 0.85
TOP_CUT_RATIO = 0.15


def parse_bbox(bbox: str) -> tuple[float, float, float, float] | None:
    if not bbox or not isinstance(bbox, str):
        return None
    parts = bbox.split()
    if len(parts) < 4:
        return None
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
    except ValueError:
        return None


def infer_page_height(page_fields: dict[str, Any]) -> float:
    """Estimate page height from the lowest bbox edge on a page."""
    max_y = 0.0
    for field in page_fields.values():
        if not isinstance(field, dict):
            continue
        bb = parse_bbox(field.get("bbox", ""))
        if bb:
            max_y = max(max_y, bb[3])
    return max_y if max_y > 0 else DEFAULT_PAGE_HEIGHT


def near_page_bottom(
    bbox: str,
    page_height: float,
    ratio: float = BOTTOM_CUT_RATIO,
) -> bool:
    bb = parse_bbox(bbox)
    if bb is None:
        return False
    return bb[3] >= page_height * ratio


def near_page_top(
    bbox: str,
    page_height: float,
    ratio: float = TOP_CUT_RATIO,
) -> bool:
    bb = parse_bbox(bbox)
    if bb is None:
        return False
    return bb[1] <= page_height * ratio


def has_page_cut_signal(
    prev_bbox: str,
    next_bbox: str,
    prev_page_height: float,
    next_page_height: float,
) -> bool:
    """True when table blocks sit at a typical page-break position."""
    if not prev_bbox or not next_bbox:
        return True
    return near_page_bottom(prev_bbox, prev_page_height) and near_page_top(
        next_bbox, next_page_height
    )
