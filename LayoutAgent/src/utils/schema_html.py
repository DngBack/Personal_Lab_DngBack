"""Thin adapter over layoutDectectionChan's schema HTML parsing utilities.

Adds the layoutDectectionChan src/ directory to sys.path on first import so
that code in LayoutAgent can reuse the parsing and visualization logic without
duplicating it. All public symbols are re-exported from this module.

Environment variable CHAN_SRC_DIR can override the auto-detected path.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap — locate layoutDectectionChan/src relative to this file
# ---------------------------------------------------------------------------

_SELF = Path(__file__).resolve()
# LayoutAgent/src/utils/schema_html.py  →  workspace root = 3 levels up
_WORKSPACE = _SELF.parent.parent.parent.parent

_CHAN_SRC = Path(os.environ.get("CHAN_SRC_DIR", "")).resolve() if os.environ.get("CHAN_SRC_DIR") else (
    _WORKSPACE / "layoutDectectionChan" / "src"
)

if not _CHAN_SRC.is_dir():
    raise ImportError(
        f"layoutDectectionChan/src not found at {_CHAN_SRC}. "
        "Set the CHAN_SRC_DIR environment variable to the correct path."
    )

if str(_CHAN_SRC) not in sys.path:
    sys.path.insert(0, str(_CHAN_SRC))

# ---------------------------------------------------------------------------
# Re-export canonical symbols from layoutDectectionChan
# ---------------------------------------------------------------------------

from schema_html_parse import (  # noqa: E402
    attach_values_to_layout,
    blocks_to_flat_map,
    extract_chandra_html_from_log,
    layout_with_extractions_to_json,
    merge_duplicate_schema_divs,
    parse_schema_divs,
)
from layout_json_utils import collect_all_names, load_layout_names  # noqa: E402


def get_extracted_schema_names(merged_html: str) -> list[str]:
    """Return the list of non-empty data-schema values found in merged HTML.

    Args:
        merged_html: HTML string produced by the schema-alignment step.

    Returns:
        Unique, ordered list of schema field names that were successfully assigned.
    """
    blocks = parse_schema_divs(merged_html)
    seen: set[str] = set()
    result: list[str] = []
    for b in blocks:
        sk = (b.get("schema") or "").strip()
        if sk and sk not in seen:
            seen.add(sk)
            result.append(sk)
    return result


def build_layout_payload(
    layout_json_path: str | Path,
    merged_html: str,
) -> dict[str, Any]:
    """Parse merged HTML and attach extracted values to the layout JSON tree.

    Args:
        layout_json_path: Path to the layout JSON file.
        merged_html: Schema-aligned HTML (output of align + dedup steps).

    Returns:
        Dictionary with keys: layout_with_values, flat_by_schema, blocks.
    """
    return layout_with_extractions_to_json(str(layout_json_path), merged_html)


__all__ = [
    "load_layout_names",
    "collect_all_names",
    "extract_chandra_html_from_log",
    "parse_schema_divs",
    "merge_duplicate_schema_divs",
    "blocks_to_flat_map",
    "attach_values_to_layout",
    "layout_with_extractions_to_json",
    "get_extracted_schema_names",
    "build_layout_payload",
]
