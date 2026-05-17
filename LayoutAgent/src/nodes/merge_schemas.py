"""merge_schemas node: deduplicate divs that share the same data-schema value.

The schema-alignment model sometimes emits two or more <div> elements for the
same schema field (e.g. one for the role label and one for the signature). This
node uses the deterministic ``merge_duplicate_schema_divs`` function from
layoutDectectionChan to collapse them into a single div with a union bounding
box.

No LLM call is made in this node — it is a pure, rule-based transformation.

Required state keys:
    merged_html (str): HTML string from the align_schema node.

Returns:
    Partial state update with ``merged_html`` (deduplicated).
"""
from __future__ import annotations

from typing import Any


def merge_schemas_node(state: dict[str, Any]) -> dict[str, Any]:
    """Merge duplicate data-schema divs into single union-bbox divs.

    Iterates over all <div> elements in the merged HTML and collapses any
    group that shares the same non-empty data-schema value into one div whose
    bounding box is the minimum enclosing rectangle of the group.

    Args:
        state: Current agent state with ``merged_html``.

    Returns:
        Partial state update with deduplicated ``merged_html``.
    """
    from utils.schema_html import merge_duplicate_schema_divs

    html_in = state.get("merged_html", "")
    if not html_in.strip():
        print("[merge_schemas] Warning: merged_html is empty", flush=True)
        return {}

    html_out = merge_duplicate_schema_divs(html_in)

    divs_before = html_in.count("<div")
    divs_after = html_out.count("<div")
    removed = divs_before - divs_after
    if removed:
        print(
            f"[merge_schemas] Merged {removed} duplicate div(s): "
            f"{divs_before} → {divs_after} divs",
            flush=True,
        )
    else:
        print(f"[merge_schemas] No duplicates found ({divs_after} divs)", flush=True)

    return {"merged_html": html_out}


__all__ = ["merge_schemas_node"]
