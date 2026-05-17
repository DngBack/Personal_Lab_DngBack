"""visualize node: draw schema bounding boxes on the PDF page and save as JPEG.

Rasterizes the source PDF at the configured DPI and overlays colored bounding
boxes for every <div data-schema=...> element in the merged HTML. The annotated
image is saved to the current iteration's output directory for inspection and
comparison with the reference ground-truth image.

Required state keys:
    merged_html (str): Deduplicated schema-aligned HTML.
    page_image_path (str): Path to the PDF to rasterize as the background.
    output_dir (str): Root run directory; iteration sub-dir is created here.
    iteration (int): Current iteration number.

Optional state keys:
    viz_page (int): 1-based PDF page number (default 1).
    viz_dpi (int): Rasterization DPI (default 200).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def visualize_node(state: dict[str, Any]) -> dict[str, Any]:
    """Render schema bounding boxes on the page image and write JPEG to disk.

    Builds the iteration output directory, draws boxes, saves the image, and
    stores the output path back in the state so the evaluate node can load it.

    Args:
        state: Current agent state.

    Returns:
        Partial state update with ``viz_source_path`` pointing to the saved JPEG.
    """
    from utils.image_io import draw_schema_boxes_on_page

    merged_html = state.get("merged_html", "")
    if not merged_html.strip():
        print("[visualize] Warning: merged_html is empty — skipping visualization", flush=True)
        return {}

    iteration = state.get("iteration", 0)
    iter_dir = _iter_dir(state["output_dir"], iteration)
    iter_dir.mkdir(parents=True, exist_ok=True)

    out_path = iter_dir / "schema_boxes.jpg"
    draw_schema_boxes_on_page(
        state["page_image_path"],
        merged_html,
        out_path,
        page=state.get("viz_page", 1),
        dpi=state.get("viz_dpi", 200),
        only_with_schema=True,
    )
    print(f"[visualize] Saved schema boxes → {out_path}", flush=True)
    return {"viz_source_path": str(out_path)}


def _iter_dir(output_dir: str, iteration: int) -> Path:
    """Return the path of the per-iteration sub-directory.

    Args:
        output_dir: Root run output directory.
        iteration: Current iteration number.

    Returns:
        Path to ``<output_dir>/iter_<NN>/``.
    """
    return Path(output_dir) / f"iter_{iteration:02d}"


__all__ = ["visualize_node", "_iter_dir"]
