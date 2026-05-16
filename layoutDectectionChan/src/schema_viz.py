"""Vẽ bbox (chuẩn hoá 0–1000) lên ảnh / trang PDF — giống sample *_schema_boxes.jpg."""
from __future__ import annotations

from pathlib import Path
from typing import Any

_PALETTE = [
    (220, 20, 60), (30, 144, 255), (50, 205, 50), (255, 165, 0),
    (148, 0, 211), (0, 191, 255), (255, 105, 180), (154, 205, 50),
    (255, 215, 0), (0, 206, 209), (199, 21, 133), (32, 178, 170),
    (255, 99, 71), (60, 179, 113), (123, 104, 238), (218, 165, 32),
]


def load_page_image(
    source: str | Path,
    *,
    page: int = 1,
    dpi: int = 200,
) -> Any:
    """PDF (trang `page` 1-based) hoặc ảnh → PIL RGB."""
    from PIL import Image as PILImage

    path = Path(source)
    if path.suffix.lower() == ".pdf":
        import fitz

        doc = fitz.open(str(path))
        try:
            if page < 1 or page > len(doc):
                raise ValueError(f"page {page} out of range 1–{len(doc)}")
            p = doc[page - 1]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = p.get_pixmap(matrix=mat, alpha=False)
            return PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
        finally:
            doc.close()
    return PILImage.open(path).convert("RGB")


def _caption(block: dict[str, Any], max_schema: int = 42, max_snippet: int = 28) -> str:
    schema = (block.get("schema") or "").strip()
    if schema:
        s = schema[:max_schema]
        if len(schema) > max_schema:
            s += "…"
        return s
    lab = (block.get("label") or "").strip()
    text = (block.get("text") or "").replace("\n", " ").strip()
    snip = (text[:max_snippet] + "…") if len(text) > max_snippet else text
    if lab:
        return f"{lab}: {snip}" if snip else lab
    return snip or "?"


def draw_schema_boxes(
    image: Any,
    blocks: list[dict[str, Any]],
    out_path: str | Path,
    *,
    jpeg_quality: int = 92,
    only_with_schema: bool = False,
) -> None:
    """
    `blocks`: như `parse_schema_divs` — `bbox` [x0,y0,x1,y1] 0–1000, optional `schema`, `label`, `text`.
    """
    from PIL import ImageDraw, ImageFont

    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font_sm = ImageFont.load_default()

    w, h = img.size
    use_blocks = [b for b in blocks if not only_with_schema or (b.get("schema") or "").strip()]
    for idx, b in enumerate(use_blocks):
        box = b.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x0, y0, x1, y1 = map(float, box)
        color = _PALETTE[idx % len(_PALETTE)]
        bx0, by0 = x0 / 1000 * w, y0 / 1000 * h
        bx1, by1 = x1 / 1000 * w, y1 / 1000 * h
        draw.rectangle([bx0, by0, bx1, by1], outline=color + (200,), width=2)
        draw.rectangle([bx0, by0, bx0 + 1, by1], fill=color + (35,))
        draw.text((bx0 + 3, by0 + 2), _caption(b), fill=color, font=font_sm)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in (".jpg", ".jpeg"):
        img.save(out_path, quality=jpeg_quality)
    else:
        img.save(out_path)


def draw_from_merged_html(
    source: str | Path,
    merged_html: str,
    out_path: str | Path,
    *,
    page: int = 1,
    dpi: int = 200,
    only_with_schema: bool = False,
) -> None:
    from schema_html_parse import parse_schema_divs

    img = load_page_image(source, page=page, dpi=dpi)
    blocks = parse_schema_divs(merged_html)
    draw_schema_boxes(img, blocks, out_path, only_with_schema=only_with_schema)


def draw_from_layout_values_json(
    source: str | Path,
    layout_values_json: str | Path,
    out_path: str | Path,
    *,
    page: int = 1,
    dpi: int = 200,
    only_with_schema: bool = True,
) -> None:
    import json

    raw = json.loads(Path(layout_values_json).read_text(encoding="utf-8"))
    blocks = raw.get("blocks") or []
    if not isinstance(blocks, list):
        blocks = []
    img = load_page_image(source, page=page, dpi=dpi)
    draw_schema_boxes(img, blocks, out_path, only_with_schema=only_with_schema)
