"""Shared helpers for PDF/image loading and bbox visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PDF_EXTENSIONS = {".pdf"}

DEFAULT_COLORS = [
    (220, 20, 60),
    (30, 144, 255),
    (50, 205, 50),
    (255, 165, 0),
    (148, 0, 211),
    (0, 191, 255),
    (255, 105, 180),
    (154, 205, 50),
    (255, 215, 0),
    (0, 206, 209),
]


def list_inputs(folder: Path, only: str | None = None) -> list[Path]:
    if not folder.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (IMAGE_EXTENSIONS | PDF_EXTENSIONS):
            continue
        if only and only not in p.name:
            continue
        out.append(p)
    return out


def render_pdf_pages(pdf_path: Path, dpi: int) -> list[Any]:
    """Rasterise every page of a PDF to PIL RGB Image via PyMuPDF."""
    import fitz
    from PIL import Image

    pages: list[Any] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            mode = "RGB" if pix.n < 4 else "RGBA"
            img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            pages.append(img.convert("RGB"))
    return pages


def load_units(path: Path, pdf_dpi: int) -> list[tuple[str, Any]]:
    """Return list of (unit_name, PIL.Image) for a single PDF or image path."""
    from PIL import Image

    suffix = path.suffix.lower()
    if suffix in PDF_EXTENSIONS:
        pages = render_pdf_pages(path, pdf_dpi)
        return [(f"{path.stem}_page{i + 1:02d}", img) for i, img in enumerate(pages)]
    return [(path.stem, Image.open(path).convert("RGB"))]


def _load_font(image_size: tuple[int, int], font_scale: float) -> Any:
    from PIL import ImageFont

    w, h = image_size
    size = max(10, int(11 + font_scale * min(w, h) / 200))
    for path in (
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_boxes(
    image: Any,
    entries: list[dict[str, Any]],
    out_path: Path,
    font_scale: float = 1.0,
    color_by: str | None = None,
) -> None:
    """Draw boxes from a list of {label, score?, box_xyxy} dicts onto a PIL image.

    Args:
        image: PIL.Image.Image (will be copied).
        entries: each must have ``box_xyxy`` and ``label``; ``score`` optional.
        out_path: file to write JPEG to (parent dirs auto-created).
        font_scale: scales label font with image min-dimension.
        color_by: if set, assigns one color per distinct value of this key.
    """
    from PIL import ImageDraw

    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font = _load_font(img.size, font_scale)
    line_w = max(2, min(w, h) // 400)

    color_map: dict[str, tuple[int, int, int]] = {}

    def color_for(idx: int, entry: dict[str, Any]) -> tuple[int, int, int]:
        if color_by and color_by in entry:
            key = str(entry[color_by])
            if key not in color_map:
                color_map[key] = DEFAULT_COLORS[len(color_map) % len(DEFAULT_COLORS)]
            return color_map[key]
        return DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]

    for i, e in enumerate(entries):
        box = e["box_xyxy"]
        label = str(e.get("label", ""))
        score = e.get("score")
        color = color_for(i, e)
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline=color, width=line_w)
        short = label if len(label) <= 48 else label[:45] + "…"
        cap = f"{short} ({score:.2f})" if isinstance(score, (int, float)) else short
        tw, th = draw.textbbox((0, 0), cap, font=font)[2:]
        pad = 2
        bg0 = (x0, max(0, y0 - th - 2 * pad))
        bg1 = (x0 + tw + 2 * pad, y0)
        draw.rectangle([bg0, bg1], fill=color)
        draw.text((x0 + pad, bg0[1] + pad), cap, fill=(255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=92)
