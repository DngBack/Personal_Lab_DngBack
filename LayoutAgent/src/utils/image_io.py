"""Image loading and encoding utilities.

Provides helpers for:
- Rasterizing a PDF page to a PIL Image
- Encoding PIL Images or file paths to base64 strings for OpenAI API calls
- Saving visualization images to disk
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any


def load_page_image(
    source: str | Path,
    *,
    page: int = 1,
    dpi: int = 200,
) -> Any:
    """Load a PDF page or image file as a PIL RGB Image.

    For PDF sources, the specified page is rasterized at the given DPI.
    For image sources (JPG, PNG, etc.), the file is opened directly.

    Args:
        source: Path to a PDF file or image file.
        page: 1-based page number (only applies to PDF files).
        dpi: Rasterization DPI (only applies to PDF files).

    Returns:
        PIL Image in RGB mode.
    """
    from PIL import Image as PILImage

    path = Path(source)
    if path.suffix.lower() == ".pdf":
        import fitz

        doc = fitz.open(str(path))
        try:
            if page < 1 or page > len(doc):
                raise ValueError(f"Page {page} out of range 1–{len(doc)}")
            pg = doc[page - 1]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = pg.get_pixmap(matrix=mat, alpha=False)
            return PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
        finally:
            doc.close()

    return PILImage.open(path).convert("RGB")


def pil_to_base64_jpeg(image: Any, quality: int = 92) -> str:
    """Encode a PIL Image as a base64 JPEG string.

    Args:
        image: PIL Image object.
        quality: JPEG quality factor (1–95).

    Returns:
        Base64-encoded JPEG string (no data URL prefix).
    """
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def file_to_base64(path: str | Path) -> str:
    """Read an image file and return its base64 encoding.

    Args:
        path: Path to image file.

    Returns:
        Base64-encoded file contents.
    """
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def pil_to_data_url(image: Any, quality: int = 92) -> str:
    """Encode a PIL Image as an OpenAI-compatible base64 JPEG data URL.

    Args:
        image: PIL Image object.
        quality: JPEG quality factor.

    Returns:
        Data URL string: ``data:image/jpeg;base64,<data>``.
    """
    data = pil_to_base64_jpeg(image, quality=quality)
    return f"data:image/jpeg;base64,{data}"


def save_image(image: Any, out_path: str | Path, *, jpeg_quality: int = 92) -> None:
    """Save a PIL Image to disk, creating parent directories as needed.

    Args:
        image: PIL Image to save.
        out_path: Destination file path (.jpg, .jpeg, or .png).
        jpeg_quality: Quality factor when saving as JPEG.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in (".jpg", ".jpeg"):
        image.save(out_path, quality=jpeg_quality)
    else:
        image.save(out_path)


def draw_schema_boxes_on_page(
    source: str | Path,
    merged_html: str,
    out_path: str | Path,
    *,
    page: int = 1,
    dpi: int = 200,
    only_with_schema: bool = True,
) -> None:
    """Rasterize a PDF page and draw schema bounding boxes on it.

    Delegates to the layoutDectectionChan schema_viz module. Boxes are drawn
    from the data-bbox and data-schema attributes in the merged HTML string.

    Args:
        source: Path to the PDF (or image) to rasterize as the background.
        merged_html: Schema-aligned HTML containing <div data-bbox=...> elements.
        out_path: Output path for the annotated JPEG image.
        page: 1-based page number.
        dpi: Rasterization DPI.
        only_with_schema: If True, only draw boxes that have a non-empty data-schema.
    """
    import sys
    from pathlib import Path as _Path

    # Bootstrap layoutDectectionChan into path (schema_html does this already,
    # but we import directly here to avoid a circular dependency).
    _chan_src = _Path(__file__).resolve().parent.parent.parent.parent / "layoutDectectionChan" / "src"
    if str(_chan_src) not in sys.path:
        sys.path.insert(0, str(_chan_src))

    from schema_viz import draw_from_merged_html

    draw_from_merged_html(
        source,
        merged_html,
        out_path,
        page=page,
        dpi=dpi,
        only_with_schema=only_with_schema,
    )


__all__ = [
    "load_page_image",
    "pil_to_base64_jpeg",
    "file_to_base64",
    "pil_to_data_url",
    "save_image",
    "draw_schema_boxes_on_page",
]
