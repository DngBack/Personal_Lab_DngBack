"""Chandra OCR-2 API client.

Calls the Chandra OCR-2 model to extract layout HTML from a PDF page image.
The model is accessed via an OpenAI-compatible endpoint configured by:

    CHANDRA_API_KEY   – API key (falls back to OPENAI_API_KEY)
    CHANDRA_BASE_URL  – endpoint URL (falls back to OPENAI_BASE_URL)
    CHANDRA_MODEL     – model name (default: "datalab-to/chandra-ocr-2")

The PDF page is rasterized at the given DPI and sent as a base64-encoded image.
Results are cached to disk: if a log file already exists for the PDF, its HTML
is returned directly without calling the API again.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from clients.openai_vision import chat_vision, image_to_data_url, make_openai_client

_DEFAULT_MODEL = "datalab-to/chandra-ocr-2"
_DIV_START = re.compile(r"<div\b", re.IGNORECASE)


def _make_chandra_client():
    """Build an OpenAI client pointed at the Chandra endpoint."""
    api_key = os.environ.get("CHANDRA_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("CHANDRA_BASE_URL") or os.environ.get("OPENAI_BASE_URL")

    from openai import OpenAI

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _strip_to_html(raw: str) -> str:
    """Return everything from the first <div to end-of-string."""
    m = _DIV_START.search(raw)
    return raw[m.start():].strip() if m else raw.strip()


def run_chandra_on_image(
    page_image: Any,
    prompt_text: str,
    *,
    model: str | None = None,
    max_tokens: int = 4096,
) -> str:
    """Call Chandra OCR-2 on a single PIL page image.

    Args:
        page_image: PIL Image (RGB) of one PDF page, rasterized at the target DPI.
        prompt_text: Chandra prompt text (e.g. contents of prompt_GGTTK.txt).
        model: Chandra model name. Defaults to CHANDRA_MODEL env var or "datalab-to/chandra-ocr-2".
        max_tokens: Maximum tokens in the Chandra response.

    Returns:
        HTML string starting with the first <div element.
    """
    model = model or os.environ.get("CHANDRA_MODEL", _DEFAULT_MODEL)
    client = _make_chandra_client()
    raw = chat_vision(
        client,
        model=model,
        system_prompt="",
        user_text=prompt_text,
        images=[page_image],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return _strip_to_html(raw)


def load_or_run_chandra(
    pdf_path: str | Path,
    prompt_path: str | Path,
    *,
    log_path: str | Path | None = None,
    page: int = 1,
    dpi: int = 200,
    model: str | None = None,
    max_tokens: int = 4096,
    force_rerun: bool = False,
) -> tuple[str, bool]:
    """Return Chandra HTML for a PDF page, using cache when available.

    If a cached log file exists and ``force_rerun`` is False, the HTML is
    extracted from it without calling the API. Otherwise the PDF page is
    rasterized and sent to Chandra OCR-2.

    Args:
        pdf_path: Path to the input PDF file.
        prompt_path: Path to the Chandra prompt text file.
        log_path: Optional path for the cached log file.
            Defaults to ``<pdf_stem>_llm.log`` next to the PDF.
        page: 1-based page number to process.
        dpi: Rasterization DPI for the PDF page.
        model: Chandra model name override.
        max_tokens: Maximum tokens in the Chandra response.
        force_rerun: If True, always call the API even if a cache exists.

    Returns:
        Tuple of (html_string, was_cached) where was_cached is True when the
        result came from an existing log file.
    """
    pdf_path = Path(pdf_path)
    prompt_path = Path(prompt_path)

    if log_path is None:
        log_path = pdf_path.parent / f"{pdf_path.stem}_llm.log"
    else:
        log_path = Path(log_path)

    if not force_rerun and log_path.is_file():
        raw = log_path.read_text(encoding="utf-8")
        html = _strip_to_html(raw)
        if html:
            print(f"[chandra] Using cached log: {log_path}", flush=True)
            return html, True

    print(f"[chandra] Running OCR on {pdf_path} page {page} (dpi={dpi}) …", flush=True)
    import fitz
    from PIL import Image as PILImage

    doc = fitz.open(str(pdf_path))
    try:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pg = doc[page - 1]
        pix = pg.get_pixmap(matrix=mat, alpha=False)
        img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
    finally:
        doc.close()

    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    html = run_chandra_on_image(img, prompt_text, model=model, max_tokens=max_tokens)

    from datetime import datetime, timezone

    log_lines = [
        "# Chandra layout LLM log",
        f"# started: {datetime.now(timezone.utc).isoformat()}",
        f"# pdf: {pdf_path}",
        f"# prompt: {prompt_path}",
        f"# model: {model or os.environ.get('CHANDRA_MODEL', _DEFAULT_MODEL)}",
        "",
        f"{'=' * 72}",
        f"PAGE {page}",
        f"{'=' * 72}",
        "",
        html,
        "",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"[chandra] Wrote log: {log_path}", flush=True)
    return html, False


__all__ = ["run_chandra_on_image", "load_or_run_chandra"]
