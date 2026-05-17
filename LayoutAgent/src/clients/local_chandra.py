"""Local Chandra OCR-2 client using HuggingFace Transformers.

Loads the Chandra OCR-2 model via AutoModelForImageTextToText (same as
layoutDectectionChan/src/load_model.py + layout_extract.py) and runs
layout extraction on a single PDF page or image file.

The model is loaded once and held in memory as a module-level singleton to
avoid reloading on every call within the same process.

Usage:
    from clients.local_chandra import LocalChandraClient

    client = LocalChandraClient("datalab-to/chandra-ocr-2", device="cuda:0")
    html = client.run(pdf_path, prompt_path, page=1, dpi=200)
    client.cleanup()          # free GPU memory when done
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

_DIV_START = re.compile(r"<div\b", re.IGNORECASE)

# Bootstrap layoutDectectionChan/src into sys.path for load_model / layout_extract.
_CHAN_SRC = Path(__file__).resolve().parent.parent.parent.parent / "layoutDectectionChan" / "src"
if _CHAN_SRC.is_dir() and str(_CHAN_SRC) not in sys.path:
    sys.path.insert(0, str(_CHAN_SRC))


class LocalChandraClient:
    """Wrapper around Chandra OCR-2 running locally via HuggingFace Transformers.

    Args:
        model_id: HuggingFace model ID or local path.
            Default: "datalab-to/chandra-ocr-2".
        device: PyTorch device string, e.g. "cuda:0" or "cpu".
        max_new_tokens: Maximum tokens to generate per page.
    """

    def __init__(
        self,
        model_id: str = "datalab-to/chandra-ocr-2",
        device: str = "cuda:0",
        max_new_tokens: int = 4096,
    ) -> None:
        from load_model import load_model  # from layoutDectectionChan/src

        print(f"[local_chandra] Loading model {model_id} on {device} ...", flush=True)
        self.model, self.processor = load_model(model_id, device=device)
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        print("[local_chandra] Model loaded.", flush=True)

    def run_on_image(self, page_image: Any, prompt_text: str) -> str:
        """Run Chandra layout extraction on a single PIL RGB image.

        Args:
            page_image: PIL Image (RGB) of one PDF page.
            prompt_text: Full Chandra prompt string.

        Returns:
            HTML string starting at the first <div element.
        """
        from layout_extract import run_layout_on_image  # from layoutDectectionChan/src

        raw = run_layout_on_image(
            prompt_text,
            page_image,
            self.model,
            self.processor,
            max_new_tokens=self.max_new_tokens,
        )
        return _strip_to_html(raw)

    def run(
        self,
        pdf_path: str | Path,
        prompt_path: str | Path,
        *,
        page: int = 1,
        dpi: int = 200,
    ) -> str:
        """Rasterize one PDF page and run Chandra layout extraction on it.

        Args:
            pdf_path: Path to the PDF file.
            prompt_path: Path to the Chandra prompt text file.
            page: 1-based page number.
            dpi: Rasterization DPI.

        Returns:
            HTML string with <div data-bbox data-label> elements.
        """
        import fitz
        from PIL import Image as PILImage

        pdf_path = Path(pdf_path)
        prompt_text = Path(prompt_path).read_text(encoding="utf-8").strip()

        doc = fitz.open(str(pdf_path))
        try:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pg = doc[page - 1]
            pix = pg.get_pixmap(matrix=mat, alpha=False)
            img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
        finally:
            doc.close()

        return self.run_on_image(img, prompt_text)

    def cleanup(self) -> None:
        """Release GPU memory and unload the model."""
        import gc
        import torch

        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[local_chandra] Model unloaded.", flush=True)


def run_local_chandra(
    pdf_path: str | Path,
    prompt_path: str | Path,
    *,
    log_path: str | Path | None = None,
    page: int = 1,
    dpi: int = 200,
    model_id: str = "datalab-to/chandra-ocr-2",
    device: str = "cuda:0",
    max_new_tokens: int = 4096,
    force_rerun: bool = False,
) -> tuple[str, bool]:
    """Run Chandra locally, with file-level caching.

    Checks for an existing log file first. If found and ``force_rerun`` is
    False, returns the cached HTML without loading the model.

    Args:
        pdf_path: Path to the input PDF.
        prompt_path: Path to the Chandra prompt text file.
        log_path: Where to cache the Chandra output log.
        page: 1-based page number.
        dpi: Rasterization DPI.
        model_id: HuggingFace model ID.
        device: PyTorch device string.
        max_new_tokens: Maximum tokens to generate.
        force_rerun: If True, re-run even when a cache exists.

    Returns:
        Tuple of (html_string, was_cached).
    """
    pdf_path = Path(pdf_path)
    if log_path is None:
        log_path = pdf_path.parent / f"{pdf_path.stem}_llm.log"
    else:
        log_path = Path(log_path)

    if not force_rerun and log_path.is_file():
        raw = log_path.read_text(encoding="utf-8")
        html = _strip_to_html(raw)
        if html:
            print(f"[local_chandra] Using cached log: {log_path}", flush=True)
            return html, True

    client = LocalChandraClient(model_id=model_id, device=device, max_new_tokens=max_new_tokens)
    try:
        html = client.run(pdf_path, prompt_path, page=page, dpi=dpi)
    finally:
        client.cleanup()

    from datetime import datetime, timezone

    log_lines = [
        "# Chandra layout LLM log (local transformer)",
        f"# started: {datetime.now(timezone.utc).isoformat()}",
        f"# pdf: {pdf_path}",
        f"# prompt: {prompt_path}",
        f"# model: {model_id} (local)",
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
    print(f"[local_chandra] Wrote log: {log_path}", flush=True)
    return html, False


def _strip_to_html(raw: str) -> str:
    """Return everything from the first <div to end-of-string."""
    m = _DIV_START.search(raw)
    return raw[m.start():].strip() if m else raw.strip()


__all__ = ["LocalChandraClient", "run_local_chandra"]
