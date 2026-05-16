#!/usr/bin/env python3
"""
PDF → Chandra OCR 2 → log file with LLM output per page.

Uses helpers from `src/`. Run from anywhere:

    python run_pdf_llm_log.py /path/to/file.pdf
    python run_pdf_llm_log.py /path/to/file.pdf --log ./out.log
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import fitz  # PyMuPDF
from PIL import Image as PILImage
from load_model import load_model
from layout_extract import run_layout_on_image


def _pdf_pages_as_images(pdf_path: Path, dpi: int):
    doc = fitz.open(str(pdf_path))
    try:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        for i in range(len(doc)):
            page = doc[i]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            yield i + 1, PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
    finally:
        doc.close()


def main() -> int:
    default_prompt = _ROOT / "prompt" / "prompt_GGTTK.txt"
    p = argparse.ArgumentParser(description="Log Chandra layout LLM output for each PDF page.")
    p.add_argument("pdf", type=Path, help="Input PDF path")
    p.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Output log path (default: <pdf_stem>_llm.log next to the PDF)",
    )
    p.add_argument("--prompt-file", type=Path, default=default_prompt)
    p.add_argument("--model", type=str, default="datalab-to/chandra-ocr-2")
    p.add_argument("--device-map", type=str, default="cuda:0")
    p.add_argument("--pdf-dpi", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    args = p.parse_args()

    pdf = args.pdf.resolve()
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        print(f"Not a PDF file: {pdf}", file=sys.stderr)
        return 1
    if not args.prompt_file.is_file():
        print(f"Prompt not found: {args.prompt_file}", file=sys.stderr)
        return 1

    log_path = args.log
    if log_path is None:
        log_path = pdf.parent / f"{pdf.stem}_llm.log"
    else:
        log_path = log_path.resolve()

    prompt_text = args.prompt_file.read_text(encoding="utf-8").strip()
    lines: list[str] = [
        f"# Chandra layout LLM log",
        f"# started: {datetime.now(timezone.utc).isoformat()}",
        f"# pdf: {pdf}",
        f"# prompt: {args.prompt_file}",
        f"# model: {args.model}",
        "",
    ]

    print(f"Loading model {args.model} …", flush=True)
    model, processor = load_model(args.model, device=args.device_map)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    model.eval()

    for page_no, img in _pdf_pages_as_images(pdf, args.pdf_dpi):
        print(f"Page {page_no} …", flush=True)
        out = run_layout_on_image(
            prompt_text,
            img,
            model,
            processor,
            max_new_tokens=args.max_new_tokens,
        )
        sep = f"{'=' * 72}\nPAGE {page_no}\n{'=' * 72}\n"
        lines.append(sep)
        lines.append(out)
        lines.append("\n")

    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote log: {log_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
