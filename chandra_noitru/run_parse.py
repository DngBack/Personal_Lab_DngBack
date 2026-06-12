#!/usr/bin/env python3
"""Parse PDF nội trú bằng Chandra OCR 2 (tối đa N trang mỗi file).

Ví dụ:
    python chandra_noitru/run_parse.py --device-map cuda:1
    python chandra_noitru/run_parse.py --input-file data/pdfs/2300030376-6263.pdf --max-pages 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "chandra4layout"))

from run import (  # noqa: E402
    _DEFAULT_MODEL,
    _draw_blocks,
    _fit,
    _parse_blocks,
    _resolve_dtype,
    _sanitize,
)

_DEFAULT_PROMPT = _HERE / "prompts/noitru_ocr.txt"
_DEFAULT_PDF_DIR = _HERE / "data/pdfs"
_DEFAULT_OUTPUT = _HERE / "results"

_DEFAULT_PDFS = [
    "2300030376-6263.pdf",
    "2300033911-5962.pdf",
    "2500064072-3637.pdf",
    "2500077856-33.pdf",
]


def _load_pdf_pages(src: Path, dpi: int, max_pages: int):
    """Yield (unit_name, PIL.Image, page_pt) cho tối đa max_pages trang đầu."""
    import fitz
    from PIL import Image as PILImage

    doc = fitz.open(str(src))
    n = min(len(doc), max_pages)
    for page_idx in range(n):
        page = doc[page_idx]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
        rect = page.rect
        yield (
            f"{src.stem}_page{page_idx + 1:02d}",
            img,
            (rect.width, rect.height),
        )
    doc.close()


def _resolve_inputs(
    input_dir: Path,
    input_files: list[Path] | None,
    only: str | None,
) -> list[Path]:
    if input_files:
        return [p for p in input_files if p.is_file()]

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        pdfs = [input_dir / name for name in _DEFAULT_PDFS]
    if only:
        pdfs = [p for p in pdfs if only in p.name]
    return pdfs


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chandra OCR 2 → parse PDF nội trú (2 trang).")
    p.add_argument("--input-dir", type=Path, default=_DEFAULT_PDF_DIR)
    p.add_argument(
        "--input-file",
        type=Path,
        action="append",
        default=None,
        help="Một hoặc nhiều PDF (có thể lặp flag).",
    )
    p.add_argument("--only", type=str, default=None, help="Lọc tên file chứa chuỗi này.")
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    p.add_argument("--prompt-file", type=Path, default=_DEFAULT_PROMPT)
    p.add_argument("--model", type=str, default=_DEFAULT_MODEL)
    p.add_argument("--device-map", type=str, default="cuda:1")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=("auto", "bfloat16", "float16", "float32"))
    p.add_argument("--pdf-dpi", type=int, default=200)
    p.add_argument("--max-pages", type=int, default=2)
    p.add_argument("--max-pixels", type=int, default=1_600_000)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


def run() -> int:
    args = _parse_args()

    inputs = _resolve_inputs(args.input_dir, args.input_file, args.only)
    missing = [p for p in inputs if not p.is_file()]
    if missing:
        print("Thiếu file PDF:", file=sys.stderr)
        for p in missing:
            print(f"  - {p}", file=sys.stderr)
        print(
            f"\nCopy 4 PDF vào {args.input_dir} hoặc dùng --input-file <đường_dẫn>.",
            file=sys.stderr,
        )
        return 1

    prompt_text = args.prompt_file.read_text(encoding="utf-8").strip()
    print(f"[run] Prompt: {args.prompt_file.name} ({len(prompt_text)} chars)")
    print(f"[run] PDFs: {len(inputs)} file, max {args.max_pages} trang/file")

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = _resolve_dtype(args.dtype)
    print(f"[run] Loading {args.model}  dtype={dtype}  device={args.device_map}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=dtype, device_map=args.device_map,
    )
    processor = AutoProcessor.from_pretrained(args.model)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "model": args.model,
        "max_pages": args.max_pages,
        "items": [],
    }

    for src_path in inputs:
        for unit_name, image, page_pt in _load_pdf_pages(
            src_path, args.pdf_dpi, args.max_pages
        ):
            page_w_pt, page_h_pt = page_pt
            img = _fit(image, args.max_pixels)

            print(f"[{unit_name}] Inference …")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt_text},
            ]}]
            inputs_t = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
            inputs_t = {
                k: (v.to(model.device) if hasattr(v, "to") else v)
                for k, v in inputs_t.items()
            }
            do_sample = args.temperature > 0
            with torch.inference_mode():
                gen = model.generate(
                    **inputs_t,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature if do_sample else 1.0,
                )
            raw = _sanitize(processor.batch_decode(
                gen[:, inputs_t["input_ids"].shape[-1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0])

            out_dir = args.output_dir
            (out_dir / f"{unit_name}_raw.html").write_text(raw, encoding="utf-8")

            blocks = _parse_blocks(raw)
            print(f"[{unit_name}] {len(blocks)} blocks parsed")

            blocks_out = {
                "source": str(src_path),
                "unit": unit_name,
                "image_size": list(img.size),
                "page_size_pt": [page_w_pt, page_h_pt],
                "n_blocks": len(blocks),
                "blocks": blocks,
            }

            json_path = out_dir / f"{unit_name}_blocks.json"
            json_path.write_text(
                json.dumps(blocks_out, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            viz_path = out_dir / f"{unit_name}_layout.jpg"
            _draw_blocks(img, blocks, viz_path)

            md_path = out_dir / f"{unit_name}.md"
            md_lines = [f"# {unit_name}", "", f"Nguồn: `{src_path.name}`", ""]
            for b in blocks:
                md_lines.append(f"## [{b['label']}]")
                md_lines.append("")
                md_lines.append(b.get("text") or "")
                md_lines.append("")
            md_path.write_text("\n".join(md_lines), encoding="utf-8")

            print(f"[ok] {unit_name} → {json_path.name}, {viz_path.name}")
            summary["items"].append({
                "unit": unit_name,
                "source": str(src_path),
                "n_blocks": len(blocks),
                "raw_html": str(out_dir / f"{unit_name}_raw.html"),
                "blocks_json": str(json_path),
                "markdown": str(md_path),
                "viz": str(viz_path),
            })

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] {len(summary['items'])} trang → {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
