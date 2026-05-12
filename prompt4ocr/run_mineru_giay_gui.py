"""MinerU2.5-Pro document parsing on GIẤY GỬI TIỀN TIẾT KIỆM test docs.

Loads ``opendatalab/MinerU2.5-Pro-2604-1.2B`` (Qwen2-VL), runs the two-step extract
(layout detection + content recognition) per page, and writes per-page structured
JSON, Markdown, and a bbox visualization colored by content type.

Bbox in MinerU output is normalised to [0, 1] (relative to image w/h); we rescale
to pixel coordinates before drawing.

Dependencies: ``pip install -r requirements_mineru.txt`` (see folder).
Docs: https://huggingface.co/opendatalab/MinerU2.5-Pro-2604-1.2B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.hf_env import ensure_writable_huggingface_cache

ensure_writable_huggingface_cache()

from doc_utils import IMAGE_EXTENSIONS, PDF_EXTENSIONS, draw_boxes, list_inputs, load_units

DEFAULT_MODEL = "opendatalab/MinerU2.5-Pro-2604-1.2B"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MinerU2.5 document parsing for test docs.")
    default_in = PROJECT_DIR / "data/test/GIAY_GUI_TIEN_TIET_KIEM"
    default_out = PROJECT_DIR / "results/mineru_giay_gui"
    p.add_argument("--input-dir", type=Path, default=default_in, help="Folder with PDFs/images.")
    p.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Optional single file (image or PDF). Overrides --input-dir scan.",
    )
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="Substring filter: only process files whose name contains this string.",
    )
    p.add_argument("--output-dir", type=Path, default=default_out, help="Folder for JSON+MD+viz.")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HF model id or local path.")
    p.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="device_map for transformers; e.g. 'auto', 'cuda:0', 'balanced'.",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=("auto", "bfloat16", "float16", "float32"),
        help="Compute dtype. 'auto' lets transformers pick (bf16 on A30).",
    )
    p.add_argument("--pdf-dpi", type=int, default=220, help="Render DPI for PDFs.")
    p.add_argument(
        "--image-analysis",
        action="store_true",
        help="Enable image/chart content analysis (slower).",
    )
    p.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip writing the Markdown file (only keep JSON+viz).",
    )
    p.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Scale label font size for the visualization.",
    )
    return p.parse_args()


def resolve_dtype(name: str) -> Any:
    import torch

    if name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return "auto"
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def block_to_entry(block: dict[str, Any], image_w: int, image_h: int) -> dict[str, Any]:
    """Convert MinerU ContentBlock dict-view to a draw_boxes entry (pixel coords)."""
    x0n, y0n, x1n, y1n = block.get("bbox", [0.0, 0.0, 0.0, 0.0])
    return {
        "label": str(block.get("type", "unknown")),
        "score": None,
        "box_xyxy": [x0n * image_w, y0n * image_h, x1n * image_w, y1n * image_h],
    }


def block_to_jsonable(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": block.get("type"),
        "bbox_normalized": list(block.get("bbox", [])),
        "angle": block.get("angle"),
        "content": block.get("content"),
    }


def run() -> int:
    import torch
    from mineru_vl_utils import MinerUClient
    from mineru_vl_utils.post_process import json2md
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    args = parse_args()

    if args.input_file is not None:
        if not args.input_file.is_file():
            print(f"Input file not found: {args.input_file}", file=sys.stderr)
            return 1
        inputs_list = [args.input_file]
    else:
        inputs_list = list_inputs(args.input_dir, args.only)

    if not inputs_list:
        print(
            f"No inputs found (images: {sorted(IMAGE_EXTENSIONS)}, pdfs: {sorted(PDF_EXTENSIONS)}). "
            "Put files into data/test/... and re-run.",
            file=sys.stderr,
        )
        return 1

    dtype = resolve_dtype(args.dtype)
    print(f"[mineru] Loading {args.model} dtype={dtype} device_map={args.device_map}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device_map,
    )
    processor = AutoProcessor.from_pretrained(args.model, use_fast=True)
    client = MinerUClient(
        backend="transformers",
        model=model,
        processor=processor,
        image_analysis=args.image_analysis,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "model": args.model,
        "dtype": str(dtype),
        "image_analysis": args.image_analysis,
        "pdf_dpi": args.pdf_dpi,
        "items": [],
    }

    for src_path in inputs_list:
        units = load_units(src_path, args.pdf_dpi)
        for unit_name, image in units:
            with torch.inference_mode():
                result = client.two_step_extract(image)

            # ExtractResult behaves like list[ContentBlock]; ContentBlock is a dict-like
            # with keys: type, bbox (normalized [0,1] xyxy), content, angle.
            blocks: list[dict[str, Any]] = [dict(b) for b in result]
            jsonable = [block_to_jsonable(b) for b in blocks]
            entries = [block_to_entry(b, image.width, image.height) for b in blocks]

            json_path = args.output_dir / f"{unit_name}_mineru.json"
            md_path = args.output_dir / f"{unit_name}_mineru.md"
            viz_path = args.output_dir / f"{unit_name}_mineru_boxes.jpg"

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source": str(src_path),
                        "unit": unit_name,
                        "image_size_hw": [image.height, image.width],
                        "blocks": jsonable,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            if not args.no_markdown:
                try:
                    md_text = json2md(blocks)
                except Exception as exc:  # noqa: BLE001
                    md_text = f"<!-- json2md failed: {exc} -->\n"
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(md_text)

            draw_boxes(image, entries, viz_path, font_scale=args.font_scale, color_by="label")

            summary["items"].append(
                {
                    "source": str(src_path),
                    "unit": unit_name,
                    "json": str(json_path),
                    "markdown": None if args.no_markdown else str(md_path),
                    "visualization": str(viz_path),
                    "num_blocks": len(blocks),
                    "block_types": sorted({str(b.get("type")) for b in blocks}),
                }
            )
            print(
                f"[ok] {src_path.name} :: {unit_name} -> "
                f"{len(blocks)} blocks -> {viz_path.name}"
            )

    summary_path = args.output_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
