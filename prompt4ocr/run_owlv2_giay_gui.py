"""Zero-shot object detection with OWLv2 on GIẤY GỬI TIỀN TIẾT KIỆM test images.

Reads images from ``data/test/GIAY_GUI_TIEN_TIET_KIEM/``, runs
``google/owlv2-base-patch16-ensemble`` (or another HF checkpoint), draws boxes, and writes
JSON + visualizations under ``results/owlv2_giay_gui/``.

Dependencies: ``pip install -r requirements_owlv2.txt`` (see repo prompt4ocr folder).
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

from doc_utils import (
    IMAGE_EXTENSIONS,
    PDF_EXTENSIONS,
    draw_boxes,
    list_inputs,
    load_units,
)
from owlv2_prompts import queries_for_profile, text_labels_for_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OWLv2 detection for GIAY_GUI_TIEN_TIET_KIEM test docs.")
    default_in = PROJECT_DIR / "data/test/GIAY_GUI_TIEN_TIET_KIEM"
    default_out = PROJECT_DIR / "results/owlv2_giay_gui"
    p.add_argument("--input-dir", type=Path, default=default_in, help="Folder with test images/PDFs.")
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
    p.add_argument("--output-dir", type=Path, default=default_out, help="Folder for JSON + bbox images.")
    p.add_argument(
        "--model",
        type=str,
        default="google/owlv2-base-patch16-ensemble",
        help="Hugging Face OWLv2 object-detection checkpoint.",
    )
    p.add_argument("--threshold", type=float, default=0.12, help="Min confidence for post-processing.")
    p.add_argument(
        "--prompt-profile",
        type=str,
        default="full",
        choices=("full", "compact"),
        help="full = more queries (better recall); compact = faster smoke test.",
    )
    p.add_argument("--device", type=str, default="auto", help="cuda, cpu, or auto.")
    p.add_argument("--max-detections", type=int, default=80, help="Cap boxes drawn per image (by score).")
    p.add_argument("--dpi-font-scale", type=float, default=1.0, help="Scale label font vs image size.")
    p.add_argument("--pdf-dpi", type=int, default=200, help="Render DPI when rasterising PDF pages.")
    return p.parse_args()


def pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _move_batch_to_device(batch: Any, device: str) -> Any:
    import torch

    if hasattr(batch, "to"):
        return batch.to(device)
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def run() -> int:
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    args = parse_args()
    device = pick_device(args.device)

    if args.input_file is not None:
        if not args.input_file.is_file():
            print(f"Input file not found: {args.input_file}", file=sys.stderr)
            return 1
        inputs_list = [args.input_file]
    else:
        inputs_list = list_inputs(args.input_dir, args.only)

    if not inputs_list:
        print(
            f"No inputs found in {args.input_dir} (images: {sorted(IMAGE_EXTENSIONS)}, "
            f"pdfs: {sorted(PDF_EXTENSIONS)}). Add files and re-run.",
            file=sys.stderr,
        )
        return 1

    queries = queries_for_profile(args.prompt_profile)
    text_labels = text_labels_for_batch(queries)

    processor = Owlv2Processor.from_pretrained(args.model)
    model = Owlv2ForObjectDetection.from_pretrained(args.model)
    model.to(device)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "model": args.model,
        "device": device,
        "threshold": args.threshold,
        "prompt_profile": args.prompt_profile,
        "num_queries": len(queries),
        "pdf_dpi": args.pdf_dpi,
        "items": [],
    }

    for src_path in inputs_list:
        units = load_units(src_path, args.pdf_dpi)
        for unit_name, image in units:
            inputs = processor(
                text=text_labels,
                images=image,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
            inputs = _move_batch_to_device(inputs, device)
            with torch.inference_mode():
                outputs = model(**inputs)

            target_sizes = torch.tensor([[image.height, image.width]])
            results = processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=args.threshold,
                text_labels=text_labels,
            )
            r0 = results[0]
            boxes = r0["boxes"]
            scores = r0["scores"]
            tlabels = r0["text_labels"]

            entries: list[dict[str, Any]] = []
            for box, score, tl in zip(boxes, scores, tlabels):
                entries.append(
                    {
                        "label": tl,
                        "score": float(score.item()),
                        "box_xyxy": [float(x) for x in box.tolist()],
                    }
                )
            entries.sort(key=lambda x: x["score"], reverse=True)
            entries = entries[: args.max_detections]

            json_path = args.output_dir / f"{unit_name}_owlv2.json"
            viz_path = args.output_dir / f"{unit_name}_owlv2_boxes.jpg"

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source": str(src_path),
                        "unit": unit_name,
                        "image_size_hw": [image.height, image.width],
                        "detections": entries,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            draw_boxes(image, entries, viz_path, font_scale=args.dpi_font_scale, color_by="label")
            summary["items"].append(
                {
                    "source": str(src_path),
                    "unit": unit_name,
                    "json": str(json_path),
                    "visualization": str(viz_path),
                    "num_boxes": len(entries),
                }
            )
            print(f"[ok] {src_path.name} :: {unit_name} -> {len(entries)} boxes -> {viz_path.name}")

    summary_path = args.output_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
