"""Few-shot VLM extraction with Qwen3-VL-4B-Instruct on GIẤY GỬI TIỀN TIẾT KIỆM.

Loads ``Qwen/Qwen3-VL-4B-Instruct`` (or any --model checkpoint that uses the
``Qwen3VLForConditionalGeneration`` class), builds a 1-shot prompt with the sample
image + golden JSON, then asks for the same JSON on a test PDF/image.

Outputs (per input page):
- ``<unit>_qwenvl.txt``   : raw model text
- ``<unit>_qwenvl.json``  : parsed JSON (if extractable) + source metadata
- ``<unit>_input.jpg``    : the resized test page actually shown to the model
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

from doc_utils import IMAGE_EXTENSIONS, PDF_EXTENSIONS, list_inputs, load_units
from qwenvl_prompts import build_few_shot_messages
from utils.io import extract_first_json_object

DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_SAMPLE_DIR = PROJECT_DIR / "data/samples/GIAY_GUI_TIEN_TIET_KIEM"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen3-VL few-shot extraction on test docs.")
    default_in = PROJECT_DIR / "data/test/GIAY_GUI_TIEN_TIET_KIEM"
    default_out = PROJECT_DIR / "results/qwenvl_giay_gui"
    p.add_argument("--input-dir", type=Path, default=default_in, help="Folder with PDFs/images.")
    p.add_argument("--input-file", type=Path, default=None, help="Single file (image or PDF).")
    p.add_argument("--only", type=str, default=None, help="Substring filter on filenames.")
    p.add_argument("--output-dir", type=Path, default=default_out, help="Output folder.")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HF model id.")
    p.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="device_map: 'auto', 'cuda:0', 'balanced'.",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=("auto", "bfloat16", "float16", "float32"),
        help="'auto' picks bf16 on Ampere+ GPUs.",
    )
    p.add_argument("--pdf-dpi", type=int, default=200, help="PDF render DPI.")
    p.add_argument(
        "--max-pixels",
        type=int,
        default=1_400_000,
        help="Max pixels (H*W) for each image sent to the VLM. Lower to fit GPU.",
    )
    p.add_argument(
        "--sample-image",
        type=Path,
        default=DEFAULT_SAMPLE_DIR / "IMAGE_LAYOUT_GIAY_GUI_TIEN_TIET_KIEM.png",
        help="Few-shot sample image (annotated layout image).",
    )
    p.add_argument(
        "--sample-json",
        type=Path,
        default=DEFAULT_SAMPLE_DIR / "OCR_GIAY_GUI_TIEN_TIET_KIEM.json",
        help="Expected JSON output for the few-shot sample.",
    )
    p.add_argument(
        "--layout-json",
        type=Path,
        default=DEFAULT_SAMPLE_DIR / "layout _GIAY_GUI_TIEN_TIET_KIEM.json",
        help="Schema layout JSON (used only to build the field guide text).",
    )
    p.add_argument("--max-new-tokens", type=int, default=2048, help="Generation budget.")
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; 0 = greedy (deterministic).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompt + render test image, but skip model loading/inference. "
        "Writes the rendered chat prompt to <unit>_prompt.txt for inspection.",
    )
    return p.parse_args()


def resolve_dtype(name: str) -> Any:
    import torch

    if name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return "auto"
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def fit_to_max_pixels(image: Any, max_pixels: int) -> Any:
    """Downscale image so width*height <= max_pixels, preserving aspect ratio."""
    from PIL import Image

    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = (max_pixels / (w * h)) ** 0.5
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return image.resize((new_w, new_h), Image.LANCZOS)


def run() -> int:
    args = parse_args()

    if args.input_file is not None:
        if not args.input_file.is_file():
            print(f"input file not found: {args.input_file}", file=sys.stderr)
            return 1
        inputs_list = [args.input_file]
    else:
        inputs_list = list_inputs(args.input_dir, args.only)
    if not inputs_list:
        print(
            f"No inputs found (images: {sorted(IMAGE_EXTENSIONS)}, pdfs: {sorted(PDF_EXTENSIONS)}).",
            file=sys.stderr,
        )
        return 1

    for required in (args.sample_image, args.sample_json, args.layout_json):
        if not required.is_file():
            print(f"missing required file: {required}", file=sys.stderr)
            return 1

    if args.dry_run:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(args.model)
        model = None  # type: ignore[assignment]
        torch = None  # type: ignore[assignment]
    else:
        import torch  # type: ignore[no-redef]
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        dtype = resolve_dtype(args.dtype)
        print(f"[qwenvl] Loading {args.model} dtype={dtype} device_map={args.device_map}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model,
            dtype=dtype,
            device_map=args.device_map,
        )
        processor = AutoProcessor.from_pretrained(args.model)
        model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "model": args.model,
        "dtype": str(resolve_dtype(args.dtype)) if not args.dry_run else None,
        "max_pixels": args.max_pixels,
        "sample_image": str(args.sample_image),
        "sample_json": str(args.sample_json),
        "pdf_dpi": args.pdf_dpi,
        "dry_run": args.dry_run,
        "items": [],
    }

    for src_path in inputs_list:
        units = load_units(src_path, args.pdf_dpi)
        for unit_name, image in units:
            test_image = fit_to_max_pixels(image, args.max_pixels)
            input_path = args.output_dir / f"{unit_name}_input.jpg"
            test_image.save(input_path, quality=92)

            messages = build_few_shot_messages(
                sample_image_path=args.sample_image,
                sample_json_path=args.sample_json,
                layout_json_path=args.layout_json,
                test_image=test_image,
            )

            if args.dry_run:
                prompt_text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt_path = args.output_dir / f"{unit_name}_prompt.txt"
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(
                        prompt_text
                        if isinstance(prompt_text, str)
                        else json.dumps(prompt_text, ensure_ascii=False)
                    )
                summary["items"].append(
                    {
                        "source": str(src_path),
                        "unit": unit_name,
                        "input_image": str(input_path),
                        "prompt_text": str(prompt_path),
                        "dry_run": True,
                    }
                )
                print(
                    f"[dry] {src_path.name} :: {unit_name} -> prompt written to {prompt_path.name}"
                )
                continue

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_device = model.device
            inputs = {
                k: (v.to(input_device) if hasattr(v, "to") else v) for k, v in inputs.items()
            }

            do_sample = args.temperature > 0
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature if do_sample else 1.0,
                )
            in_len = inputs["input_ids"].shape[-1]
            trimmed = generated[:, in_len:]
            text = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            txt_path = args.output_dir / f"{unit_name}_qwenvl.txt"
            json_path = args.output_dir / f"{unit_name}_qwenvl.json"

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            parsed = extract_first_json_object(text)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source": str(src_path),
                        "unit": unit_name,
                        "input_image": str(input_path),
                        "raw_text_path": str(txt_path),
                        "parsed": parsed,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            summary["items"].append(
                {
                    "source": str(src_path),
                    "unit": unit_name,
                    "input_image": str(input_path),
                    "raw_text": str(txt_path),
                    "parsed_json": str(json_path),
                    "parse_success": parsed is not None and parsed != {},
                }
            )
            print(
                f"[ok] {src_path.name} :: {unit_name} -> "
                f"{len(text)} chars text, parse={'ok' if parsed else 'FAILED'} -> {json_path.name}"
            )

    summary_path = args.output_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
