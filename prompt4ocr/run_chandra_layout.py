"""Layout detection with Chandra OCR 2 on GIẤY GỬI TIỀN TIẾT KIỆM.

Chandra OCR 2 ships an ``ocr_layout`` prompt that returns each region as an HTML
``<div data-bbox="x0 y0 x1 y1" data-label="...">`` with bboxes normalised to
0–1000. We strip those out, optionally match each detected block to a section
in the GIẤY GỬI TIỀN TIẾT KIỆM schema, and render two visualisations:

- ``<unit>_chandra_layout_raw.jpg``    : Chandra's English labels (Text,
  Section-Header, Table, Form, Image, ...), one colour per label.
- ``<unit>_chandra_layout_named.jpg``  : same boxes, but labelled with the best
  matching Vietnamese schema section (e.g. ``Tên khách hàng``,
  ``PHẦN DÀNH CHO NGÂN HÀNG``). Falls back to the raw label when no schema
  match is good enough.
- ``<unit>_chandra_layout_schema.jpg`` : the schema sections drawn directly on
  the test image (for reference / debug).

Per-page artefacts also include the raw HTML (``*_chandra_layout.html``) and a
JSON dump of all parsed boxes (``*_chandra_layout.json``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.hf_env import ensure_writable_huggingface_cache

ensure_writable_huggingface_cache()

from doc_utils import IMAGE_EXTENSIONS, PDF_EXTENSIONS, draw_boxes, list_inputs, load_units

DEFAULT_MODEL = "datalab-to/chandra-ocr-2"
DEFAULT_SAMPLE_DIR = PROJECT_DIR / "data/samples/GIAY_GUI_TIEN_TIET_KIEM"

# Official prompt from datalab-to/chandra (chandra/prompts.py: OCR_LAYOUT_PROMPT).
_ALLOWED_TAGS = [
    "math", "br", "i", "b", "u", "del", "sup", "sub",
    "table", "tr", "td", "p", "th", "div", "pre",
    "h1", "h2", "h3", "h4", "h5", "ul", "ol", "li",
    "input", "a", "span", "img", "hr", "tbody", "small",
    "caption", "strong", "thead", "big", "code", "chem",
]
_ALLOWED_ATTRS = [
    "class", "colspan", "rowspan", "display", "checked",
    "type", "border", "value", "style", "href", "alt",
    "align", "data-bbox", "data-label",
]
_PROMPT_ENDING = (
    f"Only use these tags {_ALLOWED_TAGS}, and these attributes {_ALLOWED_ATTRS}.\n\n"
    "Guidelines:\n"
    "* Inline math: Surround math with <math>...</math> tags.\n"
    "* Tables: Use colspan and rowspan attributes to match table structure.\n"
    "* Formatting: Maintain consistent formatting with the image.\n"
    "* Images: Include a description of any images in the alt attribute of an "
    "<img> tag. Do not fill out the src property.\n"
    "* Forms: Mark checkboxes and radio buttons properly.\n"
    "* Text: join lines together properly into paragraphs using <p>...</p> tags.\n"
    "* Use the simplest possible HTML structure that accurately represents the "
    "content of the block.\n"
    "* Make sure the text is accurate and easy for a human to read and interpret.\n"
)
OCR_LAYOUT_PROMPT = (
    "OCR this image to HTML, arranged as layout blocks. Each layout block should "
    "be a div with the data-bbox attribute representing the bounding box of the "
    "block in x0 y0 x1 y1 format. Bboxes are normalized 0-1000. The data-label "
    "attribute is the label for the block.\n\n"
    "Use the following labels:\n"
    "- Caption\n- Footnote\n- Equation-Block\n- List-Group\n- Page-Header\n"
    "- Page-Footer\n- Image\n- Section-Header\n- Table\n- Text\n- Complex-Block\n"
    "- Code-Block\n- Form\n- Table-Of-Contents\n- Figure\n- Chemical-Block\n"
    "- Diagram\n- Bibliography\n- Blank-Page\n\n"
    + _PROMPT_ENDING
)

DIV_BBOX_RE = re.compile(
    r'<div\b[^>]*data-bbox\s*=\s*"(?P<bbox>[^"]+)"[^>]*data-label\s*=\s*"(?P<label>[^"]+)"',
    re.IGNORECASE | re.DOTALL,
)
DIV_BBOX_RE_ALT = re.compile(
    r'<div\b[^>]*data-label\s*=\s*"(?P<label>[^"]+)"[^>]*data-bbox\s*=\s*"(?P<bbox>[^"]+)"',
    re.IGNORECASE | re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chandra OCR 2 layout detection.")
    default_in = PROJECT_DIR / "data/test/GIAY_GUI_TIEN_TIET_KIEM"
    default_out = PROJECT_DIR / "results/chandra_layout"
    p.add_argument("--input-dir", type=Path, default=default_in)
    p.add_argument("--input-file", type=Path, default=None)
    p.add_argument("--only", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=default_out)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--device-map", type=str, default="auto")
    p.add_argument(
        "--dtype", type=str, default="auto",
        choices=("auto", "bfloat16", "float16", "float32"),
    )
    p.add_argument("--pdf-dpi", type=int, default=200)
    p.add_argument("--max-pixels", type=int, default=1_600_000)
    p.add_argument(
        "--layout-json", type=Path,
        default=DEFAULT_SAMPLE_DIR / "layout _GIAY_GUI_TIEN_TIET_KIEM.json",
        help="Schema layout JSON (used for naming detected boxes).",
    )
    p.add_argument(
        "--page-width-pt", type=float, default=596.0,
        help="PDF page width in points the schema was authored for (A4=595).",
    )
    p.add_argument(
        "--page-height-pt", type=float, default=844.0,
        help="PDF page height in points the schema was authored for (A4=842).",
    )
    p.add_argument(
        "--match-iou-min", type=float, default=0.05,
        help="Minimum IoU between Chandra box and schema section to use the "
        "schema name; below this we fall back to the raw label.",
    )
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


def resolve_dtype(name: str) -> Any:
    import torch

    if name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return "auto"
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def fit_to_max_pixels(image: Any, max_pixels: int) -> Any:
    from PIL import Image

    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = (max_pixels / (w * h)) ** 0.5
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def build_layout_messages(test_image: Any) -> list[dict[str, Any]]:
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": test_image},
            {"type": "text", "text": OCR_LAYOUT_PROMPT},
        ],
    }]


def parse_layout_divs(text: str) -> list[dict[str, Any]]:
    """Pull (bbox=x0 y0 x1 y1 norm 0-1000, label) pairs out of Chandra HTML."""
    boxes: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for regex in (DIV_BBOX_RE, DIV_BBOX_RE_ALT):
        for m in regex.finditer(text):
            bbox_str = m.group("bbox").strip()
            label = m.group("label").strip()
            key = (bbox_str, label)
            if key in seen:
                continue
            seen.add(key)
            parts = bbox_str.replace(",", " ").split()
            if len(parts) != 4:
                continue
            try:
                x0, y0, x1, y1 = (float(v) for v in parts)
            except ValueError:
                continue
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0
            boxes.append({"label": label, "bbox_norm": [x0, y0, x1, y1]})
    return boxes


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(0.0, (bx1 - bx0) * (by1 - by0))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_in_box(point: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def load_schema_sections(path: Path, page_w_pt: float, page_h_pt: float) -> list[dict[str, Any]]:
    """Return schema sections with bbox in normalised [0,1] coords."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for sec in data.get("sections", []):
        layout = sec.get("layout") or {}
        x = layout.get("x")
        y = layout.get("y")
        w = layout.get("width")
        h = layout.get("height")
        if None in (x, y, w, h):
            continue
        x0, y0 = x / page_w_pt, y / page_h_pt
        x1, y1 = (x + w) / page_w_pt, (y + h) / page_h_pt
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(1.0, x1), min(1.0, y1)
        out.append({
            "name": sec.get("name", ""),
            "ord": sec.get("ord", -1),
            "bbox_norm": [x0, y0, x1, y1],
        })
    return out


def match_to_schema(
    chandra_box_norm01: tuple[float, float, float, float],
    schema_sections: list[dict[str, Any]],
    iou_min: float,
) -> tuple[str | None, float]:
    """Best schema section name for a Chandra box (norm [0,1]). Returns (name, score)."""
    best_name: str | None = None
    best_score = 0.0
    cx = (chandra_box_norm01[0] + chandra_box_norm01[2]) / 2
    cy = (chandra_box_norm01[1] + chandra_box_norm01[3]) / 2
    for sec in schema_sections:
        sbox = tuple(sec["bbox_norm"])  # type: ignore[arg-type]
        score = iou(chandra_box_norm01, sbox)  # type: ignore[arg-type]
        if score > best_score:
            best_score = score
            best_name = sec["name"]
    if best_score >= iou_min and best_name is not None:
        return best_name, best_score

    for sec in schema_sections:
        sbox = tuple(sec["bbox_norm"])  # type: ignore[arg-type]
        if center_in_box((cx, cy), sbox):  # type: ignore[arg-type]
            return sec["name"], best_score
    return None, best_score


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

    schema_sections = (
        load_schema_sections(args.layout_json, args.page_width_pt, args.page_height_pt)
        if args.layout_json and args.layout_json.is_file()
        else []
    )
    if schema_sections:
        print(f"[chandra-layout] loaded {len(schema_sections)} schema sections from {args.layout_json.name}")
    else:
        print("[chandra-layout] no schema sections loaded (will only draw raw Chandra labels)")

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = resolve_dtype(args.dtype)
    print(f"[chandra-layout] Loading {args.model} dtype={dtype} device_map={args.device_map}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device_map,
    )
    processor = AutoProcessor.from_pretrained(args.model)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "model": args.model,
        "dtype": str(dtype),
        "max_pixels": args.max_pixels,
        "page_size_pt": [args.page_width_pt, args.page_height_pt],
        "match_iou_min": args.match_iou_min,
        "items": [],
    }

    for src_path in inputs_list:
        units = load_units(src_path, args.pdf_dpi)
        for unit_name, image in units:
            test_image = fit_to_max_pixels(image, args.max_pixels)
            img_w, img_h = test_image.size
            input_path = args.output_dir / f"{unit_name}_input.jpg"
            test_image.save(input_path, quality=92)

            messages = build_layout_messages(test_image)
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {
                k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()
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

            html_path = args.output_dir / f"{unit_name}_chandra_layout.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(text)

            raw_boxes = parse_layout_divs(text)
            for b in raw_boxes:
                x0, y0, x1, y1 = b["bbox_norm"]
                b["bbox_norm01"] = [x0 / 1000.0, y0 / 1000.0, x1 / 1000.0, y1 / 1000.0]
                b["box_xyxy"] = [
                    int(x0 / 1000.0 * img_w),
                    int(y0 / 1000.0 * img_h),
                    int(x1 / 1000.0 * img_w),
                    int(y1 / 1000.0 * img_h),
                ]
                if schema_sections:
                    name, score = match_to_schema(
                        tuple(b["bbox_norm01"]),  # type: ignore[arg-type]
                        schema_sections,
                        args.match_iou_min,
                    )
                    b["schema_name"] = name
                    b["schema_iou"] = round(score, 3)
                else:
                    b["schema_name"] = None
                    b["schema_iou"] = 0.0

            json_path = args.output_dir / f"{unit_name}_chandra_layout.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source": str(src_path),
                        "unit": unit_name,
                        "image_size": [img_w, img_h],
                        "input_image": str(input_path),
                        "raw_html_path": str(html_path),
                        "boxes": raw_boxes,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            raw_entries = [
                {"box_xyxy": b["box_xyxy"], "label": b["label"]} for b in raw_boxes
            ]
            draw_boxes(
                test_image, raw_entries,
                args.output_dir / f"{unit_name}_chandra_layout_raw.jpg",
                font_scale=1.0, color_by="label",
            )

            named_entries: list[dict[str, Any]] = []
            for b in raw_boxes:
                name = b.get("schema_name") or b["label"]
                named_entries.append({
                    "box_xyxy": b["box_xyxy"],
                    "label": name,
                })
            draw_boxes(
                test_image, named_entries,
                args.output_dir / f"{unit_name}_chandra_layout_named.jpg",
                font_scale=1.2, color_by="label",
            )

            if schema_sections:
                schema_entries: list[dict[str, Any]] = []
                for sec in schema_sections:
                    x0, y0, x1, y1 = sec["bbox_norm"]
                    schema_entries.append({
                        "box_xyxy": [
                            int(x0 * img_w),
                            int(y0 * img_h),
                            int(x1 * img_w),
                            int(y1 * img_h),
                        ],
                        "label": sec["name"],
                    })
                draw_boxes(
                    test_image, schema_entries,
                    args.output_dir / f"{unit_name}_chandra_layout_schema.jpg",
                    font_scale=1.2, color_by="label",
                )

            matched = sum(1 for b in raw_boxes if b.get("schema_name"))
            summary["items"].append({
                "source": str(src_path),
                "unit": unit_name,
                "image_size": [img_w, img_h],
                "n_boxes": len(raw_boxes),
                "n_schema_matches": matched,
                "viz_raw": str(args.output_dir / f"{unit_name}_chandra_layout_raw.jpg"),
                "viz_named": str(args.output_dir / f"{unit_name}_chandra_layout_named.jpg"),
                "viz_schema": (
                    str(args.output_dir / f"{unit_name}_chandra_layout_schema.jpg")
                    if schema_sections else None
                ),
                "raw_html": str(html_path),
                "boxes_json": str(json_path),
            })
            print(
                f"[ok] {src_path.name} :: {unit_name} -> "
                f"{len(raw_boxes)} boxes ({matched} schema-named) -> {json_path.name}"
            )

    summary_path = args.output_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
