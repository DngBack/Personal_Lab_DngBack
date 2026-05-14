"""GIẤY GỬI TIỀN TIẾT KIỆM: Chandra OCR 2 layout HTML + schema bbox overlay.

Pipeline (single page):

  1. Render the test PDF/image to a PIL image (PDF page size in pt comes from
     PyMuPDF; raster inputs default to 596×844 pt for scaling).
  2. Run Chandra OCR 2 with the official ``ocr_layout`` prompt; by default the
     **processor** (chat template, tokenizer, image preprocessing) is loaded from
     ``--model`` even when ``--lora-path`` is set, matching non-finetuned inference
     and ``train_chandra_layout_lora``. Use ``--processor-from-lora`` only if you
     intentionally ship a tokenizer/processor inside the adapter directory.
  3. Draw **schema bboxes only** from ``--layout-json``. Coordinates are treated as
     authored in ``--schema-template-pt`` (default 596×844) and **linearly scaled**
     to each document's ``page.rect`` before mapping to pixels (fixes mild mismatch
     when PDF MediaBox differs). Pass ``--schema-template-pt 0 0`` for legacy behaviour
     (assume JSON pt already matches each page).
  4. Optional **prompt append**: ``--giay-gui-tien-layout-guide`` and/or
     ``--extra-layout-prompt-file`` (UTF-8) after the base OCR layout instructions.
  5. Render the visualisation with smart label placement (non-overlapping labels).

Outputs (per page) under ``--output-dir``:
  * ``<unit>_input.jpg``            : the resized test page.
  * ``<unit>_chandra_layout.html``  : raw model HTML.
  * ``<unit>_chandra_blocks.json``  : parsed Chandra blocks.
  * ``<unit>_schema_layout.json``   : per-schema-field bboxes (pt + px).
  * ``<unit>_schema_layout.jpg``   : overlay visualisation.
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

from doc_utils import IMAGE_EXTENSIONS, PDF_EXTENSIONS, list_inputs

DEFAULT_MODEL = "datalab-to/chandra-ocr-2"
DEFAULT_SAMPLE_DIR = PROJECT_DIR / "data/samples/GIAY_GUI_TIEN_TIET_KIEM"

# Official Chandra OCR layout prompt (chandra/prompts.py: OCR_LAYOUT_PROMPT).
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
    f"Only use these tags {_ALLOWED_TAGS}, and these attributes {_ALLOWED_ATTRS}.\n\n"
    "Guidelines: keep formatting; mark tables with colspan/rowspan; preserve "
    "reading order; describe images via the alt attribute; do not invent text."
)


def compose_ocr_layout_prompt(
    *,
    extra_file: Path | None,
    giay_gui_tien_guide: bool,
) -> str:
    """Full user text prompt: base OCR layout + optional file + optional GIẤY GỬI hints."""
    parts: list[str] = [OCR_LAYOUT_PROMPT]
    if extra_file is not None:
        if not extra_file.is_file():
            raise FileNotFoundError(f"extra layout prompt file not found: {extra_file}")
        parts.append(extra_file.read_text(encoding="utf-8").strip())
    if giay_gui_tien_guide:
        from domain_prompt_giay_gui_tien import LAYOUT_FIELD_GUIDE_VI

        parts.append(LAYOUT_FIELD_GUIDE_VI.strip())
    return "\n\n".join(p for p in parts if p)


def schema_template_tuple(args: argparse.Namespace) -> tuple[float | None, float | None]:
    w, h = args.schema_template_pt
    if w <= 0.0 or h <= 0.0:
        return None, None
    return w, h


DIV_RE = re.compile(
    r'<div\b(?P<attrs>[^>]*)>(?P<inner>.*?)</div>',
    re.IGNORECASE | re.DOTALL,
)
BBOX_RE = re.compile(r'data-bbox\s*=\s*"([^"]+)"', re.IGNORECASE)
LABEL_RE = re.compile(r'data-label\s*=\s*"([^"]+)"', re.IGNORECASE)
BR_RE = re.compile(r'<br\s*/?>', re.IGNORECASE)
TAG_RE = re.compile(r'<[^>]+>')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chandra OCR 2 + schema layout overlay.")
    default_in = PROJECT_DIR / "data/test/GIAY_GUI_TIEN_TIET_KIEM"
    default_out = PROJECT_DIR / "results/chandra_schema_layout"
    p.add_argument("--input-dir", type=Path, default=default_in)
    p.add_argument("--input-file", type=Path, default=None)
    p.add_argument("--only", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=default_out)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument(
        "--lora-path",
        type=Path,
        default=None,
        help="Optional PEFT adapter dir (e.g. outputs/.../lora_adapter from train_chandra_layout_lora.py).",
    )
    p.add_argument(
        "--processor-from-lora",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="With --lora-path: load AutoProcessor from the adapter if tokenizer_config.json "
        "exists there. Default is False: always load the processor from --model so the "
        "chat template and image preprocessing match non-finetuned inference (same as "
        "train_chandra_layout_lora, which uses the base processor only).",
    )
    p.add_argument("--device-map", type=str, default="cuda:1")
    p.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=("auto", "bfloat16", "float16", "float32"),
    )
    p.add_argument("--pdf-dpi", type=int, default=200)
    p.add_argument("--max-pixels", type=int, default=1_600_000)
    p.add_argument(
        "--layout-json", type=Path,
        default=DEFAULT_SAMPLE_DIR / "layout _GIAY_GUI_TIEN_TIET_KIEM.json",
    )
    p.add_argument("--page-width-pt", type=float, default=596.0)
    p.add_argument("--page-height-pt", type=float, default=844.0)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--font-scale", type=float, default=1.0,
        help="Label font size scale.",
    )
    p.add_argument(
        "--schema-template-pt",
        type=float,
        nargs=2,
        metavar=("W", "H"),
        default=[596.0, 844.0],
        help="Layout JSON boxes are authored in this page size (PDF pt); scale linearly "
        "to each file's page rect for overlay. Default 596 844 (sample template). "
        "Use 0 0 if JSON coordinates are already in each document's page pt (legacy).",
    )
    p.add_argument(
        "--extra-layout-prompt-file",
        type=Path,
        default=None,
        help="UTF-8 text appended after the base OCR layout prompt.",
    )
    p.add_argument(
        "--giay-gui-tien-layout-guide",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append Vietnamese zone/field hints for GIẤY GỬI TIỀN TIẾT KIỆM "
        "(domain_prompt_giay_gui_tien.py).",
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
    from PIL import Image

    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = (max_pixels / (w * h)) ** 0.5
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def load_units(path: Path, pdf_dpi: int) -> list[tuple[str, Any, tuple[float, float]]]:
    """(unit_name, PIL.Image, (page_w_pt, page_h_pt)) per page."""
    from PIL import Image

    if path.suffix.lower() in PDF_EXTENSIONS:
        import fitz

        zoom = pdf_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        out: list[tuple[str, Any, tuple[float, float]]] = []
        with fitz.open(path) as doc:
            for i, page in enumerate(doc):
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                mode = "RGB" if pix.n < 4 else "RGBA"
                img = Image.frombytes(mode, (pix.width, pix.height), pix.samples).convert("RGB")
                out.append((f"{path.stem}_page{i + 1:02d}", img,
                            (page.rect.width, page.rect.height)))
        return out
    return [(path.stem, Image.open(path).convert("RGB"), (596.0, 844.0))]


def build_messages(test_image: Any, layout_prompt: str | None = None) -> list[dict[str, Any]]:
    text = OCR_LAYOUT_PROMPT if layout_prompt is None else layout_prompt
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": test_image},
            {"type": "text", "text": text},
        ],
    }]


def parse_chandra_blocks(html: str) -> list[dict[str, Any]]:
    """Return [{label, bbox_norm:[x0,y0,x1,y1] in 0-1000, inner_html, inner_lines}]."""
    seen: set[tuple[str, str]] = set()
    blocks: list[dict[str, Any]] = []
    for m in DIV_RE.finditer(html):
        attrs = m.group("attrs") or ""
        inner = m.group("inner") or ""
        b = BBOX_RE.search(attrs)
        l = LABEL_RE.search(attrs)
        if not b or not l:
            continue
        bbox_str = b.group(1).strip()
        label = l.group(1).strip()
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

        lines_html = BR_RE.split(inner)
        lines = []
        for lh in lines_html:
            text = TAG_RE.sub("", lh).strip()
            text = re.sub(r"\s+", " ", text)
            if text:
                lines.append(text)
        plain = " ".join(lines)
        blocks.append({
            "label": label,
            "bbox_norm": [x0, y0, x1, y1],
            "inner_html": inner.strip(),
            "inner_lines": lines,
            "inner_text": plain,
        })
    return blocks


def load_schema(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for s in data.get("sections", []):
        layout = s.get("layout") or {}
        if not layout:
            continue
        out.append({
            "name": s.get("name", ""),
            "ord": s.get("ord", -1),
            "x_pt": float(layout["x"]),
            "y_pt": float(layout["y"]),
            "w_pt": float(layout["width"]),
            "h_pt": float(layout["height"]),
        })
    return out


def schema_section_box_pt(
    sec: dict[str, Any],
    page_w_pt: float,
    page_h_pt: float,
    template_w_pt: float | None = None,
    template_h_pt: float | None = None,
) -> list[float]:
    """Schema rect in **document** PDF pt (after optional linear map from template), clamped."""
    x0 = float(sec["x_pt"])
    y0 = float(sec["y_pt"])
    w = float(sec["w_pt"])
    h = float(sec["h_pt"])
    if (
        template_w_pt is not None
        and template_h_pt is not None
        and template_w_pt > 0.0
        and template_h_pt > 0.0
    ):
        sx = page_w_pt / template_w_pt
        sy = page_h_pt / template_h_pt
        x0 *= sx
        y0 *= sy
        w *= sx
        h *= sy
    x1 = x0 + w
    y1 = y0 + h
    x0 = max(0.0, x0)
    y0 = max(0.0, y0)
    x1 = min(page_w_pt, x1)
    y1 = min(page_h_pt, y1)
    if x1 <= x0:
        x1 = x0 + 1.0
    if y1 <= y0:
        y1 = y0 + 1.0
    return [x0, y0, x1, y1]


# ---------------------------------------------------------------------------
# Smart label rendering
# ---------------------------------------------------------------------------

DEFAULT_PALETTE = [
    (220, 20, 60),   (30, 144, 255),  (50, 205, 50),   (255, 165, 0),
    (148, 0, 211),   (0, 191, 255),   (255, 105, 180), (154, 205, 50),
    (255, 215, 0),   (0, 206, 209),   (199, 21, 133),  (32, 178, 170),
    (255, 99, 71),   (60, 179, 113),  (123, 104, 238), (218, 165, 32),
]


def _load_font(image_size: tuple[int, int], font_scale: float):
    from PIL import ImageFont

    w, h = image_size
    size = max(11, int(11 + font_scale * min(w, h) / 180))
    for path in (
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _rect_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def draw_schema_layout(
    image: Any,
    entries: list[dict[str, Any]],
    out_path: Path,
    font_scale: float = 1.0,
) -> None:
    """Draw boxes + labels chosen so that:
       - labels never extend past the image right/top/bottom edge,
       - labels do not overlap previously drawn labels (try alternate slots),
       - tall boxes (signature) get an inside top-left label.
    """
    from PIL import ImageDraw

    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font = _load_font(img.size, font_scale)
    line_w = max(2, min(w, h) // 500)

    color_map: dict[str, tuple[int, int, int]] = {}

    def color_for(label: str) -> tuple[int, int, int]:
        if label not in color_map:
            color_map[label] = DEFAULT_PALETTE[len(color_map) % len(DEFAULT_PALETTE)]
        return color_map[label]

    sorted_entries = sorted(
        entries,
        key=lambda e: (e["box_xyxy"][1], e["box_xyxy"][0]),
    )

    placed_labels: list[tuple[float, float, float, float]] = []

    for e in sorted_entries:
        x0, y0, x1, y1 = e["box_xyxy"]
        box_h = y1 - y0
        label = str(e.get("label", ""))
        color = color_for(label)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=line_w)

    for e in sorted_entries:
        x0, y0, x1, y1 = e["box_xyxy"]
        label = str(e.get("label", ""))
        color = color_for(label)

        cap = label
        bbox = draw.textbbox((0, 0), cap, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 3
        lab_w = tw + 2 * pad
        lab_h = th + 2 * pad

        candidates: list[tuple[float, float]] = []
        candidates.append((x0, y0 - lab_h))
        candidates.append((max(0, x1 - lab_w), y0 - lab_h))
        candidates.append((max(0, min(x0, w - lab_w)), y1 + 1))
        candidates.append((max(0, min(x1 - lab_w, w - lab_w)), y1 + 1))
        candidates.append((x1 + 2, y0))
        candidates.append((max(0, x0 - lab_w - 2), y0))
        if (y1 - y0) > lab_h * 1.4:
            candidates.append((x0 + 2, y0 + 2))
        candidates.append((max(0, min(x0, w - lab_w)), max(0, y0 - lab_h)))

        chosen: tuple[float, float, float, float] | None = None
        for lx0, ly0 in candidates:
            lx0 = max(0.0, min(lx0, w - lab_w))
            ly0 = max(0.0, min(ly0, h - lab_h))
            lx1 = lx0 + lab_w
            ly1 = ly0 + lab_h
            if any(_rect_overlap((lx0, ly0, lx1, ly1), r) for r in placed_labels):
                continue
            chosen = (lx0, ly0, lx1, ly1)
            break

        if chosen is None:
            lx0 = max(0.0, min(x0, w - lab_w))
            ly0 = max(0.0, min(y0 - lab_h, h - lab_h))
            shift = 0.0
            for _ in range(40):
                cand = (lx0, ly0 + shift, lx0 + lab_w, ly0 + shift + lab_h)
                if cand[3] > h:
                    break
                if not any(_rect_overlap(cand, r) for r in placed_labels):
                    chosen = cand
                    break
                shift += lab_h + 1
            if chosen is None:
                chosen = (lx0, max(0.0, min(y0, h - lab_h)),
                          lx0 + lab_w, max(lab_h, min(y0 + lab_h, h)))

        lx0, ly0, lx1, ly1 = chosen
        draw.rectangle([lx0, ly0, lx1, ly1], fill=color)
        text_y = ly0 + pad - bbox[1]
        draw.text((lx0 + pad, text_y), cap, fill=(255, 255, 255), font=font)
        placed_labels.append(chosen)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
            f"No inputs found (images: {sorted(IMAGE_EXTENSIONS)}, "
            f"pdfs: {sorted(PDF_EXTENSIONS)}).",
            file=sys.stderr,
        )
        return 1

    if not args.layout_json.is_file():
        print(f"layout JSON not found: {args.layout_json}", file=sys.stderr)
        return 1
    schema_sections = load_schema(args.layout_json)
    print(f"[chandra-schema] {len(schema_sections)} schema sections from "
          f"{args.layout_json.name}")

    tw, th = schema_template_tuple(args)
    try:
        layout_prompt_text = compose_ocr_layout_prompt(
            extra_file=args.extra_layout_prompt_file,
            giay_gui_tien_guide=args.giay_gui_tien_layout_guide,
        )
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    if tw is None:
        print(
            "[chandra-schema] Schema overlay: template scale OFF (use 0 0 or treat JSON pt as page pt).",
            file=sys.stderr,
        )
    else:
        print(
            f"[chandra-schema] Schema overlay: template {tw:g}×{th:g} pt → scale to each page rect.",
            file=sys.stderr,
        )

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = resolve_dtype(args.dtype)
    print(f"[chandra-schema] Loading {args.model} dtype={dtype} device_map={args.device_map}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=dtype, device_map=args.device_map,
    )
    processor_src = args.model
    if args.lora_path is not None:
        if not args.lora_path.is_dir():
            print(f"lora path not found: {args.lora_path}", file=sys.stderr)
            return 1
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.lora_path))
        print(f"[chandra-schema] Loaded LoRA from {args.lora_path}")
        if args.processor_from_lora and (args.lora_path / "tokenizer_config.json").is_file():
            processor_src = str(args.lora_path)
            print(
                f"[chandra-schema] Processor from LoRA dir (tokenizer_config.json present): "
                f"{processor_src}",
                file=sys.stderr,
            )
        else:
            print(
                f"[chandra-schema] Processor from base model (same as no LoRA): {processor_src}",
                file=sys.stderr,
            )
    processor = AutoProcessor.from_pretrained(processor_src)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "model": args.model,
        "lora_path": str(args.lora_path) if args.lora_path else None,
        "processor_source": processor_src,
        "processor_from_lora": processor_src != args.model,
        "device_map": args.device_map,
        "dtype": str(dtype),
        "layout_json": str(args.layout_json),
        "page_size_pt": [args.page_width_pt, args.page_height_pt],
        "schema_template_pt": list(args.schema_template_pt),
        "schema_template_scale_active": tw is not None,
        "giay_gui_tien_layout_guide": args.giay_gui_tien_layout_guide,
        "extra_layout_prompt_file": str(args.extra_layout_prompt_file)
        if args.extra_layout_prompt_file is not None
        else None,
        "layout_overlay": "schema_only",
        "items": [],
    }

    for src_path in inputs_list:
        for unit_name, image, page_pt in load_units(src_path, args.pdf_dpi):
            page_w_pt, page_h_pt = page_pt
            test_image = fit_to_max_pixels(image, args.max_pixels)
            img_w_px, img_h_px = test_image.size
            input_path = args.output_dir / f"{unit_name}_input.jpg"
            test_image.save(input_path, quality=92)

            messages = build_messages(test_image, layout_prompt_text)
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
            inputs = {k: (v.to(model.device) if hasattr(v, "to") else v)
                      for k, v in inputs.items()}

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
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )[0].strip()

            html_path = args.output_dir / f"{unit_name}_chandra_layout.html"
            html_path.write_text(text, encoding="utf-8")

            blocks = parse_chandra_blocks(text)

            blocks_path = args.output_dir / f"{unit_name}_chandra_blocks.json"
            blocks_path.write_text(
                json.dumps(blocks, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            entries: list[dict[str, Any]] = []
            schema_json: list[dict[str, Any]] = []
            sx = img_w_px / page_w_pt
            sy = img_h_px / page_h_pt
            for sec in sorted(schema_sections, key=lambda s: s["ord"]):
                box_pt = schema_section_box_pt(sec, page_w_pt, page_h_pt, tw, th)
                box_px = [box_pt[0] * sx, box_pt[1] * sy,
                          box_pt[2] * sx, box_pt[3] * sy]
                entries.append({"label": sec["name"], "box_xyxy": box_px})
                schema_json.append({
                    "name": sec["name"],
                    "ord": sec["ord"],
                    "box_pt": [round(v, 2) for v in box_pt],
                    "box_xyxy_px": [round(v, 2) for v in box_px],
                    "source": "schema",
                })

            json_path = args.output_dir / f"{unit_name}_schema_layout.json"
            json_path.write_text(
                json.dumps({
                    "source": str(src_path),
                    "unit": unit_name,
                    "image_size": [img_w_px, img_h_px],
                    "input_image": str(input_path),
                    "raw_html_path": str(html_path),
                    "layout_overlay": "schema_only",
                    "page_pt": [page_w_pt, page_h_pt],
                    "schema_template_pt": list(args.schema_template_pt),
                    "schema_template_scale_active": tw is not None,
                    "giay_gui_tien_layout_guide": args.giay_gui_tien_layout_guide,
                    "n_chandra_blocks": len(blocks),
                    "sections": schema_json,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            viz_path = args.output_dir / f"{unit_name}_schema_layout.jpg"
            draw_schema_layout(test_image, entries, viz_path,
                               font_scale=args.font_scale)

            summary["items"].append({
                "source": str(src_path),
                "unit": unit_name,
                "image_size": [img_w_px, img_h_px],
                "viz": str(viz_path),
                "json": str(json_path),
                "raw_html": str(html_path),
                "blocks_json": str(blocks_path),
                "n_sections": len(entries),
                "n_chandra_blocks": len(blocks),
            })
            print(f"[ok] {src_path.name} :: {unit_name} -> {len(entries)} schema boxes "
                  f"-> {viz_path.name}")

    summary_path = args.output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
