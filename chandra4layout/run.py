"""Chandra OCR 2 → Schema JSON – minimal single-pass pipeline.

Cách chạy:
    python chandra4layout/run.py --input-file test.pdf --device-map cuda:0

Pipeline:
    1. Ảnh / PDF  →  Chandra OCR 2  →  HTML output
    2. Parse từng <div> block: lấy bbox + inner_text
    3. Gán thẳng vào schema:
         • "FieldName: Value"  → đọc tên field trước dấu ':'
         • Section-Header      → so tên text với schema
         • Image block         → Logo
         • Table block         → Bảng kê tiền mặt / Bảng kê ghi số (theo kích thước)
    4. Xuất schema JSON + visualization JPG

Không có IoU, không có multi-pass matching, không có fallback phức tạp.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_DEFAULT_LAYOUT_JSON = (
    _HERE / "data/samples/GIAY_GUI_TIEN_TIET_KIEM/layout _GIAY_GUI_TIEN_TIET_KIEM.json"
)
_DEFAULT_PROMPT_FILE = _HERE / "prompts/giay_gui_tien_tiet_kiem.txt"
_DEFAULT_INPUT_DIR   = _HERE / "data/test/GIAY_GUI_TIEN_TIET_KIEM"
_DEFAULT_OUTPUT_DIR  = _HERE / "results/giay_gui_tien_tiet_kiem_direct"
_DEFAULT_MODEL       = "datalab-to/chandra-ocr-2"

# ── File extension sets ──────────────────────────────────────────────────────
_PDF_EXT   = {".pdf"}
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

# ── HTML parsing regexes ─────────────────────────────────────────────────────
_DIV_RE   = re.compile(r"<div\b(?P<attrs>[^>]*)>(?P<inner>.*?)</div>",
                       re.IGNORECASE | re.DOTALL)
_BBOX_RE  = re.compile(r'data(?:-bbox)?\s*=\s*"([^"]+)"', re.IGNORECASE)
_LABEL_RE = re.compile(r'data-label\s*=\s*"([^"]+)"', re.IGNORECASE)
_BR_RE    = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_RE   = re.compile(r"<[^>]+>")

# ── Visualization palette ────────────────────────────────────────────────────
_PALETTE = [
    (220, 20, 60),  (30, 144, 255), (50, 205, 50),  (255, 165, 0),
    (148, 0, 211),  (0, 191, 255),  (255, 105, 180),(154, 205, 50),
    (255, 215, 0),  (0, 206, 209),  (199, 21, 133), (32, 178, 170),
    (255, 99, 71),  (60, 179, 113), (123, 104, 238),(218, 165, 32),
]


# ── Text normalization ───────────────────────────────────────────────────────

def _fold(s: str) -> str:
    """Lowercase + strip Vietnamese diacritics → bare ASCII letters+digits."""
    _VN = str.maketrans("đĐơƠưƯ", "dDoOuU")
    s = s.translate(_VN)
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if "a" <= c <= "z" or c.isdigit())


# ── HTML parser ──────────────────────────────────────────────────────────────

def _parse_blocks(html: str) -> list[dict[str, Any]]:
    """Parse <div data[-bbox]="x y x y" data-label="..."> → list of blocks."""
    seen: set[tuple] = set()
    blocks = []
    for m in _DIV_RE.finditer(html):
        attrs = m.group("attrs") or ""
        inner = m.group("inner") or ""
        bm = _BBOX_RE.search(attrs)
        lm = _LABEL_RE.search(attrs)
        if not bm or not lm:
            continue
        parts = bm.group(1).replace(",", " ").split()
        if len(parts) != 4:
            continue
        try:
            x0, y0, x1, y1 = map(float, parts)
        except ValueError:
            continue
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        label = lm.group(1).strip()
        key = (round(x0), round(y0), round(x1), round(y1), label)
        if key in seen:
            continue
        seen.add(key)
        lines_raw = _BR_RE.split(inner)
        lines = [re.sub(r"\s+", " ", _TAG_RE.sub("", l)).strip() for l in lines_raw]
        lines = [l for l in lines if l]
        blocks.append({
            "label": label,
            "bbox": [x0, y0, x1, y1],
            "text": " ".join(lines),
            "lines": lines,
        })
    return blocks


# ── Direct schema extraction ─────────────────────────────────────────────────

def extract_schema(
    blocks: list[dict[str, Any]],
    schema_sections: list[dict[str, Any]],
) -> dict[str, list[float] | None]:
    """Read schema section bboxes directly from model HTML blocks.

    Strategy (in priority order, each block assigned at most once):
      1. Image block          → Logo
      2. "FieldName: Value"   → match FieldName against schema (exact fold)
      3. Section-Header text  → match against schema header names
      4. Table blocks         → Bảng kê ghi số (large) / Bảng kê tiền mặt (small)
    """
    # Build fold → schema name map
    fold2name: dict[str, str] = {_fold(s["name"]): s["name"] for s in schema_sections}
    result: dict[str, list[float] | None] = {s["name"]: None for s in schema_sections}
    used_blocks: set[int] = set()

    # ── Pass 1: Image block → Logo ───────────────────────────────────────────
    for i, b in enumerate(blocks):
        if b["label"].lower() == "image":
            result["Logo"] = b["bbox"]
            used_blocks.add(i)
            break

    # ── Pass 2: "FieldName: Value" text extraction ───────────────────────────
    for i, b in enumerate(blocks):
        if i in used_blocks:
            continue
        for line in b["lines"]:
            colon = line.find(":")
            if colon <= 0:
                continue
            candidate = line[:colon].strip()
            cf = _fold(candidate)
            if cf in fold2name:
                name = fold2name[cf]
                if result[name] is None:   # first match wins
                    result[name] = b["bbox"]
                    used_blocks.add(i)
                    break

    # ── Pass 3: Section-Header text match ────────────────────────────────────
    for i, b in enumerate(blocks):
        if i in used_blocks:
            continue
        if b["label"].lower() != "section-header":
            continue
        bf = _fold(b["text"])
        best_name, best_ratio = None, 0.0
        for sf, sname in fold2name.items():
            if result[sname] is not None:
                continue
            if sf == bf:
                best_name, best_ratio = sname, 1.0
                break
            if sf and bf and (sf in bf or bf in sf):
                ratio = min(len(sf), len(bf)) / max(len(sf), len(bf))
                if ratio > best_ratio:
                    best_ratio, best_name = ratio, sname
        if best_name and best_ratio >= 0.7:
            result[best_name] = b["bbox"]
            used_blocks.add(i)

    # ── Pass 3b: schema name as first-line prefix (catches "Người gửi tiền",
    #            "Giao dịch viên", "Ngày (Date): ...", etc.) ────────────────
    for i, b in enumerate(blocks):
        if i in used_blocks:
            continue
        for line in b["lines"][:2]:          # only first two lines
            lf = _fold(line)
            if not lf:
                continue
            for sf, sname in fold2name.items():
                if result[sname] is not None or not sf:
                    continue
                # schema fold must start the line fold (e.g. "ngay" starts "ngaydate")
                if lf.startswith(sf) or lf == sf:
                    ratio = len(sf) / max(len(lf), 1)
                    if ratio >= 0.55:        # allow short schema names in longer lines
                        result[sname] = b["bbox"]
                        used_blocks.add(i)
                        break
            if i in used_blocks:
                break

    # ── Pass 3c: substring search for longer schema names (≥10 chars folded)
    #    Catches e.g. "Xác nhận thông tin" whose fold "xacnhanthongtin" can be
    #    split and found in a block like "Tôi xác nhận ... thông tin ...". ───
    for i, b in enumerate(blocks):
        if i in used_blocks:
            continue
        bf = _fold(b["text"])
        for sf, sname in fold2name.items():
            if result[sname] is not None or len(sf) < 10:
                continue
            # Split schema fold into two halves and check both appear in block
            mid = len(sf) // 2
            part1, part2 = sf[:mid], sf[mid:]
            if part1 in bf and part2 in bf:
                result[sname] = b["bbox"]
                used_blocks.add(i)
                break

    # ── Pass 3d: schema name embedded anywhere in a line (for "Ngày" in
    #    "Liên 1/2 dành cho Ngân hàng Ngày (Date): 05-01-2026"). Only applied
    #    to short schema names (≤6 folded chars) that are sufficiently unique
    #    (not a prefix of any already-matched section name). ─────────────────
    matched_folds = {_fold(sn) for sn, bx in result.items() if bx is not None}
    for i, b in enumerate(blocks):
        if i in used_blocks:
            continue
        for line in b["lines"]:
            lf = _fold(line)
            for sf, sname in fold2name.items():
                if result[sname] is not None or not sf or len(sf) > 6:
                    continue
                # Only match if no already-matched section starts with this fold
                shadowed = any(mf.startswith(sf) and mf != sf for mf in matched_folds)
                if shadowed:
                    continue
                # The schema fold must appear as a standalone segment in the line
                # (preceded by a non-alpha char or start, followed by non-alpha)
                pattern = rf'(?<![a-z]){re.escape(sf)}(?![a-z])'
                if re.search(pattern, lf):
                    result[sname] = b["bbox"]
                    used_blocks.add(i)
                    matched_folds.add(sf)
                    break
            if i in used_blocks:
                break

    # ── Pass 4: Table blocks → Bảng kê ghi số / Bảng kê tiền mặt ───────────
    table_blocks = [
        (i, b) for i, b in enumerate(blocks)
        if i not in used_blocks and b["label"].lower() == "table"
    ]
    table_blocks.sort(key=lambda ib: (ib[1]["bbox"][2] - ib[1]["bbox"][0])
                                     * (ib[1]["bbox"][3] - ib[1]["bbox"][1]),
                      reverse=True)
    large_table = "Bảng kê ghi số"
    small_table = "Bảng kê tiền mặt"
    if table_blocks:
        i, b = table_blocks[0]
        area = (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1])
        if area > 200_000:          # large table (normalized 0-1000 coords)
            if result[large_table] is None:
                result[large_table] = b["bbox"]
                used_blocks.add(i)
        if result[small_table] is None and len(table_blocks) > 1:
            i2, b2 = table_blocks[1]
            result[small_table] = b2["bbox"]
            used_blocks.add(i2)
        elif result[small_table] is None:
            result[small_table] = b["bbox"]
            used_blocks.add(i)
    if len(table_blocks) > 1 and result[large_table] is None:
        i, b = table_blocks[0]
        result[large_table] = b["bbox"]
        used_blocks.add(i)

    return result


# ── Schema JSON builder ──────────────────────────────────────────────────────

def _norm_to_pt(bbox: list[float], w_pt: float, h_pt: float) -> dict[str, float]:
    """Convert normalized 0-1000 bbox to pt coordinates (x, y, width, height)."""
    x0, y0, x1, y1 = bbox
    x   = x0 / 1000 * w_pt
    y   = y0 / 1000 * h_pt
    w   = (x1 - x0) / 1000 * w_pt
    h   = (y1 - y0) / 1000 * h_pt
    return {"x": round(x, 2), "y": round(y, 2),
            "width": round(w, 2), "height": round(h, 2)}


def build_schema_json(
    schema_sections: list[dict[str, Any]],
    bbox_map: dict[str, list[float] | None],
    page_w_pt: float,
    page_h_pt: float,
    page_num: int = 1,
) -> list[dict[str, Any]]:
    import uuid as _uuid
    rows = []
    for sec in sorted(schema_sections, key=lambda s: s["ord"]):
        name = sec["name"]
        bbox = bbox_map.get(name)
        if bbox:
            layout = _norm_to_pt(bbox, page_w_pt, page_h_pt)
            layout["page"] = page_num
            source = "chandra"
        else:
            # Fall back to template bbox
            tpl = sec.get("layout") or {}
            layout = dict(tpl, page=page_num) if tpl else None
            source = "template"
        rows.append({
            "id": sec.get("id") or str(_uuid.uuid4()),
            "name": name,
            "layout": layout,
            "_bbox_source": source,
            "_bbox_norm": bbox,
        })
    return rows


# ── Visualization ────────────────────────────────────────────────────────────

def _draw(image: Any, sections: list[dict], out_path: Path) -> None:
    from PIL import ImageDraw, ImageFont
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font = font_sm = ImageFont.load_default()

    w, h = img.size
    for idx, sec in enumerate(sections):
        layout = sec.get("layout")
        if not layout:
            continue
        color = _PALETTE[idx % len(_PALETTE)]
        source = sec.get("_bbox_source", "template")
        alpha = 180 if source == "chandra" else 60
        x0 = layout["x"] / layout.get("_pw", 595) * w  if "_pw" in layout else layout["x"] * w / 595
        # Use normalized bbox directly if available
        bn = sec.get("_bbox_norm")
        if bn:
            bx0 = bn[0] / 1000 * w
            by0 = bn[1] / 1000 * h
            bx1 = bn[2] / 1000 * w
            by1 = bn[3] / 1000 * h
        else:
            continue
        draw.rectangle([bx0, by0, bx1, by1], outline=color + (alpha,), width=2)
        draw.rectangle([bx0, by0, bx0 + 1, by1], fill=color + (30,))
        label = sec["name"][:28]
        draw.text((bx0 + 3, by0 + 2), label, fill=color, font=font_sm)

    img.save(out_path, quality=92)


# ── PDF / image loader ───────────────────────────────────────────────────────

def _load_units(src: Path, dpi: int):
    """Yield (unit_name, PIL.Image, (page_w_pt, page_h_pt)) per page."""
    from PIL import Image as PILImage
    if src.suffix.lower() in _PDF_EXT:
        import fitz
        doc = fitz.open(str(src))
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
            rect = page.rect
            yield f"{src.stem}_page{page_idx + 1:02d}", img, (rect.width, rect.height)
        doc.close()
    else:
        img = PILImage.open(src).convert("RGB")
        yield src.stem, img, (img.width * 72 / dpi, img.height * 72 / dpi)


def _fit(img: Any, max_pixels: int) -> Any:
    w, h = img.size
    if w * h <= max_pixels:
        return img
    scale = (max_pixels / (w * h)) ** 0.5
    return img.resize((int(w * scale), int(h * scale)))


def _resolve_dtype(s: str):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}.get(s, "auto")


def _list_inputs(d: Path, only: str | None) -> list[Path]:
    exts = _PDF_EXT | _IMAGE_EXT
    files = [f for f in sorted(d.iterdir()) if f.suffix.lower() in exts]
    if only:
        files = [f for f in files if only in f.name]
    return files


def _sanitize(raw: str) -> str:
    t = raw.strip()
    for marker in ("</html>", "<|endoftext|>", "<|im_end|>"):
        if marker in t:
            t = t[: t.index(marker)]
    return t.strip()


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chandra OCR 2 → Schema JSON (direct, single-pass)."
    )
    p.add_argument("--input-dir",  type=Path, default=_DEFAULT_INPUT_DIR)
    p.add_argument("--input-file", type=Path, default=None)
    p.add_argument("--only",       type=str,  default=None)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    p.add_argument("--layout-json",type=Path, default=_DEFAULT_LAYOUT_JSON)
    p.add_argument("--prompt-file",type=Path, default=_DEFAULT_PROMPT_FILE)
    p.add_argument("--model",      type=str,  default=_DEFAULT_MODEL)
    p.add_argument("--lora-path",  type=Path, default=None)
    p.add_argument("--device-map", type=str,  default="cuda:0")
    p.add_argument("--dtype",      type=str,  default="bfloat16",
                   choices=("auto", "bfloat16", "float16", "float32"))
    p.add_argument("--pdf-dpi",    type=int,  default=200)
    p.add_argument("--max-pixels", type=int,  default=1_600_000)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--temperature",type=float,default=0.0)
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def run() -> int:
    args = _parse_args()

    inputs_list = (
        [args.input_file] if args.input_file
        else _list_inputs(args.input_dir, args.only)
    )
    if not inputs_list or (args.input_file and not args.input_file.is_file()):
        print(f"No input found.", file=sys.stderr); return 1

    _raw = json.loads(Path(args.layout_json).read_text(encoding="utf-8"))
    schema_sections = _raw["sections"] if isinstance(_raw, dict) else _raw
    for i, s in enumerate(schema_sections):
        s.setdefault("ord", i)
    print(f"[run] {len(schema_sections)} schema sections")

    prompt_text = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    print(f"[run] Prompt: {args.prompt_file.name} ({len(prompt_text)} chars)")

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = _resolve_dtype(args.dtype)
    print(f"[run] Loading {args.model}  dtype={dtype}  device={args.device_map}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=dtype, device_map=args.device_map,
    )
    if args.lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.lora_path))
        print(f"[run] LoRA loaded from {args.lora_path}")
    processor = AutoProcessor.from_pretrained(
        str(args.lora_path) if args.lora_path
        and (args.lora_path / "tokenizer_config.json").is_file()
        else args.model
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"model": args.model, "items": []}

    for src_path in inputs_list:
        for unit_name, image, page_pt in _load_units(src_path, args.pdf_dpi):
            page_w_pt, page_h_pt = page_pt
            img = _fit(image, args.max_pixels)

            # ── Inference ────────────────────────────────────────────────
            print(f"[{unit_name}] Inference …")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": prompt_text},
            ]}]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
            inputs = {k: (v.to(model.device) if hasattr(v, "to") else v)
                      for k, v in inputs.items()}
            do_sample = args.temperature > 0
            with torch.inference_mode():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature if do_sample else 1.0,
                )
            raw = _sanitize(processor.batch_decode(
                gen[:, inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0])

            # ── Save raw HTML ────────────────────────────────────────────
            (args.output_dir / f"{unit_name}_raw.html").write_text(raw, encoding="utf-8")

            # ── Parse + extract schema ───────────────────────────────────
            blocks = _parse_blocks(raw)
            print(f"[{unit_name}] {len(blocks)} blocks parsed")

            bbox_map = extract_schema(blocks, schema_sections)
            matched = sum(1 for v in bbox_map.values() if v is not None)
            missed  = [k for k, v in bbox_map.items() if v is None]
            print(f"[{unit_name}] Matched {matched}/{len(schema_sections)} sections")
            if missed:
                print(f"[{unit_name}] Missing: {', '.join(missed)}")

            # ── Build + save schema JSON ─────────────────────────────────
            sections_out = build_schema_json(
                schema_sections, bbox_map, page_w_pt, page_h_pt
            )
            schema_out = {
                "source": str(src_path),
                "unit": unit_name,
                "image_size": list(img.size),
                "page_size_pt": [page_w_pt, page_h_pt],
                "n_matched": matched,
                "n_total": len(schema_sections),
                "sections": sections_out,
            }
            json_path = args.output_dir / f"{unit_name}_schema.json"
            json_path.write_text(
                json.dumps(schema_out, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            # ── Visualize ────────────────────────────────────────────────
            viz_path = args.output_dir / f"{unit_name}_schema.jpg"
            _draw(img, sections_out, viz_path)

            print(f"[ok] {src_path.name} → {json_path.name}  viz → {viz_path.name}")
            summary["items"].append({
                "unit": unit_name,
                "n_matched": matched,
                "n_total": len(schema_sections),
                "schema_json": str(json_path),
                "viz": str(viz_path),
            })

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
