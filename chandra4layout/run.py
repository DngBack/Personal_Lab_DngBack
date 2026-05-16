"""Chandra OCR 2 – inference + bbox visualization.

Cách chạy:
    python chandra4layout/run.py --input-file test.pdf --device-map cuda:0

Pipeline:
    1. Ảnh / PDF  →  Chandra OCR 2  →  HTML (<div data-bbox data-label>)
    2. Parse các block → bbox + nội dung
    3. (Optional) `--layout-json`: gắn nhãn *tên section schema* chỉ khi khớp
       tất định với HTML (KHÔNG fuzzy/IoU/table-area trick):
       - data-field hoặc data-schema trên div (model tự đặt sau nếu có)
       - dòng khớp kiểu "Tên khách hàng: ..." (phần trước ":" = tên trong JSON)
       - khối Section-Header có text trùng tên section (chuẩn hóa)
    4. JPG vẽ bbox Chandra (+ nhãn rule-based nếu có layout-json).
    5. (Optional) `--schema-llm-model`: LLM (Qwen) đọc `chandra_blocks` + danh sách
       schema từ layout JSON → một map tên-schema → bbox; vẽ `*_schema_llm.jpg`.
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
_DEFAULT_PROMPT_FILE = _HERE / "prompts/giay_gui_tien_tiet_kiem.txt"
_DEFAULT_LAYOUT_JSON = (
    _HERE / "data/samples/GIAY_GUI_TIEN_TIET_KIEM/layout _GIAY_GUI_TIEN_TIET_KIEM.json"
)
_DEFAULT_INPUT_DIR   = _HERE / "data/test/GIAY_GUI_TIEN_TIET_KIEM"
_DEFAULT_OUTPUT_DIR  = _HERE / "results/giay_gui_tien_tiet_kiem_direct"
_DEFAULT_MODEL       = "datalab-to/chandra-ocr-2"

_DEFAULT_SCHEMA_ALIGN_PROMPT = _HERE / "prompts/schema_align_llm_system.txt"
_DEFAULT_SCHEMA_ALIGN_FEWS = _HERE / "data/samples/schema_align_fewshots.json"

_PDF_EXT   = {".pdf"}
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

_DIV_RE   = re.compile(r"<div\b(?P<attrs>[^>]*)>(?P<inner>.*?)</div>",
                       re.IGNORECASE | re.DOTALL)
_BBOX_RE  = re.compile(r'data(?:-bbox)?\s*=\s*"([^"]+)"', re.IGNORECASE)
_LABEL_RE = re.compile(r'data-label\s*=\s*"([^"]+)"', re.IGNORECASE)
_SCHEMA_ATTR_RE = re.compile(
    r'data-(?:schema|schema-name|field)\s*=\s*"([^"]+)"',
    re.IGNORECASE,
)
_BR_RE    = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_RE   = re.compile(r"<[^>]+>")

_PALETTE = [
    (220, 20, 60),  (30, 144, 255), (50, 205, 50),  (255, 165, 0),
    (148, 0, 211),  (0, 191, 255),  (255, 105, 180),(154, 205, 50),
    (255, 215, 0),  (0, 206, 209),  (199, 21, 133), (32, 178, 170),
    (255, 99, 71),  (60, 179, 113), (123, 104, 238),(218, 165, 32),
]


def _fold(s: str) -> str:
    """Lowercase + strip Vietnamese diacritics → bare ASCII letters+digits."""
    _vn = str.maketrans("đĐơƠưƯ", "dDoOuU")
    s = s.translate(_vn)
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if "a" <= c <= "z" or c.isdigit())


def _collect_all_names(layout_root: dict | list) -> list[str]:
    """Mọi chuỗi `name` trong cây layout (section, group, cột, …)."""
    out: list[str] = []

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            n = o.get("name")
            if isinstance(n, str) and n.strip():
                out.append(n.strip())
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    walk(layout_root)
    # Giữ nguyên thứ tự DFS, chỉ uniq theo normalized key
    seen: set[str] = set()
    uniq = []
    for n in out:
        k = _fold(n)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(n)
    return uniq


def _fold_to_exact_schema(names: list[str]) -> dict[str, str]:
    """Một folded key → một tên canonical (ưu tiên lần gặp đầu DFS)."""
    m: dict[str, str] = {}
    for n in names:
        f = _fold(n)
        m.setdefault(f, n)
    return m


def _parse_blocks(html: str) -> list[dict[str, Any]]:
    """Parse <div data-bbox="x y x y" data-label="..."> → list of blocks."""
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
        sam = _SCHEMA_ATTR_RE.search(attrs)
        schema_attr = sam.group(1).strip() if sam else ""
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
            "schema_attr": schema_attr or None,
        })
    return blocks


def _schema_for_block(
    b: dict[str, Any],
    fold2name: dict[str, str],
) -> str | None:
    """Chỉ khớp tất định với tên trong layout JSON (không IoU, không đoán bảng)."""
    if b.get("schema_attr"):
        raw = b["schema_attr"].strip()
        fk = _fold(raw)
        if fk in fold2name:
            return fold2name[fk]

    for line in b.get("lines") or []:
        colon = line.find(":")
        if colon <= 0:
            continue
        key = line[:colon].strip()
        fk = _fold(key)
        if fk in fold2name:
            return fold2name[fk]

    if b.get("label", "").lower() == "section-header":
        for candidate in (b.get("text") or "", (b.get("lines") or [""])[0]):
            c = candidate.strip()
            if not c:
                continue
            fk = _fold(c)
            if fk in fold2name:
                return fold2name[fk]

    # Tiêu đề một dòng trong khối Text (vd. "Người gửi tiền", "Giao dịch viên")
    lines = b.get("lines") or []
    if (b.get("label") or "").lower() == "text" and lines and ":" not in lines[0]:
        fk = _fold(lines[0].strip())
        if fk in fold2name:
            return fold2name[fk]

    return None


def _caption_for_block(b: dict[str, Any]) -> str:
    tag = b["label"]
    snippet = (b["lines"][0] if b["lines"] else b["text"])[:40]
    schema = b.get("schema_match")
    if schema:
        return f"{schema}  [{tag}]"
    return f"{tag}: {snippet}"


def _draw_blocks(
    image: Any,
    blocks: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Vẽ bbox; `schema_match` trên từng block (nếu đã enrich) dùng làm nhãn."""
    from PIL import ImageDraw, ImageFont

    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font_sm = ImageFont.load_default()

    w, h = img.size
    for idx, b in enumerate(blocks):
        color = _PALETTE[idx % len(_PALETTE)]
        x0, y0, x1, y1 = b["bbox"]
        bx0, by0 = x0 / 1000 * w, y0 / 1000 * h
        bx1, by1 = x1 / 1000 * w, y1 / 1000 * h
        draw.rectangle([bx0, by0, bx1, by1], outline=color + (200,), width=2)
        draw.rectangle([bx0, by0, bx0 + 1, by1], fill=color + (35,))
        draw.text((bx0 + 3, by0 + 2), _caption_for_block(b), fill=color,
                  font=font_sm)

    img.save(out_path, quality=92)


def _draw_schema_bbox_map(
    image: Any,
    ordered_names: list[str],
    bbox_map: dict[str, list[float] | None],
    out_path: Path,
) -> None:
    """Vẽ bbox theo kết quả LLM/schema map (normalized 0–1000 → ảnh)."""
    from PIL import ImageDraw, ImageFont

    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font_sm = ImageFont.load_default()

    w, h = img.size
    for idx, name in enumerate(ordered_names):
        bbox = bbox_map.get(name)
        if not bbox:
            continue
        color = _PALETTE[idx % len(_PALETTE)]
        bx0 = bbox[0] / 1000 * w
        by0 = bbox[1] / 1000 * h
        bx1 = bbox[2] / 1000 * w
        by1 = bbox[3] / 1000 * h
        draw.rectangle([bx0, by0, bx1, by1], outline=color + (220,), width=2)
        label = name[:30]
        draw.text((bx0 + 3, by0 + 2), label, fill=color, font=font_sm)

    img.save(out_path, quality=92)


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
    lo = t.lower()
    sep = "\nassistant\n"
    if sep in lo:
        idx = lo.index(sep)
        t = t[:idx].strip()
    for marker in ("</html>", "<|endoftext|>", "<|im_end|>"):
        if marker in t:
            t = t[: t.index(marker)]
    return t.strip()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chandra OCR 2 → block viz; optionally Qwen aligns layout JSON ↔ bbox.",
    )
    p.add_argument("--input-dir",  type=Path, default=_DEFAULT_INPUT_DIR)
    p.add_argument("--input-file", type=Path, default=None)
    p.add_argument("--only",       type=str,  default=None)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--layout-json",
        type=Path,
        default=_DEFAULT_LAYOUT_JSON,
        help="JSON layout: dùng để gắn tên section lên viz (khớp ':' / Section-Header)."
             " File không tồn tại → chỉ hiện data-label Chandra.",
    )
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
    # LLM căn chỉnh schema ← block Chandra
    p.add_argument(
        "--schema-llm-model",
        type=str,
        default=None,
        metavar="HF_REPO",
        help="Causal LM (Qwen…) map schema+bbox (vd ~4 param: Qwen/Qwen3-4B; "
             "2.5 Instruct không có ~4B trên HF thì đổi Qwen/Qwen2.5-3B-Instruct).",
    )
    p.add_argument(
        "--schema-llm-device",
        type=str,
        default="cuda:1",
        help="Device cho LLM (nên cuda:1 nếu Chandra cuda:0).",
    )
    p.add_argument(
        "--schema-llm-dtype",
        type=str,
        default="float16",
        choices=("auto", "bfloat16", "float16", "float32"),
        help="dtype LLM căn schema.",
    )
    p.add_argument("--schema-llm-max-new-tokens", type=int, default=8192)
    p.add_argument("--schema-llm-temperature", type=float, default=0.0)
    p.add_argument("--schema-align-prompt-file", type=Path,
                   default=_DEFAULT_SCHEMA_ALIGN_PROMPT)
    p.add_argument("--schema-align-fewshots", type=Path,
                   default=_DEFAULT_SCHEMA_ALIGN_FEWS)
    return p.parse_args()


def run() -> int:
    args = _parse_args()

    inputs_list = (
        [args.input_file] if args.input_file
        else _list_inputs(args.input_dir, args.only)
    )
    if not inputs_list or (args.input_file and not args.input_file.is_file()):
        print("No input found.", file=sys.stderr)
        return 1

    prompt_text = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    print(f"[run] Prompt: {args.prompt_file.name} ({len(prompt_text)} chars)")

    fold2name: dict[str, str] | None = None
    schema_names_ordered: list[str] = []
    if args.layout_json.is_file():
        layout_raw = json.loads(args.layout_json.read_text(encoding="utf-8"))
        root = layout_raw["sections"] if isinstance(layout_raw, dict) else layout_raw
        schema_names_ordered = _collect_all_names(root)
        fold2name = _fold_to_exact_schema(schema_names_ordered)
        print(f"[run] Layout names for viz labels: {args.layout_json.name} "
              f"({len(fold2name)} folded keys, {len(schema_names_ordered)} field names)")
    else:
        print(f"[run] Layout JSON không tìm thấy ({args.layout_json}) — viz chỉ data-label.")

    schema_llm: Any | None = None

    if args.schema_llm_model:
        if not schema_names_ordered:
            print(
                "--schema-llm-model requires a valid layout JSON (--layout-json).",
                file=sys.stderr,
            )
            return 1

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

    system_schema_txt = ""
    if args.schema_llm_model:
        system_schema_txt = Path(args.schema_align_prompt_file).read_text(encoding="utf-8")

    if args.schema_llm_model:
        if args.device_map.strip() == args.schema_llm_device.strip():
            print(
                "[warn] Chandra và schema LLM đều \""
                + args.device_map.strip() + "\" — hết VRAM rất có thể;"
                  " chỉnh --schema-llm-device cuda:1 nếu có 2 GPU.",
                file=sys.stderr,
            )
        from schema_align_llm import CachedSchemaCausalLM

        smid = args.schema_llm_model.strip()
        print(f"[run] Schema LLM: {smid}  device={args.schema_llm_device}  "
              f"dtype={args.schema_llm_dtype}")
        schema_llm = CachedSchemaCausalLM(
            smid,
            args.schema_llm_device,
            args.schema_llm_dtype,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model,
        "schema_llm_model": args.schema_llm_model,
        "items": [],
    }

    try:
        for src_path in inputs_list:
            for unit_name, image, page_pt in _load_units(src_path, args.pdf_dpi):
                page_w_pt, page_h_pt = page_pt
                img = _fit(image, args.max_pixels)

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

                out_dir = args.output_dir
                (out_dir / f"{unit_name}_raw.html").write_text(raw, encoding="utf-8")

                blocks = _parse_blocks(raw)
                for b in blocks:
                    b["schema_match"] = (
                        _schema_for_block(b, fold2name) if fold2name else None
                    )
                n_schema_rule = sum(1 for b in blocks if b.get("schema_match"))
                print(f"[{unit_name}] {len(blocks)} blocks parsed, "
                      f"{n_schema_rule} heuristic schema labels")

                blocks_out = {
                    "source": str(src_path),
                    "unit": unit_name,
                    "image_size": list(img.size),
                    "page_size_pt": [page_w_pt, page_h_pt],
                    "n_blocks": len(blocks),
                    "blocks": blocks,
                    "schema_alignment_llm": None,
                }

                viz_path = out_dir / f"{unit_name}_layout.jpg"

                llm_bbox_map = None
                llm_txt = ""

                if schema_llm is not None:
                    print(f"[{unit_name}] Schema LLM align … ({len(schema_names_ordered)} keys)")
                    llm_bbox_map, llm_txt = schema_llm.predict_schema_map(
                        system_prompt_text=system_schema_txt,
                        fewshots_path=args.schema_align_fewshots,
                        schema_names_ordered=schema_names_ordered,
                        blocks=blocks,
                        max_new_tokens=args.schema_llm_max_new_tokens,
                        temperature=args.schema_llm_temperature,
                    )
                    n_llm_hit = sum(1 for v in llm_bbox_map.values() if v is not None)
                    blocks_out["schema_alignment_llm"] = llm_bbox_map
                    llm_sidecar = out_dir / f"{unit_name}_schema_llm.json"
                    payload = {
                        "source": blocks_out["source"],
                        "unit": blocks_out["unit"],
                        "image_size": blocks_out["image_size"],
                        "page_size_pt": blocks_out["page_size_pt"],
                        "schema_field_order": schema_names_ordered,
                        "schema_alignment_llm": llm_bbox_map,
                        "n_schema_fields_hit": n_llm_hit,
                        "n_schema_fields_total": len(schema_names_ordered),
                    }
                    llm_sidecar.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    llm_full = out_dir / f"{unit_name}_schema_llm_raw.txt"
                    llm_full.write_text(llm_txt, encoding="utf-8")
                    viz_llm_path = out_dir / f"{unit_name}_schema_llm.jpg"
                    _draw_schema_bbox_map(
                        img, schema_names_ordered, llm_bbox_map, viz_llm_path
                    )
                    print(f"[{unit_name}] LLM matched bbox for {n_llm_hit}/{len(schema_names_ordered)} "
                          "schema fields")

                json_path = out_dir / f"{unit_name}_blocks.json"
                json_path.write_text(
                    json.dumps(blocks_out, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                _draw_blocks(img, blocks, viz_path)

                print(f"[ok] {src_path.name} → {json_path.name}  viz → {viz_path.name}")
                row = {
                    "unit": unit_name,
                    "n_blocks": len(blocks),
                    "blocks_json": str(json_path),
                    "viz_layout": str(viz_path),
                }
                if schema_llm is not None:
                    row["schema_llm_json"] = str(out_dir / f"{unit_name}_schema_llm.json")
                    row["schema_llm_raw"] = str(out_dir / f"{unit_name}_schema_llm_raw.txt")
                    row["viz_schema_llm"] = str(out_dir / f"{unit_name}_schema_llm.jpg")
                summary["items"].append(row)

    finally:
        if schema_llm is not None:
            schema_llm.cleanup()

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
