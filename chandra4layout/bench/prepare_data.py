#!/usr/bin/env python3
"""Chuẩn bị ảnh + blocks cache cho benchmark extraction (schema_align qua vLLM).

Rasterize 8 PDF trong data/test/GIAY_GUI_TIEN_TIET_KIEM → bench/data/images/
Parse blocks từ HTML Chandra có sẵn hoặc chạy --run-chandra để bổ sung.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from schema_align_llm import compact_blocks  # noqa: E402

_TEST_DIR = _ROOT / "data/test/GIAY_GUI_TIEN_TIET_KIEM"
_OUT_DIR = _HERE / "data"
_IMAGES_DIR = _OUT_DIR / "images"
_BLOCKS_DIR = _OUT_DIR / "blocks"

_DIV_RE = re.compile(
    r"<div\b(?P<attrs>[^>]*)>(?P<inner>.*?)</div>",
    re.IGNORECASE | re.DOTALL,
)
_BBOX_RE = re.compile(r'data(?:-bbox)?\s*=\s*"([^"]+)"', re.IGNORECASE)
_LABEL_RE = re.compile(r'data-label\s*=\s*"([^"]+)"', re.IGNORECASE)
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")

# HTML Chandra đã có sẵn trong repo
_KNOWN_HTML: dict[str, Path] = {
    "test_7": _ROOT / "results/giay_gui_tien_tiet_kiem_direct/test_7_page01_raw.html",
    "test_8": _ROOT / "results/giay_gui_tien_tiet_kiem_direct/test_8_page01_raw.html",
}


def _parse_html_blocks(html: str) -> list[dict]:
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
        label = lm.group(1).strip()
        key = (round(x0), round(y0), round(x1), round(y1), label)
        if key in seen:
            continue
        seen.add(key)
        lines_raw = _BR_RE.split(inner)
        lines = [re.sub(r"\s+", " ", _TAG_RE.sub("", ln)).strip() for ln in lines_raw]
        lines = [ln for ln in lines if ln]
        blocks.append({
            "label": label,
            "bbox": [x0, y0, x1, y1],
            "text": " ".join(lines),
            "lines": lines,
        })
    return blocks


def _parse_legacy_blocks(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else raw.get("blocks", [])
    out = []
    for b in items:
        bbox = b.get("bbox") or b.get("bbox_norm")
        if not bbox:
            continue
        text = b.get("text") or b.get("inner_text") or ""
        out.append({
            "label": b.get("label", "Text"),
            "bbox": list(bbox),
            "text": text,
            "lines": b.get("lines") or b.get("inner_lines") or ([text] if text else []),
        })
    return out


def _rasterize_pdf(pdf: Path, out_jpg: Path, dpi: int = 200) -> None:
    import fitz
    from PIL import Image as PILImage

    doc = fitz.open(str(pdf))
    page = doc[0]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_jpg, quality=92)
    doc.close()


def _run_chandra_html(pdf: Path, device: str) -> str:
    from run import _fit, _load_units, _sanitize  # type: ignore

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    prompt_path = _ROOT / "prompts/giay_gui_tien_tiet_kiem.txt"
    prompt_text = prompt_path.read_text(encoding="utf-8")
    model_id = "datalab-to/chandra-ocr-2"

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype="auto", device_map=device,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    model.eval()

    for _unit, image, _pt in _load_units(pdf, 200):
        img = _fit(image, 1_500_000)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt_text},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=8192, do_sample=False)
        raw = _sanitize(processor.batch_decode(
            gen[:, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0])
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return raw
    return ""


def prepare(*, run_chandra: bool, chandra_device: str, dpi: int) -> dict:
    pdfs = sorted(_TEST_DIR.glob("test_*.pdf"))
    if len(pdfs) < 8:
        raise SystemExit(f"Cần 8 PDF trong {_TEST_DIR}, hiện có {len(pdfs)}")

    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    _BLOCKS_DIR.mkdir(parents=True, exist_ok=True)

    manifest: dict = {"pdfs": [], "images": [], "blocks": []}

    for pdf in pdfs[:8]:
        stem = pdf.stem
        jpg = _IMAGES_DIR / f"{stem}_page01.jpg"
        _rasterize_pdf(pdf, jpg, dpi=dpi)
        manifest["images"].append(str(jpg.relative_to(_HERE)))

        blocks: list[dict] | None = None
        source = ""

        if stem in _KNOWN_HTML and _KNOWN_HTML[stem].is_file():
            blocks = _parse_html_blocks(_KNOWN_HTML[stem].read_text(encoding="utf-8"))
            source = f"cached_html:{_KNOWN_HTML[stem].name}"

        if blocks is None and stem == "test_1":
            legacy = _ROOT / "results/giay_gui_tien_tiet_kiem/test_1_page01_chandra_blocks.json"
            if legacy.is_file():
                blocks = _parse_legacy_blocks(legacy)
                source = f"legacy_json:{legacy.name}"

        cache_html = _BLOCKS_DIR / f"{stem}_raw.html"
        if blocks is None and cache_html.is_file():
            blocks = _parse_html_blocks(cache_html.read_text(encoding="utf-8"))
            source = f"cache:{cache_html.name}"

        if blocks is None and run_chandra:
            print(f"[chandra] {stem} …", flush=True)
            html = _run_chandra_html(pdf, chandra_device)
            cache_html.write_text(html, encoding="utf-8")
            blocks = _parse_html_blocks(html)
            source = "chandra_live"

        if blocks is None:
            # Fallback: dùng test_7 blocks (vẫn đủ để đo token/latency)
            ref = _KNOWN_HTML["test_7"]
            blocks = _parse_html_blocks(ref.read_text(encoding="utf-8"))
            source = f"proxy_from_test_7"

        blocks_path = _BLOCKS_DIR / f"{stem}_blocks.json"
        payload = {
            "pdf": str(pdf),
            "image": str(jpg),
            "source": source,
            "n_blocks": len(blocks),
            "blocks": blocks,
            "blocks_compact": compact_blocks(blocks),
        }
        blocks_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["pdfs"].append(str(pdf))
        manifest["blocks"].append(str(blocks_path.relative_to(_HERE)))
        print(f"[ok] {stem}: {len(blocks)} blocks  ({source})", flush=True)

    (_OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare bench images + blocks for extraction vLLM tests.")
    p.add_argument("--run-chandra", action="store_true", help="Chạy Chandra cho PDF chưa có HTML cache.")
    p.add_argument("--chandra-device", default="cuda:0")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()
    prepare(run_chandra=args.run_chandra, chandra_device=args.chandra_device, dpi=args.dpi)


if __name__ == "__main__":
    main()
