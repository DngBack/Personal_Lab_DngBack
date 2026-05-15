"""Layout detection for GIẤY GỬI TIỀN TIẾT KIỆM using Chandra OCR 2.

Pipeline:
  1. Render PDF page → PIL image (PyMuPDF).
  2. Run Chandra OCR 2 with the Vietnamese form prompt from
     prompts/giay_gui_tien_tiet_kiem.txt.
  3. Parse <div data-bbox="..." data-label="..."> blocks from the HTML output.
  4. Hybrid matching: assign Chandra blocks to schema sections using a 5-pass
     strategy that avoids the cascade-failure of pure IoU on narrow field rows:
       Pass 1 – Section-Header label match (text similarity on header name).
       Pass 2 – Image block → "Logo" section.
       Pass 3 – Field text match: blocks with "FieldName: value" content are
                 matched to schema sections by normalized name similarity. This
                 is the primary strategy for all field-level sections and is
                 robust to small vertical shifts that kill IoU.
       Pass 4 – Table match: large/small Table blocks → Bảng kê ghi số /
                 Bảng kê tiền mặt.
       Pass 5 – IoU fallback for remaining unmatched sections.
     Unmatched sections fall back to the template bbox.
  5. Output per-page:
       <unit>_raw_layout.html      – raw model output
       <unit>_chandra_blocks.json  – parsed blocks (norm 0-1000)
       <unit>_schema_layout.json   – schema JSON with updated bboxes
       <unit>_schema_layout.jpg    – visualisation
     Plus run_summary.json at the end.

No manual scale/offset corrections are applied; bbox coordinates come directly
from the model output.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_DEFAULT_LAYOUT_JSON = (
    _HERE / "data/samples/GIAY_GUI_TIEN_TIET_KIEM/layout _GIAY_GUI_TIEN_TIET_KIEM.json"
)
_DEFAULT_PROMPT_FILE = _HERE / "prompts/giay_gui_tien_tiet_kiem.txt"
_DEFAULT_INPUT_DIR = _HERE / "data/small_test"
_DEFAULT_OUTPUT_DIR = _HERE / "results/giay_gui_tien_tiet_kiem"
_DEFAULT_MODEL = "datalab-to/chandra-ocr-2"

# ---------------------------------------------------------------------------
# Regex patterns for parsing model HTML
# ---------------------------------------------------------------------------

_DIV_RE = re.compile(
    r"<div\b(?P<attrs>[^>]*)>(?P<inner>.*?)</div>",
    re.IGNORECASE | re.DOTALL,
)
_BBOX_RE = re.compile(r'data(?:-bbox)?\s*=\s*"([^"]+)"', re.IGNORECASE)
_LABEL_RE = re.compile(r'data-label\s*=\s*"([^"]+)"', re.IGNORECASE)
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")

# ---------------------------------------------------------------------------
# Visualization palette
# ---------------------------------------------------------------------------

_PALETTE = [
    (220, 20, 60),   (30, 144, 255),  (50, 205, 50),   (255, 165, 0),
    (148, 0, 211),   (0, 191, 255),   (255, 105, 180), (154, 205, 50),
    (255, 215, 0),   (0, 206, 209),   (199, 21, 133),  (32, 178, 170),
    (255, 99, 71),   (60, 179, 113),  (123, 104, 238), (218, 165, 32),
]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chandra OCR 2 layout detection for GIẤY GỬI TIỀN TIẾT KIỆM.",
    )
    p.add_argument("--input-dir", type=Path, default=_DEFAULT_INPUT_DIR)
    p.add_argument("--input-file", type=Path, default=None,
                   help="Process a single PDF/image instead of --input-dir.")
    p.add_argument("--only", type=str, default=None,
                   help="Only process files whose name contains this substring.")
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    p.add_argument("--model", type=str, default=_DEFAULT_MODEL)
    p.add_argument("--lora-path", type=Path, default=None,
                   help="Optional PEFT LoRA adapter directory.")
    p.add_argument("--device-map", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=("auto", "bfloat16", "float16", "float32"))
    p.add_argument("--pdf-dpi", type=int, default=200)
    p.add_argument("--max-pixels", type=int, default=1_600_000)
    p.add_argument("--layout-json", type=Path, default=_DEFAULT_LAYOUT_JSON)
    p.add_argument("--prompt-file", type=Path, default=_DEFAULT_PROMPT_FILE)
    p.add_argument("--match-iou-min", type=float, default=0.05,
                   help="Minimum IoU (0–1) to accept a Chandra block for a schema section.")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--viz-chandra-boxes", action="store_true",
                   help="Also save *_chandra_boxes.jpg with raw Chandra blocks.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_PDF_EXT = {".pdf"}


def _list_inputs(folder: Path, only: str | None) -> list[Path]:
    if not folder.is_dir():
        return []
    return [
        p for p in sorted(folder.iterdir())
        if p.is_file()
        and p.suffix.lower() in (_IMAGE_EXT | _PDF_EXT)
        and (not only or only in p.name)
    ]


def _load_units(
    path: Path, pdf_dpi: int
) -> list[tuple[str, Any, tuple[float, float]]]:
    """Return list of (unit_name, PIL.Image, (page_w_pt, page_h_pt))."""
    from PIL import Image

    if path.suffix.lower() in _PDF_EXT:
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


def _fit_image(image: Any, max_pixels: int) -> Any:
    from PIL import Image

    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = (max_pixels / (w * h)) ** 0.5
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def _resolve_dtype(name: str) -> Any:
    import torch

    if name == "auto":
        return "auto"
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------


def load_schema(path: Path) -> list[dict[str, Any]]:
    """Load the full schema JSON, preserving all section fields.

    Returns a list of section dicts each enriched with:
        x_pt, y_pt, w_pt, h_pt   – floats from layout (PDF points)
    Sections without a layout entry are skipped.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for s in data.get("sections", []):
        layout = s.get("layout") or {}
        if not layout:
            continue
        sec = dict(s)
        sec["x_pt"] = float(layout["x"])
        sec["y_pt"] = float(layout["y"])
        sec["w_pt"] = float(layout["width"])
        sec["h_pt"] = float(layout["height"])
        out.append(sec)
    return out


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------


def parse_chandra_blocks(html: str) -> list[dict[str, Any]]:
    """Parse <div data-bbox="x0 y0 x1 y1" data-label="..."> blocks.

    Returns list of dicts:
      label       – str
      bbox_norm   – [x0, y0, x1, y1] in 0–1000
      inner_html  – raw inner content
      inner_lines – list of text lines (split by <br/>)
      inner_text  – full plain text
    """
    seen: set[tuple[str, str]] = set()
    blocks: list[dict[str, Any]] = []
    for m in _DIV_RE.finditer(html):
        attrs = m.group("attrs") or ""
        inner = m.group("inner") or ""
        b_match = _BBOX_RE.search(attrs)
        l_match = _LABEL_RE.search(attrs)
        if not b_match or not l_match:
            continue
        bbox_str = b_match.group(1).strip()
        label = l_match.group(1).strip()
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
        lines_html = _BR_RE.split(inner)
        lines = [
            re.sub(r"\s+", " ", _TAG_RE.sub("", lh)).strip()
            for lh in lines_html
        ]
        lines = [ln for ln in lines if ln]
        blocks.append({
            "label": label,
            "bbox_norm": [x0, y0, x1, y1],
            "inner_html": inner.strip(),
            "inner_lines": lines,
            "inner_text": " ".join(lines),
        })
    return blocks


def sanitize_html(raw: str) -> str:
    """Trim model artefacts that can appear after the layout HTML."""
    t = raw.strip()
    lower = t.lower()
    cut = len(t)
    for needle in ("<think>", "\nassistant\n", "\r\nassistant\n", "\nassistant"):
        i = lower.find(needle.lower())
        if 0 < i < cut:
            cut = i
    return t[:cut].strip()


# ---------------------------------------------------------------------------
# Matching utilities
# ---------------------------------------------------------------------------


def _iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _ascii_fold(s: str) -> str:
    """Lowercase, map Vietnamese-specific base chars, strip combining marks.

    Handles:
    - Combining diacritics: "lính" ↔ "lĩnh" (ĩ/í both → i)
    - Vietnamese base chars that don't NFKD-decompose to ASCII:
        đ/Đ → d,  ơ/Ơ → o,  ư/Ư → u
      Without this mapping, fold("đến") = "en" but fold("den") = "den" → mismatch
      when model outputs no-diacritic text.
    """
    import unicodedata

    # Map Vietnamese-specific base characters before NFKD decomposition
    _VN_BASE = str.maketrans("đĐơƠưƯ", "dDoOuU")
    s = s.translate(_VN_BASE)
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if "a" <= c <= "z" or c.isdigit())


def _text_sim(a: str, b: str) -> float:
    """Containment-based similarity on ASCII-folded strings."""
    fa, fb = _ascii_fold(a), _ascii_fold(b)
    if not fa or not fb:
        return 0.0
    if fa == fb:
        return 1.0
    if fa.startswith(fb) or fb.startswith(fa):
        return min(len(fa), len(fb)) / max(len(fa), len(fb))
    if fa in fb or fb in fa:
        return min(len(fa), len(fb)) / max(len(fa), len(fb)) * 0.9
    return 0.0


def _extract_field_label(block: dict[str, Any]) -> str | None:
    """Return the 'FieldName' part from a 'FieldName: value' text block.

    Checks all inner lines; returns the candidate with the highest coverage
    of the section-name vocabulary (any line whose colon-prefix exists).
    """
    for line in block.get("inner_lines", []):
        if ":" not in line:
            continue
        head = line.split(":", 1)[0].strip()
        if head and len(head) <= 60:
            return head
    return None


def _make_match(
    blk: dict[str, Any],
    j: int,
    method: str,
    score: float,
) -> dict[str, Any]:
    return {
        "source": "chandra",
        "bbox_norm": list(blk["bbox_norm"]),
        "chandra_label": blk["label"],
        "match_method": method,
        "score": round(score, 4),
        "block_index": j,
    }


def _make_template() -> dict[str, Any]:
    return {
        "source": "template",
        "bbox_norm": None,
        "chandra_label": None,
        "match_method": "none",
        "score": 0.0,
        "block_index": None,
    }


# ---------------------------------------------------------------------------
# Hybrid matching  (text-primary, IoU-fallback)
# ---------------------------------------------------------------------------

# Table-type schema sections
_TABLE_SECTIONS = {"Bảng kê tiền mặt", "Bảng kê ghi số"}
# Section-header schema sections (uppercase keys in the schema JSON)
_HEADER_SECTIONS = {
    "GIẤY GỬI TIỀN TIẾT KIỆM",
    "THÔNG TIN YÊU CẦU CỦA KHÁCH HÀNG",
    "BẢNG KÊ TIỀN MẶT (CASH LIST)",
    "PHẦN DÀNH CHO NGÂN HÀNG",
}
# Sections matched primarily by label type rather than text
_IMAGE_SECTION = "Logo"
# Signature / freeform sections that fall back to IoU
_FALLBACK_SECTIONS = {
    "Ngày",
    "Xác nhận thông tin",
    "Người gửi tiền",
    "Giao dịch viên",
    "Kiểm soát viên",
}


def hybrid_match(
    schema_sections: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    page_w_pt: float,
    page_h_pt: float,
    iou_min: float = 0.05,
    text_min: float = 0.6,
) -> dict[str, dict[str, Any]]:
    """Assign Chandra blocks to schema sections via a 5-pass hybrid strategy.

    Pure IoU matching fails for narrow field rows (~13 pt tall) because even a
    small vertical shift between the template and the actual scanned form drops
    IoU to near zero, causing cascade mis-assignments.

    Instead we use:
      Pass 1 – Section-Header blocks → header schema sections (text sim).
      Pass 2 – Image block → "Logo".
      Pass 3 – Text blocks with "FieldName: value" → field sections by name sim.
               This is the primary pass and handles almost all field rows.
      Pass 4 – Table blocks → table sections (largest→Bảng kê ghi số,
               smallest→Bảng kê tiền mặt).
      Pass 5 – IoU fallback for everything still unmatched (Ngày, signature
               areas, Xác nhận thông tin, etc.).

    Returns {section_name: match_dict}.
    """
    used: set[int] = set()
    result: dict[str, dict[str, Any]] = {}

    # Pre-compute IoU-compatible 0–1 bbox for each block
    ch_01: list[tuple[float, float, float, float]] = []
    for blk in blocks:
        x0, y0, x1, y1 = blk["bbox_norm"]
        ch_01.append((x0 / 1000.0, y0 / 1000.0, x1 / 1000.0, y1 / 1000.0))

    # ------------------------------------------------------------------ #
    # Pass 1: Section-Header blocks → header-type schema sections
    # ------------------------------------------------------------------ #
    header_blocks = [(j, blk) for j, blk in enumerate(blocks)
                     if blk["label"].lower() == "section-header"]

    for sec in sorted(schema_sections, key=lambda s: s["ord"]):
        if sec["name"] not in _HEADER_SECTIONS:
            continue
        best_j, best_sc = None, 0.0
        for j, blk in header_blocks:
            if j in used:
                continue
            sc = _text_sim(sec["name"], blk["inner_text"])
            if sc > best_sc:
                best_sc, best_j = sc, j
        if best_j is not None and best_sc >= text_min:
            used.add(best_j)
            result[sec["name"]] = _make_match(blocks[best_j], best_j, "header-text", best_sc)

    # ------------------------------------------------------------------ #
    # Pass 2: Image block → Logo
    # ------------------------------------------------------------------ #
    image_blocks = [(j, blk) for j, blk in enumerate(blocks)
                    if blk["label"].lower() == "image" and j not in used]
    if image_blocks:
        j, blk = image_blocks[0]
        used.add(j)
        result[_IMAGE_SECTION] = _make_match(blk, j, "image-label", 1.0)

    # ------------------------------------------------------------------ #
    # Pass 3: "FieldName: value" text blocks → field sections by name sim
    # ------------------------------------------------------------------ #
    # For each schema section not yet matched and not in the structural sets,
    # find the block whose extracted field label best matches the section name.
    structural = _HEADER_SECTIONS | _TABLE_SECTIONS | _FALLBACK_SECTIONS | {_IMAGE_SECTION}

    field_sections = [
        s for s in sorted(schema_sections, key=lambda s: s["ord"])
        if s["name"] not in structural and s["name"] not in result
    ]

    # Build candidate list: (block_idx, field_label) for text blocks
    field_candidates: list[tuple[int, str]] = []
    for j, blk in enumerate(blocks):
        if j in used:
            continue
        if blk["label"].lower() not in ("text", "form", "complex-block"):
            continue
        label = _extract_field_label(blk)
        if label:
            field_candidates.append((j, label))

    # Greedy best-score match per section
    for sec in field_sections:
        if sec["name"] in result:
            continue
        best_j, best_sc = None, 0.0
        for j, label in field_candidates:
            if j in used:
                continue
            sc = _text_sim(sec["name"], label)
            if sc > best_sc:
                best_sc, best_j = sc, j
        if best_j is not None and best_sc >= text_min:
            used.add(best_j)
            result[sec["name"]] = _make_match(blocks[best_j], best_j, "field-text", best_sc)

    # ------------------------------------------------------------------ #
    # Pass 4: Table blocks → table sections
    # ------------------------------------------------------------------ #
    table_blocks = [(j, blk) for j, blk in enumerate(blocks)
                    if blk["label"].lower() == "table" and j not in used]

    def _table_area(blk: dict[str, Any]) -> float:
        x0, y0, x1, y1 = blk["bbox_norm"]
        return (x1 - x0) * (y1 - y0)

    table_blocks_sorted = sorted(table_blocks, key=lambda jb: _table_area(jb[1]), reverse=True)

    table_sec_names = [
        s["name"] for s in sorted(schema_sections, key=lambda s: s["ord"])
        if s["name"] in _TABLE_SECTIONS and s["name"] not in result
    ]

    # Largest table → Bảng kê ghi số (the wide bank ledger), then next → Bảng kê tiền mặt
    # Sort table sections so "Bảng kê ghi số" comes first (it's ord=31 vs ord=20)
    table_sec_names_sorted = sorted(
        table_sec_names,
        key=lambda n: 0 if "ghi số" in n.lower() else 1,
    )
    for sec_name, (j, blk) in zip(table_sec_names_sorted, table_blocks_sorted):
        used.add(j)
        result[sec_name] = _make_match(blk, j, "table-size", _table_area(blk))

    # ------------------------------------------------------------------ #
    # Pass 5: IoU fallback for all still-unmatched sections
    # ------------------------------------------------------------------ #
    for sec in sorted(schema_sections, key=lambda s: s["ord"]):
        if sec["name"] in result:
            continue
        sx0 = sec["x_pt"] / page_w_pt
        sy0 = sec["y_pt"] / page_h_pt
        sx1 = (sec["x_pt"] + sec["w_pt"]) / page_w_pt
        sy1 = (sec["y_pt"] + sec["h_pt"]) / page_h_pt
        sbox = (
            max(0.0, min(1.0, sx0)),
            max(0.0, min(1.0, sy0)),
            max(0.0, min(1.0, sx1)),
            max(0.0, min(1.0, sy1)),
        )
        best_j, best_iou_val = None, 0.0
        for j, cbox in enumerate(ch_01):
            if j in used:
                continue
            iv = _iou(sbox, cbox)
            if iv > best_iou_val:
                best_iou_val, best_j = iv, j

        if best_j is not None and best_iou_val >= iou_min:
            used.add(best_j)
            result[sec["name"]] = _make_match(blocks[best_j], best_j, "iou", best_iou_val)
        else:
            result[sec["name"]] = _make_template()

    return result


# ---------------------------------------------------------------------------
# Build final output schema JSON
# ---------------------------------------------------------------------------


def build_output_schema(
    schema_sections: list[dict[str, Any]],
    matches: dict[str, dict[str, Any]],
    page_w_pt: float,
    page_h_pt: float,
    page_num: int = 1,
) -> dict[str, Any]:
    """Build the output schema JSON with updated bboxes.

    Sections matched by Chandra use the Chandra bbox (converted to PDF pt).
    Unmatched sections keep the template bbox.
    The output structure mirrors the sample layout JSON.
    """
    out_sections: list[dict[str, Any]] = []
    for sec in sorted(schema_sections, key=lambda s: s["ord"]):
        name = sec["name"]
        m = matches.get(name, {})

        if m.get("source") == "chandra" and m.get("bbox_norm"):
            x0_n, y0_n, x1_n, y1_n = m["bbox_norm"]
            # Convert from 0–1000 norm to PDF pt
            x_pt = x0_n / 1000.0 * page_w_pt
            y_pt = y0_n / 1000.0 * page_h_pt
            w_pt = (x1_n - x0_n) / 1000.0 * page_w_pt
            h_pt = (y1_n - y0_n) / 1000.0 * page_h_pt
            bbox_source = "chandra"
        else:
            # Fallback to template bbox (preserved as-is)
            x_pt = sec["x_pt"]
            y_pt = sec["y_pt"]
            w_pt = sec["w_pt"]
            h_pt = sec["h_pt"]
            bbox_source = "template"

        layout = {
            "x": round(x_pt, 4),
            "y": round(y_pt, 4),
            "width": round(w_pt, 4),
            "height": round(h_pt, 4),
            "page": page_num,
        }

        # Reconstruct section, replacing layout only
        out_sec: dict[str, Any] = {
            "id": sec.get("id", str(uuid.uuid4())),
            "name": name,
            "required": sec.get("required", False),
            "layout": layout,
            "ord": sec["ord"],
            "items": sec.get("items", []),
            "_bbox_source": bbox_source,
            "_match_method": m.get("match_method", "none"),
            "_match_score": m.get("score"),
            "_chandra_label": m.get("chandra_label"),
        }
        if m.get("bbox_norm"):
            out_sec["_chandra_bbox_norm"] = m["bbox_norm"]
        out_sections.append(out_sec)

    return {"sections": out_sections}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _load_font(image_size: tuple[int, int], font_scale: float = 1.0) -> Any:
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


def _rect_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def draw_schema_layout(
    image: Any,
    entries: list[dict[str, Any]],
    out_path: Path,
    font_scale: float = 1.0,
) -> None:
    """Draw boxes + non-overlapping labels onto a copy of *image* and save."""
    from PIL import ImageDraw

    img = image.copy()
    draw = ImageDraw.Draw(img)
    iw, ih = img.size
    font = _load_font(img.size, font_scale)
    line_w = max(2, min(iw, ih) // 500)

    color_map: dict[str, tuple[int, int, int]] = {}

    def color_for(label: str) -> tuple[int, int, int]:
        if label not in color_map:
            color_map[label] = _PALETTE[len(color_map) % len(_PALETTE)]
        return color_map[label]

    sorted_entries = sorted(entries, key=lambda e: (e["box_xyxy"][1], e["box_xyxy"][0]))

    # Draw rectangles first
    for e in sorted_entries:
        x0, y0, x1, y1 = e["box_xyxy"]
        color = color_for(e["label"])
        draw.rectangle([x0, y0, x1, y1], outline=color, width=line_w)

    # Then labels with collision avoidance
    placed: list[tuple[float, float, float, float]] = []
    for e in sorted_entries:
        x0, y0, x1, y1 = e["box_xyxy"]
        label = str(e["label"])
        src = e.get("source", "")
        color = color_for(label)

        cap = label if not src else f"{label} [{src[:1].upper()}]"
        bbox = draw.textbbox((0, 0), cap, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 3
        lab_w, lab_h = tw + 2 * pad, th + 2 * pad

        candidates = [
            (x0, y0 - lab_h),
            (max(0, x1 - lab_w), y0 - lab_h),
            (max(0, min(x0, iw - lab_w)), y1 + 1),
            (x1 + 2, y0),
            (max(0, x0 - lab_w - 2), y0),
        ]
        if (y1 - y0) > lab_h * 1.4:
            candidates.insert(0, (x0 + 2, y0 + 2))

        chosen: tuple[float, float, float, float] | None = None
        for lx0, ly0 in candidates:
            lx0 = max(0.0, min(lx0, iw - lab_w))
            ly0 = max(0.0, min(ly0, ih - lab_h))
            r = (lx0, ly0, lx0 + lab_w, ly0 + lab_h)
            if not any(_rect_overlap(r, p) for p in placed):
                chosen = r
                break

        if chosen is None:
            lx0 = max(0.0, min(x0, iw - lab_w))
            ly0 = max(0.0, y0 - lab_h)
            for shift in range(0, 40 * (int(lab_h) + 1), int(lab_h) + 1):
                r = (lx0, ly0 + shift, lx0 + lab_w, ly0 + shift + lab_h)
                if r[3] > ih:
                    break
                if not any(_rect_overlap(r, p) for p in placed):
                    chosen = r
                    break
            if chosen is None:
                chosen = (lx0, max(0.0, min(y0, ih - lab_h)),
                          lx0 + lab_w, max(lab_h, min(y0 + lab_h, ih)))

        lx0, ly0, lx1, ly1 = chosen
        draw.rectangle([lx0, ly0, lx1, ly1], fill=color)
        text_y = ly0 + pad - bbox[1]
        draw.text((lx0 + pad, text_y), cap, fill=(255, 255, 255), font=font)
        placed.append(chosen)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)


def draw_chandra_boxes(
    image: Any,
    blocks: list[dict[str, Any]],
    out_path: Path,
    font_scale: float = 1.0,
) -> None:
    """Draw raw Chandra block bboxes (norm 0–1000) on the image."""
    iw, ih = image.size
    entries = [
        {
            "label": blk["label"],
            "source": "",
            "box_xyxy": [
                blk["bbox_norm"][0] / 1000.0 * iw,
                blk["bbox_norm"][1] / 1000.0 * ih,
                blk["bbox_norm"][2] / 1000.0 * iw,
                blk["bbox_norm"][3] / 1000.0 * ih,
            ],
        }
        for blk in blocks
    ]
    draw_schema_layout(image, entries, out_path, font_scale=font_scale)


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------


def run_inference(
    model: Any,
    processor: Any,
    image: Any,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    do_sample = temperature > 0
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
        )
    in_len = inputs["input_ids"].shape[-1]
    trimmed = generated[:, in_len:]
    text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return sanitize_html(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run() -> int:
    args = _parse_args()

    # Resolve inputs
    if args.input_file is not None:
        if not args.input_file.is_file():
            print(f"Input file not found: {args.input_file}", file=sys.stderr)
            return 1
        inputs_list = [args.input_file]
    else:
        inputs_list = _list_inputs(args.input_dir, args.only)
    if not inputs_list:
        exts = sorted(_IMAGE_EXT | _PDF_EXT)
        print(f"No inputs found in {args.input_dir} (extensions: {exts})", file=sys.stderr)
        return 1

    # Load layout schema
    if not args.layout_json.is_file():
        print(f"Layout JSON not found: {args.layout_json}", file=sys.stderr)
        return 1
    schema_sections = load_schema(args.layout_json)
    print(f"[layout] Loaded {len(schema_sections)} schema sections from {args.layout_json.name}")

    # Load prompt
    if not args.prompt_file.is_file():
        print(f"Prompt file not found: {args.prompt_file}", file=sys.stderr)
        return 1
    prompt_text = args.prompt_file.read_text(encoding="utf-8").strip()
    if not prompt_text:
        print(f"Prompt file is empty: {args.prompt_file}", file=sys.stderr)
        return 1
    print(f"[layout] Prompt: {args.prompt_file.name} ({len(prompt_text)} chars)")

    # Load model
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = _resolve_dtype(args.dtype)
    print(f"[layout] Loading {args.model}  dtype={dtype}  device_map={args.device_map}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device_map,
    )
    proc_src = args.model
    if args.lora_path is not None:
        if not args.lora_path.is_dir():
            print(f"LoRA path not found: {args.lora_path}", file=sys.stderr)
            return 1
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.lora_path))
        if (args.lora_path / "tokenizer_config.json").is_file():
            proc_src = str(args.lora_path)
        print(f"[layout] Loaded LoRA from {args.lora_path}")
    processor = AutoProcessor.from_pretrained(proc_src)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "model": args.model,
        "lora_path": str(args.lora_path) if args.lora_path else None,
        "prompt_file": str(args.prompt_file),
        "layout_json": str(args.layout_json),
        "device_map": args.device_map,
        "dtype": str(dtype),
        "match_iou_min": args.match_iou_min,
        "items": [],
    }

    for src_path in inputs_list:
        for unit_name, image, page_pt in _load_units(src_path, args.pdf_dpi):
            page_w_pt, page_h_pt = page_pt
            img = _fit_image(image, args.max_pixels)
            img_w, img_h = img.size

            # Save input image
            input_path = args.output_dir / f"{unit_name}_input.jpg"
            img.save(input_path, quality=92)

            # Run inference
            print(f"[{unit_name}] Running inference …")
            raw_html = run_inference(
                model, processor, img,
                prompt_text, args.max_new_tokens, args.temperature,
            )

            html_path = args.output_dir / f"{unit_name}_raw_layout.html"
            html_path.write_text(raw_html, encoding="utf-8")

            # Parse blocks
            blocks = parse_chandra_blocks(raw_html)
            blocks_path = args.output_dir / f"{unit_name}_chandra_blocks.json"
            blocks_path.write_text(
                json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[{unit_name}] Parsed {len(blocks)} Chandra blocks")

            # Hybrid matching (text-primary, IoU-fallback)
            matches = hybrid_match(
                schema_sections, blocks, page_w_pt, page_h_pt,
                iou_min=args.match_iou_min,
            )
            n_ch = sum(1 for v in matches.values() if v["source"] == "chandra")
            n_tmpl = sum(1 for v in matches.values() if v["source"] == "template")
            by_method: dict[str, int] = {}
            for v in matches.values():
                m = v.get("match_method", "none")
                by_method[m] = by_method.get(m, 0) + 1
            method_str = ", ".join(f"{m}={c}" for m, c in sorted(by_method.items()))
            print(f"[{unit_name}] Matched {n_ch}/{len(schema_sections)} sections "
                  f"from Chandra; {n_tmpl} template fallback [{method_str}]")

            # Build output schema JSON
            out_schema = build_output_schema(
                schema_sections, matches, page_w_pt, page_h_pt, page_num=1
            )
            json_path = args.output_dir / f"{unit_name}_schema_layout.json"
            full_json: dict[str, Any] = {
                "source": str(src_path),
                "unit": unit_name,
                "image_size": [img_w, img_h],
                "page_size_pt": [page_w_pt, page_h_pt],
                "input_image": str(input_path),
                "raw_html_path": str(html_path),
                "n_sections_chandra": n_ch,
                "n_sections_template": n_tmpl,
                "sections": out_schema["sections"],
            }
            json_path.write_text(
                json.dumps(full_json, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            # Visualization: schema layout
            viz_entries: list[dict[str, Any]] = []
            for sec in sorted(schema_sections, key=lambda s: s["ord"]):
                name = sec["name"]
                layout_out = next(
                    (s["layout"] for s in out_schema["sections"] if s["name"] == name), None
                )
                match_src = matches.get(name, {}).get("source", "template")
                if not layout_out:
                    continue
                # Convert PDF pt → pixel
                sx = img_w / page_w_pt
                sy = img_h / page_h_pt
                x0 = layout_out["x"] * sx
                y0 = layout_out["y"] * sy
                x1 = (layout_out["x"] + layout_out["width"]) * sx
                y1 = (layout_out["y"] + layout_out["height"]) * sy
                viz_entries.append({
                    "label": name,
                    "source": match_src,
                    "box_xyxy": [x0, y0, x1, y1],
                })

            viz_path = args.output_dir / f"{unit_name}_schema_layout.jpg"
            draw_schema_layout(img, viz_entries, viz_path)

            # Optional: raw Chandra boxes visualization
            chandra_viz_path: str | None = None
            if args.viz_chandra_boxes and blocks:
                cb_path = args.output_dir / f"{unit_name}_chandra_boxes.jpg"
                draw_chandra_boxes(img, blocks, cb_path)
                chandra_viz_path = str(cb_path)
                print(f"[{unit_name}] Raw Chandra boxes → {cb_path.name}")

            item: dict[str, Any] = {
                "source": str(src_path),
                "unit": unit_name,
                "image_size": [img_w, img_h],
                "viz": str(viz_path),
                "json": str(json_path),
                "raw_html": str(html_path),
                "blocks_json": str(blocks_path),
                "n_blocks": len(blocks),
                "n_sections": len(schema_sections),
                "n_sections_chandra": n_ch,
                "n_sections_template": n_tmpl,
            }
            if chandra_viz_path:
                item["viz_chandra_boxes"] = chandra_viz_path
            summary["items"].append(item)
            print(f"[ok] {src_path.name} :: {unit_name} → {viz_path.name}")

    summary_path = args.output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] Wrote summary → {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
