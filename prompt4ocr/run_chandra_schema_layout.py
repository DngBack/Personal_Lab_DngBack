"""Schema-faithful layout detection on GIẤY GỬI TIỀN TIẾT KIỆM with Chandra OCR 2.

Pipeline (single page):

  1. Render the test PDF/image to a PIL image.
  2. Run Chandra OCR 2 with the official ``ocr_layout`` prompt to get a list of
     ``<div data-bbox="..." data-label="...">...inner HTML...</div>`` blocks.
  3. From those blocks compute an **auto-alignment** offset (in PDF points)
     between the schema template and the actual scan, by matching:
       * ``Image`` block at top-left   ↔ schema ``Logo``
       * the topmost centred ``Section-Header`` ↔ schema ``GIẤY GỬI TIỀN TIẾT KIỆM``
       * any ``Section-Header`` whose text contains ``BẢNG KÊ TIỀN MẶT`` /
         ``PHẦN DÀNH CHO NGÂN HÀNG`` ↔ those schema sections
       * the largest ``Table`` ↔ schema ``Bảng kê ghi số``
     Median (dx, dy) is applied to every schema bbox.
  4. The big multi-line ``Text`` block under "THÔNG TIN YÊU CẦU CỦA KHÁCH HÀNG"
     is split by ``<br/>`` lines. Each line of the form ``Field name: value`` is
     mapped to the matching schema field; that field's bbox is recomputed from
     the line's (x0, y0_line, x1, y1_line) inside the Chandra block, so labels
     follow the actual content positions, not the rigid template grid.
  5. Signature sections (``Người gửi tiền``, ``Giao dịch viên``,
     ``Kiểm soát viên``) get their height extended downward to cover the full
     signature + name + stamp area (configurable via --sig-extend-to-pt).
  6. Render the visualisation with **smart label placement** so labels never
     overlap each other and never fall off the right / top edges.

Outputs (per page) under ``--output-dir``:
  * ``<unit>_input.jpg``                    : the resized test page.
  * ``<unit>_chandra_layout.html``          : raw model HTML.
  * ``<unit>_chandra_blocks.json``          : parsed Chandra blocks.
  * ``<unit>_schema_layout.json``           : final per-schema-field bboxes (px).
  * ``<unit>_schema_layout.jpg``            : visualisation (the headline).
  * ``<unit>_chandra_anchors.json``         : the (schema name, chandra block,
                                               offset_pt) anchor pairs used.
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

DIV_RE = re.compile(
    r'<div\b(?P<attrs>[^>]*)>(?P<inner>.*?)</div>',
    re.IGNORECASE | re.DOTALL,
)
BBOX_RE = re.compile(r'data-bbox\s*=\s*"([^"]+)"', re.IGNORECASE)
LABEL_RE = re.compile(r'data-label\s*=\s*"([^"]+)"', re.IGNORECASE)
BR_RE = re.compile(r'<br\s*/?>', re.IGNORECASE)
TAG_RE = re.compile(r'<[^>]+>')

# Per-schema-section *target bottom* Y in PDF pt. The schema's original height
# of these sections often clips the actual signature / stamp content so we
# extend their y1 to the target. Numbers are tuned for the GIẤY GỬI TIỀN
# template (A4 portrait, 842 pt tall).
DEFAULT_SECTION_BOTTOM_PT: dict[str, float] = {
    "Người gửi tiền":   555.0,   # customer signature + printed name
    "Giao dịch viên":   832.0,   # bottom-left signature + name + stamp
    "Kiểm soát viên":   832.0,   # bottom-right signature + name + stamp
}

# X-axis column boundary in PDF pt: customer/bank info forms use a two-column
# layout where left fields end around x≈280 and right fields start around x≈285.
# When the override would otherwise stretch a left field into the right column
# (because Chandra merged both columns into one Text block), we clip x1.
LEFT_COLUMN_X1_PT_MAX = 280.0
RIGHT_COLUMN_X0_PT_MIN = 285.0
LEFT_COLUMN_LABEL_PAD_PT = 60.0   # how far we extend left fields past schema x1
RIGHT_COLUMN_LABEL_PAD_PT = 30.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chandra OCR 2 + schema layout overlay.")
    default_in = PROJECT_DIR / "data/test/GIAY_GUI_TIEN_TIET_KIEM"
    default_out = PROJECT_DIR / "results/chandra_schema_layout"
    p.add_argument("--input-dir", type=Path, default=default_in)
    p.add_argument("--input-file", type=Path, default=None)
    p.add_argument("--only", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=default_out)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
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
    p.add_argument(
        "--sig-page-margin-pt", type=float, default=10.0,
        help="Min margin (pt) to keep above page bottom when extending signatures.",
    )
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--font-scale", type=float, default=1.0,
        help="Label font size scale.",
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


def build_messages(test_image: Any) -> list[dict[str, Any]]:
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": test_image},
            {"type": "text", "text": OCR_LAYOUT_PROMPT},
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


def norm_to_pt(bbox_norm: list[float], page_w_pt: float, page_h_pt: float) -> list[float]:
    return [
        bbox_norm[0] / 1000.0 * page_w_pt,
        bbox_norm[1] / 1000.0 * page_h_pt,
        bbox_norm[2] / 1000.0 * page_w_pt,
        bbox_norm[3] / 1000.0 * page_h_pt,
    ]


def _norm_text(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def find_anchors(
    blocks: list[dict[str, Any]],
    page_w_pt: float,
    page_h_pt: float,
) -> list[dict[str, Any]]:
    """Return a list of anchor pairs:
       [{schema_name, schema_box_pt, chandra_block, chandra_box_pt, dx_pt, dy_pt}].
    """
    schema_anchors: dict[str, tuple[list[float], str]] = {
        "Logo":                          ([58.0,  68.0,  153.0, 101.0], "image"),
        "GIẤY GỬI TIỀN TIẾT KIỆM":       ([226.0, 87.0,  402.0, 105.0], "title"),
        "THÔNG TIN YÊU CẦU CỦA KHÁCH HÀNG": ([55.0, 126.0, 253.0, 139.0], "section_header"),
        "BẢNG KÊ TIỀN MẶT (CASH LIST)":  ([54.0,  343.0, 213.0, 357.0], "section_header"),
        "Bảng kê tiền mặt":              ([56.0,  364.0, 300.0, 422.0], "table_small"),
        "PHẦN DÀNH CHO NGÂN HÀNG":       ([52.0,  511.0, 201.0, 524.0], "section_header"),
        "Bảng kê ghi số":                ([52.0,  600.0, 584.0, 668.0], "table_big"),
    }

    images = [b for b in blocks if b["label"].lower() == "image"]
    headers = [b for b in blocks if b["label"].lower() == "section-header"]
    tables = [b for b in blocks if b["label"].lower() == "table"]

    pairs: list[dict[str, Any]] = []

    def append_pair(name: str, schema_box_pt: list[float], blk: dict[str, Any]) -> None:
        cb_pt = norm_to_pt(blk["bbox_norm"], page_w_pt, page_h_pt)
        sx0, sy0, sx1, sy1 = schema_box_pt
        cx0, cy0, cx1, cy1 = cb_pt
        dx = ((cx0 + cx1) - (sx0 + sx1)) / 2.0
        dy = ((cy0 + cy1) - (sy0 + sy1)) / 2.0
        pairs.append({
            "schema_name": name,
            "schema_box_pt": schema_box_pt,
            "chandra_label": blk["label"],
            "chandra_box_pt": cb_pt,
            "chandra_text": blk["inner_text"][:80],
            "dx_pt": round(dx, 2),
            "dy_pt": round(dy, 2),
        })

    if images:
        top_left = min(images, key=lambda b: (b["bbox_norm"][1], b["bbox_norm"][0]))
        append_pair("Logo", schema_anchors["Logo"][0], top_left)

    for sec_name in (
        "GIẤY GỬI TIỀN TIẾT KIỆM",
        "THÔNG TIN YÊU CẦU CỦA KHÁCH HÀNG",
        "BẢNG KÊ TIỀN MẶT (CASH LIST)",
        "PHẦN DÀNH CHO NGÂN HÀNG",
    ):
        sec_key = _norm_text(sec_name)
        match = None
        for h in headers:
            hkey = _norm_text(h["inner_text"])
            if hkey and (sec_key in hkey or hkey in sec_key):
                match = h
                break
        if match is not None:
            append_pair(sec_name, schema_anchors[sec_name][0], match)

    if tables:
        big = max(tables, key=lambda b: (
            (b["bbox_norm"][2] - b["bbox_norm"][0]) *
            (b["bbox_norm"][3] - b["bbox_norm"][1])
        ))
        append_pair("Bảng kê ghi số", schema_anchors["Bảng kê ghi số"][0], big)
        if len(tables) >= 2:
            small = min(tables, key=lambda b: (
                (b["bbox_norm"][2] - b["bbox_norm"][0]) *
                (b["bbox_norm"][3] - b["bbox_norm"][1])
            ))
            if small is not big:
                append_pair("Bảng kê tiền mặt",
                            schema_anchors["Bảng kê tiền mặt"][0], small)
    return pairs


def median_offset(pairs: list[dict[str, Any]]) -> tuple[float, float]:
    if not pairs:
        return 0.0, 0.0
    dxs = sorted(p["dx_pt"] for p in pairs)
    dys = sorted(p["dy_pt"] for p in pairs)
    mid = len(dxs) // 2
    dx = dxs[mid] if len(dxs) % 2 else 0.5 * (dxs[mid - 1] + dxs[mid])
    dy = dys[mid] if len(dys) % 2 else 0.5 * (dys[mid - 1] + dys[mid])
    return dx, dy


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


def split_text_block_by_lines(
    block: dict[str, Any],
    page_w_pt: float,
    page_h_pt: float,
) -> list[dict[str, Any]]:
    """Split a multi-line Chandra Text block into one bbox per line in PDF pt.

    Returns list of {text, bbox_pt}.
    """
    lines = block["inner_lines"]
    if not lines:
        return []
    x0_pt, y0_pt, x1_pt, y1_pt = norm_to_pt(block["bbox_norm"], page_w_pt, page_h_pt)
    line_h = (y1_pt - y0_pt) / len(lines)
    out: list[dict[str, Any]] = []
    for i, ln in enumerate(lines):
        out.append({
            "text": ln,
            "bbox_pt": [x0_pt, y0_pt + i * line_h,
                        x1_pt, y0_pt + (i + 1) * line_h],
        })
    return out


def field_label_from_line(line: str) -> str | None:
    """Return the field name part of a 'Field: value' line, normalised."""
    if ":" not in line:
        return None
    head = line.split(":", 1)[0].strip()
    if not head or len(head) > 60:
        return None
    return head


CUSTOMER_FIELD_NAMES = [
    "Tên khách hàng", "CMND/CCCD/HC", "Ngày cấp", "CIF", "Số tiền gửi",
    "Nơi cấp", "Loại tiền", "Số tiền bằng chữ", "Loại hình sản phẩm",
    "Kỳ hạn", "Định kỳ lĩnh lãi", "Phương thức gửi tiền", "Tài khoản ghi nợ",
    "Tài khoản nhận lãi", "Phương thức quay vòng",
]
BANK_FIELD_NAMES = [
    "Số bút toán", "Số seri", "Ngày mở", "Salecode",
    "Company", "Lãi suất", "Ngày đến hạn",
]


def best_match_field(line_head: str, candidates: list[str]) -> str | None:
    head = _norm_text(line_head)
    if not head:
        return None
    best, best_score = None, 0.0
    for c in candidates:
        ck = _norm_text(c)
        if not ck:
            continue
        if head == ck:
            return c
        if head in ck or ck in head:
            score = min(len(head), len(ck)) / max(len(head), len(ck))
            if score > best_score:
                best_score = score
                best = c
    return best if best_score >= 0.6 else None


def assign_lines_to_fields(
    blocks: list[dict[str, Any]],
    page_w_pt: float,
    page_h_pt: float,
    candidate_fields: list[str],
) -> dict[str, list[float]]:
    """Try to map each candidate field to a line bbox extracted from the
    multi-line Text blocks Chandra detected. Returns {field_name: bbox_pt}.
    """
    out: dict[str, list[float]] = {}
    for blk in blocks:
        if blk["label"].lower() != "text":
            continue
        if not blk["inner_lines"]:
            continue
        line_boxes = split_text_block_by_lines(blk, page_w_pt, page_h_pt)
        for lb in line_boxes:
            head = field_label_from_line(lb["text"])
            if not head:
                continue
            name = best_match_field(head, candidate_fields)
            if name and name not in out:
                out[name] = lb["bbox_pt"]
    return out


def find_signature_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chandra marks signature regions with `<img alt="Signature ...">` inside Text."""
    out = []
    for b in blocks:
        if b["label"].lower() != "text":
            continue
        if "alt=\"signature" in b["inner_html"].lower():
            out.append(b)
    return out


def section_box_pt(
    sec: dict[str, Any],
    offset: tuple[float, float],
    section_bottom_pt: dict[str, float],
    page_margin_pt: float,
    field_overrides_pt: dict[str, list[float]],
    page_w_pt: float,
    page_h_pt: float,
) -> list[float]:
    """Build the final bbox in PDF pt for a schema section.

    For non-override sections, just shift schema by (dx, dy) and optionally
    extend the bottom for signatures.

    For override sections (field detected by Chandra line splitting), we trust
    Chandra's **Y** range (line position) but keep **X** from schema with
    column-aware extension. This avoids the "left field stretches into right
    column" bug seen when Chandra merges both columns into one Text block.
    """
    name = sec["name"]
    sx0 = sec["x_pt"] + offset[0]
    sy0 = sec["y_pt"] + offset[1]
    sx1 = sx0 + sec["w_pt"]
    sy1 = sy0 + sec["h_pt"]

    if name in field_overrides_pt:
        _, oy0, _, oy1 = field_overrides_pt[name]
        y0 = oy0
        y1 = oy1
        if sx0 < LEFT_COLUMN_X1_PT_MAX:
            x0 = sx0
            x1 = min(LEFT_COLUMN_X1_PT_MAX, sx1 + LEFT_COLUMN_LABEL_PAD_PT)
        else:
            x0 = max(RIGHT_COLUMN_X0_PT_MIN, sx0)
            x1 = min(page_w_pt - 6.0, sx1 + RIGHT_COLUMN_LABEL_PAD_PT)
    else:
        x0, y0, x1, y1 = sx0, sy0, sx1, sy1
        if name in section_bottom_pt:
            target_bottom = min(section_bottom_pt[name], page_h_pt - page_margin_pt)
            y1 = max(y1, target_bottom)

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

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = resolve_dtype(args.dtype)
    print(f"[chandra-schema] Loading {args.model} dtype={dtype} device_map={args.device_map}")
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
        "device_map": args.device_map,
        "dtype": str(dtype),
        "layout_json": str(args.layout_json),
        "page_size_pt": [args.page_width_pt, args.page_height_pt],
        "sig_page_margin_pt": args.sig_page_margin_pt,
        "section_bottom_pt": DEFAULT_SECTION_BOTTOM_PT,
        "items": [],
    }

    for src_path in inputs_list:
        for unit_name, image, page_pt in load_units(src_path, args.pdf_dpi):
            page_w_pt, page_h_pt = page_pt
            test_image = fit_to_max_pixels(image, args.max_pixels)
            img_w_px, img_h_px = test_image.size
            input_path = args.output_dir / f"{unit_name}_input.jpg"
            test_image.save(input_path, quality=92)

            messages = build_messages(test_image)
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

            anchors = find_anchors(blocks, page_w_pt, page_h_pt)
            dx_pt, dy_pt = median_offset(anchors)
            (args.output_dir / f"{unit_name}_chandra_anchors.json").write_text(
                json.dumps(
                    {"dx_pt": dx_pt, "dy_pt": dy_pt, "anchors": anchors},
                    ensure_ascii=False, indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[align] {unit_name}: dx={dx_pt:+.2f} dy={dy_pt:+.2f} pt "
                  f"(from {len(anchors)} anchors)")

            field_overrides_pt: dict[str, list[float]] = {}
            field_overrides_pt.update(
                assign_lines_to_fields(blocks, page_w_pt, page_h_pt, CUSTOMER_FIELD_NAMES)
            )
            field_overrides_pt.update(
                assign_lines_to_fields(blocks, page_w_pt, page_h_pt, BANK_FIELD_NAMES)
            )
            print(f"[fields] {unit_name}: {len(field_overrides_pt)} per-field overrides "
                  f"(parsed from Chandra inner lines)")

            entries: list[dict[str, Any]] = []
            schema_json: list[dict[str, Any]] = []
            sx = img_w_px / page_w_pt
            sy = img_h_px / page_h_pt
            for sec in sorted(schema_sections, key=lambda s: s["ord"]):
                box_pt = section_box_pt(
                    sec, (dx_pt, dy_pt),
                    DEFAULT_SECTION_BOTTOM_PT,
                    args.sig_page_margin_pt,
                    field_overrides_pt,
                    page_w_pt, page_h_pt,
                )
                box_px = [box_pt[0] * sx, box_pt[1] * sy,
                          box_pt[2] * sx, box_pt[3] * sy]
                entries.append({"label": sec["name"], "box_xyxy": box_px})
                schema_json.append({
                    "name": sec["name"],
                    "ord": sec["ord"],
                    "box_pt": [round(v, 2) for v in box_pt],
                    "box_xyxy_px": [round(v, 2) for v in box_px],
                    "source": ("override" if sec["name"] in field_overrides_pt
                               else "schema+offset"),
                })

            json_path = args.output_dir / f"{unit_name}_schema_layout.json"
            json_path.write_text(
                json.dumps({
                    "source": str(src_path),
                    "unit": unit_name,
                    "image_size": [img_w_px, img_h_px],
                    "input_image": str(input_path),
                    "raw_html_path": str(html_path),
                    "alignment_offset_pt": {"dx": dx_pt, "dy": dy_pt},
                    "n_anchors": len(anchors),
                    "n_field_overrides": len(field_overrides_pt),
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
                "n_field_overrides": len(field_overrides_pt),
                "alignment_offset_pt": [dx_pt, dy_pt],
            })
            print(f"[ok] {src_path.name} :: {unit_name} -> {len(entries)} sections "
                  f"({len(field_overrides_pt)} field-aligned) -> {viz_path.name}")

    summary_path = args.output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
