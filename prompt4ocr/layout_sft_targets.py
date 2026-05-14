"""Shared helpers: schema layout JSON → Chandra OCR layout HTML supervision targets."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

# Keep prompt identical to inference (`run_chandra_schema_layout`).
from run_chandra_schema_layout import OCR_LAYOUT_PROMPT

SECTION_HEADERS = {
    "GIẤY GỬI TIỀN TIẾT KIỆM",
    "THÔNG TIN YÊU CẦU CỦA KHÁCH HÀNG",
    "BẢNG KÊ TIỀN MẶT (CASH LIST)",
    "PHẦN DÀNH CHO NGÂN HÀNG",
}
TABLE_SECTIONS = {"Bảng kê tiền mặt", "Bảng kê ghi số"}


def chandra_label(section_name: str) -> str:
    if section_name == "Logo":
        return "Image"
    if section_name in SECTION_HEADERS:
        return "Section-Header"
    if section_name in TABLE_SECTIONS:
        return "Table"
    return "Text"


def pt_to_norm1000(
    box_pt: tuple[float, float, float, float],
    page_w_pt: float,
    page_h_pt: float,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box_pt
    return (
        max(0.0, min(1000.0, x0 / page_w_pt * 1000.0)),
        max(0.0, min(1000.0, y0 / page_h_pt * 1000.0)),
        max(0.0, min(1000.0, x1 / page_w_pt * 1000.0)),
        max(0.0, min(1000.0, y1 / page_h_pt * 1000.0)),
    )


def load_sections(
    layout_path: Path,
    page_w_pt: float,
    page_h_pt: float,
    schema_template_pt: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    """Load layout sections; bbox 0–1000 targets for SFT.

    If ``schema_template_pt`` is ``(tw, th)`` with tw, th > 0, layout JSON ``x,y,w,h``
    are interpreted in that **template** page size (PDF pt) and converted to 0–1000
    with ``tw, th`` as denominators (matches linear scale to any rendered page size).

    If ``schema_template_pt`` is None or ``(0, 0)``, denominators are ``page_w_pt``,
    ``page_h_pt`` (legacy: assume JSON coords already match the training page rect).
    """
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    tw, th = page_w_pt, page_h_pt
    if schema_template_pt is not None:
        ttw, tth = schema_template_pt
        if ttw > 0 and tth > 0:
            tw, th = ttw, tth
    for s in data.get("sections", []):
        layout = s.get("layout") or {}
        if not layout:
            continue
        name = (s.get("name") or "").strip()
        if not name:
            continue
        x = float(layout["x"])
        y = float(layout["y"])
        w = float(layout["width"])
        h = float(layout["height"])
        x0n, y0n, x1n, y1n = pt_to_norm1000((x, y, x + w, y + h), tw, th)
        rows.append({
            "name": name,
            "ord": int(s.get("ord", -1)),
            "label": chandra_label(name),
            "bbox_norm": (x0n, y0n, x1n, y1n),
        })
    rows.sort(key=lambda r: (r["ord"], r["name"]))
    return rows


def build_html_blocks(sections: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for sec in sections:
        x0, y0, x1, y1 = sec["bbox_norm"]
        bbox = f"{x0:.2f} {y0:.2f} {x1:.2f} {y1:.2f}"
        label = sec["label"]
        inner = f"<p>{html.escape(sec['name'])}</p>"
        parts.append(
            f'<div data-bbox="{bbox}" data-label="{html.escape(label, quote=True)}">{inner}</div>'
        )
    return "\n".join(parts)


def build_reasoning_vi(sections: list[dict[str, Any]]) -> str:
    lines = [
        "Mục tiêu: sinh các khối layout HTML (div + data-bbox 0–1000 + data-label) "
        "đúng prompt OCR layout, reading order theo ord schema.",
        "Ánh xạ nhãn: Logo→Image; tiêu đề/mục in hoa→Section-Header; "
        "bảng kê tiền mặt/ghi số→Table; còn lại→Text.",
        "",
        "Liệt kê từng vùng (đã chuẩn hóa theo trang PDF pt → 0–1000):",
    ]
    for sec in sections:
        x0, y0, x1, y1 = sec["bbox_norm"]
        lines.append(
            f"- {sec['name']}: nhãn Chandra `{sec['label']}`, "
            f"bbox [{x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}]."
        )
    lines.append("")
    lines.append("Kế tiếp: xuất đúng chuỗi HTML các <div> theo thứ tự trên, không thêm khối ngoài schema.")
    return "\n".join(lines)


def build_assistant_text(stage: str, sections: list[dict[str, Any]]) -> str:
    html_body = build_html_blocks(sections)
    if stage == "html_only":
        return html_body
    reasoning = build_reasoning_vi(sections)
    return f"<layout_reasoning>\n{reasoning}\n</layout_reasoning>\n\n{html_body}"


def resize_max_side(image: Any, max_side: int) -> Any:
    from PIL import Image

    w, h = image.size
    m = max(w, h)
    if m <= max_side:
        return image
    scale = max_side / float(m)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return image.resize((nw, nh), Image.LANCZOS)


def build_chandra_user_messages(image: Any, prompt_text: str | None = None) -> list[dict[str, Any]]:
    text = OCR_LAYOUT_PROMPT if prompt_text is None else prompt_text
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ],
    }]
