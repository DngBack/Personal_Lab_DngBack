"""Overlay the predefined schema layout boxes onto test PDFs / images.

Boxes come from a section-level layout JSON (e.g.
``prompt4ocr/data/samples/GIAY_GUI_TIEN_TIET_KIEM/layout _GIAY_GUI_TIEN_TIET_KIEM.json``).
Each section's ``layout`` is given in **PDF point space** of the original template
(595×842 for A4 portrait). When rendering on a real scan, we rescale to the actual
image pixel size using the PDF's ``page.rect`` (PyMuPDF) or assumed A4 if input is
an image.

Optional: also load a MinerU run output JSON; each MinerU block is assigned to the
schema section whose box covers it best, and that section's annotation gets the
combined recognised content. This gives you the same big "section" boxes as the
sample image, but filled with MinerU text/table content.
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

from doc_utils import IMAGE_EXTENSIONS, PDF_EXTENSIONS, draw_boxes, list_inputs

DEFAULT_LAYOUT = (
    PROJECT_DIR
    / "data/samples/GIAY_GUI_TIEN_TIET_KIEM/layout _GIAY_GUI_TIEN_TIET_KIEM.json"
)
DEFAULT_REF_W_PT = 595.0
DEFAULT_REF_H_PT = 842.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overlay schema layout boxes on test docs.")
    default_in = PROJECT_DIR / "data/test/GIAY_GUI_TIEN_TIET_KIEM"
    default_out = PROJECT_DIR / "results/schema_layout"
    p.add_argument("--layout-json", type=Path, default=DEFAULT_LAYOUT, help="Layout JSON path.")
    p.add_argument("--input-dir", type=Path, default=default_in, help="Folder with PDFs/images.")
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
        help="Substring filter applied to filenames inside --input-dir.",
    )
    p.add_argument("--output-dir", type=Path, default=default_out, help="Folder to write JPG+JSON.")
    p.add_argument("--pdf-dpi", type=int, default=220, help="Render DPI for PDFs.")
    p.add_argument(
        "--mineru-json",
        type=Path,
        default=None,
        help="Optional MinerU per-page JSON to merge into schema sections (single page).",
    )
    p.add_argument(
        "--mineru-dir",
        type=Path,
        default=None,
        help="Folder containing *_mineru.json files; auto-matched to input filenames.",
    )
    p.add_argument(
        "--min-area",
        type=float,
        default=0.0,
        help="Drop sections whose pixel area is below this (handy to hide tiny artifacts).",
    )
    p.add_argument(
        "--offset-x",
        type=float,
        default=0.0,
        help="Shift every section box right by this many PDF points (pt). Negative shifts left.",
    )
    p.add_argument(
        "--offset-y",
        type=float,
        default=0.0,
        help="Shift every section box down by this many PDF points (pt). Negative shifts up.",
    )
    p.add_argument(
        "--mineru-cover",
        type=float,
        default=0.4,
        help="Min cover ratio (block area inside section) to assign a MinerU block.",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="schema",
        choices=("schema", "mineru-named"),
        help=(
            "'schema' = draw schema layout boxes as-is (positions may be off for non-template scans). "
            "'mineru-named' = draw MinerU's accurate boxes, but labelled by the schema section "
            "they fall into."
        ),
    )
    p.add_argument(
        "--auto-align",
        action="store_true",
        help=(
            "Auto-derive offset_x/y by matching MinerU title/header content to schema section "
            "names. Requires --mineru-json or --mineru-dir."
        ),
    )
    p.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Scale label font size in the overlay.",
    )
    return p.parse_args()


def load_layout(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: list[dict[str, Any]] = []
    for s in data.get("sections", []):
        layout = s.get("layout")
        if not layout:
            continue
        out.append(
            {
                "name": s.get("name", ""),
                "page": int(layout.get("page", 1)),
                "x": float(layout["x"]),
                "y": float(layout["y"]),
                "w": float(layout["width"]),
                "h": float(layout["height"]),
                "ord": s.get("ord"),
                "id": s.get("id"),
            }
        )
    return out


def render_pdf_page_with_rect(pdf_path: Path, page_idx: int, dpi: int) -> tuple[Any, tuple[float, float]]:
    """Return (PIL.Image, (page_w_pt, page_h_pt)) for a single PDF page."""
    import fitz
    from PIL import Image

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(pdf_path) as doc:
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        mode = "RGB" if pix.n < 4 else "RGBA"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples).convert("RGB")
        return img, (page.rect.width, page.rect.height)


def iter_units(
    path: Path, pdf_dpi: int
) -> list[tuple[str, Any, tuple[float, float], int]]:
    """Yield (unit_name, image, (page_w_pt, page_h_pt), page_idx_1based) per page."""
    from PIL import Image

    suffix = path.suffix.lower()
    if suffix in PDF_EXTENSIONS:
        import fitz

        with fitz.open(path) as doc:
            n = doc.page_count
        units: list[tuple[str, Any, tuple[float, float], int]] = []
        for i in range(n):
            img, rect = render_pdf_page_with_rect(path, i, pdf_dpi)
            units.append((f"{path.stem}_page{i + 1:02d}", img, rect, i + 1))
        return units
    img = Image.open(path).convert("RGB")
    return [(path.stem, img, (DEFAULT_REF_W_PT, DEFAULT_REF_H_PT), 1)]


def section_to_entry(
    sec: dict[str, Any],
    img_w_px: int,
    img_h_px: int,
    page_w_pt: float,
    page_h_pt: float,
    offset_x_pt: float = 0.0,
    offset_y_pt: float = 0.0,
) -> dict[str, Any]:
    sx = img_w_px / page_w_pt
    sy = img_h_px / page_h_pt
    x0 = (sec["x"] + offset_x_pt) * sx
    y0 = (sec["y"] + offset_y_pt) * sy
    x1 = (sec["x"] + sec["w"] + offset_x_pt) * sx
    y1 = (sec["y"] + sec["h"] + offset_y_pt) * sy
    return {
        "label": sec["name"],
        "score": None,
        "box_xyxy": [x0, y0, x1, y1],
        "ord": sec["ord"],
        "id": sec["id"],
        "page": sec["page"],
    }


def load_mineru_blocks(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("blocks", [])


def _norm_text(s: str | None) -> str:
    if not s:
        return ""
    return "".join(ch for ch in s.lower() if ch.isalnum())


def compute_auto_offset_pt(
    sections: list[dict[str, Any]],
    mineru_blocks: list[dict[str, Any]],
    page_w_pt: float,
    page_h_pt: float,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Match MinerU title/header blocks to schema sections by name; return median
    (dx, dy) in PDF points such that schema_box + (dx, dy) ≈ mineru_box.

    Returns (dx_pt, dy_pt, anchors) where ``anchors`` lists the chosen pairs for
    transparency.
    """
    anchor_types = {"title", "header"}
    candidates = [
        b
        for b in mineru_blocks
        if (b.get("type") in anchor_types) and b.get("content") and b.get("bbox_normalized")
    ]
    pairs: list[tuple[float, float, dict[str, Any]]] = []
    for sec in sections:
        sec_key = _norm_text(sec["name"])
        if len(sec_key) < 4:
            continue
        best: tuple[float, dict[str, Any]] | None = None
        for b in candidates:
            bkey = _norm_text(b["content"])
            if not bkey or len(bkey) < 4:
                continue
            if sec_key in bkey or bkey in sec_key:
                score = min(len(sec_key), len(bkey)) / max(len(sec_key), len(bkey))
                if best is None or score > best[0]:
                    best = (score, b)
        if best is None:
            continue
        b = best[1]
        bx0, by0, bx1, by1 = b["bbox_normalized"]
        mineru_cx_pt = (bx0 + bx1) / 2 * page_w_pt
        mineru_cy_pt = (by0 + by1) / 2 * page_h_pt
        schema_cx_pt = sec["x"] + sec["w"] / 2
        schema_cy_pt = sec["y"] + sec["h"] / 2
        pairs.append(
            (
                mineru_cx_pt - schema_cx_pt,
                mineru_cy_pt - schema_cy_pt,
                {"schema": sec["name"], "mineru": b["content"], "score": round(best[0], 3)},
            )
        )

    if not pairs:
        return 0.0, 0.0, []

    dxs = sorted(p[0] for p in pairs)
    dys = sorted(p[1] for p in pairs)
    mid = len(dxs) // 2
    dx_med = dxs[mid] if len(dxs) % 2 else 0.5 * (dxs[mid - 1] + dxs[mid])
    dy_med = dys[mid] if len(dys) % 2 else 0.5 * (dys[mid - 1] + dys[mid])
    return dx_med, dy_med, [p[2] for p in pairs]


def auto_pick_mineru_json(mineru_dir: Path | None, unit_name: str) -> Path | None:
    if not mineru_dir or not mineru_dir.is_dir():
        return None
    cand = mineru_dir / f"{unit_name}_mineru.json"
    return cand if cand.is_file() else None


def _bbox_iou_to_section(block_bbox_norm: list[float], section_px: list[float], img_size: tuple[int, int]) -> float:
    """Cover ratio = intersection / block_area, using a normalised MinerU bbox
    against a section bbox already in pixel coords."""
    w, h = img_size
    bx0, by0, bx1, by1 = (
        block_bbox_norm[0] * w,
        block_bbox_norm[1] * h,
        block_bbox_norm[2] * w,
        block_bbox_norm[3] * h,
    )
    sx0, sy0, sx1, sy1 = section_px
    ix0, iy0 = max(bx0, sx0), max(by0, sy0)
    ix1, iy1 = min(bx1, sx1), min(by1, sy1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    b_area = max(1e-6, (bx1 - bx0) * (by1 - by0))
    return inter / b_area


def assign_mineru_to_sections(
    entries: list[dict[str, Any]],
    mineru_blocks: list[dict[str, Any]],
    img_size: tuple[int, int],
    min_cover: float = 0.5,
) -> dict[int, list[dict[str, Any]]]:
    """For each section index, list MinerU blocks whose cover ratio >= min_cover.

    Each block in the returned dict is the original MinerU block enriched with
    ``cover`` (so callers can also read ``bbox_normalized`` / pixel-convert it).
    """
    by_section: dict[int, list[dict[str, Any]]] = {}
    for b in mineru_blocks:
        bbox = b.get("bbox_normalized")
        if not bbox or len(bbox) != 4:
            continue
        best_idx = -1
        best_score = 0.0
        for i, sec in enumerate(entries):
            cov = _bbox_iou_to_section(bbox, sec["box_xyxy"], img_size)
            if cov > best_score:
                best_score = cov
                best_idx = i
        if best_idx >= 0 and best_score >= min_cover:
            by_section.setdefault(best_idx, []).append(
                {
                    "type": b.get("type"),
                    "content": b.get("content"),
                    "bbox_normalized": bbox,
                    "cover": round(best_score, 3),
                }
            )
    return by_section


def build_mineru_named_entries(
    sections: list[dict[str, Any]],
    mineru_blocks: list[dict[str, Any]],
    img_w_px: int,
    img_h_px: int,
    min_cover: float,
) -> list[dict[str, Any]]:
    """Use MinerU's bboxes (accurate) but label each by the schema section it best
    matches. Unmatched MinerU blocks are kept with a generic "(unmatched: type)" label.
    """
    out: list[dict[str, Any]] = []
    for b in mineru_blocks:
        bbox = b.get("bbox_normalized")
        if not bbox or len(bbox) != 4:
            continue
        block_px = [
            bbox[0] * img_w_px,
            bbox[1] * img_h_px,
            bbox[2] * img_w_px,
            bbox[3] * img_h_px,
        ]
        best_name: str | None = None
        best_cov = 0.0
        for sec in sections:
            cov = _bbox_iou_to_section(bbox, sec["box_xyxy"], (img_w_px, img_h_px))
            if cov > best_cov:
                best_cov = cov
                best_name = sec["label"]
        if best_name is None or best_cov < min_cover:
            label = f"({b.get('type', 'unknown')})"
        else:
            label = best_name
        out.append(
            {
                "label": label,
                "score": round(best_cov, 2) if best_cov > 0 else None,
                "box_xyxy": block_px,
                "mineru_type": b.get("type"),
                "content": b.get("content"),
            }
        )
    return out


def run() -> int:
    args = parse_args()

    if not args.layout_json.is_file():
        print(f"layout JSON not found: {args.layout_json}", file=sys.stderr)
        return 1

    sections = load_layout(args.layout_json)
    print(f"[layout] {len(sections)} sections from {args.layout_json.name}")

    if args.input_file is not None:
        if not args.input_file.is_file():
            print(f"input file not found: {args.input_file}", file=sys.stderr)
            return 1
        inputs_list = [args.input_file]
    else:
        inputs_list = list_inputs(args.input_dir, args.only)

    if not inputs_list:
        print(
            f"No inputs found in {args.input_dir} (images: {sorted(IMAGE_EXTENSIONS)}, "
            f"pdfs: {sorted(PDF_EXTENSIONS)}).",
            file=sys.stderr,
        )
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "layout_json": str(args.layout_json),
        "pdf_dpi": args.pdf_dpi,
        "items": [],
    }

    for src_path in inputs_list:
        units = iter_units(src_path, args.pdf_dpi)
        for unit_name, image, page_pt, page_idx in units:
            sections_for_page = [s for s in sections if s["page"] == page_idx]

            mineru_json_path_for_align = args.mineru_json
            if mineru_json_path_for_align is None and args.mineru_dir is not None:
                mineru_json_path_for_align = auto_pick_mineru_json(args.mineru_dir, unit_name)

            off_x = args.offset_x
            off_y = args.offset_y
            anchors: list[dict[str, Any]] = []
            if args.auto_align:
                pre_blocks = load_mineru_blocks(mineru_json_path_for_align)
                if not pre_blocks:
                    print(
                        "--auto-align needs MinerU JSON; pass --mineru-json or --mineru-dir.",
                        file=sys.stderr,
                    )
                    return 2
                dx_pt, dy_pt, anchors = compute_auto_offset_pt(
                    sections_for_page, pre_blocks, page_pt[0], page_pt[1]
                )
                off_x += dx_pt
                off_y += dy_pt
                print(
                    f"[auto-align] {unit_name}: dx={dx_pt:+.2f}pt dy={dy_pt:+.2f}pt "
                    f"(from {len(anchors)} anchor(s))"
                )

            entries = [
                section_to_entry(
                    s,
                    image.width,
                    image.height,
                    page_pt[0],
                    page_pt[1],
                    off_x,
                    off_y,
                )
                for s in sections_for_page
            ]
            if args.min_area > 0:
                entries = [
                    e
                    for e in entries
                    if (e["box_xyxy"][2] - e["box_xyxy"][0])
                    * (e["box_xyxy"][3] - e["box_xyxy"][1])
                    >= args.min_area
                ]

            mineru_json_path = args.mineru_json
            if mineru_json_path is None and args.mineru_dir is not None:
                mineru_json_path = auto_pick_mineru_json(args.mineru_dir, unit_name)
            mineru_blocks = load_mineru_blocks(mineru_json_path)

            if args.mode == "mineru-named":
                if not mineru_blocks:
                    print(
                        "mode=mineru-named needs --mineru-json or --mineru-dir; "
                        "no MinerU JSON found for this unit.",
                        file=sys.stderr,
                    )
                    return 2
                draw_entries = build_mineru_named_entries(
                    entries,
                    mineru_blocks,
                    image.width,
                    image.height,
                    args.mineru_cover,
                )
                suffix = "mineru_named"
                num_label = "blocks"
                num_value = len(draw_entries)
            else:
                draw_entries = entries
                suffix = "schema_boxes"
                num_label = "sections"
                num_value = len(entries)

            assigned = (
                assign_mineru_to_sections(
                    entries,
                    mineru_blocks,
                    (image.width, image.height),
                    min_cover=args.mineru_cover,
                )
                if mineru_blocks
                else {}
            )

            viz_path = args.output_dir / f"{unit_name}_{suffix}.jpg"
            json_path = args.output_dir / f"{unit_name}_{suffix}.json"

            draw_boxes(
                image,
                draw_entries,
                viz_path,
                font_scale=args.font_scale,
                color_by="label",
            )

            if args.mode == "mineru-named":
                jsonable = [
                    {
                        "label": e["label"],
                        "cover": e["score"],
                        "mineru_type": e.get("mineru_type"),
                        "content": e.get("content"),
                        "box_xyxy_px": [round(v, 2) for v in e["box_xyxy"]],
                    }
                    for e in draw_entries
                ]
                payload = {
                    "source": str(src_path),
                    "unit": unit_name,
                    "page": page_idx,
                    "image_size_hw": [image.height, image.width],
                    "mineru_json": str(mineru_json_path) if mineru_json_path else None,
                    "blocks": jsonable,
                }
            else:
                jsonable = [
                    {
                        "name": e["label"],
                        "ord": e["ord"],
                        "id": e["id"],
                        "page": e["page"],
                        "box_xyxy_px": [round(v, 2) for v in e["box_xyxy"]],
                        "mineru_assignments": assigned.get(i, []),
                    }
                    for i, e in enumerate(entries)
                ]
                payload = {
                    "source": str(src_path),
                    "unit": unit_name,
                    "page": page_idx,
                    "image_size_hw": [image.height, image.width],
                    "page_size_pt": [page_pt[0], page_pt[1]],
                    "mineru_json": str(mineru_json_path) if mineru_json_path else None,
                    "sections": jsonable,
                }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            summary["items"].append(
                {
                    "source": str(src_path),
                    "unit": unit_name,
                    "page": page_idx,
                    "mode": args.mode,
                    "visualization": str(viz_path),
                    "json": str(json_path),
                    "num_entries": num_value,
                }
            )
            print(
                f"[ok] {src_path.name} :: {unit_name} -> {num_value} {num_label} -> {viz_path.name}"
            )

    summary_path = args.output_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
