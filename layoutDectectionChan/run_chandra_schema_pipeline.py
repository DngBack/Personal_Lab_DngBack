#!/usr/bin/env python3
"""
Chandra HTML → LLM gắn schema (+ tuỳ chọn ảnh cùng trang) → HTML → JSON.

Thiết kế: chất lượng đến từ prompt + model; code chỉ vận chuyển dữ liệu, parse HTML tối thiểu,
khớp schema với layout theo đúng chuỗi UTF-8 (không chuẩn hoá tên).

Mặc định khi --schema-merge-with-image: đính ảnh mẫu bbox trong repo làm visual reference (tắt bằng --no-schema-reference-image).

Ví dụ:
  python run_chandra_schema_pipeline.py \\
    --chandra-log data/test/GIAY_GUI_TIEN_TIET_KIEM/test_7_llm.log \\
    --layout-json data/samples/GIAY_GUI_TIEN_TIET_KIEM/layout_GIAY_GUI_TIET_KIEM_MERGED.json \\
    --out-dir data/test/GIAY_GUI_TIEN_TIET_KIEM/out_schema \\
    --viz-source data/test/GIAY_GUI_TIEN_TIET_KIEM/test_7.pdf \\
    --schema-merge-with-image \\
    --qwen-vl-model Qwen/Qwen3-VL-4B-Instruct \\
    --viz-only-schema
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DEFAULT_SCHEMA_REFERENCE_IMAGE = (
    _ROOT
    / "data/samples/GIAY_GUI_TIEN_TIET_KIEM"
    / "20251018+MYNTT1_0001-p18_page01_schema_boxes.jpg"
)

from layout_json_utils import load_layout_names
from schema_html_parse import (
    extract_chandra_html_from_log,
    layout_with_extractions_to_json,
)
from schema_merge_qwen import QwenSchemaHtmlMerger, load_system_prompt
from schema_merge_qwen_vl import QwenVLSchemaHtmlMerger
from schema_viz import draw_from_merged_html, load_page_image

# Copy/paste từ web đôi khi dính dấu “ ” thay vì " — strip để không lỗi path.
_STRTIP_QUOTES = '"\'\u201c\u201d\u2018\u2019'


def _norm_user_path(p: Path | None) -> Path | None:
    if p is None:
        return None
    s = str(p).strip().strip(_STRTIP_QUOTES)
    while len(s) >= 2 and s[0] in _STRTIP_QUOTES and s[-1] in _STRTIP_QUOTES:
        s = s[1:-1].strip()
    return Path(s)


def main() -> int:
    p = argparse.ArgumentParser(description="Chandra HTML + Qwen schema merge + JSON.")
    p.add_argument(
        "--chandra-log",
        type=Path,
        default=None,
        help="File .log từ run_pdf_llm_log (chứa <div> Chandra).",
    )
    p.add_argument(
        "--chandra-html",
        type=Path,
        default=None,
        help="Hoặc file .html/.txt chỉ chứa HTML Chandra.",
    )
    p.add_argument(
        "--layout-json",
        type=Path,
        required=True,
        help="Layout gốc (VD layout _GIAY_….json hoặc MERGED).",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--system-prompt",
        type=Path,
        default=_ROOT / "prompt" / "schema_merge_html_system.txt",
    )
    p.add_argument(
        "--qwen-model",
        type=str,
        default="Qwen/Qwen3.5-4B",
        help="Qwen 3.5 4B; nếu lỗi thử Qwen/Qwen3-4B-Instruct-2507.",
    )
    p.add_argument("--qwen-device", type=str, default="cuda:0")
    p.add_argument("--qwen-dtype", type=str, default="bfloat16",
                   choices=("bfloat16", "float16", "float32"))
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--viz-source",
        type=Path,
        default=None,
        help="PDF hoặc ảnh để vẽ bbox (nên cùng DPI với Chandra). Bật visualization.",
    )
    p.add_argument("--viz-page", type=int, default=1)
    p.add_argument("--viz-dpi", type=int, default=200)
    p.add_argument(
        "--viz-only-schema",
        action="store_true",
        help="Chỉ vẽ box có data-schema",
    )
    p.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Tiền tố tên file đầu ra (mặc định: stem của --chandra-log / --chandra-html, VD test_7_llm.log → test_7_llm).",
    )
    p.add_argument(
        "--schema-merge-with-image",
        action="store_true",
        help="Dùng Qwen2.5-VL: đưa ảnh trang (--viz-source, cùng --viz-dpi/--viz-page) + HTML; bbox vùng ký tốt hơn. Cần: pip install qwen-vl-utils",
    )
    p.add_argument(
        "--qwen-vl-model",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="Chỉ checkpoint VL: Qwen3-VL-4B-Instruct hoặc Qwen2.5-VL-3B-Instruct. KHÔNG dùng Qwen/Qwen3.5-4B (text-only).",
    )
    p.add_argument(
        "--schema-reference-image",
        type=Path,
        default=None,
        help=f"Ảnh visual mẫu (bbox) gửi kèm VL trước ảnh trang. Mặc định: {DEFAULT_SCHEMA_REFERENCE_IMAGE.name} trong sample nếu có.",
    )
    p.add_argument(
        "--no-schema-reference-image",
        action="store_true",
        help="Không đính ảnh mẫu (chỉ ảnh trang đang xử lý).",
    )
    p.add_argument(
        "--schema-vl-max-pixels",
        type=int,
        default=0,
        help="0 = không resize ảnh VL (khớp bbox 0–1000 với Chandra). VD 1600000 nếu hết VRAM.",
    )
    args = p.parse_args()

    args.chandra_log = _norm_user_path(args.chandra_log)
    args.chandra_html = _norm_user_path(args.chandra_html)
    args.layout_json = _norm_user_path(args.layout_json)
    args.out_dir = _norm_user_path(args.out_dir)
    args.system_prompt = _norm_user_path(args.system_prompt)
    args.viz_source = _norm_user_path(args.viz_source)
    args.schema_reference_image = _norm_user_path(args.schema_reference_image)

    if bool(args.chandra_log) == bool(args.chandra_html):
        print("Cần đúng một trong: --chandra-log HOẶC --chandra-html", file=sys.stderr)
        return 1

    layout_path = args.layout_json.resolve()
    if not layout_path.is_file():
        print(f"Không thấy layout: {layout_path}", file=sys.stderr)
        return 1

    if args.chandra_log:
        raw = args.chandra_log.read_text(encoding="utf-8")
        chandra_html = extract_chandra_html_from_log(raw)
    else:
        chandra_html = args.chandra_html.read_text(encoding="utf-8")

    if "<div" not in chandra_html.lower():
        print("Không tìm thấy <div> trong nguồn Chandra.", file=sys.stderr)
        return 1

    if not args.system_prompt.is_file():
        print(f"Thiếu system prompt: {args.system_prompt}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem_raw = args.stem.strip() if args.stem else ""
    if stem_raw:
        s = stem_raw.strip().strip(_STRTIP_QUOTES)
        while len(s) >= 2 and s[0] in _STRTIP_QUOTES and s[-1] in _STRTIP_QUOTES:
            s = s[1:-1].strip()
        stem = s
    else:
        stem = (args.chandra_log or args.chandra_html).stem

    (args.out_dir / f"{stem}_chandra_strip.html").write_text(chandra_html, encoding="utf-8")

    schema_fields = load_layout_names(layout_path)
    system_txt = load_system_prompt(args.system_prompt)

    ref_path_logged: str | None = None
    use_vl = args.schema_merge_with_image
    if use_vl:
        if args.viz_source is None or not args.viz_source.resolve().is_file():
            print(
                "--schema-merge-with-image cần --viz-source (PDF/ảnh đúng trang đã raster cho Chandra, "
                "cùng --viz-dpi / --viz-page).",
                file=sys.stderr,
            )
            return 1
        page_img = load_page_image(
            args.viz_source, page=args.viz_page, dpi=args.viz_dpi,
        )
        ref_img = None
        ref_path_logged = None
        if not args.no_schema_reference_image:
            ref_path = args.schema_reference_image
            if ref_path is None:
                ref_path = DEFAULT_SCHEMA_REFERENCE_IMAGE
            ref_path = ref_path.resolve()
            if ref_path.is_file():
                from PIL import Image

                ref_img = Image.open(ref_path).convert("RGB")
                ref_path_logged = str(ref_path)
                print(f"Kèm ảnh visual mẫu: {ref_path.name}", flush=True)
            else:
                print(
                    f"[warn] Không tìm thấy ảnh mẫu VL ({ref_path}) — chạy không kèm reference.",
                    flush=True,
                )
        print(
            f"Qwen-VL: {args.qwen_vl_model}  (schema: {len(schema_fields)}, ảnh trang + reference: {ref_img is not None})",
            flush=True,
        )
        merger_vl = QwenVLSchemaHtmlMerger(
            args.qwen_vl_model,
            device_map=args.qwen_device,
        )
        try:
            merged_html = merger_vl.merge_to_html(
                system_prompt=system_txt,
                schema_fields=schema_fields,
                chandra_html=chandra_html,
                page_image=page_img,
                reference_image=ref_img,
                max_pixels=args.schema_vl_max_pixels,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        finally:
            merger_vl.cleanup()
        qwen_meta_model = args.qwen_vl_model
        schema_merge_mode = "vision"
    else:
        print(f"Qwen: {args.qwen_model}  (schema fields: {len(schema_fields)})", flush=True)
        merger = QwenSchemaHtmlMerger(
            args.qwen_model, device=args.qwen_device, dtype=args.qwen_dtype,
        )
        try:
            merged_html = merger.merge_to_html(
                system_prompt=system_txt,
                schema_fields=schema_fields,
                chandra_html=chandra_html,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        finally:
            merger.cleanup()
        qwen_meta_model = args.qwen_model
        schema_merge_mode = "text"

    if "<div" not in merged_html.lower():
        print("Cảnh báo: đầu ra Qwen không chứa <div> — vẫn ghi file để debug.", file=sys.stderr)

    merged_path = args.out_dir / f"{stem}_schema_merged.html"
    merged_path.write_text(merged_html, encoding="utf-8")
    print(f"Wrote {merged_path}", flush=True)

    payload = layout_with_extractions_to_json(str(layout_path), merged_html)
    payload["meta"] = {
        "chandra_source": str(args.chandra_log or args.chandra_html),
        "layout_json": str(layout_path),
        "qwen_model": qwen_meta_model,
        "schema_merge_mode": schema_merge_mode,
        "schema_reference_image": ref_path_logged,
    }
    json_path = args.out_dir / f"{stem}_layout_values.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}", flush=True)

    if args.viz_source is not None:
        vsrc = args.viz_source.resolve()
        if not vsrc.is_file():
            print(f"--viz-source không tồn tại: {vsrc}", file=sys.stderr)
            return 1
        viz_out = args.out_dir / f"{stem}_schema_boxes.jpg"
        draw_from_merged_html(
            vsrc,
            merged_html,
            viz_out,
            page=args.viz_page,
            dpi=args.viz_dpi,
            only_with_schema=args.viz_only_schema,
        )
        print(f"Wrote {viz_out}", flush=True)

    print(
        f"[out] tiền tố file: {stem} → {stem}_chandra_strip.html, {stem}_schema_merged.html, "
        f"{stem}_layout_values.json, {stem}_schema_boxes.jpg (nếu --viz-source)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
