#!/usr/bin/env python3
"""Vẽ bbox schema lên ảnh hoặc trang PDF (khớp DPI với Chandra ~200)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from schema_viz import (
    draw_from_layout_values_json,
    draw_from_merged_html,
)

_STRTIP_QUOTES = '"\'\u201c\u201d\u2018\u2019'


def _norm_user_path(p: Path | None) -> Path | None:
    if p is None:
        return None
    s = str(p).strip().strip(_STRTIP_QUOTES)
    while len(s) >= 2 and s[0] in _STRTIP_QUOTES and s[-1] in _STRTIP_QUOTES:
        s = s[1:-1].strip()
    return Path(s)


def main() -> int:
    p = argparse.ArgumentParser(description="Visualize schema/bboxes on document image.")
    p.add_argument("--source", type=Path, required=True, help="PDF hoặc ảnh (.png/.jpg)")
    p.add_argument("--out", type=Path, required=True, help="Ảnh JPG/PNG output")
    p.add_argument("--page", type=int, default=1, help="Trang PDF (1-based)")
    p.add_argument("--dpi", type=int, default=200)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--merged-html", type=Path, help="File HTML sau Qwen (data-schema)")
    g.add_argument("--layout-values-json", type=Path, help="File *_layout_values.json")
    p.add_argument(
        "--only-schema",
        action="store_true",
        help="Chỉ vẽ div có data-schema (mặc định: với --layout-values-json là True ngầm)",
    )
    p.add_argument(
        "--all-boxes",
        action="store_true",
        help="Vẽ mọi box (ghi đè only-schema)",
    )
    args = p.parse_args()

    args.source = _norm_user_path(args.source)
    args.out = _norm_user_path(args.out)
    args.merged_html = _norm_user_path(args.merged_html)
    args.layout_values_json = _norm_user_path(args.layout_values_json)

    src = args.source.resolve()
    if not src.is_file():
        print(f"Không thấy: {src}", file=sys.stderr)
        return 1

    only_schema = args.only_schema
    if args.layout_values_json and not args.all_boxes:
        only_schema = True
    if args.all_boxes:
        only_schema = False

    if args.merged_html:
        if not args.merged_html.is_file():
            print(f"Không thấy: {args.merged_html}", file=sys.stderr)
            return 1
        html = args.merged_html.read_text(encoding="utf-8")
        draw_from_merged_html(
            src,
            html,
            args.out,
            page=args.page,
            dpi=args.dpi,
            only_with_schema=only_schema,
        )
    else:
        js = args.layout_values_json
        assert js is not None
        if not js.is_file():
            print(f"Không thấy: {js}", file=sys.stderr)
            return 1
        draw_from_layout_values_json(
            src,
            js,
            args.out,
            page=args.page,
            dpi=args.dpi,
            only_with_schema=only_schema,
        )

    print(f"Wrote {args.out.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
