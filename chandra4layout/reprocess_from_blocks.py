"""Re-run matching + visualization from already-cached Chandra blocks.

Useful for quickly testing matching changes without re-running the model.

Usage:
    python chandra4layout/reprocess_from_blocks.py
    python chandra4layout/reprocess_from_blocks.py --results-dir chandra4layout/results/giay_gui_tien_tiet_kiem
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from run_layout_giay_gui import (
    _DEFAULT_LAYOUT_JSON,
    build_output_schema,
    draw_chandra_boxes,
    draw_schema_layout,
    hybrid_match,
    load_schema,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reprocess cached Chandra blocks with updated matching logic.",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=_HERE / "results/giay_gui_tien_tiet_kiem",
    )
    p.add_argument("--layout-json", type=Path, default=_DEFAULT_LAYOUT_JSON)
    p.add_argument("--match-iou-min", type=float, default=0.05)
    p.add_argument("--only", type=str, default=None,
                   help="Only reprocess units whose name contains this substring.")
    return p.parse_args()


def run() -> int:
    from PIL import Image

    args = parse_args()

    if not args.results_dir.is_dir():
        print(f"Results dir not found: {args.results_dir}", file=sys.stderr)
        return 1
    if not args.layout_json.is_file():
        print(f"Layout JSON not found: {args.layout_json}", file=sys.stderr)
        return 1

    schema_sections = load_schema(args.layout_json)
    print(f"[reprocess] {len(schema_sections)} schema sections")

    block_files = sorted(args.results_dir.glob("*_chandra_blocks.json"))
    if args.only:
        block_files = [f for f in block_files if args.only in f.name]
    if not block_files:
        print("No *_chandra_blocks.json files found.", file=sys.stderr)
        return 1

    for bf in block_files:
        unit_name = bf.name.replace("_chandra_blocks.json", "")
        input_img_path = args.results_dir / f"{unit_name}_input.jpg"
        if not input_img_path.is_file():
            print(f"[skip] {unit_name}: no input image at {input_img_path}")
            continue

        blocks = json.loads(bf.read_text(encoding="utf-8"))
        img = Image.open(input_img_path).convert("RGB")
        img_w, img_h = img.size

        # Try to get page_size from existing schema_layout.json
        old_json_path = args.results_dir / f"{unit_name}_schema_layout.json"
        page_w_pt, page_h_pt = 595.0, 844.0
        if old_json_path.is_file():
            try:
                od = json.loads(old_json_path.read_text(encoding="utf-8"))
                page_w_pt, page_h_pt = od.get("page_size_pt", [595.0, 844.0])
            except Exception:
                pass

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
        print(f"[{unit_name}] {n_ch}/{len(schema_sections)} chandra, "
              f"{n_tmpl} template  [{method_str}]")

        out_schema = build_output_schema(
            schema_sections, matches, page_w_pt, page_h_pt, page_num=1
        )

        # Update existing schema_layout.json
        if old_json_path.is_file():
            od = json.loads(old_json_path.read_text(encoding="utf-8"))
        else:
            od = {}
        od["sections"] = out_schema["sections"]
        od["n_sections_chandra"] = n_ch
        od["n_sections_template"] = n_tmpl
        od["match_by_method"] = by_method
        old_json_path.write_text(json.dumps(od, ensure_ascii=False, indent=2), encoding="utf-8")

        # Regenerate schema layout viz
        viz_entries = []
        sx = img_w / page_w_pt
        sy = img_h / page_h_pt
        for sec in sorted(schema_sections, key=lambda s: s["ord"]):
            layout_out = next(
                (s["layout"] for s in out_schema["sections"] if s["name"] == sec["name"]), None
            )
            match_src = matches.get(sec["name"], {}).get("source", "template")
            if not layout_out:
                continue
            viz_entries.append({
                "label": sec["name"],
                "source": match_src,
                "box_xyxy": [
                    layout_out["x"] * sx,
                    layout_out["y"] * sy,
                    (layout_out["x"] + layout_out["width"]) * sx,
                    (layout_out["y"] + layout_out["height"]) * sy,
                ],
            })

        viz_path = args.results_dir / f"{unit_name}_schema_layout.jpg"
        draw_schema_layout(img, viz_entries, viz_path)
        print(f"  → {viz_path.name}")

        # Regenerate chandra boxes viz too (if it existed)
        cb_path = args.results_dir / f"{unit_name}_chandra_boxes.jpg"
        if blocks:
            draw_chandra_boxes(img, blocks, cb_path)

    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
