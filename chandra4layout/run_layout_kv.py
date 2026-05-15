"""Layout detection for GIẤY GỬI TIỀN TIẾT KIỆM – Compact Key-Value approach.

Prompts the model to output a compact key-value format:

    Logo: 123 76 260 111
    GIẤY GỬI TIỀN TIẾT KIỆM: 371 96 626 113
    Tên khách hàng: 118 161 407 174
    ...

**Auto-fallback:** Base Chandra is strongly biased toward HTML output and will
ignore format instructions, producing HTML regardless of the KV prompt. This
runner detects the output format automatically:

  - KV lines detected  → parse_kv_output() + match_kv_to_schema()
  - HTML detected      → parse_chandra_blocks() + hybrid_match()  (fallback)

This makes the runner work correctly with both the base model (HTML output)
and a potential fine-tuned model (KV output), while clearly reporting which
format was actually used.

Outputs per page:
  <unit>_kv_raw.txt            – raw model output (whatever format)
  <unit>_kv_parsed.json        – parsed entries + detected format metadata
  <unit>_kv_schema_layout.json – schema JSON (same structure as sample)
  <unit>_kv_schema_layout.jpg  – visualisation
  <unit>_kv_compare.json       – side-by-side comparison vs HTML pipeline
run_kv_summary.json            – summary across all test files
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import uuid
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# Reuse helpers from the HTML pipeline
from run_layout_giay_gui import (
    _DEFAULT_LAYOUT_JSON,
    _ascii_fold,
    _fit_image,
    _list_inputs,
    _load_units,
    _resolve_dtype,
    build_output_schema,
    draw_chandra_boxes,
    draw_schema_layout,
    hybrid_match,
    load_schema,
    parse_chandra_blocks,
    sanitize_html,
)

_DEFAULT_PROMPT_FILE = _HERE / "prompts/giay_gui_tien_tiet_kiem_kv.txt"
_DEFAULT_INPUT_DIR = _HERE / "data/test/GIAY_GUI_TIEN_TIET_KIEM"
_DEFAULT_OUTPUT_DIR = _HERE / "results/giay_gui_tien_tiet_kiem_kv"
_DEFAULT_MODEL = "datalab-to/chandra-ocr-2"

# ---------------------------------------------------------------------------
# KV parser
# ---------------------------------------------------------------------------

# Pattern: "FieldName: x0 y0 x1 y1" with optional extra text after the 4 nums
_KV_RE = re.compile(
    r"^(?P<name>.+?)\s*:\s*(?P<x0>\d+(?:\.\d+)?)\s+(?P<y0>\d+(?:\.\d+)?)"
    r"\s+(?P<x1>\d+(?:\.\d+)?)\s+(?P<y1>\d+(?:\.\d+)?)(?:\s.*)?$",
    re.UNICODE,
)


def parse_kv_output(raw: str) -> list[dict[str, Any]]:
    """Parse compact key-value output into a list of {name, bbox_norm} dicts.

    Accepts lines like:
        Logo: 123 76 260 111
        GIẤY GỬI TIỀN TIẾT KIỆM: 371 96 626 113
        Tên khách hàng: 118 161 407 174

    Skips blank lines, comment lines (#), markdown fences, and lines that
    don't match the pattern.  Also deduplicates: first occurrence wins.
    """
    results: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("```"):
            continue
        m = _KV_RE.match(line)
        if not m:
            continue
        name = m.group("name").strip()
        try:
            x0, y0, x1, y1 = (
                float(m.group("x0")), float(m.group("y0")),
                float(m.group("x1")), float(m.group("y1")),
            )
        except ValueError:
            continue
        # Basic sanity: coords in [0, 1000] and x0<x1, y0<y1
        if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
            # Try to fix swapped coords
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
            if x0 == x1 or y0 == y1:
                continue

        folded = _ascii_fold(name)
        if folded in seen_names:
            continue
        seen_names.add(folded)
        results.append({
            "name": name,
            "name_folded": folded,
            "bbox_norm": [x0, y0, x1, y1],
        })
    return results


def match_kv_to_schema(
    schema_sections: list[dict[str, Any]],
    kv_entries: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Map parsed KV entries to schema sections by folded-name similarity.

    Each KV entry is matched to at most one schema section and vice versa.
    Uses exact fold-match first, then prefix/containment match.
    Returns {section_name: {source, bbox_norm, kv_name, score}}.
    """
    used_kv: set[int] = set()
    result: dict[str, dict[str, Any]] = {}

    # Build fold index for KV entries
    kv_folds = [e["name_folded"] for e in kv_entries]

    for sec in sorted(schema_sections, key=lambda s: s["ord"]):
        name = sec["name"]
        sf = _ascii_fold(name)

        best_j, best_sc = None, 0.0
        for j, (entry, kf) in enumerate(zip(kv_entries, kv_folds)):
            if j in used_kv:
                continue
            # Score
            if sf == kf:
                sc = 1.0
            elif sf.startswith(kf) or kf.startswith(sf):
                ratio = min(len(sf), len(kf)) / max(len(sf), len(kf))
                # Guard: apply a heavy discount when the shorter key is < 5
                # chars to prevent short names like "Ngày" (fold="ngay", 4
                # chars) from greedily consuming "Ngay mo" (fold="ngaymo").
                # 4/6 * 0.95 = 0.633 → would match (bad)
                # 4/6 * 0.65 = 0.433 → below threshold (good)
                # Exact-match (score=1.0) still takes priority above this.
                if min(len(sf), len(kf)) < 5:
                    sc = ratio * 0.65
                else:
                    sc = ratio * 0.95
            elif sf in kf or kf in sf:
                sc = min(len(sf), len(kf)) / max(len(sf), len(kf)) * 0.8
            else:
                continue
            if sc > best_sc:
                best_sc, best_j = sc, j

        if best_j is not None and best_sc >= 0.6:
            used_kv.add(best_j)
            entry = kv_entries[best_j]
            result[name] = {
                "source": "kv",
                "bbox_norm": entry["bbox_norm"],
                "kv_name": entry["name"],
                "score": round(best_sc, 4),
                "match_method": "kv-text",
            }
        else:
            result[name] = {
                "source": "template",
                "bbox_norm": None,
                "kv_name": None,
                "score": 0.0,
                "match_method": "none",
            }
    return result


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------


def _bbox_iou(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (ax1 - ax0) * (ay1 - ay0)
    ub = (bx1 - bx0) * (by1 - by0)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def compare_with_html(
    unit_name: str,
    schema_sections: list[dict[str, Any]],
    kv_matches: dict[str, dict[str, Any]],
    html_json_path: Path,
) -> dict[str, Any]:
    """Build a section-by-section comparison of KV vs HTML pipeline results."""
    html_sections: dict[str, dict[str, Any]] = {}
    if html_json_path.is_file():
        hd = json.loads(html_json_path.read_text(encoding="utf-8"))
        for s in hd.get("sections", []):
            html_sections[s["name"]] = s

    rows: list[dict[str, Any]] = []
    for sec in sorted(schema_sections, key=lambda s: s["ord"]):
        name = sec["name"]
        km = kv_matches.get(name, {})
        kv_bbox = km.get("bbox_norm")
        kv_src = km.get("source", "template")

        hs = html_sections.get(name, {})
        layout_out = hs.get("layout")
        html_bbox: list[float] | None = None
        html_src = hs.get("_bbox_source", "template")
        if layout_out and html_src == "chandra":
            # Reconstruct norm 0-1000 from pt (need page_size from parent)
            html_bbox = hs.get("_chandra_bbox_norm")

        iou_kv_html = _bbox_iou(kv_bbox, html_bbox)

        rows.append({
            "name": name,
            "ord": sec["ord"],
            "kv_source": kv_src,
            "kv_bbox": kv_bbox,
            "kv_name_raw": km.get("kv_name"),
            "kv_score": km.get("score"),
            "html_source": html_src,
            "html_bbox": html_bbox,
            "iou_kv_vs_html": round(iou_kv_html, 4),
        })

    n_kv = sum(1 for r in rows if r["kv_source"] == "kv")
    n_html = sum(1 for r in rows if r["html_source"] == "chandra")
    return {
        "unit": unit_name,
        "n_sections": len(rows),
        "kv_matched": n_kv,
        "html_matched": n_html,
        "sections": rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chandra KV layout detection for GIẤY GỬI TIỀN TIẾT KIỆM."
    )
    p.add_argument("--input-dir", type=Path, default=_DEFAULT_INPUT_DIR)
    p.add_argument("--input-file", type=Path, default=None)
    p.add_argument("--only", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    p.add_argument("--html-results-dir", type=Path,
                   default=_HERE / "results/giay_gui_tien_tiet_kiem",
                   help="Dir with HTML pipeline results to compare against.")
    p.add_argument("--model", type=str, default=_DEFAULT_MODEL)
    p.add_argument("--lora-path", type=Path, default=None)
    p.add_argument("--device-map", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=("auto", "bfloat16", "float16", "float32"))
    p.add_argument("--pdf-dpi", type=int, default=200)
    p.add_argument("--max-pixels", type=int, default=1_600_000)
    p.add_argument("--layout-json", type=Path, default=_DEFAULT_LAYOUT_JSON)
    p.add_argument("--prompt-file", type=Path, default=_DEFAULT_PROMPT_FILE)
    p.add_argument("--max-new-tokens", type=int, default=4096,
                   help="Set higher for HTML fallback output (model ignores KV prompt); "
                        "4096 matches the HTML pipeline default.")
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run() -> int:
    args = _parse_args()

    if args.input_file is not None:
        if not args.input_file.is_file():
            print(f"Input file not found: {args.input_file}", file=sys.stderr)
            return 1
        inputs_list = [args.input_file]
    else:
        inputs_list = _list_inputs(args.input_dir, args.only)
    if not inputs_list:
        print(f"No inputs found in {args.input_dir}", file=sys.stderr)
        return 1

    if not args.layout_json.is_file():
        print(f"Layout JSON not found: {args.layout_json}", file=sys.stderr)
        return 1
    schema_sections = load_schema(args.layout_json)
    print(f"[kv] {len(schema_sections)} schema sections")

    if not args.prompt_file.is_file():
        print(f"Prompt file not found: {args.prompt_file}", file=sys.stderr)
        return 1
    prompt_text = args.prompt_file.read_text(encoding="utf-8").strip()
    print(f"[kv] Prompt: {args.prompt_file.name} ({len(prompt_text)} chars)")

    # Load model
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = _resolve_dtype(args.dtype)
    print(f"[kv] Loading {args.model}  dtype={dtype}  device_map={args.device_map}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=dtype, device_map=args.device_map,
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
        print(f"[kv] Loaded LoRA from {args.lora_path}")
    processor = AutoProcessor.from_pretrained(proc_src)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "model": args.model,
        "prompt_file": str(args.prompt_file),
        "layout_json": str(args.layout_json),
        "device_map": args.device_map,
        "dtype": str(dtype),
        "items": [],
    }

    for src_path in inputs_list:
        for unit_name, image, page_pt in _load_units(src_path, args.pdf_dpi):
            page_w_pt, page_h_pt = page_pt
            img = _fit_image(image, args.max_pixels)
            img_w, img_h = img.size

            input_path = args.output_dir / f"{unit_name}_input.jpg"
            img.save(input_path, quality=92)

            # ── Inference ───────────────────────────────────────────────
            print(f"[{unit_name}] Running KV inference …")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt_text},
                ],
            }]
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
            raw_text = processor.batch_decode(
                generated[:, in_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            raw_text = sanitize_html(raw_text)

            # ── Save raw output ─────────────────────────────────────────
            raw_path = args.output_dir / f"{unit_name}_kv_raw.txt"
            raw_path.write_text(raw_text, encoding="utf-8")

            # ── Auto-detect output format ───────────────────────────────
            # Base Chandra ignores format instructions and always outputs HTML.
            # The model sometimes uses data-bbox="x y x y" (standard) and
            # sometimes data="x y x y" (abbreviated). Detect either form.
            _HTML_SIGNALS = re.compile(
                r'data(?:-bbox)?\s*=\s*"\d+\s+\d+\s+\d+\s+\d+"',
                re.IGNORECASE,
            )
            is_html_output = "<div" in raw_text and bool(_HTML_SIGNALS.search(raw_text))
            output_format = "html-fallback" if is_html_output else "kv"
            print(f"[{unit_name}] Detected output format: {output_format}")

            if is_html_output:
                # ── HTML fallback path ───────────────────────────────────
                chandra_blocks = parse_chandra_blocks(raw_text)
                print(f"[{unit_name}] HTML fallback: {len(chandra_blocks)} blocks parsed")
                kv_entries = []  # no KV entries in this mode
                parsed_path = args.output_dir / f"{unit_name}_kv_parsed.json"
                parsed_path.write_text(
                    json.dumps({"format": "html-fallback", "n_blocks": len(chandra_blocks)},
                               ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                kv_matches = hybrid_match(schema_sections, chandra_blocks, page_w_pt, page_h_pt)
                # Rename source field so downstream code can distinguish
                for v in kv_matches.values():
                    if v.get("source") == "chandra":
                        v["match_method"] = f"html-fallback/{v.get('match_method', '')}"
            else:
                # ── KV native path ───────────────────────────────────────
                kv_entries = parse_kv_output(raw_text)
                print(f"[{unit_name}] Parsed {len(kv_entries)}/{len(schema_sections)} KV entries")
                parsed_path = args.output_dir / f"{unit_name}_kv_parsed.json"
                parsed_path.write_text(
                    json.dumps({"format": "kv", "entries": kv_entries},
                               ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                kv_matches = match_kv_to_schema(schema_sections, kv_entries)

            n_matched = sum(
                1 for v in kv_matches.values()
                if v.get("source") in ("kv", "chandra")
            )
            n_tmpl = sum(1 for v in kv_matches.values() if v.get("source") == "template")
            print(f"[{unit_name}] Matched {n_matched}/{len(schema_sections)} sections "
                  f"({n_tmpl} template fallback)")

            # Print per-section match table (works for both KV and HTML-fallback)
            col2 = "KV name" if output_format == "kv" else "Match method"
            print(f"\n  {'Section':<40} {col2:<30} {'Score':>6}")
            print(f"  {'-'*40} {'-'*30} {'-'*6}")
            for sec in sorted(schema_sections, key=lambda s: s["ord"]):
                m = kv_matches[sec["name"]]
                is_tmpl = m.get("source") == "template"
                flag = " ✗" if is_tmpl else ""
                if output_format == "kv":
                    detail = m.get("kv_name") or "—"
                    sc = m.get("score", 0.0)
                else:
                    detail = m.get("match_method", "—") if not is_tmpl else "—"
                    sc = m.get("_match_score", 0.0)
                    if sc > 100:  # IoU area scores – normalize display
                        sc = 0.0
                print(f"  {sec['name']:<40} {detail:<30} {sc:>6.3f}{flag}")
            print()

            # ── Build schema JSON ────────────────────────────────────────
            out_schema = build_output_schema(
                schema_sections, kv_matches, page_w_pt, page_h_pt, page_num=1
            )
            schema_json: dict[str, Any] = {
                "source": str(src_path),
                "unit": unit_name,
                "image_size": [img_w, img_h],
                "page_size_pt": [page_w_pt, page_h_pt],
                "approach": "kv",
                "output_format_detected": output_format,
                "prompt_file": str(args.prompt_file),
                "raw_output_path": str(raw_path),
                "n_kv_entries": len(kv_entries),
                "n_sections_matched": n_matched,
                "n_sections_template": n_tmpl,
                "sections": out_schema["sections"],
            }
            schema_path = args.output_dir / f"{unit_name}_kv_schema_layout.json"
            schema_path.write_text(
                json.dumps(schema_json, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # ── Visualize ────────────────────────────────────────────────
            viz_entries: list[dict[str, Any]] = []
            sx, sy = img_w / page_w_pt, img_h / page_h_pt
            for sec in sorted(schema_sections, key=lambda s: s["ord"]):
                layout_out = next(
                    (s["layout"] for s in out_schema["sections"] if s["name"] == sec["name"]),
                    None,
                )
                if not layout_out:
                    continue
                src_tag = kv_matches.get(sec["name"], {}).get("source", "template")
                viz_entries.append({
                    "label": sec["name"],
                    "source": src_tag,
                    "box_xyxy": [
                        layout_out["x"] * sx,
                        layout_out["y"] * sy,
                        (layout_out["x"] + layout_out["width"]) * sx,
                        (layout_out["y"] + layout_out["height"]) * sy,
                    ],
                })

            viz_path = args.output_dir / f"{unit_name}_kv_schema_layout.jpg"
            draw_schema_layout(img, viz_entries, viz_path)

            # ── Compare vs HTML pipeline ─────────────────────────────────
            html_json = (
                args.html_results_dir / f"{unit_name}_schema_layout.json"
                if args.html_results_dir.is_dir() else Path("/dev/null")
            )
            cmp = compare_with_html(unit_name, schema_sections, kv_matches, html_json)
            cmp_path = args.output_dir / f"{unit_name}_kv_compare.json"
            cmp_path.write_text(
                json.dumps(cmp, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[{unit_name}] KV matched={cmp['kv_matched']} / "
                  f"HTML matched={cmp['html_matched']} → compare: {cmp_path.name}")

            item: dict[str, Any] = {
                "source": str(src_path),
                "unit": unit_name,
                "image_size": [img_w, img_h],
                "output_format_detected": output_format,
                "viz": str(viz_path),
                "schema_json": str(schema_path),
                "raw_output": str(raw_path),
                "n_kv_entries_parsed": len(kv_entries),
                "n_sections_matched": n_matched,
                "n_sections_template": n_tmpl,
                "compare_json": str(cmp_path),
            }
            summary["items"].append(item)
            print(f"[ok] {src_path.name} :: {unit_name} → {viz_path.name}")

    summary_path = args.output_dir / "run_kv_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] Summary → {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
