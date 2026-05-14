"""Chandra OCR 2: **tên section / ord** từ ``--layout-json`` (mỗi loại mẫu một file),
**bbox trên ảnh** mặc định lấy từ khối Chandra (giống ``*_chandra_boxes``).

- ``--section-bbox-source chandra-match`` (mặc định): với mỗi section theo ``ord``,
  chọn một khối Chandra *chưa dùng* có IoU cao nhất với bbox mẫu trong JSON; bbox
  đầu ra = ``data-bbox`` của khối đó (PDF pt rồi scale pixel). IoU thấp → fallback
  bbox mẫu (``source``: ``template``).
- ``--section-bbox-source schema-template``: bbox chỉ từ file mẫu (hành vi cũ).
- Không căn chỉnh toàn trang (dx=dy=0). Không dùng line-override.

Prompt: ``--prompt-mode`` + file / ``OCR_LAYOUT_PROMPT``. ``--viz-chandra-boxes``:
ảnh bbox thô Chandra.
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

from utils.hf_env import ensure_writable_huggingface_cache

ensure_writable_huggingface_cache()

from doc_utils import IMAGE_EXTENSIONS, PDF_EXTENSIONS, draw_boxes, list_inputs

import run_chandra_schema_layout as csl
from run_chandra_layout import iou as layout_iou

DEFAULT_MODEL = "datalab-to/chandra-ocr-2"
DEFAULT_SAMPLE_DIR = PROJECT_DIR / "data/samples/GIAY_GUI_TIEN_TIET_KIEM"
DEFAULT_PROMPT_FILE = PROJECT_DIR / "prompts/chandra_giay_gui_tk_html_layout.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chandra OCR 2 + schema layout (prompt Giấy gửi tiền tiết kiệm từ file).",
    )
    default_in = PROJECT_DIR / "data/test/GIAY_GUI_TIEN_TIET_KIEM"
    default_out = PROJECT_DIR / "results/chandra_giay_gui_tk_layout"
    p.add_argument("--input-dir", type=Path, default=default_in)
    p.add_argument("--input-file", type=Path, default=None)
    p.add_argument("--only", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=default_out)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument(
        "--lora-path",
        type=Path,
        default=None,
        help="Optional PEFT adapter dir.",
    )
    p.add_argument("--device-map", type=str, default="cuda:1")
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=("auto", "bfloat16", "float16", "float32"),
    )
    p.add_argument("--pdf-dpi", type=int, default=200)
    p.add_argument("--max-pixels", type=int, default=1_600_000)
    p.add_argument(
        "--layout-json",
        type=Path,
        default=DEFAULT_SAMPLE_DIR / "layout _GIAY_GUI_TIEN_TIET_KIEM.json",
    )
    p.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help="Khi --prompt-mode=file: file UTF-8 gửi kèm ảnh.",
    )
    p.add_argument(
        "--prompt-mode",
        type=str,
        default="file",
        choices=("file", "official"),
        help=(
            "file = đọc --prompt-file (mặc định, prompt biểu mẫu tiếng Việt). "
            "official = prompt layout ngắn của Chandra (OCR_LAYOUT_PROMPT)."
        ),
    )
    p.add_argument(
        "--viz-chandra-boxes",
        action="store_true",
        help="Ghi thêm *_chandra_boxes.jpg: bbox 0–1000 từ model map trực tiếp ra pixel ảnh.",
    )
    p.add_argument("--page-width-pt", type=float, default=596.0)
    p.add_argument("--page-height-pt", type=float, default=844.0)
    p.add_argument("--sig-page-margin-pt", type=float, default=10.0)
    p.add_argument(
        "--section-bbox-source",
        type=str,
        default="chandra-match",
        choices=("chandra-match", "schema-template"),
        help=(
            "chandra-match (default): bbox mỗi section = khối Chandra IoU cao nhất "
            "với bbox mẫu trong layout-json (mỗi khối Chandra tối đa một section). "
            "schema-template: bbox từ file mẫu (không ghép Chandra)."
        ),
    )
    p.add_argument(
        "--match-iou-min",
        type=float,
        default=0.05,
        help="Ngưỡng IoU (norm 0–1) cho chế độ chandra-match.",
    )
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


def load_prompt_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def sanitize_chandra_layout_html(raw: str) -> str:
    """Cắt phần model nối thêm (lặp layout / assistant) để parse khớp một lần sinh HTML."""
    t = raw.strip()
    lower = t.lower()
    cut = len(t)
    for needle in ("<think>", "\nassistant\n", "\r\nassistant\n", "\nassistant"):
        i = lower.find(needle.lower())
        if i != -1 and i < cut:
            cut = i
    t = t[:cut].strip()
    return t


def xyxy_px_to_norm1000(
    box_xyxy_px: list[float], img_w: int, img_h: int
) -> list[float]:
    """Cùng thang 0–1000 như data-bbox trong HTML (theo ảnh đầu vào)."""
    x0, y0, x1, y1 = box_xyxy_px
    return [
        round(x0 / img_w * 1000.0, 2),
        round(y0 / img_h * 1000.0, 2),
        round(x1 / img_w * 1000.0, 2),
        round(y1 / img_h * 1000.0, 2),
    ]


def chandra_blocks_to_draw_entries(
    blocks: list[dict[str, Any]], img_w: int, img_h: int
) -> list[dict[str, Any]]:
    """Map Chandra bbox_norm 0–1000 on the **input image** to pixel xyxy."""
    out: list[dict[str, Any]] = []
    for b in blocks:
        x0, y0, x1, y1 = b["bbox_norm"]
        out.append(
            {
                "label": b["label"],
                "box_xyxy": [
                    x0 / 1000.0 * img_w,
                    y0 / 1000.0 * img_h,
                    x1 / 1000.0 * img_w,
                    y1 / 1000.0 * img_h,
                ],
            }
        )
    return out


def build_messages(test_image: Any, prompt_text: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def _schema_sec_norm01(
    sec: dict[str, Any], page_w_pt: float, page_h_pt: float
) -> tuple[float, float, float, float]:
    x0 = sec["x_pt"] / page_w_pt
    y0 = sec["y_pt"] / page_h_pt
    x1 = (sec["x_pt"] + sec["w_pt"]) / page_w_pt
    y1 = (sec["y_pt"] + sec["h_pt"]) / page_h_pt
    return (
        max(0.0, min(1.0, x0)),
        max(0.0, min(1.0, y0)),
        max(0.0, min(1.0, x1)),
        max(0.0, min(1.0, y1)),
    )


def greedy_assign_chandra_to_sections(
    schema_sections: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    page_w_pt: float,
    page_h_pt: float,
    iou_min: float,
) -> dict[str, dict[str, Any]]:
    """Mỗi section (theo ord) chọn một khối Chandra chưa dùng có IoU cao nhất với bbox mẫu."""
    ch_norm01: list[tuple[float, float, float, float]] = []
    for b in blocks:
        x0, y0, x1, y1 = b["bbox_norm"]
        ch_norm01.append((x0 / 1000.0, y0 / 1000.0, x1 / 1000.0, y1 / 1000.0))

    used: set[int] = set()
    out: dict[str, dict[str, Any]] = {}
    for sec in sorted(schema_sections, key=lambda s: s["ord"]):
        name = sec["name"]
        sbox = _schema_sec_norm01(sec, page_w_pt, page_h_pt)
        best_j: int | None = None
        best_iou = 0.0
        for j, cbox in enumerate(ch_norm01):
            if j in used:
                continue
            iv = layout_iou(sbox, cbox)
            if iv > best_iou:
                best_iou = iv
                best_j = j
        if best_j is not None and best_iou >= iou_min:
            used.add(best_j)
            blk = blocks[best_j]
            out[name] = {
                "source": "chandra",
                "bbox_norm": list(blk["bbox_norm"]),
                "chandra_label": blk["label"],
                "iou": round(best_iou, 4),
                "block_index": best_j,
            }
        else:
            out[name] = {
                "source": "template",
                "bbox_norm": None,
                "chandra_label": None,
                "iou": round(best_iou, 4) if best_j is not None else 0.0,
                "block_index": None,
            }
    return out


def _finalize_signature_clip(
    box_pt: list[float],
    name: str,
    page_w_pt: float,
    page_h_pt: float,
    section_bottom_pt: dict[str, float],
    page_margin_pt: float,
) -> list[float]:
    """Kéo đáy vùng chữ ký + clip trong trang (PDF pt)."""
    x0, y0, x1, y1 = box_pt
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


def section_box_pt_for_entry(
    sec: dict[str, Any],
    chandra_match: dict[str, dict[str, Any]] | None,
    use_chandra_match: bool,
    page_w_pt: float,
    page_h_pt: float,
    sig_margin_pt: float,
) -> tuple[list[float], str]:
    """Trả về (box_pt, basis): chandra > template > schema."""
    name = sec["name"]

    if use_chandra_match and chandra_match is not None:
        m = chandra_match.get(name) or {}
        if m.get("source") == "chandra" and m.get("bbox_norm"):
            x0, y0, x1, y1 = m["bbox_norm"]
            box_pt = [
                x0 / 1000.0 * page_w_pt,
                y0 / 1000.0 * page_h_pt,
                x1 / 1000.0 * page_w_pt,
                y1 / 1000.0 * page_h_pt,
            ]
            box_pt = _finalize_signature_clip(
                box_pt,
                name,
                page_w_pt,
                page_h_pt,
                csl.DEFAULT_SECTION_BOTTOM_PT,
                sig_margin_pt,
            )
            return box_pt, "chandra"

    box_pt = csl.section_box_pt(
        sec,
        (0.0, 0.0),
        csl.DEFAULT_SECTION_BOTTOM_PT,
        sig_margin_pt,
        {},
        page_w_pt,
        page_h_pt,
    )
    if use_chandra_match and chandra_match is not None:
        m = chandra_match.get(name) or {}
        if m.get("source") == "template":
            return box_pt, "template"
    return box_pt, "schema"


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

    if args.prompt_mode == "official":
        prompt_text = csl.OCR_LAYOUT_PROMPT
        prompt_desc = "official OcrLayoutPrompt (short)"
    else:
        try:
            prompt_text = load_prompt_text(args.prompt_file)
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            return 1
        if not prompt_text:
            print(f"prompt file is empty: {args.prompt_file}", file=sys.stderr)
            return 1
        prompt_desc = f"file:{args.prompt_file.name}"

    schema_sections = csl.load_schema(args.layout_json)
    print(
        f"[chandra-giay-gui-tk] {len(schema_sections)} schema sections from "
        f"{args.layout_json.name}; prompt_mode={args.prompt_mode} ({prompt_desc}, "
        f"{len(prompt_text)} chars)"
    )

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = csl.resolve_dtype(args.dtype)
    print(f"[chandra-giay-gui-tk] Loading {args.model} dtype={dtype} device_map={args.device_map}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device_map,
    )
    proc_src = args.model
    if args.lora_path is not None:
        if not args.lora_path.is_dir():
            print(f"lora path not found: {args.lora_path}", file=sys.stderr)
            return 1
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.lora_path))
        if (args.lora_path / "tokenizer_config.json").is_file():
            proc_src = str(args.lora_path)
        print(f"[chandra-giay-gui-tk] Loaded LoRA from {args.lora_path}")
    processor = AutoProcessor.from_pretrained(proc_src)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "model": args.model,
        "lora_path": str(args.lora_path) if args.lora_path else None,
        "prompt_mode": args.prompt_mode,
        "prompt_file": str(args.prompt_file) if args.prompt_mode == "file" else None,
        "viz_chandra_boxes": bool(args.viz_chandra_boxes),
        "device_map": args.device_map,
        "dtype": str(dtype),
        "layout_json": str(args.layout_json),
        "page_size_pt": [args.page_width_pt, args.page_height_pt],
        "sig_page_margin_pt": args.sig_page_margin_pt,
        "section_bottom_pt": csl.DEFAULT_SECTION_BOTTOM_PT,
        "schema_alignment": "disabled",
        "section_bbox_source": args.section_bbox_source,
        "match_iou_min": args.match_iou_min,
        "items": [],
    }

    for src_path in inputs_list:
        for unit_name, image, page_pt in csl.load_units(src_path, args.pdf_dpi):
            page_w_pt, page_h_pt = page_pt
            test_image = csl.fit_to_max_pixels(image, args.max_pixels)
            img_w_px, img_h_px = test_image.size
            input_path = args.output_dir / f"{unit_name}_input.jpg"
            test_image.save(input_path, quality=92)

            messages = build_messages(test_image, prompt_text)
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {
                k: (v.to(model.device) if hasattr(v, "to") else v)
                for k, v in inputs.items()
            }

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
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            text = sanitize_chandra_layout_html(text)

            html_path = args.output_dir / f"{unit_name}_chandra_layout.html"
            html_path.write_text(text, encoding="utf-8")

            blocks = csl.parse_chandra_blocks(text)
            blocks_path = args.output_dir / f"{unit_name}_chandra_blocks.json"
            blocks_path.write_text(
                json.dumps(blocks, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            use_chandra_match = args.section_bbox_source == "chandra-match"
            chandra_match: dict[str, dict[str, Any]] | None = None
            if use_chandra_match:
                chandra_match = greedy_assign_chandra_to_sections(
                    schema_sections,
                    blocks,
                    page_w_pt,
                    page_h_pt,
                    args.match_iou_min,
                )
                n_from_ch = sum(
                    1 for v in chandra_match.values() if v.get("source") == "chandra"
                )
                print(
                    f"[match] {unit_name}: {n_from_ch}/{len(schema_sections)} sections "
                    f"bbox from Chandra (IoU>={args.match_iou_min})"
                )

            entries: list[dict[str, Any]] = []
            schema_json: list[dict[str, Any]] = []
            sx = img_w_px / page_w_pt
            sy = img_h_px / page_h_pt
            for sec in sorted(schema_sections, key=lambda s: s["ord"]):
                box_pt, basis = section_box_pt_for_entry(
                    sec,
                    chandra_match,
                    use_chandra_match,
                    page_w_pt,
                    page_h_pt,
                    args.sig_page_margin_pt,
                )
                box_px = [
                    box_pt[0] * sx,
                    box_pt[1] * sy,
                    box_pt[2] * sx,
                    box_pt[3] * sy,
                ]
                entries.append({"label": sec["name"], "box_xyxy": box_px})
                meta = (chandra_match or {}).get(sec["name"])
                row: dict[str, Any] = {
                    "name": sec["name"],
                    "ord": sec["ord"],
                    "box_pt": [round(v, 2) for v in box_pt],
                    "box_xyxy_px": [round(v, 2) for v in box_px],
                    "source": basis,
                }
                if meta:
                    row["match_iou"] = meta.get("iou")
                    row["chandra_label"] = meta.get("chandra_label")
                if meta and meta.get("bbox_norm"):
                    row["norm1000_chandra_block"] = [
                        round(float(x), 2) for x in meta["bbox_norm"]
                    ]
                else:
                    row["norm1000_chandra_block"] = None
                row["norm1000_output"] = xyxy_px_to_norm1000(box_px, img_w_px, img_h_px)
                schema_json.append(row)

            chandra_boxes_viz: str | None = None
            if args.viz_chandra_boxes:
                cb_path = args.output_dir / f"{unit_name}_chandra_boxes.jpg"
                draw_boxes(
                    test_image,
                    chandra_blocks_to_draw_entries(blocks, img_w_px, img_h_px),
                    cb_path,
                    font_scale=1.0,
                    color_by="label",
                )
                chandra_boxes_viz = str(cb_path)

            json_path = args.output_dir / f"{unit_name}_schema_layout.json"
            layout_meta: dict[str, Any] = {
                "source": str(src_path),
                "unit": unit_name,
                "image_size": [img_w_px, img_h_px],
                "input_image": str(input_path),
                "raw_html_path": str(html_path),
                "prompt_mode": args.prompt_mode,
                "prompt_file": (
                    str(args.prompt_file) if args.prompt_mode == "file" else None
                ),
                "schema_alignment": "disabled",
                "n_anchors": 0,
                "section_bbox_source": args.section_bbox_source,
                "match_iou_min": args.match_iou_min,
                "section_chandra_match": chandra_match,
                "coordinates": {
                    "norm1000_output": "bbox cuối (pt→px) đổi lại thang 0–1000, so được với data-bbox HTML",
                    "norm1000_chandra_block": "khối Chandra được greedy chọn (trùng HTML khi source=chandra); null nếu source=template/schema",
                    "box_pt": "PDF pt theo page_size_pt",
                },
                "sections": schema_json,
            }
            if chandra_boxes_viz:
                layout_meta["viz_chandra_boxes"] = chandra_boxes_viz
            json_path.write_text(
                json.dumps(layout_meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            viz_path = args.output_dir / f"{unit_name}_schema_layout.jpg"
            csl.draw_schema_layout(test_image, entries, viz_path, font_scale=1.0)

            item: dict[str, Any] = {
                "source": str(src_path),
                "unit": unit_name,
                "image_size": [img_w_px, img_h_px],
                "viz": str(viz_path),
                "json": str(json_path),
                "raw_html": str(html_path),
                "blocks_json": str(blocks_path),
                "n_sections": len(entries),
                "schema_alignment": "disabled",
                "prompt_mode": args.prompt_mode,
                "section_bbox_source": args.section_bbox_source,
                "match_iou_min": args.match_iou_min,
                "n_sections_chandra": (
                    sum(1 for r in schema_json if r.get("source") == "chandra")
                    if use_chandra_match else 0
                ),
                "n_sections_template": (
                    sum(1 for r in schema_json if r.get("source") == "template")
                    if use_chandra_match else 0
                ),
            }
            if chandra_boxes_viz:
                item["viz_chandra_boxes"] = chandra_boxes_viz
            summary["items"].append(item)
            n_ch = sum(1 for r in schema_json if r.get("source") == "chandra")
            n_tmpl = sum(1 for r in schema_json if r.get("source") == "template")
            print(
                f"[ok] {src_path.name} :: {unit_name} -> {len(entries)} sections "
                f"(chandra={n_ch} template={n_tmpl}) -> {viz_path.name}"
                + (f" + {Path(chandra_boxes_viz).name}" if chandra_boxes_viz else "")
            )

    summary_path = args.output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
