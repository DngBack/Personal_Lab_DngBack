#!/usr/bin/env python3
"""Single-shot inference: run Chandra extraction + schema alignment on one PDF.

Unlike main.py (which runs the full tuning loop), this script performs exactly
one forward pass: Chandra → schema alignment → merge → visualize → JSON.
Use this for testing any PDF without triggering prompt optimization.

The Chandra step can be performed by:
  a) GPT-4o / any OpenAI-compatible model (default, using OPENAI_API_KEY).
  b) The datalab Chandra OCR-2 API (set CHANDRA_BASE_URL + CHANDRA_API_KEY).
  c) Skipped entirely if --chandra-log points to an existing cached log.

Usage examples
--------------
Model roles
-----------
  Chandra OCR-2  (CHANDRA_MODEL / --chandra-model)  — layout extraction from image
  Qwen 3 VL 4B   (QWEN_MODEL    / --schema-model)   — schema alignment (HTML → schema)

Typical setup (local vLLM serving Qwen 3 VL 4B):
  In .env:
    QWEN_BASE_URL=http://localhost:8000/v1
    QWEN_API_KEY=EMPTY
    QWEN_MODEL=Qwen/Qwen3-VL-4B-Instruct
    CHANDRA_BASE_URL=https://api.datalab.to/api/v1
    CHANDRA_API_KEY=your-datalab-key

Reuse Chandra log + run Qwen schema alignment:

    python run_inference.py \\
        --pdf data/test/GIAY_GUI_TIEN_TIET_KIEM/test_6.pdf \\
        --layout-json "data/samples/GIAY_GUI_TIEN_TIET_KIEM/layout _GIAY_GUI_TIEN_TIET_KIEM.json" \\
        --chandra-log data/test/GIAY_GUI_TIEN_TIET_KIEM/test_6_llm.log \\
        --align-prompt data/tuning_runs/20260516_111325_193ac6/best_prompt.txt \\
        --out-dir data/test/GIAY_GUI_TIEN_TIET_KIEM/out_test6

Compare two alignment prompts (same Chandra log, different schema prompts):

    python run_inference.py \\
        --pdf data/test/GIAY_GUI_TIEN_TIET_KIEM/test_6.pdf \\
        --layout-json "data/samples/GIAY_GUI_TIEN_TIET_KIEM/layout _GIAY_GUI_TIEN_TIET_KIEM.json" \\
        --chandra-log data/test/GIAY_GUI_TIEN_TIET_KIEM/test_6_llm.log \\
        --align-prompt ../layoutDectectionChan/prompt/schema_merge_html_system.txt \\
        --align-prompt-b data/tuning_runs/20260516_111325_193ac6/best_prompt.txt \\
        --out-dir data/test/GIAY_GUI_TIEN_TIET_KIEM/out_compare_test6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _load_dotenv() -> None:
    env_path = _ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
    except ImportError:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def _resolve_schema_model() -> str:
    """Pick the schema-alignment model (Qwen 3 VL 4B) from environment.

    Priority:
    1. QWEN_MODEL env var
    2. "Qwen/Qwen3-VL-4B-Instruct" when QWEN_BASE_URL is configured
    3. "gpt-4o" fallback for testing without a Qwen endpoint
    """
    if os.environ.get("QWEN_MODEL"):
        return os.environ["QWEN_MODEL"]
    if os.environ.get("QWEN_BASE_URL"):
        return "Qwen/Qwen3-VL-4B-Instruct"
    return "gpt-4o"


def run_pipeline(
    *,
    pdf_path: Path,
    layout_json_path: Path,
    align_prompt_path: Path,
    out_dir: Path,
    chandra_prompt_path: Path | None = None,
    chandra_log_path: Path | None = None,
    schema_model: str | None = None,
    chandra_model: str | None = None,
    reference_image_path: Path | None = None,
    page: int = 1,
    dpi: int = 200,
    label: str = "",
    use_local: bool = False,
    chandra_device: str = "cuda:0",
    qwen_device: str = "cuda:0",
) -> dict:
    """Execute a full single-pass inference: Chandra → align → merge → viz → JSON.

    Model roles:
        chandra_model  — Chandra OCR-2 (layout extraction). Configured via
                         CHANDRA_MODEL env or ``--chandra-model`` CLI flag.
                         Uses CHANDRA_BASE_URL / CHANDRA_API_KEY endpoint.
        schema_model   — Qwen 3 VL 4B (schema alignment). Configured via
                         QWEN_MODEL env or ``--schema-model`` CLI flag.
                         Uses QWEN_BASE_URL / QWEN_API_KEY endpoint.
                         Falls back to GPT-4o if QWEN_BASE_URL is not set.

    Args:
        pdf_path: Input PDF file to process.
        layout_json_path: Layout JSON defining the schema field tree.
        align_prompt_path: System prompt for the schema-alignment model (Qwen VL).
        out_dir: Directory to write all output artifacts.
        chandra_prompt_path: Chandra extraction prompt (required if no chandra_log_path).
        chandra_log_path: Pre-existing Chandra log to reuse (skips OCR step).
        schema_model: Vision model for schema alignment (Qwen 3 VL 4B or fallback).
        chandra_model: Model for Chandra extraction (datalab-to/chandra-ocr-2).
        reference_image_path: Optional ground-truth image for visual context.
        page: 1-based PDF page number.
        dpi: Rasterization DPI.
        label: Optional label suffix for output filenames.
        use_local: If True, load models via HuggingFace Transformers on GPU.
        chandra_device: PyTorch device for local Chandra (e.g. "cuda:0").
        qwen_device: PyTorch device for local Qwen VL (e.g. "cuda:0").

    Returns:
        Dictionary with paths to generated output files and coverage metrics.
    """
    from utils.schema_html import (
        extract_chandra_html_from_log,
        get_extracted_schema_names,
        load_layout_names,
        merge_duplicate_schema_divs,
        build_layout_payload,
    )
    from utils.image_io import draw_schema_boxes_on_page, load_page_image
    from utils.scoring import compute_coverage

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"_{label}" if label else ""

    # ---- Resolve models -------------------------------------------------------
    _chandra_model = chandra_model or os.environ.get("CHANDRA_MODEL") or "datalab-to/chandra-ocr-2"

    # In local mode, always default to Qwen/Qwen3-VL-4B-Instruct unless explicitly overridden.
    if use_local:
        _schema_model = (
            schema_model
            or os.environ.get("ALIGN_MODEL")
            or os.environ.get("QWEN_MODEL")
            or "Qwen/Qwen3-VL-4B-Instruct"
        )
    else:
        _schema_model = schema_model or os.environ.get("ALIGN_MODEL") or _resolve_schema_model()

    mode_tag = "LOCAL (HuggingFace Transformers)" if use_local else "API"
    print(f"  Mode          : {mode_tag}", flush=True)
    print(f"  Chandra model : {_chandra_model}", flush=True)
    if use_local:
        print(f"  Qwen VL model : {_schema_model}  device={qwen_device}", flush=True)
    elif "qwen" in _schema_model.lower():
        print(f"  Schema model  : {_schema_model}  (endpoint: {os.environ.get('QWEN_BASE_URL','not set')})", flush=True)
    else:
        print(
            f"  Schema model  : {_schema_model}  "
            "[NOTE: no Qwen endpoint — OpenAI fallback. "
            "Use --local or set QWEN_BASE_URL for Qwen 3 VL 4B.]",
            flush=True,
        )

    # ---- 1. Load schema fields ------------------------------------------------
    schema_fields = load_layout_names(layout_json_path)
    print(f"  Schema fields : {len(schema_fields)}", flush=True)

    # ---- 2. Chandra extraction ------------------------------------------------
    chandra_html = ""
    if chandra_log_path:
        if not chandra_log_path.is_file():
            raise FileNotFoundError(
                f"Chandra log not found: {chandra_log_path}\n"
                "  • Generate it first with: layoutDectectionChan/run_pdf_llm_log.py <pdf>\n"
                "  • Or provide --chandra-prompt to run Chandra on-the-fly."
            )
        raw = chandra_log_path.read_text(encoding="utf-8")
        chandra_html = extract_chandra_html_from_log(raw)
        print(f"  Chandra HTML  : loaded from cache ({len(chandra_html)} chars)", flush=True)
    else:
        if chandra_prompt_path is None or not chandra_prompt_path.is_file():
            raise ValueError(
                "Provide --chandra-log (existing Chandra OCR output) OR "
                "--chandra-prompt (to run Chandra OCR-2)."
            )
        # Template injection: replace {schema_fields} placeholder if present
        _chandra_prompt_text = chandra_prompt_path.read_text(encoding="utf-8")
        if "{schema_fields}" in _chandra_prompt_text:
            _chandra_prompt_text = _chandra_prompt_text.replace(
                "{schema_fields}", "\n".join(schema_fields)
            )
            import tempfile as _tmpmod
            _tmp = _tmpmod.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            )
            _tmp.write(_chandra_prompt_text)
            _tmp.flush()
            chandra_prompt_path = Path(_tmp.name)
            print(
                f"  Chandra prompt: injected {len(schema_fields)} schema fields from template",
                flush=True,
            )

        log_out = out_dir / f"chandra_llm{stem}.log"
        if use_local:
            from clients.local_chandra import run_local_chandra
            chandra_html, was_cached = run_local_chandra(
                pdf_path,
                chandra_prompt_path,
                log_path=log_out,
                page=page,
                dpi=dpi,
                model_id=_chandra_model,
                device=chandra_device,
            )
        else:
            from clients.chandra_api import load_or_run_chandra
            chandra_html, was_cached = load_or_run_chandra(
                pdf_path,
                chandra_prompt_path,
                log_path=log_out,
                page=page,
                dpi=dpi,
                model=_chandra_model,
            )
        print(
            f"  Chandra HTML  : {len(chandra_html)} chars  "
            f"(model={_chandra_model}, cached={was_cached})",
            flush=True,
        )

    # Save stripped Chandra HTML for inspection
    chandra_strip_path = out_dir / f"chandra_strip{stem}.html"
    chandra_strip_path.write_text(chandra_html, encoding="utf-8")

    # ---- 3. Schema alignment (Qwen VL / GPT-4o) -------------------------------
    align_prompt = align_prompt_path.read_text(encoding="utf-8")
    page_image = load_page_image(pdf_path, page=page, dpi=dpi)

    images = [page_image]
    image_paths_extra: list[Path] = []
    if reference_image_path and reference_image_path.is_file():
        image_paths_extra.append(reference_image_path)

    schema_user_text = (
        "schema_fields:\n"
        + json.dumps(schema_fields, ensure_ascii=False)
        + "\n\nchandra_html:\n"
        + chandra_html.strip()
    )

    print(f"  Aligning schema with model={_schema_model} ...", flush=True)
    if use_local:
        from clients.local_qwen_vl import LocalQwenVLClient
        from PIL import Image as PILImage
        ref_img_pil = PILImage.open(reference_image_path).convert("RGB") if reference_image_path and reference_image_path.is_file() else None
        qwen_client = LocalQwenVLClient(
            model_id=_schema_model,
            device_map=qwen_device,
            max_new_tokens=8192,
            temperature=0.0,
        )
        try:
            raw_html = qwen_client.align(
                system_prompt=align_prompt,
                schema_fields=schema_fields,
                chandra_html=chandra_html.strip(),
                page_image=page_image,
                reference_image=ref_img_pil,
            )
        finally:
            qwen_client.cleanup()
    else:
        from clients.openai_vision import chat_vision, make_openai_client
        using_qwen = "qwen" in _schema_model.lower()
        client = make_openai_client(use_qwen_endpoint=using_qwen)
        raw_html = chat_vision(
            client,
            model=_schema_model,
            system_prompt=align_prompt,
            user_text=schema_user_text,
            images=images,
            image_paths=image_paths_extra,
            max_tokens=8192,
            temperature=0.0,
        )

    # Strip preamble
    raw_html = raw_html.strip()
    if raw_html.startswith("```"):
        lines = raw_html.splitlines()
        raw_html = "\n".join(l for l in lines[1:] if not l.startswith("```")).strip()
    idx = raw_html.lower().find("<div")
    if idx > 0:
        raw_html = raw_html[idx:]

    # ---- 4. Dedup / merge duplicate schemas ----------------------------------
    merged_html = merge_duplicate_schema_divs(raw_html)
    divs_before = raw_html.count("<div")
    divs_after = merged_html.count("<div")
    if divs_before != divs_after:
        print(f"  Merged duplicates: {divs_before} → {divs_after} divs", flush=True)

    merged_html_path = out_dir / f"schema_merged{stem}.html"
    merged_html_path.write_text(merged_html, encoding="utf-8")

    # ---- 5. Visualization ---------------------------------------------------
    viz_path = out_dir / f"schema_boxes{stem}.jpg"
    draw_schema_boxes_on_page(
        pdf_path, merged_html, viz_path,
        page=page, dpi=dpi, only_with_schema=True,
    )
    print(f"  Saved viz → {viz_path}", flush=True)

    # ---- 6. JSON output ------------------------------------------------------
    payload = build_layout_payload(layout_json_path, merged_html)
    payload["meta"] = {
        "pdf": str(pdf_path),
        "align_prompt": str(align_prompt_path),
        "chandra_model": _chandra_model,
        "schema_model": _schema_model,
        "label": label,
    }
    json_path = out_dir / f"layout_values{stem}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 7. Coverage summary -------------------------------------------------
    extracted = get_extracted_schema_names(merged_html)
    coverage, missing = compute_coverage(extracted, schema_fields)

    return {
        "label": label or "run",
        "coverage": coverage,
        "extracted": len(extracted),
        "total_fields": len(schema_fields),
        "missing_fields": missing,
        "viz_path": str(viz_path),
        "json_path": str(json_path),
        "merged_html_path": str(merged_html_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Single-shot inference: PDF → Chandra → schema alignment → visualization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--pdf", type=Path, required=True, help="Input PDF file.")
    p.add_argument("--layout-json", type=Path, required=True, help="Layout JSON file.")
    p.add_argument(
        "--align-prompt", type=Path, required=True,
        help="Schema alignment system prompt (prompt A, or the only prompt).",
    )
    p.add_argument(
        "--align-prompt-b", type=Path, default=None,
        help="Second alignment prompt for side-by-side comparison.",
    )
    p.add_argument("--chandra-prompt", type=Path, default=None, help="Chandra extraction prompt.")
    p.add_argument(
        "--chandra-log", type=Path, default=None,
        help="Existing Chandra log file (skips OCR step).",
    )
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    p.add_argument(
        "--local",
        action="store_true",
        help=(
            "Use local HuggingFace Transformers instead of API calls. "
            "Loads Chandra OCR-2 and Qwen 3 VL 4B on GPU. "
            "Requires CUDA and the model weights to be available locally."
        ),
    )
    p.add_argument(
        "--chandra-device", type=str, default=None,
        help="PyTorch device for Chandra local model (default: cuda:0).",
    )
    p.add_argument(
        "--qwen-device", type=str, default=None,
        help="PyTorch device for Qwen VL local model (default: cuda:0).",
    )
    p.add_argument(
        "--schema-model", type=str, default=None,
        help=(
            "Schema alignment model — intended: Qwen 3 VL 4B. "
            "Defaults to QWEN_MODEL env, or 'Qwen/Qwen3-VL-4B-Instruct' when "
            "QWEN_BASE_URL is set, or 'gpt-4o' as fallback. "
            "Requires QWEN_BASE_URL + QWEN_API_KEY for Qwen endpoints."
        ),
    )
    p.add_argument(
        "--chandra-model", type=str, default=None,
        help=(
            "Chandra OCR-2 model for layout extraction. "
            "Defaults to CHANDRA_MODEL env or 'datalab-to/chandra-ocr-2'. "
            "Requires CHANDRA_BASE_URL + CHANDRA_API_KEY."
        ),
    )
    p.add_argument(
        "--reference-image", type=Path, default=None,
        help="Ground-truth schema boxes image (optional visual reference).",
    )
    p.add_argument("--page", type=int, default=1, help="PDF page number (default: 1).")
    p.add_argument("--dpi", type=int, default=200, help="PDF rasterization DPI (default: 200).")
    return p


def main() -> int:
    _load_dotenv()
    args = _build_parser().parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("[run_inference] Warning: OPENAI_API_KEY not set.", flush=True)

    use_local = args.local or os.environ.get("USE_LOCAL_MODELS", "").lower() in ("1", "true", "yes")
    if use_local:
        print("[run_inference] Mode: LOCAL (HuggingFace Transformers)", flush=True)
    else:
        print("[run_inference] Mode: API (OpenAI-compatible)", flush=True)

    common = dict(
        pdf_path=args.pdf.resolve(),
        layout_json_path=args.layout_json.resolve(),
        chandra_prompt_path=args.chandra_prompt.resolve() if args.chandra_prompt else None,
        chandra_log_path=args.chandra_log.resolve() if args.chandra_log else None,
        schema_model=args.schema_model or os.environ.get("ALIGN_MODEL"),
        chandra_model=args.chandra_model or os.environ.get("CHANDRA_MODEL"),
        use_local=use_local,
        chandra_device=args.chandra_device or os.environ.get("CHANDRA_DEVICE", "cuda:0"),
        qwen_device=args.qwen_device or os.environ.get("QWEN_DEVICE", "cuda:0"),
        reference_image_path=args.reference_image.resolve() if args.reference_image else None,
        page=args.page,
        dpi=args.dpi,
    )

    results = []

    # ---- Run A (always) -------------------------------------------------------
    print(f"\n[run_inference] === Prompt A: {args.align_prompt.name} ===", flush=True)
    label_a = "A_" + args.align_prompt.stem
    result_a = run_pipeline(
        align_prompt_path=args.align_prompt.resolve(),
        out_dir=args.out_dir.resolve(),
        label=label_a,
        **common,
    )
    results.append(result_a)

    # ---- Run B (optional comparison) ------------------------------------------
    if args.align_prompt_b:
        print(f"\n[run_inference] === Prompt B: {args.align_prompt_b.name} ===", flush=True)
        label_b = "B_" + args.align_prompt_b.stem
        result_b = run_pipeline(
            align_prompt_path=args.align_prompt_b.resolve(),
            out_dir=args.out_dir.resolve(),
            label=label_b,
            **common,
        )
        results.append(result_b)

    # ---- Summary -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  INFERENCE RESULTS")
    print("=" * 60)
    for r in results:
        print(f"\n  [{r['label']}]")
        print(f"    Coverage  : {r['coverage']:.3f}  ({r['extracted']}/{r['total_fields']} fields)")
        print(f"    Missing   : {r['missing_fields']}")
        print(f"    Viz image : {r['viz_path']}")
        print(f"    JSON      : {r['json_path']}")
    print("=" * 60)

    # Save comparison summary
    summary_path = args.out_dir.resolve() / "inference_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Summary → {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
