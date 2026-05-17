"""run_chandra node: extract layout HTML from the input PDF via Chandra OCR-2.

Supports two execution modes:

  local  — Loads Chandra OCR-2 via HuggingFace Transformers on a local GPU.
           Requires: GPU, ``pip install transformers torch pymupdf``.
           Config:  state["chandra_device"] or env CHANDRA_DEVICE (default "cuda:0").
           Model:   state["chandra_model_id"] or env CHANDRA_MODEL
                    (default "datalab-to/chandra-ocr-2").

  api    — Calls Chandra via an OpenAI-compatible HTTP endpoint.
           Config:  env CHANDRA_BASE_URL + CHANDRA_API_KEY.

Mode selection: set state["use_local_models"]=True or env USE_LOCAL_MODELS=1
for local mode. Falls back to API mode otherwise.

This node runs ONCE per agent invocation (not per iteration). Cached HTML is
reused directly; the model is not re-loaded on subsequent calls.

Required state keys:
    page_image_path (str): Path to the PDF to process.
    chandra_prompt_path (str): Path to the Chandra prompt text file.

Optional state keys:
    chandra_log_path (str | None): Path to an existing or new cache log file.
    chandra_html (str | None): Pre-loaded HTML; skips all I/O if non-empty.
    use_local_models (bool): Force local transformer mode.
    chandra_model_id (str): HuggingFace model ID for local mode.
    chandra_device (str): PyTorch device for local mode (default "cuda:0").
    viz_page (int): 1-based page number (default 1).
    viz_dpi (int): Rasterization DPI (default 200).
    chandra_max_tokens (int): Max tokens to generate (default 4096).
    force_chandra_rerun (bool): Ignore cache and re-run (default False).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def run_chandra_node(state: dict[str, Any]) -> dict[str, Any]:
    """Call Chandra OCR-2 (local or API) or return cached HTML.

    If ``state["chandra_html"]`` is already populated, the node returns
    immediately. Otherwise it runs in local transformer mode or API mode
    depending on configuration.

    Args:
        state: Current agent state.

    Returns:
        Partial state update with ``chandra_html`` and ``chandra_log_path``.
    """
    existing_html = (state.get("chandra_html") or "").strip()
    if existing_html:
        print("[chandra] Reusing pre-loaded chandra_html from state", flush=True)
        return {}

    pdf_path = Path(state["page_image_path"])
    prompt_path = Path(state["chandra_prompt_path"])

    # Template injection: if the prompt contains {schema_fields}, substitute the
    # schema field list loaded from the sample JSON (one field per line).
    prompt_text = prompt_path.read_text(encoding="utf-8")
    if "{schema_fields}" in prompt_text:
        schema_fields: list[str] = state.get("schema_fields") or []
        if not schema_fields:
            raise ValueError(
                "Chandra prompt contains {schema_fields} placeholder but "
                "state['schema_fields'] is empty. Run setup_node first."
            )
        injected = "\n".join(schema_fields)
        prompt_text = prompt_text.replace("{schema_fields}", injected)
        # Write rendered prompt to a temp file so downstream clients can read a path
        import tempfile
        _tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        _tmp.write(prompt_text)
        _tmp.flush()
        prompt_path = Path(_tmp.name)
        print(
            f"[chandra] Injected {len(schema_fields)} schema fields into Chandra prompt template",
            flush=True,
        )
    log_path_raw = state.get("chandra_log_path")
    log_path = Path(log_path_raw) if log_path_raw else None
    force_rerun = state.get("force_chandra_rerun", False)
    page = state.get("viz_page", 1)
    dpi = state.get("viz_dpi", 200)
    max_tokens = state.get("chandra_max_tokens", 4096)

    use_local = state.get("use_local_models") or os.environ.get("USE_LOCAL_MODELS", "").lower() in ("1", "true", "yes")

    if use_local:
        from clients.local_chandra import run_local_chandra

        model_id = (
            state.get("chandra_model_id")
            or os.environ.get("CHANDRA_MODEL")
            or "datalab-to/chandra-ocr-2"
        )
        device = state.get("chandra_device") or os.environ.get("CHANDRA_DEVICE", "cuda:0")
        print(f"[chandra] Local transformer  model={model_id}  device={device}", flush=True)
        html, was_cached = run_local_chandra(
            pdf_path,
            prompt_path,
            log_path=log_path,
            page=page,
            dpi=dpi,
            model_id=model_id,
            device=device,
            max_new_tokens=max_tokens,
            force_rerun=force_rerun,
        )
    else:
        from clients.chandra_api import load_or_run_chandra

        html, was_cached = load_or_run_chandra(
            pdf_path,
            prompt_path,
            log_path=log_path,
            page=page,
            dpi=dpi,
            max_tokens=max_tokens,
            force_rerun=force_rerun,
        )

    if not html:
        raise RuntimeError(
            "Chandra returned empty HTML. "
            "Check model/API connection, prompt file, and PDF path."
        )

    actual_log = log_path or (pdf_path.parent / f"{pdf_path.stem}_llm.log")
    print(f"[chandra] HTML length={len(html)} chars  (cached={was_cached})", flush=True)
    return {
        "chandra_html": html,
        "chandra_log_path": str(actual_log),
    }


__all__ = ["run_chandra_node"]
