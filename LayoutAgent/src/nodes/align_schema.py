"""align_schema node: map Chandra HTML fragments to schema field names.

Sends the current schema-alignment prompt, Chandra HTML, schema field list, and
optionally one or two images to a vision-language model. The model returns an
HTML string where each <div> carries a ``data-schema`` attribute identifying
the schema field it represents.

Intended model: **Qwen 3 VL 4B** (or any Qwen VL variant) served through an
OpenAI-compatible API endpoint. Configure via environment variables:

    QWEN_BASE_URL   – base URL of the vLLM / Ollama / cloud endpoint
                      e.g. http://localhost:8000/v1
    QWEN_API_KEY    – API key for the Qwen endpoint (or "EMPTY" for local)
    QWEN_MODEL      – model name, e.g. Qwen/Qwen3-VL-4B-Instruct

When QWEN_BASE_URL is not set, the node falls back to the standard OpenAI
endpoint using OPENAI_API_KEY (useful for testing with GPT-4o).

Required state keys:
    current_prompt (str): Schema-alignment system prompt.
    chandra_html (str): Raw layout HTML from the Chandra node.
    schema_fields (list[str]): Ordered list of expected schema field names.
    page_image_path (str): Path to the PDF or image to rasterize as context.

Optional state keys:
    openai_model (str): Vision model identifier override. Defaults to:
        QWEN_MODEL env → "Qwen/Qwen3-VL-4B-Instruct" when QWEN_BASE_URL set,
        or "gpt-4o" otherwise.
    reference_image_path (str | None): Path to the ground-truth schema boxes JPEG.
    viz_page (int): PDF page number (default 1).
    viz_dpi (int): PDF rasterization DPI (default 200).
    align_max_tokens (int): Maximum response tokens (default 8192).
    align_temperature (float): Sampling temperature (default 0.0).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_DEFAULT_ALIGN_MODEL_QWEN = "Qwen/Qwen3-VL-4B-Instruct"
_DEFAULT_ALIGN_MODEL_OPENAI = "gpt-4o"


def _resolve_align_model() -> str:
    """Pick the schema-alignment model based on available environment config.

    Priority:
    1. QWEN_MODEL env var (explicit model name for Qwen endpoint)
    2. "Qwen/Qwen3-VL-4B-Instruct" when QWEN_BASE_URL is configured
    3. "gpt-4o" as fallback (standard OpenAI, useful for testing)
    """
    if os.environ.get("QWEN_MODEL"):
        return os.environ["QWEN_MODEL"]
    if os.environ.get("QWEN_BASE_URL"):
        return _DEFAULT_ALIGN_MODEL_QWEN
    return _DEFAULT_ALIGN_MODEL_OPENAI


def align_schema_node(state: dict[str, Any]) -> dict[str, Any]:
    """Call the Qwen VL model to assign schema names to Chandra HTML regions.

    Supports two execution modes:

    local  — Runs Qwen 3 VL 4B locally via HuggingFace Transformers.
             Set state["use_local_models"]=True or env USE_LOCAL_MODELS=1.
             Model: state["qwen_model_id"] or env QWEN_MODEL
                    (default "Qwen/Qwen3-VL-4B-Instruct").
             Device: state["qwen_device"] or env QWEN_DEVICE (default "cuda:0").

    api    — Calls Qwen (or GPT-4o fallback) via OpenAI-compatible API.
             Set QWEN_BASE_URL + QWEN_API_KEY for Qwen endpoints.

    The node sends the current schema-alignment prompt, schema field list,
    Chandra HTML, and optionally images (reference + page) to the model.

    Args:
        state: Current agent state.

    Returns:
        Partial state update with ``merged_html``.
    """
    from utils.image_io import load_page_image

    use_local = state.get("use_local_models") or os.environ.get("USE_LOCAL_MODELS", "").lower() in ("1", "true", "yes")

    iteration = state.get("iteration", 0)
    schema_fields: list[str] = state["schema_fields"]
    current_prompt: str = state["current_prompt"]
    chandra_html: str = state["chandra_html"].strip()

    page_image = load_page_image(
        state["page_image_path"],
        page=state.get("viz_page", 1),
        dpi=state.get("viz_dpi", 200),
    )

    ref_image = None
    ref_path = state.get("reference_image_path")
    if ref_path and Path(ref_path).is_file():
        from PIL import Image as PILImage
        ref_image = PILImage.open(ref_path).convert("RGB")

    if use_local:
        raw_html = _align_local(
            state=state,
            system_prompt=current_prompt,
            schema_fields=schema_fields,
            chandra_html=chandra_html,
            page_image=page_image,
            reference_image=ref_image,
            iteration=iteration,
        )
    else:
        raw_html = _align_api(
            state=state,
            system_prompt=current_prompt,
            schema_fields=schema_fields,
            chandra_html=chandra_html,
            page_image=page_image,
            ref_path=ref_path,
            iteration=iteration,
        )

    merged_html = _clean_html_output(raw_html)
    print(
        f"[align_schema] Response length={len(merged_html)} chars  "
        f"(divs≈{merged_html.count('<div')})",
        flush=True,
    )
    return {"merged_html": merged_html}


def _align_local(
    *,
    state: dict[str, Any],
    system_prompt: str,
    schema_fields: list[str],
    chandra_html: str,
    page_image: Any,
    reference_image: Any | None,
    iteration: int,
) -> str:
    """Run schema alignment using local Qwen VL transformer."""
    from clients.local_qwen_vl import LocalQwenVLClient

    model_id = (
        state.get("qwen_model_id")
        or os.environ.get("QWEN_MODEL")
        or _DEFAULT_ALIGN_MODEL_QWEN
    )
    device = state.get("qwen_device") or os.environ.get("QWEN_DEVICE", "cuda:0")

    print(
        f"[align_schema] iter={iteration}  LOCAL  model={model_id}  device={device}  "
        f"schema_fields={len(schema_fields)}",
        flush=True,
    )

    client = LocalQwenVLClient(
        model_id=model_id,
        device_map=device,
        max_new_tokens=state.get("align_max_tokens", 8192),
        temperature=state.get("align_temperature", 0.0),
    )
    try:
        return client.align(
            system_prompt=system_prompt,
            schema_fields=schema_fields,
            chandra_html=chandra_html,
            page_image=page_image,
            reference_image=reference_image,
        )
    finally:
        client.cleanup()


def _align_api(
    *,
    state: dict[str, Any],
    system_prompt: str,
    schema_fields: list[str],
    chandra_html: str,
    page_image: Any,
    ref_path: str | None,
    iteration: int,
) -> str:
    """Run schema alignment using OpenAI-compatible API (Qwen endpoint or GPT-4o)."""
    from clients.openai_vision import chat_vision, make_openai_client

    model = (
        state.get("openai_model")
        or os.environ.get("ALIGN_MODEL")
        or _resolve_align_model()
    )
    print(
        f"[align_schema] iter={iteration}  API  model={model}  "
        f"schema_fields={len(schema_fields)}",
        flush=True,
    )

    schema_json = json.dumps(schema_fields, ensure_ascii=False)
    user_text = (
        "schema_fields:\n"
        f"{schema_json}\n\n"
        "chandra_html:\n"
        f"{chandra_html}\n"
    )

    images: list[Any] = [page_image]
    image_paths: list[str | Path] = []
    if ref_path and Path(ref_path).is_file():
        image_paths.append(ref_path)

    client = make_openai_client(use_qwen_endpoint=_is_qwen_model(model))
    return chat_vision(
        client,
        model=model,
        system_prompt=system_prompt,
        user_text=user_text,
        images=images,
        image_paths=image_paths,
        max_tokens=state.get("align_max_tokens", 8192),
        temperature=state.get("align_temperature", 0.0),
    )


def _is_qwen_model(model: str) -> bool:
    """Return True if the model name looks like a Qwen model.

    When True, the client is built using the QWEN_BASE_URL / QWEN_API_KEY
    environment variables so that local vLLM deployments are used.
    """
    return "qwen" in model.lower()


def _clean_html_output(raw: str) -> str:
    """Strip markdown fences and non-HTML preamble from the model response.

    The schema-alignment prompt instructs the model to output only HTML, but
    models occasionally add a code fence or short preamble.

    Args:
        raw: Raw text returned by the vision model.

    Returns:
        Cleaned HTML string starting at the first <div element.
    """
    text = raw.strip()
    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        inner = [l for l in lines[1:] if not l.startswith("```")]
        text = "\n".join(inner).strip()
    # Find first <div to strip any remaining preamble
    lower = text.lower()
    idx = lower.find("<div")
    if idx > 0:
        text = text[idx:]
    return text.strip()


__all__ = ["align_schema_node"]
