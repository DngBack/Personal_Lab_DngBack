"""OpenAI-compatible vision client.

Supports both the standard OpenAI endpoint and any OpenAI-compatible server
(e.g. local vLLM serving Qwen 3 VL 4B). Configure via environment variables:

    OPENAI_API_KEY       – required for standard OpenAI
    OPENAI_BASE_URL      – optional override (e.g. http://localhost:8000/v1)
    QWEN_BASE_URL        – optional separate base URL for the schema-alignment model
    QWEN_API_KEY         – optional separate API key for the schema-alignment model
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI


def _encode_pil(image: Any) -> str:
    """Encode a PIL Image to base64 JPEG string."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _encode_path(image_path: str | Path) -> str:
    """Encode an image file to base64 string."""
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def image_to_data_url(image: Any, path: str | Path | None = None) -> str:
    """Convert a PIL Image or file path to an OpenAI data URL."""
    if path is not None:
        data = _encode_path(path)
        suffix = Path(path).suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
    else:
        data = _encode_pil(image)
        mime = "image/jpeg"
    return f"data:{mime};base64,{data}"


def make_openai_client(
    *,
    use_qwen_endpoint: bool = False,
) -> OpenAI:
    """Build an OpenAI client, optionally pointing at the Qwen-compatible endpoint.

    Args:
        use_qwen_endpoint: If True, uses QWEN_BASE_URL / QWEN_API_KEY when set.
            Falls back to standard OPENAI_* variables if unset.
    """
    if use_qwen_endpoint:
        api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("QWEN_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL")

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def chat_vision(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_text: str,
    images: list[Any] | None = None,
    image_paths: list[str | Path] | None = None,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    json_mode: bool = False,
) -> str:
    """Send a multimodal chat request to an OpenAI-compatible endpoint.

    Images are passed as base64 data URLs in the user message, in the order:
    image_paths first (if provided), then PIL images.

    Args:
        client: An OpenAI client instance.
        model: Model identifier (e.g. "gpt-4o" or "Qwen/Qwen3-VL-4B-Instruct").
        system_prompt: System message text.
        user_text: Text portion of the user message.
        images: Optional list of PIL Image objects.
        image_paths: Optional list of local image file paths.
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature (0 = greedy).
        json_mode: If True, requests JSON output format.

    Returns:
        Raw text content from the first response choice.
    """
    user_content: list[dict[str, Any]] = []

    for p in image_paths or []:
        url = image_to_data_url(None, path=p)
        user_content.append({"type": "image_url", "image_url": {"url": url}})

    for img in images or []:
        url = image_to_data_url(img)
        user_content.append({"type": "image_url", "image_url": {"url": url}})

    user_content.append({"type": "text", "text": user_text})

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def chat_text(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_text: str,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    json_mode: bool = False,
) -> str:
    """Send a text-only chat request to an OpenAI-compatible endpoint.

    Args:
        client: An OpenAI client instance.
        model: Model identifier.
        system_prompt: System message text.
        user_text: User message text.
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature.
        json_mode: If True, requests JSON output format.

    Returns:
        Raw text content from the first response choice.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def parse_json_response(text: str) -> dict[str, Any]:
    """Extract and parse a JSON object from a model response.

    Handles both raw JSON and JSON wrapped in markdown code fences.

    Args:
        text: Raw text response from the model.

    Returns:
        Parsed JSON as a dictionary.

    Raises:
        ValueError: If no valid JSON object can be found.
    """
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        inner = [l for l in lines if not l.startswith("```")]
        text = "\n".join(inner).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot parse JSON from model response: {exc}\n---\n{text[:500]}") from exc


__all__ = [
    "make_openai_client",
    "chat_vision",
    "chat_text",
    "image_to_data_url",
    "parse_json_response",
]
