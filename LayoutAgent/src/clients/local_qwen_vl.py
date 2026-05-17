"""Local Qwen 3 VL client using HuggingFace Transformers.

Wraps QwenVLSchemaHtmlMerger from layoutDectectionChan/src/schema_merge_qwen_vl.py
to perform schema alignment locally without any API call.

The model (Qwen/Qwen3-VL-4B-Instruct or Qwen2.5-VL-*) is loaded via the
existing transformer pipeline and run in inference mode.

Usage:
    from clients.local_qwen_vl import LocalQwenVLClient

    client = LocalQwenVLClient("Qwen/Qwen3-VL-4B-Instruct", device_map="cuda:0")
    merged_html = client.align(
        system_prompt=prompt_text,
        schema_fields=fields,
        chandra_html=chandra_html,
        page_image=pil_image,
        reference_image=ref_pil_image,  # optional
    )
    client.cleanup()
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Bootstrap layoutDectectionChan/src into sys.path.
_CHAN_SRC = Path(__file__).resolve().parent.parent.parent.parent / "layoutDectectionChan" / "src"
if _CHAN_SRC.is_dir() and str(_CHAN_SRC) not in sys.path:
    sys.path.insert(0, str(_CHAN_SRC))


class LocalQwenVLClient:
    """Wrapper around Qwen 3 VL running locally via HuggingFace Transformers.

    Delegates to QwenVLSchemaHtmlMerger from layoutDectectionChan, which
    handles both Qwen2.5-VL and Qwen3-VL model types automatically.

    Args:
        model_id: HuggingFace model ID or local path.
            Supported: "Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct", etc.
        device_map: Device mapping string, e.g. "cuda:0" or "auto".
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 = greedy).
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        device_map: str = "cuda:0",
        max_new_tokens: int = 8192,
        temperature: float = 0.0,
    ) -> None:
        from schema_merge_qwen_vl import QwenVLSchemaHtmlMerger  # layoutDectectionChan/src

        print(f"[local_qwen_vl] Loading model {model_id} on {device_map} ...", flush=True)
        self._merger = QwenVLSchemaHtmlMerger(model_id, device_map=device_map)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        print("[local_qwen_vl] Model loaded.", flush=True)

    def align(
        self,
        *,
        system_prompt: str,
        schema_fields: list[str],
        chandra_html: str,
        page_image: Any,
        reference_image: Any | None = None,
        max_pixels: int = 0,
    ) -> str:
        """Run schema alignment: map Chandra HTML fragments to schema field names.

        Sends the system prompt, schema field list, Chandra HTML, and images to
        the Qwen VL model. Returns an HTML string where each <div> has a
        ``data-schema`` attribute identifying its schema field.

        Args:
            system_prompt: Schema-alignment system prompt text.
            schema_fields: Ordered list of expected schema field names.
            chandra_html: Raw layout HTML from Chandra OCR-2.
            page_image: PIL Image (RGB) of the page being processed.
            reference_image: Optional PIL Image of the reference schema boxes.
            max_pixels: Resize images to at most this many pixels (0 = no resize).

        Returns:
            Merged HTML string starting at the first <div element.
        """
        return self._merger.merge_to_html(
            system_prompt=system_prompt,
            schema_fields=schema_fields,
            chandra_html=chandra_html,
            page_image=page_image,
            reference_image=reference_image,
            max_pixels=max_pixels,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

    def cleanup(self) -> None:
        """Release GPU memory and unload the model."""
        self._merger.cleanup()
        self._merger = None
        print("[local_qwen_vl] Model unloaded.", flush=True)


__all__ = ["LocalQwenVLClient"]
