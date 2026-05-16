"""Vision–language Qwen: ảnh trang + HTML Chandra + schema → HTML có data-schema.

Hỗ trợ:
  - ``qwen2_5_vl`` (VD ``Qwen/Qwen2.5-VL-3B-Instruct``) — qua ``qwen_vl_utils.process_vision_info``
  - ``qwen3_vl`` (VD ``Qwen/Qwen3-VL-4B-Instruct``) — ``processor.apply_chat_template(..., tokenize=True, ...)``

Không dùng checkpoint **text-only** như ``Qwen/Qwen3.5-4B`` (model_type ``qwen3_5``).
"""
from __future__ import annotations

import json
from typing import Any

from schema_merge_qwen import _strip_assistant_noise


def _fit_image(img: Any, max_pixels: int) -> Any:
    from PIL import Image as PILImage

    if not isinstance(img, PILImage.Image) or max_pixels <= 0:
        return img
    w, h = img.size
    if w * h <= max_pixels:
        return img
    scale = (max_pixels / (w * h)) ** 0.5
    return img.resize((int(w * scale), int(h * scale)), PILImage.Resampling.LANCZOS)


def _load_vl_model(model_id: str, device_map: str):
    from transformers import (
        AutoConfig,
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
    )

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    mt = getattr(config, "model_type", "") or ""
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    if mt == "qwen2_5_vl":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map=device_map,
            trust_remote_code=True,
        )
        return "qwen2_5_vl", processor, model

    if mt == "qwen3_vl":
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                dtype="auto",
                device_map=device_map,
                trust_remote_code=True,
            )
        except TypeError:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map=device_map,
                trust_remote_code=True,
            )
        return "qwen3_vl", processor, model

    hint = (
        f"Model '{model_id}' có model_type={mt!r} — không phải vision-language được hỗ trợ.\n"
        "  • Bật --schema-merge-with-image: dùng VL, ví dụ:\n"
        "      Qwen/Qwen2.5-VL-3B-Instruct   (nhẹ hơn)\n"
        "      Qwen/Qwen3-VL-4B-Instruct     (~4B, đa phương thức)\n"
        "  • Không dùng text-only Qwen/Qwen3.5-4B ở đây (chỉ hợp với --qwen-model khi tắt --schema-merge-with-image)."
    )
    raise ValueError(hint)


class QwenVLSchemaHtmlMerger:
    def __init__(
        self,
        model_id: str,
        device_map: str = "cuda:0",
    ) -> None:
        import torch

        self._torch = torch
        self._vl_kind, self.processor, self.model = _load_vl_model(model_id, device_map)
        self.model.eval()
        self._device = next(self.model.parameters()).device

    def merge_to_html(
        self,
        *,
        system_prompt: str,
        schema_fields: list[str],
        chandra_html: str,
        page_image: Any,
        reference_image: Any | None = None,
        max_pixels: int = 0,
        max_new_tokens: int = 8192,
        temperature: float = 0.0,
    ) -> str:
        if reference_image is not None:
            reference_image = _fit_image(reference_image, max_pixels)
        page_image = _fit_image(page_image, max_pixels)
        schema_json = json.dumps(schema_fields, ensure_ascii=False)
        user_text = (
            "schema_fields:\n"
            f"{schema_json}\n\n"
            "chandra_html:\n"
            f"{chandra_html.strip()}\n"
        )
        user_content: list[dict[str, Any]] = []
        if reference_image is not None:
            user_content.append({"type": "image", "image": reference_image})
        user_content.append({"type": "image", "image": page_image})
        user_content.append({"type": "text", "text": user_text})

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_content},
        ]

        do_sample = temperature > 0
        gen_kw: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kw["temperature"] = max(0.05, temperature)

        if self._vl_kind == "qwen2_5_vl":
            from qwen_vl_utils import process_vision_info

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

        inputs = inputs.to(self._device)

        with self._torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **gen_kw)
        in_len = inputs["input_ids"].shape[1]
        trimmed = generated_ids[:, in_len:]
        decoded = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return _strip_assistant_noise(decoded)

    def cleanup(self) -> None:
        import gc

        self.model = None
        self.processor = None
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


__all__ = ["QwenVLSchemaHtmlMerger"]
