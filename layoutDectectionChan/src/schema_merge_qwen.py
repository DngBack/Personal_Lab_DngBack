"""Qwen (3.5 4B class): Chandra HTML + schema list → merged HTML with data-schema."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _strip_assistant_noise(text: str) -> str:
    """Tối thiểu: model đã được yêu cầu chỉ in HTML; chỉ lấy từ thẻ đầu tiên."""
    t = text.strip()
    if "</think>" in t:
        t = t.split("</think>", 1)[-1].strip()
    i = t.find("<div")
    if i > 0:
        t = t[i:]
    return t.strip()


class QwenSchemaHtmlMerger:
    def __init__(
        self,
        model_id: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        dt_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dt = dt_map.get(dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dt if dt is not None else "auto",
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self._device = next(self.model.parameters()).device

    def merge_to_html(
        self,
        *,
        system_prompt: str,
        schema_fields: list[str],
        chandra_html: str,
        max_new_tokens: int = 8192,
        temperature: float = 0.0,
        user_prefix: str | None = None,
    ) -> str:
        if user_prefix is not None:
            user = user_prefix + chandra_html.strip() + "\n"
        elif schema_fields:
            schema_json = json.dumps(schema_fields, ensure_ascii=False)
            user = (
                "schema_fields:\n"
                f"{schema_json}\n\n"
                "chandra_html:\n"
                f"{chandra_html.strip()}\n"
            )
        else:
            user = chandra_html.strip() + "\n"
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user},
        ]
        try:
            templ = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            templ = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        enc = self.tokenizer(templ, return_tensors="pt")
        enc = {k: v.to(self._device) for k, v in enc.items()}
        do_sample = temperature > 0
        gen_kw: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }
        if do_sample:
            gen_kw["temperature"] = max(0.05, temperature)
        with self._torch.inference_mode():
            out = self.model.generate(**enc, **gen_kw)
        pref = enc["input_ids"].shape[-1]
        decoded = self.tokenizer.decode(out[0, pref:], skip_special_tokens=True)
        return _strip_assistant_noise(decoded)

    def cleanup(self) -> None:
        import gc

        self.model = None
        self.tokenizer = None
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def load_system_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")
