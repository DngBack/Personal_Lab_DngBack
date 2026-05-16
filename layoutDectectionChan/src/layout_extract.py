"""Minimal Chandra OCR 2 layout sample: prompt file + image/PDF + model from load_model."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image as PILImage
from transformers import AutoModelForImageTextToText, AutoProcessor

from load_model import load_model

_PDF = {".pdf"}


def _first_page_image(document_path: str | Path, pdf_dpi: int = 200) -> PILImage.Image:
    path = Path(document_path)
    if path.suffix.lower() in _PDF:
        import fitz

        doc = fitz.open(str(path))
        try:
            page = doc[0]
            mat = fitz.Matrix(pdf_dpi / 72, pdf_dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            return PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
        finally:
            doc.close()
    return PILImage.open(path).convert("RGB")


def _clean_decoded(raw: str) -> str:
    t = raw.strip()
    lo = t.lower()
    if "\nassistant\n" in lo:
        t = t[: lo.index("\nassistant\n")].strip()
    for m in ("</html>", "<|endoftext|>", "<|im_end|>"):
        if m in t:
            t = t[: t.index(m)]
    return t.strip()


def run_layout_on_image(
    prompt_text: str,
    image: PILImage.Image,
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    *,
    max_new_tokens: int = 4096,
) -> str:
    """Run Chandra layout on one RGB image; `prompt_text` is the full prompt string."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt_text},
    ]}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens)
    new_tokens = gen[:, inputs["input_ids"].shape[-1]:]
    text = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return _clean_decoded(text)


def sample_layout_output(
    prompt_file: str | Path,
    document_path: str | Path,
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    *,
    pdf_dpi: int = 200,
    max_new_tokens: int = 4096,
) -> str:
    """Run layout detection once (first PDF page or whole image); return model HTML/text."""
    prompt = Path(prompt_file).read_text(encoding="utf-8").strip()
    img = _first_page_image(document_path, pdf_dpi)
    return run_layout_on_image(
        prompt, img, model, processor, max_new_tokens=max_new_tokens,
    )


if __name__ == "__main__":
    import sys

    # Edit DOC, then:  cd layoutDectectionChan/src && python layout_extract.py
    _root = Path(__file__).resolve().parent.parent
    PROMPT = _root / "prompt" / "prompt_GGTTK.txt"
    DOC = _root / "data" / "test" / "YOUR_FILE.pdf"  # or .png / .jpg

    for p in (PROMPT, DOC):
        if not p.is_file():
            print(f"Missing file (edit path in layout_extract.py): {p}", file=sys.stderr)
            sys.exit(1)

    model, processor = load_model("datalab-to/chandra-ocr-2", device="cuda:0")
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    model.eval()

    out = sample_layout_output(PROMPT, DOC, model, processor)
    print(out)
