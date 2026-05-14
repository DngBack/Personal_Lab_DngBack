"""LoRA fine-tune ``datalab-to/chandra-ocr-2`` (HF ``AutoModelForImageTextToText``) on layout HTML.

Uses the same chat messages as ``run_chandra_schema_layout.build_messages`` (image then text),
and the same ``OCR_LAYOUT_PROMPT``. Supervision targets come from schema layout JSON
(see ``layout_sft_targets``).

Typical command (single GIẤY GỬI TIỀN sample, bf16, tiny LoRA):

  python train_chandra_layout_lora.py \\
    --output-dir outputs/chandra_layout_lora \\
    --num-train-epochs 2 --dataset-repeats 16

Train from a **clean PDF** (rasterise first page → RGB; page size in pt is read from
the PDF for ``load_sections`` so bbox 0–1000 targets match the document):

  python train_chandra_layout_lora.py \\
    --cuda-device 1 \\
    --pdf data/samples/GIAY_GUI_TIEN_TIET_KIEM/20251018+MYNTT1_0001-p18.pdf \\
    --pdf-dpi 220 \\
    --save-rendered-image outputs/chandra_layout_lora/train_page_render.png \\
    --output-dir outputs/chandra_layout_lora_pdf \\
    --num-train-epochs 2 --dataset-repeats 24

4-bit (less VRAM; install bitsandbytes):

  python train_chandra_layout_lora.py --load-in-4bit --output-dir outputs/chandra_lora_4bit

Load adapter for inference: pass ``PeftModel.from_pretrained(base, adapter)`` or merge
weights; then point ``run_chandra_schema_layout.py --model`` to the merged dir / adapter.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

# Select physical GPU **before** importing torch (``torch.utils.data.Dataset`` pulls torch in).
if "--cuda-device" in sys.argv:
    i = sys.argv.index("--cuda-device")
    if i + 1 < len(sys.argv) and sys.argv[i + 1].strip("-").isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
        print(
            f"[train] Early CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} "
            "(this process will see it as cuda:0)",
            file=sys.stderr,
        )

from torch.utils.data import Dataset

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from doc_utils import render_pdf_pages  # noqa: E402
from layout_sft_targets import (  # noqa: E402
    build_assistant_text,
    build_chandra_user_messages,
    load_sections,
    resize_max_side,
)
from run_chandra_schema_layout import (  # noqa: E402
    DEFAULT_MODEL,
    compose_ocr_layout_prompt,
    fit_to_max_pixels,
)

DEFAULT_IMAGE = (
    PROJECT_DIR
    / "data/samples/GIAY_GUI_TIEN_TIET_KIEM/IMAGE_LAYOUT_GIAY_GUI_TIEN_TIET_KIEM.png"
)
DEFAULT_LAYOUT = PROJECT_DIR / "data/samples/GIAY_GUI_TIEN_TIET_KIEM/layout _GIAY_GUI_TIEN_TIET_KIEM.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA SFT for Chandra OCR 2 layout HTML.")
    p.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        metavar="N",
        help="Physical GPU index (e.g. 1). Must match early argv parsing at import time; "
        "prefer passing as the first flags, e.g. ``python ... --cuda-device 1 --pdf ...``. "
        "Alternatively: ``CUDA_VISIBLE_DEVICES=1 python ...`` (no flag needed).",
    )
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help="Training RGB image. Ignored if --pdf is set.",
    )
    p.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Optional PDF: rasterise one page (see --pdf-page, --pdf-dpi) as the training image. "
        "Page width/height in pt are taken from the PDF for layout→0–1000 targets.",
    )
    p.add_argument("--pdf-page", type=int, default=0, help="0-based page index inside the PDF.")
    p.add_argument("--pdf-dpi", type=int, default=220, help="Rasterisation DPI (match inference PDF render).")
    p.add_argument(
        "--save-rendered-image",
        type=Path,
        default=None,
        help="If set, save the preprocessed training PIL image here (after resize) for QC.",
    )
    p.add_argument("--layout-json", type=Path, default=DEFAULT_LAYOUT)
    p.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs/chandra_layout_lora")
    p.add_argument(
        "--stage",
        choices=("reasoning_html", "html_only"),
        default="html_only",
        help="html_only is safer for 1-sample (shorter); reasoning_html matches Unsloth two-stage.",
    )
    p.add_argument(
        "--page-width-pt",
        type=float,
        default=596.0,
        help="Used only with --image (PNG). With --pdf, size is read from the PDF page rect.",
    )
    p.add_argument(
        "--page-height-pt",
        type=float,
        default=844.0,
        help="Used only with --image. With --pdf, size is read from the PDF page rect.",
    )
    p.add_argument("--max-pixels", type=int, default=1_600_000, help="Match run_chandra_schema_layout resize.")
    p.add_argument("--max-image-side", type=int, default=0, help="If >0, also cap longest image side (px).")
    p.add_argument(
        "--schema-template-pt",
        type=float,
        nargs=2,
        metavar=("W", "H"),
        default=[596.0, 844.0],
        help="Same meaning as run_chandra_schema_layout: layout JSON in template pt; "
        "SFT bbox 0–1000 uses these denominators when W,H > 0. Use 0 0 for legacy (divide by page pt).",
    )
    p.add_argument(
        "--extra-layout-prompt-file",
        type=Path,
        default=None,
        help="UTF-8 text appended after base OCR layout prompt (same as inference script).",
    )
    p.add_argument(
        "--giay-gui-tien-layout-guide",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append Vietnamese GIẤY GỬI zone hints (same as run_chandra_schema_layout).",
    )
    p.add_argument("--dataset-repeats", type=int, default=24)
    p.add_argument("--num-train-epochs", type=float, default=2.0)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1.5e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    return p.parse_args()


def _prefix_match_len(a: Any, b: Any) -> int:
    import torch

    n = min(int(a.shape[0]), int(b.shape[0]))
    for i in range(n):
        if int(a[i]) != int(b[i]):
            return i
    return n


def infer_lora_target_modules(model: Any) -> list[str]:
    want = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    hit: set[str] = set()
    for name, module in model.named_modules():
        if module.__class__.__name__ != "Linear":
            continue
        suf = name.rsplit(".", 1)[-1]
        if suf in want:
            hit.add(suf)
    return sorted(hit) if hit else sorted(want)


class ChandraLayoutDataset(Dataset):
    """Repeats one (image, assistant_text) pair; tokenizes with the Chandra processor."""

    def __init__(
        self,
        processor: Any,
        image: Any,
        assistant_text: str,
        repeats: int,
        user_layout_prompt: str | None = None,
    ) -> None:
        self.processor = processor
        self.image = image
        self.assistant_text = assistant_text
        self.user_messages = build_chandra_user_messages(
            image, prompt_text=user_layout_prompt,
        )
        self.full_messages = self.user_messages + [{
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_text}],
        }]
        self.repeats = max(1, repeats)

    def __len__(self) -> int:
        return self.repeats

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch

        prompt_batch = self.processor.apply_chat_template(
            self.user_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        full_batch = self.processor.apply_chat_template(
            self.full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        p_ids = prompt_batch["input_ids"][0]
        f_ids = full_batch["input_ids"][0]
        prompt_len = _prefix_match_len(p_ids, f_ids)
        if prompt_len < p_ids.shape[0]:
            print(
                f"[warn] prompt tokenization prefix length {prompt_len} < "
                f"standalone prompt len {p_ids.shape[0]}; masking uses prefix match.",
                file=sys.stderr,
            )

        labels = f_ids.clone()
        labels[:prompt_len] = -100

        item: dict[str, Any] = {}
        for k, v in full_batch.items():
            if hasattr(v, "shape") and v.shape and v.shape[0] == 1:
                item[k] = v[0]
            elif hasattr(v, "shape"):
                item[k] = v
            else:
                item[k] = v
        item["labels"] = labels
        return item


def collate_and_filter(model: Any, batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    sig = inspect.signature(model.forward)
    params = sig.parameters
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    tensor_keys = [k for k in batch[0] if isinstance(batch[0][k], torch.Tensor)]
    if not has_varkw:
        ok = {n for n in params if n != "self"}
        tensor_keys = [k for k in tensor_keys if k in ok]

    out: dict[str, Any] = {}
    for k in tensor_keys:
        vals = [b[k] for b in batch]
        out[k] = torch.stack(vals, dim=0)
    return out


def load_training_image_from_pdf(
    pdf_path: Path, page_idx: int, dpi: int
) -> tuple[Any, float, float]:
    """Return (PIL RGB image, page_width_pt, page_height_pt)."""
    import fitz

    with fitz.open(pdf_path) as doc:
        if page_idx < 0 or page_idx >= doc.page_count:
            raise ValueError(f"pdf page {page_idx} out of range (0..{doc.page_count - 1})")
        rect = doc[page_idx].rect
        w_pt, h_pt = float(rect.width), float(rect.height)
    pages = render_pdf_pages(pdf_path, dpi)
    return pages[page_idx], w_pt, h_pt


def run() -> int:
    args = parse_args()
    if not args.layout_json.is_file():
        print(f"layout json not found: {args.layout_json}", file=sys.stderr)
        return 1

    from PIL import Image

    page_w_pt = args.page_width_pt
    page_h_pt = args.page_height_pt
    source_desc: str

    if args.pdf is not None:
        if not args.pdf.is_file():
            print(f"pdf not found: {args.pdf}", file=sys.stderr)
            return 1
        try:
            pil, page_w_pt, page_h_pt = load_training_image_from_pdf(
                args.pdf, args.pdf_page, args.pdf_dpi
            )
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 1
        pil = pil.convert("RGB")
        source_desc = f"pdf:{args.pdf} page={args.pdf_page} dpi={args.pdf_dpi} pt={page_w_pt}x{page_h_pt}"
        print(f"[train] Using {source_desc}")
    else:
        if not args.image.is_file():
            print(f"image not found: {args.image}", file=sys.stderr)
            return 1
        pil = Image.open(args.image).convert("RGB")
        source_desc = f"image:{args.image}"

    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        print(
            "Install: pip install -r prompt4ocr/requirements_chandra.txt "
            "(torch, transformers, accelerate, peft; bitsandbytes optional for 4-bit).\n"
            f"ImportError: {e}",
            file=sys.stderr,
        )
        return 2

    st = args.schema_template_pt
    schema_tpl: tuple[float, float] | None = None
    if st[0] > 0.0 and st[1] > 0.0:
        schema_tpl = (float(st[0]), float(st[1]))

    try:
        user_layout_prompt = compose_ocr_layout_prompt(
            extra_file=args.extra_layout_prompt_file,
            giay_gui_tien_guide=args.giay_gui_tien_layout_guide,
        )
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1

    sections = load_sections(args.layout_json, page_w_pt, page_h_pt, schema_tpl)
    if not sections:
        print("No sections with layout in JSON.", file=sys.stderr)
        return 1

    assistant_text = build_assistant_text(args.stage, sections)
    pil = fit_to_max_pixels(pil, args.max_pixels)
    if args.max_image_side > 0:
        pil = resize_max_side(pil, args.max_image_side)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_rendered_image is not None:
        args.save_rendered_image.parent.mkdir(parents=True, exist_ok=True)
        pil.save(args.save_rendered_image, quality=95)
        print(f"[train] Saved rendered training image -> {args.save_rendered_image}")

    (args.output_dir / "train_target_preview.txt").write_text(assistant_text, encoding="utf-8")
    (args.output_dir / "train_meta.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "stage": args.stage,
                "source": source_desc,
                "page_size_pt": [page_w_pt, page_h_pt],
                "pdf": str(args.pdf) if args.pdf else None,
                "pdf_page": args.pdf_page if args.pdf else None,
                "pdf_dpi": args.pdf_dpi if args.pdf else None,
                "image": str(args.image) if args.pdf is None else None,
                "layout_json": str(args.layout_json),
                "n_sections": len(sections),
                "max_pixels": args.max_pixels,
                "max_image_side": args.max_image_side,
                "schema_template_pt": list(args.schema_template_pt),
                "schema_template_for_sft": schema_tpl is not None,
                "giay_gui_tien_layout_guide": args.giay_gui_tien_layout_guide,
                "extra_layout_prompt_file": str(args.extra_layout_prompt_file)
                if args.extra_layout_prompt_file
                else None,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "cuda_device_flag": args.cuda_device,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    use_bf16 = args.bf16
    if use_bf16 is None:
        use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())

    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        )

    processor = AutoProcessor.from_pretrained(args.model)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    if args.load_in_4bit:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            device_map="auto",
            quantization_config=quant_config,
        )
    else:
        load_dtype = torch.bfloat16 if use_bf16 else torch.float32
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            dtype=load_dtype,
            device_map="auto",
        )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    targets = infer_lora_target_modules(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=targets,
        ),
    )
    model.print_trainable_parameters()

    ds = ChandraLayoutDataset(
        processor, pil, assistant_text, args.dataset_repeats,
        user_layout_prompt=user_layout_prompt,
    )

    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return collate_and_filter(model, batch)

    cuda = torch.cuda.is_available()
    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        bf16=cuda and bool(use_bf16),
        fp16=cuda and not bool(use_bf16),
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,  # type: ignore[arg-type]
        data_collator=_collate,
    )
    trainer.train()

    adapter_dir = args.output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    print(f"Saved LoRA adapter + processor to {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
