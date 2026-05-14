"""Fine-tune a small Qwen3-VL model (Unsloth) on schema layout → Chandra-style layout HTML.

Designed for **very few samples** (default: one PNG + one layout JSON): strong LoRA,
moderate epochs, optional dataset repetition, and toggles to train only language layers.

Follows Unsloth vision SFT patterns (FastVisionModel, UnslothVisionDataCollator, TRL SFTTrainer):
https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_(2B)_Vision.ipynb
Dataset message format:
https://docs.unsloth.ai/basics/vision-fine-tuning

User prompt text is aligned with ``run_chandra_schema_layout.OCR_LAYOUT_PROMPT`` so the
same instruction can be used at inference time.

Example (single GIẤY GỬI TIỀN sample, reasoning then HTML):

  python train_layout_unsloth_qwen_vl.py \\
    --stage reasoning_html \\
    --image data/samples/GIAY_GUI_TIEN_TIET_KIEM/IMAGE_LAYOUT_GIAY_GUI_TIEN_TIET_KIEM.png \\
    --layout-json 'data/samples/GIAY_GUI_TIEN_TIET_KIEM/layout _GIAY_GUI_TIEN_TIET_KIEM.json' \\
    --output-dir outputs/layout_lora_reasoning

HTML-only stage (shorter targets, good as second-stage distill):

  python train_layout_unsloth_qwen_vl.py \\
    --stage html_only \\
    --image ... --layout-json ... \\
    --output-dir outputs/layout_lora_html
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

from layout_sft_targets import (  # noqa: E402
    OCR_LAYOUT_PROMPT,
    build_assistant_text,
    load_sections,
    resize_max_side,
)

DEFAULT_IMAGE = (
    PROJECT_DIR
    / "data/samples/GIAY_GUI_TIEN_TIET_KIEM/IMAGE_LAYOUT_GIAY_GUI_TIEN_TIET_KIEM.png"
)
DEFAULT_LAYOUT = PROJECT_DIR / "data/samples/GIAY_GUI_TIEN_TIET_KIEM/layout _GIAY_GUI_TIEN_TIET_KIEM.json"
DEFAULT_MODEL = "unsloth/Qwen3-VL-2B-Instruct"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unsloth Qwen3-VL SFT: layout JSON → Chandra-style layout HTML.",
    )
    p.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    p.add_argument("--layout-json", type=Path, default=DEFAULT_LAYOUT)
    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    p.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs/layout_unsloth_lora")
    p.add_argument(
        "--stage",
        type=str,
        default="reasoning_html",
        choices=("reasoning_html", "html_only"),
        help="reasoning_html: Vietnamese scratchpad + HTML; html_only: HTML blocks only.",
    )
    p.add_argument("--page-width-pt", type=float, default=596.0)
    p.add_argument("--page-height-pt", type=float, default=844.0)
    p.add_argument(
        "--max-image-side",
        type=int,
        default=1024,
        help="Resize so max(w,h) <= this (Unsloth recommends ~300–1000px per side for speed).",
    )
    p.add_argument(
        "--dataset-repeats",
        type=int,
        default=32,
        help="Repeat the single conversation N times (tiny-data trick; acts like epochs multiplier).",
    )
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-seq-length", type=int, default=8192)
    p.add_argument("--lora-r", type=int, default=8, help="Small r reduces overfit on 1 sample.")
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument(
        "--finetune-vision-layers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Layout detection benefits from vision LoRA; disable to tune language/connector only.",
    )
    p.add_argument("--finetune-language-layers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--user-text-then-image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Unsloth vision docs use text then image in training messages.",
    )
    p.add_argument(
        "--export-dataset-jsonl",
        type=Path,
        default=None,
        help="If set, write one JSONL record (messages as JSON-serializable dict) and exit.",
    )
    return p.parse_args()


def messages_for_sample(
    image: Any,
    assistant_text: str,
    text_then_image: bool,
) -> list[dict[str, Any]]:
    user_text = OCR_LAYOUT_PROMPT
    if text_then_image:
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": user_text},
            {"type": "image", "image": image},
        ]
    else:
        user_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
        ]
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
    ]


def export_jsonl_serializable(messages: list[dict[str, Any]], path: Path) -> None:
    def _ser(o: Any) -> Any:
        if hasattr(o, "tolist"):  # numpy
            return o.tolist()
        if o.__class__.__name__ == "Image" or o.__class__.__name__ == "PngImageFile":
            return f"<PIL.Image size={getattr(o, 'size', '?')}>"
        return o

    def _walk(d: Any) -> Any:
        if isinstance(d, dict):
            return {k: _walk(v) for k, v in d.items()}
        if isinstance(d, list):
            return [_walk(v) for v in d]
        return _ser(d)

    path.parent.mkdir(parents=True, exist_ok=True)
    one = {"messages": _walk(messages)}
    path.write_text(json.dumps(one, ensure_ascii=False) + "\n", encoding="utf-8")


def run() -> int:
    args = parse_args()
    if not args.image.is_file():
        print(f"image not found: {args.image}", file=sys.stderr)
        return 1
    if not args.layout_json.is_file():
        print(f"layout json not found: {args.layout_json}", file=sys.stderr)
        return 1

    from PIL import Image

    sections = load_sections(args.layout_json, args.page_width_pt, args.page_height_pt)
    if not sections:
        print("No sections with layout in JSON.", file=sys.stderr)
        return 1

    assistant_text = build_assistant_text(args.stage, sections)
    pil = Image.open(args.image).convert("RGB")
    pil = resize_max_side(pil, args.max_image_side)

    messages = messages_for_sample(pil, assistant_text, args.user_text_then_image)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "model_name": args.model_name,
        "stage": args.stage,
        "image": str(args.image),
        "layout_json": str(args.layout_json),
        "n_sections": len(sections),
        "page_pt": [args.page_width_pt, args.page_height_pt],
        "dataset_repeats": args.dataset_repeats,
        "lora_r": args.lora_r,
        "user_text_then_image": args.user_text_then_image,
    }
    (args.output_dir / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "train_target_preview.txt").write_text(assistant_text, encoding="utf-8")

    if args.export_dataset_jsonl is not None:
        export_jsonl_serializable(messages, args.export_dataset_jsonl)
        print(f"Wrote dataset preview to {args.export_dataset_jsonl}")
        return 0

    try:
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastVisionModel
        from unsloth.trainer import UnslothVisionDataCollator
    except ImportError as e:
        print(
            "Missing deps. Install torch (CUDA), then:\n"
            "  pip install -r prompt4ocr/requirements_unsloth_layout.txt\n"
            "and Unsloth per https://docs.unsloth.ai/get-started/install\n"
            f"ImportError: {e}",
            file=sys.stderr,
        )
        return 2

    rows = [{"messages": messages} for _ in range(max(1, args.dataset_repeats))]
    train_ds = Dataset.from_list(rows)

    dtype = None  # let Unsloth pick (matches notebook style)
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=args.finetune_vision_layers,
        finetune_language_layers=args.finetune_language_layers,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        random_state=args.seed,
        target_modules="all-linear",
        modules_to_save=["lm_head", "embed_tokens"],
    )

    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_ds,
        args=SFTConfig(
            output_dir=str(args.output_dir),
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            logging_steps=args.logging_steps,
            optim="adamw_8bit",
            lr_scheduler_type="linear",
            seed=args.seed,
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=args.max_length,
        ),
    )
    trainer.train()
    model.save_pretrained(str(args.output_dir / "lora_adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "lora_adapter"))
    print(f"Done. LoRA saved under {args.output_dir / 'lora_adapter'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
