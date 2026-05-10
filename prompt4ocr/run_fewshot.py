from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.hf_env import ensure_writable_huggingface_cache

ensure_writable_huggingface_cache()

from infer.vllm_client import VllmGenerationConfig, VllmTextGenerator
from prompt.fewshot import build_fewshot_prompt
from utils.io import (
    as_pretty_json,
    dump_json,
    extract_first_json_object,
    read_json,
    timestamp_slug,
)

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_DOC_TYPE = "GIAY_GUI_TIEN_TIET_KIEM"
DEFAULT_SCHEMA_PATH = PROJECT_DIR / "prompt/schema/GIAY_GUI_TIEN_TIET_KIEM.json"
DEFAULT_SAMPLE_OCR_PATH = (
    PROJECT_DIR / "data/samples/GIAY_GUI_TIEN_TIET_KIEM/OCR_GIAY_GUI_TIEN_TIET_KIEM.json"
)
DEFAULT_INPUT_OCR_PATH = (
    PROJECT_DIR / "data/samples/GIAY_GUI_TIEN_TIET_KIEM/OCR_GIAY_GUI_TIEN_TIET_KIEM.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results"
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for few-shot inference script."""
    parser = argparse.ArgumentParser(
        description="Run few-shot OCR extraction prompt with vLLM."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name/path.")
    parser.add_argument("--document-type", default=DEFAULT_DOC_TYPE, help="Document type.")
    parser.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH), help="Schema JSON path.")
    parser.add_argument(
        "--sample-ocr-path",
        default=str(DEFAULT_SAMPLE_OCR_PATH),
        help="Reference OCR JSON used for few-shot context.",
    )
    parser.add_argument(
        "--input-ocr-path", default=str(DEFAULT_INPUT_OCR_PATH), help="OCR input JSON path."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling.")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max generated tokens.")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help="Max context length allocated in vLLM.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help="Fraction of GPU memory reserved by vLLM.",
    )
    return parser.parse_args()


def run_fewshot(args: argparse.Namespace) -> dict[str, Any]:
    """Build prompt, run vLLM generation, and parse the resulting JSON."""
    schema = read_json(args.schema_path)
    sample_ocr = read_json(args.sample_ocr_path)
    input_ocr = read_json(args.input_ocr_path)
    prompt = build_fewshot_prompt(
        document_type=args.document_type,
        schema_json=as_pretty_json(schema),
        sample_ocr_json=as_pretty_json(sample_ocr),
        input_ocr_json=as_pretty_json(input_ocr),
    )

    generator = VllmTextGenerator(
        model_name=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    config = VllmGenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    raw_text = generator.generate_text(prompt=prompt, config=config)
    parsed = extract_first_json_object(raw_text)
    return {"parsed_output": parsed, "raw_model_text": raw_text}


def main() -> None:
    """CLI entrypoint for few-shot experiment."""
    args = parse_args()
    result = run_fewshot(args)

    output_path = Path(args.output_dir) / f"fewshot_{timestamp_slug()}.json"
    dump_json(result, output_path)
    print(f"Saved few-shot result to: {output_path}")


if __name__ == "__main__":
    main()
