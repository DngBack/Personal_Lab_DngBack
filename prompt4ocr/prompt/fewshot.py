from __future__ import annotations

from prompt.baseline import EXTRACTION_BASE_PROMPT


def build_fewshot_prompt(
    document_type: str,
    schema_json: str,
    sample_ocr_json: str,
    input_ocr_json: str,
) -> str:
    """Build a prompt with sample OCR context as few-shot guidance.

    Args:
        document_type: Human-readable document name.
        schema_json: Target output schema serialized as JSON string.
        sample_ocr_json: OCR sample JSON used as in-context reference.
        input_ocr_json: OCR input for the current inference serialized as JSON string.

    Returns:
        Prompt text containing shared rules, one example, and the final query.
    """
    rules = EXTRACTION_BASE_PROMPT.format(document_type=document_type)
    return (
        f"{rules}\n\n"
        "Target output schema (follow keys and structure as strictly as possible):\n"
        f"{schema_json}\n\n"
        "Few-shot reference context (same document type, no answer shown):\n"
        "Reference OCR JSON:\n"
        f"{sample_ocr_json}\n\n"
        "Now extract from this OCR input JSON following the schema:\n"
        f"{input_ocr_json}\n"
    )
