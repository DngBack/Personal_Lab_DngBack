from __future__ import annotations

EXTRACTION_BASE_PROMPT = """
 Extract the key-value information from the provided document which is of type: {document_type}.

FOLLOW THESE CRITICAL EXTRACTION RULES:
1. Extract all relevant information from the document, EXCEPT parts considered noise.
2. Remove noise: Do not extract logos, round stamps, handwritten signatures, watermarks, drawings, or QR codes.
3. Skip any section labeled "ĐIỀU KHOẢN CHUNG", "GENERAL TERMS", "TERMS AND CONDITIONS", or similar legal boilerplate — do NOT include it in the output.
4. Automatically detect key-value pairs from the original document. Do NOT edit or make up keys. Extract them exactly as written in the origin document.
5. Represent empty values as "".
6. Checkboxes should be represented as a boolean object grouping. E.g. "Currency": {{"VND": true, "USD": false}}.
7. If there are duplicate keys at the same hierarchy level, append _1, _2 only, or merge into one value / a short list / "FreeTextN".
8. Continuous free-text paragraphs should be merged into a single free-text string, and named "FreeText1", "FreeText2", etc.
9. If a table has a title or header (e.g., "BẢNG KÊ TIỀN MẶT"), use that EXACT title as the JSON key. The table must be represented as a list of objects. DO NOT use "Table1" if the table has a real title.
10. If a group of fields falls under a section header (e.g., "THÔNG TIN CHI TIẾT"), use that EXACT header as the JSON key, and represent its fields as a nested JSON object. DO NOT extract the header as a flat value named "Section1".
11. ONLY when tables or sections have absolutely no identifiable header, you may fall back to naming them "Table1", "Table2" or "Section1", "Section2", etc.
12. Put document heading and common header metadata under "general_info": {{ "Title": "...", "datetime": "...", "company_name": "...", "company_code": "..." }} (datetime e.g. "01/01/2026 17:01"; company e.g. branch name "CN HUNG YEN" and code "VN0010121" when present).
13. Respond with one valid JSON object only: well-formed strings (closed quotes), balanced braces, no markdown fences, no keys or text outside the document.

Respond with a single JSON object containing the extracted key-value data (the document content as structured JSON). No markdown, no extra keys except the data fields.
"""

# Backward-compatible alias for previous typo.
EXTRACTION_BASE_PROMP = EXTRACTION_BASE_PROMPT


def build_baseline_prompt(
    document_type: str,
    schema_json: str,
    ocr_input_json: str,
) -> str:
    """Build the baseline prompt from document type, schema, and OCR input.

    Args:
        document_type: Human-readable document name.
        schema_json: Target output schema serialized as JSON string.
        ocr_input_json: OCR result serialized as JSON string.

    Returns:
        Prompt text ready to send to the LLM.
    """
    rules = EXTRACTION_BASE_PROMPT.format(document_type=document_type)
    return (
        f"{rules}\n\n"
        "Target output schema (follow keys and structure as strictly as possible):\n"
        f"{schema_json}\n\n"
        "OCR input JSON:\n"
        f"{ocr_input_json}\n"
    )