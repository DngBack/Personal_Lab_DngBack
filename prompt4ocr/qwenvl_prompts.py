"""Few-shot prompt builder for Qwen3-VL on GIẤY GỬI TIỀN TIẾT KIỆM bank slips.

We pass the model:
1. A system message describing the role and output contract.
2. A first user turn containing the SAMPLE annotated layout image + a detailed field
   guide derived from the schema layout JSON, asking for the structured JSON output.
3. The assistant's golden answer = the OCR sample JSON for that image.
4. A second user turn with the TEST image and a short reminder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "Bạn là một trợ lý trích xuất thông tin tài liệu ngân hàng tiếng Việt. "
    "Khi được cung cấp ảnh một tờ GIẤY GỬI TIỀN TIẾT KIỆM, hãy đọc kỹ cả chữ in và chữ viết "
    "tay, sau đó trả về MỘT đối tượng JSON duy nhất chứa toàn bộ key đúng theo schema mẫu. "
    "Tuân thủ nghiêm các quy tắc sau:\n"
    "  1. Giữ NGUYÊN tên key bằng tiếng Việt (có dấu) như trong schema mẫu.\n"
    "  2. Mọi key trong schema PHẢI xuất hiện trong output, dù không có giá trị thì để \"\".\n"
    "  3. Số tiền giữ nguyên dấu phẩy phân nhóm (ví dụ \"1,000,000,000\"). Không quy đổi.\n"
    "  4. Ngày giữ nguyên định dạng xuất hiện trên giấy (vd 18-10-2025 hoặc 18/10/2025).\n"
    "  5. Bỏ qua logo, dấu mộc tròn, watermark, QR. Chữ ký KHÔNG OCR mà bỏ thành \"\".\n"
    "  6. Bảng (Table2, BẢNG KÊ TIỀN MẶT) phải là LIST các object với đúng cột schema.\n"
    "  7. Tuyệt đối CHỈ trả về JSON hợp lệ, không thêm markdown, không thêm comment."
)


def _format_schema_fields(layout_json_path: Path) -> str:
    """Render a compact human-readable field list from the layout JSON."""
    with open(layout_json_path, "r", encoding="utf-8") as f:
        layout = json.load(f)
    lines: list[str] = []
    for sec in layout.get("sections", []):
        name = sec.get("name")
        items = sec.get("items") or []
        if items:
            sub_names = []
            for it in items:
                if it.get("kind") == "group":
                    for col in it.get("columns", []):
                        sub_names.append(col.get("name"))
                elif it.get("kind") == "table":
                    cols = [c.get("name") for c in it.get("columns", [])]
                    sub_names.append(f"{it.get('name')} [TABLE: {', '.join(cols)}]")
                else:
                    sub_names.append(it.get("name"))
            sub_names = [s for s in sub_names if s]
            if sub_names:
                lines.append(f"- {name}: {', '.join(sub_names)}")
                continue
        lines.append(f"- {name}")
    return "\n".join(lines)


def _expected_keys(sample_json: Any) -> str:
    """Show the JSON path hierarchy of the golden sample to make the schema explicit."""
    out: list[str] = []

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                p = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (dict, list)):
                    out.append(p)
                    walk(v, p)
                else:
                    out.append(p)
        elif isinstance(node, list):
            if node and isinstance(node[0], (dict, list)):
                walk(node[0], prefix + "[*]")

    walk(sample_json, "")
    return "\n".join(f"  - {p}" for p in out)


def build_few_shot_messages(
    *,
    sample_image_path: Path,
    sample_json_path: Path,
    layout_json_path: Path,
    test_image: Any,
    document_type: str = "GIẤY GỬI TIỀN TIẾT KIỆM",
) -> list[dict[str, Any]]:
    """Return Qwen3-VL chat messages with one annotated few-shot pair + test image.

    ``test_image`` is a PIL.Image (so the test PDF page can be rendered in-memory).
    """
    with open(sample_json_path, "r", encoding="utf-8") as f:
        sample_json = json.load(f)
    sample_json_text = json.dumps(sample_json, ensure_ascii=False, indent=2)

    field_guide = _format_schema_fields(layout_json_path)
    key_paths = _expected_keys(sample_json)

    user_intro = (
        f"Đây là một ví dụ tham khảo cho loại tài liệu **{document_type}**.\n\n"
        "Hình minh hoạ kèm theo có vẽ bbox màu mô tả VÙNG CỦA TỪNG TRƯỜNG để bạn hiểu "
        "cấu trúc form. Khi xử lý tài liệu thật, ảnh sẽ KHÔNG có các bbox này — bạn "
        "phải tự định vị các trường dựa vào nhãn (label) in trên giấy.\n\n"
        "DANH SÁCH TRƯỜNG CẦN TRÍCH (từ schema):\n"
        f"{field_guide}\n\n"
        "CẤU TRÚC JSON ĐẦY ĐỦ (đường dẫn key kỳ vọng):\n"
        f"{key_paths}\n\n"
        "Hãy đọc ảnh sau và xuất ra JSON đúng theo schema trên."
    )

    assistant_reply = sample_json_text

    user_test = (
        "Bây giờ hãy trích xuất JSON theo CÙNG schema từ ảnh tài liệu sau. "
        "Trả về duy nhất một đối tượng JSON, không markdown, không chú thích thêm. "
        "Nếu một trường không xuất hiện trên giấy, hãy để giá trị là \"\"."
    )

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_intro},
                {"type": "image", "image": str(sample_image_path)},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_reply}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_test},
                {"type": "image", "image": test_image},
            ],
        },
    ]
