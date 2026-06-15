"""Heuristic text stitching for fields split across consecutive pages."""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r'[.!?…][\'")\]]?\s*$')
_HYPHEN_END = re.compile(r"-\s*$")
_HEADER_PREFIX = re.compile(
    r"^(Chẩn bệnh|Họ tên|Ngày sinh|Giường|Buồng|Ngày vào|Mã số|Bộ Y tế|Ghi chú|Người lập)",
    re.IGNORECASE,
)


def looks_incomplete(text: str) -> bool:
    t = text.rstrip()
    if _HYPHEN_END.search(t):
        return True
    last_line = t.split("\n")[-1].strip()
    if not last_line:
        return True
    if _SENTENCE_END.search(last_line):
        return False
    if re.search(r"[.!?:;]$", last_line):
        return False
    return len(last_line) > 0


def looks_continuation(text: str) -> bool:
    t = text.lstrip()
    first_line = t.split("\n")[0].strip()
    if _HEADER_PREFIX.match(first_line):
        return False
    if not first_line:
        return True
    if first_line[0].islower() or first_line[0].isdigit():
        return True
    if first_line.startswith(('"', "'", "(", "[")):
        return True
    return False


def stitch_text(a: str, b: str) -> str:
    left = a.rstrip()
    right = b.lstrip()
    if _HYPHEN_END.search(left):
        return left.rstrip("-").rstrip() + right
    if left.endswith("-") and right and right[0].islower():
        return left.rstrip("-") + right
    if left.endswith("\n") or right.startswith("\n"):
        return left + right
    return left + " " + right


def should_stitch(a: str, b: str) -> bool:
    a = a.strip()
    b = b.strip()
    if not a or not b:
        return False
    if _HYPHEN_END.search(a.rstrip()):
        return True
    return looks_incomplete(a) and looks_continuation(b)


def merge_text_pages(texts: list[str]) -> tuple[str, list[str]]:
    """Stitch consecutive page texts when heuristics agree."""
    notes: list[str] = []
    if not texts:
        return "", ["empty"]
    if len(texts) == 1:
        return texts[0], ["single occurrence"]

    merged = texts[0]
    notes.append("page_part_0: base")
    for i, nxt in enumerate(texts[1:], start=1):
        if should_stitch(merged, nxt):
            merged = stitch_text(merged, nxt)
            notes.append(f"page_part_{i}: stitched (incomplete+continuation)")
        else:
            merged = merged.rstrip() + "\n" + nxt.lstrip()
            notes.append(f"page_part_{i}: concatenated with newline")
    return merged, notes
