"""Field merge strategy configuration."""

from __future__ import annotations

from enum import Enum


class MergeStrategy(str, Enum):
    AS_IS = "as_is"
    FIRST_PAGE = "first_page"
    LAST_PAGE = "last_page"
    FOOTER_PER_PAGE = "footer_per_page"
    TABLE_CONCAT = "table_concat_drop_repeated_headers"
    TEXT_STITCH = "text_stitch"


# Name-based overrides for fields that cannot be inferred from type alone.
FIELD_STRATEGIES: dict[str, MergeStrategy] = {
    "THÔNG TIN CHÂN TRANG": MergeStrategy.FOOTER_PER_PAGE,
    "XÁC NHẬN": MergeStrategy.LAST_PAGE,
    "GHI CHÚ": MergeStrategy.LAST_PAGE,
    "BÁC SỸ ĐIỀU TRỊ": MergeStrategy.LAST_PAGE,
}

FIELD_ALIASES: dict[str, str] = {
    "BANG KE DICH VU": "BẢNG KÊ DỊCH VỤ",
    "Bảng kê dịch vụ": "BẢNG KÊ DỊCH VỤ",
    "THONG TIN CHAN TRANG": "THÔNG TIN CHÂN TRANG",
    "Thông tin chân trang": "THÔNG TIN CHÂN TRANG",
}


def canonical_field_name(name: str) -> str:
    return FIELD_ALIASES.get(name, name)


def resolve_strategy(
    field_name: str,
    occurrences: int,
    field_type: str = "Text",
) -> MergeStrategy:
    """Pick merge strategy for a field given how many pages it appears on."""
    canonical = canonical_field_name(field_name)
    if canonical in FIELD_STRATEGIES:
        return FIELD_STRATEGIES[canonical]
    if occurrences == 1:
        return MergeStrategy.AS_IS
    if field_type.lower() == "table":
        return MergeStrategy.TABLE_CONCAT
    return MergeStrategy.TEXT_STITCH
