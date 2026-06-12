"""Tests for type-based merge strategy resolution."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from stitch.config import MergeStrategy, resolve_strategy
from stitch.page_layout import has_page_cut_signal, infer_page_height, parse_bbox
from stitch.table_merge import extract_stt, refine_skip_with_stt, stt_continues


class TestResolveStrategy(unittest.TestCase):
    def test_table_type_uses_concat_without_field_name(self) -> None:
        strategy = resolve_strategy("ANY_TABLE_FIELD", 3, field_type="Table")
        self.assertEqual(strategy, MergeStrategy.TABLE_CONCAT)

    def test_footer_still_name_based(self) -> None:
        strategy = resolve_strategy("THÔNG TIN CHÂN TRANG", 3, field_type="Section")
        self.assertEqual(strategy, MergeStrategy.FOOTER_PER_PAGE)

    def test_single_page_as_is(self) -> None:
        strategy = resolve_strategy("Chẩn bệnh", 1, field_type="Section")
        self.assertEqual(strategy, MergeStrategy.AS_IS)

    def test_multi_page_text_stitch(self) -> None:
        strategy = resolve_strategy("Mô tả", 2, field_type="Text")
        self.assertEqual(strategy, MergeStrategy.TEXT_STITCH)


class TestPageLayout(unittest.TestCase):
    def test_parse_bbox(self) -> None:
        self.assertEqual(parse_bbox("64 92 446 221"), (64.0, 92.0, 446.0, 221.0))

    def test_infer_page_height(self) -> None:
        fields = {"a": {"bbox": "0 0 100 1134"}, "b": {"bbox": "0 0 200 900"}}
        self.assertEqual(infer_page_height(fields), 1134.0)

    def test_page_cut_signal(self) -> None:
        self.assertTrue(
            has_page_cut_signal("64 316 1620 1053", "64 92 1620 1053", 1134.0, 1134.0)
        )
        self.assertFalse(
            has_page_cut_signal("64 92 446 221", "64 260 927 309", 1134.0, 1134.0)
        )


class TestSttHelpers(unittest.TestCase):
    def test_extract_stt(self) -> None:
        row = "<tr><td>5</td><td>Định lượng Glucose máu</td></tr>"
        self.assertEqual(extract_stt(row), 5)

    def test_stt_continues(self) -> None:
        prev = ["<tr><td>4</td><td>x</td></tr>"]
        nxt = ["<tr><td>5</td><td>y</td></tr>"]
        self.assertTrue(stt_continues(prev, nxt))

    def test_refine_skip_with_stt(self) -> None:
        accumulated = ["<tr><td>4</td><td>prev</td></tr>"]
        page_rows = [
            "<tr><td>1</td><td>spurious restart</td></tr>",
            "<tr><td>5</td><td>next</td></tr>",
        ]
        self.assertEqual(refine_skip_with_stt(accumulated, page_rows, 0), 1)


if __name__ == "__main__":
    unittest.main()
