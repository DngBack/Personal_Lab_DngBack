"""Golden tests for sample_000014 cross-page merge."""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from stitch.io import discover_samples, load_sample_pages
from stitch.merge_document import merge_sample
from stitch.table_merge import extract_rows, merge_tables
from stitch.text_merge import should_stitch, stitch_text

_LABELS = _ROOT / "data" / "labels"
_GOLDEN = _ROOT / "output" / "merged" / "sample_000014_merged.json"
_SAMPLE = "000014"


class TestTableMerge(unittest.TestCase):
    def test_merge_sample_014_row_count(self) -> None:
        paths = discover_samples(_LABELS, sample_id=_SAMPLE)[_SAMPLE]
        sample = load_sample_pages(paths)
        doc = merge_sample(sample)
        table_html = doc["fields"]["BẢNG KÊ DỊCH VỤ"]["content"]
        rows = extract_rows(table_html)
        self.assertEqual(len(rows), 56)
        self.assertIn("total_rows=56", doc["fields"]["BẢNG KÊ DỊCH VỤ"]["merge_notes"])

    def test_page2_starts_at_stt_5(self) -> None:
        paths = discover_samples(_LABELS, sample_id=_SAMPLE)[_SAMPLE]
        pages = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
        htmls = [p["BẢNG KÊ DỊCH VỤ"]["content"] for p in pages]
        merged, _ = merge_tables(htmls)
        rows = extract_rows(merged)
        body_after_p1 = rows[20:]
        joined = "".join(body_after_p1)
        self.assertNotEqual(joined.count(">STT<"), 2)
        r = re.search(r"<td>5</td><td>Định lượng Glucose", joined)
        self.assertIsNotNone(r)
        stt5_rows = [row for row in rows if "Định lượng Glucose máu" in row]
        self.assertIn("Định lượng Glucose máu", stt5_rows[0])

    def test_no_duplicate_date_header_mid_table(self) -> None:
        paths = discover_samples(_LABELS, sample_id=_SAMPLE)[_SAMPLE]
        pages = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
        htmls = [p["BẢNG KÊ DỊCH VỤ"]["content"] for p in pages]
        merged, _ = merge_tables(htmls)
        rows = extract_rows(merged)
        date_header_count = sum(
            1 for r in rows if "11.09" in r and "28.09" in r
        )
        self.assertEqual(date_header_count, 1)


class TestTextMerge(unittest.TestCase):
    def test_hyphen_stitch(self) -> None:
        a = "Chẩn đoán: suy thận mạn giai đoạn"
        b = "5 và loãng xương"
        self.assertTrue(should_stitch(a, b))

    def test_stitch_removes_hyphen(self) -> None:
        out = stitch_text("suy thận-", "mạn")
        self.assertEqual(out, "suy thậnmạn")


class TestGoldenSample014(unittest.TestCase):
    def test_matches_golden_structure(self) -> None:
        paths = discover_samples(_LABELS, sample_id=_SAMPLE)[_SAMPLE]
        doc = merge_sample(load_sample_pages(paths))
        self.assertEqual(doc["sample_id"], _SAMPLE)
        self.assertEqual(doc["n_pages"], 3)
        self.assertEqual(doc["page_numbers"], [1, 2, 3])
        table = doc["fields"]["BẢNG KÊ DỊCH VỤ"]
        self.assertEqual(table["type"], "Table")
        self.assertEqual(table["source_pages"], [1, 2, 3])
        footer = doc["fields"]["THÔNG TIN CHÂN TRANG"]
        self.assertIn("Trang 1:", footer["content"])
        self.assertIn("Trang 3:", footer["content"])
        self.assertIn("XÁC NHẬN", doc["fields"])

    def test_against_saved_golden_if_present(self) -> None:
        if not _GOLDEN.is_file():
            self.skipTest("golden file not present")
        paths = discover_samples(_LABELS, sample_id=_SAMPLE)[_SAMPLE]
        doc = merge_sample(load_sample_pages(paths))
        golden = json.loads(_GOLDEN.read_text(encoding="utf-8"))
        self.assertEqual(
            doc["fields"]["BẢNG KÊ DỊCH VỤ"]["merge_notes"],
            golden["fields"]["BẢNG KÊ DỊCH VỤ"]["merge_notes"],
        )
        g_rows = extract_rows(golden["fields"]["BẢNG KÊ DỊCH VỤ"]["content"])
        d_rows = extract_rows(doc["fields"]["BẢNG KÊ DỊCH VỤ"]["content"])
        self.assertEqual(len(d_rows), len(g_rows))


if __name__ == "__main__":
    unittest.main()
