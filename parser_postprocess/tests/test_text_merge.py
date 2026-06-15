"""Synthetic tests for cross-page text stitching."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from stitch.text_merge import merge_text_pages, should_stitch


class TestTextMergeSynthetic(unittest.TestCase):
    def test_hyphen_across_pages(self) -> None:
        pages = ["Chẩn đoán: suy thận mạn giai đoạn-", "5 và loãng xương."]
        out, notes = merge_text_pages(pages)
        self.assertIn("giaiđoạn5", out.replace(" ", ""))
        self.assertTrue(any("stitched" in n for n in notes))

    def test_single_page_unchanged(self) -> None:
        out, notes = merge_text_pages(["Một câu hoàn chỉnh."])
        self.assertEqual(out, "Một câu hoàn chỉnh.")
        self.assertEqual(notes, ["single occurrence"])


if __name__ == "__main__":
    unittest.main()
