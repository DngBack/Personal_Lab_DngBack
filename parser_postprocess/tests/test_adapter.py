"""Tests for model JSON adapter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from stitch.adapters.model_json import from_chandra_blocks, normalize_page_dict


class TestModelAdapter(unittest.TestCase):
    def test_normalize_canonical(self) -> None:
        page = {
            "Bảng kê dịch vụ": {
                "content": "<table></table>",
                "bbox": "1 2 3 4",
                "type": "Table",
            }
        }
        out = normalize_page_dict(page)
        self.assertIn("BẢNG KÊ DỊCH VỤ", out)

    def test_chandra_blocks(self) -> None:
        doc = {
            "blocks": [
                {
                    "label": "Table",
                    "text": "<table><tr></tr></table>",
                    "bbox": (0, 0, 100, 100),
                },
                {
                    "label": "Page-Footer",
                    "text": "Trang 1/3",
                    "bbox": (0, 900, 100, 950),
                },
            ]
        }
        out = from_chandra_blocks(doc)
        self.assertIn("BẢNG KÊ DỊCH VỤ", out)
        self.assertIn("THÔNG TIN CHÂN TRANG", out)


if __name__ == "__main__":
    unittest.main()
