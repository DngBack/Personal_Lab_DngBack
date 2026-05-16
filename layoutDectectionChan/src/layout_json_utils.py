"""Đọc layout JSON: lấy danh sách tên trường cho LLM (DFS, không chuẩn hoá/fuzzy)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def collect_all_names(layout_root: dict | list) -> list[str]:
    """Mọi `name` trong cây layout, thứ tự DFS; trùng chuỗi y hệt chỉ giữ lần đầu."""
    out: list[str] = []

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            n = o.get("name")
            if isinstance(n, str) and n.strip():
                out.append(n.strip())
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    walk(layout_root)
    seen: set[str] = set()
    uniq: list[str] = []
    for n in out:
        if n in seen:
            continue
        seen.add(n)
        uniq.append(n)
    return uniq


def load_layout_names(layout_path: str | Path) -> list[str]:
    path = Path(layout_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    root = raw["sections"] if isinstance(raw, dict) and "sections" in raw else raw
    return collect_all_names(root)
