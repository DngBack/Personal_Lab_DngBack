"""Discover and load per-page label JSON files grouped by sample."""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PAGE_RE = re.compile(r"^sample_(\d+)_page_(\d+)\.json$")


@dataclass
class PageRecord:
    sample_id: str
    page_num: int
    path: Path
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class SamplePages:
    sample_id: str
    pages: list[PageRecord]

    @property
    def page_numbers(self) -> list[int]:
        return [p.page_num for p in self.pages]

    @property
    def n_pages(self) -> int:
        return len(self.pages)


def parse_page_filename(name: str) -> tuple[str, int] | None:
    m = _PAGE_RE.match(name)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def load_page(path: Path) -> PageRecord:
    parsed = parse_page_filename(path.name)
    if parsed is None:
        raise ValueError(f"Not a page label file: {path}")
    sample_id, page_num = parsed
    data = json.loads(path.read_text(encoding="utf-8"))
    return PageRecord(sample_id=sample_id, page_num=page_num, path=path, data=data)


def discover_samples(
    input_dir: Path,
    sample_id: str | None = None,
    limit: int | None = None,
) -> dict[str, list[Path]]:
    """Return {sample_id: [paths sorted by page]}."""
    grouped: dict[str, list[tuple[int, Path]]] = {}
    for path in sorted(input_dir.glob("*.json")):
        parsed = parse_page_filename(path.name)
        if parsed is None:
            continue
        sid, page_num = parsed
        if sample_id is not None and sid != sample_id:
            continue
        grouped.setdefault(sid, []).append((page_num, path))

    result: dict[str, list[Path]] = {}
    for sid, pages in grouped.items():
        pages.sort(key=lambda x: x[0])
        if limit is not None:
            pages = pages[:limit]
        result[sid] = [p for _, p in pages]
    return result


def load_sample_pages(paths: list[Path]) -> SamplePages:
    if not paths:
        raise ValueError("No page paths provided")

    records = [load_page(p) for p in paths]
    sample_id = records[0].sample_id
    for r in records[1:]:
        if r.sample_id != sample_id:
            raise ValueError(f"Mixed sample ids: {sample_id} vs {r.sample_id}")

    nums = {r.page_num for r in records}
    expected = set(range(min(nums), max(nums) + 1))
    missing = expected - nums
    if missing:
        warnings.warn(
            f"sample_{sample_id}: missing pages {sorted(missing)}",
            stacklevel=2,
        )

    records.sort(key=lambda r: r.page_num)
    return SamplePages(sample_id=sample_id, pages=records)
