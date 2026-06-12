"""Merge HTML tables across pages using structural fingerprint + LCP."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

from stitch.page_layout import has_page_cut_signal

_TR_RE = re.compile(r"<tr\b[^>]*>.*?</tr>", re.IGNORECASE | re.DOTALL)
_TD_RE = re.compile(r"<td\b[^>]*>(.*?)</td>", re.IGNORECASE | re.DOTALL)
_SPAN_RE = re.compile(r"(rowspan|colspan)\s*=\s*['\"]?\d+", re.IGNORECASE)


class _TableRowParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_tr = False
        self._buf: list[str] = []
        self.rows: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() == "tr":
            self._in_tr = True
            attr_s = "".join(f" {k}='{v}'" for k, v in attrs)
            self._buf = [f"<tr{attr_s}>"]

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "tr" and self._in_tr:
            self._buf.append("</tr>")
            self.rows.append("".join(self._buf))
            self._in_tr = False
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._in_tr:
            self._buf.append(data)

    def handle_startendtag(self, tag: str, attrs) -> None:
        if self._in_tr:
            attr_s = "".join(f" {k}='{v}'" for k, v in attrs)
            self._buf.append(f"<{tag}{attr_s}/>")


@dataclass(frozen=True)
class RowFingerprint:
    n_cells: int
    n_spans: int
    cell_text_lens: tuple[int, ...]
    n_numeric_cells: int


def normalize_row(row: str) -> str:
    return re.sub(r"\s+", " ", row.strip()).lower()


def _cell_text(raw: str) -> str:
    return re.sub(r"\s+", " ", raw.strip())


def row_fingerprint(row: str) -> RowFingerprint:
    cells = [_cell_text(c) for c in _TD_RE.findall(row)]
    n_numeric = sum(
        1 for c in cells if c and re.fullmatch(r"[\d.,]+", c.replace(" ", ""))
    )
    return RowFingerprint(
        n_cells=len(cells),
        n_spans=len(_SPAN_RE.findall(row)),
        cell_text_lens=tuple(len(c) for c in cells),
        n_numeric_cells=n_numeric,
    )


def fingerprints_match(a: RowFingerprint, b: RowFingerprint) -> bool:
    return (
        a.n_cells == b.n_cells
        and a.n_spans == b.n_spans
        and a.cell_text_lens == b.cell_text_lens
    )


def rows_equivalent(left: str, right: str) -> bool:
    if normalize_row(left) == normalize_row(right):
        return True
    return fingerprints_match(row_fingerprint(left), row_fingerprint(right))


def is_header_like(row: str) -> bool:
    """Structural hint: row looks like table header, not data."""
    fp = row_fingerprint(row)
    if fp.n_spans > 0:
        return True
    if fp.n_cells >= 8 and fp.n_numeric_cells >= fp.n_cells - 2:
        return True
    return False


def extract_stt(row: str) -> int | None:
    """First-column STT for medical service tables."""
    cells = [_cell_text(c) for c in _TD_RE.findall(row)]
    if not cells:
        return None
    first = cells[0]
    if re.fullmatch(r"\d+", first):
        return int(first)
    return None


def last_data_stt(rows: list[str]) -> int | None:
    for row in reversed(rows):
        if is_header_like(row):
            continue
        stt = extract_stt(row)
        if stt is not None:
            return stt
    return None


def first_data_stt(rows: list[str]) -> int | None:
    for row in rows:
        if is_header_like(row):
            continue
        stt = extract_stt(row)
        if stt is not None:
            return stt
    return None


def stt_continues(accumulated_rows: list[str], continuation_rows: list[str]) -> bool:
    """True when the first data STT on the next part follows the last data STT."""
    prev = last_data_stt(accumulated_rows)
    nxt = first_data_stt(continuation_rows)
    if prev is None or nxt is None:
        return False
    return nxt == prev + 1


def refine_skip_with_stt(
    accumulated_rows: list[str],
    page_rows: list[str],
    skip: int,
    max_extra: int = 5,
) -> int:
    """Shift skip forward when STT indicates more repeated header rows."""
    body = page_rows[skip:]
    if stt_continues(accumulated_rows, body):
        return skip
    for extra in range(1, min(max_extra, len(page_rows) - skip)):
        trial_body = page_rows[skip + extra :]
        if stt_continues(accumulated_rows, trial_body):
            return skip + extra
    return skip


def count_leading_repeat_rows(
    reference_rows: list[str],
    page_rows: list[str],
) -> int:
    """
    Longest common prefix of rows between reference (page 1) and a continuation page.

    Only skips rows that match structurally or exactly; stops at first data row mismatch.
    """
    n = 0
    for ref, cur in zip(reference_rows, page_rows):
        if not rows_equivalent(ref, cur):
            break
        if n >= 2 and not is_header_like(cur):
            break
        n += 1
    return n


def extract_rows(html: str) -> list[str]:
    """Extract <tr>...</tr> fragments from table HTML."""
    rows = _TR_RE.findall(html)
    if rows:
        return rows
    if "<tr" in html.lower():
        parser = _TableRowParser()
        parser.feed(html)
        return parser.rows
    return []


def wrap_table(rows: list[str]) -> str:
    return "<table>" + "".join(rows) + "</table>"


def merge_tables(
    page_htmls: list[str],
    *,
    bboxes: list[str] | None = None,
    page_heights: list[float] | None = None,
) -> tuple[str, list[str]]:
    """
    Concatenate table HTML from multiple pages.

    Drops repeated leading rows on continuation pages via LCP against page 1,
    refines skip with STT when available, and records bbox page-cut signals.
    Returns (merged_html, merge_notes).
    """
    notes: list[str] = []
    if not page_htmls:
        return "", ["empty input"]

    all_parts: list[str] = []
    first_rows = extract_rows(page_htmls[0])
    if not first_rows:
        return "", ["no rows in first page"]

    all_parts.extend(first_rows)
    notes.append("page_part_0: kept %d rows (incl. header)" % len(first_rows))

    for idx, html in enumerate(page_htmls[1:], start=1):
        rows = extract_rows(html)
        if not rows:
            notes.append(f"page_part_{idx}: no rows, skipped")
            continue

        if bboxes and page_heights and idx < len(bboxes) and idx < len(page_heights):
            prev_h = page_heights[idx - 1]
            next_h = page_heights[idx]
            if prev_h and next_h:
                if has_page_cut_signal(
                    bboxes[idx - 1], bboxes[idx], prev_h, next_h
                ):
                    notes.append(f"page_part_{idx}: bbox page-cut continuation")
                else:
                    notes.append(f"page_part_{idx}: bbox weak page-cut signal")

        skip = count_leading_repeat_rows(first_rows, rows)
        refined = refine_skip_with_stt(all_parts, rows, skip)
        if refined != skip:
            notes.append(
                f"page_part_{idx}: stt adjusted skip {skip}->{refined}"
            )
            skip = refined

        body = rows[skip:]
        if body and stt_continues(all_parts, body):
            notes.append(f"page_part_{idx}: stt continuation confirmed")

        if skip:
            notes.append(
                f"page_part_{idx}: skipped {skip} repeated prefix rows, appended {len(body)} rows"
            )
        else:
            notes.append(f"page_part_{idx}: appended {len(body)} rows")
        all_parts.extend(body)

    notes.append(f"total_rows={len(all_parts)}")
    return wrap_table(all_parts), notes
