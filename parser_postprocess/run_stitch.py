#!/usr/bin/env python3
"""CLI: stitch per-page label JSON into one merged document."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stitch.io import discover_samples, load_sample_pages
from stitch.merge_document import merge_sample


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge per-page parser JSON labels into one document.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/labels"),
        help="Directory with sample_XXXXXX_page_YYYY.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/merged"),
        help="Directory for merged JSON output",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Sample id (e.g. 000014). Default: all samples in input dir",
    )
    args = parser.parse_args()

    grouped = discover_samples(args.input_dir, sample_id=args.sample_id)
    if not grouped:
        print(f"No page JSON files found in {args.input_dir}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sid, paths in sorted(grouped.items()):
        sample = load_sample_pages(paths)
        doc = merge_sample(sample)
        out_path = args.output_dir / f"sample_{sid}_merged.json"
        out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        n_rows = doc["fields"].get("BẢNG KÊ DỊCH VỤ", {}).get("merge_notes", [])
        row_note = next((n for n in n_rows if n.startswith("total_rows=")), "")
        print(f"Wrote {out_path} ({doc['n_pages']} pages{', ' + row_note if row_note else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
