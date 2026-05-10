from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    """Read and deserialize JSON from a file path."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(data: Any, path: str | Path) -> None:
    """Serialize and write JSON data to a file path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def as_pretty_json(data: Any) -> str:
    """Serialize data into pretty JSON text for prompt embedding."""
    return json.dumps(data, ensure_ascii=False, indent=2)


def timestamp_slug() -> str:
    """Create a filesystem-friendly timestamp slug."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def extract_first_json_object(text: str) -> Any:
    """Extract and parse the first valid JSON object from arbitrary text.

    The model may prepend or append notes around JSON. This parser scans
    candidate object spans and returns the first parseable JSON object.
    """
    decoder = json.JSONDecoder()
    for start_idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start_idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("No valid JSON object found in model output.")
