from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any


def encode_prompt(prompt: str) -> str:
    return base64.b64encode(prompt.encode("utf-8")).decode("ascii")


def decode_prompt(payload: dict[str, Any]) -> str:
    if "prompt_b64" in payload:
        return base64.b64decode(payload["prompt_b64"]).decode("utf-8")
    return str(payload.get("prompt") or "")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_index(suite_dir: Path) -> dict[str, Any]:
    return read_json(suite_dir / "index.json")


def load_payload(suite_dir: Path, rel_path: str) -> dict[str, Any]:
    return read_json(suite_dir / rel_path)
