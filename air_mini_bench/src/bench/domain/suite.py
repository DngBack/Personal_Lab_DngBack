"""Value objects for suite directories and generation parameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import DEFAULT_CONTEXT_SAFETY_TOKENS, DEFAULT_MAX_CONTEXT_TOKENS


@dataclass(frozen=True)
class GenerationConfig:
    """Token-length and context limits applied when building a suite."""

    length_profile: str = "heavy"
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    safety_tokens: int = DEFAULT_CONTEXT_SAFETY_TOKENS
    warmup_ratio: float | None = None  # None → dùng phase.warmup_ratio (mặc định 5%)


@dataclass(frozen=True)
class SuitePaths:
    """Standard on-disk layout for one generated scenario suite."""

    root: Path

    @property
    def index(self) -> Path:
        return self.root / "index.json"

    @property
    def trace(self) -> Path:
        return self.root / "trace.jsonl"

    @property
    def trace_meta(self) -> Path:
        return self.root / "trace_meta.json"

    @property
    def probes(self) -> Path:
        return self.root / "probes.jsonl"

    @property
    def payloads_dir(self) -> Path:
        return self.root / "payloads"

    def payload(self, request_id: str) -> Path:
        return self.payloads_dir / f"{request_id}.json"

    def exists(self) -> bool:
        return self.index.is_file() and self.trace.is_file()
