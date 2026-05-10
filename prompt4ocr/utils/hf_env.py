"""Hugging Face cache env: avoid broken or unwritable global cache paths."""

from __future__ import annotations

import os
from pathlib import Path


def _is_dir_writable(dir_path: Path) -> bool:
    """Return True if we can create the dir and write a small file inside."""
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        probe = dir_path / ".prompt4ocr_hf_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def resolved_hf_home() -> Path:
    """Resolve the directory Hugging Face would use as HF_HOME (before our override)."""
    explicit = os.environ.get("HF_HOME")
    if explicit:
        return Path(explicit).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "huggingface"
    return Path.home() / ".cache" / "huggingface"


def ensure_writable_huggingface_cache() -> Path:
    """If the effective HF cache root is not writable, point caches under ~/.cache.

    Call this before importing ``transformers`` / ``vllm`` so downloads use a
    safe location when e.g. ``XDG_CACHE_HOME`` points to a read-only mount.

    Returns:
        The HF_HOME path that will be used (existing or overridden).
    """
    base = resolved_hf_home()
    hub_override = os.environ.get("HUGGINGFACE_HUB_CACHE")
    hub_path = Path(hub_override).expanduser() if hub_override else base / "hub"

    if _is_dir_writable(base) and _is_dir_writable(hub_path):
        return base

    safe = Path.home() / ".cache" / "huggingface"
    safe.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(safe)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(safe / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(safe / "transformers")
    return safe
