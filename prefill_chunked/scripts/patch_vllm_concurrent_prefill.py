#!/usr/bin/env python3
"""Enable concurrent partial prefill on vLLM 0.23 without replacing whole modules."""

from __future__ import annotations

import importlib.util
import sys
import urllib.error
import urllib.request
from pathlib import Path

VLLM_TAG = "v0.23.0"
TAG_BASE = f"https://raw.githubusercontent.com/vllm-project/vllm/{VLLM_TAG}"
SCHEDULER_CONFIG = "vllm/config/scheduler.py"
SCHEDULER_CORE = "vllm/v1/core/sched/scheduler.py"
ARG_UTILS_BLOCK = """        # No Concurrent Partial Prefills so far.
        if (
            self.max_num_partial_prefills != SchedulerConfig.max_num_partial_prefills
            or self.max_long_partial_prefills
            != SchedulerConfig.max_long_partial_prefills
        ):
            _raise_unsupported_error(feature_name="Concurrent Partial Prefill")

"""


def vllm_package_root() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("vllm is not installed in this environment")
    return Path(spec.origin).resolve().parent


def fetch_text(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            return response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to download {url}: {exc}") from exc


def install_relative(vllm_root: Path, relative_path: str, content: str) -> Path:
    prefix = "vllm/"
    if not relative_path.startswith(prefix):
        raise ValueError(f"unexpected patch path: {relative_path}")
    target = vllm_root / relative_path[len(prefix) :]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def patch_arg_utils(vllm_root: Path) -> None:
    path = vllm_root / "engine" / "arg_utils.py"
    original = path.read_text(encoding="utf-8")
    if ARG_UTILS_BLOCK not in original:
        if "Concurrent Partial Prefill" not in original:
            print(f"[patch] arg_utils already patched: {path}", flush=True)
            return
        raise RuntimeError(f"unexpected arg_utils layout: {path}")
    path.write_text(original.replace(ARG_UTILS_BLOCK, ""), encoding="utf-8")
    print(f"[patch] removed concurrent-prefill startup guard: {path}", flush=True)


def restore_scheduler_config(vllm_root: Path) -> None:
    content = fetch_text(f"{TAG_BASE}/{SCHEDULER_CONFIG}")
    target = install_relative(vllm_root, SCHEDULER_CONFIG, content)
    print(f"[patch] restored {SCHEDULER_CONFIG} from {VLLM_TAG}: {target}", flush=True)


def restore_scheduler_core(vllm_root: Path) -> Path:
    content = fetch_text(f"{TAG_BASE}/{SCHEDULER_CORE}")
    return install_relative(vllm_root, SCHEDULER_CORE, content)


def apply_scheduler_patch(vllm_root: Path) -> None:
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from scheduler_concurrent_prefill_v023 import apply_scheduler_patch as apply_v023

    scheduler_path = restore_scheduler_core(vllm_root)
    apply_v023(scheduler_path)


def main() -> int:
    if sys.argv[1:] == ["--check"]:
        root = vllm_package_root()
        arg_utils = (root / "engine" / "arg_utils.py").read_text(encoding="utf-8")
        scheduler = (root / "v1" / "core" / "sched" / "scheduler.py").read_text(
            encoding="utf-8"
        )
        config = (root / "config" / "scheduler.py").read_text(encoding="utf-8")
        enabled = (
            "Concurrent Partial Prefill" not in arg_utils
            and "scheduler_reserve_full_isl" in config
            and (
                "enable_concurrent_partial_prefill_scheduling" in scheduler
                or "enable_short_prefill_priority" in scheduler
            )
        )
        print("patched" if enabled else "not_patched")
        return 0 if enabled else 1

    root = vllm_package_root()
    restore_scheduler_config(root)
    patch_arg_utils(root)
    apply_scheduler_patch(root)
    print("[patch] concurrent partial prefill enabled", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
