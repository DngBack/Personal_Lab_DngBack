#!/usr/bin/env python3
"""Fix a one-line signature bug in stock vLLM 0.24.0 that makes
--gdn-prefill-backend=flashqla crash on EVERY real request (confirmed
2026-07-13 on r11-rustwarm-v1: EngineCore dies with TypeError on the very
first prefill).

Root cause (source-read of qwen_gdn_linear_attn.py, ChunkGatedDeltaRule):
forward_cuda and forward_native both accept `g_already_exp: bool = False`;
forward_flashqla is missing that parameter. The caller
(_forward_core/_forward_core_decode_aiter) always passes
`g_already_exp=(self.gdn_prefill_backend == "flashinfer")` to whichever
_forward_method was selected -- when flashqla is active this becomes
`g_already_exp=False` passed into a function that doesn't declare it ->
TypeError: ChunkGatedDeltaRule.forward_flashqla() got an unexpected keyword
argument 'g_already_exp'.

Fix: add the same `g_already_exp: bool = False` parameter to
forward_flashqla's signature (accepted, unused -- flashqla's own
air_chunk_gated_delta_rule call doesn't take it, matching that this flag is
only meaningful for the flashinfer path per the existing
`g_already_exp=(self.gdn_prefill_backend == "flashinfer")` caller logic).
Purely additive: no other backend's behavior changes.
"""
from __future__ import annotations

import argparse
from pathlib import Path


PATCH_MARKER = "# R_FLASHQLA_SIG_FIX_PATCH"


class PatchError(RuntimeError):
    pass


def find_vllm_root(explicit_root: str | None) -> Path:
    if explicit_root:
        root = Path(explicit_root)
        return root if (root / "vllm").is_dir() else root.parent
    import importlib.util

    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise PatchError("Unable to locate installed vllm package")
    return Path(spec.origin).resolve().parent.parent


def replace_once(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise PatchError(
            f"Expected exactly one anchor for {description}, found {count}."
        )
    return text.replace(old, new, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("vllm_root", nargs="?", default=None)
    args = parser.parse_args()

    vllm_root = find_vllm_root(args.vllm_root)
    target = (
        vllm_root
        / "vllm"
        / "model_executor"
        / "layers"
        / "mamba"
        / "gdn"
        / "qwen_gdn_linear_attn.py"
    )
    if not target.exists():
        raise PatchError(f"Missing target file: {target}")

    text = target.read_text()
    if PATCH_MARKER in text:
        print(f"already patched: {target}")
        return 0

    anchor = """        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
    ):
        o, final_state = air_chunk_gated_delta_rule("""

    replacement = f"""        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
        g_already_exp: bool = False,  {PATCH_MARKER}: matches forward_cuda/
        # forward_native's signature; unused here (flashqla's own kernel call
        # below doesn't take it), only meaningful for the flashinfer path.
    ):
        o, final_state = air_chunk_gated_delta_rule("""

    text = replace_once(text, anchor, replacement, "forward_flashqla signature")
    target.write_text(text)

    import py_compile

    py_compile.compile(str(target), doraise=True)
    print(f"patched {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
