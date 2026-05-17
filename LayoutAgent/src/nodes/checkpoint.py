"""checkpoint node: persist artifacts for the current iteration.

After each evaluation, this node writes the following files to the iteration
sub-directory so that results can be inspected, diffed, or replayed:

    iter_NN/
        prompt.txt          — the schema-alignment prompt used this iteration
        schema_merged.html  — deduplicated schema-aligned HTML
        eval.json           — evaluation metrics and feedback
        layout_values.json  — enriched layout tree with extracted field values

Additionally, the run-level summary.json is updated so that the overall tuning
progress can be monitored without opening individual iteration directories.
If this iteration produced the best composite score, its prompt and HTML are
also written as best_prompt.txt and best_schema_merged.html at the run level.

Required state keys:
    output_dir (str): Root run directory.
    iteration (int): Current iteration number.
    current_prompt (str): Schema-alignment prompt used this iteration.
    merged_html (str): Deduplicated HTML from the merge node.
    layout_json_path (str): Path to the layout JSON for field enrichment.
    eval_history (list): Accumulated EvalRecord list.

Optional state keys:
    run_id (str): Identifier for log messages.
    best_score (float): Best composite score so far.
    best_iteration (int): Iteration that produced the best score.
    best_prompt (str): Best prompt so far.
    best_html (str): Best merged HTML so far.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def checkpoint_node(state: dict[str, Any]) -> dict[str, Any]:
    """Write per-iteration artifacts and update the run summary.

    Args:
        state: Current agent state.

    Returns:
        Empty dict — checkpoint is a side-effect-only node.
    """
    from utils.schema_html import build_layout_payload
    from nodes.visualize import _iter_dir

    iteration = state.get("iteration", 0)
    output_dir = Path(state["output_dir"])
    iter_dir = _iter_dir(str(output_dir), iteration)
    iter_dir.mkdir(parents=True, exist_ok=True)

    # --- prompt.txt ---
    prompt = state.get("current_prompt", "")
    (iter_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    # --- schema_merged.html ---
    merged_html = state.get("merged_html", "")
    (iter_dir / "schema_merged.html").write_text(merged_html, encoding="utf-8")

    # --- eval.json ---
    eval_history: list[dict] = state.get("eval_history") or []
    current_eval = eval_history[-1] if eval_history else {}
    (iter_dir / "eval.json").write_text(
        json.dumps(current_eval, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --- layout_values.json ---
    layout_path = state.get("layout_json_path")
    if layout_path and merged_html:
        try:
            payload = build_layout_payload(layout_path, merged_html)
            payload["meta"] = {
                "run_id": state.get("run_id", ""),
                "iteration": iteration,
                "schema_merge_mode": "openai_vision",
            }
            (iter_dir / "layout_values.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"[checkpoint] Could not write layout_values.json: {exc}", flush=True)

    # --- best artifacts at run level ---
    best_iteration = state.get("best_iteration", 0)
    if best_iteration == iteration:
        best_prompt = state.get("best_prompt", "")
        best_html = state.get("best_html", "")
        if best_prompt:
            (output_dir / "best_prompt.txt").write_text(best_prompt, encoding="utf-8")
        if best_html:
            (output_dir / "best_schema_merged.html").write_text(best_html, encoding="utf-8")

    # --- summary.json ---
    summary = {
        "run_id": state.get("run_id", ""),
        "best_score": state.get("best_score", 0.0),
        "best_iteration": best_iteration,
        "stop_reason": state.get("stop_reason"),
        "eval_history": eval_history,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"[checkpoint] iter={iteration}  "
        f"score={current_eval.get('composite_score', 0):.3f}  "
        f"best={state.get('best_score', 0):.3f}  →  {iter_dir}",
        flush=True,
    )
    return {}


__all__ = ["checkpoint_node"]
