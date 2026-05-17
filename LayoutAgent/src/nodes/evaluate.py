"""evaluate node: assess the quality of the current schema-alignment output.

Computes two complementary metrics:

1. **Coverage** — the fraction of expected schema fields that appear in the
   merged HTML with a non-empty ``data-schema`` attribute. Computed locally
   without any API call.

2. **LLM visual judge** — GPT-4o receives the reference ground-truth image and
   the current iteration's schema-boxes image, then returns a score in [0, 1]
   and structured textual feedback about missing, mis-aligned, or extra regions.

A composite score is derived from these two metrics and stored in the state so
that the graph's routing function can decide whether to continue tuning.

Required state keys:
    merged_html (str): Deduplicated schema-aligned HTML.
    schema_fields (list[str]): Full list of expected schema field names.
    iteration (int): Current iteration number.
    output_dir (str): Root run directory.
    viz_source_path (str): Path to the current visualization JPEG.

Optional state keys:
    reference_image_path (str | None): Ground-truth schema boxes JPEG.
    openai_judge_model (str): Model for visual evaluation (default: "gpt-4o").
    eval_history (list): Accumulated evaluation records.
    best_score (float): Best composite score seen so far.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_DEFAULT_JUDGE_MODEL = "gpt-4o"

_JUDGE_SYSTEM = """\
You are a strict layout quality evaluator for Vietnamese banking form OCR.

You will receive two images:
1. The REFERENCE image — ground truth with correctly labelled schema bounding boxes.
2. The OUTPUT image — the current extraction attempt to evaluate.

Compare the two images carefully and assess:
- Which schema fields are MISSING in the output (visible in reference but have no box in output)?
- Which fields have WRONG or misaligned bounding boxes (box is in wrong position, wrong size, or covers wrong region)?
- Are there EXTRA/spurious boxes in the output that are not in the reference?
- For TABLE fields: does the output draw individual boxes per column/row as in the reference, or does it collapse them into one big box?

Scoring rules (be strict):
- Start at 1.0.
- Deduct 0.05 per missing field (field present in reference, absent from output).
- Deduct 0.03 per misaligned field (box exists but position/size is significantly wrong).
- Deduct 0.02 per extra/spurious field.
- Deduct 0.05 if table sub-fields (individual columns/rows) are collapsed into a single box instead of separate boxes.
- Floor at 0.0.

Respond with a JSON object exactly like this:
{
  "score": <float in [0.0, 1.0]>,
  "missing_fields_visual": [<field names as strings>],
  "misaligned_fields": [<field names as strings>],
  "extra_fields": [<field names as strings>],
  "feedback": "<one short paragraph of specific, actionable feedback: what fields are wrong and what the prompt should do differently>"
}

score = 1.0 means perfect match. Apply the deduction rules above precisely.
"""


def evaluate_node(state: dict[str, Any]) -> dict[str, Any]:
    """Compute coverage and LLM visual judge score for the current iteration.

    Reads schema field names extracted from the merged HTML, computes coverage
    against the full schema field list, then calls GPT-4o with both the
    reference and output images to get a visual quality score and feedback.

    Args:
        state: Current agent state.

    Returns:
        Partial state update with evaluation metrics, feedback, and best tracking.
    """
    from utils.schema_html import get_extracted_schema_names
    from utils.scoring import (
        build_eval_record,
        compute_composite_score,
        compute_coverage,
        compute_duplicate_schemas,
    )

    iteration = state.get("iteration", 0)
    schema_fields: list[str] = state.get("schema_fields", [])
    merged_html: str = state.get("merged_html", "")

    extracted = get_extracted_schema_names(merged_html)
    coverage, missing_fields = compute_coverage(extracted, schema_fields)
    duplicate_schemas = compute_duplicate_schemas(merged_html)

    print(
        f"[evaluate] iter={iteration}  coverage={coverage:.3f}  "
        f"extracted={len(extracted)}/{len(schema_fields)}  "
        f"missing={len(missing_fields)}  duplicates={len(duplicate_schemas)}",
        flush=True,
    )

    llm_judge, judge_feedback, visual_details = _run_llm_judge(state)

    composite = compute_composite_score(coverage, llm_judge)
    print(
        f"[evaluate] llm_judge={llm_judge:.3f}  composite={composite:.3f}",
        flush=True,
    )

    record = build_eval_record(
        iteration=iteration,
        coverage=coverage,
        llm_judge=llm_judge,
        missing_fields=missing_fields,
        duplicate_schemas=duplicate_schemas,
        feedback=judge_feedback,
        composite_score=composite,
    )
    record.update(visual_details)

    eval_history: list[dict] = list(state.get("eval_history") or [])
    eval_history.append(record)

    best_score = state.get("best_score", -1.0)
    best_prompt = state.get("best_prompt", state.get("current_prompt", ""))
    best_html = state.get("best_html", "")
    best_iteration = state.get("best_iteration", 0)

    if composite > best_score:
        best_score = composite
        best_prompt = state.get("current_prompt", "")
        best_html = merged_html
        best_iteration = iteration
        print(f"[evaluate] New best score: {best_score:.3f}", flush=True)

    return {
        "coverage": coverage,
        "llm_judge": llm_judge,
        "composite_score": composite,
        "missing_fields": missing_fields,
        "duplicate_schemas": duplicate_schemas,
        "eval_feedback": judge_feedback,
        "eval_history": eval_history,
        "best_score": best_score,
        "best_prompt": best_prompt,
        "best_html": best_html,
        "best_iteration": best_iteration,
    }


# ---------------------------------------------------------------------------
# LLM visual judge helpers
# ---------------------------------------------------------------------------

def _run_llm_judge(
    state: dict[str, Any],
) -> tuple[float, str, dict[str, Any]]:
    """Call the LLM visual judge and parse its structured response.

    Returns:
        Tuple of (score, feedback_text, extra_detail_dict).
    """
    from clients.openai_vision import chat_vision, make_openai_client, parse_json_response

    judge_model = (
        state.get("openai_judge_model")
        or os.environ.get("JUDGE_MODEL")
        or _DEFAULT_JUDGE_MODEL
    )

    viz_path = state.get("viz_source_path")
    ref_path = state.get("reference_image_path")

    if not viz_path or not Path(viz_path).is_file():
        print("[evaluate] No visualization image found — skipping visual judge", flush=True)
        return 0.5, "No visualization available for visual comparison.", {}

    image_paths: list[str | Path] = []
    if ref_path and Path(ref_path).is_file():
        image_paths.append(ref_path)
        image_paths.append(viz_path)
        user_text = (
            "Image 1 is the REFERENCE (ground truth). "
            "Image 2 is the OUTPUT to evaluate. "
            "Compare them and return the JSON evaluation."
        )
    else:
        image_paths.append(viz_path)
        user_text = (
            "No reference image is available. Evaluate the OUTPUT image alone: "
            "check that schema boxes are reasonable, cover the correct regions, "
            "and do not overlap excessively. Return the JSON evaluation."
        )

    missing_hint = ""
    missing = state.get("missing_fields", [])
    if missing:
        missing_hint = (
            f"\n\nNote: coverage analysis already found these fields missing from the HTML: "
            f"{missing}. Check if they are also absent visually."
        )

    client = make_openai_client(use_qwen_endpoint=False)
    try:
        raw = chat_vision(
            client,
            model=judge_model,
            system_prompt=_JUDGE_SYSTEM,
            user_text=user_text + missing_hint,
            image_paths=image_paths,
            max_tokens=1024,
            temperature=0.0,
            json_mode=True,
        )
        result = parse_json_response(raw)
    except Exception as exc:
        print(f"[evaluate] LLM judge failed: {exc}", flush=True)
        return 0.5, f"Judge error: {exc}", {}

    score = float(result.get("score", 0.5))
    score = max(0.0, min(1.0, score))
    feedback = str(result.get("feedback", ""))
    extra = {
        "missing_fields_visual": result.get("missing_fields_visual", []),
        "misaligned_fields": result.get("misaligned_fields", []),
        "extra_fields": result.get("extra_fields", []),
    }
    return score, feedback, extra


__all__ = ["evaluate_node"]
