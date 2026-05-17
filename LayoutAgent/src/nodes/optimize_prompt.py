"""optimize_prompt node: rewrite the schema-alignment prompt using GPT-4o.

This node is the core of the prompt-tuning loop. It receives the current
schema-alignment system prompt, the evaluation feedback from the current
iteration (missing fields, scores, visual diff images), and produces an improved
prompt via GPT-4o structured output.

The response JSON format:
    {
        "improved_prompt": "<full new system prompt>",
        "changes_summary": "<bullet list of changes and rationale>"
    }

The node is placed behind a LangGraph interrupt_before breakpoint so the human
operator can review the evaluation before allowing optimization to proceed.

Required state keys:
    current_prompt (str): The prompt used in the iteration just evaluated.
    eval_feedback (str): Textual feedback from the LLM visual judge.
    missing_fields (list[str]): Fields absent from the last output.
    duplicate_schemas (list[str]): Fields duplicated in the last output.
    composite_score (float): Composite score for the last iteration.
    iteration (int): Current iteration (will be incremented here).

Optional state keys:
    reference_image_path (str | None): Ground-truth schema boxes image.
    viz_source_path (str | None): Current iteration output image.
    openai_model (str): GPT-4o model to use for optimization.
    schema_fields (list[str]): Full schema field list (for context).
    eval_history (list): Full history for multi-iteration context.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_DEFAULT_OPTIMIZER_MODEL = "gpt-4o"

_OPTIMIZER_SYSTEM = "\n".join([
    "You are an expert prompt engineer specializing in multimodal document layout extraction.",
    "",
    "Your task: rewrite a system prompt that instructs a vision-language model to map",
    "Chandra OCR HTML fragments onto schema field names for Vietnamese banking forms.",
    "",
    "The current prompt has weaknesses identified by an evaluator. Your rewrite must:",
    "1. Keep all rules that are working (do not remove useful guidance).",
    "2. Add or sharpen rules that address the identified weaknesses.",
    "3. Preserve the output format contract: HTML-only, one <div> per schema field,",
    "   data-bbox integers in [0,1000], data-label and data-schema with the exact field name.",
    "4. Be concrete: reference specific field names when they are consistently missing.",
    "5. Do NOT add hard-coded bounding boxes or values -- the model must detect them visually.",
    "6. Keep the prompt concise: avoid bloat that distracts the model from the core task.",
    "",
    "Output a single JSON object (no markdown fences):",
    "{",
    '  "improved_prompt": "<the complete, revised system prompt>",',
    '  "changes_summary": "<bullet list: what changed and why, max 8 bullets>"',
    "}",
])


def optimize_prompt_node(state: dict[str, Any]) -> dict[str, Any]:
    """Use GPT-4o to generate an improved schema-alignment prompt.

    Builds a rich context message containing the current prompt, evaluation
    metrics, feedback, and optional visual evidence (both images), then requests
    a structured JSON response with the improved prompt and a summary of changes.

    This node sits behind an interrupt_before breakpoint so the human operator
    can review the current evaluation score before allowing optimization to proceed.

    Args:
        state: Current agent state. Must have passed through at least one
               evaluate -> checkpoint cycle.

    Returns:
        Partial state update with new ``current_prompt`` and incremented ``iteration``.
    """
    from clients.openai_vision import chat_vision, make_openai_client, parse_json_response

    # Optimizer always uses GPT-4o (or OpenAI-compatible), never Qwen.
    # Prefer explicit optimizer_model key, then openai_judge_model (GPT-4o),
    # then OPTIMIZER_MODEL env, then default.
    optimizer_model = (
        state.get("optimizer_model")
        or state.get("openai_judge_model")
        or os.environ.get("OPTIMIZER_MODEL")
        or _DEFAULT_OPTIMIZER_MODEL
    )

    iteration = state.get("iteration", 0)
    print(
        f"[optimize_prompt] iter={iteration}  model={optimizer_model}  "
        f"score={state.get('composite_score', 0):.3f}",
        flush=True,
    )

    user_text = _build_optimizer_user_message(state)

    image_paths: list[str | Path] = []
    ref_path = state.get("reference_image_path")
    viz_path = state.get("viz_source_path")
    if ref_path and Path(ref_path).is_file():
        image_paths.append(ref_path)
    if viz_path and Path(viz_path).is_file():
        image_paths.append(viz_path)

    client = make_openai_client(use_qwen_endpoint=False)
    try:
        raw = chat_vision(
            client,
            model=optimizer_model,
            system_prompt=_OPTIMIZER_SYSTEM,
            user_text=user_text,
            image_paths=image_paths,
            max_tokens=4096,
            temperature=0.2,
            json_mode=True,
        )
        result = parse_json_response(raw)
        new_prompt = result.get("improved_prompt", "").strip()
        changes_summary = result.get("changes_summary", "")
    except Exception as exc:
        print(f"[optimize_prompt] Optimizer failed: {exc} -- keeping current prompt", flush=True)
        new_prompt = state.get("current_prompt", "")
        changes_summary = f"Optimization failed: {exc}"

    if not new_prompt:
        print("[optimize_prompt] Empty improved_prompt received -- keeping current", flush=True)
        new_prompt = state.get("current_prompt", "")

    print(
        f"[optimize_prompt] New prompt length={len(new_prompt)} chars",
        flush=True,
    )
    if changes_summary:
        print(f"  Changes: {changes_summary[:300]}", flush=True)

    return {
        "current_prompt": new_prompt,
        "iteration": iteration + 1,
    }


def _build_optimizer_user_message(state: dict[str, Any]) -> str:
    """Construct the user message for the prompt optimizer.

    Includes current prompt, evaluation metrics, full feedback, and historical
    scores so the optimizer can reason about multi-iteration trends.

    Args:
        state: Current agent state.

    Returns:
        Formatted user message string.
    """
    current_prompt = state.get("current_prompt", "")
    composite = state.get("composite_score", 0.0)
    coverage = state.get("coverage", 0.0)
    llm_judge = state.get("llm_judge", 0.0)
    missing = state.get("missing_fields", [])
    duplicates = state.get("duplicate_schemas", [])
    feedback = state.get("eval_feedback", "")
    schema_fields = state.get("schema_fields", [])
    eval_history = state.get("eval_history", [])

    image_note = ""
    ref_path = state.get("reference_image_path")
    viz_path = state.get("viz_source_path")
    if ref_path and Path(ref_path).is_file() and viz_path and Path(viz_path).is_file():
        image_note = (
            "Two images are attached:\n"
            "  Image 1: REFERENCE (ground truth schema boxes)\n"
            "  Image 2: OUTPUT from the current iteration\n\n"
        )
    elif viz_path and Path(viz_path).is_file():
        image_note = "One image is attached: the OUTPUT from the current iteration.\n\n"

    history_lines = []
    for rec in eval_history:
        history_lines.append(
            "  iter {}: composite={:.3f}  coverage={:.3f}  judge={:.3f}  missing={}".format(
                rec.get("iteration", "?"),
                rec.get("composite_score", 0),
                rec.get("coverage", 0),
                rec.get("llm_judge", 0),
                rec.get("missing_fields", []),
            )
        )
    history_block = "\n".join(history_lines) if history_lines else "  (none)"

    parts = [
        image_note,
        "## Current evaluation (iteration {})".format(state.get("iteration", 0)),
        "- Composite score: {:.3f}".format(composite),
        "- Coverage: {:.3f}".format(coverage),
        "- LLM judge: {:.3f}".format(llm_judge),
        "- Missing fields ({}): {}".format(len(missing), json.dumps(missing, ensure_ascii=False)),
        "- Duplicate schemas ({}): {}".format(
            len(duplicates), json.dumps(duplicates, ensure_ascii=False)
        ),
        "",
        "## Visual judge feedback",
        feedback,
        "",
        "## Score history",
        history_block,
        "",
        "## Full schema field list",
        json.dumps(schema_fields, ensure_ascii=False, indent=2),
        "",
        "## Current system prompt",
        "```",
        current_prompt,
        "```",
        "",
        "Rewrite the prompt to fix the issues above. Return JSON only.",
    ]
    return "\n".join(parts)


__all__ = ["optimize_prompt_node"]
