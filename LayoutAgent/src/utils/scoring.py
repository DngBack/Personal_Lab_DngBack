"""Evaluation scoring utilities.

Computes quantitative metrics to assess the quality of a schema-alignment step:

- Coverage: fraction of expected schema fields that were successfully extracted.
- Composite score: weighted combination of coverage and LLM visual judge score.

These metrics drive the stopping condition and guide prompt optimization.
"""
from __future__ import annotations

from typing import Any


# Weights for the composite score formula.
# Coverage is given equal weight to prevent a generous judge score from masking
# missing fields — both dimensions must improve for the composite to rise.
_COVERAGE_WEIGHT = 0.5
_LLM_JUDGE_WEIGHT = 0.5


def compute_coverage(
    extracted_fields: list[str],
    schema_fields: list[str],
) -> tuple[float, list[str]]:
    """Compute the fraction of schema fields that were successfully extracted.

    Exact UTF-8 string matching is used (same convention as layoutDectectionChan).

    Args:
        extracted_fields: Schema field names found in the merged HTML output.
        schema_fields: Full list of expected schema field names.

    Returns:
        Tuple of (coverage_ratio, missing_fields) where coverage_ratio is in
        [0.0, 1.0] and missing_fields lists the fields not found in the output.
    """
    if not schema_fields:
        return 1.0, []

    extracted_set = set(extracted_fields)
    missing = [f for f in schema_fields if f not in extracted_set]
    coverage = (len(schema_fields) - len(missing)) / len(schema_fields)
    return coverage, missing


def compute_duplicate_schemas(merged_html: str) -> list[str]:
    """Return schema names that appear more than once in the merged HTML.

    Ideally zero — duplicates indicate the align_schema step emitted the same
    field more than once. The merge_schemas node should handle them, but this
    metric helps track whether prompt optimization is reducing the issue.

    Args:
        merged_html: HTML string produced by the alignment step.

    Returns:
        List of schema field names that have duplicate <div> entries.
    """
    import re

    schema_re = re.compile(r'data-schema\s*=\s*"([^"]*)"', re.IGNORECASE)
    counts: dict[str, int] = {}
    for m in schema_re.finditer(merged_html):
        sk = m.group(1).strip()
        if sk:
            counts[sk] = counts.get(sk, 0) + 1
    return [sk for sk, cnt in counts.items() if cnt > 1]


def compute_composite_score(
    coverage: float,
    llm_judge: float,
    *,
    coverage_weight: float = _COVERAGE_WEIGHT,
    judge_weight: float = _LLM_JUDGE_WEIGHT,
) -> float:
    """Combine coverage and LLM visual judge scores into a single metric.

    Args:
        coverage: Field coverage ratio in [0.0, 1.0].
        llm_judge: Visual quality score from the LLM judge in [0.0, 1.0].
        coverage_weight: Weight applied to the coverage term.
        judge_weight: Weight applied to the judge term.

    Returns:
        Composite score in [0.0, 1.0].
    """
    return coverage_weight * coverage + judge_weight * llm_judge


def build_eval_record(
    *,
    iteration: int,
    coverage: float,
    llm_judge: float,
    missing_fields: list[str],
    duplicate_schemas: list[str],
    feedback: str,
    composite_score: float | None = None,
) -> dict[str, Any]:
    """Construct an EvalRecord dictionary for storage in the agent state.

    Args:
        iteration: Current tuning iteration number.
        coverage: Field coverage ratio.
        llm_judge: Visual quality score from the LLM judge.
        missing_fields: Fields not found in the output.
        duplicate_schemas: Fields duplicated in the output.
        feedback: Textual feedback from the LLM judge.
        composite_score: Pre-computed composite; computed here if None.

    Returns:
        EvalRecord-compatible dictionary.
    """
    if composite_score is None:
        composite_score = compute_composite_score(coverage, llm_judge)
    return {
        "iteration": iteration,
        "coverage": coverage,
        "llm_judge": llm_judge,
        "composite_score": composite_score,
        "missing_fields": missing_fields,
        "duplicate_schemas": duplicate_schemas,
        "feedback": feedback,
    }


__all__ = [
    "compute_coverage",
    "compute_duplicate_schemas",
    "compute_composite_score",
    "build_eval_record",
]
