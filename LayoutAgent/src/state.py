"""LangGraph agent state for prompt tuning."""
from __future__ import annotations

from typing import Any, TypedDict


class EvalRecord(TypedDict, total=False):
    iteration: int
    coverage: float
    text_similarity: float
    llm_judge: float
    composite_score: float
    missing_fields: list[str]
    duplicate_schemas: list[str]
    feedback: str


class AgentState(TypedDict, total=False):
    # Run identity
    run_id: str
    doc_type: str

    # Input paths (fixed for a tuning run)
    chandra_log_path: str
    chandra_prompt_path: str
    chandra_html: str
    force_chandra_rerun: bool
    layout_json_path: str
    reference_ocr_path: str
    reference_image_path: str | None
    page_image_path: str
    viz_source_path: str
    viz_page: int
    viz_dpi: int

    # Schema
    schema_fields: list[str]
    reference_ocr_flat: dict[str, str]

    # Prompt tuning
    current_prompt: str
    initial_prompt_path: str
    optimizer_prompt_path: str
    iteration: int
    max_iterations: int
    stop_threshold: float
    coverage_threshold: float   # Minimum coverage required to stop (default 0.95)

    # Model config
    # openai_model      → Qwen 3 VL 4B (schema alignment, local or API)
    # openai_judge_model → GPT-4o (visual evaluation, always OpenAI)
    # optimizer_model    → GPT-4o (prompt optimization, always OpenAI)
    openai_model: str
    openai_judge_model: str
    optimizer_model: str

    # Local HuggingFace Transformers mode
    use_local_models: bool
    chandra_device: str
    qwen_device: str

    # Outputs (per iteration)
    merged_html: str
    layout_values: dict[str, Any]
    output_dir: str

    # Evaluation
    coverage: float
    text_similarity: float
    llm_judge: float
    composite_score: float
    missing_fields: list[str]
    duplicate_schemas: list[str]
    eval_feedback: str
    eval_history: list[EvalRecord]

    # Best-so-far tracking
    best_score: float
    best_prompt: str
    best_html: str
    best_iteration: int

    # Control flow
    should_stop: bool
    stop_reason: str
    human_approved_stop: bool
