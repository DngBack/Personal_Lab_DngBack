"""Setup node: initialize a tuning run.

Responsibilities:
- Load schema field names from the layout JSON using exact DFS traversal.
- Create the output directory for this run: data/tuning_runs/<run_id>/
- Load the initial schema-alignment prompt from file.
- Record paths to reference files (image, layout JSON).
- Initialize iteration counter and best-score tracking.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


def setup_node(state: dict[str, Any]) -> dict[str, Any]:
    """Initialize agent state for a new tuning run.

    Reads the layout JSON to extract the ordered list of schema field names,
    creates the run output directory, and loads the initial prompt text. The
    iteration counter is set to 0 and best-score tracking is reset.

    Required state keys:
        layout_json_path (str): Path to the layout JSON file.
        initial_prompt_path (str): Path to the initial schema-alignment prompt.

    Optional state keys (populated with defaults if missing):
        run_id (str): Identifier for this run; auto-generated if absent.
        output_dir (str): Root directory for run artifacts; auto-generated if absent.
        max_iterations (int): Maximum tuning iterations (default 3).
        stop_threshold (float): Score threshold to stop early (default 0.85).
        reference_image_path (str | None): Path to the ground-truth schema boxes JPEG.

    Returns:
        Partial state update dictionary.
    """
    from utils.schema_html import load_layout_names

    layout_path = Path(state["layout_json_path"])
    if not layout_path.is_file():
        raise FileNotFoundError(f"Layout JSON not found: {layout_path}")

    schema_fields = load_layout_names(layout_path)
    print(f"[setup] Loaded {len(schema_fields)} schema fields from {layout_path.name}", flush=True)

    run_id = state.get("run_id") or _generate_run_id()
    output_dir = Path(
        state.get("output_dir")
        or Path(state["layout_json_path"]).parent.parent.parent
        / "data"
        / "tuning_runs"
        / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = Path(state["initial_prompt_path"])
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Initial prompt not found: {prompt_path}")
    current_prompt = prompt_path.read_text(encoding="utf-8")
    print(f"[setup] Loaded initial prompt from {prompt_path.name} ({len(current_prompt)} chars)", flush=True)

    print(f"[setup] Run ID: {run_id}  →  {output_dir}", flush=True)

    return {
        "run_id": run_id,
        "schema_fields": schema_fields,
        "output_dir": str(output_dir),
        "current_prompt": current_prompt,
        "initial_prompt_path": str(prompt_path),
        "iteration": 0,
        "max_iterations": state.get("max_iterations", 3),
        "stop_threshold": state.get("stop_threshold", 0.85),
        "best_score": -1.0,
        "best_prompt": current_prompt,
        "best_html": "",
        "best_iteration": 0,
        "eval_history": [],
        "should_stop": False,
        "stop_reason": "",
        "composite_score": 0.0,
        "layout_json_path": str(layout_path),
        "reference_image_path": state.get("reference_image_path"),
    }


def _generate_run_id() -> str:
    """Generate a run ID of the form YYYYMMDD_HHMMSS_<hex6>."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6]
    return f"{ts}_{uid}"


__all__ = ["setup_node"]
