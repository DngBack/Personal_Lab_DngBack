"""LangGraph definition for the prompt-tuning agent.

Graph topology
--------------

    setup
      └── run_chandra
            └── align_schema          ← re-entry point for each tuning iteration
                  └── merge_schemas
                        └── visualize
                              └── evaluate
                                    └── checkpoint
                                          ├── [score >= threshold or iter >= max] → END
                                          └── [continue] → optimize_prompt ← INTERRUPT BEFORE
                                                              └── align_schema (loop)

Human breakpoint
----------------
``interrupt_before=["optimize_prompt"]`` causes LangGraph to pause the graph
execution before the optimize_prompt node runs. The operator can inspect the
current evaluation and then resume (or abort) the run via the checkpoint
mechanism. When running headlessly, pass ``auto_resume=True`` in the agent
config to skip the interactive prompt.

Routing function
----------------
``should_continue_router`` reads the current iteration count and composite score
from the state and returns either:
- ``"optimize_prompt"`` — to loop back and improve the prompt.
- ``"__end__"`` — to terminate the run.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# Bootstrap src/ so all relative imports work regardless of CWD.
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from state import AgentState
from nodes.setup import setup_node
from nodes.run_chandra import run_chandra_node
from nodes.align_schema import align_schema_node
from nodes.merge_schemas import merge_schemas_node
from nodes.visualize import visualize_node
from nodes.evaluate import evaluate_node
from nodes.checkpoint import checkpoint_node
from nodes.optimize_prompt import optimize_prompt_node


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def should_continue_router(state: AgentState) -> str:
    """Decide whether to optimize the prompt or end the run.

    Conditions that trigger a stop:
    - The composite score has reached or exceeded ``stop_threshold``.
    - The iteration counter has reached or exceeded ``max_iterations``.

    Args:
        state: Current agent state after the checkpoint node.

    Returns:
        ``"optimize_prompt"`` to continue tuning, or ``"__end__"`` to stop.
    """
    iteration: int = state.get("iteration", 0)
    max_iterations: int = state.get("max_iterations", 3)
    composite: float = state.get("composite_score", 0.0)
    threshold: float = state.get("stop_threshold", 0.85)

    # Require both composite score AND coverage to be high enough to stop early.
    # A high judge score should not mask low coverage (missing fields).
    coverage: float = state.get("coverage", 0.0)
    coverage_min = state.get("coverage_threshold", 0.95)

    if composite >= threshold and coverage >= coverage_min:
        print(
            f"[router] Score {composite:.3f} >= {threshold:.3f} AND "
            f"coverage {coverage:.3f} >= {coverage_min:.3f} — stopping.",
            flush=True,
        )
        return "__end__"

    if composite >= threshold and coverage < coverage_min:
        print(
            f"[router] Composite {composite:.3f} >= threshold but "
            f"coverage {coverage:.3f} < {coverage_min:.3f} (missing fields) — continuing.",
            flush=True,
        )

    if iteration >= max_iterations:
        print(
            f"[router] Reached max iterations ({max_iterations}) — stopping.",
            flush=True,
        )
        return "__end__"

    print(
        f"[router] Score {composite:.3f} < {threshold:.3f}, iter {iteration}/{max_iterations} "
        "— continuing to optimize.",
        flush=True,
    )
    return "optimize_prompt"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(*, with_human_interrupt: bool = True) -> Any:
    """Construct and compile the prompt-tuning LangGraph.

    Args:
        with_human_interrupt: If True (default), add an ``interrupt_before``
            breakpoint at the ``optimize_prompt`` node. Set to False for fully
            automated headless runs.

    Returns:
        Compiled LangGraph application ready to be invoked.
    """
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("setup", setup_node)
    builder.add_node("run_chandra", run_chandra_node)
    builder.add_node("align_schema", align_schema_node)
    builder.add_node("merge_schemas", merge_schemas_node)
    builder.add_node("visualize", visualize_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("checkpoint", checkpoint_node)
    builder.add_node("optimize_prompt", optimize_prompt_node)

    # Linear edges: setup → run_chandra → align_schema → ... → checkpoint
    builder.set_entry_point("setup")
    builder.add_edge("setup", "run_chandra")
    builder.add_edge("run_chandra", "align_schema")
    builder.add_edge("align_schema", "merge_schemas")
    builder.add_edge("merge_schemas", "visualize")
    builder.add_edge("visualize", "evaluate")
    builder.add_edge("evaluate", "checkpoint")

    # Conditional edge: checkpoint → [optimize_prompt | END]
    builder.add_conditional_edges(
        "checkpoint",
        should_continue_router,
        {
            "optimize_prompt": "optimize_prompt",
            "__end__": END,
        },
    )

    # Loop back: optimize_prompt → align_schema (skips setup + run_chandra)
    builder.add_edge("optimize_prompt", "align_schema")

    # Compile with in-memory checkpoint store and optional human breakpoint
    interrupt_before = ["optimize_prompt"] if with_human_interrupt else []
    app = builder.compile(
        checkpointer=MemorySaver(),
        interrupt_before=interrupt_before,
    )
    return app


__all__ = ["build_graph", "should_continue_router"]
