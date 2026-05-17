#!/usr/bin/env python3
"""LayoutAgent — prompt-tuning agent for Vietnamese banking form layout extraction.

Usage
-----
Tune the schema-alignment prompt on the GIAY_GUI_TIEN_TIET_KIEM sample:

    python main.py \\
        --pdf data/samples/GIAY_GUI_TIEN_TIET_KIEM/20251018+MYNTT1_0001-p18.pdf \\
        --layout-json "data/samples/GIAY_GUI_TIEN_TIET_KIEM/layout _GIAY_GUI_TIEN_TIET_KIEM.json" \\
        --chandra-prompt ../layoutDectectionChan/prompt/prompt_GGTTK.txt \\
        --initial-prompt ../layoutDectectionChan/prompt/schema_merge_html_system.txt \\
        --reference-image "data/samples/GIAY_GUI_TIEN_TIET_KIEM/20251018+MYNTT1_0001-p18_page01_schema_boxes.jpg"

Run on a test file (no ground truth, auto-resume):

    python main.py \\
        --pdf data/small_test/test_1.pdf \\
        --layout-json "data/samples/GIAY_GUI_TIEN_TIET_KIEM/layout _GIAY_GUI_TIEN_TIET_KIEM.json" \\
        --chandra-prompt ../layoutDectectionChan/prompt/prompt_GGTTK.txt \\
        --initial-prompt ../layoutDectectionChan/prompt/schema_merge_html_system.txt \\
        --auto-resume

Resume from an existing Chandra cache (fastest iteration):

    python main.py \\
        --pdf data/samples/GIAY_GUI_TIEN_TIET_KIEM/20251018+MYNTT1_0001-p18.pdf \\
        --layout-json "data/samples/GIAY_GUI_TIEN_TIET_KIEM/layout _GIAY_GUI_TIEN_TIET_KIEM.json" \\
        --chandra-log ../layoutDectectionChan/data/test/GIAY_GUI_TIEN_TIET_KIEM/test_7_llm.log \\
        --initial-prompt ../layoutDectectionChan/prompt/schema_merge_html_system.txt \\
        --reference-image "data/samples/GIAY_GUI_TIEN_TIET_KIEM/20251018+MYNTT1_0001-p18_page01_schema_boxes.jpg"

Environment variables (via .env or shell):
    OPENAI_API_KEY      Required for GPT-4o (evaluation and optimization).
    OPENAI_BASE_URL     Optional: custom OpenAI-compatible endpoint.
    QWEN_BASE_URL       Optional: separate endpoint for the align model (Qwen VL).
    QWEN_API_KEY        Optional: separate API key for the align model.
    ALIGN_MODEL         Model for schema alignment (default: gpt-4o).
    JUDGE_MODEL         Model for visual evaluation (default: gpt-4o).
    OPTIMIZER_MODEL     Model for prompt optimization (default: gpt-4o).
    CHANDRA_BASE_URL    Optional: Chandra OCR-2 endpoint.
    CHANDRA_API_KEY     Optional: Chandra API key.
    CHANDRA_MODEL       Chandra model name (default: datalab-to/chandra-ocr-2).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: add src/ to sys.path so all node imports resolve
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _load_dotenv() -> None:
    """Load .env from the LayoutAgent directory if python-dotenv is available."""
    env_path = _ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
        print(f"[main] Loaded env from {env_path}", flush=True)
    except ImportError:
        # Manual parse fallback
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LangGraph prompt-tuning agent for layout schema extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required inputs
    p.add_argument(
        "--pdf",
        type=Path,
        required=True,
        help="Input PDF file (tuning sample or test document).",
    )
    p.add_argument(
        "--layout-json",
        type=Path,
        required=True,
        help="Layout JSON file defining the schema field tree.",
    )
    p.add_argument(
        "--initial-prompt",
        type=Path,
        required=True,
        help="Path to the initial schema-alignment system prompt.",
    )

    # Chandra options
    p.add_argument(
        "--chandra-prompt",
        type=Path,
        default=None,
        help="Chandra OCR-2 prompt file. Required unless --chandra-log is provided.",
    )
    p.add_argument(
        "--chandra-log",
        type=Path,
        default=None,
        help="Existing Chandra log file to reuse (skips Chandra API call).",
    )
    p.add_argument(
        "--force-chandra-rerun",
        action="store_true",
        help="Force Chandra re-run even if a cached log exists.",
    )

    # Reference / evaluation
    p.add_argument(
        "--reference-image",
        type=Path,
        default=None,
        help="Ground-truth schema-boxes JPEG for visual evaluation.",
    )

    # Model overrides
    p.add_argument(
        "--schema-model",
        type=str,
        default=None,
        help=(
            "Schema alignment model — intended: Qwen 3 VL 4B. "
            "Defaults to QWEN_MODEL env, or 'Qwen/Qwen3-VL-4B-Instruct' when "
            "QWEN_BASE_URL is set, or 'gpt-4o' as OpenAI fallback. "
            "Overrides ALIGN_MODEL env var."
        ),
    )
    p.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="GPT-4o model for visual evaluation (overrides JUDGE_MODEL env var).",
    )

    # Tuning parameters
    p.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum number of prompt-tuning iterations (default: 3).",
    )
    p.add_argument(
        "--stop-threshold",
        type=float,
        default=0.85,
        help="Composite score threshold to stop early (default: 0.85).",
    )
    p.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.95,
        help=(
            "Minimum field coverage required to stop early (default: 0.95). "
            "Both --stop-threshold AND --coverage-threshold must be met."
        ),
    )

    # PDF rendering
    p.add_argument("--page", type=int, default=1, help="PDF page number to process (default: 1).")
    p.add_argument("--dpi", type=int, default=200, help="PDF rasterization DPI (default: 200).")

    # Local model mode
    p.add_argument(
        "--local",
        action="store_true",
        help=(
            "Use local HuggingFace Transformers for Chandra OCR-2 and Qwen 3 VL 4B "
            "instead of API calls. Requires CUDA and model weights."
        ),
    )
    p.add_argument(
        "--chandra-device", type=str, default=None,
        help="PyTorch device for local Chandra model (default: cuda:0).",
    )
    p.add_argument(
        "--qwen-device", type=str, default=None,
        help="PyTorch device for local Qwen VL model (default: cuda:0).",
    )

    # Control
    p.add_argument(
        "--auto-resume",
        action="store_true",
        help=(
            "Disable the human interrupt breakpoint and run all iterations automatically. "
            "Useful for headless/CI runs."
        ),
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Custom run ID (auto-generated if not provided).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root output directory for tuning artifacts. Default: data/tuning_runs/<run_id>/",
    )

    return p


def _build_initial_state(args: argparse.Namespace) -> dict:
    """Translate CLI arguments into the initial AgentState dictionary.

    Handles path resolution, existence checks, and environment variable overrides
    for model selection.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Initial state dict ready to be passed to the LangGraph graph.
    """
    pdf_path = args.pdf.resolve()
    layout_path = args.layout_json.resolve()
    prompt_path = args.initial_prompt.resolve()

    # Validate required paths
    for p, name in [(pdf_path, "--pdf"), (layout_path, "--layout-json"), (prompt_path, "--initial-prompt")]:
        if not p.is_file():
            print(f"[main] Error: {name} not found: {p}", file=sys.stderr)
            sys.exit(1)

    # Chandra: at least one of --chandra-prompt or --chandra-log must be provided
    chandra_prompt = args.chandra_prompt.resolve() if args.chandra_prompt else None
    chandra_log = args.chandra_log.resolve() if args.chandra_log else None

    # Load pre-existing Chandra HTML from log if provided
    chandra_html = ""
    if chandra_log and chandra_log.is_file():
        from utils.schema_html import extract_chandra_html_from_log
        raw = chandra_log.read_text(encoding="utf-8")
        chandra_html = extract_chandra_html_from_log(raw)
        print(f"[main] Pre-loaded Chandra HTML from {chandra_log} ({len(chandra_html)} chars)", flush=True)
    elif chandra_prompt is None:
        print(
            "[main] Error: provide either --chandra-prompt (to run OCR) "
            "or --chandra-log (to reuse cached output).",
            file=sys.stderr,
        )
        sys.exit(1)

    use_local = args.local or os.environ.get("USE_LOCAL_MODELS", "").lower() in ("1", "true", "yes")

    # ---- Resolve the three model roles ---------------------------------------
    # Role 1: Schema alignment → Qwen 3 VL 4B (local or API)
    def _resolve_schema_model_main() -> str:
        if os.environ.get("QWEN_MODEL"):
            return os.environ["QWEN_MODEL"]
        if use_local:
            return "Qwen/Qwen3-VL-4B-Instruct"
        if os.environ.get("QWEN_BASE_URL"):
            return "Qwen/Qwen3-VL-4B-Instruct"
        return "gpt-4o"

    align_model = args.schema_model or os.environ.get("ALIGN_MODEL") or _resolve_schema_model_main()

    # Role 2 & 3: Evaluate + Optimize → always OpenAI (GPT-4o)
    judge_model = args.judge_model or os.environ.get("JUDGE_MODEL") or "gpt-4o"
    optimizer_model = os.environ.get("OPTIMIZER_MODEL") or judge_model

    print(f"[main] Model roles:", flush=True)
    print(f"  Chandra (layout)   : datalab-to/chandra-ocr-2  ({'local' if use_local else 'API'})", flush=True)
    print(f"  Qwen VL (align)    : {align_model}  ({'local' if use_local else 'API'})", flush=True)
    print(f"  GPT-4o (eval+opt)  : {judge_model}  (OpenAI API)", flush=True)

    if not use_local and "qwen" not in align_model.lower() and not os.environ.get("QWEN_BASE_URL"):
        print(
            f"[main] NOTE: Schema alignment using '{align_model}' (OpenAI fallback). "
            "Use --local or set QWEN_BASE_URL + QWEN_MODEL in .env to use Qwen 3 VL 4B.",
            flush=True,
        )

    # Reference image
    ref_image = None
    if args.reference_image:
        ref_path = args.reference_image.resolve()
        if ref_path.is_file():
            ref_image = str(ref_path)
        else:
            print(f"[main] Warning: reference image not found: {ref_path}", flush=True)

    state: dict = {
        # Input paths
        "page_image_path": str(pdf_path),
        "layout_json_path": str(layout_path),
        "initial_prompt_path": str(prompt_path),
        "reference_image_path": ref_image,
        # Chandra
        "chandra_prompt_path": str(chandra_prompt) if chandra_prompt else "",
        "chandra_log_path": str(chandra_log) if chandra_log else None,
        "chandra_html": chandra_html,
        "force_chandra_rerun": args.force_chandra_rerun,
        # Model roles (three distinct roles):
        #   openai_model      → Qwen 3 VL 4B: schema alignment (local or Qwen API)
        #   openai_judge_model → GPT-4o: visual evaluation (always OpenAI)
        #   optimizer_model    → GPT-4o: prompt optimization (always OpenAI)
        "openai_model": align_model,
        "openai_judge_model": judge_model,
        "optimizer_model": optimizer_model,
        # Local transformer mode
        "use_local_models": use_local,
        "chandra_device": args.chandra_device or os.environ.get("CHANDRA_DEVICE", "cuda:0"),
        "qwen_device": args.qwen_device or os.environ.get("QWEN_DEVICE", "cuda:0"),
        # PDF rendering
        "viz_page": args.page,
        "viz_dpi": args.dpi,
        # Tuning
        "max_iterations": args.max_iterations,
        "stop_threshold": args.stop_threshold,
        "coverage_threshold": args.coverage_threshold,
        # Optional overrides
        "run_id": args.run_id or "",
        "output_dir": str(args.output_dir.resolve()) if args.output_dir else "",
    }
    return state


def _run_interactive(app: object, initial_state: dict, thread_config: dict) -> None:
    """Drive the graph with human-in-the-loop at each optimize_prompt breakpoint.

    After each interrupted state, displays the evaluation results and asks the
    operator whether to continue optimization or stop.

    Args:
        app: Compiled LangGraph application.
        initial_state: Initial state dictionary.
        thread_config: LangGraph thread configuration dict.
    """
    print("\n[main] Starting interactive tuning run (interrupt before each optimization)...\n")
    current_input = initial_state

    while True:
        events = list(app.stream(current_input, config=thread_config, stream_mode="values"))
        last_state = events[-1] if events else {}

        snap = app.get_state(thread_config)
        next_nodes = list(snap.next) if snap else []

        if not next_nodes:
            print("\n[main] Run complete (graph reached END).")
            _print_summary(last_state)
            break

        if "optimize_prompt" in next_nodes:
            _print_iteration_summary(last_state)
            answer = _ask_continue()
            if not answer:
                print("[main] Operator chose to stop. Saving best prompt and exiting.")
                _print_summary(last_state)
                break
            # Resume: pass None to continue from breakpoint
            current_input = None
        else:
            # Graph stopped at an unexpected node
            print(f"[main] Stopped at unexpected node(s): {next_nodes}")
            _print_summary(last_state)
            break


def _run_auto(app: object, initial_state: dict, thread_config: dict) -> None:
    """Drive the graph automatically without human interaction.

    Streams all events and prints a summary at completion.

    Args:
        app: Compiled LangGraph application.
        initial_state: Initial state dictionary.
        thread_config: LangGraph thread configuration dict.
    """
    print("\n[main] Starting automatic tuning run (no human interrupts)...\n")
    last_state: dict = {}
    for state in app.stream(initial_state, config=thread_config, stream_mode="values"):
        last_state = state
    _print_summary(last_state)


def _print_iteration_summary(state: dict) -> None:
    """Print a compact summary of the latest evaluation iteration."""
    iteration = state.get("iteration", "?")
    composite = state.get("composite_score", 0.0)
    coverage = state.get("coverage", 0.0)
    judge = state.get("llm_judge", 0.0)
    missing = state.get("missing_fields", [])
    feedback = state.get("eval_feedback", "")
    output_dir = state.get("output_dir", "")

    print("\n" + "=" * 60)
    print(f"  ITERATION {iteration} RESULTS")
    print("=" * 60)
    print(f"  Composite score : {composite:.3f}")
    print(f"  Coverage        : {coverage:.3f}")
    print(f"  LLM judge       : {judge:.3f}")
    print(f"  Missing fields  : {missing}")
    if feedback:
        print(f"  Feedback        : {feedback[:300]}")
    if output_dir:
        print(f"  Artifacts       : {output_dir}/iter_{iteration:02d}/")
    print("=" * 60 + "\n")


def _print_summary(state: dict) -> None:
    """Print final run summary."""
    print("\n" + "=" * 60)
    print("  TUNING RUN COMPLETE")
    print("=" * 60)
    print(f"  Run ID          : {state.get('run_id', '?')}")
    print(f"  Best score      : {state.get('best_score', 0):.3f}")
    print(f"  Best iteration  : {state.get('best_iteration', '?')}")
    print(f"  Total iterations: {state.get('iteration', '?')}")
    output_dir = state.get("output_dir", "")
    if output_dir:
        print(f"  Output dir      : {output_dir}")
        print(f"  Best prompt     : {output_dir}/best_prompt.txt")
    print("=" * 60 + "\n")


def _ask_continue() -> bool:
    """Prompt the operator to continue or stop the tuning loop.

    Returns:
        True to continue optimization, False to stop.
    """
    while True:
        try:
            answer = input("Continue with prompt optimization? [y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[main] Interrupted — stopping.")
            return False
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  Please enter 'y' or 'n'.")


def main() -> int:
    _load_dotenv()

    parser = _build_arg_parser()
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("[main] Warning: OPENAI_API_KEY not set.", flush=True)

    initial_state = _build_initial_state(args)

    from graph import build_graph

    with_interrupt = not args.auto_resume
    app = build_graph(with_human_interrupt=with_interrupt)

    thread_id = args.run_id or uuid.uuid4().hex[:8]
    thread_config = {"configurable": {"thread_id": thread_id}}

    if with_interrupt:
        _run_interactive(app, initial_state, thread_config)
    else:
        _run_auto(app, initial_state, thread_config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
