"""
Mini Mooncake-style bench for LLM Inference Optimization Challenge V2.

Layout:
  - ``bench.domain`` — scenario specs, suite paths, report models
  - ``bench.services`` — trace generation, replay, dataset analysis
  - ``bench.*`` (root) — config, arrivals, prompts, storage, thin CLIs

Entry points:
  ``python -m bench.generate``
  ``python -m bench.analyze``
  ``python -m bench.run_bench``
  ``python -m bench.run_scenarios``
"""

from .domain import (
    BenchReport,
    GenerationConfig,
    RequestResult,
    ScenarioSpec,
    SuitePaths,
)
from .services import (
    DatasetAnalyzer,
    ReplayEngine,
    TraceGenerator,
    generate_suite,
    replay_phase,
)

__all__ = [
    "BenchReport",
    "DatasetAnalyzer",
    "GenerationConfig",
    "ReplayEngine",
    "RequestResult",
    "ScenarioSpec",
    "SuitePaths",
    "TraceGenerator",
    "generate_suite",
    "replay_phase",
]
