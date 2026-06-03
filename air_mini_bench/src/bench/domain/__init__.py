"""Domain layer: scenario specs, suite layout, and benchmark result models."""

from .report import BenchReport, RequestResult
from .scenario import (
    PRIORITY_SUITES_BY_PHASE,
    PHASE1_SUITES,
    PHASE2_EXTENDED,
    PHASE2_PRIORITY,
    PHASE2_SUITES,
    SUITES_BY_PHASE,
    ScenarioSpec,
    get_suite,
    list_suite_names,
)
from .suite import GenerationConfig, SuitePaths

__all__ = [
    "BenchReport",
    "GenerationConfig",
    "PRIORITY_SUITES_BY_PHASE",
    "PHASE1_SUITES",
    "PHASE2_EXTENDED",
    "PHASE2_PRIORITY",
    "PHASE2_SUITES",
    "RequestResult",
    "SUITES_BY_PHASE",
    "ScenarioSpec",
    "SuitePaths",
    "get_suite",
    "list_suite_names",
]
