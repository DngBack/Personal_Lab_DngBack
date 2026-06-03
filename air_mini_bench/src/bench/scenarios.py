"""Backward-compatible re-exports from ``bench.domain.scenario``."""

from .domain.scenario import (  # noqa: F401
    PHASE1_SUITES,
    PHASE2_EXTENDED,
    PHASE2_PRIORITY,
    PHASE2_SUITES,
    PRIORITY_SUITES_BY_PHASE,
    SUITES_BY_PHASE,
    ScenarioSpec,
    get_suite,
    list_suite_names,
)

__all__ = [
    "PHASE1_SUITES",
    "PHASE2_EXTENDED",
    "PHASE2_PRIORITY",
    "PHASE2_SUITES",
    "PRIORITY_SUITES_BY_PHASE",
    "SUITES_BY_PHASE",
    "ScenarioSpec",
    "get_suite",
    "list_suite_names",
]
