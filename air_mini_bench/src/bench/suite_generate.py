"""Backward-compatible re-exports from ``bench.services.trace_generator``."""

from .services.trace_generator import (  # noqa: F401
    TraceGenerator,
    generate_phase_suites,
    generate_suite,
)

__all__ = ["TraceGenerator", "generate_phase_suites", "generate_suite"]
