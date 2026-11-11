"""Lawn-mowing coverage path planning on a grid."""
from __future__ import annotations

import time

from .grid import DIRECTIONS, Cell, Lawn
from .planners import PLANNERS
from .sim import IllegalMove, Metrics, Result, evaluate, validate

__version__ = "2.0.0"


def plan(lawn: Lawn, planner: str = "stc", **options) -> Result:
    """Run a planner by name and return its evaluated :class:`Result`."""
    try:
        fn = PLANNERS[planner]
    except KeyError:
        raise ValueError(f"unknown planner {planner!r}; choose from {sorted(PLANNERS)}") from None
    t0 = time.perf_counter()
    path = fn(lawn, **options)
    dt = time.perf_counter() - t0
    metrics, visits = evaluate(lawn, path)
    return Result(planner, lawn, path, metrics, visits, dt)


__all__ = ["Lawn", "Cell", "DIRECTIONS", "PLANNERS", "plan", "evaluate", "validate",
           "IllegalMove", "Metrics", "Result"]
