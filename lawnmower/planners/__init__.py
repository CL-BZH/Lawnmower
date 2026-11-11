"""Coverage planners.  Each takes a :class:`~lawnmower.grid.Lawn` and returns
a list of cells starting at ``lawn.start``."""
from __future__ import annotations

from typing import Callable

from ..grid import Cell
from .baselines import greedy, random_walk
from .boustrophedon import boustrophedon
from .stc import stc

Planner = Callable[..., list[Cell]]

PLANNERS: dict[str, Planner] = {
    "random": random_walk,
    "greedy": greedy,
    "boustrophedon": boustrophedon,
    "stc": stc,
}

__all__ = ["PLANNERS", "Planner", "greedy", "random_walk", "boustrophedon", "stc"]
