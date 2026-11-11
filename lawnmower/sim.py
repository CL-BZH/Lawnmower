"""Replay a path on a lawn, check it is legal, and measure it.

Every planner returns a plain list of cells; :func:`evaluate` is the single
place where coverage and costs are computed, so planners cannot disagree on
what "mowed" means.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from .grid import DIR_INDEX, Cell, Lawn


class IllegalMove(ValueError):
    """Raised when a path contains a step the mower cannot make."""


@dataclass(frozen=True)
class Metrics:
    """Cost and quality of a mowing path.

    Attributes
    ----------
    target:       reachable lawn cells (what has to be mowed)
    mowed:        target cells the path visits
    steps:        number of moves
    length:       travelled distance in cells (diagonal = sqrt 2)
    turns:        number of heading changes
    turn_angle:   total absolute heading change, in degrees
    revisits:     moves that land on an already-mowed cell (wasted passes)
    """

    target: int
    mowed: int
    steps: int
    length: float
    turns: int
    turn_angle: float
    revisits: int

    @property
    def coverage(self) -> float:
        return self.mowed / self.target if self.target else 1.0

    @property
    def complete(self) -> bool:
        return self.mowed == self.target

    @property
    def overhead(self) -> float:
        """steps / (target - 1): 1.0 means no cell visited twice (optimal)."""
        return self.steps / max(self.target - 1, 1)

    def cost(self, turn_weight: float = 0.0) -> tuple[float, int]:
        """Comparison key used to pick between candidate paths.

        ``steps + turn_weight * turns``, ties broken by fewer turns.
        A turn costs a real mower time (slow down, pivot), so a positive
        ``turn_weight`` trades a few extra cells for straighter passes.
        """
        return (self.steps + turn_weight * self.turns, self.turns)

    def as_dict(self) -> dict[str, float]:
        return {
            "coverage_%": round(100 * self.coverage, 2),
            "steps": self.steps,
            "length": round(self.length, 1),
            "turns": self.turns,
            "turn_angle": round(self.turn_angle),
            "revisits": self.revisits,
            "overhead": round(self.overhead, 3),
        }


@dataclass
class Result:
    planner: str
    lawn: Lawn
    path: list[Cell]
    metrics: Metrics
    visits: Counter = field(repr=False, default_factory=Counter)
    runtime_s: float = 0.0


def heading_of(a: Cell, b: Cell) -> int:
    return DIR_INDEX[(b[0] - a[0], b[1] - a[1])]


def turn_between(h1: int, h2: int) -> int:
    """Smallest rotation between two headings, in multiples of 45 degrees."""
    d = (h2 - h1) % 8
    return min(d, 8 - d)


def validate(lawn: Lawn, path: list[Cell]) -> None:
    if not path:
        raise IllegalMove("empty path")
    if path[0] != lawn.start:
        raise IllegalMove(f"path starts at {path[0]}, expected start {lawn.start}")
    for i, (a, b) in enumerate(zip(path, path[1:]), start=1):
        if not lawn.can_move(a, b):
            raise IllegalMove(f"step {i}: {a} -> {b} is not a legal move")


def evaluate(lawn: Lawn, path: list[Cell], check: bool = True) -> tuple[Metrics, Counter]:
    """Return the metrics of ``path`` and the visit count of each cell."""
    if check:
        validate(lawn, path)
    visits: Counter = Counter()
    visits[path[0]] += 1
    length = 0.0
    turns = 0
    angle = 0
    revisits = 0
    prev_h: int | None = None
    for a, b in zip(path, path[1:]):
        h = heading_of(a, b)
        length += math.sqrt(2) if h % 2 else 1.0
        if prev_h is not None and h != prev_h:
            turns += 1
            angle += 45 * turn_between(prev_h, h)
        prev_h = h
        if visits[b]:
            revisits += 1
        visits[b] += 1
    target = lawn.reachable
    mowed = sum(1 for c in visits if c in target)
    m = Metrics(len(target), mowed, len(path) - 1, length, turns, float(angle), revisits)
    return m, visits
