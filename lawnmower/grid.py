"""The lawn: a grid of cells, each one blade-width wide.

Model
-----
* One grid cell is exactly the area the blade cuts when the mower sits on it.
* The mower occupies one cell and mows it on arrival.
* It moves to one of its 8 neighbours per step.  A diagonal move is legal only
  if both orthogonal cells it squeezes between are free (no corner cutting),
  so the mower never clips an obstacle corner.

Because of the no-corner-cutting rule, every diagonal step can be replaced by
two orthogonal ones, hence the cells the mower can reach from the start are
exactly the 4-connected component of the start.

Coordinates are ``(row, col)``, row 0 at the top.

Map file format
---------------
``L`` (or ``.``) lawn, ``S`` lawn + start position, ``+`` (or ``#``) obstacle.
Anything else, ragged line ends and the area outside the file count as obstacles.
"""
from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

Cell = tuple[int, int]

# The 8 moves, ordered counter-clockwise starting East. Index = heading.
# Heading h corresponds to an angle of 45*h degrees (screen "up" = North).
DIRECTIONS: tuple[Cell, ...] = (
    (0, 1),    # 0  E
    (-1, 1),   # 1  NE
    (-1, 0),   # 2  N
    (-1, -1),  # 3  NW
    (0, -1),   # 4  W
    (1, -1),   # 5  SW
    (1, 0),    # 6  S
    (1, 1),    # 7  SE
)
DIRECTION_NAMES = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")
DIR_INDEX: dict[Cell, int] = {d: i for i, d in enumerate(DIRECTIONS)}

LAWN_CHARS = {"L", ".", "S"}
START_CHAR = "S"


@dataclass
class Lawn:
    """Grid of free (lawn) and blocked cells plus a start position."""

    free: np.ndarray                     # bool array (H, W); True = lawn
    start: Cell
    name: str = "lawn"
    _reachable: frozenset[Cell] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.free = np.asarray(self.free, dtype=bool)
        if not self.in_bounds(self.start) or not self.free[self.start]:
            raise ValueError(f"start {self.start} is not a lawn cell")
        self._reachable = frozenset(self._component(self.start))

    # ------------------------------------------------------------------ build
    @classmethod
    def from_string(cls, text: str, start: Cell | None = None,
                    name: str = "lawn") -> "Lawn":
        lines = [ln.rstrip("\r\n") for ln in text.splitlines()]
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            raise ValueError("empty map")
        width = max(len(ln) for ln in lines)
        free = np.zeros((len(lines), width), dtype=bool)
        found_start: Cell | None = None
        for r, ln in enumerate(lines):
            for c, ch in enumerate(ln):
                if ch in LAWN_CHARS:
                    free[r, c] = True
                if ch == START_CHAR:
                    if found_start is not None:
                        raise ValueError("map has more than one 'S'")
                    found_start = (r, c)
        if not free.any():
            raise ValueError("map contains no lawn")
        if start is None:
            start = found_start
        if start is None:  # first lawn cell in reading order
            rows, cols = np.nonzero(free)
            start = (int(rows[0]), int(cols[0]))
        return cls(free, start, name)

    @classmethod
    def from_file(cls, path: str | Path, start: Cell | None = None) -> "Lawn":
        path = Path(path)
        return cls.from_string(path.read_text(), start=start, name=path.stem)

    def with_start(self, start: Cell) -> "Lawn":
        return Lawn(self.free.copy(), start, self.name)

    # ---------------------------------------------------------------- queries
    @property
    def shape(self) -> tuple[int, int]:
        return self.free.shape  # type: ignore[return-value]

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.free.shape[0] and 0 <= c < self.free.shape[1]

    def is_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and bool(self.free[cell])

    def can_move(self, a: Cell, b: Cell) -> bool:
        """True if a single step from ``a`` to ``b`` is legal."""
        dr, dc = b[0] - a[0], b[1] - a[1]
        if (dr, dc) not in DIR_INDEX or not self.is_free(b):
            return False
        if dr and dc:  # diagonal: no corner cutting
            return self.is_free((a[0] + dr, a[1])) and self.is_free((a[0], a[1] + dc))
        return True

    def neighbors(self, cell: Cell, diagonal: bool = True) -> Iterator[Cell]:
        for dr, dc in DIRECTIONS:
            if not diagonal and dr and dc:
                continue
            nb = (cell[0] + dr, cell[1] + dc)
            if self.can_move(cell, nb):
                yield nb

    @property
    def lawn_cells(self) -> frozenset[Cell]:
        rows, cols = np.nonzero(self.free)
        return frozenset(zip(rows.tolist(), cols.tolist()))

    @property
    def reachable(self) -> frozenset[Cell]:
        """Lawn cells reachable from the start: the coverage target."""
        return self._reachable

    @property
    def unreachable(self) -> frozenset[Cell]:
        return self.lawn_cells - self._reachable

    def _component(self, start: Cell) -> set[Cell]:
        seen = {start}
        todo = deque([start])
        while todo:
            cur = todo.popleft()
            for nb in self.neighbors(cur, diagonal=False):
                if nb not in seen:
                    seen.add(nb)
                    todo.append(nb)
        return seen

    # ------------------------------------------------------------ path search
    def bfs(self, source: Cell) -> dict[Cell, Cell | None]:
        """Shortest-path tree (in steps) from ``source``: child -> parent."""
        parent: dict[Cell, Cell | None] = {source: None}
        todo = deque([source])
        while todo:
            cur = todo.popleft()
            for nb in self.neighbors(cur):
                if nb not in parent:
                    parent[nb] = cur
                    todo.append(nb)
        return parent

    def shortest_path(self, a: Cell, b: Cell) -> list[Cell]:
        """Shortest path from ``a`` to ``b`` (both included).

        Among paths with the fewest steps, prefers few turns: orthogonal moves
        are explored first and the search keeps the current heading when it can.
        """
        if a == b:
            return [a]
        # Dijkstra on (cell, heading); cost = steps * big + turns, so steps
        # dominate and turns only break ties.
        big = 10_000
        start_state = (a, -1)
        dist: dict[tuple[Cell, int], int] = {start_state: 0}
        parent: dict[tuple[Cell, int], tuple[Cell, int] | None] = {start_state: None}
        heap: list[tuple[int, int, tuple[Cell, int]]] = [(0, 0, start_state)]
        counter = 0
        while heap:
            d, _, state = heapq.heappop(heap)
            if d != dist.get(state):
                continue
            cell, head = state
            if cell == b:
                out = []
                s: tuple[Cell, int] | None = state
                while s is not None:
                    out.append(s[0])
                    s = parent[s]
                return out[::-1]
            for h, (dr, dc) in enumerate(DIRECTIONS):
                nb = (cell[0] + dr, cell[1] + dc)
                if not self.can_move(cell, nb):
                    continue
                cost = d + big + (0 if head in (-1, h) else 1)
                ns = (nb, h)
                if cost < dist.get(ns, 1 << 62):
                    dist[ns] = cost
                    parent[ns] = state
                    counter += 1
                    heapq.heappush(heap, (cost, counter, ns))
        raise ValueError(f"no path from {a} to {b}")

    # ---------------------------------------------------------------- display
    def render(self, marks: dict[Cell, str] | None = None) -> str:
        marks = marks or {}
        rows = []
        for r in range(self.shape[0]):
            row = []
            for c in range(self.shape[1]):
                if (r, c) in marks:
                    row.append(marks[(r, c)])
                elif (r, c) == self.start:
                    row.append("S")
                else:
                    row.append("L" if self.free[r, c] else "+")
            rows.append("".join(row))
        return "\n".join(rows)

    def __repr__(self) -> str:
        h, w = self.shape
        return (f"Lawn({self.name!r}, {h}x{w}, {len(self._reachable)} reachable "
                f"cells, start={self.start})")
