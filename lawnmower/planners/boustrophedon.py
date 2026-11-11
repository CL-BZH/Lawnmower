"""Boustrophedon cellular decomposition (Choset & Pignon, 1997), on a grid.

1. Sweep a line across the lawn column by column.  In each column the lawn
   splits into vertical *segments* (maximal runs of free cells).
2. Group consecutive segments into *cells*: a segment extends the cell of the
   segment to its left when they overlap one-to-one; whenever segments split
   or merge (an obstacle starts or ends — a "critical event"), new cells begin.
   Each cell is then obstacle-free in the sweep direction and can be mowed
   with plain back-and-forth passes.
3. Choose an order of the cells and, for each, which corner to enter by
   (see :func:`order_cells`), then sweep them one after another.

The whole thing is run with vertical passes and with horizontal passes
(by transposing the lawn); the cheaper of the two is kept.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..grid import Cell, Lawn
from .common import bfs_dist, finish_greedily, go_to, transpose_lawn, transpose_path


@dataclass
class Segment:
    col: int
    top: int
    bottom: int  # inclusive

    def overlaps(self, other: "Segment") -> bool:
        return self.top <= other.bottom and other.top <= self.bottom


def column_segments(lawn: Lawn) -> list[list[Segment]]:
    reach = lawn.reachable
    h, w = lawn.shape
    cols: list[list[Segment]] = []
    for c in range(w):
        segs: list[Segment] = []
        r = 0
        while r < h:
            if (r, c) in reach:
                top = r
                while r + 1 < h and (r + 1, c) in reach:
                    r += 1
                segs.append(Segment(c, top, r))
            r += 1
        cols.append(segs)
    return cols


def decompose(lawn: Lawn) -> list[list[Segment]]:
    """Return the boustrophedon cells, each a left-to-right list of segments."""
    cells: list[list[Segment]] = []
    prev: list[tuple[Segment, int]] = []  # (segment, cell id) in previous column
    for segs in column_segments(lawn):
        cur: list[tuple[Segment, int]] = []
        for s in segs:
            left = [(p, cid) for p, cid in prev if p.overlaps(s)]
            if len(left) == 1:
                p, cid = left[0]
                if sum(1 for t in segs if t.overlaps(p)) == 1:
                    cells[cid].append(s)
                    cur.append((s, cid))
                    continue
            cells.append([s])
            cur.append((s, len(cells) - 1))
        prev = cur
    return cells


def sweep_cell(lawn: Lawn, path: list[Cell], segs: list[Segment],
               reverse: bool, downward: bool) -> None:
    """Append back-and-forth passes over one cell to ``path``."""
    order = segs[::-1] if reverse else segs
    for s in order:
        rows = range(s.top, s.bottom + 1) if downward else range(s.bottom, s.top - 1, -1)
        cells = [(r, s.col) for r in rows]
        go_to(lawn, path, cells[0])
        path.extend(cells[1:])
        downward = not downward


def entry_exit(segs: list[Segment], reverse: bool, downward: bool) -> tuple[Cell, Cell]:
    """Where the mower enters and leaves a cell swept with these options."""
    first = segs[-1] if reverse else segs[0]
    last = segs[0] if reverse else segs[-1]
    entry = (first.top if downward else first.bottom, first.col)
    last_down = downward if len(segs) % 2 == 1 else not downward
    exit_ = (last.bottom if last_down else last.top, last.col)
    return entry, exit_


SWEEP_OPTIONS = [(rev, down) for rev in (False, True) for down in (True, False)]


def order_cells(lawn: Lawn, cells: list[list[Segment]]) -> list[tuple[int, bool, bool]]:
    """Choose the visiting order of the cells and how each is swept.

    This is a small generalised TSP: each cell can be swept 4 ways, each
    with its own entry and exit corner.  Solved with a *rollout* of the
    nearest-neighbour rule: at every decision, each candidate is scored by
    the travel to its entry plus the cost of finishing nearest-first from its
    exit, and the best candidate is kept.  Travel costs are exact grid
    distances (one BFS per corner).
    """
    ends = {(i, rev, down): entry_exit(cells[i], rev, down)
            for i in range(len(cells)) for rev, down in SWEEP_OPTIONS}
    sources = {lawn.start} | {ex for _, ex in ends.values()}
    dist = {src: bfs_dist(lawn, src) for src in sources}

    def nearest_first(pos: Cell, todo: set[int]) -> int:
        total = 0
        todo = set(todo)
        while todo:
            d, opt = min((dist[pos][ends[(i, r, w)][0]], (i, r, w))
                         for i in todo for r, w in SWEEP_OPTIONS)
            total += d
            todo.discard(opt[0])
            pos = ends[opt][1]
        return total

    order = []
    pos = lawn.start
    todo = set(range(len(cells)))
    while todo:
        best = min(
            (dist[pos][ends[(i, r, w)][0]] + nearest_first(ends[(i, r, w)][1], todo - {i}),
             (i, r, w))
            for i in todo for r, w in SWEEP_OPTIONS)
        opt = best[1]
        order.append(opt)
        todo.discard(opt[0])
        pos = ends[opt][1]
    return order


def _is_mowed(cell: list[Segment], mowed: set[Cell]) -> bool:
    return all((r, s.col) in mowed for s in cell for r in range(s.top, s.bottom + 1))


def _plan_nearest(lawn: Lawn, cells: list[list[Segment]]) -> list[Cell]:
    """Each time, sweep the unmowed cell whose entry corner is closest."""
    path = [lawn.start]
    todo = set(range(len(cells)))
    while True:
        mowed = set(path)
        todo = {i for i in todo if not _is_mowed(cells[i], mowed)}
        if not todo:
            return finish_greedily(lawn, path)
        dist = bfs_dist(lawn, path[-1])
        _, i, rev, down = min((dist[entry_exit(cells[i], r, w)[0]], i, r, w)
                              for i in todo for r, w in SWEEP_OPTIONS)
        sweep_cell(lawn, path, cells[i], rev, down)
        todo.discard(i)


def _plan_ordered(lawn: Lawn, cells: list[list[Segment]]) -> list[Cell]:
    path = [lawn.start]
    for i, rev, down in order_cells(lawn, cells):
        if not _is_mowed(cells[i], set(path)):
            sweep_cell(lawn, path, cells[i], rev, down)
    return finish_greedily(lawn, path)


def _candidates(lawn: Lawn, max_rollout_cells: int) -> list[list[Cell]]:
    cells = decompose(lawn)
    out = [_plan_nearest(lawn, cells)]
    if 1 < len(cells) <= max_rollout_cells:
        out.append(_plan_ordered(lawn, cells))
    return out


def boustrophedon(lawn: Lawn, orientation: str = "best",
                  turn_weight: float = 0.0, max_rollout_cells: int = 30) -> list[Cell]:
    """Plan with vertical passes, horizontal passes, or whichever is cheaper.

    For each orientation two cell orders are tried: plain nearest-first and
    the rollout of :func:`order_cells` (skipped above ``max_rollout_cells``
    cells, where it gets slow).  The cheapest resulting path is returned.

    ``orientation``: ``"vertical"``, ``"horizontal"`` or ``"best"``.
    ``turn_weight``: see :meth:`lawnmower.sim.Metrics.cost`.
    """
    from ..sim import evaluate

    options = []
    if orientation in ("vertical", "best"):
        options += _candidates(lawn, max_rollout_cells)
    if orientation in ("horizontal", "best"):
        options += [transpose_path(p)
                    for p in _candidates(transpose_lawn(lawn), max_rollout_cells)]
    if not options:
        raise ValueError(f"unknown orientation {orientation!r}")

    return min(options, key=lambda p: evaluate(lawn, p, check=False)[0].cost(turn_weight))
