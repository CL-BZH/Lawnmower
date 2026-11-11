"""Spanning Tree Coverage (Gabriely & Rimon, 2001), with a clean-up pass.

1. Group the grid into 2x2 *mega-cells*.  Keep those whose 4 sub-cells are
   all reachable lawn; join 4-adjacent ones into a graph.
2. Build a spanning tree of that graph.
3. Walk around the tree, keeping it on the mower's left, through the
   sub-cells.  This is a closed loop visiting every sub-cell of every kept
   mega-cell exactly once, moving only orthogonally.

   Construction used here: each sub-cell touches two sides of its mega-cell.
   On each such side, if the tree has an edge crossing that side, link the
   sub-cell to the facing sub-cell across it; otherwise link it to its sibling
   along that side.  Every sub-cell then has degree 2 and, because the tree
   is connected and acyclic, the links form one single cycle.
4. Lawn the mega-cells could not cover (strips along obstacles whose width or
   alignment is odd) is mowed afterwards: while walking the loop, the mower
   makes short detours into adjacent leftover cells, and anything still
   missing at the end is reached with nearest-first moves.

The grid alignment of the mega-cells (4 offsets) and the shape of the tree
(long horizontal or long vertical runs) are tried exhaustively and the
cheapest path is kept.
"""
from __future__ import annotations

from collections import defaultdict

from ..grid import Cell, Lawn
from ..sim import evaluate
from .common import bfs_dist, finish_greedily, go_to


def mega_cells(lawn: Lawn, off_r: int, off_c: int) -> set[Cell]:
    """Mega-cell indices (I, J) whose 4 sub-cells are all reachable."""
    reach = lawn.reachable
    h, w = lawn.shape
    out = set()
    for big_r in range((h - off_r) // 2):
        for big_c in range((w - off_c) // 2):
            r, c = off_r + 2 * big_r, off_c + 2 * big_c
            if {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)} <= reach:
                out.add((big_r, big_c))
    return out


def spanning_forest(nodes: set[Cell], horizontal_first: bool) -> set[frozenset[Cell]]:
    """Kruskal with cheap edges along one axis -> comb-shaped trees.

    Long straight branches give long straight passes around them.
    """
    parent = {n: n for n in nodes}

    def find(x: Cell) -> Cell:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = []
    for (r, c) in nodes:
        for dr, dc in ((0, 1), (1, 0)):
            nb = (r + dr, c + dc)
            if nb in nodes:
                along = (dc == 1) == horizontal_first
                edges.append((0 if along else 1, r, c, (r, c), nb))
    edges.sort()
    tree: set[frozenset[Cell]] = set()
    for *_, a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            tree.add(frozenset((a, b)))
    return tree


def circumnavigate(nodes: set[Cell], tree: set[frozenset[Cell]],
                   off_r: int, off_c: int) -> list[list[Cell]]:
    """Return one closed loop of sub-cells per tree of the forest."""
    links: dict[Cell, list[Cell]] = defaultdict(list)

    def sub(big: Cell, dr: int, dc: int) -> Cell:
        return (off_r + 2 * big[0] + dr, off_c + 2 * big[1] + dc)

    for big in nodes:
        br, bc = big
        for (dr, dc) in ((0, 0), (0, 1), (1, 0), (1, 1)):
            me = sub(big, dr, dc)
            # top side (dr == 0) or bottom side (dr == 1)
            nb = (br + (-1 if dr == 0 else 1), bc)
            if frozenset((big, nb)) in tree:
                links[me].append(sub(nb, 1 - dr, dc))      # cross into nb
            else:
                links[me].append(sub(big, dr, 1 - dc))     # run along the side
            # left side (dc == 0) or right side (dc == 1)
            nb = (br, bc + (-1 if dc == 0 else 1))
            if frozenset((big, nb)) in tree:
                links[me].append(sub(nb, dr, 1 - dc))
            else:
                links[me].append(sub(big, 1 - dr, dc))

    loops = []
    seen: set[Cell] = set()
    for startcell in sorted(links):
        if startcell in seen:
            continue
        loop = [startcell]
        seen.add(startcell)
        prev, cur = startcell, links[startcell][0]
        while cur != startcell:
            if cur in seen:
                raise AssertionError("STC links do not form a simple cycle")
            loop.append(cur)
            seen.add(cur)
            a, b = links[cur]
            prev, cur = cur, (b if a == prev else a)
        loops.append(loop)
    return loops


def _walk_loops(lawn: Lawn, loops: list[list[Cell]], detours: bool) -> list[Cell]:
    path = [lawn.start]
    pending = {c for lp in loops for c in lp}       # loop cells not walked yet
    leftovers = set(lawn.reachable) - pending       # lawn no loop covers
    remaining = list(range(len(loops)))
    while remaining:
        dist = bfs_dist(lawn, path[-1])
        # enter the closest loop at its closest cell
        k, i = min(((k, i) for k in remaining for i in range(len(loops[k]))),
                   key=lambda ki: dist[loops[ki[0]][ki[1]]])
        loop = loops[k]
        remaining.remove(k)
        n = len(loop)
        order = [loop[(i + j) % n] for j in range(n)]
        # walk the loop in the direction whose second cell is closer
        if n > 2:
            rev = [order[0]] + order[:0:-1]
            if dist[rev[1]] < dist[order[1]]:
                order = rev
        for cell in order:
            go_to(lawn, path, cell)
            pending.difference_update(path)
            leftovers.difference_update(path)
            if detours:
                _detour(lawn, path, leftovers, pending)
    return finish_greedily(lawn, path)


def _detour(lawn: Lawn, path: list[Cell], leftovers: set[Cell],
            pending: set[Cell]) -> None:
    """Step into leftover cells hanging off the current position.

    The mower steps into an adjacent leftover cell and keeps following a chain
    of leftover cells only while the next one is *not* also next to a loop
    cell still to be walked (those are picked up later at almost no cost, by
    stepping out and back diagonally as the loop goes on).  The next
    ``go_to`` brings the mower back to the loop.
    """
    cur = path[-1]
    while True:
        nxt = next((nb for nb in lawn.neighbors(cur, diagonal=False)
                    if nb in leftovers), None)
        if nxt is None:
            return
        path.append(nxt)
        leftovers.discard(nxt)
        cur = nxt
        if any(nb in pending for nb in lawn.neighbors(cur, diagonal=False)):
            return



def stc(lawn: Lawn, detours: bool = True, turn_weight: float = 0.0) -> list[Cell]:
    """Best STC path over the 4 grid alignments and 2 tree shapes.

    ``detours``: mow leftover cells next to the loop while walking it
    (otherwise they are all left to the final nearest-first clean-up).
    ``turn_weight``: see :meth:`lawnmower.sim.Metrics.cost`.
    """
    best: tuple | None = None
    for off_r in (0, 1):
        for off_c in (0, 1):
            nodes = mega_cells(lawn, off_r, off_c)
            for horizontal in (True, False):
                tree = spanning_forest(nodes, horizontal)
                loops = circumnavigate(nodes, tree, off_r, off_c)
                path = _walk_loops(lawn, loops, detours)
                m, _ = evaluate(lawn, path, check=False)
                key = m.cost(turn_weight)
                if best is None or key < best[0]:
                    best = (key, path)
    assert best is not None
    return best[1]
