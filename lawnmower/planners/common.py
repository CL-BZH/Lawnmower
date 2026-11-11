"""Helpers shared by the planners."""
from __future__ import annotations

from collections import deque

from ..grid import Cell, Lawn
from ..sim import heading_of, turn_between


def bfs_dist(lawn: Lawn, src: Cell) -> dict[Cell, int]:
    """Step distance from ``src`` to every reachable cell."""
    parent = lawn.bfs(src)
    dist = {src: 0}
    # parents are discovered in BFS order, so a single pass suffices
    for cell, par in parent.items():
        if par is not None:
            dist[cell] = dist[par] + 1
    return dist


def go_to(lawn: Lawn, path: list[Cell], target: Cell) -> None:
    """Extend ``path`` (in place) with a shortest path to ``target``."""
    if path[-1] != target:
        path.extend(lawn.shortest_path(path[-1], target)[1:])


def nearest_unmowed(lawn: Lawn, src: Cell, mowed: set[Cell],
                    heading: int | None = None) -> list[Cell] | None:
    """Shortest path from ``src`` to the closest reachable unmowed cell.

    Ties between equally close cells are broken in favour of the one whose
    first step needs the smallest turn from ``heading``.
    Returns ``None`` when everything is mowed.
    """
    parent: dict[Cell, Cell | None] = {src: None}
    dist = {src: 0}
    todo = deque([src])
    best_d: int | None = None
    candidates: list[Cell] = []
    while todo:
        cur = todo.popleft()
        if best_d is not None and dist[cur] > best_d:
            break
        if cur != src and cur not in mowed:
            best_d = dist[cur]
            candidates.append(cur)
            continue
        for nb in lawn.neighbors(cur):
            if nb not in parent:
                parent[nb] = cur
                dist[nb] = dist[cur] + 1
                todo.append(nb)
    if not candidates:
        return None

    def first_turn(target: Cell) -> int:
        if heading is None:
            return 0
        c = target
        while parent[c] != src:
            c = parent[c]  # type: ignore[assignment]
        return turn_between(heading, heading_of(src, c))

    target = min(candidates, key=first_turn)
    return lawn.shortest_path(src, target)


def finish_greedily(lawn: Lawn, path: list[Cell]) -> list[Cell]:
    """Extend ``path`` until every reachable cell is mowed (nearest-first)."""
    mowed = set(path)
    heading = heading_of(path[-2], path[-1]) if len(path) > 1 else None
    while True:
        leg = nearest_unmowed(lawn, path[-1], mowed, heading)
        if leg is None:
            return path
        path.extend(leg[1:])
        mowed.update(leg)
        heading = heading_of(leg[-2], leg[-1])


def transpose_lawn(lawn: Lawn) -> Lawn:
    return Lawn(lawn.free.T.copy(), (lawn.start[1], lawn.start[0]), lawn.name)


def transpose_path(path: list[Cell]) -> list[Cell]:
    return [(c, r) for r, c in path]
