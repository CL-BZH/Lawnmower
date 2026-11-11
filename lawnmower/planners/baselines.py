"""Reference planners: a biased random walk and a greedy local planner."""
from __future__ import annotations

import random

from ..grid import DIRECTIONS, Cell, Lawn
from ..sim import heading_of, turn_between
from .common import nearest_unmowed


def random_walk(lawn: Lawn, seed: int = 0, max_steps_factor: int = 50) -> list[Cell]:
    """Random walk that prefers unmowed neighbours.

    At each step pick uniformly among unmowed neighbours, or among all
    neighbours if none is unmowed.  Stops at full coverage or after
    ``max_steps_factor * N`` steps, so the result may be incomplete.
    This is the "robot mower bouncing around" baseline.
    """
    rng = random.Random(seed)
    target = lawn.reachable
    path = [lawn.start]
    mowed = {lawn.start}
    budget = max_steps_factor * len(target)
    while len(mowed) < len(target) and len(path) <= budget:
        nbs = list(lawn.neighbors(path[-1]))
        fresh = [n for n in nbs if n not in mowed]
        nxt = rng.choice(fresh or nbs)
        path.append(nxt)
        mowed.add(nxt)
    return path


def greedy(lawn: Lawn, prefer_walls: bool = True) -> list[Cell]:
    """Greedy local planner.

    1. Keep going straight while the cell ahead is unmowed.
    2. Otherwise move to the unmowed neighbour needing the smallest turn,
       preferring orthogonal moves and, if ``prefer_walls``, the neighbour
       with the fewest unmowed neighbours of its own (hugging the boundary
       of the unmowed area, so we do not strand isolated patches).
    3. If no neighbour is unmowed, take the shortest path to the nearest
       unmowed cell.
    """
    target = lawn.reachable
    path = [lawn.start]
    mowed = {lawn.start}
    heading = 0

    def free_degree(cell: Cell) -> int:
        return sum(1 for n in lawn.neighbors(cell, diagonal=False) if n not in mowed)

    while len(mowed) < len(target):
        cur = path[-1]
        fresh = [n for n in lawn.neighbors(cur) if n not in mowed]
        if fresh:
            ahead = (cur[0] + DIRECTIONS[heading][0], cur[1] + DIRECTIONS[heading][1])

            scored = []
            for i, n in enumerate(fresh):
                h = heading_of(cur, n)
                scored.append(((
                    n != ahead,
                    h % 2,  # diagonal moves last
                    free_degree(n) if prefer_walls else 0,
                    turn_between(heading, h),
                ), i, n))
            nxt = min(scored)[2]
            path.append(nxt)
            mowed.add(nxt)
            heading = heading_of(cur, nxt)
        else:
            leg = nearest_unmowed(lawn, cur, mowed, heading)
            assert leg is not None
            path.extend(leg[1:])
            mowed.update(leg)
            heading = heading_of(leg[-2], leg[-1])
    return path
