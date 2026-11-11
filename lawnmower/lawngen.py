"""Random lawns, to train agents that generalise beyond the example maps."""
from __future__ import annotations

from collections import deque

import numpy as np

from .grid import Lawn


def random_lawn(rng: np.random.Generator, min_size: int = 8, max_size: int = 32,
                obstacle_density: float = 0.15, max_tries: int = 50) -> Lawn:
    """A walled rectangular garden with rectangular obstacles and cut corners.

    The outer size (walls included) is drawn in ``[min_size, max_size]``.
    Lawn not 4-connected to the largest region is turned into obstacle, so
    every lawn cell is reachable.  The start is a random lawn cell.
    """
    for _ in range(max_tries):
        h, w = (int(v) for v in rng.integers(min_size, max_size + 1, size=2))
        free = np.zeros((h, w), dtype=bool)
        free[1:-1, 1:-1] = True
        inner = (h - 2) * (w - 2)
        # cut corners: L/T/U shaped gardens
        for _ in range(int(rng.integers(0, 3))):
            ch = int(rng.integers(2, max(3, h // 2)))
            cw = int(rng.integers(2, max(3, w // 2)))
            r0 = 0 if rng.random() < 0.5 else h - ch
            c0 = 0 if rng.random() < 0.5 else w - cw
            free[r0:r0 + ch, c0:c0 + cw] = False
        # rectangular obstacles (flower beds, trees, sheds) until the density is reached
        target = rng.uniform(0.3, 1.0) * obstacle_density * inner
        blocked = 0
        while blocked < target:
            oh = int(rng.integers(1, max(2, min(6, h // 3)) + 1))
            ow = int(rng.integers(1, max(2, min(6, w // 3)) + 1))
            r0 = int(rng.integers(1, max(2, h - oh)))
            c0 = int(rng.integers(1, max(2, w - ow)))
            blocked += int(free[r0:r0 + oh, c0:c0 + ow].sum())
            free[r0:r0 + oh, c0:c0 + ow] = False
        free = _largest_component(free)
        if free.sum() >= max(9, 0.35 * inner):
            cells = np.argwhere(free)
            r, c = cells[int(rng.integers(len(cells)))]
            return Lawn(free, (int(r), int(c)), name="random")
    raise RuntimeError("could not generate a lawn")


def _largest_component(free: np.ndarray) -> np.ndarray:
    h, w = free.shape
    label = np.full((h, w), -1, dtype=np.int32)
    sizes = []
    for r0, c0 in np.argwhere(free):
        if label[r0, c0] >= 0:
            continue
        k = len(sizes)
        label[r0, c0] = k
        todo = deque([(r0, c0)])
        n = 0
        while todo:
            r, c = todo.popleft()
            n += 1
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and free[rr, cc] and label[rr, cc] < 0:
                    label[rr, cc] = k
                    todo.append((rr, cc))
        sizes.append(n)
    if not sizes:
        return free
    return label == int(np.argmax(sizes))


def transform_lawn(lawn: Lawn, k: int, flip: bool) -> Lawn:
    """Rotate ``lawn`` by ``k`` quarter turns, then mirror it if ``flip``.

    The 8 combinations are the symmetries of the square (the dihedral group
    D4).  A mowing problem and its transformed copy are equally hard, so they
    are free extra training lawns.
    """
    free = lawn.free
    marker = np.zeros(free.shape, dtype=bool)
    marker[lawn.start] = True
    free, marker = np.rot90(free, k), np.rot90(marker, k)
    if flip:
        free, marker = np.fliplr(free), np.fliplr(marker)
    r, c = np.argwhere(marker)[0]
    return Lawn(np.ascontiguousarray(free), (int(r), int(c)), lawn.name)
