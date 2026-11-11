"""Matplotlib figures and animations of mowing paths."""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from .grid import Lawn
from .sim import Result

COLORS = {
    "obstacle": "#3a3a3a",
    "unreachable": "#6b6b3a",
    "unmowed": "#2e7d32",
    "mowed": "#b9f6ca",
    "overlap": "#ffcc80",
    "start": "#7e57c2",
    "end": "#e53935",
}


def _base_image(lawn: Lawn, visits=None) -> np.ndarray:
    img = np.empty((*lawn.shape, 3))
    img[:] = to_rgb(COLORS["obstacle"])
    for c in lawn.unreachable:
        img[c] = to_rgb(COLORS["unreachable"])
    for c in lawn.reachable:
        n = visits.get(c, 0) if visits is not None else 0
        key = "unmowed" if n == 0 else "mowed" if n == 1 else "overlap"
        img[c] = to_rgb(COLORS[key])
    return img


def _decorate(ax, lawn: Lawn) -> None:
    h, w = lawn.shape
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.3, alpha=0.5)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)


def plot_result(result: Result, ax=None, title: str | None = None, show_path: bool = True):
    """Draw the lawn coloured by visit count, with the path on top."""
    lawn, path = result.lawn, result.path
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7 * lawn.shape[0] / lawn.shape[1] + 0.6))
    ax.imshow(_base_image(lawn, result.visits), interpolation="nearest")
    _decorate(ax, lawn)
    if show_path and len(path) > 1:
        pts = np.array([(c, r) for r, c in path], dtype=float)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", linewidths=1.6, alpha=0.9)
        lc.set_array(np.linspace(0, 1, len(segs)))
        ax.add_collection(lc)
    sr, sc = path[0]
    er, ec = path[-1]
    ax.plot(sc, sr, "o", color=COLORS["start"], ms=9, mec="white", label="start")
    ax.plot(ec, er, "s", color=COLORS["end"], ms=8, mec="white", label="end")
    m = result.metrics
    ax.set_title(title or (
        f"{result.planner} — {lawn.name}\n"
        f"coverage {100 * m.coverage:.1f}%  steps {m.steps}  "
        f"turns {m.turns}  revisits {m.revisits}"), fontsize=10)
    return ax


def legend_handles() -> list[Patch]:
    return [Patch(color=COLORS[k], label=lbl) for k, lbl in [
        ("unmowed", "not mowed"), ("mowed", "mowed once"), ("overlap", "mowed 2+ times"),
        ("obstacle", "obstacle"), ("unreachable", "unreachable lawn")]]


def plot_comparison(results: Sequence[Result], ncols: int | None = None):
    n = len(results)
    ncols = ncols or min(n, 3)
    nrows = -(-n // ncols)
    h, w = results[0].lawn.shape
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, layout="constrained",
                             figsize=(4.6 * ncols, (4.6 * h / w + 0.6) * nrows + 0.5))
    for ax, res in zip(axes.flat, results):
        plot_result(res, ax=ax)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.legend(handles=legend_handles(), loc="outside lower center", ncol=5, fontsize=9,
               frameon=False)
    return fig


def animate(result: Result, filename: str, fps: int = 20, max_frames: int = 300,
            dpi: int = 72) -> None:
    """Save a GIF of the mower working through its path."""
    lawn, path = result.lawn, result.path
    stride = max(1, -(-len(path) // max_frames))
    frames = list(range(0, len(path), stride))
    if frames[-1] != len(path) - 1:
        frames.append(len(path) - 1)

    fig, ax = plt.subplots(figsize=(6, 6 * lawn.shape[0] / lawn.shape[1] + 0.5))
    img = ax.imshow(_base_image(lawn), interpolation="nearest")
    _decorate(ax, lawn)
    (trail,) = ax.plot([], [], "-", color="#1565c0", lw=1.2, alpha=0.7)
    (mower,) = ax.plot([], [], "o", color="#1565c0", ms=9, mec="white")
    title = ax.set_title("")

    def update(k: int):
        visits: dict = {}
        for c in path[: k + 1]:
            visits[c] = visits.get(c, 0) + 1
        img.set_data(_base_image(lawn, visits))
        cols = [c for _, c in path[max(0, k - 30): k + 1]]
        rows = [r for r, _ in path[max(0, k - 30): k + 1]]
        trail.set_data(cols, rows)
        mower.set_data([path[k][1]], [path[k][0]])
        done = sum(1 for c in visits if c in lawn.reachable)
        title.set_text(f"{result.planner} — step {k}/{len(path) - 1} — "
                       f"{100 * done / len(lawn.reachable):.0f}% mowed")
        return img, trail, mower, title

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(filename, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
