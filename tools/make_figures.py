"""Draw the figures of the documentation (docs/figures/*.svg).

    python tools/make_figures.py

Every figure is computed from the code it illustrates (decomposition,
spanning tree, environment observation, receptive field...), so the
pictures follow the implementation.  Needs numpy and matplotlib; the
observation figure also needs gymnasium.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lawnmower import Lawn, evaluate, plan  # noqa: E402
from lawnmower.grid import DIRECTIONS  # noqa: E402
from lawnmower.planners.boustrophedon import boustrophedon, decompose  # noqa: E402
from lawnmower.planners.stc import circumnavigate, mega_cells, spanning_forest  # noqa: E402
from lawnmower.viz import COLORS  # noqa: E402

OUT = ROOT / "docs" / "figures"
plt.rcParams.update({"font.size": 10, "svg.fonttype": "path", "svg.hashsalt": "lawnmower",
                     "figure.facecolor": "white",
                     "savefig.facecolor": "white", "axes.titlesize": 10})
CELL_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#edc948",
               "#76b7b2", "#ff9da7", "#9c755f", "#bab0ac"]


def load(name: str) -> Lawn:
    return Lawn.from_file(ROOT / "maps" / f"{name}.txt")


def rect_lawn(rows: list[str]) -> Lawn:
    return Lawn.from_string("\n".join(rows))


def draw_lawn(ax, lawn: Lawn, fill: dict | None = None, grid: bool = True) -> None:
    """Obstacles dark, lawn green; ``fill`` overrides the colour of some cells."""
    img = np.empty((*lawn.shape, 3))
    img[:] = to_rgb(COLORS["obstacle"])
    for c in lawn.lawn_cells:
        img[c] = to_rgb(COLORS["unmowed"]) if c in lawn.reachable else to_rgb(COLORS["unreachable"])
    for c, col in (fill or {}).items():
        img[c] = to_rgb(col)
    ax.imshow(img, interpolation="nearest")
    h, w = lawn.shape
    if grid:
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.4, alpha=0.6)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    for s in ax.spines.values():
        s.set_visible(False)


def draw_path(ax, path, color="#1565c0", lw=1.6, start=True) -> None:
    pts = np.array([(c, r) for r, c in path], dtype=float)
    ax.plot(pts[:, 0], pts[:, 1], "-", color=color, lw=lw, solid_capstyle="round")
    if start:
        ax.plot(*pts[0], "o", color=COLORS["start"], ms=8, mec="white", zorder=5)
        ax.plot(*pts[-1], "s", color=COLORS["end"], ms=7, mec="white", zorder=5)


def arrow(ax, a, b, color="black", lw=1.4, style="-|>", ms=10, ls="-") -> None:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms, color=color,
                                 lw=lw, linestyle=ls, shrinkA=0, shrinkB=0))


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # no date in the file, and fixed ids, so that redrawing gives identical files
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.08, metadata={"Date": None})
    plt.close(fig)
    # matplotlib leaves trailing spaces, which git reports as whitespace errors
    svg = OUT / name
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    print("wrote", OUT / name)


# ------------------------------------------------------------------ the model
def fig_moves() -> None:
    lawn = rect_lawn(["+++++++", "+LLLLL+", "+LLL+L+", "+LLLLL+", "+LLLLL+", "+LLLLL+",
                      "+++++++"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8))
    for ax, (pos, title) in zip(axes, [((4, 2), "8 moves from an open cell"),
                                       ((3, 4), "below an obstacle: N is blocked, and NW, NE\n"
                                                "would clip its corners (✕)")]):
        draw_lawn(ax, lawn)
        ax.plot(pos[1], pos[0], "o", color="#1565c0", ms=13, mec="white", zorder=5)
        for dr, dc in DIRECTIONS:
            nb = (pos[0] + dr, pos[1] + dc)
            ok = lawn.can_move(pos, nb)
            if not lawn.is_free(nb):
                continue
            arrow(ax, (pos[1] + 0.25 * dc, pos[0] + 0.25 * dr),
                  (pos[1] + 0.8 * dc, pos[0] + 0.8 * dr),
                  color="white" if ok else "#ff5252", lw=1.8 if ok else 1.4,
                  ls="-" if ok else (0, (2, 2)))
            if not ok:
                ax.text(nb[1], nb[0], "✕", color="#ff5252", ha="center", va="center",
                        fontsize=14, fontweight="bold")
        ax.set_title(title)
    save(fig, "moves.svg")


# --------------------------------------------------------------------- greedy
def fig_greedy() -> None:
    lawn = load("maze7")
    res = plan(lawn, "greedy")
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    visits = res.visits
    fill = {c: COLORS["mowed"] if n == 1 else COLORS["overlap"] for c, n in visits.items()}
    draw_lawn(ax, lawn, fill)
    draw_path(ax, res.path)
    m = res.metrics
    ax.set_title(f"greedy on maze7: {m.steps} steps (N−1 = {m.target - 1}), {m.turns} turns")
    save(fig, "greedy_maze7.svg")


# -------------------------------------------------------------- boustrophedon
def fig_boustrophedon() -> None:
    lawn = load("maze0")
    cells = decompose(lawn)
    fill = {}
    for k, cell in enumerate(cells):
        for s in cell:
            for r in range(s.top, s.bottom + 1):
                fill[(r, s.col)] = CELL_COLORS[k % len(CELL_COLORS)]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8))
    draw_lawn(axes[0], lawn, fill)
    for k, cell in enumerate(cells):
        rs = [r for s in cell for r in (s.top, s.bottom)]
        cs = [s.col for s in cell]
        axes[0].text(np.mean(cs), (min(rs) + max(rs)) / 2, str(k + 1), ha="center",
                     va="center", color="white", fontsize=12, fontweight="bold")
    # critical events: where the segments of two neighbouring columns do not match 1-to-1
    for k, cell in enumerate(cells):
        x = cell[0].col - 0.5
        if cell[0].col > 1:
            axes[0].plot([x, x], [cell[0].top - 0.5, cell[0].bottom + 0.5], color="white", lw=2.2)
    axes[0].set_title(f"vertical decomposition: {len(cells)} cells\n"
                      "(white lines: critical events, where a cell ends)")
    path = boustrophedon(lawn, orientation="vertical")
    m, visits = evaluate(lawn, path)
    fill2 = {c: COLORS["mowed"] if n == 1 else COLORS["overlap"] for c, n in visits.items()}
    draw_lawn(axes[1], lawn, fill2)
    draw_path(axes[1], path, lw=1.3)
    axes[1].set_title(f"cells swept one after another\n{m.steps} steps, {m.turns} turns "
                      f"(orange: mowed twice)")
    save(fig, "boustrophedon_maze0.svg")


def fig_cell_sweep() -> None:
    """The 4 ways to sweep one cell, with their entry (●) and exit (■) corners."""
    from lawnmower.planners.boustrophedon import Segment, entry_exit, sweep_cell
    lawn = rect_lawn(["+" * 7] + ["+LLLLL+"] * 4 + ["+" * 7])
    segs = [Segment(c, 1, 4) for c in range(1, 6)]
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.6))
    for ax, (rev, down) in zip(axes, [(False, True), (False, False), (True, True), (True, False)]):
        entry, _ = entry_exit(segs, rev, down)
        path = [entry]
        sweep_cell(lawn, path, segs, rev, down)
        draw_lawn(ax, lawn)
        draw_path(ax, path, color="white", lw=1.8)
        ax.set_title(f"{'right→left' if rev else 'left→right'}, first pass "
                     f"{'down' if down else 'up'}", fontsize=9)
    save(fig, "cell_sweeps.svg")


# ------------------------------------------------------------------------ STC
STC_LAWN = ["++++++++++++",
            "+LLLLLLLLLL+",
            "+LLLLLLLLLL+",
            "+LLLL++LLLL+",
            "+LLLL++LLLL+",
            "+LLLLLLLLLL+",
            "+LLLLLLLLLL+",
            "+LLLLLLLLLL+",
            "+LLLLLLLLLL+",
            "++++++++++++"]


def fig_stc() -> None:
    lawn = rect_lawn(STC_LAWN)
    off = (1, 1)
    nodes = mega_cells(lawn, *off)
    tree = spanning_forest(nodes, horizontal_first=True)
    loop = circumnavigate(nodes, tree, *off)[0]

    def centre(big):
        return (off[1] + 2 * big[1] + 0.5, off[0] + 2 * big[0] + 0.5)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    for ax in axes:
        draw_lawn(ax, lawn)
        for big in nodes:
            x, y = centre(big)
            ax.add_patch(Rectangle((x - 1, y - 1), 2, 2, fill=False, ec="white", lw=1.8))
    for big in nodes:
        axes[0].plot(*centre(big), "o", color="white", ms=7)
    axes[0].set_title(f"1. 2×2 mega-cells ({len(nodes)})")
    for ax in axes[1:]:
        for e in tree:
            a, b = sorted(e)
            (x1, y1), (x2, y2) = centre(a), centre(b)
            ax.plot([x1, x2], [y1, y2], "-", color="#ffd54f", lw=3 if ax is axes[1] else 2,
                    alpha=1 if ax is axes[1] else 0.7)
        for big in nodes:
            ax.plot(*centre(big), "o", color="#ffd54f", ms=6 if ax is axes[1] else 4)
    axes[1].set_title("2. spanning tree (comb: horizontal edges first)")
    pts = np.array([(c, r) for r, c in loop + loop[:1]], dtype=float)
    axes[2].plot(pts[:, 0], pts[:, 1], "-", color="#1565c0", lw=2.2)
    # direction marks
    for i in range(0, len(loop), 7):
        (r1, c1), (r2, c2) = loop[i], loop[(i + 1) % len(loop)]
        arrow(axes[2], (c1, r1), (c1 + 0.6 * (c2 - c1), r1 + 0.6 * (r2 - r1)), color="#1565c0",
              ms=12)
    axes[2].set_title(f"3. walk around the tree: one loop through\nall {len(loop)} sub-cells, "
                      "each visited once")
    save(fig, "stc_construction.svg")


def fig_stc_links() -> None:
    """Zoom on the link rule of one mega-cell."""
    lawn = rect_lawn(["++++++++", "+LLLLLL+", "+LLLLLL+", "+LLLLLL+", "+LLLLLL+", "++++++++"])
    off = (1, 1)
    nodes = {(0, 0), (0, 1), (0, 2), (1, 1)}
    tree = {frozenset({(0, 0), (0, 1)}), frozenset({(0, 1), (0, 2)}), frozenset({(0, 1), (1, 1)})}
    loop = circumnavigate(nodes, tree, *off)[0]
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    fill = {(r, c): COLORS["obstacle"] for r in (3, 4) for c in (1, 2, 5, 6)}
    draw_lawn(ax, lawn, fill)

    def centre(big):
        return (off[1] + 2 * big[1] + 0.5, off[0] + 2 * big[0] + 0.5)

    for big in nodes:
        x, y = centre(big)
        ax.add_patch(Rectangle((x - 1, y - 1), 2, 2, fill=False, ec="white", lw=1.5))
    for e in tree:
        (x1, y1), (x2, y2) = (centre(b) for b in sorted(e))
        ax.plot([x1, x2], [y1, y2], "-", color="#ffd54f", lw=3)
    pts = np.array([(c, r) for r, c in loop + loop[:1]], dtype=float)
    ax.plot(pts[:, 0], pts[:, 1], "-", color="#1565c0", lw=2.2)
    ax.set_title("a tree edge is crossed, a side without\none is followed: the loop hugs the tree",
                 fontsize=9)
    save(fig, "stc_links.svg")


# -------------------------------------------------------------------- the env
def fig_observation() -> None:
    try:
        from lawnmower.rl import LawnmowerEnv, Rewards, path_to_actions
    except ImportError:
        print("skip observation figure (needs gymnasium)")
        return
    lawn = load("maze7")
    rw = Rewards(turn_repeat=-0.1)
    env = LawnmowerEnv(lawn, rewards=rw, turn_memory=rw.memory_horizon())
    obs, _ = env.reset()
    for a in path_to_actions(plan(lawn, "greedy").path)[:23]:
        obs, *_ = env.step(a)
    names = ["0  obstacle", "1  mowed", "2  mower", "3  cell ahead\n(heading)",
             "4  steps since last turn\n(only with turn_repeat)"]
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.9))
    for k, ax in enumerate(axes):
        ax.imshow(obs[k], cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        h, w = obs[k].shape
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(which="minor", color="#bbbbbb", linewidth=0.4)
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        ax.set_title(names[k], fontsize=9)
        if k == 4:
            r, c = env.pos
            ax.text(c, r, f"{obs[4][r, c]:.1f}", color="#e53935", ha="center", va="center",
                    fontsize=8, fontweight="bold")
    save(fig, "observation.svg")


def fig_turn_cost() -> None:
    try:
        from lawnmower.rl import Rewards
    except ImportError:
        print("skip turn-cost figure (needs gymnasium)")
        return
    x = np.arange(1, 21)
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for rep, col in [(0.0, "#888888"), (-0.1, "#4e79a7"), (-0.2, "#e15759")]:
        rw = Rewards(turn=-0.05, turn_repeat=rep, turn_decay=0.01)
        y = [-rw.turn_cost(v) for v in x]
        label = "turn_repeat = 0 (flat)" if rep == 0 else f"turn_repeat = {-rep}"
        ax.plot(x, y, "o-", ms=3.5, color=col, label=label)
    ax.set_xlabel("steps x since the previous turn")
    ax.set_ylabel("cost per 45° turned")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(0, 0.21)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("cost of a turn: max(p₀, p_max − d·x), p₀ = 0.05, d = 0.01")
    save(fig, "turn_cost.svg")


# -------------------------------------------------------------------- network
def receptive_radii(dilations=(1, 2, 4, 8, 16)) -> list[int]:
    """Radius of the region that can influence one output cell, layer by layer.

    Computed by propagating a set of cells through the actual kernel offsets
    (stem 3x3, then per block a 3x3 conv with dilation d and a plain 3x3 conv,
    plus the residual identity), not by a formula.
    """
    size = 2 * (1 + sum(d + 1 for d in dilations)) + 3
    c = size // 2
    region = np.zeros((size, size), dtype=bool)
    region[c, c] = True

    def conv(reg, d):
        out = reg.copy()
        for dr in (-d, 0, d):
            for dc in (-d, 0, d):
                out |= np.roll(np.roll(reg, dr, 0), dc, 1)
        return out

    radii = []
    region = conv(region, 1)                         # stem
    radii.append(int(np.abs(np.argwhere(region) - c).max()))
    for d in dilations:
        region = conv(conv(region, d), 1) | region   # block + residual
        assert region[c - radii[-1] - d - 1: c + radii[-1] + d + 2,
                      c - radii[-1] - d - 1: c + radii[-1] + d + 2].all()  # a full square
        radii.append(int(np.abs(np.argwhere(region) - c).max()))
    return radii


def fig_receptive_field() -> None:
    radii = receptive_radii()
    lawn = load("maze1")
    h, w = lawn.shape
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.9), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    draw_lawn(ax, lawn, grid=False)
    pos = (14, 15)
    ax.plot(pos[1], pos[0], "o", color="#1565c0", ms=9, mec="white", zorder=6)
    cols = ["#cfd8dc", "#ffd54f", "#ffb74d", "#ff8a65", "#e57373", "#ba68c8"]
    labels = ["stem"] + [f"block {i} (d={d})" for i, d in enumerate((1, 2, 4, 8, 16), 1)]
    for r, col, lab in zip(radii, cols, labels):
        ax.add_patch(Rectangle((pos[1] - r - 0.5, pos[0] - r - 0.5), 2 * r + 1, 2 * r + 1,
                               fill=False, ec=col, lw=1.8, label=f"{lab}: {2 * r + 1}×{2 * r + 1}"))
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=8, frameon=False,
              ncol=3)
    ax.set_title(f"cells that can influence the features at the mower\n(maze1, {h}×{w}: from "
                 f"block 4 on, the square is larger than the lawn)")
    ax = axes[1]
    blocks = np.arange(0, 6)
    plain = [1 + 2 * k for k in blocks]
    ax.plot(blocks, [2 * r + 1 for r in radii], "o-", color="#e15759",
            label="dilations 1, 2, 4, 8, 16")
    ax.plot(blocks, [2 * r + 1 for r in plain], "o-", color="#4e79a7",
            label="no dilation (all d = 1)")
    ax.set_xticks(blocks, ["stem"] + [f"block {i}" for i in range(1, 6)], fontsize=8)
    ax.set_ylabel("receptive field width (cells)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("same number of layers and weights,\nexponential instead of linear growth")
    save(fig, "receptive_field.svg")


def fig_size_independence() -> None:
    small, large = load("maze3"), load("maze0")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw={"width_ratios": [9, 21]})
    for ax, lawn, mower, kern in [(axes[0], small, (6, 2), [(2, 5)]),
                                  (axes[1], large, (9, 12), [(2, 3), (15, 7), (12, 17)])]:
        draw_lawn(ax, lawn)
        h, w = lawn.shape
        for (r, c) in kern:   # the same 3x3 kernel, slid everywhere
            ax.add_patch(Rectangle((c - 1.5, r - 1.5), 3, 3, fill=False, ec="#ffd54f", lw=2,
                                   ls=(0, (3, 2))))
        r, c = mower
        ax.add_patch(Rectangle((c - 1.5, r - 1.5), 3, 3, fill=True, fc="#e5393555", ec="#e53935",
                               lw=2.2))
        ax.plot(c, r, "o", color="#1565c0", ms=8, mec="white", zorder=6)
        ax.set_title(f"{lawn.name}: {h}×{w} → features (C, {h}, {w})\n"
                     f"policy reads 9·C numbers, value pools over {len(lawn.reachable)} cells",
                     fontsize=9)
    fig.text(0.5, 0.03, "dashed yellow: the same 3×3 kernel (same weights) at a few of its "
             "positions;  red: the mower's 3×3 neighbourhood read by the policy head",
             ha="center", fontsize=9)
    save(fig, "size_independence.svg")


def main() -> None:
    fig_moves()
    fig_greedy()
    fig_boustrophedon()
    fig_cell_sweep()
    fig_stc()
    fig_stc_links()
    fig_observation()
    fig_turn_cost()
    fig_receptive_field()
    fig_size_independence()


if __name__ == "__main__":
    main()
