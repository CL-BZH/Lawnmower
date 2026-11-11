# Lawnmower — coverage path planning on a grid

Plan a path that mows every reachable cell of a lawn with as little wasted
travel as possible, compare planners on the same footing, and train
reinforcement-learning agents on the same model.

![STC on maze8](docs/stc_maze8.gif)

## Quick start

```bash
pip install -e .            # numpy + matplotlib
pip install -e '.[rl]'      # + gymnasium, for the RL environment
pip install -e '.[dev]'     # + pytest

python -m lawnmower run maps/maze8.txt -p stc --plot stc.png
python -m lawnmower run maps/maze7.txt -p greedy --ascii --gif greedy.gif
python -m lawnmower compare maps/maze6.txt maps/maze8.txt --plot compare.png
pytest
```

```python
from lawnmower import Lawn, plan

lawn = Lawn.from_file("maps/maze8.txt")          # or Lawn.from_string(...)
res = plan(lawn, "boustrophedon")
print(res.metrics.as_dict(), res.path[:5])
```

## The model

* One grid cell is exactly the area the blade cuts. The mower sits on one
  cell and mows it on arrival.
* It moves to any of its 8 neighbours per step. A diagonal move is only legal
  if both orthogonal cells it squeezes between are free — the mower never
  clips an obstacle corner.
* As a consequence, the lawn to mow is the 4-connected region around the
  start. Lawn cut off from it is reported as *unreachable* and excluded from
  coverage (see `maps/maze2.txt`).

Map files: `L` lawn, `S` lawn + start, `+` obstacle (`.` and `#` also work).
Without `S` the start is the first lawn cell in reading order; `--start ROW,COL`
overrides both.

Every planner just returns a list of cells. `lawnmower.sim.evaluate` replays
it, **rejects any illegal step**, and computes all metrics, so planners (and RL
agents) are always scored by the same rules:

| metric     | meaning |
|------------|---------|
| `coverage` | share of reachable cells visited |
| `steps`    | number of moves |
| `length`   | distance travelled (diagonal = √2) |
| `turns` / `turn_angle` | heading changes, and their total in degrees |
| `revisits` | moves landing on an already-mowed cell |
| `overhead` | `steps / (N − 1)`; 1.0 means no cell was visited twice |

## Planners

| name | idea |
|------|------|
| `random` | Random walk preferring unmowed neighbours — the "bump-and-turn robot" baseline. Seeded; may stop before full coverage. |
| `greedy` | Keep straight while the cell ahead is unmowed, else take the unmowed neighbour needing the smallest turn (hugging the unmowed boundary); when boxed in, shortest path to the nearest unmowed cell. |
| `boustrophedon` | Boustrophedon cellular decomposition (Choset & Pignon 1997): split the lawn at obstacle "critical events" into cells that can each be mowed with straight back-and-forth passes, then sweep them. The cell order and entry corner is a small generalised TSP, solved by nearest-neighbour and by a rollout of it; vertical and horizontal passes are both tried. |
| `stc` | Spanning Tree Coverage (Gabriely & Rimon 2001): build a spanning tree on 2×2 blocks of lawn and walk around it, which visits every cell of those blocks exactly once. Strips the blocks miss (odd widths, obstacle edges) are mowed by short detours along the way. All 4 block alignments and 2 tree shapes are tried. |

`boustrophedon` and `stc` pick their best candidate on steps, then turns.
`--turn-weight W` charges each turn `W` steps instead, for mowers where
turning is slow.

### Results

`python -m lawnmower compare ...` (N = cells to mow):

| map (N) | planner | steps | turns | revisits | overhead |
|---|---|---:|---:|---:|---:|
| maze2 (404) | greedy | 457 | 133 | 54 | 1.13 |
| | boustrophedon | 443 | **71** | 40 | 1.10 |
| | stc | **427** | 182 | **24** | **1.06** |
| | random | 12 721 | 11 089 | 12 318 | 31.6 |
| maze8 (406) | greedy | 465 | 127 | 60 | 1.15 |
| | boustrophedon | 464 | **79** | 59 | 1.15 |
| | stc | **420** | 180 | **15** | **1.04** |
| test_map (347) | greedy | **440** | 173 | **94** | **1.27** |
| | boustrophedon | 442 | **167** | 96 | 1.28 |
| | stc | 478 | 283 | 132 | 1.38 |

STC wastes the fewest moves on open, blocky lawns; boustrophedon gives by far
the straightest passes; on a lawn peppered with single obstacles
(`test_map`) the simple greedy rule is as good as anything.
On an open rectangle both boustrophedon and STC are exact (no revisit).

![planners on maze8](docs/compare.png)

## Reinforcement learning

`lawnmower.rl.LawnmowerEnv` is a Gymnasium environment on the same model
(also registered as `Lawnmower-v0`):

* **observation** `(4, H, W)` float planes — obstacles, mowed, mower, cell
  ahead (heading) — or an egocentric `(4, 2r+1, 2r+1)` window with
  `view_radius=r`, which lets one policy work across maps of any size;
* **action** `Discrete(8)`; illegal moves leave the mower in place and are
  penalised; `env.action_masks()` / `info["action_mask"]` give the legal ones
  (works with sb3-contrib `MaskablePPO`);
* **reward** `+1` per new cell, `−0.1` per step, `−0.05` per 45° turned,
  `−1` per illegal move, `+10` on completion — all set via `Rewards(...)`;
* pass several lawns and/or `random_start=True` to train for generalisation;
* `env.result()` scores the episode with the same metrics as the planners, and
  `path_to_actions(planner_path)` turns any planner into expert
  demonstrations for imitation learning.

```python
from lawnmower import Lawn
from lawnmower.rl import LawnmowerEnv

env = LawnmowerEnv([Lawn.from_file(f"maps/maze{i}.txt") for i in (0, 3, 4, 7)],
                   view_radius=5, random_start=True)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

No learning library is a dependency; plug in your own DQN / PPO / A2C.

## Layout

```
lawnmower/
  grid.py          Lawn: map parsing, legal moves, reachability, shortest paths
  sim.py           path validation and metrics (single source of truth)
  planners/        baselines.py (random, greedy), boustrophedon.py, stc.py
  rl.py            Gymnasium environment (optional)
  viz.py           figures and GIF animations
  __main__.py      command line interface
maps/              example lawns
tests/             pytest suite (all planners on all maps, legality, RL env)
```

The previous code (the turtle version and `lawnmower_optimizer/`) remains on
the `master` branch.
