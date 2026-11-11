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

## Learned agent

`lawnmower/learn/` trains an agent for this environment with PPO:

* **Network**: fully convolutional, so the same network runs on lawns of any
  size. Residual 3×3 convolutions with growing dilations keep full
  resolution while seeing the whole lawn. The policy reads the features of
  the 8 cells the mower can move to, and illegal moves are masked out.
* **Training**: behaviour cloning of a planner, then masked PPO. Episodes
  use random gardens (`lawnmower/lawngen.py`) and the example maps, with
  random start cells, so the agent has to generalise rather than memorise
  one path.
* **Evaluation**: the agent's path goes through the same `evaluate` as the
  planners. `agent` plays its most likely move; `agent_best_of_N` samples N
  episodes and keeps the shortest (the planners also keep the best of
  several candidates).

```bash
pip install -e '.[train]'
python -m lawnmower.learn train --out runs/small --size 8 14 --p-random 0.7 \
    --maps maps/maze3.txt maps/maze4.txt maps/maze5.txt maps/maze7.txt maps/maze7a.txt maps/maze7b.txt \
    --bc-teacher greedy --steps 1500000
python -m lawnmower.learn eval models/agent_small.pt maps/maze7.txt --random 100 --size 8 14 --samples 16
```

### Results: small lawns (`models/agent_small.pt`)

Trained for 1.5M PPO steps on CPU (about 1.5 h on 2 cores) after cloning the
greedy planner. These are steps to mow everything from the map's usual start;
N − 1 is the minimum possible:

| map (N) | greedy | boustrophedon | stc | agent | agent best of 16 |
|---|---:|---:|---:|---:|---:|
| maze3 (43) | 45 | 48 | 44 | **42** | **42** |
| maze4 (42) | 45 | 44 | 43 | **41** | **41** |
| maze5 (9) | **8** | **8** | **8** | **8** | **8** |
| maze7 (51) | 52 | 55 | 54 | **50** | **50** |
| maze7a (25) | 27 | 26 | 26 | 25 | **24** |
| maze7b (26) | **25** | **25** | 26 | **25** | **25** |

The agent reaches the minimum N − 1 on every one of these maps. They were in
its training set, though always with random starts.

On **100 random gardens it never saw** (outer size 8–14), the mean overhead is
(1.0 = optimal):

| greedy | boustrophedon | stc | agent | agent best of 16 |
|---:|---:|---:|---:|---:|
| 1.119 | 1.110 | 1.092 | 1.060 (completes 99%) | **1.035** (completes 100%) |

The agent beats the best planner, but only at this scale. On the 30×30 maps it
was never trained on, sampling still finds shorter paths than STC on maze0
(266 vs 273) and maze2 (404 cells: 418 vs 427). On maze6 and maze8, however,
its most likely move can cycle and it stops at 91–95% coverage. Large lawns
need training on large lawns: `tools/train_kaggle.ipynb` does that on a GPU,
and can then fine-tune on a single map.

## Layout

```
lawnmower/
  grid.py          Lawn: map parsing, legal moves, reachability, shortest paths
  sim.py           path validation and metrics (single source of truth)
  planners/        baselines.py (random, greedy), boustrophedon.py, stc.py
  rl.py            Gymnasium environment (optional)
  lawngen.py       random gardens for training
  learn/           network, PPO + behaviour cloning, train/eval CLI (needs torch)
  viz.py           figures and GIF animations
  __main__.py      command line interface
maps/              example lawns
models/            trained agent checkpoints
tools/             Kaggle notebook for GPU training
tests/             pytest suite (all planners on all maps, legality, RL env)
```

The previous code (the turtle version and `lawnmower_optimizer/`) remains on
the `master` branch.
