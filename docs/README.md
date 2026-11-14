# How the mowing paths are computed

This folder explains, method by method, how this project computes a path
that mows a whole lawn. The README at the root of the repository says
*what* each method gives; these pages explain *how* and *why it works*. Each
page refers to the source file it describes, and every figure is drawn from
the code itself by [`tools/make_figures.py`](../tools/make_figures.py).

| | page | what you will find |
|---|---|---|
| 1 | [The problem](01-problem.md) | the grid model, which cells can be reached, the N − 1 lower bound, why the true optimum is out of reach, how paths are scored |
| 2 | [Random walk and greedy](02-greedy.md) | two local rules, nearest-unmowed search, shortest paths that prefer few turns |
| 3 | [Boustrophedon decomposition](03-boustrophedon.md) | cutting the lawn into cells that can be mowed in straight passes, and ordering them (a small generalised TSP) |
| 4 | [Spanning Tree Coverage](04-stc.md) | 2×2 blocks, a spanning tree, and why walking around it visits every cell exactly once |
| 5 | [Reinforcement learning](05-reinforcement-learning.md) | mowing as a Markov decision process, reward design, the Markov property and the 5th channel, behaviour cloning, PPO and GAE |
| 6 | [The network](06-network.md) | a fully convolutional network: why it runs on lawns of any size, dilated convolutions, receptive field |
| 7 | [Single-player MCTS](07-mcts.md) | searching with the network: PUCT, expansion with a value head, what changes when there is no opponent |

Reading order: 1 first; 2, 3 and 4 are independent; 5, 6 and 7 build on each
other.

**One rule underlies everything:** every method, planner or agent, returns a
plain list of cells, and a single function, `lawnmower.sim.evaluate`, checks
that every move is legal and computes the metrics. So the methods are always
compared by the same rules.

To redraw the figures after a change in the code:

```bash
python tools/make_figures.py      # needs matplotlib; the observation figure also gymnasium
```

## References

* A. Itai, C. H. Papadimitriou, J. L. Szwarcfiter, *Hamilton paths in grid
  graphs*, SIAM Journal on Computing, 1982.
* H. Choset, P. Pignon, *Coverage path planning: the boustrophedon cellular
  decomposition*, Field and Service Robotics, 1997.
* Y. Gabriely, E. Rimon, *Spanning-tree based coverage of continuous areas
  by a mobile robot*, Annals of Mathematics and Artificial Intelligence, 2001.
* E. Galceran, M. Carreras, *A survey on coverage path planning for
  robotics*, Robotics and Autonomous Systems, 2013.
* J. Schulman et al., *High-dimensional continuous control using generalized
  advantage estimation*, ICLR 2016.
* J. Schulman et al., *Proximal policy optimization algorithms*, 2017.
* J. Long, E. Shelhamer, T. Darrell, *Fully convolutional networks for
  semantic segmentation*, CVPR 2015.
* F. Yu, V. Koltun, *Multi-scale context aggregation by dilated
  convolutions*, ICLR 2016.
* K. He et al., *Deep residual learning for image recognition*, CVPR 2016.
* D. Silver et al., *A general reinforcement learning algorithm that masters
  chess, shogi, and Go through self-play*, Science, 2018 (AlphaZero).
* J. Schrittwieser et al., *Mastering Atari, Go, chess and shogi by planning
  with a learned model*, Nature, 2020 (MuZero).
* M. Schadd et al., *Single-player Monte-Carlo tree search*, Computers and
  Games, 2008.
