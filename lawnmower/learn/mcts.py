"""Monte Carlo Tree Search guided by the trained network.

AlphaZero-style search (Silver et al., 2018): at every move the mower runs
``simulations`` simulations from the current state, each one

1. **selecting** moves down the tree with the PUCT rule
       a* = argmax_a  Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)),
   where P is the policy network's prior, N the visit counts and Q the
   (normalised) return observed after (s, a);
2. **expanding** the first new state with one network evaluation: the policy
   gives the priors of its moves, the value head estimates the return still
   to come, so no random roll-out is needed;
3. **backing up** that estimate along the path:  G = r + gamma * G.

It then plays the most visited move and keeps the subtree below it.

What changes for a one-player "game"
------------------------------------
* No opponent: there is no alternation of maximising and minimising players,
  every node maximises the same return.
* Returns are not bounded in [-1, 1] (they grow with the size of the lawn),
  so Q is rescaled to [0, 1] with the smallest and largest values seen in the
  tree (as in MuZero, Schrittwieser et al., 2020) before being added to the
  exploration term.
* The environment is deterministic: replaying the same moves always gives
  the same state, so the tree is exact and is reused after each move, and a
  node can be valued by the *best* return found below it rather than the
  mean (``backup="max"``, in the spirit of single-player MCTS, Schadd et al.,
  2008).  The mean is more robust to errors of the value head, the max more
  aggressive.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from ..grid import Lawn
from ..rl import LawnmowerEnv, Rewards
from ..sim import Result, evaluate
from .net import CoverageNet


@dataclass
class Edge:
    prior: float
    visits: int = 0
    total: float = 0.0          # sum of the returns backed up through this edge
    best: float = -math.inf     # largest of them
    reward: float = 0.0         # (scaled) reward of taking this move
    child: "Node | None" = None
    terminal: bool = False


@dataclass
class Node:
    value: float                # value-head estimate of this state
    edges: dict[int, Edge] = field(default_factory=dict)

    @property
    def visits(self) -> int:
        return sum(e.visits for e in self.edges.values())


def _snapshot(env: LawnmowerEnv) -> tuple:
    return (env.pos, env.heading, env.steps, env.since_turn, env._mowed.copy(), env._todo)


def _restore(env: LawnmowerEnv, snap: tuple) -> None:
    env.pos, env.heading, env.steps, env.since_turn, mowed, env._todo = snap
    env._mowed = mowed.copy()
    env.path = [env.pos]


class MCTS:
    def __init__(self, net: CoverageNet, rewards: Rewards = Rewards(), gamma: float = 0.995,
                 reward_scale: float = 0.1, simulations: int = 64, c_puct: float = 1.25,
                 backup: str = "mean", device: torch.device | None = None):
        if backup not in ("mean", "max"):
            raise ValueError("backup must be 'mean' or 'max'")
        self.net = net.eval()
        self.rewards = rewards
        self.gamma = gamma
        self.scale = reward_scale
        self.simulations = simulations
        self.c_puct = c_puct
        self.backup = backup
        self.device = device or next(net.parameters()).device
        self.root: Node | None = None
        self.q_min, self.q_max = math.inf, -math.inf

    @torch.no_grad()
    def _expand(self, obs: np.ndarray, mask: np.ndarray) -> Node:
        logits, value = self.net(torch.as_tensor(obs[None], device=self.device),
                                 torch.as_tensor(mask[None], device=self.device))
        prior = F.softmax(logits[0], dim=0).cpu().numpy()
        return Node(float(value), {int(a): Edge(float(prior[a])) for a in np.flatnonzero(mask)})

    def _q(self, edge: Edge) -> float:
        return edge.best if self.backup == "max" else edge.total / edge.visits

    def _normalise(self, q: float) -> float:
        if self.q_max > self.q_min:
            return (q - self.q_min) / (self.q_max - self.q_min)
        return 0.5

    def _select(self, node: Node) -> int:
        sqrt_n = math.sqrt(node.visits + 1)
        fpu = self._normalise(node.value)  # unvisited moves start at the node's own value
        best_a, best_score = -1, -math.inf
        for a, e in node.edges.items():
            q = self._normalise(self._q(e)) if e.visits else fpu
            score = q + self.c_puct * e.prior * sqrt_n / (1 + e.visits)
            if score > best_score:
                best_a, best_score = a, score
        return best_a

    def _simulate(self, sim: LawnmowerEnv, root_snap: tuple) -> None:
        _restore(sim, root_snap)
        node = self.root
        path: list[Edge] = []
        leaf_value = 0.0
        while True:
            a = self._select(node)
            edge = node.edges[a]
            path.append(edge)
            if edge.terminal:
                break
            obs, r, term, trunc, info = sim.step(a)
            if edge.child is None:
                edge.reward = r * self.scale
                if term or trunc:
                    edge.terminal = True
                else:
                    edge.child = self._expand(obs, info["action_mask"])
                    leaf_value = edge.child.value
                break
            node = edge.child
        g = leaf_value
        for edge in reversed(path):
            g = edge.reward + self.gamma * g
            edge.visits += 1
            edge.total += g
            edge.best = max(edge.best, g)
            q = self._q(edge)
            self.q_min, self.q_max = min(self.q_min, q), max(self.q_max, q)

    def choose(self, env: LawnmowerEnv, obs: np.ndarray, mask: np.ndarray) -> int:
        """Search from the env's current state and return the move to play."""
        if self.root is None:
            self.root = self._expand(obs, mask)
        # returns shrink as the lawn gets mowed: rebuild the normalisation bounds
        self.q_min, self.q_max = math.inf, -math.inf
        snap = _snapshot(env)
        sim = LawnmowerEnv(env.lawn, rewards=self.rewards,
                           max_steps_factor=env.max_steps_factor,
                           turn_memory=env.turn_memory)
        sim.reset(options={"lawn": env.lawn})
        for _ in range(self.simulations):
            self._simulate(sim, snap)
        a, edge = max(self.root.edges.items(),
                      key=lambda ae: (ae[1].visits, self._q(ae[1]) if ae[1].visits else -math.inf))
        self.root = edge.child  # reuse the subtree (None if the move ends the episode)
        return a


def mcts_mow(net: CoverageNet, lawn: Lawn, simulations: int = 64, c_puct: float = 1.25,
             backup: str = "mean", rewards: Rewards = Rewards(), gamma: float = 0.995,
             reward_scale: float = 0.1, max_steps_factor: float = 4.0,
             device: torch.device | None = None) -> Result:
    """Mow ``lawn`` choosing every move with MCTS; return the evaluated path."""
    env = LawnmowerEnv(lawn, rewards=rewards, max_steps_factor=max_steps_factor,
                       turn_memory=net.config["in_channels"] == 5)
    obs, info = env.reset()
    search = MCTS(net, rewards, gamma, reward_scale, simulations, c_puct, backup, device)
    done = False
    while not done:
        a = search.choose(env, obs, info["action_mask"])
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
    m, visits = evaluate(lawn, env.path)
    return Result(f"mcts{simulations}", lawn, list(env.path), m, visits)


def search_settings(checkpoint: dict) -> dict:
    """Rewards, gamma and reward scale the checkpoint's network was trained with."""
    ppo = checkpoint.get("ppo", {})
    rw = ppo.get("rewards")
    return {
        "rewards": Rewards(**rw) if isinstance(rw, dict) else Rewards(turn=-0.01),
        "gamma": ppo.get("gamma", 0.995),
        "reward_scale": ppo.get("reward_scale", 0.1),
    }
