"""Gymnasium environment for learning to mow.

Same model as the planners (one cell = one blade footprint, 8 moves, no
corner cutting), so a learned policy can be scored with exactly the same
:func:`lawnmower.sim.evaluate` as the classical planners.

Requires ``gymnasium`` (``pip install -e .[rl]``).  The environment is
framework-agnostic: observations are NumPy arrays, so any DQN / PPO / A2C
implementation (your own, Stable-Baselines3, CleanRL...) can drive it.

Observation
    ``float32`` array of shape ``(4, H, W)`` (full view) or
    ``(4, 2r+1, 2r+1)`` (egocentric window of radius ``r`` = ``view_radius``):

    0. obstacle or out of the lawn
    1. already mowed
    2. mower position
    3. the cell straight ahead of the mower (encodes its heading spatially)

Action
    ``Discrete(8)``: index into :data:`lawnmower.grid.DIRECTIONS`
    (0 = E, then counter-clockwise).  An illegal move leaves the mower in
    place and is penalised.  :meth:`LawnmowerEnv.action_masks` gives the legal
    actions (compatible with sb3-contrib's ``MaskablePPO``) and the same mask
    is in ``info["action_mask"]``.

Reward (all configurable through :class:`Rewards`)
    ``+new_cell`` when the mower reaches an unmowed cell, ``step`` every
    step, ``turn`` per 45 degrees of heading change, ``invalid`` for an
    illegal move, ``complete`` once everything reachable is mowed.

With ``obs_size=(H, W)`` the full view is padded to that size, so lawns
passed later through ``reset(options={"lawn": ...})`` fit the same shape.

Episodes end (``terminated``) at full coverage and are truncated after
``max_steps_factor * N`` steps, N being the number of reachable cells.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "lawnmower.rl needs gymnasium: pip install -e '.[rl]'  (or pip install gymnasium)"
    ) from exc

from .grid import DIRECTIONS, Cell, Lawn
from .sim import Result, evaluate, heading_of, turn_between


@dataclass(frozen=True)
class Rewards:
    new_cell: float = 1.0
    step: float = -0.1
    turn: float = -0.05      # per 45 degrees
    invalid: float = -1.0
    complete: float = 10.0


class LawnmowerEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        lawns: Lawn | Sequence[Lawn],
        rewards: Rewards = Rewards(),
        max_steps_factor: float = 4.0,
        view_radius: int | None = None,
        random_start: bool = False,
        render_mode: str | None = None,
        obs_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.lawns = [lawns] if isinstance(lawns, Lawn) else list(lawns)
        if not self.lawns:
            raise ValueError("need at least one lawn")
        self.rewards = rewards
        self.max_steps_factor = max_steps_factor
        self.view_radius = view_radius
        self.random_start = random_start
        self.render_mode = render_mode

        if view_radius is None:
            # pad every lawn to a common size so the observation shape is fixed
            # (obs_size lets lawns passed later through reset(options=...) fit too)
            self._H, self._W = obs_size or (max(lw.shape[0] for lw in self.lawns),
                                            max(lw.shape[1] for lw in self.lawns))
            shape = (4, self._H, self._W)
        else:
            side = 2 * view_radius + 1
            shape = (4, side, side)
        self.observation_space = spaces.Box(0.0, 1.0, shape=shape, dtype=np.float32)
        self.action_space = spaces.Discrete(len(DIRECTIONS))

        self.lawn: Lawn = self.lawns[0]
        self.pos: Cell = self.lawn.start
        self.heading: int | None = None
        self.path: list[Cell] = []
        self.steps = 0
        self._mowed: np.ndarray = np.zeros(self.lawn.shape, dtype=bool)
        self._todo = 0

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        lawn = options.get("lawn")
        if lawn is None:
            lawn = self.lawns[int(self.np_random.integers(len(self.lawns)))]
        if options.get("start") is not None:
            lawn = lawn.with_start(tuple(options["start"]))
        elif self.random_start:
            cells = sorted(lawn.reachable)
            lawn = lawn.with_start(cells[int(self.np_random.integers(len(cells)))])
        if self.view_radius is None and (lawn.shape[0] > self._H or lawn.shape[1] > self._W):
            raise ValueError(f"lawn {lawn.shape} larger than the observation {(self._H, self._W)}")
        self.lawn = lawn
        self.pos = lawn.start
        self.heading = None
        self.path = [lawn.start]
        self.steps = 0
        self._reach = np.zeros(lawn.shape, dtype=bool)
        for c in lawn.reachable:
            self._reach[c] = True
        self._mowed = np.zeros(lawn.shape, dtype=bool)
        self._mowed[lawn.start] = True
        self._todo = len(lawn.reachable) - 1
        self._max_steps = int(self.max_steps_factor * len(lawn.reachable))
        return self._obs(), self._info()

    def step(self, action: int):
        a = int(action)
        dr, dc = DIRECTIONS[a]
        nxt = (self.pos[0] + dr, self.pos[1] + dc)
        rw = self.rewards
        reward = rw.step
        self.steps += 1
        if self.lawn.can_move(self.pos, nxt):
            if self.heading is not None:
                reward += rw.turn * turn_between(self.heading, a)
            self.heading = a
            self.pos = nxt
            self.path.append(nxt)
            if not self._mowed[nxt]:
                self._mowed[nxt] = True
                self._todo -= 1
                reward += rw.new_cell
        else:
            reward += rw.invalid
        terminated = self._todo == 0
        if terminated:
            reward += rw.complete
        truncated = not terminated and self.steps >= self._max_steps
        return self._obs(), float(reward), terminated, truncated, self._info()

    def action_masks(self) -> np.ndarray:
        r, c = self.pos
        return np.array([self.lawn.can_move(self.pos, (r + dr, c + dc))
                         for dr, dc in DIRECTIONS], dtype=bool)

    def result(self) -> Result:
        """Score the current episode's path like any planner's."""
        metrics, visits = evaluate(self.lawn, self.path)
        return Result("rl", self.lawn, list(self.path), metrics, visits)

    # ------------------------------------------------------------ internals
    def _planes(self) -> np.ndarray:
        h, w = self.lawn.shape
        planes = np.zeros((4, h, w), dtype=np.float32)
        planes[0] = ~self._reach
        planes[1] = self._mowed
        planes[2][self.pos] = 1.0
        if self.heading is not None:
            dr, dc = DIRECTIONS[self.heading]
            ahead = (self.pos[0] + dr, self.pos[1] + dc)
            if self.lawn.in_bounds(ahead):
                planes[3][ahead] = 1.0
        return planes

    def _obs(self) -> np.ndarray:
        planes = self._planes()
        if self.view_radius is None:
            h, w = self.lawn.shape
            out = np.zeros(self.observation_space.shape, dtype=np.float32)
            out[0] = 1.0  # padding = obstacle
            out[:, :h, :w] = planes
            return out
        rad = self.view_radius
        padded = np.zeros((4, planes.shape[1] + 2 * rad, planes.shape[2] + 2 * rad),
                          dtype=np.float32)
        padded[0] = 1.0
        padded[:, rad:-rad or None, rad:-rad or None] = planes
        r, c = self.pos
        return padded[:, r:r + 2 * rad + 1, c:c + 2 * rad + 1].copy()

    def _info(self) -> dict[str, Any]:
        total = len(self.lawn.reachable)
        return {
            "coverage": (total - self._todo) / total,
            "steps": self.steps,
            "action_mask": self.action_masks(),
        }

    def render(self):
        if self.render_mode == "ansi":
            marks = {(int(r), int(c)): "." for r, c in zip(*np.nonzero(self._mowed))}
            marks[self.pos] = "@"
            return self.lawn.render(marks)
        if self.render_mode == "rgb_array":
            img = np.zeros((*self.lawn.shape, 3), dtype=np.uint8)
            img[:] = (60, 60, 60)
            img[self._reach] = (30, 110, 40)
            img[self._mowed] = (150, 230, 150)
            img[self.pos] = (30, 90, 220)
            return np.kron(img, np.ones((8, 8, 1), dtype=np.uint8))
        return None


def path_to_actions(path: Sequence[Cell]) -> list[int]:
    """Convert a planner's path into the env's action sequence.

    Useful to replay a planner in the env, e.g. to collect expert
    demonstrations for imitation learning.
    """
    return [heading_of(a, b) for a, b in zip(path, path[1:])]


if "Lawnmower-v0" not in gym.registry:
    gym.register(id="Lawnmower-v0", entry_point="lawnmower.rl:LawnmowerEnv")
