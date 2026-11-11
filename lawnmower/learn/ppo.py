"""PPO (with action masking) and behaviour cloning for the lawnmower env.

Training loop
-------------
* ``n_envs`` environments run in lock-step; each episode is a new lawn
  (random garden and/or an example map, with a random start).
* Every ``n_steps`` steps, the batch of ``n_envs * n_steps`` transitions is
  used for ``epochs`` epochs of clipped-PPO updates (Schulman et al., 2017)
  with generalised advantage estimation (GAE).  Episodes cut by the step limit
  are bootstrapped with the value of their last observation.
* Optionally the network is first trained to imitate a planner (behaviour
  cloning), so that PPO starts from a policy that already completes lawns
  instead of from a random walk.
"""
from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .. import plan
from ..grid import Lawn
from ..lawngen import random_lawn, transform_lawn
from ..rl import LawnmowerEnv, Rewards, path_to_actions
from ..sim import Result, evaluate
from .net import CoverageNet


# --------------------------------------------------------------------- lawns
class LawnSource:
    """Draws the lawn of each new episode.

    ``maps``: example lawns, drawn with probability ``1 - p_random``, with a
    random start if ``random_start``.  ``p_random``: probability of a random
    garden of outer size in ``random_size``.

    ``symmetries``: rotate/mirror every lawn by one of the 8 symmetries of the
    square, drawn at random, so each map gives 8 training lawns.
    ``curriculum``: the random gardens start small and reach their full size
    after this fraction of the training (0 = full size from the start); the
    trainer reports its progress through :attr:`progress`.
    """

    def __init__(self, maps: Sequence[Lawn] = (), p_random: float = 1.0,
                 random_size: tuple[int, int] = (8, 32), random_start: bool = True,
                 symmetries: bool = True, curriculum: float = 0.0, seed: int = 0):
        if not maps and p_random < 1:
            p_random = 1.0
        self.maps = list(maps)
        self.p_random = p_random
        self.random_size = random_size
        self.random_start = random_start
        self.symmetries = symmetries
        self.curriculum = curriculum
        self.progress = 0.0
        self.rng = np.random.default_rng(seed)

    @property
    def max_shape(self) -> tuple[int, int]:
        """Observation size that fits every lawn this source can draw."""
        shapes = [m.shape for m in self.maps]
        if self.p_random > 0:
            shapes.append((self.random_size[1], self.random_size[1]))
        h, w = max(s[0] for s in shapes), max(s[1] for s in shapes)
        return (max(h, w), max(h, w)) if self.symmetries else (h, w)

    @property
    def current_size(self) -> tuple[int, int]:
        lo, hi = self.random_size
        if self.curriculum <= 0:
            return lo, hi
        frac = min(1.0, self.progress / self.curriculum)
        return lo, max(lo, round(lo + frac * (hi - lo)))

    def __call__(self) -> Lawn:
        if self.maps and self.rng.random() >= self.p_random:
            lawn = self.maps[int(self.rng.integers(len(self.maps)))]
            if self.random_start:
                cells = sorted(lawn.reachable)
                lawn = lawn.with_start(cells[int(self.rng.integers(len(cells)))])
        else:
            lawn = random_lawn(self.rng, *self.current_size)
        if self.symmetries:
            lawn = transform_lawn(lawn, int(self.rng.integers(4)), bool(self.rng.integers(2)))
        return lawn


# ---------------------------------------------------------------- vector env
class VecEnv:
    """``n`` LawnmowerEnv in lock-step, with automatic reset on a new lawn."""

    def __init__(self, n: int, source: LawnSource, rewards: Rewards,
                 max_steps_factor: float, obs_size: tuple[int, int],
                 turn_memory: bool = False):
        self.source = source
        first = source()
        self.envs = [LawnmowerEnv(first, rewards=rewards, max_steps_factor=max_steps_factor,
                                  obs_size=obs_size, turn_memory=turn_memory)
                     for _ in range(n)]
        self.episode_return = np.zeros(n)
        self.finished: list[dict] = []

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        out = [env.reset(options={"lawn": self.source()}) for env in self.envs]
        return (np.stack([o for o, _ in out]),
                np.stack([i["action_mask"] for _, i in out]))

    def step(self, actions: np.ndarray):
        obs, masks, rewards, dones, truncs, finals = [], [], [], [], [], []
        for k, (env, a) in enumerate(zip(self.envs, actions)):
            o, r, term, trunc, info = env.step(int(a))
            self.episode_return[k] += r
            final = None
            if term or trunc:
                m = env.result().metrics
                self.finished.append({
                    "return": self.episode_return[k], "coverage": m.coverage,
                    "complete": m.complete, "steps": m.steps,
                    "overhead": m.overhead, "cells": m.target})
                self.episode_return[k] = 0.0
                if trunc and not term:
                    final = o
                o, info = env.reset(options={"lawn": self.source()})
            obs.append(o)
            masks.append(info["action_mask"])
            rewards.append(r)
            dones.append(term or trunc)
            truncs.append(trunc and not term)
            finals.append(final)
        return (np.stack(obs), np.stack(masks), np.array(rewards, dtype=np.float32),
                np.array(dones), np.array(truncs), finals)


# -------------------------------------------------------------------- config
@dataclass
class PPOConfig:
    total_steps: int = 2_000_000
    n_envs: int = 32
    n_steps: int = 128
    epochs: int = 4
    minibatches: int = 4
    lr: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    reward_scale: float = 0.1
    max_steps_factor: float = 4.0
    rewards: Rewards = field(default_factory=Rewards)
    save_every: int = 250_000
    seed: int = 0


def uses_turn_memory(net: CoverageNet) -> bool:
    """True if the network was built for the 5-channel observation."""
    return net.config["in_channels"] == 5


def get_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(path: str | Path, net: CoverageNet, step: int, extra: dict | None = None):
    torch.save({"net": net.state_dict(), "net_config": net.config, "step": step,
                **(extra or {})}, path)


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> tuple[CoverageNet, dict]:
    ck = torch.load(path, map_location=device or "cpu", weights_only=False)
    net = CoverageNet(**ck["net_config"])
    net.load_state_dict(ck["net"])
    return net.to(device or "cpu"), ck


# ---------------------------------------------------------- behaviour cloning
def collect_demos(source: Callable[[], Lawn], n_lawns: int, teacher: str,
                  obs_size: tuple[int, int], turn_memory: bool = False):
    """Replay the planner ``teacher`` on ``n_lawns`` lawns: (obs, mask, action)."""
    obs, masks, acts = [], [], []
    for _ in range(n_lawns):
        lawn = source()
        path = plan(lawn, teacher).path
        env = LawnmowerEnv(lawn, obs_size=obs_size, turn_memory=turn_memory)
        o, info = env.reset()
        for a in path_to_actions(path):
            obs.append(o)
            masks.append(info["action_mask"])
            acts.append(a)
            o, _, term, _, info = env.step(a)
        assert term
    return np.stack(obs), np.stack(masks), np.array(acts)


def behaviour_cloning(net: CoverageNet, demos, epochs: int = 3, batch: int = 256,
                      lr: float = 1e-3, device: torch.device | None = None,
                      log: Callable[[str], None] = print) -> float:
    device = device or next(net.parameters()).device
    obs, masks, acts = demos
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(acts)
    acc = 0.0
    for ep in range(epochs):
        perm = np.random.permutation(n)
        correct = total_loss = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            o = torch.as_tensor(obs[idx], device=device)
            m = torch.as_tensor(masks[idx], device=device)
            a = torch.as_tensor(acts[idx], device=device)
            logits, _ = net(o, m)
            loss = F.cross_entropy(logits, a)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == a).sum().item()
        acc = correct / n
        log(f"BC epoch {ep + 1}/{epochs}: loss {total_loss / n:.3f}, accuracy {100 * acc:.1f}%")
    return acc


# ----------------------------------------------------------------------- PPO
def train_ppo(net: CoverageNet, source: LawnSource, cfg: PPOConfig, out_dir: str | Path,
              device: torch.device | None = None, start_step: int = 0,
              log: Callable[[str], None] = print) -> CoverageNet:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or get_device()
    net.to(device)
    torch.manual_seed(cfg.seed)
    obs_size = source.max_shape
    channels = net.config["in_channels"]
    vec = VecEnv(cfg.n_envs, source, cfg.rewards, cfg.max_steps_factor, obs_size,
                 turn_memory=uses_turn_memory(net))
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)

    n, T = cfg.n_envs, cfg.n_steps
    obs_buf = torch.zeros((T, n, channels, *obs_size), device=device)
    mask_buf = torch.zeros((T, n, 8), dtype=torch.bool, device=device)
    act_buf = torch.zeros((T, n), dtype=torch.long, device=device)
    logp_buf = torch.zeros((T, n), device=device)
    val_buf = torch.zeros((T, n), device=device)
    rew_buf = torch.zeros((T, n), device=device)
    done_buf = torch.zeros((T, n), device=device)

    obs, masks = vec.reset()
    step = start_step
    next_save = (step // cfg.save_every + 1) * cfg.save_every
    log_path = out_dir / "training_log.csv"
    new_log = not log_path.exists() or start_step == 0
    log_file = open(log_path, "w" if new_log else "a", newline="")
    writer = csv.writer(log_file)
    if new_log:
        writer.writerow(["step", "episodes", "return", "coverage_%", "complete_%",
                         "overhead", "policy_loss", "value_loss", "entropy", "fps"])
    t0 = time.time()
    n_updates = math.ceil((cfg.total_steps - start_step) / (n * T))
    for update in range(n_updates):
        frac = 1 - update / max(n_updates, 1)
        source.progress = (step - start_step) / max(cfg.total_steps - start_step, 1)
        for g in opt.param_groups:
            g["lr"] = cfg.lr * frac
        net.eval()
        for t in range(T):
            o = torch.as_tensor(obs, device=device)
            m = torch.as_tensor(masks, device=device)
            with torch.no_grad():
                logits, v = net(o, m)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            obs_buf[t], mask_buf[t], act_buf[t] = o, m, a
            logp_buf[t], val_buf[t] = dist.log_prob(a), v
            obs, masks, r, d, tr, finals = vec.step(a.cpu().numpy())
            r = r * cfg.reward_scale
            # bootstrap episodes cut by the step limit
            for k in np.flatnonzero(tr):
                with torch.no_grad():
                    fo = torch.as_tensor(finals[k][None], device=device)
                    fm = torch.ones((1, 8), dtype=torch.bool, device=device)
                    _, fv = net(fo, fm)
                r[k] += cfg.gamma * fv.item()
            rew_buf[t] = torch.as_tensor(r, device=device)
            done_buf[t] = torch.as_tensor(d, dtype=torch.float32, device=device)
        step += n * T

        with torch.no_grad():
            _, next_v = net(torch.as_tensor(obs, device=device),
                            torch.as_tensor(masks, device=device))
        adv = torch.zeros_like(rew_buf)
        last = torch.zeros(n, device=device)
        for t in reversed(range(T)):
            nv = next_v if t == T - 1 else val_buf[t + 1]
            nonterm = 1.0 - done_buf[t]
            delta = rew_buf[t] + cfg.gamma * nv * nonterm - val_buf[t]
            last = delta + cfg.gamma * cfg.gae_lambda * nonterm * last
            adv[t] = last
        ret = adv + val_buf

        b_obs = obs_buf.reshape(n * T, channels, *obs_size)
        b_mask, b_act = mask_buf.reshape(n * T, 8), act_buf.reshape(-1)
        b_logp, b_adv, b_ret = logp_buf.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        net.train()
        mb = n * T // cfg.minibatches
        stats = []
        for _ in range(cfg.epochs):
            perm = torch.randperm(n * T, device=device)
            for i in range(0, n * T, mb):
                idx = perm[i:i + mb]
                logits, v = net(b_obs[idx], b_mask[idx])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(b_act[idx])
                ratio = (logp - b_logp[idx]).exp()
                a_ = b_adv[idx]
                a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                pl = -torch.min(ratio * a_, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a_).mean()
                vl = F.mse_loss(v, b_ret[idx])
                ent = dist.entropy().mean()
                loss = pl + cfg.vf_coef * vl - cfg.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()
                stats.append((pl.item(), vl.item(), ent.item()))

        eps, vec.finished = vec.finished, []
        pl_, vl_, ent_ = np.mean(stats, axis=0)
        fps = (step - start_step) / (time.time() - t0)
        if eps:
            done_eps = [e for e in eps if e["complete"]]
            row = [step, len(eps), np.mean([e["return"] for e in eps]),
                   100 * np.mean([e["coverage"] for e in eps]),
                   100 * len(done_eps) / len(eps),
                   np.mean([e["overhead"] for e in done_eps]) if done_eps else float("nan"),
                   pl_, vl_, ent_, fps]
            writer.writerow([f"{x:.4g}" if isinstance(x, float) else x for x in row])
            log_file.flush()
            log(f"step {step:>9,}  episodes {len(eps):3d}  coverage {row[3]:5.1f}%  "
                f"complete {row[4]:5.1f}%  overhead {row[5]:.3f}  entropy {ent_:.2f}  "
                f"{fps:.0f} steps/s")
        if step >= next_save:
            save_checkpoint(out_dir / f"agent_{step}.pt", net, step, {"ppo": asdict(cfg)})
            next_save += cfg.save_every
    save_checkpoint(out_dir / "agent_final.pt", net, step, {"ppo": asdict(cfg)})
    log_file.close()
    return net


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def run_policy(net: CoverageNet, lawn: Lawn, greedy: bool = True, samples: int = 1,
               max_steps_factor: float = 4.0, seed: int = 0,
               device: torch.device | None = None) -> Result:
    """Let the agent mow ``lawn``; return the best of ``samples`` episodes.

    ``greedy=True`` plays the most likely move (one episode).  Otherwise
    moves are sampled from the policy and the shortest complete episode wins
    (a cheap search: the planners also pick the best of several candidates).
    """
    device = device or next(net.parameters()).device
    net.eval()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best: Result | None = None
    for _ in range(1 if greedy else samples):
        env = LawnmowerEnv(lawn, max_steps_factor=max_steps_factor,
                           turn_memory=uses_turn_memory(net))
        o, info = env.reset()
        done = False
        while not done:
            logits, _ = net(torch.as_tensor(o[None], device=device),
                            torch.as_tensor(info["action_mask"][None], device=device))
            if greedy:
                a = int(logits.argmax())
            else:
                a = int(torch.multinomial(F.softmax(logits.cpu(), 1), 1, generator=gen))
            o, _, term, trunc, info = env.step(a)
            done = term or trunc
        m, visits = evaluate(lawn, env.path)
        res = Result("agent", lawn, list(env.path), m, visits)
        key = (not m.complete, m.steps, m.turns)
        if best is None or key < (not best.metrics.complete, best.metrics.steps, best.metrics.turns):
            best = res
    assert best is not None
    return best
