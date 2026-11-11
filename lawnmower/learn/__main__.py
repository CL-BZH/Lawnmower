"""Train and evaluate a PPO agent.

    # behaviour cloning from a planner, then PPO, on random gardens + example maps
    python -m lawnmower.learn train --out runs/small --size 8 14 \\
        --maps maps/maze3.txt maps/maze4.txt --bc-teacher greedy --steps 1000000

    # compare the agent with the planners on example maps
    python -m lawnmower.learn eval runs/small/agent_final.pt maps/maze3.txt maps/maze7.txt
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .. import Lawn, plan
from ..rl import Rewards
from .mcts import mcts_mow, search_settings
from .net import CoverageNet
from .ppo import (LawnSource, PPOConfig, behaviour_cloning, collect_demos, get_device,
                  load_checkpoint, run_policy, save_checkpoint, train_ppo)

PLANNERS = ("greedy", "boustrophedon", "stc")


def cmd_train(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(a.device)
    if device.type == "cpu" and a.threads:
        torch.set_num_threads(a.threads)
    maps = [Lawn.from_file(m) for m in a.maps]
    source = LawnSource(maps, p_random=a.p_random if maps else 1.0,
                        random_size=tuple(a.size), symmetries=not a.no_symmetries,
                        curriculum=a.curriculum, seed=a.seed)
    cfg = PPOConfig(total_steps=a.steps, n_envs=a.envs, n_steps=a.n_steps, lr=a.lr,
                    gamma=a.gamma, ent_coef=a.ent_coef, seed=a.seed,
                    save_every=a.save_every, max_steps_factor=a.max_steps_factor,
                    rewards=Rewards(turn=-a.turn_penalty, turn_repeat=-a.turn_repeat,
                                    turn_decay=a.turn_decay))
    start = 0
    if a.resume:
        net, ck = load_checkpoint(a.resume, device)
        start = ck.get("step", 0) if a.continue_steps else 0
        print(f"resumed from {a.resume} (step {ck.get('step', 0)})")
    else:
        torch.manual_seed(a.seed)
        # the consecutive-turn penalty needs the 5th channel (steps since the last turn)
        memory = cfg.rewards.memory_horizon() if a.turn_repeat > 0 else 0
        net = CoverageNet(in_channels=5 if memory else 4, channels=a.channels,
                          turn_memory=memory).to(device)
    (out / "config.json").write_text(json.dumps(
        {"args": vars(a), "ppo": asdict(cfg), "net": net.config}, indent=2, default=str))

    if a.bc_teacher and not a.resume:
        print(f"collecting {a.bc_lawns} demonstrations of '{a.bc_teacher}'...")
        demo_source = LawnSource(maps, p_random=source.p_random, random_size=tuple(a.size),
                                 symmetries=source.symmetries, seed=a.seed + 1)
        demos = collect_demos(demo_source, a.bc_lawns, a.bc_teacher, source.max_shape,
                              turn_memory=net.config["turn_memory"])
        behaviour_cloning(net, demos, epochs=a.bc_epochs, device=device)
        save_checkpoint(out / "agent_bc.pt", net, 0, {"bc_teacher": a.bc_teacher})
    if a.steps > start:
        train_ppo(net, source, cfg, out, device=device, start_step=start)
    else:
        save_checkpoint(out / "agent_final.pt", net, start)


def compare(net: CoverageNet, lawns: list[Lawn], samples: int = 0, seed: int = 0,
            mcts: int = 0, settings: dict | None = None) -> list[dict]:
    rows = []
    for lawn in lawns:
        row = {"map": lawn.name, "cells": len(lawn.reachable)}
        for p in PLANNERS:
            row[p] = plan(lawn, p).metrics
        row["agent"] = run_policy(net, lawn, greedy=True).metrics
        if samples:
            row[f"agent_best_of_{samples}"] = run_policy(
                net, lawn, greedy=False, samples=samples, seed=seed).metrics
        if mcts:
            row[f"mcts_{mcts}"] = mcts_mow(net, lawn, simulations=mcts, **(settings or {})).metrics
        rows.append(row)
    return rows


def format_table(rows: list[dict]) -> str:
    keys = [k for k in rows[0] if k not in ("map", "cells")]
    head = ["map", "N", *keys]
    lines = []
    for row in rows:
        cells = [row["map"], str(row["cells"])]
        for k in keys:
            m = row[k]
            cell = f"{m.steps}/{m.turns}"
            cells.append(cell if m.complete else f"{cell} ({100 * m.coverage:.0f}%)")
        lines.append(cells)
    widths = [max(len(x) for x in col) for col in zip(head, *lines)]
    fmt = lambda cs: "  ".join(c.rjust(w) for c, w in zip(cs, widths))  # noqa: E731
    return "\n".join([fmt(head), fmt(["-" * w for w in widths]), *map(fmt, lines)])


def cmd_eval(a) -> None:
    net, ck = load_checkpoint(a.checkpoint, get_device(a.device))
    lawns = [Lawn.from_file(m) for m in a.maps]
    if a.random:
        src = LawnSource(random_size=tuple(a.size), seed=10_000 + a.seed)
        lawns += [src() for _ in range(a.random)]
    rows = compare(net, lawns, samples=a.samples, seed=a.seed, mcts=a.mcts,
                   settings=search_settings(ck))
    print(f"checkpoint {a.checkpoint} (step {ck.get('step')}) — steps/turns to mow everything "
          f"(coverage in brackets when incomplete)")
    print(format_table(rows))
    if a.random:
        keys = [k for k in rows[0] if k not in ("map", "cells")]
        print("\nover all lawns: mean overhead and turns per 100 cells on complete runs, "
              "completion rate")
        for k in keys:
            ms = [r[k] for r in rows]
            done = [m for m in ms if m.complete]
            print(f"  {k:18s} {np.mean([m.overhead for m in done]):.3f}  "
                  f"{np.mean([100 * m.turns / m.target for m in done]):5.1f}  "
                  f"{100 * len(done) / len(ms):.0f}%")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m lawnmower.learn")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    t.add_argument("--out", required=True, help="output directory (checkpoints, log)")
    t.add_argument("--maps", nargs="*", default=[], help="example maps to train on")
    t.add_argument("--p-random", type=float, default=0.5,
                   help="share of episodes on random gardens when --maps is given")
    t.add_argument("--size", type=int, nargs=2, default=(8, 32),
                   help="outer size range of random gardens")
    t.add_argument("--no-symmetries", action="store_true",
                   help="do not rotate/mirror the lawns (8 symmetries of the square)")
    t.add_argument("--curriculum", type=float, default=0.0,
                   help="random gardens grow from the smallest to the largest size over "
                        "this fraction of the training (0 = off)")
    t.add_argument("--steps", type=int, default=2_000_000, help="PPO environment steps")
    t.add_argument("--envs", type=int, default=32)
    t.add_argument("--n-steps", type=int, default=128, help="steps per env between updates")
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--gamma", type=float, default=0.995)
    t.add_argument("--ent-coef", type=float, default=0.01)
    t.add_argument("--turn-penalty", type=float, default=0.05,
                   help="penalty per 45 degrees turned (a step costs 0.1, so 0.05 makes "
                        "a 90-degree turn cost as much as one step)")
    t.add_argument("--turn-repeat", type=float, default=0.0,
                   help="penalty per 45 degrees for a turn right after another; the surcharge "
                        "over --turn-penalty fades by --turn-decay per step (0 = off; >0 adds "
                        "a 5th observation channel)")
    t.add_argument("--turn-decay", type=float, default=0.01)
    t.add_argument("--max-steps-factor", type=float, default=4.0)
    t.add_argument("--channels", type=int, default=32)
    t.add_argument("--bc-teacher", choices=PLANNERS, help="behaviour cloning before PPO")
    t.add_argument("--bc-lawns", type=int, default=500)
    t.add_argument("--bc-epochs", type=int, default=5)
    t.add_argument("--resume", help="checkpoint to start from")
    t.add_argument("--continue-steps", action="store_true",
                   help="with --resume: count --steps from the checkpoint's step")
    t.add_argument("--save-every", type=int, default=250_000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device")
    t.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval")
    e.add_argument("checkpoint")
    e.add_argument("maps", nargs="*")
    e.add_argument("--random", type=int, default=0, help="also evaluate on N random gardens")
    e.add_argument("--size", type=int, nargs=2, default=(8, 32))
    e.add_argument("--samples", type=int, default=0,
                   help="also report the best of N sampled episodes")
    e.add_argument("--mcts", type=int, default=0,
                   help="also report MCTS with N simulations per move")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--device")
    e.set_defaults(func=cmd_eval)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
