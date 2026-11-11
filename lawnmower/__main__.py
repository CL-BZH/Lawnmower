"""Command line interface.

    python -m lawnmower run maps/maze6.txt -p stc --plot stc.png
    python -m lawnmower run maps/maze6.txt -p greedy --gif greedy.gif
    python -m lawnmower compare maps/maze6.txt maps/maze8.txt --plot cmp.png
    python -m lawnmower run maps/maze7.txt -p agent --gif agent.gif   # needs torch
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import PLANNERS, Lawn, plan
from .sim import Result

COLUMNS = ["coverage_%", "steps", "length", "turns", "revisits", "overhead"]
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "agent_small.pt"
_models: dict = {}


def _solve(lawn: Lawn, planner: str, args) -> Result:
    """Run a planner, or the trained agent when ``planner == "agent"``."""
    if planner != "agent":
        opts = {"seed": args.seed} if planner == "random" else {}
        if getattr(args, "turn_weight", 0) and planner in ("boustrophedon", "stc"):
            opts["turn_weight"] = args.turn_weight
        return plan(lawn, planner, **opts)
    try:
        from .learn.ppo import load_checkpoint, run_policy
    except ImportError:
        sys.exit("the agent needs PyTorch: pip install -e '.[train]'")
    if args.model not in _models:
        _models[args.model] = load_checkpoint(args.model)[0]
    t0 = time.perf_counter()
    res = run_policy(_models[args.model], lawn, greedy=args.samples <= 1,
                     samples=args.samples, seed=args.seed)
    res.runtime_s = time.perf_counter() - t0
    return res


def _start(text: str | None):
    if text is None:
        return None
    try:
        r, c = (int(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("--start expects ROW,COL") from None
    return (r, c)


def _table(rows: list[tuple[str, str, Result]]) -> str:
    head = ["map", "planner", *COLUMNS, "time_s"]
    body = [[m, p, *(str(r.metrics.as_dict()[k]) for k in COLUMNS), f"{r.runtime_s:.2f}"]
            for m, p, r in rows]
    widths = [max(len(x) for x in col) for col in zip(head, *body)]
    fmt = lambda cells: "  ".join(c.rjust(w) if i > 1 else c.ljust(w)  # noqa: E731
                                  for i, (c, w) in enumerate(zip(cells, widths)))
    return "\n".join([fmt(head), fmt(["-" * w for w in widths]), *map(fmt, body)])


def _describe(lawn: Lawn) -> str:
    msg = f"{lawn.name}: {lawn.shape[0]}x{lawn.shape[1]} grid, " \
          f"{len(lawn.reachable)} lawn cells to mow, start {lawn.start}"
    if lawn.unreachable:
        msg += f" ({len(lawn.unreachable)} lawn cells unreachable from the start)"
    return msg


def cmd_run(args) -> None:
    lawn = Lawn.from_file(args.map, start=args.start)
    print(_describe(lawn))
    res = _solve(lawn, args.planner, args)
    print(_table([(lawn.name, args.planner, res)]))
    if args.ascii:
        marks = {c: "." for c in res.visits if res.visits[c] == 1}
        marks.update({c: "o" for c in res.visits if res.visits[c] > 1})
        print(lawn.render(marks))
    if args.path:
        Path(args.path).write_text("\n".join(f"{r},{c}" for r, c in res.path) + "\n")
        print(f"path written to {args.path}")
    if args.plot or args.show or args.gif:
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .viz import animate, plot_result
        if args.plot or args.show:
            ax = plot_result(res)
            if args.plot:
                ax.figure.savefig(args.plot, dpi=130, bbox_inches="tight")
                print(f"figure written to {args.plot}")
            if args.show:
                plt.show()
        if args.gif:
            animate(res, args.gif)
            print(f"animation written to {args.gif}")


def cmd_compare(args) -> None:
    planners = args.planners or list(PLANNERS)
    rows = []
    for map_file in args.maps:
        lawn = Lawn.from_file(map_file, start=args.start)
        print(_describe(lawn), file=sys.stderr)
        results = [_solve(lawn, p, args) for p in planners]
        rows += [(lawn.name, p, r) for p, r in zip(planners, results)]
        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            from .viz import plot_comparison
            out = Path(args.plot)
            if len(args.maps) > 1:
                out = out.with_name(f"{out.stem}_{lawn.name}{out.suffix}")
            plot_comparison(results).savefig(out, dpi=110, bbox_inches="tight")
            print(f"figure written to {out}", file=sys.stderr)
    print(_table(rows))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m lawnmower",
                                 description="Lawn-mowing coverage path planning.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="plan one map with one planner")
    run.add_argument("map")
    run.add_argument("-p", "--planner", default="stc", choices=[*sorted(PLANNERS), "agent"])
    run.add_argument("--start", type=_start, help="start cell ROW,COL (default: 'S' or first lawn cell)")
    run.add_argument("--seed", type=int, default=0, help="seed for the random planner and the agent")
    run.add_argument("--turn-weight", type=float, default=0.0,
                     help="cost of a turn in steps when choosing between candidate plans")
    run.add_argument("--plot", help="save a figure (png/pdf/svg)")
    run.add_argument("--gif", help="save an animation")
    run.add_argument("--show", action="store_true", help="open an interactive window")
    run.add_argument("--ascii", action="store_true", help="print the coverage as text")
    run.add_argument("--path", help="write the path as ROW,COL lines")
    run.set_defaults(func=cmd_run)

    cmp_ = sub.add_parser("compare", help="compare planners on one or more maps")
    cmp_.add_argument("maps", nargs="+")
    cmp_.add_argument("-p", "--planners", nargs="+", choices=[*sorted(PLANNERS), "agent"],
                      help="default: all planners (add 'agent' for the trained agent)")
    cmp_.add_argument("--start", type=_start)
    cmp_.add_argument("--seed", type=int, default=0)
    cmp_.add_argument("--plot", help="save a side-by-side figure (one per map)")
    cmp_.set_defaults(func=cmd_compare)

    for p in (run, cmp_):
        p.add_argument("--model", default=str(DEFAULT_MODEL),
                       help="-p agent: checkpoint of the trained agent")
        p.add_argument("--samples", type=int, default=16,
                       help="-p agent: sample N episodes and keep the shortest "
                            "(1 = always play the most likely move)")

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
