from pathlib import Path

import numpy as np
import pytest

from lawnmower import PLANNERS, IllegalMove, Lawn, evaluate, plan
from lawnmower.planners.boustrophedon import decompose
from lawnmower.planners.stc import circumnavigate, mega_cells, spanning_forest

MAPS = sorted((Path(__file__).parent.parent / "maps").glob("*.txt"))
DETERMINISTIC = [p for p in PLANNERS if p != "random"]


def rect(h: int, w: int) -> Lawn:
    """An h x w lawn surrounded by a wall."""
    rows = ["+" * (w + 2)] + ["+" + "L" * w + "+" for _ in range(h)] + ["+" * (w + 2)]
    return Lawn.from_string("\n".join(rows))


# ------------------------------------------------------------------ the grid
def test_parse_start_and_padding():
    lawn = Lawn.from_string("+++\n+LS\n+L")
    assert lawn.start == (1, 2)
    assert lawn.shape == (3, 3)
    assert not lawn.is_free((2, 2))  # ragged line end = obstacle
    assert lawn.reachable == {(1, 1), (1, 2), (2, 1)}


def test_default_start_is_first_lawn_cell():
    assert Lawn.from_string("+++\n++L\n+LL").start == (1, 2)


def test_bad_maps():
    with pytest.raises(ValueError):
        Lawn.from_string("+++\n+++")
    with pytest.raises(ValueError):
        Lawn.from_string("+S+\n+S+")
    with pytest.raises(ValueError):
        Lawn.from_string("LL", start=(5, 5))


def test_no_corner_cutting():
    lawn = Lawn.from_string("L+\nLL")
    assert lawn.can_move((1, 0), (1, 1))
    assert not lawn.can_move((0, 0), (1, 1))  # (0,1) is an obstacle
    assert Lawn.from_string("LL\nLL").can_move((0, 0), (1, 1))


def test_diagonal_only_connection_is_unreachable():
    lawn = Lawn.from_string("L+\n+L")
    assert lawn.reachable == {(0, 0)}
    assert lawn.unreachable == {(1, 1)}


def test_shortest_path_prefers_fewer_turns():
    lawn = rect(5, 5)
    p = lawn.shortest_path((1, 1), (1, 5))
    assert len(p) == 5 and all(r == 1 for r, _ in p)


# ---------------------------------------------------------------- simulation
def test_evaluate_counts():
    lawn = rect(1, 4)  # cells (1,1)..(1,4)
    m, visits = evaluate(lawn, [(1, 1), (1, 2), (1, 3), (1, 2), (1, 3), (1, 4)])
    assert m.complete and m.steps == 5 and m.revisits == 2
    assert m.turns == 2 and m.turn_angle == 360
    assert visits[(1, 2)] == 2


def test_evaluate_rejects_illegal_paths():
    lawn = rect(3, 3)
    with pytest.raises(IllegalMove):
        evaluate(lawn, [(1, 1), (1, 3)])          # jump
    with pytest.raises(IllegalMove):
        evaluate(lawn, [(2, 2)])                  # wrong start
    with pytest.raises(IllegalMove):
        evaluate(lawn, [(1, 1), (0, 1)])          # into the wall


# ------------------------------------------------------------------ planners
@pytest.mark.parametrize("map_file", MAPS, ids=lambda p: p.stem)
@pytest.mark.parametrize("planner", DETERMINISTIC)
def test_planners_cover_every_map(map_file, planner):
    res = plan(Lawn.from_file(map_file), planner)  # plan() validates the path
    assert res.metrics.complete
    assert res.metrics.overhead < 1.5


@pytest.mark.parametrize("planner", DETERMINISTIC)
def test_planners_from_other_starts(planner):
    lawn = Lawn.from_file(MAPS[0])
    for start in sorted(lawn.reachable)[::37]:
        assert plan(lawn.with_start(start), planner).metrics.complete


@pytest.mark.parametrize("planner", ["boustrophedon", "stc"])
@pytest.mark.parametrize("h,w", [(4, 6), (6, 4), (8, 8)])
def test_open_rectangle_is_mowed_without_overlap(planner, h, w):
    m = plan(rect(h, w), planner).metrics
    assert m.complete and m.revisits == 0


def test_random_walk_is_seeded():
    lawn = Lawn.from_file(MAPS[0])
    assert plan(lawn, "random", seed=3).path == plan(lawn, "random", seed=3).path


def test_unknown_planner():
    with pytest.raises(ValueError):
        plan(rect(2, 2), "teleport")


def test_boustrophedon_cells_partition_the_lawn():
    lawn = Lawn.from_file(MAPS[0])
    cells = [(r, s.col) for cell in decompose(lawn) for s in cell
             for r in range(s.top, s.bottom + 1)]
    assert len(cells) == len(set(cells)) and set(cells) == lawn.reachable


def test_stc_loop_is_a_hamiltonian_cycle_of_the_mega_cells():
    lawn = Lawn.from_file(MAPS[0])
    nodes = mega_cells(lawn, 0, 0)
    loops = circumnavigate(nodes, spanning_forest(nodes, True), 0, 0)
    sub = [c for loop in loops for c in loop]
    assert len(sub) == 4 * len(nodes) == len(set(sub))
    for loop in loops:
        for a, b in zip(loop, loop[1:] + loop[:1]):
            assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


# ----------------------------------------------------------------------- CLI
def test_cli_run_and_compare(tmp_path, capsys):
    from lawnmower.__main__ import main

    main(["run", str(MAPS[0]), "-p", "greedy", "--ascii", "--path", str(tmp_path / "p.txt"),
          "--plot", str(tmp_path / "p.png")])
    main(["compare", str(MAPS[0]), "-p", "greedy", "stc"])
    out = capsys.readouterr().out
    assert "greedy" in out and "stc" in out
    assert (tmp_path / "p.png").stat().st_size > 0
    assert (tmp_path / "p.txt").read_text().count("\n") > 10


# ------------------------------------------------------------------------ RL
gym = pytest.importorskip("gymnasium")


def test_env_passes_gymnasium_checks():
    from gymnasium.utils.env_checker import check_env
    from lawnmower.rl import LawnmowerEnv

    lawns = [Lawn.from_file(m) for m in MAPS]
    check_env(LawnmowerEnv(lawns, random_start=True), skip_render_check=True)
    check_env(LawnmowerEnv(lawns, view_radius=3), skip_render_check=True)


def test_env_replay_matches_planner():
    from lawnmower.rl import LawnmowerEnv, Rewards, path_to_actions

    lawn = Lawn.from_file(MAPS[0])
    res = plan(lawn, "stc")
    rw = Rewards(new_cell=1, step=0, turn=0, invalid=0, complete=0)
    env = LawnmowerEnv(lawn, rewards=rw)
    env.reset(seed=0)
    total = 0.0
    for a in path_to_actions(res.path):
        _, r, term, trunc, _ = env.step(a)
        total += r
    assert term and not trunc
    assert total == len(lawn.reachable) - 1
    assert env.result().metrics == res.metrics


def test_env_invalid_move_and_mask():
    from lawnmower.rl import LawnmowerEnv

    env = LawnmowerEnv(rect(2, 2))
    _, info = env.reset(seed=0)
    mask = info["action_mask"]
    assert mask.sum() == 3  # E, S, SE from the top-left corner
    bad = int(np.flatnonzero(~mask)[0])
    _, r, *_ = env.step(bad)
    assert env.pos == (1, 1) and r < 0


# ------------------------------------------------------------------- learning
def test_random_lawns_are_connected():
    from lawnmower.lawngen import random_lawn

    rng = np.random.default_rng(0)
    for _ in range(20):
        lawn = random_lawn(rng, 8, 20)
        assert lawn.reachable == lawn.lawn_cells and len(lawn.reachable) >= 9


def test_agent_pipeline_smoke(tmp_path):
    pytest.importorskip("torch")
    from lawnmower.learn.__main__ import main as learn_main
    from lawnmower.learn.ppo import load_checkpoint, run_policy

    learn_main(["train", "--out", str(tmp_path), "--size", "6", "9", "--steps", "1024",
                "--envs", "4", "--n-steps", "128", "--bc-teacher", "greedy",
                "--bc-lawns", "5", "--bc-epochs", "1", "--channels", "8"])
    net, ck = load_checkpoint(tmp_path / "agent_final.pt")
    assert ck["step"] == 1024
    res = run_policy(net, rect(3, 4), greedy=False, samples=2)
    from lawnmower.__main__ import main as cli
    cli(["run", str(MAPS[0]), "-p", "agent", "--model", str(tmp_path / "agent_final.pt"),
         "--samples", "1"])
    assert res.path[0] == (1, 1)
    assert (tmp_path / "training_log.csv").exists()
