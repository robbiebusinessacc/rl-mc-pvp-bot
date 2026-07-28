"""League: Elo math, outcome bookkeeping, sampling, gating."""
import numpy as np
import torch

from pvpbot.models import PolicyNet
from pvpbot.spec import ACTION_HEAD_SIZES, NUM_ACTION_HEADS, OBS_DIM
from pvpbot.train.league import SELF_PLAY, League, elo_expected, elo_update
from pvpbot.train.ppo import RunningNorm


def test_elo_expected_symmetry():
    assert elo_expected(1000, 1000) == 0.5
    assert abs(elo_expected(1200, 1000) + elo_expected(1000, 1200) - 1.0) < 1e-12
    assert elo_expected(1200, 1000) > 0.5


def test_elo_update_directions():
    # equal ratings, A wins: A up, B down, zero-sum
    ra, rb = elo_update(1000.0, 1000.0, 1.0, k=16.0)
    assert ra > 1000.0 > rb
    assert abs((ra - 1000.0) - (1000.0 - rb)) < 1e-9
    # favourite winning gains less than an equal-rated winner
    ra2, _ = elo_update(1200.0, 1000.0, 1.0, k=16.0)
    assert 0.0 < ra2 - 1200.0 < ra - 1000.0
    # a draw moves the higher-rated player down and the lower one up
    ra3, rb3 = elo_update(1200.0, 1000.0, 0.5, k=16.0)
    assert ra3 < 1200.0 and rb3 > 1000.0
    # underdog losing barely moves
    ra4, rb4 = elo_update(1000.0, 1400.0, 0.0, k=16.0)
    assert 0.0 < 1000.0 - ra4 < 2.0 and 0.0 < rb4 - 1400.0 < 2.0


def _league_with_ckpt(num_envs=4, seed=0, **kw):
    torch.manual_seed(seed)
    policy = PolicyNet()
    norm = RunningNorm(OBS_DIM)
    league = League(num_envs, seed=seed, **kw)
    assert league.gate(policy, norm, step=0)  # empty pool always accepts
    return league, policy, norm


def test_league_on_done_moves_elo_right_direction():
    league, _, _ = _league_with_ckpt()
    league.assign[:] = 0  # everyone plays the checkpoint
    entry = league.pool[0]
    e0, l0 = entry.elo, league.learner_elo

    win = np.zeros((4, 2), dtype=np.float32)
    win[0, 0] = 1.0  # learner wins env 0
    done = np.array([True, False, False, False])
    league.on_done(done, win)
    assert league.learner_elo > l0
    assert entry.elo < e0
    assert entry.games == 1 and league.games_vs_pool == 1

    # learner loss moves ratings back the other way
    league.assign[:] = 0
    l1, e1 = league.learner_elo, entry.elo
    win = np.zeros((4, 2), dtype=np.float32)
    win[1, 1] = 1.0  # opponent wins env 1
    league.on_done(np.array([False, True, False, False]), win)
    assert league.learner_elo < l1
    assert entry.elo > e1


def test_league_draw_between_equals_is_neutral():
    league, _, _ = _league_with_ckpt()
    league.assign[:] = 0
    entry = league.pool[0]
    entry.elo = league.learner_elo = 1000.0
    win = np.zeros((4, 2), dtype=np.float32)  # timeout: nobody wins
    league.on_done(np.array([True, False, False, False]), win)
    assert abs(league.learner_elo - 1000.0) < 1e-9
    assert abs(entry.elo - 1000.0) < 1e-9
    assert league.winrate_vs_pool() == 0.5


def test_league_self_play_games_do_not_move_elo():
    league, _, _ = _league_with_ckpt()
    league.assign[:] = SELF_PLAY
    l0 = league.learner_elo
    win = np.zeros((4, 2), dtype=np.float32)
    win[:, 0] = 1.0
    league.on_done(np.ones(4, dtype=bool), win)
    assert league.learner_elo == l0
    assert league.games_vs_pool == 0


def test_league_gate_rules():
    league, policy, norm = _league_with_ckpt(min_gate_games=8, gate_winrate=0.55)
    assert league.num_ckpts() == 1
    # not enough recent games -> accept (pool must not starve)
    assert league.gate(policy, norm, step=1)
    assert league.num_ckpts() == 2
    # losing streak -> gate refuses
    league.recent.clear()
    league.recent.extend([0.0] * 16)
    assert not league.gate(policy, norm, step=2)
    assert league.num_ckpts() == 2
    # winning record -> gate accepts
    league.recent.clear()
    league.recent.extend([1.0] * 16)
    assert league.gate(policy, norm, step=3)
    assert league.num_ckpts() == 3


def test_league_pool_capped_and_lowest_elo_evicted():
    league, policy, norm = _league_with_ckpt(max_pool=2, min_gate_games=10**9)
    # the pool may open with discovered scripted baselines; ckpt entries only
    ck = [i for i, e in enumerate(league.pool) if e.kind == "ckpt"]
    league.pool[ck[0]].elo = 500.0  # weakest ckpt, should be evicted first
    assert league.gate(policy, norm, step=1)
    ck = [i for i, e in enumerate(league.pool) if e.kind == "ckpt"]
    league.pool[ck[1]].elo = 900.0
    assert league.gate(policy, norm, step=2)  # third ckpt -> evict elo 500
    assert league.num_ckpts() == 2
    assert all(e.elo != 500.0 for e in league.pool if e.kind == "ckpt")


def test_league_sampling_prefers_close_elo():
    league, policy, norm = _league_with_ckpt(
        num_envs=2000, p_self=0.0, min_gate_games=10**9
    )
    assert league.gate(policy, norm, step=1)
    league.pool[0].elo = 1000.0
    league.pool[1].elo = 1600.0
    league.learner_elo = 1000.0
    league.resample(np.arange(2000))
    counts = np.bincount(league.assign, minlength=2)
    assert counts[0] > 5 * counts[1]
    assert (league.assign != SELF_PLAY).all()  # p_self = 0


def test_league_opponent_actions_shapes_and_bounds():
    torch.manual_seed(0)
    league, policy, norm = _league_with_ckpt(num_envs=8)
    league.assign[:4] = SELF_PLAY  # mix mirror and checkpoint opponents
    league.assign[4:] = 0
    raw = np.random.default_rng(0).normal(size=(8, OBS_DIM)).astype(np.float32)
    acts = league.opponent_actions(raw, policy, norm)
    assert acts.shape == (8, NUM_ACTION_HEADS) and acts.dtype == np.int64
    for i, n in enumerate(ACTION_HEAD_SIZES):
        assert acts[:, i].min() >= 0 and acts[:, i].max() < n


def test_league_state_dict_roundtrip():
    league, policy, norm = _league_with_ckpt(min_gate_games=10**9)
    league.gate(policy, norm, step=5)
    league.learner_elo = 1234.5
    league.recent.extend([1.0, 0.0, 0.5])
    sd = league.state_dict()

    league2 = League(4, seed=1)
    league2.load_state_dict(sd)
    assert league2.learner_elo == 1234.5
    assert league2.num_ckpts() == 2
    assert list(league2.recent) == [1.0, 0.0, 0.5]
    names = [e.name for e in league2.pool]
    assert names == [e.name for e in league.pool]
