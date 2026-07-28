"""Tier ordering in stub-env matches.

Hard requirement: T2 beats T0.  The stronger orderings hold clearly with the
current tiers on the stub, but stub physics is crude (no knockback, no aim
gating on hits), so they are asserted with tolerance (decisive-game win
share) rather than exact scores.  All matches are seeded -> deterministic.
"""
import numpy as np
import pytest

from pvpbot.eval.arena import run_match
from pvpbot.sim.stub import DuelVecEnv as StubEnv
from pvpbot.eval.scripted import (
    AimbotStationary,
    Chaser,
    Idle,
    Pro,
    Strafer,
)

DUELS = 24


def _duel(hi_cls, lo_cls, seed):
    # tiers here are calibrated against stub physics (no knockback, no aim
    # gating); real-sim orderings are reported by the ladder, not asserted
    return run_match(hi_cls(seed=1), lo_cls(seed=2), num_duels=DUELS,
                     seed=seed, env_cls=StubEnv)


def test_t2_beats_t0_hard():
    res = _duel(Chaser, Idle, seed=11)
    assert res.wins_b == 0
    assert res.wins_a >= int(0.9 * DUELS), res.to_dict()


def test_t3_and_t4_flatten_t0():
    for cls, seed in ((Strafer, 12), (Pro, 13)):
        res = _duel(cls, Idle, seed=seed)
        assert res.wins_b == 0
        assert res.wins_a >= int(0.9 * DUELS), res.to_dict()


def test_t1_never_loses_to_t0_and_wins_reachable_spawns():
    res = _duel(AimbotStationary, Idle, seed=14)
    assert res.wins_b == 0
    assert res.wins_a >= 1  # only wins spawns that start within reach
    assert res.wins_a + res.draws == DUELS


@pytest.mark.parametrize("hi,lo,seed", [
    (Chaser, AimbotStationary, 21),
    (Strafer, Chaser, 22),
    (Pro, Strafer, 23),
    (Pro, Chaser, 24),   # the ladder's headline check
    (Strafer, AimbotStationary, 25),
    (Pro, AimbotStationary, 26),
])
def test_higher_tier_beats_lower_with_tolerance(hi, lo, seed):
    res = _duel(hi, lo, seed=seed)
    decisive = res.wins_a + res.wins_b
    assert decisive > 0, res.to_dict()
    share = res.wins_a / decisive
    assert share >= 0.7, (hi.__name__, lo.__name__, res.to_dict())


def test_matches_are_deterministic_given_seed():
    r1 = _duel(Pro, Chaser, seed=31)
    r2 = _duel(Pro, Chaser, seed=31)
    assert (r1.wins_a, r1.wins_b, r1.draws) == (r2.wins_a, r2.wins_b, r2.draws)
    assert r1.mean_episode_len == r2.mean_episode_len


def test_higher_tiers_land_more_hits_than_they_take():
    res = _duel(Pro, Chaser, seed=41)
    assert res.stats_a.hits_landed > res.stats_a.hits_taken
    assert res.stats_b.hits_landed < res.stats_b.hits_taken
