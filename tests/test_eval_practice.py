"""Practice-server-style tiers (pvpbot/eval/practice.py)."""
import numpy as np

from pvpbot.eval.arena import run_match
from pvpbot.eval.practice import (
    PracticeEasy,
    PracticeHacker,
    PracticeHard,
    practice_tiers,
)
from pvpbot.eval.scripted import Idle
from pvpbot.sim.env import DuelVecEnv
from pvpbot.spec import ACTION_HEAD_SIZES


def test_actions_valid_and_deterministic():
    env = DuelVecEnv(6, seed=0)
    obs = env.reset()
    for tier in practice_tiers(seed=3):
        a = tier.act(obs[:, 0])
        assert a.shape == (6, len(ACTION_HEAD_SIZES))
        for h, size in enumerate(ACTION_HEAD_SIZES):
            assert a[:, h].min() >= 0 and a[:, h].max() < size, tier.name
    t1 = PracticeHard(seed=5)
    t2 = PracticeHard(seed=5)
    a1 = t1.act(obs[:, 0])
    a2 = t2.act(obs[:, 0])
    assert (a1 == a2).all()


def test_hacker_flattens_idle():
    res = run_match(PracticeHacker(seed=1), Idle(seed=2), num_duels=16, seed=9)
    assert res.wins_b == 0
    assert res.wins_a >= 14, res.to_dict()


def test_hard_beats_easy():
    res = run_match(PracticeHard(seed=1), PracticeEasy(seed=2),
                    num_duels=24, seed=11)
    decisive = res.wins_a + res.wins_b
    assert decisive > 0
    assert res.wins_a / max(decisive, 1) > 0.6, res.to_dict()
