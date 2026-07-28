"""Bait-and-punish reward shaping (RewardConfig.punish_close_bonus).

The official-bar hacker wins via distance control; its exploitable state
is the committed approach. The bonus credits hits landed while the
OPPONENT was recently closing fast through the outer engage band.
"""
import numpy as np

from pvpbot.sim.env import DuelVecEnv, RewardConfig, SimConfig
from pvpbot.spec import CAMERA_BINS, NUM_ACTION_HEADS

ZERO_BIN = CAMERA_BINS.index(0.0)
BIG = SimConfig(arena_radius=1000.0, spawn_radius=5.0)


def _noop(n=1):
    a = np.zeros((n, 2, NUM_ACTION_HEADS), dtype=np.int64)
    a[:, :, 0] = 1
    a[:, :, 1] = 1
    a[:, :, 5] = ZERO_BIN
    a[:, :, 6] = ZERO_BIN
    return a


def _env(bonus):
    env = DuelVecEnv(1, seed=0, config=BIG,
                     reward=RewardConfig(punish_close_bonus=bonus,
                                         aim_coeff=0.0,
                                         sprint_toggle_coeff=0.0))
    env.reset()
    env.pos[0, 0] = (0.0, 0.0, 0.0)
    env.pos[0, 1] = (3.4, 0.0, 0.0)   # inside the commit band, out of reach
    env.vel[:] = 0.0
    env.yaw[0, 0] = 0.0               # facing +x = straight at the opponent
    env.yaw[0, 1] = 180.0
    env.pitch[:] = 0.0
    env.on_ground[:] = True
    return env


def _swing_in_reach(env):
    """Force a clean deterministic hit and return side-0's reward."""
    env.pos[0, 0] = (0.0, 0.0, 0.0)
    env.pos[0, 1] = (2.0, 0.0, 0.0)
    env.vel[:] = 0.0
    a = _noop()
    a[0, 0, 4] = 1
    _, rew, _, _ = env.step(a)
    assert float(env.hp[0, 1]) < 20.0  # the hit really landed
    return float(rew[0, 0])


def test_bonus_credits_hit_inside_commit_window():
    # opponent sprints inward through the band, then we land the punish
    base_env, boost_env = _env(0.0), _env(5.0)
    for env in (base_env, boost_env):
        a = _noop()
        a[0, 1, 0] = 2                # opponent holds forward (toward us)
        a[0, 1, 3] = 1                # sprinting: a committed close
        for _ in range(3):            # sustained sprint exceeds _COMMIT_V
            env.step(a)
        if env.rew_cfg.punish_close_bonus:
            assert int(env._commit[0, 0]) > 0   # window armed for side 0
    base = _swing_in_reach(base_env)
    boosted = _swing_in_reach(boost_env)
    assert abs(boosted - (base + 5.0)) < 1e-4


def test_no_bonus_without_commitment():
    env = _env(5.0)
    env.step(_noop())                 # opponent never moves: no commitment
    rew = _swing_in_reach(env)
    assert abs(rew - 1.0) < 1e-4      # plain hit reward only


def test_window_expires():
    env = _env(5.0)
    env.vel[0, 1, 0] = -0.25
    env.step(_noop())                 # commitment armed (timer = 6)
    for _ in range(7):                # idle past the window
        env.pos[0, 1] = (3.4, 0.0, 0.0)
        env.vel[:] = 0.0
        env.step(_noop())
    rew = _swing_in_reach(env)
    assert abs(rew - 1.0) < 1e-4      # bonus gone after expiry


def _swing_env(bonus):
    env = DuelVecEnv(1, seed=0, config=BIG,
                     reward=RewardConfig(punish_swing_bonus=bonus,
                                         aim_coeff=0.0,
                                         sprint_toggle_coeff=0.0))
    env.reset()
    env.pos[0, 0] = (0.0, 0.0, 0.0)
    env.pos[0, 1] = (5.0, 0.0, 0.0)   # out of reach: opponent swing whiffs
    env.vel[:] = 0.0
    env.yaw[0, 0] = 0.0
    env.yaw[0, 1] = 180.0
    env.pitch[:] = 0.0
    env.on_ground[:] = True
    return env


def test_swing_punish_credits_hit_after_opponent_whiff():
    base_env, boost_env = _swing_env(0.0), _swing_env(3.0)
    for env in (base_env, boost_env):
        a = _noop()
        a[0, 1, 4] = 1                # opponent swings (whiffs at 5 blocks)
        env.step(a)
    base = _swing_in_reach(base_env)      # our punish 1 tick later
    boosted = _swing_in_reach(boost_env)
    assert abs(boosted - (base + 3.0)) < 1e-4


def test_swing_punish_expires():
    env = _swing_env(3.0)
    a = _noop()
    a[0, 1, 4] = 1                    # opponent whiffs once
    env.step(a)
    for _ in range(9):                # idle past the 8-tick window
        env.pos[0, 1] = (5.0, 0.0, 0.0)
        env.vel[:] = 0.0
        env.step(_noop())
    rew = _swing_in_reach(env)
    assert abs(rew - 1.0) < 1e-4      # plain hit only
