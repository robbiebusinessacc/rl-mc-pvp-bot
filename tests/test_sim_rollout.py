"""Long random rollout: numerical stability and episode termination."""
import numpy as np

from pvpbot.sim.env import DuelVecEnv
from pvpbot.spec import ACTION_HEAD_SIZES, OBS_LAYOUT


def test_random_rollout_2000_ticks():
    num_envs = 64
    env = DuelVecEnv(num_envs, seed=7)
    obs = env.reset()
    rng = np.random.default_rng(3)
    total_dones = 0
    for t in range(2000):
        actions = np.stack(
            [rng.integers(0, n, (num_envs, 2)) for n in ACTION_HEAD_SIZES],
            axis=-1,
        )
        obs, rew, done, info = env.step(actions)
        assert np.isfinite(obs).all(), "NaN/inf in obs at tick %d" % t
        assert np.isfinite(rew).all(), "NaN/inf in rew at tick %d" % t
        assert obs.dtype == np.float32 and rew.dtype == np.float32
        total_dones += int(done.sum())
        w = info["win"]
        assert np.isin(w, (0.0, 1.0)).all()
        assert (w[~done] == 0.0).all()      # win only reported on done ticks
        assert not (w.sum(axis=1) > 1.0).any()  # at most one winner
    # MAX_TICKS=1200 guarantees every env terminated at least once in 2000
    assert total_dones >= num_envs, total_dones
    # post-auto-reset invariants
    assert (env.hp > 0.0).all() and (env.hp <= 20.0).all()
    assert (env.ticks < env.MAX_TICKS).all()
    assert (env.pos[:, :, 1] >= 0.0).all()  # never below the floor
    hxz = np.sqrt(env.pos[:, :, 0] ** 2 + env.pos[:, :, 2] ** 2)
    assert (hxz <= env.cfg.arena_radius + 1e-4).all()  # inside the wall
    assert np.abs(obs).max() < 60.0
    # dtype/shape of internal state stayed intact across resets
    assert env.pos.dtype == np.float32 and env.vel.dtype == np.float32
    assert env.hp.dtype == np.float32


def test_fights_actually_happen():
    # with random actions some swings must land within a few episodes
    num_envs = 32
    env = DuelVecEnv(num_envs, seed=11)
    env.reset()
    rng = np.random.default_rng(4)
    s = OBS_LAYOUT["ticks_since_hit_dealt"][0]
    saw_hit = False
    wins = 0.0
    for _ in range(1500):
        actions = np.stack(
            [rng.integers(0, n, (num_envs, 2)) for n in ACTION_HEAD_SIZES],
            axis=-1,
        )
        obs, rew, done, info = env.step(actions)
        saw_hit = saw_hit or (obs[:, :, s] == 0.0).any() or (rew >= 0.9).any()
        wins += float(info["win"].sum())
    assert saw_hit, "no hit ever landed in 1500 random ticks"
