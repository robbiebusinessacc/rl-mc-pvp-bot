"""API-contract conformance for the real sim (pvpbot.sim.env.DuelVecEnv).

Mirrors the stub smoke test: the real env must be a drop-in replacement
per the docstring at the bottom of pvpbot/spec.py.
"""
import numpy as np

from pvpbot.sim.env import DuelVecEnv
from pvpbot.spec import ACTION_HEAD_SIZES, NUM_ACTION_HEADS, OBS_DIM, OBS_LAYOUT


def _random_actions(rng, num_envs):
    return np.stack(
        [rng.integers(0, n, (num_envs, 2)) for n in ACTION_HEAD_SIZES], axis=-1
    )


def test_env_contract_shapes():
    env = DuelVecEnv(num_envs=8, seed=1)
    obs = env.reset()
    assert obs.shape == (8, 2, OBS_DIM) and obs.dtype == np.float32
    rng = np.random.default_rng(0)
    for _ in range(60):
        obs, rew, done, info = env.step(_random_actions(rng, 8))
        assert obs.shape == (8, 2, OBS_DIM) and obs.dtype == np.float32
        assert rew.shape == (8, 2) and rew.dtype == np.float32
        assert done.shape == (8,) and done.dtype == bool
        assert info["win"].shape == (8, 2)
        assert info["win"].dtype == np.float32


def test_drop_in_signature():
    # Constructable exactly like the stub: DuelVecEnv(num_envs, seed=0)
    env = DuelVecEnv(4)
    obs = env.reset()
    assert obs.shape == (4, 2, OBS_DIM)
    assert env.REACH == 3.0
    assert env.MAX_TICKS == 1200


def test_returned_obs_not_aliased():
    # step() must not hand back a view of internal state that a later
    # step() silently overwrites
    env = DuelVecEnv(4, seed=3)
    rng = np.random.default_rng(1)
    obs1, _, _, _ = env.step(_random_actions(rng, 4))
    snap = obs1.copy()
    env.step(_random_actions(rng, 4))
    assert np.array_equal(obs1, snap)


def test_auto_reset_gives_fresh_obs():
    env = DuelVecEnv(6, seed=2)
    env.reset()
    # force a timeout on every env: next step must auto-reset all of them
    env.ticks[:] = env.MAX_TICKS - 1
    rng = np.random.default_rng(1)
    a = _random_actions(rng, 6)
    a[:, :, 4] = 0  # no attacks -> pure timeout, no winner
    obs, rew, done, info = env.step(a)
    assert done.all()
    assert (info["win"] == 0.0).all()  # draw on timeout
    assert (env.ticks == 0).all()
    assert (env.hp == 20.0).all()
    s = OBS_LAYOUT["self_hp"][0]
    assert np.allclose(obs[:, :, s], 1.0)
    s = OBS_LAYOUT["prev_action"][0]
    assert np.allclose(obs[:, :, s : s + NUM_ACTION_HEADS], 0.0)


def test_prev_action_obs_normalized_by_head_size():
    env = DuelVecEnv(3, seed=0)
    env.reset()
    rng = np.random.default_rng(5)
    a = _random_actions(rng, 3)
    a[:, :, 4] = 0  # avoid hits/dones interfering
    obs, _, done, _ = env.step(a)
    assert not done.any()
    s, e = OBS_LAYOUT["prev_action"]
    sizes = np.asarray(ACTION_HEAD_SIZES, dtype=np.float32)
    expect = a.astype(np.float32) / sizes
    assert np.allclose(obs[:, :, s:e], expect, atol=1e-6)


def test_reserved_slots_zero():
    env = DuelVecEnv(4, seed=9)
    obs = env.reset()
    rng = np.random.default_rng(2)
    s, e = OBS_LAYOUT["reserved"]
    assert (obs[:, :, s:e] == 0.0).all()
    for _ in range(30):
        obs, _, _, _ = env.step(_random_actions(rng, 4))
        assert (obs[:, :, s:e] == 0.0).all()


def test_determinism_given_seed():
    env_a = DuelVecEnv(16, seed=42)
    env_b = DuelVecEnv(16, seed=42)
    obs_a = env_a.reset()
    obs_b = env_b.reset()
    assert np.array_equal(obs_a, obs_b)
    rng = np.random.default_rng(123)
    for _ in range(400):
        a = _random_actions(rng, 16)
        oa, ra, da, ia = env_a.step(a)
        ob, rb, db, ib = env_b.step(a)
        assert np.array_equal(oa, ob)
        assert np.array_equal(ra, rb)
        assert np.array_equal(da, db)
        assert np.array_equal(ia["win"], ib["win"])
        assert np.array_equal(ia["dist"], ib["dist"])


def test_different_seeds_differ():
    env_a = DuelVecEnv(16, seed=1)
    env_b = DuelVecEnv(16, seed=2)
    assert not np.array_equal(env_a.reset(), env_b.reset())
