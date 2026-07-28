"""obs_delay_ticks: delayed env emits exactly k-tick-old observations."""
import numpy as np

from pvpbot.sim.env import DuelVecEnv, SimConfig
from pvpbot.spec import NUM_ACTION_HEADS


def _mk(delay):
    cfg = SimConfig(obs_delay_ticks=delay)
    return DuelVecEnv(num_envs=4, seed=7, config=cfg)


def test_delayed_obs_matches_fresh_history():
    fresh_env, del_env = _mk(0), _mk(3)
    rng = np.random.default_rng(0)
    fresh_env.reset()
    del_env.reset()
    fresh_hist, delayed = [], []
    for t in range(12):
        a = rng.integers(0, 2, size=(4, 2, NUM_ACTION_HEADS))
        a[:, :, 5] = rng.integers(0, 11, size=(4, 2))
        a[:, :, 6] = rng.integers(0, 11, size=(4, 2))
        of, _, d1, _ = fresh_env.step(a)
        od, _, d2, _ = del_env.step(a)
        assert not d1.any() and not d2.any()  # no resets in 12 ticks
        fresh_hist.append(of.copy())
        delayed.append(od.copy())
    for t in range(3, 12):
        np.testing.assert_allclose(delayed[t], fresh_hist[t - 3], atol=1e-6)


def test_delay_zero_is_identity():
    e1, e2 = _mk(0), _mk(0)
    e1.reset(); e2.reset()
    a = np.ones((4, 2, NUM_ACTION_HEADS), dtype=np.int64)
    o1 = e1.step(a)[0]
    o2 = e2.step(a)[0]
    np.testing.assert_allclose(o1, o2)
