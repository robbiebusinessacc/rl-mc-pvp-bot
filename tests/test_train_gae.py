"""GAE(lambda) correctness on hand-computed toy sequences."""
import numpy as np

from pvpbot.train.ppo import compute_gae


def test_gae_hand_computed_with_mid_episode_done():
    # gamma = lam = 0.5, T = 3, one env, done at t=1.
    rewards = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    values = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
    dones = np.array([[0.0], [1.0], [0.0]], dtype=np.float32)
    last_values = np.array([2.0], dtype=np.float32)

    adv, ret = compute_gae(rewards, values, dones, last_values, gamma=0.5, lam=0.5)

    # By hand (backwards):
    # t=2: delta = 3 + 0.5*2 - 1 = 3.0            -> adv2 = 3.0
    # t=1: done, no bootstrap: delta = 2 - 1 = 1  -> adv1 = 1.0
    # t=0: delta = 1 + 0.5*1 - 1 = 0.5
    #      adv0 = 0.5 + 0.5*0.5*adv1 = 0.75
    np.testing.assert_allclose(adv[:, 0], [0.75, 1.0, 3.0], rtol=1e-6)
    np.testing.assert_allclose(ret[:, 0], [1.75, 2.0, 4.0], rtol=1e-6)


def test_gae_lambda_one_equals_discounted_returns():
    # With lam = 1 and no dones, adv + v must equal the full discounted
    # return including the bootstrap value.
    gamma = 0.9
    rewards = np.array([[1.0], [1.0]], dtype=np.float32)
    values = np.array([[0.3], [0.4]], dtype=np.float32)
    dones = np.zeros((2, 1), dtype=np.float32)
    last_values = np.array([0.5], dtype=np.float32)

    adv, ret = compute_gae(rewards, values, dones, last_values, gamma=gamma, lam=1.0)

    # returns: t=1: 1 + 0.9*0.5 = 1.45; t=0: 1 + 0.9*1.45 = 2.305
    np.testing.assert_allclose(ret[:, 0], [2.305, 1.45], rtol=1e-6)
    np.testing.assert_allclose(adv, ret - values, rtol=1e-6)


def test_gae_multiple_envs_independent():
    # Env 0 never terminates, env 1 terminates at t=0; columns must not mix.
    rewards = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    values = np.zeros((2, 2), dtype=np.float32)
    dones = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    last_values = np.array([10.0, 10.0], dtype=np.float32)

    adv, ret = compute_gae(rewards, values, dones, last_values, gamma=1.0, lam=1.0)

    # env 0: t=1: 1 + 10 = 11; t=0: 1 + 11 = 12
    np.testing.assert_allclose(adv[:, 0], [12.0, 11.0], rtol=1e-6)
    # env 1: t=0 done cuts everything after it: adv = 1; t=1 bootstraps: 11
    np.testing.assert_allclose(adv[:, 1], [1.0, 11.0], rtol=1e-6)
