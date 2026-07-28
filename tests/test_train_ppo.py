"""PPO core: multi-discrete math, obs normalization, loss decreases."""
import numpy as np
import torch

from pvpbot.models import PolicyNet
from pvpbot.sim.stub import DuelVecEnv
from pvpbot.spec import ACTION_HEAD_SIZES, NUM_ACTION_HEADS, OBS_DIM
from pvpbot.train.league import League
from pvpbot.train.ppo import (
    PPOConfig,
    PPOTrainer,
    RolloutBuffer,
    RunningNorm,
    heads_log_prob_entropy,
    sample_multi_discrete,
)
from pvpbot.train.run import Collector


def test_multi_discrete_logp_entropy_matches_manual():
    torch.manual_seed(0)
    b = 6
    logits = [torch.randn(b, n) for n in ACTION_HEAD_SIZES]
    actions = torch.stack(
        [torch.randint(0, n, (b,)) for n in ACTION_HEAD_SIZES], dim=-1
    )
    logp, ent = heads_log_prob_entropy(logits, actions)
    assert logp.shape == (b,)
    assert ent.shape == (b, NUM_ACTION_HEADS)

    manual_logp = torch.zeros(b)
    for i, l in enumerate(logits):
        dist = torch.distributions.Categorical(logits=l)
        manual_logp += dist.log_prob(actions[:, i])
        torch.testing.assert_close(ent[:, i], dist.entropy(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(logp, manual_logp, rtol=1e-5, atol=1e-6)


def test_sample_multi_discrete_bounds_and_logp():
    torch.manual_seed(1)
    b = 512
    logits = [torch.randn(b, n) for n in ACTION_HEAD_SIZES]
    actions, logp = sample_multi_discrete(logits)
    assert actions.shape == (b, NUM_ACTION_HEADS)
    for i, n in enumerate(ACTION_HEAD_SIZES):
        assert int(actions[:, i].min()) >= 0
        assert int(actions[:, i].max()) < n
    # returned logp must equal the joint log-prob of the sampled actions
    expected, _ = heads_log_prob_entropy(logits, actions)
    torch.testing.assert_close(logp, expected, rtol=1e-5, atol=1e-6)


def test_running_norm_tracks_moments():
    rng = np.random.default_rng(0)
    norm = RunningNorm(4)
    data = rng.normal(loc=3.0, scale=2.0, size=(4096, 4))
    for i in range(0, 4096, 256):
        norm.update(data[i : i + 256])
    np.testing.assert_allclose(norm.mean, data.mean(axis=0), atol=0.05)
    np.testing.assert_allclose(norm.var, data.var(axis=0), rtol=0.05)
    z = norm.normalize(data)
    assert z.dtype == np.float32
    assert abs(z.mean()) < 0.05 and abs(z.std() - 1.0) < 0.05


def _tiny_rollout(seed=0, num_envs=16, rollout_len=8, chunk_len=4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = DuelVecEnv(num_envs, seed=seed)
    policy = PolicyNet()
    cfg = PPOConfig(
        lr=1e-3, epochs=3, num_minibatches=2, ent_coef=0.0,
        chunk_len=chunk_len, rollout_len=rollout_len,
    )
    trainer = PPOTrainer(policy, cfg)
    league = League(num_envs, p_self=1.0, seed=seed)
    collector = Collector(env, policy, trainer, league)
    buf = RolloutBuffer(rollout_len, num_envs, chunk_len=chunk_len)
    collector.rollout(buf)
    return trainer, buf


def test_ppo_update_decreases_loss_on_fixed_batch():
    trainer, buf = _tiny_rollout()
    loss_before = trainer.evaluate_loss(buf)
    metrics = trainer.update(buf)
    loss_after = trainer.evaluate_loss(buf)
    assert np.isfinite(loss_before) and np.isfinite(loss_after)
    assert loss_after < loss_before
    for key in ("loss", "pg_loss", "v_loss", "entropy", "approx_kl", "grad_norm"):
        assert np.isfinite(metrics[key]), key


def test_ppo_update_reports_per_head_entropy():
    trainer, buf = _tiny_rollout(seed=2)
    metrics = trainer.update(buf)
    head_keys = [k for k in metrics if k.startswith("entropy_")]
    assert len(head_keys) == NUM_ACTION_HEADS
    # summed per-head entropies should match the total entropy metric
    total = sum(metrics[k] for k in head_keys)
    assert abs(total - metrics["entropy"]) < 1e-4


def test_rollout_buffer_requires_chunk_alignment():
    try:
        RolloutBuffer(10, 4, chunk_len=4)
    except ValueError:
        return
    raise AssertionError("expected ValueError for T not divisible by chunk_len")


def test_obs_normalization_is_applied_and_stored():
    trainer, buf = _tiny_rollout(seed=3)
    # stored observations are the normalized ones: bounded by the clip range
    assert np.abs(buf.obs).max() <= trainer.obs_norm.clip + 1e-6
    assert trainer.obs_norm.count > 1.0  # stats actually accumulated
    assert buf.obs.shape == (buf.T, buf.N, OBS_DIM)
