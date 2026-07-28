"""Checkpoint format compliance and save/load round-trips."""
import numpy as np
import torch

from pvpbot.models import PolicyNet
from pvpbot.sim.stub import DuelVecEnv
from pvpbot.spec import ACTION_HEADS, OBS_DIM
from pvpbot.train.league import League
from pvpbot.train.ppo import PPOConfig, PPOTrainer, RolloutBuffer
from pvpbot.train.run import Collector, load_checkpoint, save_checkpoint


def _trained_setup(seed=0, num_envs=8):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = DuelVecEnv(num_envs, seed=seed)
    policy = PolicyNet()
    cfg = PPOConfig(epochs=1, num_minibatches=2, chunk_len=4, rollout_len=8)
    trainer = PPOTrainer(policy, cfg)
    league = League(num_envs, seed=seed)
    collector = Collector(env, policy, trainer, league)
    buf = RolloutBuffer(cfg.rollout_len, num_envs, chunk_len=cfg.chunk_len)
    collector.rollout(buf)
    trainer.update(buf)  # populate optimizer state
    league.gate(policy, trainer.obs_norm, step=64)
    league.learner_elo = 1042.0
    return policy, trainer, league


def test_checkpoint_matches_spec_format(tmp_path):
    policy, trainer, league = _trained_setup()
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, policy, trainer, league, step=1234, update=7)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert "model" in ckpt and "meta" in ckpt
    meta = ckpt["meta"]
    assert meta["obs_dim"] == OBS_DIM
    assert tuple(meta["action_heads"]) == ACTION_HEADS
    assert meta["step"] == 1234
    assert meta["elo"] == 1042.0
    # the model state dict must load into a fresh shared-architecture net
    fresh = PolicyNet()
    fresh.load_state_dict(ckpt["model"])


def test_checkpoint_roundtrip_preserves_outputs(tmp_path):
    policy, trainer, league = _trained_setup(seed=1)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, policy, trainer, league, step=999, update=3)

    torch.manual_seed(123)  # different init for the fresh copies
    policy2 = PolicyNet()
    trainer2 = PPOTrainer(policy2, PPOConfig())
    league2 = League(8, seed=99)
    step, update, pin_stage = load_checkpoint(path, policy2, trainer2, league2)
    assert step == 999 and update == 3
    assert pin_stage is None  # not saved -> not invented

    save_checkpoint(path, policy, trainer, league, step=999, update=3,
                    pin_stage=2)
    _, _, pin_stage = load_checkpoint(path, PolicyNet(), None, None)
    assert pin_stage == 2  # curriculum ladder survives restarts

    raw = np.random.default_rng(7).normal(size=(5, OBS_DIM)).astype(np.float32)
    obs_a = trainer.obs_norm.normalize(raw)
    obs_b = trainer2.obs_norm.normalize(raw)
    np.testing.assert_allclose(obs_a, obs_b, rtol=1e-6)

    h = policy.initial_state(5)
    with torch.no_grad():
        logits_a, val_a, _ = policy(torch.from_numpy(obs_a), h)
        logits_b, val_b, _ = policy2(torch.from_numpy(obs_b), h)
    for la, lb in zip(logits_a, logits_b):
        torch.testing.assert_close(la, lb, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(val_a, val_b, rtol=1e-6, atol=1e-7)

    # league state restored too
    assert league2.learner_elo == league.learner_elo
    assert league2.num_ckpts() == league.num_ckpts()
    # optimizer state came back (adam moments exist)
    assert len(trainer2.opt.state_dict()["state"]) > 0


def test_pool_checkpoint_policies_survive_roundtrip(tmp_path):
    policy, trainer, league = _trained_setup(seed=2)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, policy, trainer, league, step=5, update=1)

    league2 = League(8, seed=5)
    load_checkpoint(path, PolicyNet(), None, league2)
    ck_a = next(e for e in league.pool if e.kind == "ckpt")
    ck_b = next(e for e in league2.pool if e.kind == "ckpt")
    obs = torch.randn(3, OBS_DIM)
    h = torch.zeros(3, PolicyNet.CORE)
    with torch.no_grad():
        la, va, _ = ck_a.net(obs, h)
        lb, vb, _ = ck_b.net(obs, h)
    for a, b in zip(la, lb):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(va, vb, rtol=1e-6, atol=1e-7)
