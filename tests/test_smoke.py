"""Contract smoke tests: stub env API shapes and shared model shapes."""
import numpy as np
import torch

from pvpbot.models import PerceptionCNN, PolicyNet
from pvpbot.sim.stub import DuelVecEnv
from pvpbot.spec import (
    ACTION_HEAD_SIZES,
    FRAME_SHAPE,
    NUM_ACTION_HEADS,
    OBS_DIM,
    OBS_LAYOUT,
    PERCEPTION_DIM,
)


def test_obs_layout_covers_48():
    covered = np.zeros(OBS_DIM, dtype=int)
    for start, stop in OBS_LAYOUT.values():
        covered[start:stop] += 1
    assert (covered == 1).all(), "OBS_LAYOUT must tile [0,48) exactly once"


def test_stub_env_contract():
    env = DuelVecEnv(num_envs=8, seed=1)
    obs = env.reset()
    assert obs.shape == (8, 2, OBS_DIM) and obs.dtype == np.float32
    rng = np.random.default_rng(0)
    for _ in range(50):
        actions = np.stack(
            [rng.integers(0, n, (8, 2)) for n in ACTION_HEAD_SIZES], axis=-1
        )
        obs, rew, done, info = env.step(actions)
        assert obs.shape == (8, 2, OBS_DIM)
        assert rew.shape == (8, 2) and rew.dtype == np.float32
        assert done.shape == (8,) and done.dtype == bool
        assert info["win"].shape == (8, 2)


def test_policy_net_shapes():
    net = PolicyNet()
    b = 4
    state = net.initial_state(b)
    obs = torch.randn(b, OBS_DIM)
    logits, value, h = net(obs, state)
    assert len(logits) == NUM_ACTION_HEADS
    for l, n in zip(logits, ACTION_HEAD_SIZES):
        assert l.shape == (b, n)
    assert value.shape == (b,)
    actions, h2 = net.act(obs, h)
    assert actions.shape == (b, NUM_ACTION_HEADS)
    assert h2.shape == h.shape


def test_perception_cnn_shapes():
    net = PerceptionCNN()
    frames = torch.randn(4, *FRAME_SHAPE)
    out = net(frames)
    assert out.shape == (4, PERCEPTION_DIM)
