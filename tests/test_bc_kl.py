"""KL prior loss: zero against itself, positive against a perturbed policy,
gradients reach the RL policy only."""
import copy
import math

import pytest
import torch

from pvpbot.bc.kl import KLPriorLoss, kl_categorical
from pvpbot.models import PolicyNet
from pvpbot.spec import OBS_DIM


def _obs(b=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, OBS_DIM, generator=g)


def test_kl_categorical_known_value():
    # p uniform (zeros logits), q = [0.9, 0.1]: KL = ln(5/3)
    logits_p = torch.zeros(1, 2)
    logits_q = torch.log(torch.tensor([[0.9, 0.1]]))
    kl = kl_categorical(logits_p, logits_q)
    expected = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)
    assert float(kl.detach()) == pytest.approx(expected, rel=1e-5)
    # KL(p || p) == 0
    assert float(kl_categorical(logits_q, logits_q)) == pytest.approx(0.0, abs=1e-7)


def test_kl_zero_against_itself():
    torch.manual_seed(0)
    bc = PolicyNet()
    rl = copy.deepcopy(bc)
    prior = KLPriorLoss(bc)
    obs = _obs()
    rl_state = rl.initial_state(obs.shape[0])
    bc_state = prior.initial_state(obs.shape[0])
    rl_logits, _, _ = rl(obs, rl_state)
    kl, new_bc_state = prior(rl_logits, obs, bc_state)
    assert float(kl.detach()) == pytest.approx(0.0, abs=1e-6)
    assert new_bc_state.shape == bc_state.shape
    assert not new_bc_state.requires_grad


def test_kl_positive_against_perturbed_policy():
    torch.manual_seed(1)
    bc = PolicyNet()
    rl = copy.deepcopy(bc)
    with torch.no_grad():
        for head in rl.heads:
            head.weight.add_(0.5 * torch.randn_like(head.weight))
    prior = KLPriorLoss(bc)
    obs = _obs(seed=1)
    rl_logits, _, _ = rl(obs, rl.initial_state(obs.shape[0]))
    kl, _ = prior(rl_logits, obs, prior.initial_state(obs.shape[0]))
    assert float(kl.detach()) > 0.01


def test_gradients_flow_to_rl_policy_only():
    torch.manual_seed(2)
    bc = PolicyNet()
    rl = PolicyNet()
    prior = KLPriorLoss(bc)
    obs = _obs(seed=2)
    rl_logits, _, _ = rl(obs, rl.initial_state(obs.shape[0]))
    kl, _ = prior(rl_logits, obs, prior.initial_state(obs.shape[0]))
    kl.backward()
    rl_grads = [p.grad for p in rl.parameters() if p.grad is not None]
    assert rl_grads, "RL policy received no gradients"
    assert any(float(g.abs().sum()) > 0 for g in rl_grads)
    # the frozen internal prior: no grads, no grad tracking
    for p in prior.policy.parameters():
        assert not p.requires_grad
        assert p.grad is None
    # the caller's BC net was deep-copied, not mutated
    for p in bc.parameters():
        assert p.requires_grad
        assert p.grad is None


def test_from_policy_convenience_and_state_threading():
    torch.manual_seed(3)
    bc, rl = PolicyNet(), PolicyNet()
    prior = KLPriorLoss(bc)
    obs = _obs(b=4, seed=3)
    rl_state = rl.initial_state(4)
    bc_state = prior.initial_state(4)
    kl, new_rl_state, new_bc_state = prior.from_policy(rl, obs, rl_state, bc_state)
    assert kl.dim() == 0 and float(kl.detach()) >= 0.0
    assert new_rl_state.shape == rl_state.shape
    assert new_bc_state.shape == bc_state.shape
    assert new_rl_state.requires_grad  # RL graph intact
    assert not new_bc_state.requires_grad
    # state actually advances (nonzero after one step)
    assert float(new_bc_state.abs().sum()) > 0
    # reset_state zeros done rows only
    done = torch.tensor([True, False, True, False])
    reset = KLPriorLoss.reset_state(new_bc_state, done)
    assert float(reset[0].abs().sum()) == 0.0
    assert torch.equal(reset[1], new_bc_state[1])


def test_from_checkpoint_round_trip(tmp_path):
    torch.manual_seed(4)
    bc = PolicyNet()
    path = str(tmp_path / "bc.pt")
    torch.save(
        {"model": bc.state_dict(),
         "meta": {"obs_dim": OBS_DIM, "action_heads": [], "step": 0}},
        path,
    )
    prior = KLPriorLoss.from_checkpoint(path)
    obs = _obs(b=8, seed=4)
    rl_logits, _, _ = bc(obs, bc.initial_state(8))
    kl, _ = prior(rl_logits, obs, prior.initial_state(8))
    assert float(kl.detach()) == pytest.approx(0.0, abs=1e-6)


def test_rejects_malformed_logits():
    bc = PolicyNet()
    prior = KLPriorLoss(bc)
    obs = _obs(b=2)
    state = prior.initial_state(2)
    with pytest.raises(ValueError):
        prior([torch.zeros(2, 3)], obs, state)  # wrong head count
    with pytest.raises(ValueError):
        prior([torch.zeros(2, 99)] * 7, obs, state)  # wrong class count
