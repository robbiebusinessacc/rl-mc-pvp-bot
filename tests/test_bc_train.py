"""End-to-end BC training on synth demos: metrics, imbalance handling,
checkpoint contract. The attack head must beat the majority-class baseline --
this is precisely the hold-W collapse check (accuracy alone would pass a
policy that never clicks)."""
import numpy as np
import pytest
import torch

from pvpbot.bc.kl import KLPriorLoss
from pvpbot.bc.synth_demos import generate_demos
from pvpbot.bc.train_bc import (
    load_bc_checkpoint,
    macro_f1,
    train_bc,
)
from pvpbot.models import PolicyNet
from pvpbot.spec import ACTION_HEADS, OBS_DIM


def test_macro_f1_properties():
    t = np.array([0] * 90 + [1] * 10)
    # majority-class predictor: high accuracy, mediocre macro-F1
    p_major = np.zeros(100, dtype=int)
    assert float(np.mean(p_major == t)) == 0.9
    assert macro_f1(p_major, t, 2) == pytest.approx(2 * 90 / (2 * 90 + 10) / 2)
    # perfect predictor
    assert macro_f1(t, t, 2) == pytest.approx(1.0)
    # classes with zero support in the target are excluded from the average
    assert macro_f1(np.zeros(5, dtype=int), np.zeros(5, dtype=int), 3) == 1.0


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("bc_train")
    data_dir = root / "demos"
    generate_demos(str(data_dir), episodes=8, seed=0, max_ticks=240)
    results = train_bc(
        data_dir=str(data_dir),
        out_path=str(root / "bc_policy.pt"),
        epochs=20,
        seq_len=32,
        burn_in=8,
        batch_size=32,
        lr=1e-3,
        val_frac=0.25,
        max_weight=10.0,
        seed=0,
        log=None,
    )
    return results


def test_attack_macro_f1_beats_trivial_baseline(trained):
    val = trained["val"]
    f1 = val["macro_f1"]["attack"]
    baseline = val["baseline_macro_f1"]["attack"]
    # a hold-W-collapsed policy scores exactly `baseline` here (~0.48)
    assert baseline < 0.55, "baseline should reflect heavy skew"
    assert f1 > baseline + 0.03, (
        "attack macro-F1 %.3f did not beat majority baseline %.3f: "
        "hold-W collapse" % (f1, baseline)
    )
    assert f1 > 0.5


def test_learnable_heads_beat_their_baselines(trained):
    val = trained["val"]
    for head in ("forward", "sprint", "yaw"):
        assert val["macro_f1"][head] > val["baseline_macro_f1"][head], head
    # accuracy is reported for every head
    assert set(val["acc"].keys()) == {name for name, _ in ACTION_HEADS}


def test_checkpoint_spec_format_and_reload(trained):
    path = trained["checkpoint"]
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert set(ckpt.keys()) == {"model", "meta"}
    meta = ckpt["meta"]
    for key in ("obs_dim", "action_heads", "step"):  # spec.py contract
        assert key in meta
    assert meta["obs_dim"] == OBS_DIM
    assert [tuple(h) for h in meta["action_heads"]] == list(ACTION_HEADS)
    assert meta["step"] == trained["steps"] > 0
    # weights load into a fresh PolicyNet
    net = load_bc_checkpoint(path)
    assert isinstance(net, PolicyNet)
    obs = torch.zeros(2, OBS_DIM)
    logits, value, h = net(obs, net.initial_state(2))
    assert len(logits) == len(ACTION_HEADS)
    # and straight into the KL prior for the PPO trainer
    prior = KLPriorLoss.from_checkpoint(path)
    kl, _ = prior(logits, obs, prior.initial_state(2))
    assert float(kl.detach()) == pytest.approx(0.0, abs=1e-6)


def test_history_and_selection(trained):
    hist = trained["history"]
    assert len(hist) == 20
    assert all("val" in h and "train_loss" in h for h in hist)
    # training loss decreased overall
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    assert 0 <= trained["selected_epoch"] < 20
