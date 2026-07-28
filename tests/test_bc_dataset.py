"""Dataset: imbalance weights on a constructed skewed dataset, chunking, split."""
import numpy as np
import pytest
import torch

from pvpbot.bc.dataset import (
    BCSequenceDataset,
    compute_head_class_weights,
    split_recordings,
)
from pvpbot.bc.recording import Recording
from pvpbot.spec import ACTION_HEAD_SIZES, NUM_ACTION_HEADS, OBS_DIM

ATTACK, JUMP = 4, 2  # head indices


def _make_recording(ticks, attack_ticks=0, jump_ticks=0, seed=0):
    """Recording with exactly `attack_ticks` attack=1 and `jump_ticks` jump=1."""
    rng = np.random.default_rng(seed)
    obs = rng.normal(0, 0.3, size=(ticks, OBS_DIM)).astype(np.float32)
    actions = np.zeros((ticks, NUM_ACTION_HEADS), dtype=np.int64)
    actions[:, 0] = 2  # hold W everywhere: the classic skew
    actions[:attack_ticks, ATTACK] = 1
    actions[:jump_ticks, JUMP] = 1
    return Recording(obs, actions, {"tick_rate": 20, "source": "t", "map": "t"})


def test_class_weights_exact_on_skewed_dataset():
    # 1000 ticks, 100 attacks (10%), 50 jumps (5%): the video author's setup
    rec = _make_recording(1000, attack_ticks=100, jump_ticks=50)
    w = compute_head_class_weights([rec], smoothing=0.0)
    assert len(w) == NUM_ACTION_HEADS
    # balanced heuristic: w_c = N / (K * count_c)
    np.testing.assert_allclose(w[ATTACK], [1000 / (2 * 900), 1000 / (2 * 100)],
                               rtol=1e-6)
    np.testing.assert_allclose(w[JUMP], [1000 / (2 * 950), 1000 / (2 * 50)],
                               rtol=1e-6)
    # rare class boosted ~9x relative to majority
    assert w[ATTACK][1] / w[ATTACK][0] == pytest.approx(9.0, rel=1e-5)
    # expected weight under the data distribution is 1 (loss scale unchanged)
    counts = np.array([900, 100])
    assert float((w[ATTACK] * counts).sum()) == pytest.approx(1000, rel=1e-5)


def test_class_weights_zero_count_and_cap():
    rec = _make_recording(200, attack_ticks=20)
    # jump=1 never occurs: with smoothing it gets a large finite weight
    w = compute_head_class_weights([rec], smoothing=1.0)
    assert np.isfinite(w[JUMP]).all()
    assert w[JUMP][1] > w[JUMP][0]
    # with smoothing=0 the unseen class gets weight 0 (never a target anyway)
    w0 = compute_head_class_weights([rec], smoothing=0.0)
    assert w0[JUMP][1] == 0.0
    # cap applies
    wc = compute_head_class_weights([rec], smoothing=1.0, max_weight=3.0)
    assert max(float(x.max()) for x in wc) <= 3.0


def test_class_weights_accumulate_over_recordings():
    recs = [_make_recording(100, attack_ticks=10, seed=i) for i in range(3)]
    w_multi = compute_head_class_weights(recs, smoothing=0.0)
    w_single = compute_head_class_weights(
        [_make_recording(300, attack_ticks=30)], smoothing=0.0
    )
    for a, b in zip(w_multi, w_single):
        np.testing.assert_allclose(a, b, rtol=1e-6)


def test_dataset_chunking_and_mask():
    rec = _make_recording(100, attack_ticks=30, seed=3)
    ds = BCSequenceDataset([rec], seq_len=32, burn_in=8, stride=16)
    window = 40
    expected_chunks = (100 - window) // 16 + 1
    assert len(ds) == expected_chunks
    sample = ds[0]
    assert sample["obs"].shape == (window, OBS_DIM)
    assert sample["obs"].dtype == torch.float32
    assert sample["actions"].shape == (window, NUM_ACTION_HEADS)
    assert sample["actions"].dtype == torch.int64
    mask = sample["loss_mask"]
    assert mask.shape == (window,)
    assert (mask[:8] == 0).all() and (mask[8:] == 1).all()
    # chunk content matches the source slice (second chunk starts at 16)
    s1 = ds[1]
    np.testing.assert_array_equal(s1["obs"].numpy(), rec.obs[16:56])
    np.testing.assert_array_equal(s1["actions"].numpy(), rec.actions[16:56])


def test_dataset_skips_short_episodes():
    long_rec = _make_recording(80, seed=1)
    short_rec = _make_recording(10, seed=2)  # < window of 40
    ds = BCSequenceDataset([long_rec, short_rec], seq_len=32, burn_in=8)
    assert ds.num_skipped == 1
    with pytest.raises(ValueError):
        BCSequenceDataset([short_rec], seq_len=32, burn_in=8)


def test_split_recordings_deterministic_and_disjoint():
    recs = [_make_recording(50, seed=i) for i in range(10)]
    tr1, va1 = split_recordings(recs, val_frac=0.3, seed=42)
    tr2, va2 = split_recordings(recs, val_frac=0.3, seed=42)
    assert [id(r) for r in tr1] == [id(r) for r in tr2]
    assert [id(r) for r in va1] == [id(r) for r in va2]
    assert len(va1) == 3 and len(tr1) == 7
    assert {id(r) for r in tr1}.isdisjoint({id(r) for r in va1})
    with pytest.raises(ValueError):
        split_recordings(recs[:1], val_frac=0.3)
