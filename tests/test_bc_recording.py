"""Recording format: round-trip fidelity, validation, mouse<->bin quantization."""
import json

import numpy as np
import pytest
import torch

from pvpbot.bc.recording import (
    RecordingWriter,
    bins_to_mouse,
    iter_recordings,
    list_recordings,
    load_recording,
    mouse_to_bins,
)
from pvpbot.spec import ACTION_HEAD_SIZES, CAMERA_BINS, NUM_ACTION_HEADS, OBS_DIM


def _random_episode(rng, ticks):
    obs = rng.normal(0, 0.5, size=(ticks, OBS_DIM)).astype(np.float32)
    actions = np.stack(
        [rng.integers(0, n, size=ticks) for n in ACTION_HEAD_SIZES], axis=1
    ).astype(np.int64)
    return obs, actions


def test_round_trip_fidelity(tmp_path):
    rng = np.random.default_rng(0)
    obs, actions = _random_episode(rng, 37)
    path = tmp_path / "ep.npz"
    w = RecordingWriter(path, tick_rate=20, source="human", map_name="flat",
                        extra_meta={"session": 7})
    for t in range(37):
        w.append(obs[t], actions[t])
    assert len(w) == 37
    out = w.finalize()
    rec = load_recording(out)
    np.testing.assert_array_equal(rec.obs, obs)
    np.testing.assert_array_equal(rec.actions, actions)
    assert rec.obs.dtype == np.float32 and rec.actions.dtype == np.int64
    assert rec.meta["tick_rate"] == 20
    assert rec.meta["source"] == "human"
    assert rec.meta["map"] == "flat"
    assert rec.meta["session"] == 7
    assert rec.num_ticks == 37
    to, ta = rec.torch_tensors()
    assert to.dtype == torch.float32 and to.shape == (37, OBS_DIM)
    assert ta.dtype == torch.int64 and ta.shape == (37, NUM_ACTION_HEADS)


def test_writer_context_manager(tmp_path):
    rng = np.random.default_rng(1)
    obs, actions = _random_episode(rng, 5)
    path = tmp_path / "ctx.npz"
    with RecordingWriter(path) as w:
        for t in range(5):
            w.append(obs[t], actions[t])
    assert load_recording(path).num_ticks == 5


def test_writer_validation(tmp_path):
    w = RecordingWriter(tmp_path / "bad.npz")
    good_obs = np.zeros(OBS_DIM, dtype=np.float32)
    good_act = np.zeros(NUM_ACTION_HEADS, dtype=np.int64)
    with pytest.raises(ValueError):
        w.append(np.zeros(OBS_DIM - 1), good_act)  # wrong obs size
    with pytest.raises(ValueError):
        w.append(good_obs, np.zeros(NUM_ACTION_HEADS - 1, dtype=np.int64))
    with pytest.raises(ValueError):
        w.append(good_obs, good_act.astype(np.float32))  # non-integer action
    bad = good_act.copy()
    bad[4] = 2  # attack head has 2 classes -> max index 1
    with pytest.raises(ValueError):
        w.append(good_obs, bad)
    nan_obs = good_obs.copy()
    nan_obs[0] = np.nan
    with pytest.raises(ValueError):
        w.append(nan_obs, good_act)
    with pytest.raises(ValueError):
        w.finalize()  # empty
    w.append(good_obs, good_act)
    w.finalize()
    with pytest.raises(RuntimeError):
        w.finalize()  # double finalize
    with pytest.raises(RuntimeError):
        w.append(good_obs, good_act)  # append after finalize


def test_loader_rejects_out_of_range_actions(tmp_path):
    obs = np.zeros((4, OBS_DIM), dtype=np.float32)
    actions = np.zeros((4, NUM_ACTION_HEADS), dtype=np.int64)
    actions[2, 5] = 11  # yaw head only has 11 bins (0..10)
    path = tmp_path / "corrupt.npz"
    np.savez(path, obs=obs, actions=actions,
             meta=np.asarray(json.dumps({"tick_rate": 20, "source": "x", "map": "y"})))
    with pytest.raises(ValueError, match="yaw"):
        load_recording(path)


def test_loader_rejects_missing_meta_key(tmp_path):
    obs = np.zeros((2, OBS_DIM), dtype=np.float32)
    actions = np.zeros((2, NUM_ACTION_HEADS), dtype=np.int64)
    path = tmp_path / "nometa.npz"
    np.savez(path, obs=obs, actions=actions,
             meta=np.asarray(json.dumps({"tick_rate": 20})))
    with pytest.raises(ValueError, match="meta"):
        load_recording(path)


def test_loader_rejects_shape_mismatch(tmp_path):
    obs = np.zeros((3, OBS_DIM), dtype=np.float32)
    actions = np.zeros((4, NUM_ACTION_HEADS), dtype=np.int64)  # T mismatch
    path = tmp_path / "mismatch.npz"
    np.savez(path, obs=obs, actions=actions,
             meta=np.asarray(json.dumps({"tick_rate": 20, "source": "x", "map": "y"})))
    with pytest.raises(ValueError):
        load_recording(path)


def test_iter_recordings_sorted(tmp_path):
    for name in ("b.npz", "a.npz"):
        with RecordingWriter(tmp_path / name) as w:
            w.append(np.zeros(OBS_DIM), np.zeros(NUM_ACTION_HEADS, dtype=np.int64))
    paths = list_recordings(tmp_path)
    assert [p.split("/")[-1] for p in paths] == ["a.npz", "b.npz"]
    assert len(list(iter_recordings(tmp_path))) == 2


# ---------------------------------------------------------------------------
# mouse <-> bins
# ---------------------------------------------------------------------------

def test_bins_round_trip_all_indices():
    for i in range(len(CAMERA_BINS)):
        for j in range(len(CAMERA_BINS)):
            dx, dy = bins_to_mouse(i, j)
            assert (dx, dy) == (CAMERA_BINS[i], CAMERA_BINS[j])
            assert mouse_to_bins(dx, dy) == (i, j)


def test_mouse_to_bins_nearest_and_clamp():
    # nearest-bin quantization
    assert mouse_to_bins(0.4, 0.0)[0] == CAMERA_BINS.index(0.0)
    assert mouse_to_bins(0.6, 0.0)[0] == CAMERA_BINS.index(1.0)
    assert mouse_to_bins(-2.4, 0.0)[0] == CAMERA_BINS.index(-3.0)
    assert mouse_to_bins(11.5, 0.0)[0] == CAMERA_BINS.index(15.0)
    # far outside the bin range clamps to the extremes
    assert mouse_to_bins(500.0, -500.0) == (len(CAMERA_BINS) - 1, 0)


def test_mouse_to_bins_array_and_types():
    dx = np.array([-30.0, 0.2, 6.9])
    dy = np.array([0.0, 0.0, 0.0])
    yi, pi = mouse_to_bins(dx, dy)
    assert yi.dtype == np.int64 and yi.shape == (3,)
    np.testing.assert_array_equal(
        yi, [0, CAMERA_BINS.index(0.0), CAMERA_BINS.index(7.0)]
    )
    bx, by = bins_to_mouse(yi, pi)
    yi2, _ = mouse_to_bins(bx, by)
    np.testing.assert_array_equal(yi, yi2)  # quantize(dequantize(i)) == i
    # scalars come back as plain python numbers
    a, b = mouse_to_bins(3.0, -3.0)
    assert isinstance(a, int) and isinstance(b, int)
    with pytest.raises(ValueError):
        bins_to_mouse(len(CAMERA_BINS), 0)
    with pytest.raises(ValueError):
        mouse_to_bins(float("nan"), 0.0)
