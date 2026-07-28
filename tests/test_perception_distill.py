"""DAgger-style collector: (frame, obs, action) triples from the stub env."""
import numpy as np

from pvpbot.spec import ACTION_HEAD_SIZES, FRAME_SHAPE, NUM_ACTION_HEADS, OBS_DIM
from pvpbot.perception.distill import collect, collect_arrays


def test_collect_yields_valid_triples():
    count = 0
    for frame, obs, action in collect(ticks=30, seed=3):
        assert frame.shape == FRAME_SHAPE and frame.dtype == np.uint8
        assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
        assert np.isfinite(obs).all()
        assert action.shape == (NUM_ACTION_HEADS,) and action.dtype == np.int64
        for i, n in enumerate(ACTION_HEAD_SIZES):
            assert 0 <= action[i] < n
        count += 1
    assert count == 30


def test_collect_arrays_stacks_and_is_deterministic():
    f1, o1, a1 = collect_arrays(ticks=20, seed=9)
    f2, o2, a2 = collect_arrays(ticks=20, seed=9)
    assert f1.shape == (20,) + FRAME_SHAPE
    assert o1.shape == (20, OBS_DIM) and a1.shape == (20, NUM_ACTION_HEADS)
    assert np.array_equal(f1, f2)
    assert np.array_equal(o1, o2)
    assert np.array_equal(a1, a2)
    # frames are not all identical: the duel actually evolves
    assert not np.array_equal(f1[0], f1[-1])
