"""Humanization: delay length, smoother time structure, jitter determinism."""
import numpy as np
import pytest

from pvpbot.bc.humanize import (
    IDLE_ACTION,
    ClickJitter,
    MouseSmoother,
    ReactionDelay,
)
from pvpbot.spec import CAMERA_BINS, NUM_ACTION_HEADS, TICK_RATE

TICK = 1.0 / TICK_RATE


def _action(v):
    a = np.zeros(NUM_ACTION_HEADS, dtype=np.int64)
    a[0] = v % 3
    a[4] = v % 2
    a[5] = v % 11
    return a


# ---------------------------------------------------------------------------
# ReactionDelay
# ---------------------------------------------------------------------------

def test_reaction_delay_length():
    delay = ReactionDelay(delay_ticks=3)
    assert delay.delay_seconds == pytest.approx(0.150)
    inputs = [_action(i) for i in range(10)]
    outputs = [delay.step(a) for a in inputs]
    # first 3 outputs are idle, then output t == input t-3
    for t in range(3):
        np.testing.assert_array_equal(outputs[t], IDLE_ACTION)
    for t in range(3, 10):
        np.testing.assert_array_equal(outputs[t], inputs[t - 3])


def test_reaction_delay_reset_and_zero():
    delay = ReactionDelay(delay_ticks=2)
    delay.step(_action(1))
    delay.reset()
    np.testing.assert_array_equal(delay.step(_action(5)), IDLE_ACTION)
    passthrough = ReactionDelay(delay_ticks=0)
    np.testing.assert_array_equal(passthrough.step(_action(7)), _action(7))


def test_reaction_delay_output_is_a_copy():
    delay = ReactionDelay(delay_ticks=1)
    out = delay.step(_action(0))
    out[0] = 99  # mutating the returned idle must not poison later outputs
    np.testing.assert_array_equal(delay.step(_action(1)), _action(0))
    out2 = delay.step(_action(2))
    np.testing.assert_array_equal(out2, _action(1))


# ---------------------------------------------------------------------------
# MouseSmoother
# ---------------------------------------------------------------------------

def test_smoother_monotone_time_summing_to_one_tick():
    sm = MouseSmoother(substeps=8)
    yaw_idx = CAMERA_BINS.index(7.0)
    pitch_idx = CAMERA_BINS.index(-3.0)
    moves = sm.smooth(yaw_idx, pitch_idx)
    assert len(moves) == 8
    dts = [m[0] for m in moves]
    assert all(dt > 0 for dt in dts)  # cumulative time strictly increasing
    assert sum(dts) == pytest.approx(TICK, abs=1e-12)
    # deltas integrate to the full binned delta
    assert sum(m[1] for m in moves) == pytest.approx(7.0, abs=1e-9)
    assert sum(m[2] for m in moves) == pytest.approx(-3.0, abs=1e-9)


def test_smoother_minimum_jerk_shape():
    sm = MouseSmoother(substeps=10)
    moves = sm.smooth_degrees(10.0, 0.0)
    dxs = [m[1] for m in moves]
    # bell-shaped velocity: peak in the middle, slow at the ends
    peak = max(range(10), key=lambda i: dxs[i])
    assert peak in (4, 5)
    assert dxs[0] < dxs[4] and dxs[-1] < dxs[5]
    # symmetric profile
    for a, b in zip(dxs, reversed(dxs)):
        assert a == pytest.approx(b, abs=1e-9)
    # all micro-moves point the same way as the total (no backtracking)
    assert all(dx >= 0 for dx in dxs)


def test_smoother_zero_delta_and_determinism():
    sm = MouseSmoother(substeps=4)
    zero_idx = CAMERA_BINS.index(0.0)
    moves = sm.smooth(zero_idx, zero_idx)
    assert all(m[1] == 0.0 and m[2] == 0.0 for m in moves)
    assert sum(m[0] for m in moves) == pytest.approx(TICK, abs=1e-12)
    assert sm.smooth(8, 3) == sm.smooth(8, 3)  # deterministic
    with pytest.raises(ValueError):
        sm.smooth(len(CAMERA_BINS), 0)


# ---------------------------------------------------------------------------
# ClickJitter
# ---------------------------------------------------------------------------

def test_click_jitter_deterministic_under_seed():
    j1 = ClickJitter(sigma_ms=12.0, seed=123)
    j2 = ClickJitter(sigma_ms=12.0, seed=123)
    seq1 = [j1.offset(1) for _ in range(50)]
    seq2 = [j2.offset(1) for _ in range(50)]
    assert seq1 == seq2
    assert len(set(seq1)) > 1  # actually jittered, not constant
    j1.reset()
    assert [j1.offset(1) for _ in range(50)] == seq1


def test_click_jitter_bounds_and_no_press():
    j = ClickJitter(sigma_ms=25.0, seed=0)
    assert j.offset(0) is None
    for _ in range(200):
        off = j.offset(1)
        assert 0.0 <= off < TICK
    assert ClickJitter(sigma_ms=0.0, seed=0).offset(1) == 0.0
    with pytest.raises(ValueError):
        ClickJitter(sigma_ms=-1.0)
