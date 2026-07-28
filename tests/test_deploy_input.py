"""Deploy module: input injection — mock sink, calibration math, guards."""
import numpy as np
import pytest

from pvpbot.deploy import input_inject
from pvpbot.deploy.input_inject import (
    Calibration,
    MockInputSink,
    PixelAccumulator,
    check_input_available,
)
from pvpbot.spec import CAMERA_BINS


# ---------------------------------------------------------------------------
# MockInputSink
# ---------------------------------------------------------------------------

def test_mock_sink_records_calls_with_timestamps():
    t = {"now": 100.0}

    def fake_time():
        t["now"] += 0.05
        return t["now"]

    sink = MockInputSink(time_fn=fake_time)
    sink.move_mouse(5, -3)
    sink.key("w", True)
    sink.click(True)
    sink.click(False)
    sink.key("w", False)

    assert [e.kind for e in sink.events] == ["move", "key", "click", "click", "key"]
    assert sink.events[0].data == (5, -3)
    assert sink.events[1].data == ("w", True)
    times = [e.t for e in sink.events]
    assert times == sorted(times) and times[0] == pytest.approx(100.05)
    assert sink.held_keys == set() and sink.mouse_down is False


def test_mock_sink_release_all():
    sink = MockInputSink()
    sink.key("w", True)
    sink.key("space", True)
    sink.click(True)
    sink.release_all()
    assert sink.held_keys == set()
    assert sink.mouse_down is False
    ups = [e for e in sink.events_of("key") if e.data[1] is False]
    assert {e.data[0] for e in ups} == {"w", "space"}


def test_mock_sink_total_mouse_delta():
    sink = MockInputSink()
    sink.move_mouse(10, 2)
    sink.move_mouse(-4, 1)
    assert sink.total_mouse_delta() == (6, 3)


# ---------------------------------------------------------------------------
# Calibration math
# ---------------------------------------------------------------------------

def test_calibration_roundtrip_degrees_pixels():
    cal = Calibration(px_per_degree_x=12.9, px_per_degree_y=8.4)
    for yaw, pitch in [(30.0, -15.0), (-7.0, 3.0), (0.0, 0.0), (1.0, -1.0)]:
        dx, dy = cal.degrees_to_pixels(yaw, pitch)
        back = cal.pixels_to_degrees(dx, dy)
        assert back[0] == pytest.approx(yaw, abs=1e-9)
        assert back[1] == pytest.approx(pitch, abs=1e-9)


def test_calibration_pixel_values():
    cal = Calibration(px_per_degree_x=10.0, px_per_degree_y=20.0)
    assert cal.degrees_to_pixels(3.0, -1.5) == (30.0, -30.0)
    assert cal.pixels_to_degrees(30.0, -30.0) == (3.0, -1.5)


def test_calibration_bins_to_pixels_matches_camera_bins():
    cal = Calibration(px_per_degree_x=10.0, px_per_degree_y=10.0)
    for i, deg in enumerate(CAMERA_BINS):
        dx, dy = cal.bins_to_pixels(i, i)
        assert dx == pytest.approx(deg * 10.0)
        assert dy == pytest.approx(deg * 10.0)
    # zero bin injects nothing
    zero = CAMERA_BINS.index(0.0)
    assert cal.bins_to_pixels(zero, zero) == (0.0, 0.0)


def test_calibration_rejects_nonpositive():
    with pytest.raises(ValueError):
        Calibration(px_per_degree_x=0.0)
    with pytest.raises(ValueError):
        Calibration(px_per_degree_y=-3.0)


def test_pixel_accumulator_carries_fractions():
    acc = PixelAccumulator()
    # 0.3 px/tick: integers emitted must track the exact sum within 1 px
    emitted = 0
    for i in range(1000):
        ox, oy = acc.step(0.3, -0.3)
        emitted += ox
        assert oy == -ox
    assert abs(emitted - 300) <= 1


def test_pixel_accumulator_integer_passthrough():
    acc = PixelAccumulator()
    assert acc.step(5.0, -2.0) == (5, -2)
    assert acc.step(0.0, 0.0) == (0, 0)


def test_bins_through_accumulator_roundtrip_total_degrees():
    """Sum of injected pixels over many ticks == sum of commanded degrees."""
    cal = Calibration(px_per_degree_x=12.9, px_per_degree_y=12.9)
    rng = np.random.default_rng(0)
    acc = PixelAccumulator()
    total_deg = 0.0
    total_px = 0
    for _ in range(500):
        b = int(rng.integers(0, len(CAMERA_BINS)))
        total_deg += CAMERA_BINS[b]
        dx, _ = cal.bins_to_pixels(b, CAMERA_BINS.index(0.0))
        ox, _ = acc.step(dx, 0.0)
        total_px += ox
    back_deg = cal.pixels_to_degrees(total_px, 0.0)[0]
    assert back_deg == pytest.approx(total_deg, abs=1.0 / 12.9)  # within 1 px


# ---------------------------------------------------------------------------
# Quartz guards
# ---------------------------------------------------------------------------

def _raise_import_error():
    raise ImportError("No module named 'Quartz'")


def test_quartz_input_sink_guard_message(monkeypatch):
    monkeypatch.setattr(input_inject, "_load_quartz", _raise_import_error)
    with pytest.raises(RuntimeError) as ei:
        input_inject.QuartzInputSink()
    msg = str(ei.value)
    assert "pyobjc-framework-Quartz" in msg
    assert "pip install" in msg
    assert "MockInputSink" in msg


def test_check_input_available_without_quartz(monkeypatch):
    monkeypatch.setattr(input_inject, "_load_quartz", _raise_import_error)
    report = check_input_available()
    assert report["quartz_importable"] is False
    assert "pyobjc" in report["detail"]


def test_calibration_procedure_documented():
    doc = input_inject.__doc__
    assert "calibration" in doc.lower()
    assert "px_per_degree" in doc
    assert "F3" in doc  # references the in-game yaw readout
