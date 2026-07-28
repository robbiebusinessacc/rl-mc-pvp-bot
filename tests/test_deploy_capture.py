"""Deploy module: frame capture — downscaling, mock playback, import guards."""
import numpy as np
import pytest

from pvpbot.deploy import capture
from pvpbot.deploy.capture import (
    FRAME_H,
    FRAME_W,
    FrameSourceExhausted,
    MockFrameSource,
    check_capture_available,
    downscale_area,
)
from pvpbot.spec import FRAME_SHAPE


# ---------------------------------------------------------------------------
# downscale_area
# ---------------------------------------------------------------------------

def test_downscale_output_contract():
    img = np.random.default_rng(0).integers(0, 256, (720, 1280), dtype=np.uint8)
    out = downscale_area(img)
    assert out.shape == (FRAME_H, FRAME_W, 3)
    assert out.dtype == np.uint8


def test_downscale_constant_image_is_exact():
    for value in (0, 97, 255):
        img = np.full((640, 1140), value, dtype=np.uint8)
        out = downscale_area(img)
        assert (out == value).all()


def test_downscale_integer_ratio_is_exact_block_mean():
    # 2x in both axes: each output pixel is the mean of a 2x2 block.
    rng = np.random.default_rng(1)
    blocks = rng.integers(0, 256, (FRAME_H, FRAME_W), dtype=np.uint8)
    img = np.repeat(np.repeat(blocks, 2, axis=0), 2, axis=1)
    out = downscale_area(img)
    np.testing.assert_array_equal(out[..., 0], blocks)

    # 2x2 checkerboard of 0/255 collapses to the exact block mean (127.5 -> 128)
    cb = np.zeros((2 * FRAME_H, 2 * FRAME_W), dtype=np.uint8)
    cb[::2, ::2] = 255
    cb[1::2, 1::2] = 255
    out = downscale_area(cb)
    assert (out == 128).all()


def test_downscale_10x_block_mean():
    rng = np.random.default_rng(2)
    blocks = rng.integers(0, 256, (FRAME_H, FRAME_W)).astype(np.float64)
    img = np.repeat(np.repeat(blocks, 10, axis=0), 10, axis=1)
    out = downscale_area(img)
    np.testing.assert_allclose(out[..., 0], np.rint(blocks), atol=1)


def test_downscale_non_integer_ratio_preserves_gradient():
    # 719x1279 -> non-integer ratio in both axes.
    col = np.linspace(0, 255, 719)[:, None]
    img = np.repeat(col, 1279, axis=1).astype(np.uint8)
    out = downscale_area(img)[..., 0]
    assert out.shape == (FRAME_H, FRAME_W)
    # rows must be non-decreasing top->bottom and span most of the range
    rowmeans = out.astype(np.float64).mean(axis=1)
    assert (np.diff(rowmeans) >= 0).all()
    assert rowmeans[0] < 10 and rowmeans[-1] > 245
    # columns constant
    assert (out == out[:, :1]).all()


def test_downscale_mean_is_preserved():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (720, 1280), dtype=np.uint8)
    out = downscale_area(img)
    assert abs(out.mean() - img.mean()) < 1.5


def test_downscale_rgb_input_preserves_color():
    img = np.zeros((128, 228, 3), dtype=np.uint8)
    img[..., 0] = 255  # pure red must STAY red (color is the signal now)
    out = downscale_area(img)
    assert out.shape == (FRAME_H, FRAME_W, 3)
    assert (out[..., 0] == 255).all() and (out[..., 1] == 0).all()


def test_downscale_upscales_small_input():
    img = np.full((32, 57), 42, dtype=np.uint8)  # smaller than target
    out = downscale_area(img)
    assert out.shape == (FRAME_H, FRAME_W, 3)
    assert (out == 42).all()


# ---------------------------------------------------------------------------
# MockFrameSource
# ---------------------------------------------------------------------------

def test_mock_single_frame_repeats():
    frame = np.full((FRAME_H, FRAME_W), 7, dtype=np.uint8)
    src = MockFrameSource(frame)
    for _ in range(5):
        out = src.get_frame()
        assert out.shape == (FRAME_H, FRAME_W, 3) and out.dtype == np.uint8
        assert (out == 7).all()
    src.close()


def test_mock_sequence_plays_in_order_then_exhausts():
    frames = np.stack(
        [np.full((FRAME_H, FRAME_W), i, dtype=np.uint8) for i in range(3)]
    )
    src = MockFrameSource(frames)
    assert [int(src.get_frame()[0, 0, 0]) for _ in range(3)] == [0, 1, 2]
    with pytest.raises(FrameSourceExhausted):
        src.get_frame()


def test_mock_sequence_loops_when_requested():
    frames = [np.full((FRAME_H, FRAME_W), i, dtype=np.uint8) for i in range(2)]
    src = MockFrameSource(frames, loop=True)
    assert [int(src.get_frame()[0, 0, 0]) for _ in range(5)] == [0, 1, 0, 1, 0]


def test_mock_generator_and_conforming():
    def gen():
        yield np.zeros((720, 1280), dtype=np.uint8)  # wrong size -> downscaled
        yield np.full((FRAME_H, FRAME_W), 300.0)     # wrong dtype -> clipped u8

    src = MockFrameSource(gen())
    a = src.get_frame()
    assert a.shape == (FRAME_H, FRAME_W, 3) and a.dtype == np.uint8 and (a == 0).all()
    b = src.get_frame()
    assert b.dtype == np.uint8 and (b == 255).all()
    with pytest.raises(FrameSourceExhausted):
        src.get_frame()


def test_mock_noise_source_endless():
    src = MockFrameSource.noise(seed=0)
    frames = [src.get_frame() for _ in range(3)]
    assert all(f.shape == (FRAME_H, FRAME_W, 3) for f in frames)
    assert not (frames[0] == frames[1]).all()  # actually noise


def test_mock_closed_raises():
    src = MockFrameSource(np.zeros((FRAME_H, FRAME_W), dtype=np.uint8))
    src.close()
    with pytest.raises(FrameSourceExhausted):
        src.get_frame()


# ---------------------------------------------------------------------------
# Quartz guards (no real Quartz used; loader is monkeypatched)
# ---------------------------------------------------------------------------

def _raise_import_error():
    raise ImportError("No module named 'Quartz'")


def test_quartz_frame_source_guard_message(monkeypatch):
    monkeypatch.setattr(capture, "_load_quartz", _raise_import_error)
    with pytest.raises(RuntimeError) as ei:
        capture.QuartzFrameSource()
    msg = str(ei.value)
    assert "pyobjc-framework-Quartz" in msg
    assert "pip install" in msg
    assert "MockFrameSource" in msg  # points at the offline alternative


def test_check_capture_available_without_quartz(monkeypatch):
    monkeypatch.setattr(capture, "_load_quartz", _raise_import_error)
    report = check_capture_available()
    assert report["quartz_importable"] is False
    assert report["minecraft_window_found"] is False
    assert "pyobjc" in report["detail"]


def test_check_cli_never_crashes_without_quartz(monkeypatch, capsys):
    monkeypatch.setattr(capture, "_load_quartz", _raise_import_error)
    rc = capture._main(["--check"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "NOT ready" in out
