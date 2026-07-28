"""Frame capture for live deployment.

``FrameSource`` is the interface the tick loop consumes: one grayscale
``HxW uint8`` frame per call, shaped ``FRAME_SHAPE[1:]`` = (64, 114).

Implementations:

* ``MockFrameSource``  -- plays back caller-provided frames (tests, dry-run).
* ``QuartzFrameSource`` -- real capture of the Minecraft window via
  ``CGWindowListCreateImage`` (macOS, requires ``pyobjc-framework-Quartz``
  and the Screen Recording permission). Import-guarded: constructing it
  without the dependency raises a ``RuntimeError`` that explains how to
  install it; the rest of this module works everywhere.

CLI::

    python3 -m pvpbot.deploy.capture --check

reports whether Quartz is importable, whether screen-capture permission is
granted, and whether a Minecraft window is currently visible -- without
crashing on machines that have none of that.
"""
import sys
from typing import Iterable, Iterator, Optional, Tuple, Union

import numpy as np

from pvpbot.spec import FRAME_SHAPE

FRAME_H: int = FRAME_SHAPE[1]
FRAME_W: int = FRAME_SHAPE[2]

_PYOBJC_HINT = (
    "Quartz (pyobjc) is required for real screen capture but could not be "
    "imported. Install it with:\n"
    "    python3 -m pip install --user pyobjc-framework-Quartz\n"
    "or, with uv:\n"
    "    uv pip install pyobjc-framework-Quartz\n"
    "Tests and --dry-run mode do not need it (use MockFrameSource)."
)


def _load_quartz():
    """Import and return the Quartz module. Separated out for testability:
    tests monkeypatch this to simulate a missing dependency."""
    import Quartz  # noqa: PLC0415 (deliberate lazy import)

    return Quartz


# ---------------------------------------------------------------------------
# Downscaling
# ---------------------------------------------------------------------------

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """(H, W) passthrough; (H, W, 3|4) -> luma-weighted grayscale float64."""
    img = np.asarray(img)
    if img.ndim == 2:
        return img.astype(np.float64)
    if img.ndim == 3 and img.shape[2] >= 3:
        rgb = img[..., :3].astype(np.float64)
        return rgb @ np.array([0.299, 0.587, 0.114])
    raise ValueError("expected (H,W) or (H,W,3/4) image, got shape %r" % (img.shape,))


def _area_reduce_axis(img: np.ndarray, out_n: int, axis: int) -> np.ndarray:
    """Area-average one axis down to ``out_n`` (or nearest-sample up if the
    source is smaller). Pure numpy, exact block-mean when the ratio is an
    integer."""
    src_n = img.shape[axis]
    img = np.moveaxis(img, axis, 0)
    if src_n >= out_n:
        # bucket index of every source row: surjective onto [0, out_n)
        bucket = np.arange(src_n) * out_n // src_n
        offsets = np.searchsorted(bucket, np.arange(out_n), side="left")
        sums = np.add.reduceat(img, offsets, axis=0)
        counts = np.diff(np.append(offsets, src_n)).astype(np.float64)
        out = sums / counts.reshape((-1,) + (1,) * (img.ndim - 1))
    else:
        idx = np.arange(out_n) * src_n // out_n
        out = img[idx].astype(np.float64)
    return np.moveaxis(out, 0, axis)


def downscale_area(
    img: np.ndarray, out_hw: Tuple[int, int] = (FRAME_H, FRAME_W)
) -> np.ndarray:
    """Downscale to ``out_hw`` with area averaging, preserving color.

    (H, W) grayscale input -> replicated to (out_h, out_w, 3);
    (H, W, 3|4) color input -> (out_h, out_w, 3). Returns uint8.
    FRAME_SHAPE is color: perception's aim precision comes largely from the
    armor's color signature, so capture must not collapse to luma anymore.
    """
    img = np.asarray(img)
    out_h, out_w = out_hw
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    chans = []
    for ci in range(3):
        red = _area_reduce_axis(img[..., ci].astype(np.float64), out_h, axis=0)
        red = _area_reduce_axis(red, out_w, axis=1)
        chans.append(red)
    out = np.stack(chans, axis=2)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# FrameSource interface
# ---------------------------------------------------------------------------

class FrameSourceExhausted(RuntimeError):
    """Raised by finite MockFrameSources when playback runs out."""


class FrameSource:
    """Abstract frame provider for the tick loop."""

    def get_frame(self) -> np.ndarray:
        """Return one grayscale frame, shape (FRAME_H, FRAME_W), dtype uint8."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any OS resources. Safe to call more than once."""

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class MockFrameSource(FrameSource):
    """Plays back provided frames. Accepts:

    * a single (H, W) array           -> repeated forever
    * an (N, H, W) array or list      -> played in order
    * a generator / iterator of (H, W) frames

    Frames not already (FRAME_H, FRAME_W) uint8 are downscaled/converted.
    ``loop=True`` restarts finite sequences instead of raising
    ``FrameSourceExhausted``.
    """

    def __init__(
        self,
        frames: Union[np.ndarray, Iterable[np.ndarray], Iterator[np.ndarray]],
        loop: bool = False,
    ):
        self._loop = loop
        self._closed = False
        self._sequence: Optional[list] = None
        self._iter: Optional[Iterator[np.ndarray]] = None
        self._pos = 0
        if isinstance(frames, np.ndarray):
            if frames.ndim == 2:
                self._sequence = [frames]
                self._loop = True  # a single frame always repeats
            elif frames.ndim == 3:
                self._sequence = [frames[i] for i in range(frames.shape[0])]
            else:
                raise ValueError("frames array must be (H,W) or (N,H,W)")
        elif isinstance(frames, (list, tuple)):
            self._sequence = list(frames)
        else:
            self._iter = iter(frames)
        if self._sequence is not None and not self._sequence:
            raise ValueError("MockFrameSource needs at least one frame")

    @classmethod
    def noise(cls, seed: int = 0) -> "MockFrameSource":
        """Endless uniform-noise frames -- the dry-run default."""
        rng = np.random.default_rng(seed)

        def gen():
            while True:
                yield rng.integers(0, 256, (FRAME_H, FRAME_W), dtype=np.uint8)

        return cls(gen())

    def _conform(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)
        if frame.shape[:2] != (FRAME_H, FRAME_W):
            frame = downscale_area(frame)
        if frame.ndim == 2:  # grayscale -> replicate to the color contract
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        if frame.dtype != np.uint8:
            frame = np.clip(np.rint(frame.astype(np.float64)), 0, 255).astype(np.uint8)
        return frame

    def get_frame(self) -> np.ndarray:
        if self._closed:
            raise FrameSourceExhausted("MockFrameSource is closed")
        if self._sequence is not None:
            if self._pos >= len(self._sequence):
                if not self._loop:
                    raise FrameSourceExhausted(
                        "MockFrameSource: out of frames after %d" % self._pos
                    )
                self._pos = 0
            frame = self._sequence[self._pos]
            self._pos += 1
        else:
            assert self._iter is not None
            try:
                frame = next(self._iter)
            except StopIteration:
                raise FrameSourceExhausted("MockFrameSource generator exhausted")
        return self._conform(frame)

    def close(self) -> None:
        self._closed = True


# ---------------------------------------------------------------------------
# Real capture via Quartz
# ---------------------------------------------------------------------------

class QuartzFrameSource(FrameSource):
    """Captures the Minecraft window via CGWindowListCreateImage.

    Finds the first on-screen window whose title (or owner name) contains
    ``window_title`` case-insensitively, captures it each tick, and
    area-averages down to FRAME_SHAPE. Requires macOS Screen Recording
    permission for the terminal running the bot (see pvpbot/deploy/SETUP.md).
    """

    def __init__(self, window_title: str = "Minecraft"):
        try:
            self._quartz = _load_quartz()
        except ImportError as exc:
            raise RuntimeError(_PYOBJC_HINT) from exc
        self.window_title = window_title
        self._window_id = self._find_window()
        if self._window_id is None:
            raise RuntimeError(
                "No on-screen window matching %r was found. Start Minecraft "
                "1.8.9 in windowed mode first (see pvpbot/deploy/SETUP.md)."
                % window_title
            )

    def _find_window(self) -> Optional[int]:
        q = self._quartz
        wanted = self.window_title.lower()
        infos = q.CGWindowListCopyWindowInfo(
            q.kCGWindowListOptionOnScreenOnly | q.kCGWindowListExcludeDesktopElements,
            q.kCGNullWindowID,
        )
        for info in infos or []:
            title = str(info.get("kCGWindowName") or "")
            owner = str(info.get("kCGWindowOwnerName") or "")
            if wanted in title.lower() or wanted in owner.lower():
                return int(info["kCGWindowNumber"])
        return None

    def _capture_raw(self) -> np.ndarray:
        q = self._quartz
        image = q.CGWindowListCreateImage(
            q.CGRectNull,
            q.kCGWindowListOptionIncludingWindow,
            self._window_id,
            q.kCGWindowImageBoundsIgnoreFraming | q.kCGWindowImageNominalResolution,
        )
        if image is None:
            # Window closed, crashed, or permission revoked mid-run. Do NOT
            # rediscover by title: a substring match can silently latch onto
            # a different window (the "Minecraft Launcher" matched
            # "Minecraft" after a client crash, so the loop kept running and
            # leaked keystrokes into whatever app was frontmost). A vanished
            # window ends the session, full stop.
            raise FrameSourceExhausted(
                "Lost the %r window (game closed or crashed) -- stopping."
                % self.window_title
            )
        w = q.CGImageGetWidth(image)
        h = q.CGImageGetHeight(image)
        bpr = q.CGImageGetBytesPerRow(image)
        data = q.CGDataProviderCopyData(q.CGImageGetDataProvider(image))
        buf = np.frombuffer(data, dtype=np.uint8)
        buf = buf[: h * bpr].reshape(h, bpr)[:, : w * 4].reshape(h, w, 4)
        # CGWindowListCreateImage yields BGRA on little-endian macOS.
        return buf[..., [2, 1, 0]]

    def get_frame(self) -> np.ndarray:
        return downscale_area(self._capture_raw())

    def get_frame_fullres(self) -> np.ndarray:
        """Full-resolution RGB frame (no downscale) for UI-element
        detection -- finding the actual Respawn button beats guessing its
        geometry from GUI-scale/title-bar/retina assumptions (all guessed
        wrong once)."""
        return self._capture_raw()

    def window_owner_pid(self):
        """PID of the process owning the captured window, or None.

        This is the injection target: events posted to this pid (instead
        of the global HID tap) can never land in another application, no
        matter what the focus state claims."""
        q = self._quartz
        try:
            infos = q.CGWindowListCopyWindowInfo(
                q.kCGWindowListOptionIncludingWindow, self._window_id)
            for info in infos or []:
                pid = info.get("kCGWindowOwnerPID")
                if pid:
                    return int(pid)
        except Exception:
            pass
        return None

    def window_bounds(self):
        """(x, y, w, h) of the captured window in global screen points
        (top-left origin, includes the title bar), or None if unavailable.
        Used by the loop to aim deliberate UI clicks (respawn button)."""
        q = self._quartz
        try:
            infos = q.CGWindowListCopyWindowInfo(
                q.kCGWindowListOptionIncludingWindow, self._window_id)
            for info in infos or []:
                b = info.get("kCGWindowBounds")
                if b:
                    return (float(b["X"]), float(b["Y"]),
                            float(b["Width"]), float(b["Height"]))
        except Exception:
            pass
        return None

    def close(self) -> None:
        pass  # CGImages are released by pyobjc refcounting


# ---------------------------------------------------------------------------
# --check CLI
# ---------------------------------------------------------------------------

def check_capture_available(window_title: str = "Minecraft") -> dict:
    """Best-effort environment probe. Never raises."""
    report = {
        "quartz_importable": False,
        "screen_recording_permission": None,  # None = unknown
        "minecraft_window_found": False,
        "detail": "",
    }
    try:
        q = _load_quartz()
    except ImportError:
        report["detail"] = _PYOBJC_HINT
        return report
    except Exception as exc:  # pyobjc present but broken
        report["detail"] = "Quartz import failed: %r" % (exc,)
        return report
    report["quartz_importable"] = True
    try:
        if hasattr(q, "CGPreflightScreenCaptureAccess"):
            report["screen_recording_permission"] = bool(
                q.CGPreflightScreenCaptureAccess()
            )
    except Exception as exc:
        report["detail"] += "permission preflight failed: %r; " % (exc,)
    try:
        infos = q.CGWindowListCopyWindowInfo(
            q.kCGWindowListOptionOnScreenOnly | q.kCGWindowListExcludeDesktopElements,
            q.kCGNullWindowID,
        )
        wanted = window_title.lower()
        for info in infos or []:
            title = str(info.get("kCGWindowName") or "")
            owner = str(info.get("kCGWindowOwnerName") or "")
            if wanted in title.lower() or wanted in owner.lower():
                report["minecraft_window_found"] = True
                break
    except Exception as exc:
        report["detail"] += "window enumeration failed: %r" % (exc,)
    return report


def _main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python3 -m pvpbot.deploy.capture",
        description="Screen-capture environment check.",
    )
    parser.add_argument(
        "--check", action="store_true", help="report capture readiness and exit"
    )
    parser.add_argument("--window-title", default="Minecraft")
    args = parser.parse_args(argv)
    if not args.check:
        parser.print_help()
        return 2
    report = check_capture_available(args.window_title)
    ok = lambda b: {True: "OK", False: "MISSING", None: "UNKNOWN"}[b]  # noqa: E731
    print("capture --check")
    print("  quartz importable ............ %s" % ok(report["quartz_importable"]))
    print("  screen recording permission .. %s" % ok(report["screen_recording_permission"]))
    print("  minecraft window found ....... %s" % ok(report["minecraft_window_found"]))
    if report["detail"]:
        print("  detail: %s" % report["detail"])
    ready = report["quartz_importable"] and report["minecraft_window_found"] and (
        report["screen_recording_permission"] is not False
    )
    print("  => %s" % ("ready for live capture" if ready else "NOT ready (see SETUP.md)"))
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(_main())
