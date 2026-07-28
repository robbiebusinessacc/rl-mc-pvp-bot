"""Input injection for live deployment, plus degrees<->pixels calibration.

``InputSink`` is the interface the tick loop drives:

* ``move_mouse(dx_px, dy_px)``  -- relative mouse motion in pixels
* ``key(name, down)``           -- press/release a named key ("w", "space", ...)
* ``click(down)``               -- left mouse button press/release

Implementations: ``MockInputSink`` (records every call with a timestamp;
used by tests and dry-run) and ``QuartzInputSink`` (CGEventPost; requires
``pyobjc-framework-Quartz`` and the Accessibility permission -- import
guarded, tests never need it).

Manual calibration procedure (degrees -> pixels)
================================================

Minecraft converts mouse pixels to camera degrees through its Sensitivity
slider. The policy outputs camera deltas in *degrees per tick*
(``CAMERA_BINS`` in ``pvpbot/spec.py``); we must know how many pixels of
mouse motion produce one degree of yaw. Measure it once per
sensitivity/DPI setup:

1. In Minecraft 1.8.9: Options -> Controls -> set **Raw Input: OFF** (if on
   a fork that has it) and note the Sensitivity percentage. Do not change
   either afterwards -- the calibration is only valid for those settings.
2. Join an empty world/arena. Press F3 so the debug overlay shows your
   facing (the "Facing" line includes yaw in degrees).
3. Note the current yaw, e.g. ``yaw0 = -87.3``.
4. Run ``python3 -m pvpbot.deploy.run --nudge 1000`` (or use any tool that
   injects an exact relative mouse move of ``N = 1000`` pixels horizontally),
   with the Minecraft window focused.
5. Read the new yaw ``yaw1`` from F3. The horizontal calibration is::

       px_per_degree_x = N / abs(yaw1 - yaw0)

   (Unwrap manually if you crossed the +/-180 boundary.)
6. Repeat with a vertical nudge against the pitch readout for
   ``px_per_degree_y``. In vanilla 1.8.9 pitch and yaw share the same
   scale, so ``px_per_degree_y == px_per_degree_x`` is expected; measuring
   both is a sanity check.
7. Verify: at Minecraft's default 100%% sensitivity the theoretical value is
   ``1 degree = 1 / (0.6666 * 0.6 + 0.2)^3 / 0.15`` ~= 12.9 px... in
   practice just trust your measurement: a correct calibration makes a
   commanded 30-degree flick land within ~1 degree on the F3 readout.
8. Pass the numbers to the CLI: ``--px-per-degree 12.9`` (or
   ``--px-per-degree-x/-y`` separately) or construct
   ``Calibration(px_per_degree_x=..., px_per_degree_y=...)`` in code.

``python3 -m pvpbot.deploy.run --calibrate`` prints this procedure.
"""
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from pvpbot.spec import CAMERA_BINS

_PYOBJC_HINT = (
    "Quartz (pyobjc) is required for real input injection but could not be "
    "imported. Install it with:\n"
    "    python3 -m pip install --user pyobjc-framework-Quartz\n"
    "or, with uv:\n"
    "    uv pip install pyobjc-framework-Quartz\n"
    "Tests and --dry-run mode do not need it (use MockInputSink)."
)


def _load_quartz():
    """Import and return the Quartz module. Tests monkeypatch this."""
    import Quartz  # noqa: PLC0415 (deliberate lazy import)

    return Quartz


# ---------------------------------------------------------------------------
# Calibration: degrees <-> pixels
# ---------------------------------------------------------------------------

@dataclass
class Calibration:
    """Pixels of relative mouse motion per degree of camera rotation.

    Measured manually per sensitivity/DPI setup -- see the module docstring
    for the exact procedure. The default is the common value for Minecraft
    1.8.9 at 100% sensitivity; treat it as a placeholder until measured.
    """

    px_per_degree_x: float = 12.9
    px_per_degree_y: float = 12.9

    def __post_init__(self):
        if self.px_per_degree_x <= 0 or self.px_per_degree_y <= 0:
            raise ValueError("px_per_degree must be positive")

    def degrees_to_pixels(self, yaw_deg: float, pitch_deg: float) -> Tuple[float, float]:
        """Camera delta in degrees -> (dx_px, dy_px) mouse delta.

        Positive yaw turns right -> positive dx. Positive pitch looks down
        in Minecraft's convention -> positive dy (screen-down).
        """
        return yaw_deg * self.px_per_degree_x, pitch_deg * self.px_per_degree_y

    def pixels_to_degrees(self, dx_px: float, dy_px: float) -> Tuple[float, float]:
        """Inverse of :meth:`degrees_to_pixels`."""
        return dx_px / self.px_per_degree_x, dy_px / self.px_per_degree_y

    def bins_to_pixels(self, yaw_bin: int, pitch_bin: int) -> Tuple[float, float]:
        """CAMERA_BINS indices (policy heads 5 and 6) -> pixel deltas."""
        return self.degrees_to_pixels(CAMERA_BINS[yaw_bin], CAMERA_BINS[pitch_bin])


class PixelAccumulator:
    """Carries the fractional pixel remainder across ticks.

    Mouse events take integer pixels; naively rounding each tick loses up
    to 0.5 px/tick of aim (10 deg/s of drift at ~1 px/deg error rates over
    a fight). Accumulate the exact command, emit integers, keep the
    remainder.
    """

    def __init__(self) -> None:
        self._rx = 0.0
        self._ry = 0.0

    def step(self, dx_px: float, dy_px: float) -> Tuple[int, int]:
        self._rx += dx_px
        self._ry += dy_px
        ox = int(round(self._rx))
        oy = int(round(self._ry))
        self._rx -= ox
        self._ry -= oy
        return ox, oy

    def reset(self) -> None:
        self._rx = self._ry = 0.0


# ---------------------------------------------------------------------------
# InputSink interface
# ---------------------------------------------------------------------------

class InputSink:
    """Abstract input backend the tick loop drives."""

    def move_mouse(self, dx_px: int, dy_px: int) -> None:
        raise NotImplementedError

    def key(self, name: str, down: bool) -> None:
        raise NotImplementedError

    def click(self, down: bool) -> None:
        raise NotImplementedError

    def release_all(self) -> None:
        """Release every key/button this sink may have left held."""

    def cursor_visible(self) -> bool:
        """True when the OS cursor is visible, i.e. the game does NOT have
        mouse grab (death screen, Esc menu, chat, inventory). Relative-delta
        injection is unsafe then: deltas move the real cursor and clicks
        land wherever it happens to sit -- including outside the game
        window (observed live: death screen freed the cursor and inputs
        leaked into other apps). Mock sinks report False (grabbed)."""
        return False

    def close(self) -> None:
        self.release_all()


@dataclass
class SinkEvent:
    t: float
    kind: str  # "move" | "key" | "click"
    data: tuple


class MockInputSink(InputSink):
    """Records every call with a timestamp. ``time_fn`` is injectable so
    tests can use a fake clock."""

    def __init__(self, time_fn: Callable[[], float] = time.monotonic):
        self._time_fn = time_fn
        self.events: List[SinkEvent] = []
        self.held_keys: set = set()
        self.mouse_down = False

    def move_mouse(self, dx_px: int, dy_px: int) -> None:
        self.events.append(SinkEvent(self._time_fn(), "move", (dx_px, dy_px)))

    def key(self, name: str, down: bool) -> None:
        self.events.append(SinkEvent(self._time_fn(), "key", (name, down)))
        (self.held_keys.add if down else self.held_keys.discard)(name)

    def click(self, down: bool) -> None:
        self.events.append(SinkEvent(self._time_fn(), "click", (down,)))
        self.mouse_down = down

    def release_all(self) -> None:
        for name in sorted(self.held_keys):
            self.key(name, False)
        if self.mouse_down:
            self.click(False)

    # -- test conveniences ---------------------------------------------------
    def events_of(self, kind: str) -> List[SinkEvent]:
        return [e for e in self.events if e.kind == kind]

    def total_mouse_delta(self) -> Tuple[int, int]:
        dx = sum(e.data[0] for e in self.events if e.kind == "move")
        dy = sum(e.data[1] for e in self.events if e.kind == "move")
        return dx, dy


# ---------------------------------------------------------------------------
# Real injection via Quartz
# ---------------------------------------------------------------------------

# ANSI (US layout) virtual keycodes for the keys the bot uses.
# Sprint is X, NOT Left Ctrl: macOS synthesizes Ctrl+left-click into a
# RIGHT click, so holding Ctrl-sprint turned every attack into a 1.8 sword
# block. Rebind Sprint to X in Minecraft (Options -> Controls) to match.
MAC_KEYCODES = {
    "w": 13, "a": 0, "s": 1, "d": 2,
    "space": 49, "shift": 56, "ctrl": 59, "sprint": 7,  # sprint == X
    "x": 7, "esc": 53, "f3": 99, "e": 14, "q": 12,
}


class QuartzInputSink(InputSink):
    """Posts input events to the game.

    Keyboard and click events are posted DIRECTLY to the game's pid with
    CGEventPostToPid: addressed events cannot land in any other app, no
    matter what focus claims (NSWorkspace's frontmost answer is stale for
    a few hundred ms around an app switch -- observed live: movement keys
    typed into the user's terminal while they were writing "stop"). Only
    relative mouse motion still goes through the global HID tap, because
    the grabbed-cursor delta path needs it -- and a stray mouse delta
    cannot type or click anything.

    Relative mouse motion is delivered as a kCGEventMouseMoved event whose
    integer delta fields are set explicitly -- Minecraft 1.8.9 (raw input
    off) reads the OS cursor deltas, so the cursor is also warped by the
    delta. Requires the Accessibility permission (see SETUP.md).
    """

    def __init__(self, target_pid: Optional[int] = None) -> None:
        try:
            self._q = _load_quartz()
        except ImportError as exc:
            raise RuntimeError(_PYOBJC_HINT) from exc
        self._held: set = set()
        self._mouse_down = False
        self._pid = int(target_pid) if target_pid else None
        if self._pid is not None and not hasattr(self._q, "CGEventPostToPid"):
            raise RuntimeError(
                "CGEventPostToPid unavailable in this Quartz build; refusing "
                "to fall back to global keyboard posting (leak risk)."
            )

    def _post(self, event) -> None:
        self._q.CGEventPost(self._q.kCGHIDEventTap, event)

    def _post_targeted(self, event) -> None:
        """Post to the game's pid when known; global tap otherwise."""
        if self._pid is not None:
            self._q.CGEventPostToPid(self._pid, event)
        else:
            self._post(event)

    def _cursor(self) -> Tuple[float, float]:
        ev = self._q.CGEventCreate(None)
        loc = self._q.CGEventGetLocation(ev)
        return loc.x, loc.y

    def move_mouse(self, dx_px: int, dy_px: int) -> None:
        # Post AT the current cursor position with only the delta fields set,
        # from an explicit HID-state source. A pointer-grabbed LWJGL2 game
        # (Minecraft 1.8.9) reads exactly these deltas; posting at position
        # + delta instead makes the WindowServer synthesize an implicit
        # second delta from the position jump (verified live: camera drifted
        # vertically on a pure-horizontal nudge until this was fixed).
        q = self._q
        if getattr(self, "_src", None) is None:
            self._src = q.CGEventSourceCreate(q.kCGEventSourceStateHIDSystemState)
        pos = self._cursor()
        ev = q.CGEventCreateMouseEvent(
            self._src, q.kCGEventMouseMoved, pos, q.kCGMouseButtonLeft
        )
        q.CGEventSetIntegerValueField(ev, q.kCGMouseEventDeltaX, int(dx_px))
        q.CGEventSetIntegerValueField(ev, q.kCGMouseEventDeltaY, int(dy_px))
        self._post(ev)

    def key(self, name: str, down: bool) -> None:
        code = MAC_KEYCODES.get(name.lower())
        if code is None:
            raise ValueError("no macOS keycode mapped for key %r" % name)
        self._post_targeted(self._q.CGEventCreateKeyboardEvent(None, code, down))
        (self._held.add if down else self._held.discard)(name)

    def click(self, down: bool) -> None:
        # HID tap, not pid: LWJGL2 ignores pid-addressed mouse events
        # (verified live -- 50 pid-posted clicks on the respawn button did
        # nothing, while pid-posted Esc worked). Mouse events cannot type
        # text, and the loop's frontmost + cursor gates bound the risk.
        q = self._q
        pos = self._cursor()
        kind = q.kCGEventLeftMouseDown if down else q.kCGEventLeftMouseUp
        self._post(q.CGEventCreateMouseEvent(None, kind, pos, q.kCGMouseButtonLeft))
        self._mouse_down = down

    def cursor_visible(self) -> bool:
        # Fail toward "visible" (= unsafe, no injection) on any doubt.
        try:
            return bool(self._q.CGCursorIsVisible())
        except Exception:
            return True

    def move_cursor_to(self, x: float, y: float) -> None:
        """Warp + real HID MOVE event: parks the UI cursor somewhere
        neutral so no button is hovered (a hovered 1.8 button tints
        bluish and vanishes from the gray-bar detector)."""
        q = self._q
        pt = (x, y)
        q.CGWarpMouseCursorPosition(pt)
        self._post(q.CGEventCreateMouseEvent(
            None, q.kCGEventMouseMoved, pt, q.kCGMouseButtonLeft))

    def click_at(self, x: float, y: float) -> None:
        """One deliberate absolute-position click for UI buttons (respawn).

        Only meaningful in cursor-VISIBLE states; gameplay clicks go through
        :meth:`click` at the grabbed cursor. Coordinates are global screen
        points (top-left origin, same space as kCGWindowBounds)."""
        # HID tap for the whole sequence (see click()); a real MOVE event
        # first, because LWJGL2/AWT track their own cursor from move
        # events -- warping alone leaves the game's UI cursor stale.
        q = self._q
        pt = (x, y)
        q.CGWarpMouseCursorPosition(pt)
        self._post(q.CGEventCreateMouseEvent(
            None, q.kCGEventMouseMoved, pt, q.kCGMouseButtonLeft))
        time.sleep(0.03)
        down = q.CGEventCreateMouseEvent(
            None, q.kCGEventLeftMouseDown, pt, q.kCGMouseButtonLeft)
        q.CGEventSetIntegerValueField(down, q.kCGMouseEventClickState, 1)
        self._post(down)
        time.sleep(0.05)
        up = q.CGEventCreateMouseEvent(
            None, q.kCGEventLeftMouseUp, pt, q.kCGMouseButtonLeft)
        q.CGEventSetIntegerValueField(up, q.kCGMouseEventClickState, 1)
        self._post(up)

    def release_all(self) -> None:
        for name in sorted(self._held):
            try:
                self.key(name, False)
            except Exception:
                pass
        if self._mouse_down:
            try:
                self.click(False)
            except Exception:
                pass


def check_input_available() -> dict:
    """Best-effort probe of injection readiness. Never raises."""
    report = {"quartz_importable": False, "accessibility_permission": None, "detail": ""}
    try:
        q = _load_quartz()
    except ImportError:
        report["detail"] = _PYOBJC_HINT
        return report
    except Exception as exc:
        report["detail"] = "Quartz import failed: %r" % (exc,)
        return report
    report["quartz_importable"] = True
    try:
        if hasattr(q, "CGPreflightPostEventAccess"):
            report["accessibility_permission"] = bool(q.CGPreflightPostEventAccess())
        elif hasattr(q, "AXIsProcessTrusted"):
            report["accessibility_permission"] = bool(q.AXIsProcessTrusted())
    except Exception as exc:
        report["detail"] = "accessibility preflight failed: %r" % (exc,)
    return report
