"""Output-layer humanization for live deployment.

Three independent, composable pieces sit between the policy's per-tick action
and the OS input injection layer:

  ReactionDelay -- delays obs->action by ~3 ticks (150 ms at 20 tps) so the
                   bot cannot respond faster than a human physically can.
  MouseSmoother -- expands each per-tick binned camera delta into a smooth
                   sub-tick trajectory of micro-moves (minimum-jerk profile),
                   so the injected mouse stream has no per-tick teleports.
  ClickJitter   -- jitters attack-press timing within the tick with a
                   configurable millisecond sigma; humans don't click on a
                   50 ms grid.

Everything here is pure python/numpy, has no torch dependency, and is
deterministic: the only stochastic piece (ClickJitter) draws from a
``numpy.random.default_rng(seed)`` stream.
"""
from collections import deque
from typing import Deque, List, Optional, Tuple, Union

import numpy as np

from pvpbot.spec import CAMERA_BINS, NUM_ACTION_HEADS, TICK_RATE

ZERO_BIN = CAMERA_BINS.index(0.0)

# forward=none, strafe=none, no jump, no sprint, no attack, zero camera delta
IDLE_ACTION = np.array([1, 1, 0, 0, 0, ZERO_BIN, ZERO_BIN], dtype=np.int64)


class ReactionDelay:
    """FIFO ring buffer delaying the action stream by ``delay_ticks`` ticks.

    ``step(action_t)`` returns ``action_{t - delay_ticks}``; until the buffer
    has filled (the first ``delay_ticks`` calls, and again after ``reset()``)
    it returns copies of ``idle_action``. Default 3 ticks = 150 ms at 20 tps,
    the low end of human visual reaction time.
    """

    def __init__(self, delay_ticks: int = 3, idle_action: Optional[np.ndarray] = None):
        if delay_ticks < 0:
            raise ValueError("delay_ticks must be >= 0")
        self.delay_ticks = int(delay_ticks)
        idle = IDLE_ACTION if idle_action is None else np.asarray(idle_action)
        idle = idle.astype(np.int64).reshape(-1)
        if idle.shape != (NUM_ACTION_HEADS,):
            raise ValueError("idle_action must have %d heads" % NUM_ACTION_HEADS)
        self.idle_action = idle
        self._buf: Deque[np.ndarray] = deque()

    @property
    def delay_seconds(self) -> float:
        return self.delay_ticks / float(TICK_RATE)

    def reset(self) -> None:
        self._buf.clear()

    def step(self, action: np.ndarray) -> np.ndarray:
        """Push this tick's action, pop the action from ``delay_ticks`` ago."""
        a = np.asarray(action).astype(np.int64).reshape(-1)
        if a.shape != (NUM_ACTION_HEADS,):
            raise ValueError("action must have %d heads" % NUM_ACTION_HEADS)
        if self.delay_ticks == 0:
            return a.copy()
        self._buf.append(a.copy())
        if len(self._buf) > self.delay_ticks:
            return self._buf.popleft()
        return self.idle_action.copy()


class MouseSmoother:
    """Expand a per-tick binned camera delta into sub-tick micro-moves.

    A raw policy emits one (yaw_bin, pitch_bin) jump per 50 ms tick; injecting
    that directly produces stepwise, obviously robotic mouse motion. This
    interpolates each tick's total delta along a minimum-jerk position profile

        s(u) = 10 u^3 - 15 u^4 + 6 u^5,   u in [0, 1]

    (zero velocity and acceleration at both ends -- the standard model of
    human point-to-point movement), sampled at ``substeps`` uniform time
    slices. ``smooth()`` returns a list of ``(dt_seconds, dx_degrees,
    dy_degrees)`` micro-moves whose dts are all positive and sum to exactly
    one tick, and whose deltas sum to the bin's full delta. Deterministic;
    no randomness involved.
    """

    def __init__(self, substeps: int = 8, tick_rate: int = TICK_RATE):
        if substeps < 1:
            raise ValueError("substeps must be >= 1")
        if tick_rate <= 0:
            raise ValueError("tick_rate must be > 0")
        self.substeps = int(substeps)
        self.tick_seconds = 1.0 / float(tick_rate)
        u = np.linspace(0.0, 1.0, self.substeps + 1)
        s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        self._fractions = np.diff(s)  # sums to exactly 1.0 (s(0)=0, s(1)=1)

    def smooth_degrees(
        self, dx_degrees: float, dy_degrees: float
    ) -> List[Tuple[float, float, float]]:
        """Continuous per-tick delta -> [(dt_s, dx_deg, dy_deg), ...]."""
        dt = self.tick_seconds / self.substeps
        dx = float(dx_degrees)
        dy = float(dy_degrees)
        return [(dt, dx * f, dy * f) for f in self._fractions]

    def smooth(
        self, yaw_idx: Union[int, np.integer], pitch_idx: Union[int, np.integer]
    ) -> List[Tuple[float, float, float]]:
        """CAMERA_BINS indices -> [(dt_s, dx_deg, dy_deg), ...] micro-moves."""
        n = len(CAMERA_BINS)
        yi, pi = int(yaw_idx), int(pitch_idx)
        if not (0 <= yi < n and 0 <= pi < n):
            raise ValueError("camera bin index out of range [0, %d)" % n)
        return self.smooth_degrees(CAMERA_BINS[yi], CAMERA_BINS[pi])


class ClickJitter:
    """Human-like timing jitter for attack presses.

    ``offset(attack)`` returns None when ``attack`` is 0; otherwise a delay in
    seconds from the start of the tick at which to fire the click, drawn as
    ``|N(0, sigma_ms)|`` (half-normal -- a human is late, never early,
    relative to the decision instant) and clipped to stay inside the tick.
    Deterministic under ``seed``: the same seed yields the same offset
    sequence. ``sigma_ms=0`` gives exactly 0.0 for every press.
    """

    def __init__(
        self, sigma_ms: float = 12.0, seed: int = 0, tick_rate: int = TICK_RATE
    ):
        if sigma_ms < 0:
            raise ValueError("sigma_ms must be >= 0")
        self.sigma_ms = float(sigma_ms)
        self.tick_seconds = 1.0 / float(tick_rate)
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)

    def offset(self, attack: Union[int, np.integer]) -> Optional[float]:
        """Seconds into the tick to fire this press, or None if no press."""
        if int(attack) == 0:
            return None
        if self.sigma_ms == 0.0:
            return 0.0
        raw = abs(float(self._rng.normal(0.0, self.sigma_ms))) / 1000.0
        # keep the press inside its own tick
        return min(raw, self.tick_seconds * 0.95)
