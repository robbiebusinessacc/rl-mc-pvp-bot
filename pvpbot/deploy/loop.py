"""Real-time tick loop: frame -> perception -> obs -> policy -> humanize -> input.

Runs at ``TICK_RATE`` Hz (50 ms budget). Per tick:

1. capture   -- ``FrameSource.get_frame()``
2. encode    -- perception CNN estimate + observation assembly (48 floats)
3. policy    -- ``PolicyNet.act`` (GRU state carried across ticks)
4. humanize  -- optional BC humanization pass over the raw action
5. inject    -- action -> ``InputSink`` (keys / clicks / calibrated mouse)

Perception (``pvpbot.perception.infer`` / ``pvpbot.perception.adapter``) and
humanization (``pvpbot.bc.humanize``) are developed on parallel branches, so
they are imported **lazily** with graceful fallbacks: without them the loop
still runs end-to-end in dry-run mode using a zero-perception encoder, a
built-in obs assembler, and identity humanization.

Safety / observability:

* **Kill-switch**: the loop stops if the sentinel file ``/tmp/pvpbot-stop``
  exists (checked every tick; path configurable) or if an injected
  ``stop_check`` callback (e.g. a hotkey listener) returns True. On stop the
  sink releases all held keys/buttons.
* **Match state machine**: ``idle`` (neutral inputs only) -> ``fighting``
  (policy in control) -> ``stopped``.
* **Flight recorder**: one JSONL line per tick (state, obs, perception,
  action, per-stage latencies) for post-match analysis and future BC data.
* **Latency**: per-stage timings are collected each tick; a warning is
  logged whenever a tick exceeds the 50 ms budget.

Time is injectable (``clock``/``sleep``) so tests run instantly on a fake
clock.
"""
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from pvpbot.spec import (
    ACTION_HEAD_SIZES,
    CAMERA_BINS,
    FRAME_SHAPE,
    NUM_ACTION_HEADS,
    OBS_DIM,
    OBS_LAYOUT,
    PERCEPTION_DIM,
    PERCEPTION_LAYOUT,
    TICK_RATE,
)
from pvpbot.deploy.capture import FrameSource, FrameSourceExhausted, downscale_area
from pvpbot.deploy.input_inject import Calibration, InputSink, PixelAccumulator

log = logging.getLogger(__name__)

TICK_PERIOD = 1.0 / TICK_RATE          # 0.05 s
TICK_BUDGET_MS = 1000.0 / TICK_RATE    # 50 ms
KILL_SWITCH_PATH = "/tmp/pvpbot-stop"
# Authoritative death cue: match orchestration (watching the server log)
# touches this file when our player dies; the loop then clicks Respawn on
# cursor-free ticks and removes the file once the mouse grab is stably
# back. The frame classifier is only a fallback cue -- the real 1.8 death
# tint is too weak against a bright sky to detect reliably.
DEATH_FLAG_PATH = "/tmp/pvpbot-dead"
# CGCursorIsVisible flaps on the death screen (observed live: 1-5 tick
# alternations), so injection only resumes after the cursor has read
# "grabbed" this many consecutive ticks.
GRAB_SETTLE_TICKS = 10
STAGES = ("capture", "encode", "policy", "inject")
NEUTRAL_ACTION = np.array(
    [1, 1, 0, 0, 0, CAMERA_BINS.index(0.0), CAMERA_BINS.index(0.0)], dtype=np.int64
)
_BINS = np.asarray(CAMERA_BINS, dtype=np.float32)


class _MoveLatch:
    """Live mirror of SimConfig.act_hold_ticks (heads: forward/strafe/sprint).

    A changed movement bin takes effect only after persisting ``hold`` ticks.
    Without this, flickered per-tick intents reach the adapter's dead-reckoner
    (which then reports fictitious self-motion the client never performed)
    while the real keys produce nothing -- the policy ends up acting inside
    the PWM fantasy the sim latch was built to forbid. Filtering the action
    ONCE here keeps keys, dead-reckoning, and the flight log all consistent
    with the action semantics the policy was trained under.
    """

    HEADS = (0, 1, 3)

    def __init__(self, hold: int = 3):
        self.hold = int(hold)
        self.eff = {0: 1, 1: 1, 3: 0}
        self.pend = dict(self.eff)
        self.cnt = {h: 0 for h in self.HEADS}

    def filter(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action).copy()
        for h in self.HEADS:
            raw = int(a[h])
            self.cnt[h] = self.cnt[h] + 1 if raw == self.pend[h] else 1
            self.pend[h] = raw
            if self.cnt[h] >= self.hold:
                self.eff[h] = raw
            a[h] = self.eff[h]
        return a


def _nearest_bin(err_deg: float) -> int:
    return int(np.abs(_BINS - err_deg).argmin())


class MatchState(Enum):
    IDLE = "idle"
    FIGHTING = "fighting"
    STOPPED = "stopped"


# ---------------------------------------------------------------------------
# Lazy perception / humanization with fallbacks
# ---------------------------------------------------------------------------

def _fallback_encode(frame: np.ndarray) -> np.ndarray:
    """Zero perception estimate (enemy not visible). Keeps the loop able to
    run before the perception branch merges."""
    return np.zeros(PERCEPTION_DIM, dtype=np.float32)


def _fallback_assemble(
    percep: np.ndarray, prev_action: np.ndarray
) -> np.ndarray:
    """Minimal perception -> observation mapping per OBS_LAYOUT.

    Fills the slots perception can inform (aim errors, distance from bbox
    height, hp, on-ground, pitch) plus prev_action; leaves the rest zero.
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    def put(name: str, value) -> None:
        s, e = OBS_LAYOUT[name]
        obs[s:e] = value

    put("aim_err_yaw", percep[PERCEPTION_LAYOUT["aim_err_yaw"]])
    put("aim_err_pitch", percep[PERCEPTION_LAYOUT["aim_err_pitch"]])
    pitch_rad = float(percep[PERCEPTION_LAYOUT["self_pitch"]]) * np.deg2rad(90.0)
    put("self_pitch_sincos", [np.sin(pitch_rad), np.cos(pitch_rad)])
    put("self_hp", percep[PERCEPTION_LAYOUT["self_hp"]])
    put("enemy_hurt", percep[PERCEPTION_LAYOUT["hurt_flash"]])
    put("enemy_on_ground", percep[PERCEPTION_LAYOUT["enemy_on_ground"]])
    visible = float(percep[PERCEPTION_LAYOUT["visible"]])
    bbox_h = float(percep[PERCEPTION_LAYOUT["bbox_height"]])
    if visible > 0.5 and bbox_h > 1e-3:
        # A 1.8-block player subtends roughly (1.8 / dist) of a ~1.2 rad
        # vertical FOV -> dist ~ 1.5 / bbox_height. Crude; the real
        # adapter replaces this.
        dist = min(1.5 / bbox_h, 16.0)
        put("dist", dist / 8.0)
        put("in_reach", 1.0 if dist < 3.0 else 0.0)
    else:
        put("dist", 16.0 / 8.0)
    put(
        "prev_action",
        prev_action.astype(np.float32) / np.asarray(ACTION_HEAD_SIZES, np.float32),
    )
    return obs


def _resolve_perception() -> Dict[str, Callable]:
    """Try the real perception modules; fall back to built-ins.

    Returns {"encode": frame->(PERCEPTION_DIM,), "assemble":
    (percep, prev_action)->(OBS_DIM,), "source": str}.
    """
    encode = None
    assemble = None
    source = "fallback"
    try:
        from pvpbot.perception import infer as p_infer  # noqa: PLC0415

        for name in ("encode_frame", "encode", "infer_frame", "infer"):
            cand = getattr(p_infer, name, None)
            if callable(cand):
                encode = cand
                break
    except Exception as exc:  # ImportError or anything broken mid-development
        log.info("perception.infer unavailable (%s); using zero encoder", exc)
    try:
        from pvpbot.perception import adapter as p_adapter  # noqa: PLC0415

        for name in ("assemble_obs", "assemble", "adapt"):
            cand = getattr(p_adapter, name, None)
            if callable(cand):
                assemble = cand
                break
    except Exception as exc:
        log.info("perception.adapter unavailable (%s); using builtin assembler", exc)
    if encode is not None and assemble is not None:
        source = "pvpbot.perception"
    return {
        "encode": encode or _fallback_encode,
        "assemble": assemble or _fallback_assemble,
        "source": source,
    }


def _resolve_humanizer() -> Dict[str, object]:
    """Try pvpbot.bc.humanize; fall back to identity."""
    try:
        from pvpbot.bc import humanize as bc_humanize  # noqa: PLC0415

        for name in ("humanize_action", "humanize", "apply"):
            cand = getattr(bc_humanize, name, None)
            if callable(cand):
                return {"fn": cand, "source": "pvpbot.bc.humanize"}
    except Exception as exc:
        log.info("bc.humanize unavailable (%s); actions pass through raw", exc)
    return {"fn": lambda action, **_kw: action, "source": "identity"}


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------

class TorchPolicy:
    """Wraps the shared PolicyNet with its GRU state for single-stream use.

    Satisfies the loop's policy protocol: ``__call__(obs_f32[48]) ->
    int64[NUM_ACTION_HEADS]``.
    """

    def __init__(self, checkpoint: Optional[str] = None, deterministic: bool = False):
        import torch  # noqa: PLC0415 (keep module import light)

        from pvpbot.models import PolicyNet  # noqa: PLC0415

        self._torch = torch
        self.net = PolicyNet()
        self.meta: Dict = {}
        if checkpoint:
            blob = torch.load(checkpoint, map_location="cpu")
            self.net.load_state_dict(blob["model"])
            self.meta = dict(blob.get("meta", {}))
            if self.meta.get("obs_dim") not in (None, OBS_DIM):
                raise ValueError(
                    "checkpoint obs_dim=%r != spec OBS_DIM=%d"
                    % (self.meta.get("obs_dim"), OBS_DIM)
                )
        self.net.eval()
        # apply the training-time obs normalization when the checkpoint
        # carries it (same fix as eval/arena.py: raw obs silently cripple
        # policies whose RunningNorm is far from identity)
        self.norm = None
        if checkpoint and "obs_norm" in blob:
            from pvpbot.train.ppo import RunningNorm

            self.norm = RunningNorm(OBS_DIM)
            self.norm.load_state_dict(blob["obs_norm"])
        self.deterministic = deterministic
        self.state = self.net.initial_state(1)

    def reset(self) -> None:
        self.state = self.net.initial_state(1)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        torch = self._torch
        obs = np.ascontiguousarray(obs, dtype=np.float32)
        if self.norm is not None:
            obs = self.norm.normalize(obs[None])[0].astype(np.float32)
        obs_t = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32)).unsqueeze(0)
        actions, self.state = self.net.act(
            obs_t, self.state, deterministic=self.deterministic
        )
        return actions.squeeze(0).numpy().astype(np.int64)


# ---------------------------------------------------------------------------
# Action -> InputSink
# ---------------------------------------------------------------------------

class ActionApplier:
    """Translates a 7-head action into key transitions, clicks, and
    calibrated mouse motion. Tracks held-key state so only edges are sent."""

    PITCH_LIMIT_DEG = 55.0  # sim duels never need more; prevents sky-lock

    # Injection is only ever allowed while the game itself is the frontmost
    # app. After a client crash the frontmost app is a dialog, the launcher,
    # or whatever the user was working in -- keystrokes must never go there
    # (observed once: inputs leaked into the user's browser; never again).
    FRONTMOST_OK = ("java", "minecraft")

    def __init__(self, sink: InputSink, calibration: Optional[Calibration] = None):
        self.sink = sink
        self.calibration = calibration or Calibration()
        self.accum = PixelAccumulator()
        self._down: Dict[str, bool] = {}
        # integrated commanded pitch (sim frame, up-positive). We issue every
        # delta ourselves, so this tracks true pitch closely; clamping makes
        # staring at the sky/feet physically impossible.
        self.est_pitch = 0.0
        # only guard real HID sinks; mocks in tests have no side effects
        self._guard = type(sink).__name__ == "QuartzInputSink"
        self._ws = None
        if self._guard:
            from AppKit import NSWorkspace  # pyobjc, present with Quartz

            self._ws = NSWorkspace.sharedWorkspace()

    def sync_pitch(self, measured_deg: float) -> None:
        """Blend the open-loop pitch integrator toward the CNN's measured
        self-pitch. Pure command integration drifts (measured response
        gain ~0.8, not 1.0) and then pins at the +/-55 clamp while the
        real camera is elsewhere -- silently eating every further pitch
        command in that direction (observed live: aim-assist pitch error
        stuck at -13 deg median)."""
        self.est_pitch = 0.85 * self.est_pitch + 0.15 * float(measured_deg)

    def game_frontmost(self) -> bool:
        """True when injecting is safe. Fail closed on any doubt."""
        if not self._guard:
            return True
        try:
            app = self._ws.frontmostApplication()
            name = (app.localizedName() or "").lower()
            return any(k in name for k in self.FRONTMOST_OK)
        except Exception:
            return False

    def _set_key(self, name: str, want_down: bool) -> None:
        if self._down.get(name, False) != want_down:
            self.sink.key(name, want_down)
            self._down[name] = want_down

    def injection_safe(self) -> bool:
        """Game frontmost AND mouse grabbed. A visible cursor (death screen,
        Esc menu, chat) makes delta injection leak: deltas move the real
        cursor and clicks land wherever it drifts, possibly outside the
        game window entirely."""
        return self.game_frontmost() and not self.sink.cursor_visible()

    def apply(self, action: Sequence[int]) -> None:
        if not self.injection_safe():
            # release everything and inject nothing this tick
            for name in list(self._down):
                self._set_key(name, False)
            return
        fwd, strafe, jump, sprint, attack, yaw_bin, pitch_bin = (
            int(a) for a in action
        )
        self._set_key("w", fwd == 2)
        self._set_key("s", fwd == 0)
        self._set_key("a", strafe == 0)
        self._set_key("d", strafe == 2)
        self._set_key("space", jump == 1)
        self._set_key("sprint", sprint == 1)
        dx, dy = self.calibration.bins_to_pixels(yaw_bin, pitch_bin)
        delta = float(CAMERA_BINS[pitch_bin])
        allowed = max(-self.PITCH_LIMIT_DEG,
                      min(self.PITCH_LIMIT_DEG, self.est_pitch + delta)
                      ) - self.est_pitch
        if delta != 0.0 and allowed != delta:
            dy = dy * (allowed / delta)
        self.est_pitch += allowed
        # sim/spec pitch is UP-positive (env.py: "positive = looking up");
        # Calibration expects Minecraft's DOWN-positive convention. The
        # missing negation inverted the whole vertical axis at the
        # sim->screen boundary (root cause of sky-lock and of aim-assist
        # pitch error growing instead of converging: measured +0.12 err
        # change per unit command, yaw -0.74).
        dy = -dy
        ix, iy = self.accum.step(dx, dy)
        if ix or iy:
            self.sink.move_mouse(ix, iy)
        if attack == 1:
            self.sink.click(True)
            self.sink.click(False)

    def neutral(self) -> None:
        self.apply(NEUTRAL_ACTION)

    def release_all(self) -> None:
        for name, down in list(self._down.items()):
            if down:
                self.sink.key(name, False)
                self._down[name] = False
        self.sink.release_all()
        self.accum.reset()


# ---------------------------------------------------------------------------
# Flight recorder
# ---------------------------------------------------------------------------

class FlightRecorder:
    """One JSON object per tick, newline-delimited. Cheap enough for 20 Hz."""

    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "w")

    def record(self, payload: dict) -> None:
        self._fh.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()


def _round_list(arr: np.ndarray, ndigits: int = 5) -> List[float]:
    return [round(float(x), ndigits) for x in np.asarray(arr).ravel()]


# ---------------------------------------------------------------------------
# Latency accounting
# ---------------------------------------------------------------------------

@dataclass
class LatencyStats:
    samples_ms: Dict[str, List[float]] = field(
        default_factory=lambda: {s: [] for s in STAGES + ("tick",)}
    )
    overruns: int = 0

    def add(self, stage: str, ms: float) -> None:
        self.samples_ms[stage].append(ms)

    def summary(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for stage, xs in self.samples_ms.items():
            if not xs:
                continue
            a = np.asarray(xs)
            out[stage] = {
                "mean": float(a.mean()),
                "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)),
                "max": float(a.max()),
            }
        return out

    def format_table(self) -> str:
        rows = ["%-10s %9s %9s %9s %9s" % ("stage", "mean ms", "p50 ms", "p95 ms", "max ms")]
        rows.append("-" * 50)
        for stage in STAGES + ("tick",):
            s = self.summary().get(stage)
            if s is None:
                continue
            rows.append(
                "%-10s %9.3f %9.3f %9.3f %9.3f"
                % (stage, s["mean"], s["p50"], s["p95"], s["max"])
            )
        rows.append(
            "budget %.0f ms/tick; overruns: %d" % (TICK_BUDGET_MS, self.overruns)
        )
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

@dataclass
class LoopResult:
    ticks_run: int
    state: MatchState
    stats: LatencyStats
    stop_reason: str


class TickLoop:
    """Drives one live (or dry-run) match.

    Parameters
    ----------
    frame_source, input_sink : the I/O backends (mock or Quartz).
    policy : callable ``obs_f32[48] -> int64[7]``; defaults to a fresh
        ``TorchPolicy`` (pass ``TorchPolicy(checkpoint=...)`` for a trained one).
    calibration : degrees->pixels mapping for camera bins.
    recorder_path : JSONL flight-recorder output (None disables).
    kill_switch_path : sentinel file checked every tick.
    stop_check : optional callback (e.g. hotkey listener) polled every tick.
    clock, sleep : injectable time source; defaults to wall clock.
    auto_fight : transition idle -> fighting on the first tick (dry-run and
        arranged duels); otherwise call :meth:`start_fight`.
    """

    def __init__(
        self,
        frame_source: FrameSource,
        input_sink: InputSink,
        policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        calibration: Optional[Calibration] = None,
        recorder_path: Optional[str] = None,
        kill_switch_path: str = KILL_SWITCH_PATH,
        stop_check: Optional[Callable[[], bool]] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        auto_fight: bool = True,
        clear_death_flag: bool = True,
        aim_assist: bool = False,
        click_discipline: bool = False,
        crit_assist: bool = False,
        dump_frames_dir: Optional[str] = None,
    ):
        self.frame_source = frame_source
        self.input_sink = input_sink
        self.policy = policy if policy is not None else TorchPolicy()
        self.applier = ActionApplier(input_sink, calibration)
        self.recorder = FlightRecorder(recorder_path) if recorder_path else None
        self.kill_switch_path = kill_switch_path
        self.stop_check = stop_check
        self.clock = clock
        self.sleep = sleep
        self.auto_fight = auto_fight
        self.aim_assist = aim_assist
        self.click_discipline = click_discipline
        self.crit_assist = crit_assist
        self._crit_t = 0
        self._ticks_unseen = 999
        # target-motion lead state for the visible-branch aim assist:
        # stationary-target probe landed 65% of the 1.8 hit cap vs 25%
        # against a strafing P4 -- pure pursuit on delayed perception
        # trails a mover by (lag x angular rate). Lead reconstructs the
        # target's angular motion and aims where it will be.
        self._aim_prev_err = None
        self._aim_prev_cmd = 0.0
        self._aim_motion = 0.0
        self._scan_dir = 1.0  # search direction when holds go stale
        self._prev_percep_frame = None  # temporal pair buffer (stack-2)
        # optional combat-frame dump for perception fine-tuning: the
        # collector never sprints, so its frames miss the sprint-FOV /
        # swing-animation domain the model actually fights in
        self.dump_frames_dir = dump_frames_dir
        self._dump_frames: List = []
        self._dump_times: List = []
        self.state = MatchState.IDLE
        self.stats = LatencyStats()
        self._percep = _resolve_perception()
        self._human = _resolve_humanizer()
        self._prev_action = NEUTRAL_ACTION.copy()
        self._move_latch = _MoveLatch(hold=3)  # mirror of sim act_hold_ticks
        self._focus_miss = 0
        self._grab_miss = 0     # cursor-free ticks since the last settled grab
        self._grab_streak = GRAB_SETTLE_TICKS  # consecutive grabbed ticks
        self._was_dead = False  # death cue seen; reset policy on respawn
        self._respawn_i = 0     # cycles through candidate button positions
        self._respawn_cd = 0    # ticks until the next respawn click attempt
        self._respawn_wait = 0  # ticks since the death cue appeared
        self._respawn_won = None  # candidate index that worked last death
        self.death_flag_path = DEATH_FLAG_PATH
        if clear_death_flag:
            # a stale flag from a previous MATCH must not read as "dead";
            # supervisor RESTARTS pass False so a pending death survives
            try:
                os.remove(self.death_flag_path)
            except OSError:
                pass
        log.info(
            "TickLoop ready (perception=%s, humanizer=%s)",
            self._percep["source"],
            self._human["source"],
        )

    # -- death screen / respawn ----------------------------------------------
    @staticmethod
    def _looks_like_death_screen(frame: np.ndarray) -> bool:
        """1.8's death screen overlays the world with a translucent red
        gradient, so the mean red channel dominates green and blue by a
        margin no gameplay or menu state reaches."""
        m = frame.reshape(-1, frame.shape[-1]).mean(axis=0)
        return float(m[0]) > 25.0 and m[0] > 1.5 * m[1] and m[0] > 1.5 * m[2]

    @staticmethod
    def _find_button_runs(img: np.ndarray) -> List:
        """Locate 1.8 button bars in a full-resolution frame.

        Minecraft 1.8 buttons are wide low-saturation gray bars; scan the
        vertical band 35-80% of height (middle-20% width strip) for
        contiguous mostly-gray row runs of plausible button thickness.
        Returns a list of (x_px, y_px) run centers, top to bottom.
        """
        h, w = img.shape[:2]
        top = int(h * 0.35)
        # probe only the middle 20% of the width: the 200-GUI-px button is
        # centered and covers that strip at every realistic GUI scale
        band = img[top:int(h * 0.8), int(w * 0.40):int(w * 0.60)]
        r = band[..., 0].astype(np.int16)
        g = band[..., 1].astype(np.int16)
        b = band[..., 2].astype(np.int16)
        low_sat = (np.abs(r - g) < 14) & (np.abs(g - b) < 14)
        body = low_sat & (g > 70) & (g < 215)
        label = low_sat & (g >= 215)  # the button's white text
        frac = (body | label).mean(axis=1)
        body_frac = body.mean(axis=1)
        runs, start = [], None
        for i, f in enumerate(frac):
            if f > 0.55 and start is None:
                start = i
            elif f <= 0.55 and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, len(frac)))
        lo, hi = max(2, int(h * 0.015)), int(h * 0.12)
        # a real button run must also contain actual gray body rows (a
        # block of pure white text alone is not a button)
        runs = [(a, b) for a, b in runs
                if lo <= (b - a) <= hi and body_frac[a:b].max() > 0.55]
        return [(w // 2, top + (a + b) // 2) for a, b in runs]

    @classmethod
    def _find_respawn_button_px(cls, img: np.ndarray):
        """Topmost button-bar center, or None (kept for tests)."""
        runs = cls._find_button_runs(img)
        return runs[0] if runs else None

    def _respawn_candidates(self) -> List:
        """Candidate screen points (global, top-left origin) for the 1.8
        Respawn button: GUI y = height/4 + 82 in scaled units, but the
        title-bar height and retina factor are unknown, so try the plausible
        combinations; a miss lands inside the game window and is harmless."""
        bounds_fn = getattr(self.frame_source, "window_bounds", None)
        b = bounds_fn() if callable(bounds_fn) else None
        if not b:
            return []
        x0, y0, w, h = b
        out: List = []
        for title_bar in (28.0, 0.0):
            ch = h - title_bar
            for retina in (2, 1):
                pw, ph = w * retina, ch * retina
                s = 1
                while s < 4 and 320 * (s + 1) <= pw and 240 * (s + 1) <= ph:
                    s += 1
                y = y0 + title_bar + ch / 4.0 + 82.0 * s / retina
                if all(abs(y - oy) > 2.0 for _, oy in out):
                    out.append((x0 + w / 2.0, y))
        return out

    def _try_respawn_click(self) -> None:
        """Rate-limited absolute click on a respawn-button candidate.
        Only ever fires with the game frontmost and the cursor visible.

        1.8's death screen ignores clicks for its first second, so the
        first attempt waits ~25 ticks; after that, candidates are cycled
        every 10 ticks, starting from whichever position worked last time
        (the geometry is stable within a session)."""
        click_at = getattr(self.input_sink, "click_at", None)
        if click_at is None or not self.applier.game_frontmost():
            return
        self._respawn_wait += 1
        if self._respawn_wait < 25:
            return
        self._respawn_cd -= 1
        if self._respawn_cd > 0:
            return
        self._respawn_cd = 10  # one attempt per ~0.5 s

        # preferred: find the actual buttons in the full-res frame
        fullres_fn = getattr(self.frame_source, "get_frame_fullres", None)
        bounds_fn = getattr(self.frame_source, "window_bounds", None)
        b = bounds_fn() if callable(bounds_fn) else None
        if callable(fullres_fn) and b:
            x0, y0, w_pt, h_pt = b
            move_to = getattr(self.input_sink, "move_cursor_to", None)
            if callable(move_to):
                # park the cursor over the "You died" text first: a
                # hovered button tints bluish and vanishes from the
                # detector, which once left only Title Screen visible
                # and got it clicked (game exited to the main menu)
                move_to(x0 + w_pt / 2.0, y0 + h_pt * 0.22)
                time.sleep(0.06)
            try:
                img = fullres_fn()
                runs = self._find_button_runs(img)
            except Exception:
                runs = []
            # the death screen always shows Respawn above Title Screen;
            # refuse to click unless BOTH are visible, and never fall
            # back to blind geometry when detection ran but disagreed
            if len(runs) >= 2:
                hit = runs[0]
                fx = img.shape[1] / max(w_pt, 1.0)
                fy = img.shape[0] / max(h_pt, 1.0)
                x, y = x0 + hit[0] / fx, y0 + hit[1] / fy
                log.info("respawn button located in frame -- clicking "
                         "(%.0f, %.0f)", x, y)
                click_at(x, y)
            return

        # fallback: cycle the geometry candidates
        cands = self._respawn_candidates()
        if not cands:
            return
        if self._respawn_won is not None and self._respawn_i == 0:
            idx = self._respawn_won
        else:
            idx = self._respawn_i % len(cands)
        x, y = cands[idx % len(cands)]
        self._last_respawn_idx = idx % len(cands)
        self._respawn_i += 1
        log.info("death screen detected -- clicking respawn at (%.0f, %.0f)", x, y)
        click_at(x, y)

    # -- state machine -------------------------------------------------------
    def start_fight(self) -> None:
        if self.state is MatchState.IDLE:
            self.state = MatchState.FIGHTING

    def _should_stop(self) -> Optional[str]:
        if os.path.exists(self.kill_switch_path):
            return "kill-switch file %s" % self.kill_switch_path
        if self.stop_check is not None and self.stop_check():
            return "stop_check callback"
        return None

    # -- main entry ----------------------------------------------------------
    def run(self, num_ticks: int) -> LoopResult:
        stop_reason = "completed %d ticks" % num_ticks
        tick = 0
        next_deadline = self.clock() + TICK_PERIOD
        try:
            for tick in range(num_ticks):
                reason = self._should_stop()
                if reason:
                    stop_reason = reason
                    break
                if self.state is MatchState.IDLE and self.auto_fight:
                    self.state = MatchState.FIGHTING

                t0 = self.clock()
                try:
                    frame = self.frame_source.get_frame()
                except FrameSourceExhausted:
                    stop_reason = "frame source exhausted"
                    break
                if frame.ndim != 3 or frame.shape[:2] != FRAME_SHAPE[1:]:
                    frame = downscale_area(frame)
                t1 = self.clock()

                # temporal pair for stack-2 perception (v13+): the hurt
                # flash is a between-frames event; single frames capped
                # recall at 0.18, pairs reach 0.51 (held-out round)
                if self._prev_percep_frame is not None:
                    enc_in = np.stack([self._prev_percep_frame, frame])
                else:
                    enc_in = frame
                self._prev_percep_frame = frame
                percep = np.asarray(
                    self._percep["encode"](enc_in), dtype=np.float32
                ).reshape(-1)[:PERCEPTION_DIM]
                obs = np.asarray(
                    self._percep["assemble"](percep, self._prev_action),
                    dtype=np.float32,
                ).reshape(-1)
                t2 = self.clock()

                cursor_free = self.input_sink.cursor_visible()
                if cursor_free:
                    self._grab_streak = 0
                else:
                    self._grab_streak += 1
                # the grab signal flaps around GUI transitions; only a
                # sustained grabbed streak counts as "the game has control"
                settled = not cursor_free and self._grab_streak >= GRAB_SETTLE_TICKS

                if self.dump_frames_dir is not None and settled:
                    # settled-only: death-screen/menu frames would carry
                    # wrong ground-truth labels
                    self._dump_frames.append(np.asarray(frame, dtype=np.uint8))
                    self._dump_times.append(time.time() * 1000.0)

                if settled and self._was_dead:
                    # grab is stably back after a death: fresh episode
                    self._was_dead = False
                    self._respawn_won = getattr(self, "_last_respawn_idx", None)
                    self._respawn_i = 0
                    self._respawn_wait = 0
                    try:
                        os.remove(self.death_flag_path)
                    except OSError:
                        pass
                    if hasattr(self.policy, "reset"):
                        self.policy.reset()
                    self.applier.est_pitch = 0.0
                    self.applier.accum.reset()

                if settled and self.state is MatchState.FIGHTING:
                    s0, _s1 = OBS_LAYOUT["self_pitch_sincos"]
                    self.applier.sync_pitch(np.degrees(
                        np.arctan2(float(obs[s0]), float(obs[s0 + 1]))))
                    raw_action = np.asarray(self.policy(obs), dtype=np.int64).reshape(-1)
                    action = np.asarray(
                        self._human["fn"](raw_action), dtype=np.int64
                    ).reshape(-1)
                    if self.click_discipline and action[4] == 1:
                        # 1.8: any swing while sprinting resets sprint, so
                        # spam-clicking (measured 10.6 CPS) cripples the
                        # chase. Practice bots click only when the shot is
                        # real: crosshair on the hitbox and within reach.
                        vis_ = obs[OBS_LAYOUT["enemy_visible"][0]] > 0.5
                        d_ = max(float(obs[OBS_LAYOUT["dist"][0]]) * 8.0, 0.5)
                        y_ = abs(float(obs[OBS_LAYOUT["aim_err_yaw"][0]])) * 180.0
                        p_ = abs(float(obs[OBS_LAYOUT["aim_err_pitch"][0]])) * 90.0
                        on_box = (y_ < np.degrees(np.arctan2(0.35, d_))
                                  and p_ < np.degrees(np.arctan2(1.05, d_)))
                        # generous range: perceived dist carries ~1 block
                        # of error and a strangled gate produced 0.5 CPS
                        # (0 kills) live; wasted swings only cost a brief
                        # sprint reset, the client validates real reach
                        if not (vis_ and d_ < 5.0 and on_box):
                            action[4] = 0
                        # hurt-window timing (P4-Hacker's edge): a swing
                        # while the enemy's 10-tick hurt window is still
                        # open deals nothing and resets our sprint --
                        # hold the click until the window expires
                        elif obs[OBS_LAYOUT["enemy_hurt"][0]] > 0.15:
                            action[4] = 0
                    # assist gates on the RAW CNN sighting for THIS tick,
                    # not the obs bit: the adapter's last-seen holds keep
                    # obs.enemy_visible high ~70% of ticks while the CNN
                    # truly sees ~39% -- and homing on a held (stale) aim
                    # error is the measured spin loop (telemetry round
                    # 2026-07-25: obs-gated aim = 40 deg median true err,
                    # corr +0.19; raw-CNN frames = 15 deg, corr +0.56).
                    # Held ticks now fall to predictive re-acquisition.
                    raw_vis = float(percep[PERCEPTION_LAYOUT["visible"]]) > 0.5
                    if self.aim_assist and raw_vis:
                        # bounded-rate homing on the perceived aim error --
                        # the exact bins the sim's T1-Aimbot commands, so
                        # the sign/scale chain is proven. PLUS lead: err
                        # evolution + our own commanded turn = the target's
                        # angular motion; aim LEAD_TICKS ahead of it so a
                        # strafer meets the crosshair instead of escaping it.
                        self._ticks_unseen = 0
                        err_deg = float(obs[OBS_LAYOUT["aim_err_yaw"][0]]) * 180.0
                        # DAMPED homing: the CNN error estimate carries
                        # ~15 deg noise at corr 0.56 (telemetry round);
                        # full-error homing every tick thrashed a median
                        # 15 deg/tick (32% of ticks >=20). Command half
                        # the estimate, cap 20 deg/tick, no lead --
                        # extrapolating a corr-0.56 signal amplifies noise.
                        # retuned for v11 (corr +0.83, 6.9 deg): the
                        # 0.5/±20 damping was sized for corr-0.56 noise
                        action[5] = _nearest_bin(
                            float(np.clip(0.7 * err_deg, -25.0, 25.0)))
                        self._aim_prev_err = err_deg
                        self._aim_prev_cmd = float(_BINS[int(action[5])])
                        action[6] = _nearest_bin(float(np.clip(
                            0.65 * float(obs[OBS_LAYOUT["aim_err_pitch"][0]]) * 90.0,
                            -18.0, 18.0)))
                    elif self.aim_assist:
                        # no unseen-tick cutoff: the training env's assist
                        # sweeps on EVERY unseen tick (cam_assist=1), and
                        # the policy camera heads are untrained noise under
                        # that regime -- handing them yaw after 25 unseen
                        # ticks was a measured spin driver (39% true
                        # visibility => constant >1.25s unseen stretches).
                        self._aim_prev_err = None
                        # predictive re-acquisition: P4 circle-strafes out
                        # of the 120 deg FOV cone (visibility was ~50%), and
                        # a blind policy flails while it keeps hitting. Keep
                        # swinging the camera toward the enemy's last-known
                        # position (held rel_pos, self-yaw frame [fwd,up,lat]),
                        # velocity-extrapolated to lead the circle.
                        self._ticks_unseen += 1
                        if self._ticks_unseen <= 6:
                            # fresh hold: aim at the last-seen bearing
                            # (holds are frame-consistent, verified by
                            # unit rotation test)
                            lead = 4.0
                            fwd = float(obs[0]) + lead * float(obs[3])
                            lat = float(obs[2]) + lead * float(obs[5])
                            if abs(fwd) > 1e-3 or abs(lat) > 1e-3:
                                yaw_to = np.degrees(np.arctan2(lat, fwd))
                                action[5] = _nearest_bin(
                                    float(np.clip(yaw_to, -30.0, 30.0)))
                                self._scan_dir = 1.0 if yaw_to >= 0 else -1.0
                        else:
                            # stale hold: measured live, hold-chasing
                            # pointed at the true enemy only 38% of the
                            # time at the +/-30 clip -- worse than chance
                            # and the visible half of "it just spins".
                            # Deterministic scan in the last plausible
                            # direction finds a circler and reads as
                            # purposeful.
                            action[5] = _nearest_bin(15.0 * self._scan_dir)
                        # ALSO recover pitch toward level while searching:
                        # without this the policy can drive pitch off during
                        # the unseen window (enemy leaves frame, stays
                        # unseen, pitch runs to the clamp -- observed as the
                        # camera pegged looking up). Steer the integrated
                        # camera pitch back toward 0 so it re-acquires level.
                        pitch_recover = float(np.clip(-self.applier.est_pitch,
                                                      -30.0, 30.0))
                        action[6] = _nearest_bin(pitch_recover)
                    if self.crit_assist and action[4] == 1:
                        # P4-Hacker crits 100%%; the policy crit ~12%% live,
                        # so every P4 hit dealt 1.5x ours. Mirror the
                        # practice-bot crit: on a wanted swing while
                        # grounded, JUMP instead of hitting this tick, then
                        # release the swing once airborne and falling (the
                        # 1.5x window). est_vy comes from the jump timer.
                        on_ground = obs[OBS_LAYOUT["self_on_ground"][0]] > 0.5
                        if on_ground and self._crit_t <= 0:
                            self._crit_t = 6          # ticks to falling state
                            action[2] = 1             # jump (start crit arc)
                            action[4] = 0             # hold the swing
                        elif self._crit_t > 3:
                            action[4] = 0             # still rising: hold
                        # else: airborne & falling -> let the crit swing land
                    if self._crit_t > 0:
                        self._crit_t -= 1
                else:
                    # un-grabbed or not yet settled: policy paused, nothing
                    # may be injected
                    action = NEUTRAL_ACTION.copy()
                assert action.shape == (NUM_ACTION_HEADS,), action.shape
                t3 = self.clock()

                if not settled:
                    self.applier.release_all()
                    self._grab_miss += 1
                    if cursor_free:
                        dead_cue = os.path.exists(self.death_flag_path) \
                            or self._looks_like_death_screen(frame)
                        if dead_cue:
                            self._was_dead = True
                            self._try_respawn_click()
                    if self._grab_miss >= 1200:  # ~60 s
                        stop_reason = "mouse grab lost for 60 s"
                        break
                elif self.applier.game_frontmost():
                    self._grab_miss = 0
                    self._focus_miss = 0
                    action = self._move_latch.filter(action)
                    self.applier.apply(action)
                else:
                    self._focus_miss += 1
                    self.applier.apply(action)  # releases held keys, injects nothing
                    if self._focus_miss >= 10:  # ~0.5 s
                        stop_reason = "game lost frontmost focus"
                        break
                self._prev_action = action
                t4 = self.clock()

                lat = {
                    "capture": (t1 - t0) * 1000.0,
                    "encode": (t2 - t1) * 1000.0,
                    "policy": (t3 - t2) * 1000.0,
                    "inject": (t4 - t3) * 1000.0,
                    "tick": (t4 - t0) * 1000.0,
                }
                for stage in STAGES + ("tick",):
                    self.stats.add(stage, lat[stage])
                if lat["tick"] > TICK_BUDGET_MS:
                    self.stats.overruns += 1
                    log.warning(
                        "tick %d exceeded budget: %.1f ms > %.0f ms "
                        "(capture=%.1f encode=%.1f policy=%.1f inject=%.1f)",
                        tick, lat["tick"], TICK_BUDGET_MS,
                        lat["capture"], lat["encode"], lat["policy"], lat["inject"],
                    )

                if self.recorder:
                    self.recorder.record(
                        {
                            "tick": tick,
                            "t": round(t0, 6),
                            "state": self.state.value,
                            "grabbed": 0 if cursor_free else 1,
                            "settled": 1 if settled else 0,
                            "obs": _round_list(obs),
                            "percep": _round_list(percep),
                            "action": [int(a) for a in action],
                            "latency_ms": {k: round(v, 3) for k, v in lat.items()},
                        }
                    )

                # -- pacing ---------------------------------------------------
                now = self.clock()
                if now < next_deadline:
                    self.sleep(next_deadline - now)
                else:
                    next_deadline = now  # resync after an overrun
                next_deadline += TICK_PERIOD
            else:
                tick = num_ticks - 1
            ticks_run = tick + 1 if num_ticks > 0 else 0
            if stop_reason != "completed %d ticks" % num_ticks:
                ticks_run = tick  # the breaking tick did not execute stages
        finally:
            self.state = MatchState.STOPPED
            self.applier.release_all()
            if self.recorder:
                self.recorder.close()
            if self.dump_frames_dir is not None and self._dump_frames:
                os.makedirs(self.dump_frames_dir, exist_ok=True)
                np.save(os.path.join(self.dump_frames_dir, "frames.npy"),
                        np.stack(self._dump_frames))
                np.save(os.path.join(self.dump_frames_dir, "times.npy"),
                        np.asarray(self._dump_times, dtype=np.float64))
            self.frame_source.close()
        return LoopResult(
            ticks_run=ticks_run, state=self.state, stats=self.stats,
            stop_reason=stop_reason,
        )
