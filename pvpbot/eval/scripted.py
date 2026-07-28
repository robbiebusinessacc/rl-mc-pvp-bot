"""Scripted opponent policies, difficulty tiers T0..T4.

Interface
---------
Every tier implements ``ScriptedPolicy.act(obs) -> actions`` where ``obs`` is
``(N, 48) float32`` (one side's observation rows, laid out per
``pvpbot.spec.OBS_LAYOUT``) and the return value is ``(N, 7) int64`` action
indices per ``pvpbot.spec.ACTION_HEADS``.  All tiers are vectorized numpy and
use only the observation vector -- exactly the same information a trained
policy sees (aim_err_* slots, dist / in_reach, hurt timers, prev_action).
Stochastic tiers draw from a seeded ``numpy`` Generator, so behaviour is
reproducible for a fixed seed and batch size.

Statefulness: tiers keep small per-env state (attack cooldowns, strafe
direction, aim-tracking EMA).  ``begin(num_envs)`` (re)allocates it and
``on_done(mask)`` resets the rows of envs that just finished; the arena calls
both.  ``act`` self-heals if the batch size changes.

Skill model / intended Elo gaps
-------------------------------
Separation between tiers comes from three axes: aim, movement, and attack
cadence/timing (clicks per second and timing hits to the enemy's hurt-time
expiry -- the classic 1.8 "combo timing" skill).  Intended anchors, roughly
200 Elo per tier below T2 and ~150 above:

    T0 Idle               ~ 800  Elo   never acts at all
    T1 Aimbot-Stationary  ~1000  Elo   perfect aim, no movement, slow clicks
    T2 Chaser             ~1300  Elo   sprints at you, spam clicks
    T3 Strafer            ~1450  Elo   human-ish aim noise/lag, circle-strafe,
                                       good-but-jittery click timing
    T4 Pro                ~1600  Elo   near-perfect hurt-expiry timing, W-tap
                                       sprint resets, crit jumps, spacing
                                       while in own hurt-time

The attack cadences (T1 every 6 ticks, T2 every 4, T3 every 2-3 with jitter
and occasional whiffs, T4 exactly on hurt-expiry) are what make the gradient
measurable even in the crude stub sim, where hits are gated only by reach and
the target's hurt-timer: expected delay between the enemy's hurt-time expiry
and your next swing directly sets your damage rate.
"""
from typing import List, Optional

import numpy as np

from pvpbot.spec import ACTION_HEAD_SIZES, CAMERA_BINS, NUM_ACTION_HEADS, OBS_LAYOUT

_BINS = np.asarray(CAMERA_BINS, dtype=np.float32)
_NOOP_CAM = int(np.argmin(np.abs(_BINS)))  # index of the 0.0 deg/tick bin

# Action head indices (order fixed by spec.ACTION_HEADS).
FORWARD, STRAFE, JUMP, SPRINT, ATTACK, YAW, PITCH = range(NUM_ACTION_HEADS)

# Observation slot indices we read (scalars).
_AIM_YAW = OBS_LAYOUT["aim_err_yaw"][0]
_AIM_PITCH = OBS_LAYOUT["aim_err_pitch"][0]
_DIST = OBS_LAYOUT["dist"][0]
_IN_REACH = OBS_LAYOUT["in_reach"][0]
_SELF_HURT = OBS_LAYOUT["self_hurt"][0]
_ENEMY_HURT = OBS_LAYOUT["enemy_hurt"][0]
_PREV_ATTACK = OBS_LAYOUT["prev_action"][0] + ATTACK


def nearest_camera_bin(err_deg: np.ndarray) -> np.ndarray:
    """Map desired camera deltas (degrees) to nearest CAMERA_BINS index."""
    err = np.asarray(err_deg, dtype=np.float32)
    return np.abs(err[:, None] - _BINS[None, :]).argmin(axis=1).astype(np.int64)


class ScriptedPolicy:
    """Base class: batch bookkeeping + seeded rng + noop action template."""

    name = "scripted"
    intended_elo = 1000  # documented anchor, see module docstring

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self._num_envs: Optional[int] = None

    # -- lifecycle ---------------------------------------------------------
    def begin(self, num_envs: int) -> None:
        """(Re)allocate per-env state for a batch of `num_envs` duels."""
        self._num_envs = int(num_envs)
        self.rng = np.random.default_rng(self.seed)
        self._alloc(self._num_envs)

    def on_done(self, done: np.ndarray) -> None:
        """Reset per-env state rows for envs that just finished."""
        mask = np.asarray(done, dtype=bool)
        if self._num_envs is not None and mask.any():
            self._reset_rows(mask)

    def _alloc(self, n: int) -> None:  # override in stateful tiers
        pass

    def _reset_rows(self, mask: np.ndarray) -> None:  # override
        pass

    # -- acting ------------------------------------------------------------
    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        assert obs.ndim == 2, "obs must be (N, OBS_DIM)"
        if self._num_envs != obs.shape[0]:
            self.begin(obs.shape[0])
        actions = self._act(obs)
        assert actions.shape == (obs.shape[0], NUM_ACTION_HEADS)
        return actions.astype(np.int64, copy=False)

    def _act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _noop(n: int) -> np.ndarray:
        """forward=none, strafe=none, no jump/sprint/attack, camera still."""
        a = np.zeros((n, NUM_ACTION_HEADS), dtype=np.int64)
        a[:, FORWARD] = 1
        a[:, STRAFE] = 1
        a[:, YAW] = _NOOP_CAM
        a[:, PITCH] = _NOOP_CAM
        return a

    @staticmethod
    def _aim(actions: np.ndarray, err_yaw_deg: np.ndarray,
             err_pitch_deg: np.ndarray) -> None:
        actions[:, YAW] = nearest_camera_bin(err_yaw_deg)
        actions[:, PITCH] = nearest_camera_bin(err_pitch_deg)


class Idle(ScriptedPolicy):
    """T0 -- stands perfectly still, never attacks.

    Rating anchor at the bottom of the ladder (~800 Elo intent).  Everything
    above it must at least beat it whenever the spawn allows contact.
    """

    name = "T0-Idle"
    intended_elo = 800

    def _act(self, obs: np.ndarray) -> np.ndarray:
        return self._noop(obs.shape[0])


class AimbotStationary(ScriptedPolicy):
    """T1 -- perfect aim, zero movement, casual click speed.

    Reads aim_err_yaw / aim_err_pitch and snaps the camera with the nearest
    CAMERA_BINS delta each tick (converges from any angle in <= 6 ticks).
    Attacks only when in_reach, at a leisurely one swing per 7 ticks
    (~3 cps; with a 10-tick hurt window that means it lands a hit at most
    every 14 ticks -- clearly slower than T2's spam).  Never moves, so it
    only ever fights opponents that come to it (or spawns already in
    reach).  Intended ~+200 Elo over T0.
    """

    name = "T1-Aimbot"
    intended_elo = 1000
    click_cooldown = 7

    def _alloc(self, n: int) -> None:
        self._cd = np.zeros(n, dtype=np.int32)

    def _reset_rows(self, mask: np.ndarray) -> None:
        self._cd[mask] = 0

    def _act(self, obs: np.ndarray) -> np.ndarray:
        n = obs.shape[0]
        a = self._noop(n)
        self._aim(a, obs[:, _AIM_YAW] * 180.0, obs[:, _AIM_PITCH] * 90.0)
        self._cd = np.maximum(self._cd - 1, 0)
        swing = (obs[:, _IN_REACH] > 0.5) & (self._cd <= 0)
        a[swing, ATTACK] = 1
        self._cd[swing] = self.click_cooldown
        return a


class Chaser(ScriptedPolicy):
    """T2 -- perfect aim + sprints straight at the enemy + spam clicks.

    Movement: always holds forward + sprint; because the camera is locked
    onto the enemy, "forward" closes the distance.  Attacks whenever the
    enemy is in reach, one swing per 4 ticks (~5 cps spam clicking), with no
    regard for the enemy's hurt-time (wasted clicks reset the cooldown, so
    its expected delay behind the enemy's hurt-expiry is ~1.5 ticks).
    Intended ~+300 Elo over T1: it can actually reach passive opponents and
    out-damages T1's slow clicks.
    """

    name = "T2-Chaser"
    intended_elo = 1300
    click_cooldown = 4

    def _alloc(self, n: int) -> None:
        self._cd = np.zeros(n, dtype=np.int32)

    def _reset_rows(self, mask: np.ndarray) -> None:
        self._cd[mask] = 0

    def _act(self, obs: np.ndarray) -> np.ndarray:
        n = obs.shape[0]
        a = self._noop(n)
        self._aim(a, obs[:, _AIM_YAW] * 180.0, obs[:, _AIM_PITCH] * 90.0)
        a[:, FORWARD] = 2
        a[:, SPRINT] = 1
        self._cd = np.maximum(self._cd - 1, 0)
        swing = (obs[:, _IN_REACH] > 0.5) & (self._cd <= 0)
        a[swing, ATTACK] = 1
        self._cd[swing] = self.click_cooldown
        return a


class Strafer(ScriptedPolicy):
    """T3 -- human-ish aim (noise + lag), circle-strafe, good click timing.

    Aim: the raw aim error is corrupted with Gaussian noise (sigma
    ~4 deg) and passed through an EMA tracker (alpha 0.55) to model human
    reaction lag, then snapped to the nearest camera bin.

    Movement: sprints straight in until ~3.5 blocks, then keeps forward
    pressure while circle-strafing; the strafe direction flips every 15-35
    ticks (per-env randomized) like a human juking.

    Attack timing: swings only when in reach and the enemy's hurt-timer is
    nearly out (<= 2 ticks left), with a 2-3 tick jittered click cooldown and
    a 10% chance to flub the click entirely.  Net effect: lands hits ~0.5-1
    tick after the enemy's hurt-expiry -- better than T2's blind spam,
    worse than T4's frame-perfect timing.  Intended ~+150 Elo over T2.
    """

    name = "T3-Strafer"
    intended_elo = 1450
    aim_noise_deg = 4.0
    aim_alpha = 0.55
    whiff_prob = 0.10
    strafe_range = 3.5

    def _alloc(self, n: int) -> None:
        self._cd = np.zeros(n, dtype=np.int32)
        self._tracked = np.zeros(n, dtype=np.float32)
        self._strafe_dir = self.rng.choice(np.asarray([0, 2]), size=n).astype(np.int64)
        self._strafe_t = self.rng.integers(15, 36, size=n).astype(np.int32)

    def _reset_rows(self, mask: np.ndarray) -> None:
        self._cd[mask] = 0
        self._tracked[mask] = 0.0
        k = int(mask.sum())
        self._strafe_dir[mask] = self.rng.choice(np.asarray([0, 2]), size=k)
        self._strafe_t[mask] = self.rng.integers(15, 36, size=k)

    def _aim_noisy(self, obs: np.ndarray) -> np.ndarray:
        n = obs.shape[0]
        err = obs[:, _AIM_YAW] * 180.0
        err = err + self.rng.normal(0.0, self.aim_noise_deg, n).astype(np.float32)
        self._tracked += self.aim_alpha * (err - self._tracked)
        return self._tracked

    def _movement(self, a: np.ndarray, obs: np.ndarray) -> None:
        dist = obs[:, _DIST] * 8.0
        a[:, FORWARD] = 2
        a[:, SPRINT] = 1
        close = dist <= self.strafe_range
        self._strafe_t -= 1
        flip = self._strafe_t <= 0
        if flip.any():
            self._strafe_dir[flip] = 2 - self._strafe_dir[flip]
            self._strafe_t[flip] = self.rng.integers(15, 36, size=int(flip.sum()))
        a[close, STRAFE] = self._strafe_dir[close]

    def _swing(self, obs: np.ndarray) -> np.ndarray:
        n = obs.shape[0]
        self._cd = np.maximum(self._cd - 1, 0)
        want = (
            (obs[:, _IN_REACH] > 0.5)
            & (obs[:, _ENEMY_HURT] <= 0.21)
            & (self._cd <= 0)
        )
        want &= self.rng.random(n) >= self.whiff_prob
        self._cd[want] = self.rng.integers(2, 4, size=int(want.sum()))
        return want

    def _act(self, obs: np.ndarray) -> np.ndarray:
        n = obs.shape[0]
        a = self._noop(n)
        self._aim(a, self._aim_noisy(obs), obs[:, _AIM_PITCH] * 90.0)
        self._movement(a, obs)
        a[self._swing(obs), ATTACK] = 1
        return a


class Pro(ScriptedPolicy):
    """T4 -- T3's movement plus frame-perfect timing, W-tap, crits, spacing.

    On top of the Strafer it:

    * Attacks exactly when the enemy's hurt-timer reads zero (reads the
      enemy_hurt slot; zero expected delay -> highest damage rate).
    * W-taps: when its previous action was an attack and the enemy's
      hurt-timer just went to max (prev_action + enemy_hurt cues = "I just
      landed that hit"), it releases forward and sprint for 2 ticks to reset
      sprint for the extra knockback on the next hit.
    * Crit jumps: short hops when very close (< 2.0 blocks) while the enemy
      is mid hurt-time, so it is falling when the next window opens.
    * Spacing: while in its own hurt-time and jammed against the enemy
      (< 1.2 blocks) it backs off to deny follow-up pressure.

    Aim noise is tighter (sigma 1.5 deg, alpha 0.85) than T3.  Intended
    ~+150 Elo over T3, ~+300 over T2 (the ladder's headline sanity check).
    """

    name = "T4-Pro"
    intended_elo = 1600
    aim_noise_deg = 1.5
    aim_alpha = 0.85
    strafe_range = 3.5
    wtap_ticks = 2

    def _alloc(self, n: int) -> None:
        self._tracked = np.zeros(n, dtype=np.float32)
        self._strafe_dir = self.rng.choice(np.asarray([0, 2]), size=n).astype(np.int64)
        self._strafe_t = self.rng.integers(15, 36, size=n).astype(np.int32)
        self._wtap = np.zeros(n, dtype=np.int32)

    def _reset_rows(self, mask: np.ndarray) -> None:
        self._tracked[mask] = 0.0
        k = int(mask.sum())
        self._strafe_dir[mask] = self.rng.choice(np.asarray([0, 2]), size=k)
        self._strafe_t[mask] = self.rng.integers(15, 36, size=k)
        self._wtap[mask] = 0

    def _act(self, obs: np.ndarray) -> np.ndarray:
        n = obs.shape[0]
        a = self._noop(n)
        dist = obs[:, _DIST] * 8.0
        in_reach = obs[:, _IN_REACH] > 0.5
        enemy_hurt = obs[:, _ENEMY_HURT]
        self_hurt = obs[:, _SELF_HURT]

        # aim (tight noise + fast tracker)
        err = obs[:, _AIM_YAW] * 180.0
        err = err + self.rng.normal(0.0, self.aim_noise_deg, n).astype(np.float32)
        self._tracked += self.aim_alpha * (err - self._tracked)
        self._aim(a, self._tracked, obs[:, _AIM_PITCH] * 90.0)

        # movement: approach + circle strafe (as T3)
        a[:, FORWARD] = 2
        a[:, SPRINT] = 1
        close = dist <= self.strafe_range
        self._strafe_t -= 1
        flip = self._strafe_t <= 0
        if flip.any():
            self._strafe_dir[flip] = 2 - self._strafe_dir[flip]
            self._strafe_t[flip] = self.rng.integers(15, 36, size=int(flip.sum()))
        a[close, STRAFE] = self._strafe_dir[close]

        # W-tap: prev tick attacked and enemy hurt-timer just maxed => we
        # landed it; release forward+sprint briefly to reset sprint.
        landed = (obs[:, _PREV_ATTACK] > 0.05) & (enemy_hurt >= 0.95)
        self._wtap[landed] = self.wtap_ticks
        wtapping = self._wtap > 0
        a[wtapping, FORWARD] = 1
        a[wtapping, SPRINT] = 0
        self._wtap = np.maximum(self._wtap - 1, 0)

        # spacing: while in own hurt-time and jammed in, back off.
        back = (self_hurt > 0.3) & (dist < 1.2)
        a[back, FORWARD] = 0

        # crit jumps: hop while enemy rides out mid hurt-time very close by
        # (still in reach at the apex, so timing is unaffected).
        hop = (
            (dist < 2.0)
            & (enemy_hurt > 0.4) & (enemy_hurt < 0.75)
            & (self.rng.random(n) < 0.5)
        )
        a[hop, JUMP] = 1

        # frame-perfect swing: exactly when the hurt window opens.
        a[in_reach & (enemy_hurt <= 0.0), ATTACK] = 1
        return a


def default_tiers(seed: int = 0) -> List[ScriptedPolicy]:
    """All scripted tiers T0..T4 with distinct sub-seeds."""
    return [
        Idle(seed),
        AimbotStationary(seed + 1),
        Chaser(seed + 2),
        Strafer(seed + 3),
        Pro(seed + 4),
    ]


# quick sanity: head sizes assumed by the action template
assert ACTION_HEAD_SIZES[FORWARD] == 3 and ACTION_HEAD_SIZES[ATTACK] == 2
