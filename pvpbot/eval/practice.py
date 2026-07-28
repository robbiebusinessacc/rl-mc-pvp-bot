"""Practice-server-style rule-based PvP bots (the real competition).

These four tiers are modeled on how actual rule-based Minecraft PvP bots
work -- the bots players train against on practice servers (pvp.land bot
fights) and the published mechanics of practice-bot plugins:

* PracticeBotPvP (Spigot/Modrinth) documents a four-tier difficulty table
  (Easy/Medium/Hard/Hacker) scaling crit rate (10/40/80/100%), combo
  (w-tap) rate (10/50/90/100%), strafe rate (30/60/85/100%) and reaction
  speed (1.5x slower .. 3x faster), plus: frame-accurate crits (jump,
  ~7 ticks to falling state, attack while falling), w-tap tick sequence
  (attack -> release forward 1 tick -> re-press), circle/zigzag strafing
  at a managed 3-3.5 block engage distance (back up under 2, chase past
  4), and random hops every 20-60 ticks.
* Wurst's AimAssist documents the standard aim model: rotate toward the
  target at a capped rotation speed (default 600 deg/s, range 10-3600)
  within a FOV gate -- i.e. bounded-rate homing, not teleport aim.
* Competitive guides put humanlike combat at 8-14 CPS with smooth
  tracking, unpredictable strafe-direction switches, and w-tapping as the
  core combo mechanic.

Milestone anchors (intended_elo) express the ladder we are climbing:
beating P4-Hacker in-sim at >90% is the "proper PvP bot" bar the learned
policy must clear; P1..P3 are the intermediate milestones.
"""
from typing import List

import numpy as np

from pvpbot.eval.scripted import (
    ATTACK,
    FORWARD,
    JUMP,
    PITCH,
    SPRINT,
    STRAFE,
    YAW,
    ScriptedPolicy,
    nearest_camera_bin,
)
from pvpbot.spec import OBS_LAYOUT, TICK_RATE

_AIM_YAW = OBS_LAYOUT["aim_err_yaw"][0]
_AIM_PITCH = OBS_LAYOUT["aim_err_pitch"][0]
_DIST = OBS_LAYOUT["dist"][0]
_ENEMY_HURT = OBS_LAYOUT["enemy_hurt"][0]
_SELF_GROUND = OBS_LAYOUT["self_on_ground"][0]
_SELF_SPRINT = OBS_LAYOUT["self_sprinting"][0]


class PracticeBot(ScriptedPolicy):
    """Parameterized practice bot; subclasses set the difficulty table row."""

    name = "practice"
    intended_elo = 1300

    rot_speed_dps = 240.0     # aim-assist style rotation cap (deg/sec)
    reaction_ticks = 3        # obs delay before the aim loop sees it
    cps = 8.0                 # click rate cap (competitive band is 8-14)
    crit_rate = 0.4           # PracticeBotPvP difficulty-table probabilities
    wtap_rate = 0.5
    strafe_rate = 0.6
    aim_jitter_deg = 2.5      # tracking noise while rotating
    # PracticeBotPvP manages 3-3.5 blocks with its default 3.5 reach;
    # our sim uses the honest client reach of 3.0, so the band shifts
    # inside it (engage INSIDE sword range or you never land a hit)
    engage_lo = 1.8           # back up under this (blocks)
    engage_hi = 2.7           # walk in beyond this
    chase_dist = 3.2          # sprint-chase beyond this
    hurt_timed_swing = False  # P4: swing exactly as the hurt window expires

    # -- state -------------------------------------------------------------
    def _alloc(self, n: int) -> None:
        self.click_cd = np.zeros(n, dtype=np.int32)
        self.strafe_dir = self.rng.choice([-1, 1], n)
        self.strafe_timer = self.rng.integers(10, 50, n)
        self.strafing = self.rng.random(n) < self.strafe_rate
        self.wtap_phase = np.zeros(n, dtype=np.int32)
        self.crit_phase = np.zeros(n, dtype=np.int32)
        self.hop_timer = self.rng.integers(20, 60, n)  # natural movement
        self.prev_hurt = np.zeros(n, dtype=np.float32)
        k = max(self.reaction_ticks, 1)
        self._err_buf = np.zeros((k, n, 2), dtype=np.float32)
        self._buf_i = 0

    def _reset_rows(self, mask: np.ndarray) -> None:
        self.click_cd[mask] = 0
        self.wtap_phase[mask] = 0
        self.crit_phase[mask] = 0
        self.prev_hurt[mask] = 0.0
        self._err_buf[:, mask] = 0.0

    # -- behavior ----------------------------------------------------------
    def _act(self, obs: np.ndarray) -> np.ndarray:
        n = obs.shape[0]
        a = self._noop(n)
        rng = self.rng

        yaw_now = obs[:, _AIM_YAW] * 180.0
        pitch_now = obs[:, _AIM_PITCH] * 90.0
        dist = obs[:, _DIST] * 8.0
        enemy_hurt = obs[:, _ENEMY_HURT] * 10.0
        on_ground = obs[:, _SELF_GROUND] > 0.5

        # reaction delay: aim uses errors from `reaction_ticks` ago
        if self.reaction_ticks > 0:
            self._err_buf[self._buf_i] = np.stack([yaw_now, pitch_now], 1)
            self._buf_i = (self._buf_i + 1) % self._err_buf.shape[0]
            delayed = self._err_buf[self._buf_i]
            yaw_err, pitch_err = delayed[:, 0], delayed[:, 1]
        else:
            yaw_err, pitch_err = yaw_now, pitch_now

        # bounded-rate aim (wurst-style) + tracking jitter
        cap = self.rot_speed_dps / TICK_RATE
        jit = rng.normal(0.0, self.aim_jitter_deg, n)
        a[:, YAW] = nearest_camera_bin(np.clip(yaw_err + jit, -cap, cap))
        a[:, PITCH] = nearest_camera_bin(
            np.clip(pitch_err + 0.5 * jit, -cap, cap))

        # distance management (PracticeBotPvP: back <2, engage 3-3.5, chase >4)
        a[:, FORWARD] = np.where(dist > self.engage_hi, 2,
                                 np.where(dist < self.engage_lo, 0, 1))
        a[:, SPRINT] = (dist > self.chase_dist).astype(np.int64)

        # strafing with random direction switches
        self.strafe_timer -= 1
        switch = self.strafe_timer <= 0
        if switch.any():
            self.strafe_dir[switch] = rng.choice([-1, 1], int(switch.sum()))
            self.strafing[switch] = rng.random(int(switch.sum())) < self.strafe_rate
            self.strafe_timer[switch] = rng.integers(10, 50, int(switch.sum()))
        engaged = (dist < self.chase_dist)
        a[:, STRAFE] = np.where(self.strafing & engaged,
                                np.where(self.strafe_dir > 0, 2, 0), 1)

        # natural hops
        self.hop_timer -= 1
        hop = (self.hop_timer <= 0) & on_ground
        if hop.any():
            self.hop_timer[hop] = rng.integers(20, 60, int(hop.sum()))

        # crit sequence: jump, attack while falling (ticks ~5-8 after)
        can_crit = (on_ground & (dist > 1.8) & (dist < self.engage_hi)
                    & (enemy_hurt <= 0.5) & (self.crit_phase == 0)
                    & (rng.random(n) < self.crit_rate * 0.04))
        self.crit_phase[can_crit] = 9
        in_crit = self.crit_phase > 0
        a[in_crit & (self.crit_phase == 9), JUMP] = 1
        a[hop & ~in_crit, JUMP] = 1
        self.crit_phase[in_crit] -= 1

        # clicking: CPS-capped, ray-aware (only swing when the crosshair is
        # actually on the hitbox), optionally timed to the hurt window
        self.click_cd = np.maximum(self.click_cd - 1, 0)
        half_w_ang = np.degrees(np.arctan2(0.35, np.maximum(dist, 0.5)))
        half_h_ang = np.degrees(np.arctan2(1.05, np.maximum(dist, 0.5)))
        aimed = (np.abs(yaw_now) < half_w_ang) & (np.abs(pitch_now) < half_h_ang)
        in_reach = dist < 3.05
        want = aimed & in_reach & (self.click_cd == 0)
        if self.hurt_timed_swing:
            want &= enemy_hurt <= 0.5
        if self.crit_rate > 0:
            crit_window = in_crit & (self.crit_phase <= 4) & ~on_ground
            want |= aimed & in_reach & crit_window & (self.click_cd == 0)
        a[want, ATTACK] = 1
        self.click_cd[want] = max(int(round(TICK_RATE / self.cps)), 1)

        # w-tap: after a hit lands (enemy hurt-time rising edge), release
        # forward for one tick to reset sprint, then re-press
        landed = (enemy_hurt > self.prev_hurt + 0.5) & (
            rng.random(n) < self.wtap_rate)
        self.prev_hurt = enemy_hurt.copy()
        self.wtap_phase[landed] = 2
        wt = self.wtap_phase > 0
        a[wt & (self.wtap_phase == 2), FORWARD] = 1   # release
        a[wt & (self.wtap_phase == 1), FORWARD] = 2   # re-press
        a[wt & (self.wtap_phase == 1), SPRINT] = 1
        self.wtap_phase[wt] -= 1

        return a


class PracticeEasy(PracticeBot):
    """PracticeBotPvP 'Easy' row: slow reactions, low CPS, rare techniques."""
    name = "P1-Easy"
    intended_elo = 1100
    rot_speed_dps = 120.0
    reaction_ticks = 6
    cps = 4.0
    crit_rate, wtap_rate, strafe_rate = 0.10, 0.10, 0.30
    aim_jitter_deg = 4.0


class PracticeMedium(PracticeBot):
    """'Medium': normal reactions, competitive CPS, frequent techniques."""
    name = "P2-Medium"
    intended_elo = 1300


class PracticeHard(PracticeBot):
    """'Hard': fast reactions, high CPS, near-constant techniques."""
    name = "P3-Hard"
    intended_elo = 1500
    rot_speed_dps = 480.0
    reaction_ticks = 1
    cps = 12.0
    crit_rate, wtap_rate, strafe_rate = 0.80, 0.90, 0.85
    aim_jitter_deg = 1.5


class PracticeHacker(PracticeBot):
    """'Hacker': 3x-faster-than-normal reaction, hurt-window-perfect swings,
    max technique rates.

    reaction_ticks = 0 (frame-perfect). A spec-reading argument ("3x faster
    than Medium's 3 ticks = 1") once softened this to 1 -- and Robbie then
    beat the port consistently in a fresh-equal-diamond duel (2026-07-23).
    The real practice-server hacker beats most humans; the react=0 port is
    the one that matched that feel (killed Robbie in ~6 s). Human-calibrated
    truth outranks the spec reading: 0 stands."""
    name = "P4-Hacker"
    intended_elo = 1700
    # OFFICIAL BAR (Robbie's calibration duels, 2026-07-24): frame-perfect
    # reaction kept, but rot 600 / jitter 1.5 / cps 11 -- the config he
    # judged to match the real practice-server hacker. The old full-power
    # port (720/0.5/13) was harder than the real thing; react:1 was weaker.
    rot_speed_dps = 600.0
    reaction_ticks = 0
    cps = 11.0
    crit_rate, wtap_rate, strafe_rate = 1.0, 1.0, 1.0
    aim_jitter_deg = 1.5
    hurt_timed_swing = True
    engage_lo, engage_hi = 2.0, 2.8   # optimal spacing


def practice_tiers(seed: int = 0) -> List[ScriptedPolicy]:
    return [
        PracticeEasy(seed + 10),
        PracticeMedium(seed + 11),
        PracticeHard(seed + 12),
        PracticeHacker(seed + 13),
    ]
