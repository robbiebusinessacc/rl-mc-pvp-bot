"""ObsAssembler: perception stream + own issued actions -> 48-float obs.

Called once per tick with the newest PERCEPTION_DIM estimate and the action
the bot issued on the previous tick. Maintains a small recurrent state
(velocity dead-reckoning, distance/pitch filters, hurt timers, finite
differences) and returns a full OBS_LAYOUT vector.

Per-slot source map (P = perception slot, A = dead-reckoned from issued
actions, E = recurrent estimate / finite difference, D = default):

    rel_pos              E  from P.aim_err_yaw/pitch angles + distance
                            (dist inverted from P.bbox_height)
    rel_vel              E  finite difference of the rel_pos estimate (EMA)
    self_vel             A  stub-physics integrator over issued move/jump/
                            sprint actions, rotated with issued yaw deltas
    self_pitch_sincos    A+P camera-pitch integrator from issued pitch bins,
                            corrected toward P.self_pitch (complementary)
    dist                 P  dist_from_bbox_height(P.bbox_height), EMA; held
                            when enemy not visible
    in_reach             E  dist_est < 3.0 and P.visible
    self_hurt            E  timer set to 10 when P.self_hp drops >= ~1 hp
    enemy_hurt           P  timer set to 10 on P.hurt_flash rising edge
    self_hp              P  direct
    enemy_hp             E  starts full; -1 hp per hurt_flash rising edge
                            (sword-hit heuristic; true value unobservable)
    self_on_ground       A  from the jump/gravity integrator
    enemy_on_ground      P  direct (thresholded)
    self_sprinting       A  sprint head of the issued action
    ticks_since_hit_dealt E reset on enemy_hurt rising edge, else +1 (cap 100)
    ticks_since_hit_taken E reset on self_hurt trigger, else +1 (cap 100)
    aim_err_yaw          P  direct (already /180)
    aim_err_pitch        P  direct (already /90)
    prev_action          A  issued action index per head, / head_size (spec;
                            note stub.py divides by a flat 10.0 instead)
    reserved             D  zeros
"""
import math
from typing import Optional

import numpy as np

from pvpbot.spec import (
    ACTION_HEAD_SIZES,
    CAMERA_BINS,
    NUM_ACTION_HEADS,
    OBS_DIM,
    OBS_LAYOUT,
    PERCEPTION_LAYOUT,
)
from pvpbot.perception.synth import dist_from_bbox_height

import os

_DIST_OFFSET = float(os.environ.get("PVPBOT_DIST_OFFSET", "0.0"))
_HP_MEDIAN = 9          # frames of self_hp median filter (kills read jitter)
# self_hp perception noise (std ~3hp) made single-frame HP differencing fire
# "I got hit" on 42%% of frames. In SIM training the policy dominates and
# almost never sees self_hurt active or self_hp below full, so its response
# to those channels is essentially untrained -- feeding live noise on them is
# pure OOD harm (measured 15-19 sigma). Trigger self_hurt only on a
# high-confidence multi-hp drop of the median-smoothed hp, matching the
# sim-typical "~never hurt" distribution the policy actually learned in.
_HURT_DROP_HP = 3.0

_P = PERCEPTION_LAYOUT

# stub-compatible movement constants (see pvpbot/sim/stub.py)
_FRICTION = 0.546
_ACCEL = 0.1
_SPRINT_MULT = 1.3
_JUMP_VY = 0.42
_GRAVITY = 0.08
_REACH = 3.0
EYE = 1.62       # camera eye height (matches build_dataset labels)
CENTER = 0.9     # enemy hitbox center height the aim labels target


class ObsAssembler:
    """Maps (perception estimates, issued actions) ticks to OBS_DIM vectors."""

    def __init__(self, dist_alpha: float = 0.5, vel_alpha: float = 0.2,
                 pitch_gain: float = 0.15):
        self.dist_alpha = dist_alpha    # EMA weight for fresh dist measurements
        self.vel_alpha = vel_alpha      # EMA weight for rel_vel differences
        self.pitch_gain = pitch_gain    # complementary-filter correction gain
        self.reset()

    def reset(self) -> None:
        self.tick = 0
        # A: dead-reckoned self state (self-yaw frame: [fwd, up, lat])
        self.self_vel = np.zeros(3, dtype=np.float32)
        self.self_y = 0.0               # height above ground (jump integrator)
        self.on_ground = True
        self.pitch_deg = 0.0
        self.sprinting = 0.0
        self.prev_action = np.zeros(NUM_ACTION_HEADS, dtype=np.int64)
        # E: enemy estimates
        self.dist_est = 4.0
        self.rel_pos = np.zeros(3, dtype=np.float32)
        self.rel_vel = np.zeros(3, dtype=np.float32)
        self.have_rel = False
        self.enemy_hp = 20.0
        self.enemy_unseen = 100     # ticks the enemy has been out of view
        self.enemy_hurt_t = 0
        self.self_hurt_t = 0
        # self_hp perception jitters ~1.4hp/frame (CNN reads the heart bar
        # off a 96x170 frame); differencing raw HP made "I got hit" fire on
        # 42%% of frames, pinning self_hurt/ticks_since_hit_taken permanently
        # ON -- a distribution the policy almost never saw in training
        # (measured sim-vs-live gap: 15-19 sigma). Median-filter the HP and
        # derive both the obs and hit-detection from the smoothed value.
        self._hp_hist = [20.0] * _HP_MEDIAN
        # perception under-reads full HP (~15/20 when actually full), so
        # calibrate the smoothed reading up toward the training-typical
        # full-HP distribution: track the running max as the "full" ref.
        self.smooth_hp = 20.0
        self.prev_smooth_hp = 20.0
        self.hp_ref = 20.0      # running estimate of the "full" reading
        self.prev_hurt_flash = False
        self.ticks_since_dealt = 100
        self.ticks_since_taken = 100

    # ------------------------------------------------------------------
    def _integrate_action(self, action: Optional[np.ndarray]) -> None:
        """Dead-reckon own kinematics from the issued action (stub physics)."""
        if action is None:
            action = np.zeros(NUM_ACTION_HEADS, dtype=np.int64)
            action[0] = 1  # forward head: none
            action[1] = 1  # strafe head: none
            action[5] = 5  # yaw bin: 0 deg
            action[6] = 5  # pitch bin: 0 deg
        action = np.asarray(action, dtype=np.int64).reshape(NUM_ACTION_HEADS)
        self.prev_action = action.copy()

        # camera: issued yaw delta rotates the self-yaw frame; issued pitch
        # delta integrates the pitch estimate (corrected by perception later)
        dyaw = CAMERA_BINS[int(action[5])]
        dpitch = CAMERA_BINS[int(action[6])]
        self.pitch_deg = float(np.clip(self.pitch_deg + dpitch, -90.0, 90.0))
        rad = math.radians(dyaw)
        c, s = math.cos(rad), math.sin(rad)
        # rotate horizontal velocity into the new frame (frame turned by dyaw)
        fwd, lat = self.self_vel[0], self.self_vel[2]
        self.self_vel[0] = c * fwd + s * lat
        self.self_vel[2] = -s * fwd + c * lat
        # same rotation for the enemy position estimate
        if self.have_rel:
            rx, rz = self.rel_pos[0], self.rel_pos[2]
            self.rel_pos[0] = c * rx + s * rz
            self.rel_pos[2] = -s * rx + c * rz

        # movement: mirror stub.py's crude integrator, in the self frame
        move_f = float(action[0]) - 1.0
        move_l = float(action[1]) - 1.0
        self.sprinting = float(action[3])
        mult = _ACCEL * (1.0 + 0.3 * self.sprinting)
        self.self_vel[0] = self.self_vel[0] * _FRICTION + move_f * mult
        self.self_vel[2] = self.self_vel[2] * _FRICTION + move_l * mult
        if action[2] == 1 and self.on_ground:
            self.self_vel[1] = _JUMP_VY
        else:
            self.self_vel[1] = self.self_vel[1] - _GRAVITY
        self.self_y = max(self.self_y + float(self.self_vel[1]), 0.0)
        self.on_ground = self.self_y <= 0.0
        if self.on_ground:
            self.self_vel[1] = max(self.self_vel[1], 0.0) * 0.0

    # ------------------------------------------------------------------
    def update(self, perception: np.ndarray,
               action: Optional[np.ndarray] = None) -> np.ndarray:
        """One tick: newest perception vector + previously issued action.

        perception: (PERCEPTION_DIM,) as produced by FrameEncoder.encode.
        action:     (NUM_ACTION_HEADS,) int indices issued last tick, or None.
        Returns a float32 (OBS_DIM,) vector per OBS_LAYOUT.
        """
        p = np.asarray(perception, dtype=np.float32).reshape(-1)
        self._integrate_action(action)
        self.tick += 1

        visible = p[_P["visible"]] > 0.5
        yaw_err_n = float(np.clip(p[_P["aim_err_yaw"]], -1.0, 1.0))
        # perception/synth aim_err_pitch is DOWN-positive ("+ = enemy below
        # crosshair"); the sim obs contract (env.py) is UP-positive. Flip
        # once here so everything downstream, including the OBS slot and the
        # rel_pos geometry, is in the sim frame.
        pitch_err_n = -float(np.clip(p[_P["aim_err_pitch"]], -1.0, 1.0))
        hp_raw = float(np.clip(p[_P["self_hp"]], 0.0, 1.0)) * 20.0
        self._hp_hist.append(hp_raw)
        if len(self._hp_hist) > _HP_MEDIAN:
            self._hp_hist.pop(0)
        self.smooth_hp = float(np.median(self._hp_hist))
        # calibrate against the running-max "full" reference so a systematic
        # under-read maps to ~full (sim self_hp is ~1.0); ref drifts up
        # slowly toward higher readings, decays very slowly otherwise
        self.hp_ref = max(self.hp_ref * 0.999, self.smooth_hp, 12.0)
        hp = min(self.smooth_hp / self.hp_ref * 20.0, 20.0)
        hurt_flash = p[_P["hurt_flash"]] > 0.5 and visible

        # pitch complementary filter: integrated pitch pulled toward the
        # perception measurement (P.self_pitch is /90). The CNN reads pitch
        # off the horizon, so its estimate is garbage when the horizon is
        # out of frame -- observed live: true pitch +90 (sky) while the CNN
        # said -10, which convinced the policy it was level and locked it
        # staring at the sky. The integrator is fed our OWN issued deltas
        # (exact), so only apply the perception correction when the horizon
        # is plausibly visible.
        # SIGN: perception self_pitch is DOWN-positive (MC/synth
        # convention); the integrator and the whole sim obs contract are
        # UP-positive. Blending the raw measurement inverted the belief
        # once v11's confident readings dominated (telemetry forensics
        # 2026-07-25: believed +44 while truly 44 DOWN; recovery then
        # commanded deeper down and the +/-55 clamp ate real up-commands
        # -> the "stares at the floor forever" lock). Convert here, the
        # one perception->sim boundary for this slot.
        meas_pitch = -float(np.clip(p[_P["self_pitch"]], -1.0, 1.0)) * 90.0
        if abs(self.pitch_deg) < 40.0:
            self.pitch_deg += self.pitch_gain * (meas_pitch - self.pitch_deg)

        # distance from bbox height (inverse perspective); hold when unseen.
        # PVPBOT_DIST_OFFSET subtracts a live-calibrated bias: perceived
        # dist measured +2.2 blocks long against server truth in combat
        # (sprint-FOV/clipping domain), which starved the click gate and
        # spacing. Set from truth-joined round data, not offline tails.
        if visible:
            d_meas = dist_from_bbox_height(float(p[_P["bbox_height"]]))
            d_meas = max(d_meas - _DIST_OFFSET, 0.5)
            self.dist_est += self.dist_alpha * (d_meas - self.dist_est)
        self.dist_est = float(np.clip(self.dist_est, 0.3, 30.0))

        # rel_pos from aim angles + distance (self-yaw frame [fwd, up, lat])
        prev_rel = self.rel_pos.copy()
        if visible:
            yaw_err_rad = math.radians(yaw_err_n * 180.0)
            total_pitch_deg = self.pitch_deg + pitch_err_n * 90.0
            d = self.dist_est
            # NOTE: rel_pos[1] intentionally uses the eye-to-chest line of
            # sight (NOT the sim's feet-to-feet dy). Adding the EYE-CENTER
            # offset to "match sim" destabilized the pitch loop live: while
            # the enemy is unseen the policy drives pitch toward the shifted
            # height, the enemy leaves the frame, and pitch runs away to the
            # clamp (observed: camera pegged looking up). The distribution
            # mismatch here is less harmful than the runaway, so keep the
            # line-of-sight form until the search-assist also stabilizes
            # pitch.
            rel = np.array([
                d * math.cos(yaw_err_rad),
                d * math.sin(math.radians(
                    float(np.clip(total_pitch_deg, -89.0, 89.0)))),
                d * math.sin(yaw_err_rad),
            ], dtype=np.float32)
            if self.have_rel:
                diff = rel - prev_rel          # blocks/tick
                self.rel_vel += self.vel_alpha * (diff - self.rel_vel)
            self.rel_pos = rel
            self.have_rel = True
        else:
            self.rel_vel *= 0.8                # decay when unseen

        # enemy_hp lifecycle: the live assembler is a session singleton that
        # is never reset, but enemy_hp only ever DECREMENTED on hits -- so
        # across a multi-duel session it ratcheted to 0 and stayed there,
        # convincing the policy the enemy was permanently near death (live
        # 0.19 vs sim 0.71, 1.8 sigma; sim resets it every episode). Fix:
        # reset to full when a fresh enemy reappears after a visibility gap
        # (a respawn or re-engage), and regen slowly like a real 1.8 player.
        if visible:
            if self.enemy_unseen > 20:      # was gone a while -> fresh enemy
                self.enemy_hp = 20.0
            self.enemy_unseen = 0
        else:
            self.enemy_unseen = min(self.enemy_unseen + 1, 1000)

        # hurt timers / hp estimates
        if hurt_flash and not self.prev_hurt_flash:
            self.enemy_hurt_t = 10
            self.enemy_hp = max(self.enemy_hp - 1.0, 0.0)
            self.ticks_since_dealt = 0
        else:
            self.enemy_hurt_t = max(self.enemy_hurt_t - 1, 0)
            self.ticks_since_dealt = min(self.ticks_since_dealt + 1, 100)
            if self.ticks_since_dealt > 40:   # not hitting -> enemy regens
                self.enemy_hp = min(self.enemy_hp + 0.05, 20.0)
        self.prev_hurt_flash = bool(hurt_flash)

        # hit only on a real drop of the SMOOTHED hp (raw jitter removed)
        if self.prev_smooth_hp - self.smooth_hp >= _HURT_DROP_HP:
            self.self_hurt_t = 10
            self.ticks_since_taken = 0
        else:
            self.self_hurt_t = max(self.self_hurt_t - 1, 0)
            self.ticks_since_taken = min(self.ticks_since_taken + 1, 100)
        self.prev_smooth_hp = self.smooth_hp

        # ------------------------------------------------------------------
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        s, e = OBS_LAYOUT["rel_pos"]
        obs[s:e] = self.rel_pos / 8.0
        s, e = OBS_LAYOUT["rel_vel"]
        obs[s:e] = self.rel_vel
        s, e = OBS_LAYOUT["self_vel"]
        obs[s:e] = self.self_vel
        s, e = OBS_LAYOUT["self_pitch_sincos"]
        pr = math.radians(self.pitch_deg)
        obs[s:e] = (math.sin(pr), math.cos(pr))
        s, e = OBS_LAYOUT["dist"]
        obs[s] = self.dist_est / 8.0
        s, e = OBS_LAYOUT["in_reach"]
        obs[s] = 1.0 if (visible and self.dist_est < _REACH) else 0.0
        s, e = OBS_LAYOUT["self_hurt"]
        obs[s] = self.self_hurt_t / 10.0
        s, e = OBS_LAYOUT["enemy_hurt"]
        obs[s] = self.enemy_hurt_t / 10.0
        s, e = OBS_LAYOUT["self_hp"]
        obs[s] = hp / 20.0
        s, e = OBS_LAYOUT["enemy_hp"]
        obs[s] = self.enemy_hp / 20.0
        s, e = OBS_LAYOUT["self_on_ground"]
        obs[s] = 1.0 if self.on_ground else 0.0
        s, e = OBS_LAYOUT["enemy_on_ground"]
        obs[s] = 1.0 if p[_P["enemy_on_ground"]] > 0.5 else 0.0
        s, e = OBS_LAYOUT["self_sprinting"]
        obs[s] = self.sprinting
        s, e = OBS_LAYOUT["ticks_since_hit_dealt"]
        obs[s] = self.ticks_since_dealt / 100.0
        s, e = OBS_LAYOUT["ticks_since_hit_taken"]
        obs[s] = self.ticks_since_taken / 100.0
        s, e = OBS_LAYOUT["aim_err_yaw"]
        obs[s] = yaw_err_n if visible else 0.0
        s, e = OBS_LAYOUT["aim_err_pitch"]
        obs[s] = pitch_err_n if visible else 0.0
        s, e = OBS_LAYOUT["enemy_visible"]
        obs[s] = 1.0 if visible else 0.0
        s, e = OBS_LAYOUT["prev_action"]
        sizes = np.asarray(ACTION_HEAD_SIZES, dtype=np.float32)
        obs[s:e] = self.prev_action.astype(np.float32) / sizes
        # "reserved" stays zero
        return obs


# ---------------------------------------------------------------------------
# Module-level convenience for the deploy loop (probes for ``assemble_obs``).
# One process-global recurrent assembler; the live loop is single-stream.
# ---------------------------------------------------------------------------
_DEFAULT_ASSEMBLER = None


def assemble_obs(perception: np.ndarray, prev_action: np.ndarray) -> np.ndarray:
    global _DEFAULT_ASSEMBLER
    if _DEFAULT_ASSEMBLER is None:
        _DEFAULT_ASSEMBLER = ObsAssembler()
    return _DEFAULT_ASSEMBLER.update(perception, prev_action)
