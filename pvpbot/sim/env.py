"""Vectorized Minecraft 1.8-style sword-duel physics environment.

Drop-in replacement for ``pvpbot.sim.stub.DuelVecEnv``: same class name,
same API (see the docstring at the bottom of ``pvpbot/spec.py``), real
1.8 movement/combat mechanics, fully vectorized in NumPy across
``num_envs`` parallel duels with two players each on a flat arena.

Conventions (chosen to match the stub, not raw Minecraft data values --
only internal consistency matters):
  * yaw in degrees [0, 360); horizontal facing vector is (cos yaw, 0, sin yaw)
  * pitch in degrees [-90, 90], positive = looking up
  * the floor is the plane y = 0, +y is up; arena is a disc of
    ``SimConfig.arena_radius`` around the origin (positions are clamped
    to the disc -- a simple "arena wall")

Per-tick update order follows 1.8 ``EntityLivingBase.onLivingUpdate`` /
``moveEntityWithHeading``:
  camera -> sprint state -> jump -> movement acceleration -> integrate
  position (floor + wall collision) -> gravity + drag/friction -> combat
  (hits, damage, knockback) -> rewards/termination -> auto-reset -> obs.

Documented simplifications vs. real 1.8 (see also the test suite):
  * Flat featureless arena: no blocks, no fall damage, LOS always clear.
  * Hit detection approximates ray-vs-AABB with a generous per-axis
    angular cone against the 0.6 x 1.8 hitbox (padded + slack degrees),
    plus the real eye-to-AABB 3.0-block reach test.
  * Hurt-time is a flat 10-tick invulnerability window (1.8's
    hurtResistantTime > maxHurtResistantTime/2 rule, without the
    "extra damage still applies" corner case).
  * Knockback is applied at end-of-tick (after friction) instead of
    mid-entity-update; no knockback-resistance random roll.
  * Taking damage breaks sprint, and both a landed sprint-hit and taken
    damage set a "blocked" latch that requires releasing forward/sprint
    for a tick before sprint can re-engage (the W-tap mechanic).
  * No hunger, potions, enchantments, sword blocking, fishing rods, or
    random damage spread; armor is a flat 1.8-style percentage
    (defense points * 4% reduction), no durability.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from pvpbot.spec import (
    ACTION_HEAD_SIZES,
    CAMERA_BINS,
    NUM_ACTION_HEADS,
    OBS_DIM,
    OBS_LAYOUT,
)

_DEG2RAD = np.float32(np.pi / 180.0)
_PI = np.float32(np.pi)
_TWO_PI = np.float32(2.0 * np.pi)
_INV_PI = np.float32(1.0 / np.pi)        # radians -> "degrees/180" norm
_INV_HALF_PI = np.float32(2.0 / np.pi)   # radians -> "degrees/90" norm

# Slot start index per OBS_LAYOUT entry (layout is the contract; we index
# through it so a layout change on main fails loudly here, not silently).
_S = {name: se[0] for name, se in OBS_LAYOUT.items()}


@dataclass
class SimConfig:
    """Physics / combat / episode constants. Defaults are Minecraft 1.8."""

    # --- movement ---------------------------------------------------------
    ground_slipperiness: float = 0.6   # default block slipperiness; ground
    #   friction multiplier each tick is slipperiness * 0.91 = 0.546
    air_drag_h: float = 0.91           # horizontal drag while airborne
    y_drag: float = 0.98               # vertical drag applied after gravity
    gravity: float = 0.08              # blocks/tick^2
    walk_speed: float = 0.1            # movementSpeed attribute
    air_accel: float = 0.02            # jumpMovementFactor (air acceleration)
    sprint_mult: float = 1.3           # sprint = +30% movement speed
    input_scale: float = 0.98          # keyboard input magnitude (1.8 client)
    jump_velocity: float = 0.42        # motionY set on jump
    sprint_jump_boost: float = 0.2     # horizontal boost along yaw on sprint jump

    # --- combat -----------------------------------------------------------
    reach: float = 3.0                 # blocks, from eye to target AABB
    eye_height: float = 1.62           # 1.8 player eye height
    hitbox_half_width: float = 0.3     # player AABB is 0.6 wide
    hitbox_height: float = 1.8         # ... and 1.8 tall
    attack_damage: float = 8.0         # 1.8 diamond sword (7.0 is the 1.9
                                       # rebalance; 8 confirmed on a real
                                       # 1.8.8 server: 1.6 hp through 80% armor)
    armor_points: float = 20.0         # full diamond armor = 20 defense points
    armor_reduction_per_point: float = 0.04  # 1.8: 4%/point -> 80% reduction
    crit_mult: float = 1.5             # falling & airborne = critical hit
    hurt_ticks: int = 10               # post-hit invulnerability window
    kb_horizontal: float = 0.4         # base knockback along attacker->target
    kb_vertical: float = 0.4           # base vertical knockback ...
    kb_vertical_cap: float = 0.4       # ... capped at 0.4 (1.8 knockBack)
    # sprint-hit extras, additive on top of the base knockback. Calibrated
    # against a real 1.8.8 trace: measured launch 0.9 h (0.4+0.5, then one
    # ground-friction bite 0.546 before pure 0.91 air decay -- which the
    # start-of-tick friction rule reproduces) and 0.46 v (first rise 0.461,
    # 4.94-block total slide, 1.49-block pop).
    sprint_kb_horizontal: float = 0.5
    sprint_kb_vertical: float = 0.06
    sprint_hit_slowdown: float = 0.6   # attacker motionX/Z *= 0.6 on sprint hit
    aim_pad: float = 0.15              # extra effective hitbox radius (generous)
    aim_slack_deg: float = 10.0        # extra cone half-angle in degrees
    # clickable box: Entity.getCollisionBorderSize() expands the target AABB
    # by 0.1 on every axis for entity picking (EntityRenderer.getMouseOver,
    # MCP-919 1.8.9 source) -- the true clickable half-width is 0.4, not 0.3
    click_border: float = 0.1

    # --- observation realism (train like you deploy) -----------------------
    # Live, the policy only knows about the enemy through a camera + CNN:
    # enemy-derived obs exist only inside the view cone, and are noisy.
    # Training with X-ray obs taught the policy to never look at the target
    # (observed live 2026-07-21). Gating emulates pvpbot.perception.adapter:
    # aim errors zero when unseen, rel_pos/dist hold last-seen, rel_vel
    # decays 0.8/tick, in_reach/enemy_on_ground gate to 0.
    fov_limited_obs: bool = False      # enable the camera-cone gating
    fov_h_deg: float = 120.0           # horizontal cone (client FOV 90 -> ~122)
    fov_v_deg: float = 90.0            # vertical cone
    obs_max_dist: float = 16.0         # perception range limit
    obs_noise: float = 0.0             # noise scale; 1.0 = measured real MAEs
                                       # (yaw ~14 deg, pitch ~8, dist ~12%,
                                       # visible flake ~4%/tick)
    # live sensing lags ~3 ticks behind server truth (client renders remote
    # entities ~150 ms late + capture latency); a policy trained on fresh
    # obs loses trades it would win with learned lead/timing compensation
    obs_delay_ticks: int = 0
    # aim-execution jitter (deg, OU-correlated) added to the LEARNER's camera
    # each tick -- models the live perception->aim-assist->pixel-injection
    # error (~5 deg). Without it the sim rewards a perfect-aim knockback-kite
    # that fails live. 0 = perfect aim (old behavior).
    aim_exec_noise_deg: float = 0.0
    # live pixel-pipeline artifacts on the LEARNER's obs (flight-log matched):
    # rel_vel from frame-differencing is jitter-dominated (measured sigma
    # ~0.35 after adapter smoothing vs ~0.09 sim-true), and rel_pos[1] carries
    # the eye-vs-box-center vertical offset residual (~-0.16).
    rel_vel_noise: float = 0.0         # stationary sigma, OU-correlated
    rel_pos_y_bias: float = 0.0        # constant additive bias
    # live ACTION-channel artifacts (learner side only). The live round at
    # 37.5B proved the obs side faithful (top OOD z=1.9) while outcomes
    # inverted (sim 99% -> live 0-7): the policy "PWMs" movement bins each
    # 50 ms tick, which instant-velocity sim physics honors but the real
    # client (frame-sampled held keys + momentum + sprint reset) cannot --
    # live median speed was 0.00 m/s vs sim 2.16 while MEANS matched.
    act_delay_ticks: int = 0           # inject->client-frame->server lag
    act_hold_ticks: int = 0            # movement/sprint bins must persist
                                       # this many ticks to take effect
    # live click->hit conversion is lossy even with the crosshair VERIFIED
    # on the box (render-vs-server offset on a strafing target + CNN box
    # error): measured 33% on-box conversion in round 4 vs this sim's
    # deterministic ray. Learner side only -- the scripted opponent keeps
    # full registration, which errs conservative.
    hit_reg_prob: float = 1.0
    # the CNN misses a truly-on-screen enemy some fraction of frames
    # (v9 recall 0.73 -> ~0.27 misses); the sim's near-perfect FOV flag
    # is fantasy without it. Learner side only; missed ticks fall back to
    # the last-seen cache exactly like FOV exits.
    # With vis_reacq_prob > 0 this becomes a TWO-STATE tracked/lost process
    # (vis_miss_prob = per-tick p(lose track), vis_reacq_prob = per-tick
    # p(reacquire)): live blindness comes in runs (fitted mean 4.5 ticks
    # blind / 4.3 visible), and IID misses can't match both dwell times --
    # the policy must learn what to DO during a multi-tick dropout.
    vis_miss_prob: float = 0.0
    vis_reacq_prob: float = 0.0
    # deploy runs policy + aim/search assist as ONE closed-loop system
    # (deploy/loop.py); training the raw policy alone was a systematic
    # evaluator mismatch (raw sim 5% vs deployed-system live 44% on the
    # same weights). With cam_assist the env overrides the learner's
    # camera heads exactly like deploy: visible -> home on the perceived
    # aim error; unseen -> sweep toward held rel_pos (+lead) and recover
    # pitch toward level. PPO stays consistent: the buffer stores the
    # policy's own sampled camera actions; the override is env dynamics
    # (same precedent as act_hold_ticks).
    cam_assist: bool = False
    # zero the learner's enemy_hurt/enemy_hp/ticks_since_hit_dealt
    # obs (live these derive from the blind hurt-flash head)
    mask_dead_senses: bool = False

    # --- episode ----------------------------------------------------------
    arena_radius: float = 8.0          # hard wall (position clamp); disc
    #   radius, or HALF-SIDE when arena_square (live arena = 14x14 walled
    #   square: corners exist, and corners kill backpedal policies)
    arena_square: bool = False
    spawn_radius: float = 6.0          # spawn uniformly in this disc
    # player-player shove strength per tick; 0.10 = vanilla's 0.05 applied
    # twice (once from each entity's update loop). Set 0.0 to disable, e.g.
    # when replaying mineflayer ground truth (no client-side pushing there).
    collision_push: float = 0.10
    max_ticks: int = 1200              # 60 s timeout at 20 tps
    max_hp: float = 20.0


@dataclass
class RewardConfig:
    """Per-side reward weights."""

    hit: float = 1.0                   # landed a hit
    hurt: float = -1.0                 # took a hit
    win: float = 10.0                  # opponent died
    loss: float = -10.0                # died
    aim_coeff: float = 0.01            # -coeff * (|yaw err|/180 + |pitch err|/90)
    sprint_toggle_coeff: float = 0.005  # sprint-key toggle spam penalty
    # bait-and-punish shaping (0 = off): extra credit for a hit landed
    # while the OPPONENT is committed to closing -- moving toward us fast
    # through the outer engage band, where it cannot instantly disengage.
    # Motivated by the 0-20 official-bar live probe (Jul 25): the hacker
    # wins via distance control; its only exploitable state is the
    # committed approach. Symmetric across sides (self-play-safe).
    punish_close_bonus: float = 0.0
    # swing-punish shaping (0 = off): extra credit for a hit landed
    # within _SWING_TICKS of the OPPONENT's swing. Robbie's recorded
    # stage-4 duels: 79% of his hits within 400ms of the bot's swing,
    # median 151ms -- humans punish the swing, not just the approach.
    punish_swing_bonus: float = 0.0


class DuelVecEnv:
    """1.8 sword-duel physics, vectorized over (num_envs, 2 players)."""

    REACH = 3.0
    MAX_TICKS = 1200  # 60s episode cap (class attrs kept for stub parity)
    # opponent-commitment window (bait-and-punish shaping): closing
    # faster than _COMMIT_V inside the approach band arms a _COMMIT_TICKS
    # punish window on the other side. Units are STORED (post-friction)
    # motion, not displacement: sustained sprint converges to ~0.146
    # there (~0.25 blocks/tick displacement), walking to ~0.10 -- so 0.11
    # reads "sprint-committed" from ~3 ticks of a real chase.
    _COMMIT_V = 0.11
    _COMMIT_LO, _COMMIT_HI = 1.8, 3.6
    _COMMIT_TICKS = 6
    _SWING_TICKS = 8   # punish window after the opponent's swing (400ms)

    def __init__(
        self,
        num_envs: int,
        seed: int = 0,
        config: Optional[SimConfig] = None,
        reward: Optional[RewardConfig] = None,
    ):
        self.num_envs = num_envs
        self.cfg = config if config is not None else SimConfig()
        self.rew_cfg = reward if reward is not None else RewardConfig()
        self.rng = np.random.default_rng(seed)
        self.REACH = self.cfg.reach
        self.MAX_TICKS = self.cfg.max_ticks
        self._precompute()
        self._alloc()

    # ------------------------------------------------------------------
    def _precompute(self) -> None:
        """Bake config into float32 scalars so the hot path stays float32."""
        cfg = self.cfg
        f32 = np.float32
        self._bins = np.asarray(CAMERA_BINS, dtype=np.float32)
        self._inv_heads = (1.0 / np.asarray(ACTION_HEAD_SIZES)).astype(np.float32)

        # ground friction each tick = slipperiness * 0.91 (1.8), => 0.546
        fric_g = cfg.ground_slipperiness * 0.91
        self._fric_ground = f32(fric_g)
        self._fric_air = f32(cfg.air_drag_h)
        # 1.8 moveEntityWithHeading ground accel: speed * 0.16277136 / f^3
        # (numerically ~= walk_speed for default slipperiness)
        self._acc_ground = f32(cfg.walk_speed * 0.16277136 / fric_g ** 3)
        self._acc_air = f32(cfg.air_accel)
        self._sprint_mult = f32(cfg.sprint_mult)
        self._input = f32(cfg.input_scale)
        self._gravity = f32(cfg.gravity)
        self._ydrag = f32(cfg.y_drag)
        self._jump_v = f32(cfg.jump_velocity)
        self._sj_boost = f32(cfg.sprint_jump_boost)

        self._reach = f32(cfg.reach)
        self._eye_h = f32(cfg.eye_height)
        self._half_w = f32(cfg.hitbox_half_width)
        self._box_h = f32(cfg.hitbox_height)
        self._click_hw = f32(cfg.hitbox_half_width + cfg.click_border)
        self._click_b = f32(cfg.click_border)
        # aim reference point: hitbox centre ("chest"), 0.9 above the feet
        self._eye_to_chest = f32(0.5 * cfg.hitbox_height - cfg.eye_height)
        armor_mult = 1.0 - cfg.armor_reduction_per_point * cfg.armor_points
        self._dmg_base = f32(cfg.attack_damage * armor_mult)
        self._dmg_crit = f32(cfg.attack_damage * cfg.crit_mult * armor_mult)
        self._hurt_ticks = int(cfg.hurt_ticks)
        self._kb_h = f32(cfg.kb_horizontal)
        self._kb_v = f32(cfg.kb_vertical)
        self._kb_cap = f32(cfg.kb_vertical_cap)
        self._skb_h = f32(cfg.sprint_kb_horizontal)
        self._skb_v = f32(cfg.sprint_kb_vertical)
        self._push = f32(cfg.collision_push)
        self._sprint_slow = f32(cfg.sprint_hit_slowdown)
        self._aim_hw = f32(cfg.hitbox_half_width + cfg.aim_pad)
        self._aim_hh = f32(0.5 * cfg.hitbox_height + cfg.aim_pad)
        self._slack = f32(np.deg2rad(cfg.aim_slack_deg))

        self._arena_r = f32(cfg.arena_radius)
        self._arena_square = bool(cfg.arena_square)
        self._max_ticks = int(cfg.max_ticks)
        self._max_hp = f32(cfg.max_hp)

        self._fov_gate = bool(cfg.fov_limited_obs)
        self._obs_delay = int(cfg.obs_delay_ticks)
        self._aim_exec_nz = float(cfg.aim_exec_noise_deg)
        self._rv_nz = float(cfg.rel_vel_noise)
        self._rpy_bias = f32(cfg.rel_pos_y_bias)
        self._act_delay = int(cfg.act_delay_ticks)
        self._act_hold = int(cfg.act_hold_ticks)
        self._hit_reg_p = float(cfg.hit_reg_prob)
        self._vis_miss = float(cfg.vis_miss_prob)
        self._vis_reacq = float(cfg.vis_reacq_prob)
        self._cam_assist = bool(cfg.cam_assist)
        self._mask_dead_senses = bool(cfg.mask_dead_senses)
        self._aim_bias = np.zeros(self.num_envs, dtype=np.float32)
        self._delay_buf = None   # (delay, n, 2, OBS_DIM) ring, lazy
        self._delay_i = 0
        self._fov_hy = f32(np.deg2rad(cfg.fov_h_deg / 2.0))
        self._fov_hp = f32(np.deg2rad(cfg.fov_v_deg / 2.0))
        self._obs_maxd = f32(cfg.obs_max_dist)
        self._obs_nz = f32(cfg.obs_noise)

    def _alloc(self) -> None:
        n = self.num_envs
        self.pos = np.zeros((n, 2, 3), dtype=np.float32)
        self.vel = np.zeros((n, 2, 3), dtype=np.float32)
        self.yaw = np.zeros((n, 2), dtype=np.float32)
        self.pitch = np.zeros((n, 2), dtype=np.float32)
        self.hp = np.full((n, 2), self._max_hp, dtype=np.float32)
        self.hurt = np.zeros((n, 2), dtype=np.int32)
        self.on_ground = np.ones((n, 2), dtype=bool)
        self.sprinting = np.zeros((n, 2), dtype=bool)
        self.sprint_blocked = np.zeros((n, 2), dtype=bool)
        self.last_dmg = np.zeros((n, 2), dtype=np.float32)
        self.since_dealt = np.full((n, 2), 100.0, dtype=np.float32)
        self.since_taken = np.full((n, 2), 100.0, dtype=np.float32)
        self.ticks = np.zeros(n, dtype=np.int32)
        self.prev_actions = np.zeros((n, 2, NUM_ACTION_HEADS), dtype=np.int64)
        self._obs_buf = np.zeros((n, 2, OBS_DIM), dtype=np.float32)
        # ungated omniscient obs (for scripted/mineflayer-style opponents)
        self._obs_ungated = np.zeros((n, 2, OBS_DIM), dtype=np.float32)
        self._relvel_ou = np.zeros((n, 3), dtype=np.float32)
        self._vis_lost = np.zeros(n, dtype=bool)
        self._last_obs0 = None
        # learner action-channel state: delay ring buffer + move-hold latch
        cam0 = int(np.argmin(np.abs(self._bins)))
        self._act_neutral = np.array([1, 1, 0, 0, 0, cam0, cam0], dtype=np.int64)
        if self._act_delay > 0:
            self._act_queue = np.tile(self._act_neutral, (self._act_delay, n, 1))
            self._aq_pos = 0
        # latched heads: 0=forward 1=strafe 3=sprint (jump/attack/camera are
        # tap- or event-driven live and DO register per tick)
        self._mv_eff = np.tile(np.array([1, 1, 0], dtype=np.int64), (n, 1))
        self._mv_pend = self._mv_eff.copy()
        self._mv_cnt = np.zeros((n, 3), dtype=np.int32)
        # last-seen caches for FOV-gated obs (normalized units, like the obs)
        self._seen_rel = np.zeros((n, 2, 3), dtype=np.float32)
        self._seen_relvel = np.zeros((n, 2, 3), dtype=np.float32)
        self._seen_dist = np.full((n, 2), 2.0, dtype=np.float32)  # 16/8
        # OU-correlated perception bias per env/side: [yaw, pitch] normalized.
        # The real CNN's aim error is biased in the same direction for
        # seconds at a time (temporally correlated), which iid noise never
        # taught the policy to survive (live: 39% camera-on-target vs a
        # 500-0-0 ladder sweep in-sim).
        self._noise_bias = np.zeros((n, 2, 2), dtype=np.float32)
        # opponent-commitment punish timers per side (bait-and-punish)
        self._commit = np.zeros((n, 2), dtype=np.int32)
        # opponent-swing punish timers per side (swing-punish shaping)
        self._opp_swing = np.zeros((n, 2), dtype=np.int32)
        # cam-assist search state (mirrors deploy: hold 6 ticks, then scan)
        self._cam_unseen = np.zeros(n, dtype=np.int32)
        self._scan_dir = np.ones(n, dtype=np.float32)

    # ------------------------------------------------------------------
    def _reset_envs(self, mask: np.ndarray) -> None:
        k = int(mask.sum())
        if k == 0:
            return
        # uniform in the spawn disc: sqrt-uniform radius, uniform angle
        ang = self.rng.uniform(0.0, 2.0 * np.pi, (k, 2))
        rad = self.cfg.spawn_radius * np.sqrt(self.rng.uniform(0.0, 1.0, (k, 2)))
        p = np.zeros((k, 2, 3), dtype=np.float32)
        p[:, :, 0] = rad * np.cos(ang)
        p[:, :, 2] = rad * np.sin(ang)
        self.pos[mask] = p
        self.vel[mask] = 0.0
        self.yaw[mask] = self.rng.uniform(0.0, 360.0, (k, 2)).astype(np.float32)
        self.pitch[mask] = 0.0
        self.hp[mask] = self._max_hp
        self.hurt[mask] = 0
        self.on_ground[mask] = True
        self.sprinting[mask] = False
        self.sprint_blocked[mask] = False
        self.last_dmg[mask] = 0.0
        self.since_dealt[mask] = 100.0
        self.since_taken[mask] = 100.0
        self.ticks[mask] = 0
        self.prev_actions[mask] = 0
        self._seen_rel[mask] = 0.0
        self._seen_relvel[mask] = 0.0
        self._seen_dist[mask] = 2.0   # "unknown / far" (16 blocks, /8)
        self._noise_bias[mask] = 0.0
        self._commit[mask] = 0
        self._opp_swing[mask] = 0
        self._cam_unseen[mask] = 0
        self._scan_dir[mask] = 1.0
        self._aim_bias[mask] = 0.0
        self._relvel_ou[mask] = 0.0
        self._vis_lost[mask] = False
        if self._act_delay > 0:
            self._act_queue[:, mask] = self._act_neutral
        self._mv_eff[mask] = np.array([1, 1, 0], dtype=np.int64)
        self._mv_pend[mask] = self._mv_eff[mask]
        self._mv_cnt[mask] = 0

    def reset(self) -> np.ndarray:
        self._reset_envs(np.ones(self.num_envs, dtype=bool))
        fresh = self._obs()
        if self._obs_delay > 0:
            self._delay_buf = np.repeat(fresh[None], self._obs_delay, 0).copy()
            self._delay_i = 0
        return fresh

    def _delayed(self, fresh: np.ndarray, reset_mask: np.ndarray) -> np.ndarray:
        """Emit obs from ``obs_delay_ticks`` ago (live sensing lags server
        truth ~3 ticks). Freshly reset envs see their new episode's first
        obs immediately -- no cross-episode leakage."""
        if self._obs_delay <= 0:
            return fresh
        if self._delay_buf is None:
            self._delay_buf = np.repeat(fresh[None], self._obs_delay, 0).copy()
        if reset_mask.any():
            self._delay_buf[:, reset_mask] = fresh[reset_mask]
        out = self._delay_buf[self._delay_i].copy()
        self._delay_buf[self._delay_i] = fresh
        self._delay_i = (self._delay_i + 1) % self._obs_delay
        return out

    # ------------------------------------------------------------------
    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        a = np.asarray(actions)
        assert a.shape == (self.num_envs, 2, NUM_ACTION_HEADS)
        if self._act_delay > 0 or self._act_hold > 0:
            a = a.copy()
            x = a[:, 0]
            if self._act_delay > 0:
                # ring buffer: effective action is the one issued D ticks ago
                eff = self._act_queue[self._aq_pos].copy()
                self._act_queue[self._aq_pos] = x
                self._aq_pos = (self._aq_pos + 1) % self._act_delay
                x = eff
            if self._act_hold > 0:
                # a changed movement/sprint bin takes effect only after
                # persisting act_hold_ticks -- no per-tick velocity PWM
                for k, head in enumerate((0, 1, 3)):
                    raw = x[:, head]
                    same = raw == self._mv_pend[:, k]
                    self._mv_cnt[:, k] = np.where(same, self._mv_cnt[:, k] + 1, 1)
                    self._mv_pend[:, k] = raw
                    commit = self._mv_cnt[:, k] >= self._act_hold
                    self._mv_eff[commit, k] = raw[commit]
                    x[:, head] = self._mv_eff[:, k]
            a[:, 0] = x
        if self._cam_assist and self._last_obs0 is not None:
            # mirrors the DEPLOYED assist exactly (train/deploy parity,
            # retuned 2026-07-25): damped homing (0.5 x err, capped) on
            # fresh sightings; short hold-aim on the last bearing, then a
            # deterministic 15 deg/tick scan once holds go stale.
            o0 = self._last_obs0
            vis0 = o0[:, _S["enemy_visible"]] > 0.5
            self._cam_unseen = np.where(vis0, 0, self._cam_unseen + 1)
            fwd = o0[:, _S["rel_pos"]] + 4.0 * o0[:, _S["rel_vel"]]
            lat = o0[:, _S["rel_pos"] + 2] + 4.0 * o0[:, _S["rel_vel"] + 2]
            sweep = np.degrees(np.arctan2(lat, np.where(np.abs(fwd) < 1e-6, 1e-6, fwd)))
            sweep = np.clip(sweep, -30.0, 30.0)
            self._scan_dir = np.where(vis0 | (self._cam_unseen <= 6),
                                      np.where(sweep >= 0, 1.0, -1.0),
                                      self._scan_dir)
            search = np.where(self._cam_unseen <= 6, sweep,
                              15.0 * self._scan_dir)
            yaw_t = np.where(
                vis0,
                np.clip(0.5 * o0[:, _S["aim_err_yaw"]] * 180.0, -20.0, 20.0),
                search).astype(np.float32)
            pitch_cur = np.degrees(np.arctan2(
                o0[:, _S["self_pitch_sincos"]], o0[:, _S["self_pitch_sincos"] + 1]))
            pitch_t = np.where(
                vis0,
                np.clip(0.5 * o0[:, _S["aim_err_pitch"]] * 90.0, -15.0, 15.0),
                np.clip(-pitch_cur, -30.0, 30.0)).astype(np.float32)
            if a.base is not None or not a.flags.writeable:
                a = a.copy()
            a[:, 0, 5] = np.argmin(np.abs(self._bins[None, :] - yaw_t[:, None]), 1)
            a[:, 0, 6] = np.argmin(np.abs(self._bins[None, :] - pitch_t[:, None]), 1)
        rc = self.rew_cfg

        # -- 1) camera: yaw/pitch from binned deltas, before movement ------
        self.yaw = (self.yaw + self._bins[a[:, :, 5]]) % np.float32(360.0)
        self.pitch = np.clip(self.pitch + self._bins[a[:, :, 6]], -90.0, 90.0)
        # aim-EXECUTION error: the sim used to land the learner's clicks with
        # pixel-perfect aim, so it learned a knockback-kite needing perfect
        # aim it CANNOT deliver live (perception -> aim-assist -> pixel
        # injection lands ~5 deg off). That single gap explains the whole
        # sim(99%)->live(25%) collapse: +3 deg jitter alone drops sim winrate
        # to ~49%, +6 deg to ~6%. Add OU-correlated aim jitter to side 0 (the
        # learner) so the sim is faithful AND training forces an aim-robust
        # strategy. Side 1 (scripted mineflayer P4) keeps its near-perfect
        # aim; a self-play side-1 fights the noised learner, which is fine.
        if self._aim_exec_nz > 0.0:
            self._aim_bias *= 0.9
            self._aim_bias += self.rng.normal(
                0.0, self._aim_exec_nz * 0.44, size=self.num_envs).astype(np.float32)
            self.yaw[:, 0] = (self.yaw[:, 0] + self._aim_bias) % np.float32(360.0)
        yaw_rad = self.yaw * _DEG2RAD
        cos_y = np.cos(yaw_rad)
        sin_y = np.sin(yaw_rad)

        # -- 2) sprint state (W-tap latch) ---------------------------------
        # sprint requires the sprint key AND forward held; a "blocked" latch
        # (set by sprint-hits landed and damage taken) clears only once the
        # player releases forward or sprint for a tick -- the W-tap.
        want = (a[:, :, 3] == 1) & (a[:, :, 0] == 2)
        self.sprint_blocked &= want
        self.sprinting = want & ~self.sprint_blocked

        # -- 3) jump (1.8 jump(): motionY = 0.42, sprint adds 0.2 fwd) -----
        jump = (a[:, :, 2] == 1) & self.on_ground
        self.vel[:, :, 1] = np.where(jump, self._jump_v, self.vel[:, :, 1])
        sj = (jump & self.sprinting).astype(np.float32) * self._sj_boost
        self.vel[:, :, 0] += cos_y * sj
        self.vel[:, :, 2] += sin_y * sj

        # -- 4) movement acceleration (1.8 moveFlying) ---------------------
        # inputs in {-0.98, 0, +0.98}; diagonal input is renormalized so it
        # never accelerates faster than straight (f = accel / max(|in|, 1))
        fwd = (a[:, :, 0] - 1).astype(np.float32) * self._input
        strafe = (a[:, :, 1] - 1).astype(np.float32) * self._input
        mag2 = fwd * fwd + strafe * strafe
        accel = np.where(self.on_ground, self._acc_ground, self._acc_air)
        accel = np.where(self.sprinting, accel * self._sprint_mult, accel)
        scale = np.where(
            mag2 >= 1e-4, accel / np.maximum(np.sqrt(mag2), 1.0), 0.0
        )
        self.vel[:, :, 0] += (fwd * cos_y - strafe * sin_y) * scale
        self.vel[:, :, 2] += (fwd * sin_y + strafe * cos_y) * scale

        # -- 5) integrate position; floor + arena-wall collision -----------
        # vanilla picks the tick's horizontal friction from the on-ground
        # state BEFORE the move: the jump tick still gets ground friction
        # (eating most of the sprint-jump boost) and the landing tick still
        # gets air drag -- validated against a real 1.8 server trace
        was_grounded = self.on_ground
        self.pos += self.vel
        on_floor = self.pos[:, :, 1] <= 0.0
        self.pos[:, :, 1] = np.where(on_floor, 0.0, self.pos[:, :, 1])
        self.on_ground = on_floor
        if self._arena_square:
            # live-matched square arena: axis-independent clamp -> corners
            np.clip(self.pos[:, :, 0], -self._arena_r, self._arena_r,
                    out=self.pos[:, :, 0])
            np.clip(self.pos[:, :, 2], -self._arena_r, self._arena_r,
                    out=self.pos[:, :, 2])
        else:
            x = self.pos[:, :, 0]
            z = self.pos[:, :, 2]
            rr = np.sqrt(x * x + z * z)
            wall = np.minimum(1.0, self._arena_r / np.maximum(rr, 1e-6))
            self.pos[:, :, 0] = x * wall
            self.pos[:, :, 2] = z * wall

        # -- 6) gravity 0.08 + 0.98 y-drag; ground/air horizontal friction -
        self.vel[:, :, 1] = np.where(
            self.on_ground, 0.0, (self.vel[:, :, 1] - self._gravity) * self._ydrag
        )
        fric = np.where(was_grounded, self._fric_ground, self._fric_air)
        self.vel[:, :, 0] *= fric
        self.vel[:, :, 2] *= fric

        # -- 6b) player-player collision push (1.8 applyEntityCollision) ---
        # overlapping players shove each other apart; without this a walking
        # attacker phases through its target instead of herding it along
        # (validated against a real 1.8.8 trace). Runs after friction, as in
        # vanilla, so the push carries undamped into next tick's move.
        cdx = self.pos[:, 1, 0] - self.pos[:, 0, 0]
        cdz = self.pos[:, 1, 2] - self.pos[:, 0, 2]
        cdy = np.abs(self.pos[:, 1, 1] - self.pos[:, 0, 1])
        absmax = np.maximum(np.abs(cdx), np.abs(cdz))
        touch_r = 2.0 * self._half_w + np.float32(0.2)  # AABBs expanded 0.2
        touching = (
            (np.abs(cdx) < touch_r)
            & (np.abs(cdz) < touch_r)
            & (cdy < self._box_h)
            & (absmax >= 0.01)
        )
        s = np.sqrt(np.maximum(absmax, np.float32(1e-9)))
        pscale = np.minimum(1.0 / s, 1.0) * self._push / s
        px = np.where(touching, cdx * pscale, 0.0).astype(np.float32)
        pz = np.where(touching, cdz * pscale, 0.0).astype(np.float32)
        self.vel[:, 0, 0] -= px
        self.vel[:, 0, 2] -= pz
        self.vel[:, 1, 0] += px
        self.vel[:, 1, 2] += pz

        # -- 7) combat ------------------------------------------------------
        # per-side geometry: row s sees its opponent via the [:, ::-1] view
        pos = self.pos
        tpos = pos[:, ::-1]
        dx = tpos[:, :, 0] - pos[:, :, 0]
        dz = tpos[:, :, 2] - pos[:, :, 2]
        dy_feet = tpos[:, :, 1] - pos[:, :, 1]
        dy = dy_feet + self._eye_to_chest        # own eye -> target chest
        hd2 = dx * dx + dz * dz
        hdist = np.sqrt(hd2)
        dist_c = np.sqrt(hd2 + dy * dy)

        # reach: eye to closest point of the target 0.6 x 1.8 AABB <= 3.0
        ax = np.maximum(np.abs(dx) - self._half_w, 0.0)
        az = np.maximum(np.abs(dz) - self._half_w, 0.0)
        eye_y = pos[:, :, 1] + self._eye_h
        ay = np.maximum(
            np.maximum(tpos[:, :, 1] - eye_y, eye_y - (tpos[:, :, 1] + self._box_h)),
            0.0,
        )
        reach_d = np.sqrt(ax * ax + ay * ay + az * az)

        # exact ray-vs-AABB, like the real 1.8 client: a click only connects
        # when the crosshair ray from the eye intersects the target's
        # 0.6 x 1.8 box within reach. The earlier padded cone (+10 deg
        # slack) let a policy spam-click with ~15 deg aim slop; the real
        # client rejects that (observed live: 73% camera-on-target, 77%
        # click rate, zero hits landed). Precision is the point.
        # (true angular errors still feed the dense aim-shaping reward)
        tgt_yaw = np.arctan2(dz, dx)
        yaw_err = (tgt_yaw - yaw_rad + _PI) % _TWO_PI - _PI
        pitch_rad = self.pitch * _DEG2RAD
        pitch_err = np.arctan2(dy, hdist) - pitch_rad
        cp = np.cos(pitch_rad)
        vx = cp * np.cos(yaw_rad)
        vy = np.sin(pitch_rad)
        vz = cp * np.sin(yaw_rad)
        # slab test against the target box, origin = own eye
        with np.errstate(divide="ignore", invalid="ignore"):
            lo_x = (dx - self._click_hw) / vx
            hi_x = (dx + self._click_hw) / vx
            t0x, t1x = np.minimum(lo_x, hi_x), np.maximum(lo_x, hi_x)
            oy = dy_feet - self._eye_h          # eye -> target FEET y
            lo_y = (oy - self._click_b) / vy
            hi_y = (oy + self._box_h + self._click_b) / vy
            t0y, t1y = np.minimum(lo_y, hi_y), np.maximum(lo_y, hi_y)
            lo_z = (dz - self._click_hw) / vz
            hi_z = np.copy((dz + self._click_hw) / vz)
            t0z, t1z = np.minimum(lo_z, hi_z), np.maximum(lo_z, hi_z)
        # degenerate axes (view component ~0): inside-slab -> +/-inf bounds
        for (t0, t1, v, o, half) in (
            (t0x, t1x, vx, dx, self._click_hw),
            (t0y, t1y, vy, oy + 0.5 * self._box_h,
             0.5 * self._box_h + self._click_b),
            (t0z, t1z, vz, dz, self._click_hw),
        ):
            deg = np.abs(v) < 1e-8
            inside = np.abs(o) <= half + 1e-6
            t0[deg] = np.where(inside[deg], -np.inf, np.inf)
            t1[deg] = np.where(inside[deg], np.inf, -np.inf)
        tmin = np.maximum(np.maximum(t0x, t0y), t0z)
        tmax = np.minimum(np.minimum(t1x, t1y), t1z)
        aim_ok = (tmax >= np.maximum(tmin, 0.0)) & (tmin <= self._reach)

        swing = a[:, :, 4] == 1
        landed = swing & aim_ok & (reach_d <= self._reach)
        if self._hit_reg_p < 1.0:
            landed[:, 0] &= self.rng.random(self.num_envs) < self._hit_reg_p
        # 1.8 hurt window (EntityLivingBase.attackEntityFrom, MCP-919): a hit
        # inside the 10-tick window is NOT simply ignored -- if it deals MORE
        # than the last hit, the EXCESS damage goes through (lastDamage
        # bookkeeping); only fresh-window hits re-arm the window, trigger the
        # flash, and knock back. A crit chasing a normal hit lands 1/3 of its
        # damage mid-window.
        crit = (~self.on_ground) & (self.vel[:, :, 1] < 0.0)
        dmg_amt = np.where(crit, self._dmg_crit, self._dmg_base)
        in_window = self.hurt[:, ::-1] > 0      # victim state, attacker rows
        last_v = self.last_dmg[:, ::-1]         # victim lastDamage, attacker rows
        fresh = landed & ~in_window
        excess = landed & in_window & (dmg_amt > last_v)
        hits = fresh                            # rewards/kb/flash: fresh only
        hits_f = hits.astype(np.float32)
        taken = hits[:, ::-1]           # row s: was hit by its opponent
        taken_f = hits_f[:, ::-1]
        dealt = np.where(fresh, dmg_amt, 0.0) + np.where(
            excess, dmg_amt - last_v, 0.0)
        self.hp -= dealt[:, ::-1].astype(np.float32)
        upd = (fresh | excess)[:, ::-1]
        self.last_dmg = np.where(upd, dmg_amt[:, ::-1], self.last_dmg)
        # 10-tick hurt invulnerability window: re-armed on fresh hits only
        # (excess hits do not reset it, per source), otherwise ticks down
        self.hurt = np.where(taken, self._hurt_ticks, np.maximum(self.hurt - 1, 0))

        # knockback (1.8 knockBack): halve all axes, +0.4 horizontal along
        # attacker->target, +0.4 vertical capped at 0.4
        dxr = dx[:, ::-1]               # attacker -> victim horizontal delta
        dzr = dz[:, ::-1]
        khr = hdist[:, ::-1]
        khs = np.maximum(khr, 1e-6)
        degen = khr < 1e-4              # overlapping: fall back to attacker yaw
        kx = np.where(degen, cos_y[:, ::-1], dxr / khs)
        kz = np.where(degen, sin_y[:, ::-1], dzr / khs)
        keep = 1.0 - 0.5 * taken_f
        self.vel[:, :, 0] = self.vel[:, :, 0] * keep + kx * self._kb_h * taken_f
        self.vel[:, :, 2] = self.vel[:, :, 2] * keep + kz * self._kb_h * taken_f
        vy = self.vel[:, :, 1] * keep + self._kb_v * taken_f
        self.vel[:, :, 1] = np.where(taken, np.minimum(vy, self._kb_cap), vy)

        # sprint hit: additive extras along the attacker's facing (validated
        # against a real trace -- no second halving on the horizontal axes);
        # attacker slows to 60% and loses sprint (W-tap mechanic)
        sprint_hit = hits & self.sprinting
        sh_f = sprint_hit[:, ::-1].astype(np.float32)
        self.vel[:, :, 0] += cos_y[:, ::-1] * self._skb_h * sh_f
        self.vel[:, :, 2] += sin_y[:, ::-1] * self._skb_h * sh_f
        self.vel[:, :, 1] += self._skb_v * sh_f
        slow = 1.0 - (1.0 - self._sprint_slow) * sprint_hit.astype(np.float32)
        self.vel[:, :, 0] *= slow
        self.vel[:, :, 2] *= slow
        self.sprint_blocked |= sprint_hit   # attacker must W-tap to re-sprint
        self.sprint_blocked |= taken        # damage taken breaks sprint too
        self.sprinting &= ~(sprint_hit | taken)

        self.since_dealt = np.where(
            hits, 0.0, np.minimum(self.since_dealt + 1.0, 100.0)
        )
        self.since_taken = np.where(
            taken, 0.0, np.minimum(self.since_taken + 1.0, 100.0)
        )

        # -- 8) rewards -----------------------------------------------------
        rew = hits_f * np.float32(rc.hit) + taken_f * np.float32(rc.hurt)
        rew -= np.float32(rc.aim_coeff) * (
            np.abs(yaw_err) * _INV_PI + np.abs(pitch_err) * _INV_HALF_PI
        )
        rew -= np.float32(rc.sprint_toggle_coeff) * (
            a[:, :, 3] != self.prev_actions[:, :, 3]
        ).astype(np.float32)
        if rc.punish_close_bonus != 0.0:
            d10 = self.pos[:, 0] - self.pos[:, 1]
            dist_c = np.sqrt((d10 * d10).sum(axis=1))
            u10 = d10 / np.maximum(dist_c, 1e-6)[:, None]
            close0 = -(self.vel[:, 0] * u10).sum(axis=1)  # side0 toward side1
            close1 = (self.vel[:, 1] * u10).sum(axis=1)   # side1 toward side0
            band = (dist_c > self._COMMIT_LO) & (dist_c < self._COMMIT_HI)
            self._commit[band & (close1 > self._COMMIT_V), 0] = self._COMMIT_TICKS
            self._commit[band & (close0 > self._COMMIT_V), 1] = self._COMMIT_TICKS
            rew += np.float32(rc.punish_close_bonus) * hits_f * (self._commit > 0)
            np.maximum(self._commit - 1, 0, out=self._commit)
        if rc.punish_swing_bonus != 0.0:
            # opponent pressed attack this tick -> arm my punish window
            # (same-tick credit intended: human punishes land ~50-150ms
            # after the swing, i.e. within the very next frames)
            self._opp_swing[a[:, 1, 4] == 1, 0] = self._SWING_TICKS
            self._opp_swing[a[:, 0, 4] == 1, 1] = self._SWING_TICKS
            rew += np.float32(rc.punish_swing_bonus) * hits_f * (self._opp_swing > 0)
            np.maximum(self._opp_swing - 1, 0, out=self._opp_swing)

        # -- 9) termination -------------------------------------------------
        self.ticks += 1
        dead = self.hp <= 0.0
        done = dead.any(axis=1) | (self.ticks >= self._max_ticks)
        win = (dead[:, ::-1] & ~dead).astype(np.float32)   # opponent died
        loss = (dead & ~dead[:, ::-1]).astype(np.float32)  # self died
        rew += np.float32(rc.win) * win + np.float32(rc.loss) * loss

        pd = pos[:, 1] - pos[:, 0]
        info = {"win": win, "dist": np.sqrt((pd * pd).sum(axis=1))}

        self.prev_actions = a.astype(np.int64)  # always copies
        self._reset_envs(done)
        return self._delayed(self._obs(), done), rew, done, info

    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        """Fill every OBS_LAYOUT slot for both sides (see pvpbot/spec.py)."""
        obs = self._obs_buf
        pos = self.pos
        vel = self.vel
        for side in (0, 1):
            other = 1 - side
            o = obs[:, side]
            dx = pos[:, other, 0] - pos[:, side, 0]
            dy = pos[:, other, 1] - pos[:, side, 1]
            dz = pos[:, other, 2] - pos[:, side, 2]
            yr = self.yaw[:, side] * _DEG2RAD
            c = np.cos(yr)
            s = np.sin(yr)
            # rel_pos: enemy - self, rotated into the self-yaw frame, /8
            i = _S["rel_pos"]
            o[:, i] = (dx * c + dz * s) * 0.125
            o[:, i + 1] = dy * 0.125
            o[:, i + 2] = (-dx * s + dz * c) * 0.125
            # rel_vel: enemy vel - self vel, self-yaw frame (blocks/tick)
            rvx = vel[:, other, 0] - vel[:, side, 0]
            rvz = vel[:, other, 2] - vel[:, side, 2]
            i = _S["rel_vel"]
            o[:, i] = rvx * c + rvz * s
            o[:, i + 1] = vel[:, other, 1] - vel[:, side, 1]
            o[:, i + 2] = -rvx * s + rvz * c
            # self_vel: own velocity, self-yaw frame
            svx = vel[:, side, 0]
            svz = vel[:, side, 2]
            i = _S["self_vel"]
            o[:, i] = svx * c + svz * s
            o[:, i + 1] = vel[:, side, 1]
            o[:, i + 2] = -svx * s + svz * c
            # self_pitch_sincos
            pr = self.pitch[:, side] * _DEG2RAD
            i = _S["self_pitch_sincos"]
            o[:, i] = np.sin(pr)
            o[:, i + 1] = np.cos(pr)
            # dist /8, in_reach (dist < 3.0; LOS always clear on flat arena)
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            o[:, _S["dist"]] = dist * 0.125
            o[:, _S["in_reach"]] = dist < self._reach
            # timers / hp / flags
            o[:, _S["self_hurt"]] = self.hurt[:, side] * 0.1
            o[:, _S["self_hp"]] = self.hp[:, side] * 0.05
            o[:, _S["self_on_ground"]] = self.on_ground[:, side]
            o[:, _S["enemy_on_ground"]] = self.on_ground[:, other]
            o[:, _S["self_sprinting"]] = self.sprinting[:, side]
            o[:, _S["ticks_since_hit_taken"]] = self.since_taken[:, side] * 0.01
            # dead-sense masking (learner side only): live, enemy_hurt /
            # enemy_hp / ticks_since_hit_dealt all derive from the CNN
            # hurt-flash head, which detects 0% of real hits (KB-truth,
            # 2026-07-25). Training with these as ground truth taught
            # sim-only trade timing: live 10-22 (deaths 2x) vs sim 0.88.
            # Physics is source-exact; the SENSES must match the pipeline.
            if self._mask_dead_senses and side == 0:
                o[:, _S["enemy_hurt"]] = 0.0
                o[:, _S["enemy_hp"]] = 1.0    # adapter's live default: full
                o[:, _S["ticks_since_hit_dealt"]] = 1.0   # capped/stale
            else:
                o[:, _S["enemy_hurt"]] = self.hurt[:, other] * 0.1
                o[:, _S["enemy_hp"]] = self.hp[:, other] * 0.05
                o[:, _S["ticks_since_hit_dealt"]] = self.since_dealt[:, side] * 0.01
            # aim error: own crosshair (eye) -> enemy hitbox centre
            hd = np.sqrt(dx * dx + dz * dz)
            ey = dy + self._eye_to_chest
            ye = (np.arctan2(dz, dx) - yr + _PI) % _TWO_PI - _PI
            o[:, _S["aim_err_yaw"]] = ye * _INV_PI
            o[:, _S["aim_err_pitch"]] = (np.arctan2(ey, hd) - pr) * _INV_HALF_PI
            # prev_action, each index normalized by its head size
            i = _S["prev_action"]
            o[:, i : i + NUM_ACTION_HEADS] = (
                self.prev_actions[:, side].astype(np.float32) * self._inv_heads
            )

            # Capture the UNGATED, noiseless obs before gating. Scripted
            # opponents model real mineflayer bots, which have perfect
            # server-side knowledge of the player's position (no camera FOV,
            # no perception noise). Feeding them the gated/noisy learner-view
            # made the sim opponent a blindfolded weakling -- the learner
            # "won" sim 100%% then lost to the omniscient live bot. Only the
            # learner (camera-based) should be FOV-gated + noised.
            self._obs_ungated[:, side] = o
            self._obs_ungated[:, side, _S["enemy_visible"]] = 1.0

            if not self._fov_gate:
                o[:, _S["enemy_visible"]] = 1.0
            else:
                # ---- camera-cone gating (emulates the live adapter) ------
                pe = np.arctan2(ey, hd) - pr
                vis = (
                    (np.abs(ye) < self._fov_hy)
                    & (np.abs(pe) < self._fov_hp)
                    & (dist < self._obs_maxd)
                )
                if self._obs_nz > 0.0:
                    # perception false-negatives (measured visible_p ~0.08;
                    # use a conservative per-tick flake)
                    vis &= self.rng.random(len(vis)) > 0.04 * self._obs_nz
                if side == 0 and self._vis_miss > 0.0:
                    if self._vis_reacq > 0.0:
                        r = self.rng.random(len(vis))
                        lost = self._vis_lost
                        self._vis_lost = np.where(
                            lost, r >= self._vis_reacq, r < self._vis_miss)
                        vis &= ~self._vis_lost
                    else:
                        vis &= self.rng.random(len(vis)) > self._vis_miss
                inv = ~vis

                # update last-seen caches from the freshly written truth
                i = _S["rel_pos"]
                self._seen_rel[vis, side] = o[vis, i : i + 3]
                iv = _S["rel_vel"]
                self._seen_relvel[vis, side] = o[vis, iv : iv + 3]
                self._seen_relvel[inv, side] *= 0.8
                self._seen_dist[vis, side] = o[vis, _S["dist"]]

                # overwrite enemy-derived slots with the gated view
                o[:, i : i + 3] = self._seen_rel[:, side]
                o[:, iv : iv + 3] = self._seen_relvel[:, side]
                o[:, _S["dist"]] = self._seen_dist[:, side]
                o[inv, _S["in_reach"]] = 0.0
                o[inv, _S["aim_err_yaw"]] = 0.0
                o[inv, _S["aim_err_pitch"]] = 0.0
                o[inv, _S["enemy_on_ground"]] = 0.0
                o[:, _S["enemy_visible"]] = vis

                if self._obs_nz > 0.0:
                    # ---- perception-grade noise (real fine-tuned MAEs) ---
                    # aim noise = OU-correlated bias (~2 s time constant,
                    # like the real CNN's persistent per-view bias) plus a
                    # smaller white component.
                    nz = float(self._obs_nz)
                    nv = int(vis.sum())
                    rng = self.rng
                    nb = self._noise_bias[:, side]
                    nb += -0.05 * nb + rng.normal(
                        0.0, 0.020 * nz, nb.shape).astype(np.float32)
                    if nv:
                        o[vis, _S["aim_err_yaw"]] += nb[vis, 0] + rng.normal(
                            0.0, 0.04 * nz, nv).astype(np.float32)
                        o[vis, _S["aim_err_pitch"]] += nb[vis, 1] + rng.normal(
                            0.0, 0.05 * nz, nv).astype(np.float32)
                        o[vis, _S["dist"]] *= 1.0 + np.clip(
                            rng.normal(0.0, 0.12 * nz, nv), -0.3, 0.3
                        ).astype(np.float32)
                        o[vis, i : i + 3] += rng.normal(
                            0.0, 0.03 * nz, (nv, 3)).astype(np.float32)
                        o[vis, iv : iv + 3] += rng.normal(
                            0.0, 0.02 * nz, (nv, 3)).astype(np.float32)
                    isv = _S["self_vel"]
                    o[:, isv : isv + 3] += rng.normal(
                        0.0, 0.01 * nz, (len(o), 3)).astype(np.float32)
                # live pixel-pipeline artifacts, learner side only (see
                # SimConfig.rel_vel_noise / rel_pos_y_bias)
                if side == 0:
                    if self._rv_nz > 0.0:
                        ou = self._relvel_ou
                        ou *= 0.8
                        ou += self.rng.normal(
                            0.0, self._rv_nz * 0.6, ou.shape
                        ).astype(np.float32)
                        o[:, iv : iv + 3] += ou
                    if self._rpy_bias != 0.0:
                        o[:, i + 1] += self._rpy_bias
            # "reserved" [32:48] stays zero (buffer is never written there)
        self._last_obs0 = obs[:, 0].copy()
        return obs.copy()
