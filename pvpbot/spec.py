"""Single source of truth for interface contracts between modules.

DO NOT MODIFY on module branches. All modules import constants from here.
If a contract must change, it happens on main after integration discussion.
"""

TICK_RATE = 20          # game ticks per second
OBS_DIM = 48            # per-player observation vector length
PERCEPTION_DIM = 12     # what the vision CNN estimates from one frame
FRAME_SHAPE = (3, 96, 170)  # RGB, HxW downscaled frame for perception
# (was (1, 64, 114) grayscale; color exists because cyan diamond armor on
# green/blue terrain is the strongest localization signal available)

# ---------------------------------------------------------------------------
# Action space: multi-discrete, one categorical head per entry, in this order.
# ---------------------------------------------------------------------------
ACTION_HEADS = (
    ("forward", 3),   # 0=back, 1=none, 2=forward
    ("strafe", 3),    # 0=left, 1=none, 2=right
    ("jump", 2),      # 0=no, 1=jump
    ("sprint", 2),    # 0=no, 1=sprint held
    ("attack", 2),    # 0=no, 1=swing this tick
    ("yaw", 11),      # index into CAMERA_BINS -> yaw delta, degrees/tick
    ("pitch", 11),    # index into CAMERA_BINS -> pitch delta, degrees/tick
)
NUM_ACTION_HEADS = len(ACTION_HEADS)
ACTION_HEAD_SIZES = tuple(n for _, n in ACTION_HEADS)

# Approximately mu-law spaced camera deltas (degrees per tick).
CAMERA_BINS = (-30.0, -15.0, -7.0, -3.0, -1.0, 0.0, 1.0, 3.0, 7.0, 15.0, 30.0)

# ---------------------------------------------------------------------------
# Observation vector layout: name -> (start, stop) into the 48-float vector.
# All values are float32. Angles are normalized by /180, distances by /8,
# velocities by /1.0 (blocks/tick), hp by /20, timers by /10 unless noted.
# Frame of reference: self yaw-aligned frame ("rel" = enemy relative to self).
# ---------------------------------------------------------------------------
OBS_LAYOUT = {
    "rel_pos": (0, 3),            # enemy pos minus self pos, self-yaw frame, /8
    "rel_vel": (3, 6),            # enemy vel minus self vel, self-yaw frame
    "self_vel": (6, 9),           # own velocity, self-yaw frame
    "self_pitch_sincos": (9, 11), # sin(pitch), cos(pitch)
    "dist": (11, 12),             # horizontal+vertical euclidean distance, /8
    "in_reach": (12, 13),         # 1.0 if enemy hittable (dist < 3.0 & LOS)
    "self_hurt": (13, 14),        # own hurt-time ticks remaining, /10
    "enemy_hurt": (14, 15),       # enemy hurt-time ticks remaining, /10
    "self_hp": (15, 16),          # /20
    "enemy_hp": (16, 17),         # /20
    "self_on_ground": (17, 18),
    "enemy_on_ground": (18, 19),
    "self_sprinting": (19, 20),
    "ticks_since_hit_dealt": (20, 21),  # capped 100, /100
    "ticks_since_hit_taken": (21, 22),  # capped 100, /100
    "aim_err_yaw": (22, 23),      # signed angle self crosshair -> enemy center, /180
    "aim_err_pitch": (23, 24),    # signed, /90
    "prev_action": (24, 31),      # previous tick's action index per head, /head_size
    "enemy_visible": (31, 32),    # 1.0 iff the enemy is inside the camera
                                  # cone (live: perception's visible output;
                                  # sim: FOV-gated when fov_limited_obs)
    "reserved": (32, 48),         # zeros for now; future features
}

# ---------------------------------------------------------------------------
# Perception output layout (PERCEPTION_DIM floats estimated from one frame).
# The obs-assembly adapter maps these into the OBS slots marked * below and
# fills the rest from OCR-free heuristics / defaults / recurrent estimates.
# ---------------------------------------------------------------------------
PERCEPTION_LAYOUT = {
    "aim_err_yaw": 0,      # * signed, /180
    "aim_err_pitch": 1,    # * signed, /90
    "bbox_height": 2,      # enemy bbox height / frame height (~ inverse dist)
    "visible": 3,          # enemy on screen at all
    "self_pitch": 4,       # from view geometry, /90
    "self_hp": 5,          # from hearts HUD, /20
    "hurt_flash": 6,       # enemy red damage-flash visible
    "rel_screen_vx": 7,    # enemy screen-space velocity x (frames apart)
    "rel_screen_vy": 8,
    "self_speed": 9,       # apparent own speed from optic flow, blocks/tick
    "enemy_on_ground": 10,
    "reserved": 11,
}

# ---------------------------------------------------------------------------
# Vectorized env API (implemented by pvpbot.sim; stub in pvpbot/sim/stub.py):
#
#   env = DuelVecEnv(num_envs, seed=0)
#   obs = env.reset()                          # (num_envs, 2, OBS_DIM) f32
#   obs, rew, done, info = env.step(actions)   # actions (num_envs, 2, NUM_ACTION_HEADS) i64
#     rew:  (num_envs, 2) f32   -- per side
#     done: (num_envs,)  bool   -- env auto-resets; obs after done is fresh
#     info: dict of arrays, must include "win" (num_envs, 2) f32 on done ticks
#
# Checkpoint format: torch.save({"model": state_dict, "meta": {...}}, path)
#   meta must include: {"obs_dim", "action_heads", "step", "elo"(optional)}
# ---------------------------------------------------------------------------
