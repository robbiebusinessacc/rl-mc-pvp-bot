"""ObsAssembler: valid 48-vectors with correctly sourced slots over a stream."""
import numpy as np

from pvpbot.spec import (
    ACTION_HEAD_SIZES,
    NUM_ACTION_HEADS,
    OBS_DIM,
    OBS_LAYOUT,
    PERCEPTION_DIM,
    PERCEPTION_LAYOUT,
)
from pvpbot.perception.adapter import ObsAssembler
from pvpbot.perception.synth import bbox_height_frac, generate_batch

P = PERCEPTION_LAYOUT


def _perc(**kw):
    p = np.zeros(PERCEPTION_DIM, dtype=np.float32)
    for name, v in kw.items():
        p[P[name]] = v
    return p


def _idle_action():
    a = np.zeros(NUM_ACTION_HEADS, dtype=np.int64)
    a[0] = 1  # forward: none
    a[1] = 1  # strafe: none
    a[5] = 5  # yaw bin 0
    a[6] = 5  # pitch bin 0
    return a


def test_100_tick_stream_produces_valid_obs():
    """Feed 100 ticks of real synthetic labels + random actions."""
    _, labels = generate_batch(100, seed=42)
    rng = np.random.default_rng(0)
    asm = ObsAssembler()
    for t in range(100):
        action = np.array([rng.integers(0, n) for n in ACTION_HEAD_SIZES],
                          dtype=np.int64)
        obs = asm.update(labels[t], action)
        assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
        assert np.isfinite(obs).all(), "non-finite obs at tick {}".format(t)
        s, e = OBS_LAYOUT["reserved"]
        assert (obs[s:e] == 0).all()
        s, e = OBS_LAYOUT["self_pitch_sincos"]
        assert abs(float(np.square(obs[s:e]).sum()) - 1.0) < 1e-5
        s, e = OBS_LAYOUT["dist"]
        assert 0.0 < obs[s] < 30.0 / 8.0
        s, e = OBS_LAYOUT["prev_action"]
        assert (obs[s:e] >= 0).all() and (obs[s:e] < 1.0).all()
        for name in ("in_reach", "self_hp", "enemy_hp", "self_on_ground",
                     "enemy_on_ground", "self_sprinting"):
            s, e = OBS_LAYOUT[name]
            assert 0.0 <= obs[s] <= 1.0, name


def test_perception_slots_pass_through():
    asm = ObsAssembler()
    p = _perc(aim_err_yaw=0.1, aim_err_pitch=-0.2, visible=1.0,
              bbox_height=bbox_height_frac(4.0), self_hp=0.5,
              enemy_on_ground=1.0)
    obs = asm.update(p, _idle_action())
    s, _ = OBS_LAYOUT["aim_err_yaw"]
    assert abs(obs[s] - 0.1) < 1e-6
    s, _ = OBS_LAYOUT["aim_err_pitch"]
    # perception pitch err is down-positive (synth "+ = below"); the obs
    # contract (sim env) is up-positive, so the adapter flips the sign
    assert abs(obs[s] - 0.2) < 1e-6
    s, _ = OBS_LAYOUT["enemy_on_ground"]
    assert obs[s] == 1.0
    # self_hp is NO LONGER a raw pass-through: it is median-filtered and
    # calibrated (perception HP jitters ~3hp, which used to false-trigger
    # self_hurt on 42%% of frames). A single low reading is rejected as
    # noise, so one self_hp=0.5 frame stays near full.
    s, _ = OBS_LAYOUT["self_hp"]
    assert obs[s] > 0.9


def test_self_hp_median_filter_rejects_single_frame_noise():
    asm = ObsAssembler()
    for _ in range(9):
        asm.update(_perc(visible=0.0, self_hp=1.0), _idle_action())
    # one noisy low frame among full-HP readings must not move self_hp much
    obs = asm.update(_perc(visible=0.0, self_hp=0.2), _idle_action())
    s, _ = OBS_LAYOUT["self_hp"]
    assert obs[s] > 0.9
    s, _ = OBS_LAYOUT["self_hurt"]
    assert obs[s] == 0.0    # single-frame dip is not a hit


def test_dist_converges_to_bbox_inverse():
    asm = ObsAssembler()
    p = _perc(visible=1.0, bbox_height=bbox_height_frac(2.5), self_hp=1.0)
    for _ in range(20):
        obs = asm.update(p, _idle_action())
    s, _ = OBS_LAYOUT["dist"]
    assert abs(obs[s] * 8.0 - 2.5) < 0.1
    s, _ = OBS_LAYOUT["in_reach"]
    assert obs[s] == 1.0  # 2.5 < 3.0 reach and visible


def test_invisible_enemy_zeroes_aim_and_holds_dist():
    asm = ObsAssembler()
    seen = _perc(visible=1.0, aim_err_yaw=0.2,
                 bbox_height=bbox_height_frac(5.0), self_hp=1.0)
    for _ in range(10):
        asm.update(seen, _idle_action())
    unseen = _perc(visible=0.0, aim_err_yaw=0.4, self_hp=1.0)
    obs = asm.update(unseen, _idle_action())
    s, _ = OBS_LAYOUT["aim_err_yaw"]
    assert obs[s] == 0.0
    s, _ = OBS_LAYOUT["dist"]
    assert abs(obs[s] * 8.0 - 5.0) < 0.3     # held estimate
    s, _ = OBS_LAYOUT["in_reach"]
    assert obs[s] == 0.0


def test_hurt_flash_drives_enemy_hurt_hp_and_hit_timers():
    asm = ObsAssembler()
    calm = _perc(visible=1.0, bbox_height=0.3, self_hp=1.0)
    asm.update(calm, _idle_action())
    flash = _perc(visible=1.0, bbox_height=0.3, self_hp=1.0, hurt_flash=1.0)
    obs = asm.update(flash, _idle_action())
    s, _ = OBS_LAYOUT["enemy_hurt"]
    assert obs[s] == 1.0                     # timer set to 10 -> /10
    s, _ = OBS_LAYOUT["enemy_hp"]
    assert abs(obs[s] - 19.0 / 20.0) < 1e-6  # -1 hp heuristic
    s, _ = OBS_LAYOUT["ticks_since_hit_dealt"]
    assert obs[s] == 0.0
    obs = asm.update(calm, _idle_action())   # flash gone: timer decays
    s, _ = OBS_LAYOUT["enemy_hurt"]
    assert abs(obs[s] - 0.9) < 1e-6


def test_sustained_hp_drop_sets_self_hurt():
    # self_hurt fires only on a SUSTAINED, high-confidence drop of the
    # median-smoothed hp (>=3hp) -- a real hit, not read jitter. Feed full
    # HP long enough to fill the median window, then a sustained large drop.
    asm = ObsAssembler()
    for _ in range(12):
        asm.update(_perc(visible=0.0, self_hp=1.0), _idle_action())   # 20 hp
    for _ in range(12):
        obs = asm.update(_perc(visible=0.0, self_hp=0.6), _idle_action())  # 12 hp
    s, _ = OBS_LAYOUT["self_hurt"]
    assert obs[s] > 0.0                       # sustained -8hp drop = a hit
    # and a single noisy dip does NOT trigger it (noise rejection)
    asm2 = ObsAssembler()
    for _ in range(12):
        asm2.update(_perc(visible=0.0, self_hp=1.0), _idle_action())
    obs = asm2.update(_perc(visible=0.0, self_hp=0.9), _idle_action())  # one -2hp frame
    s, _ = OBS_LAYOUT["self_hurt"]
    assert obs[s] == 0.0


def test_dead_reckoned_self_state_from_actions():
    asm = ObsAssembler()
    run = _idle_action()
    run[0] = 2   # forward
    run[3] = 1   # sprint held
    p = _perc(visible=0.0, self_hp=1.0)
    for _ in range(15):
        obs = asm.update(p, run)
    s, e = OBS_LAYOUT["self_vel"]
    assert obs[s] > 0.2          # forward velocity built up (~0.29 terminal)
    s, _ = OBS_LAYOUT["self_sprinting"]
    assert obs[s] == 1.0
    s, _ = OBS_LAYOUT["self_on_ground"]
    assert obs[s] == 1.0
    jump = run.copy()
    jump[2] = 1
    obs = asm.update(p, jump)
    s, _ = OBS_LAYOUT["self_on_ground"]
    assert obs[s] == 0.0         # airborne after the jump impulse
    # prev_action normalized per head size (spec convention)
    s, e = OBS_LAYOUT["prev_action"]
    expect = jump.astype(np.float32) / np.asarray(ACTION_HEAD_SIZES,
                                                  dtype=np.float32)
    assert np.allclose(obs[s:e], expect)


def test_pitch_integration_tracks_issued_camera_deltas():
    asm = ObsAssembler()
    look_down = _idle_action()
    look_down[6] = 8             # +7 deg/tick pitch bin (sim UP-positive)
    # perception agrees with the integrated pitch: the CNN reports
    # DOWN-positive, so "7t degrees up" reads as -7t/90 from perception
    # (the adapter converts at the boundary)
    for t in range(1, 6):
        p = _perc(visible=0.0, self_hp=1.0, self_pitch=-7.0 * t / 90.0)
        obs = asm.update(p, look_down)
    s, e = OBS_LAYOUT["self_pitch_sincos"]
    pitch = np.degrees(np.arctan2(obs[s], obs[s + 1]))
    assert abs(pitch - 35.0) < 2.0


def test_reset_clears_state():
    asm = ObsAssembler()
    for _ in range(5):
        asm.update(_perc(visible=1.0, bbox_height=0.5, hurt_flash=1.0,
                         self_hp=0.2), _idle_action())
    asm.reset()
    assert asm.tick == 0 and asm.enemy_hp == 20.0
    obs = asm.update(_perc(visible=0.0, self_hp=1.0), None)
    assert np.isfinite(obs).all()
