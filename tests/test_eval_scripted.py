"""Scripted tiers: valid actions, slot reading, determinism, aim behaviour."""
import numpy as np
import pytest

from pvpbot.eval.scripted import (
    ATTACK,
    FORWARD,
    JUMP,
    SPRINT,
    STRAFE,
    YAW,
    AimbotStationary,
    Chaser,
    Idle,
    Pro,
    Strafer,
    default_tiers,
    nearest_camera_bin,
)
from pvpbot.sim.stub import DuelVecEnv
from pvpbot.spec import ACTION_HEAD_SIZES, CAMERA_BINS, NUM_ACTION_HEADS, OBS_LAYOUT

AIM_YAW = OBS_LAYOUT["aim_err_yaw"][0]
IN_REACH = OBS_LAYOUT["in_reach"][0]
ENEMY_HURT = OBS_LAYOUT["enemy_hurt"][0]
SELF_HURT = OBS_LAYOUT["self_hurt"][0]
DIST = OBS_LAYOUT["dist"][0]
PREV_ATTACK = OBS_LAYOUT["prev_action"][0] + ATTACK


def _obs_row(slots):
    row = np.zeros(48, dtype=np.float32)
    for slot, val in slots.items():
        row[slot] = val
    return row


def test_all_tiers_valid_actions_on_stub_rollout():
    env = DuelVecEnv(num_envs=6, seed=3)
    for tier in default_tiers(1):
        obs = env.reset()
        tier.begin(6)
        for _ in range(40):
            acts = np.stack([tier.act(obs[:, 0]), tier.act(obs[:, 1])], axis=1)
            assert acts.shape == (6, 2, NUM_ACTION_HEADS)
            assert acts.dtype == np.int64
            for h, size in enumerate(ACTION_HEAD_SIZES):
                assert (acts[..., h] >= 0).all() and (acts[..., h] < size).all()
            obs, _, done, _ = env.step(acts)
            tier.on_done(done)


def test_idle_is_noop():
    obs = np.random.default_rng(0).normal(size=(5, 48)).astype(np.float32)
    a = Idle().act(obs)
    assert (a[:, FORWARD] == 1).all() and (a[:, STRAFE] == 1).all()
    assert (a[:, JUMP] == 0).all() and (a[:, SPRINT] == 0).all()
    assert (a[:, ATTACK] == 0).all()
    zero_bin = CAMERA_BINS.index(0.0)
    assert (a[:, YAW] == zero_bin).all() and (a[:, 6] == zero_bin).all()


def test_nearest_camera_bin_mapping():
    got = nearest_camera_bin(np.array([-30.0, -1.2, 2.5, 8.0, 200.0]))
    assert got.tolist() == [
        CAMERA_BINS.index(-30.0),
        CAMERA_BINS.index(-1.0),
        CAMERA_BINS.index(3.0),
        CAMERA_BINS.index(7.0),
        CAMERA_BINS.index(30.0),
    ]
    assert got.dtype == np.int64


def test_aimbot_corrects_toward_error():
    # aim error of +90 deg -> largest positive bin; -2.8 deg -> the -3 bin
    tier = AimbotStationary()
    obs = np.stack([
        _obs_row({AIM_YAW: 90.0 / 180.0}),
        _obs_row({AIM_YAW: -2.8 / 180.0}),
    ])
    a = tier.act(obs)
    assert a[0, YAW] == CAMERA_BINS.index(30.0)
    assert a[1, YAW] == CAMERA_BINS.index(-3.0)


def test_aimbot_attack_gated_on_reach_and_cadence():
    tier = AimbotStationary()
    in_reach = _obs_row({IN_REACH: 1.0})[None]
    out_reach = _obs_row({IN_REACH: 0.0})[None]
    assert tier.act(out_reach)[0, ATTACK] == 0
    assert tier.act(in_reach)[0, ATTACK] == 1
    # click cooldown: no second swing immediately after
    assert tier.act(in_reach)[0, ATTACK] == 0


def test_aimbot_converges_on_stub():
    env = DuelVecEnv(num_envs=8, seed=5)
    obs = env.reset()
    t1 = AimbotStationary()
    t0 = Idle()
    for _ in range(40):
        acts = np.stack([t1.act(obs[:, 0]), t0.act(obs[:, 1])], axis=1)
        obs, _, done, _ = env.step(acts)
        t1.on_done(done)
    err_deg = np.abs(obs[:, 0, AIM_YAW]) * 180.0
    assert (err_deg < 1.0).all(), err_deg


def test_chaser_moves_forward_with_sprint():
    a = Chaser().act(_obs_row({AIM_YAW: 0.1, DIST: 6.0 / 8.0})[None])
    assert a[0, FORWARD] == 2 and a[0, SPRINT] == 1


def test_strafer_strafes_close_but_not_far():
    tier = Strafer(seed=2)
    far = _obs_row({DIST: 6.0 / 8.0})[None]
    near = _obs_row({DIST: 2.0 / 8.0})[None]
    assert tier.act(far)[0, STRAFE] == 1  # straight approach
    assert tier.act(near)[0, STRAFE] in (0, 2)  # circling


def test_pro_attack_timing_exact():
    tier = Pro()
    window_open = _obs_row({IN_REACH: 1.0, ENEMY_HURT: 0.0})[None]
    window_shut = _obs_row({IN_REACH: 1.0, ENEMY_HURT: 0.5})[None]
    out_of_reach = _obs_row({IN_REACH: 0.0, ENEMY_HURT: 0.0})[None]
    assert tier.act(window_open)[0, ATTACK] == 1
    assert tier.act(window_shut)[0, ATTACK] == 0
    assert tier.act(out_of_reach)[0, ATTACK] == 0
    # unlike cooldown tiers it swings again the very next open tick
    assert tier.act(window_open)[0, ATTACK] == 1


def test_pro_wtap_on_landed_hit_cue():
    tier = Pro()
    landed = _obs_row({PREV_ATTACK: 0.1, ENEMY_HURT: 1.0,
                         DIST: 1.0 / 8.0, IN_REACH: 1.0})[None]
    a = tier.act(landed)
    assert a[0, FORWARD] == 1 and a[0, SPRINT] == 0  # sprint reset
    # keeps releasing for the configured wtap window
    neutral = _obs_row({DIST: 1.0 / 8.0, IN_REACH: 1.0})[None]
    a2 = tier.act(neutral)
    assert a2[0, FORWARD] == 1 and a2[0, SPRINT] == 0
    a3 = tier.act(neutral)
    assert a3[0, FORWARD] == 2 and a3[0, SPRINT] == 1


def test_pro_backs_off_in_own_hurt_time_when_jammed():
    tier = Pro()
    jammed = _obs_row({SELF_HURT: 0.8, DIST: 0.8 / 8.0, IN_REACH: 1.0})[None]
    assert tier.act(jammed)[0, FORWARD] == 0


@pytest.mark.parametrize("cls", [AimbotStationary, Chaser, Strafer, Pro])
def test_seeded_determinism(cls):
    rng = np.random.default_rng(9)
    obs_seq = [rng.uniform(-1, 1, (4, 48)).astype(np.float32) for _ in range(20)]
    p1, p2 = cls(seed=7), cls(seed=7)
    for obs in obs_seq:
        np.testing.assert_array_equal(p1.act(obs), p2.act(obs))


def test_batch_size_change_self_heals():
    tier = Strafer(seed=1)
    a = tier.act(np.zeros((4, 48), dtype=np.float32))
    b = tier.act(np.zeros((9, 48), dtype=np.float32))
    assert a.shape == (4, NUM_ACTION_HEADS) and b.shape == (9, NUM_ACTION_HEADS)


def test_default_tiers_order_and_elo_docs():
    tiers = default_tiers()
    names = [t.name for t in tiers]
    assert names == ["T0-Idle", "T1-Aimbot", "T2-Chaser", "T3-Strafer", "T4-Pro"]
    elos = [t.intended_elo for t in tiers]
    assert elos == sorted(elos) and len(set(elos)) == len(elos)
