"""Physics unit tests for the 1.8 duel sim (pvpbot.sim.env.DuelVecEnv).

These tests place players in known states (the env exposes its state
arrays) and check the documented 1.8 mechanics: jump apex, terminal
fall speed, sprint speed-up, knockback, hurt-time, crits, W-tap.
"""
import numpy as np

from pvpbot.sim.env import DuelVecEnv, RewardConfig, SimConfig
from pvpbot.spec import CAMERA_BINS, NUM_ACTION_HEADS, OBS_LAYOUT

ZERO_BIN = CAMERA_BINS.index(0.0)
DMG = 8.0 * 0.2          # 1.8 diamond sword through full diamond armor
DMG_CRIT = 8.0 * 1.5 * 0.2

BIG = SimConfig(arena_radius=1000.0, spawn_radius=5.0)


def noop_actions(n=1):
    a = np.zeros((n, 2, NUM_ACTION_HEADS), dtype=np.int64)
    a[:, :, 0] = 1  # forward: none
    a[:, :, 1] = 1  # strafe: none
    a[:, :, 5] = ZERO_BIN
    a[:, :, 6] = ZERO_BIN
    return a


def make_env(p0=(0.0, 0.0, 0.0), p1=(6.0, 0.0, 0.0), yaw0=0.0, yaw1=180.0,
             config=BIG):
    env = DuelVecEnv(1, seed=0, config=config)
    env.reset()
    env.pos[0, 0] = p0
    env.pos[0, 1] = p1
    env.vel[:] = 0.0
    env.yaw[0, 0] = yaw0
    env.yaw[0, 1] = yaw1
    env.pitch[:] = 0.0
    env.on_ground[:] = env.pos[:, :, 1] <= 0.0
    return env


def obs_slot(obs, side, name):
    s = OBS_LAYOUT[name][0]
    return float(obs[0, side, s])


# ---------------------------------------------------------------------------
# movement
# ---------------------------------------------------------------------------

def test_jump_apex_about_1_25_blocks():
    # v0=0.42, then v=(v-0.08)*0.98 each tick -> apex ~= 1.2522 blocks
    env = make_env()
    a = noop_actions()
    a[0, 0, 2] = 1  # hold jump (no double jumps midair)
    apex = 0.0
    for _ in range(12):
        env.step(a)
        apex = max(apex, float(env.pos[0, 0, 1]))
    assert 1.20 <= apex <= 1.30, apex


def test_terminal_fall_speed():
    # fixed point of v' = (v - 0.08) * 0.98 is -3.92 blocks/tick
    env = make_env()
    env.pos[0, 0, 1] = 2000.0
    env.on_ground[0, 0] = False
    a = noop_actions()
    for _ in range(300):
        env.step(a)
        vy = float(env.vel[0, 0, 1])
        assert vy < 0.0            # falling: always negative
        assert vy >= -3.921        # never faster than terminal
    assert abs(vy + 3.92) < 0.02   # converged to terminal speed
    assert float(env.pos[0, 0, 1]) > 0.0  # still airborne after 300 ticks


def test_sprint_faster_than_walk():
    # both run +x; side 1 sprints -> 30% faster in steady state
    env = make_env(p0=(0.0, 0.0, -5.0), p1=(0.0, 0.0, 5.0), yaw0=0.0, yaw1=0.0)
    a = noop_actions()
    a[:, :, 0] = 2      # forward
    a[0, 1, 3] = 1      # side 1 sprints
    for _ in range(40):
        env.step(a)
    walk = float(env.pos[0, 0, 0])
    sprint = float(env.pos[0, 1, 0])
    # 1.8 walk ~0.216 blocks/tick, sprint ~0.281
    assert 7.5 <= walk <= 9.5, walk
    assert 1.2 <= sprint / walk <= 1.4, sprint / walk


def test_sprint_requires_forward_held():
    env = make_env(p0=(0.0, 0.0, -5.0), p1=(0.0, 0.0, 5.0))
    a = noop_actions()
    a[0, 0, 3] = 1      # sprint key without forward: no sprint
    a[0, 1, 3] = 1
    a[0, 1, 0] = 2      # side 1 holds forward too
    obs, _, _, _ = env.step(a)
    assert obs_slot(obs, 0, "self_sprinting") == 0.0
    assert obs_slot(obs, 1, "self_sprinting") == 1.0
    # sprint-only input must not move the player
    assert abs(float(env.vel[0, 0, 0])) < 1e-6


def test_sprint_jump_boost():
    # sprint jump gets the 0.2 horizontal boost -> faster first airborne tick
    env_w = make_env(p0=(0.0, 0.0, -5.0), p1=(0.0, 0.0, 5.0), yaw1=0.0)
    a = noop_actions()
    a[:, :, 0] = 2
    a[:, :, 2] = 1              # jump
    a[0, 1, 3] = 1              # side 1 sprints
    env_w.step(a)
    vx_walk = float(env_w.vel[0, 0, 0])
    vx_sprint = float(env_w.vel[0, 1, 0])
    # the jump tick still applies GROUND friction (start-of-tick rule,
    # validated on a real 1.8.8 server), so the 0.2 boost survives only
    # as 0.2 * 0.546 ~= 0.11
    assert vx_sprint > vx_walk + 0.08


# ---------------------------------------------------------------------------
# combat
# ---------------------------------------------------------------------------

def test_hit_damage_and_knockback_direction():
    env = make_env(p0=(0.0, 0.0, 0.0), p1=(2.0, 0.0, 0.0))
    a = noop_actions()
    a[0, 0, 4] = 1  # side 0 swings, aimed straight at side 1 (+x)
    obs, rew, done, _ = env.step(a)
    assert abs(float(env.hp[0, 1]) - (20.0 - DMG)) < 1e-3
    # base knockback: 0.4 horizontal away from attacker, 0.4 up (capped)
    assert abs(float(env.vel[0, 1, 0]) - 0.4) < 1e-5
    assert abs(float(env.vel[0, 1, 1]) - 0.4) < 1e-5
    assert abs(float(env.vel[0, 1, 2])) < 1e-5
    assert obs_slot(obs, 0, "enemy_hurt") == 1.0   # 10/10
    assert obs_slot(obs, 1, "self_hurt") == 1.0
    assert obs_slot(obs, 0, "ticks_since_hit_dealt") == 0.0
    assert obs_slot(obs, 1, "ticks_since_hit_taken") == 0.0
    assert rew[0, 0] > 0.9 and rew[0, 1] < -0.9    # +-1 hit reward + shaping
    assert not done[0]


def test_attack_misses_out_of_reach_or_off_aim():
    # out of reach (5 blocks)
    env = make_env(p0=(0.0, 0.0, 0.0), p1=(5.0, 0.0, 0.0))
    a = noop_actions()
    a[0, 0, 4] = 1
    env.step(a)
    assert float(env.hp[0, 1]) == 20.0
    # in reach but looking 90 degrees away
    env = make_env(p0=(0.0, 0.0, 0.0), p1=(2.0, 0.0, 0.0), yaw0=90.0)
    env.step(a)
    assert float(env.hp[0, 1]) == 20.0


def test_hurt_time_blocks_double_hits():
    env = make_env(p0=(0.0, 0.0, 0.0), p1=(2.0, 0.0, 0.0))
    a = noop_actions()
    a[0, 0, 4] = 1  # swing every tick
    hits_landed = []
    for t in range(1, 13):
        hp_before = float(env.hp[0, 1])
        env.step(a)
        hits_landed.append(float(env.hp[0, 1]) < hp_before)
        # pin the victim in place so knockback can't push it out of reach
        env.pos[0, 1] = (2.0, 0.0, 0.0)
        env.vel[0, 1] = 0.0
        env.on_ground[0, 1] = True
    # hit lands on tick 1, is blocked for the 10-tick hurt window
    # (ticks 2..11), lands again on tick 12
    assert hits_landed == [True] + [False] * 10 + [True]
    assert abs(float(env.hp[0, 1]) - (20.0 - 2 * DMG)) < 1e-3


def test_critical_hit_multiplier():
    # attacker airborne and falling -> 1.5x damage
    env = make_env(p0=(0.0, 1.0, 0.0), p1=(2.0, 0.0, 0.0))
    env.on_ground[0, 0] = False
    env.vel[0, 0, 1] = -0.1
    env.pitch[0, 0] = -35.0  # look down at the target
    a = noop_actions()
    a[0, 0, 4] = 1
    env.step(a)
    assert abs(float(env.hp[0, 1]) - (20.0 - DMG_CRIT)) < 1e-3


def test_sprint_hit_extra_knockback_and_wtap_reset():
    env = make_env(p0=(0.0, 0.0, 0.0), p1=(2.5, 0.0, 0.0), yaw1=0.0)
    run = noop_actions()
    run[0, :, 0] = 2   # both hold forward (+x: chase / flee)
    run[0, :, 3] = 1   # both hold sprint
    obs, _, _, _ = env.step(run)
    assert obs_slot(obs, 0, "self_sprinting") == 1.0
    assert obs_slot(obs, 1, "self_sprinting") == 1.0

    swing = run.copy()
    swing[0, 0, 4] = 1
    obs, _, _, _ = env.step(swing)
    # sprint hit: 0.4 base + 0.5 extra horizontal knockback (attacker faces +x)
    assert float(env.vel[0, 1, 0]) > 0.75
    # W-tap mechanic: attacker's sprint resets on the hit...
    assert obs_slot(obs, 0, "self_sprinting") == 0.0
    # ...and damage breaks the victim's sprint
    assert obs_slot(obs, 1, "self_sprinting") == 0.0

    # still holding forward+sprint: sprint stays blocked
    obs, _, _, _ = env.step(run)
    assert obs_slot(obs, 0, "self_sprinting") == 0.0
    assert obs_slot(obs, 1, "self_sprinting") == 0.0

    # W-tap: release forward for one tick, then re-press -> sprint again
    release = run.copy()
    release[0, 0, 0] = 1
    env.step(release)
    obs, _, _, _ = env.step(run)
    assert obs_slot(obs, 0, "self_sprinting") == 1.0
    assert obs_slot(obs, 1, "self_sprinting") == 0.0  # victim never released


def test_sprint_toggle_spam_penalized():
    env = make_env(p0=(0.0, 0.0, -5.0), p1=(0.0, 0.0, 5.0))
    hold = noop_actions()
    hold[0, 0, 0] = 2
    toggle_on = hold.copy()
    toggle_on[0, 0, 3] = 1
    env.step(hold)
    _, rew_toggle, _, _ = env.step(toggle_on)   # sprint key changed 0 -> 1
    _, rew_hold, _, _ = env.step(toggle_on)     # unchanged
    assert rew_toggle[0, 0] < rew_hold[0, 0] - 1e-4


# ---------------------------------------------------------------------------
# observations
# ---------------------------------------------------------------------------

def test_aim_err_zero_when_looking_at_enemy():
    # enemy at bearing atan2(4, 3) = 53.13 deg, dist 5
    env = make_env(p0=(0.0, 0.0, 0.0), p1=(3.0, 0.0, 4.0),
                   yaw0=np.degrees(np.arctan2(4.0, 3.0)))
    obs, _, _, _ = env.step(noop_actions())
    assert abs(obs_slot(obs, 0, "aim_err_yaw")) < 1e-3
    # enemy centre (chest) is below own eye -> must look slightly down
    assert -0.15 < obs_slot(obs, 0, "aim_err_pitch") < 0.0
    # rel_pos in the self-yaw frame: straight ahead, 5 blocks, /8
    s = OBS_LAYOUT["rel_pos"][0]
    rel = obs[0, 0, s : s + 3]
    assert abs(rel[0] - 5.0 / 8.0) < 1e-3
    assert abs(rel[1]) < 1e-6
    assert abs(rel[2]) < 1e-3
    assert abs(obs_slot(obs, 0, "dist") - 5.0 / 8.0) < 1e-3
    assert obs_slot(obs, 0, "in_reach") == 0.0  # 5 > 3


def test_in_reach_and_hp_obs():
    env = make_env(p0=(0.0, 0.0, 0.0), p1=(2.0, 0.0, 0.0))
    env.hp[0, 1] = 10.0
    obs, _, _, _ = env.step(noop_actions())
    assert obs_slot(obs, 0, "in_reach") == 1.0
    assert obs_slot(obs, 1, "in_reach") == 1.0
    assert abs(obs_slot(obs, 0, "enemy_hp") - 0.5) < 1e-6
    assert abs(obs_slot(obs, 1, "self_hp") - 0.5) < 1e-6
    assert obs_slot(obs, 0, "self_on_ground") == 1.0


def test_camera_bins_update_view():
    env = make_env()
    a = noop_actions()
    a[0, 0, 5] = CAMERA_BINS.index(15.0)
    a[0, 0, 6] = CAMERA_BINS.index(-7.0)
    env.step(a)
    assert abs(float(env.yaw[0, 0]) - 15.0) < 1e-4
    assert abs(float(env.pitch[0, 0]) + 7.0) < 1e-4
    # pitch clamps at +-90
    a[0, 0, 6] = CAMERA_BINS.index(-30.0)
    for _ in range(10):
        env.step(a)
    assert float(env.pitch[0, 0]) == -90.0


def test_win_loss_on_death():
    env = make_env(p0=(0.0, 0.0, 0.0), p1=(2.0, 0.0, 0.0))
    env.hp[0, 1] = 1.0  # next hit kills
    a = noop_actions()
    a[0, 0, 4] = 1
    _, rew, done, info = env.step(a)
    assert done[0]
    assert info["win"][0, 0] == 1.0 and info["win"][0, 1] == 0.0
    assert rew[0, 0] > 10.0  # +1 hit +10 win - shaping
    assert rew[0, 1] < -10.0
    # auto-reset happened
    assert float(env.hp[0, 1]) == 20.0 and int(env.ticks[0]) == 0


def test_reward_config_is_used():
    rc = RewardConfig(hit=5.0, hurt=-2.0, aim_coeff=0.0, sprint_toggle_coeff=0.0)
    env = DuelVecEnv(1, seed=0, config=BIG, reward=rc)
    env.reset()
    env.pos[0, 0] = (0.0, 0.0, 0.0)
    env.pos[0, 1] = (2.0, 0.0, 0.0)
    env.vel[:] = 0.0
    env.yaw[0] = (0.0, 180.0)
    env.pitch[:] = 0.0
    env.on_ground[:] = True
    a = noop_actions()
    a[0, 0, 4] = 1
    _, rew, _, _ = env.step(a)
    assert abs(rew[0, 0] - 5.0) < 1e-5
    assert abs(rew[0, 1] + 2.0) < 1e-5
