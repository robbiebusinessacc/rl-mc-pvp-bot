"""FOV-gated observations + perception noise (SimConfig.fov_limited_obs)."""
import numpy as np

from pvpbot.sim.env import DuelVecEnv, SimConfig
from pvpbot.spec import CAMERA_BINS, OBS_LAYOUT

_Z = CAMERA_BINS.index(0.0)


def idle(n):
    a = np.zeros((n, 2, 7), dtype=np.int64)
    a[:, :, :2] = 1
    a[:, :, 5] = _Z
    a[:, :, 6] = _Z
    return a


def slot(obs, env_i, name):
    s, e = OBS_LAYOUT[name]
    v = obs[env_i, 0, s:e]
    return float(v[0]) if e - s == 1 else v


def make(n=2, noise=0.0, fov=True, seed=0):
    env = DuelVecEnv(n, seed=seed, config=SimConfig(
        fov_limited_obs=fov, obs_noise=noise,
        arena_radius=100.0, spawn_radius=1.0))
    env.reset()
    for e in range(n):
        env.pos[e, 0] = (0.0, 0.0, 0.0)
        env.pos[e, 1] = (4.0, 0.0, 0.0)
        env.vel[e] = 0.0
        env.pitch[e] = 0.0
        env.yaw[e, 1] = 180.0
        env.yaw[e, 0] = 0.0     # sim yaw 0 faces +x = toward the enemy
    return env


def test_facing_enemy_is_visible_and_exact():
    env = make()
    obs, _, _, _ = env.step(idle(2))
    assert slot(obs, 0, "enemy_visible") == 1.0
    assert abs(slot(obs, 0, "aim_err_yaw")) < 0.02
    assert abs(slot(obs, 0, "dist") - 0.5) < 0.02          # 4 blocks /8


def test_facing_away_gates_enemy_slots():
    env = make()
    env.yaw[1, 0] = 180.0        # env 1: look away from the enemy
    obs, _, _, _ = env.step(idle(2))
    assert slot(obs, 1, "enemy_visible") == 0.0
    assert slot(obs, 1, "aim_err_yaw") == 0.0
    assert slot(obs, 1, "aim_err_pitch") == 0.0
    assert slot(obs, 1, "in_reach") == 0.0
    assert slot(obs, 1, "enemy_on_ground") == 0.0
    assert abs(slot(obs, 1, "dist") - 2.0) < 1e-6          # never-seen default
    assert np.allclose(slot(obs, 1, "rel_pos"), 0.0)


def test_hold_last_seen_after_turning_away():
    env = make(n=1)
    obs, _, _, _ = env.step(idle(1))
    d_seen = slot(obs, 0, "dist")
    rel_seen = slot(obs, 0, "rel_pos").copy()
    assert abs(d_seen - 0.5) < 0.02
    env.yaw[0, 0] = 180.0        # snap the camera away
    obs, _, _, _ = env.step(idle(1))
    assert slot(obs, 0, "enemy_visible") == 0.0
    assert abs(slot(obs, 0, "dist") - d_seen) < 0.02       # held, not reset
    assert np.allclose(slot(obs, 0, "rel_pos"), rel_seen, atol=0.02)
    assert slot(obs, 0, "aim_err_yaw") == 0.0


def test_noise_perturbs_visible_only():
    env = make(n=64, noise=1.0, seed=3)
    for e in range(32, 64):
        env.yaw[e, 0] = 180.0    # half the envs look away
    obs, _, _, _ = env.step(idle(64))
    s, _ = OBS_LAYOUT["aim_err_yaw"]
    vis_col = OBS_LAYOUT["enemy_visible"][0]
    vis = obs[:, 0, vis_col] > 0.5
    away = np.zeros(64, dtype=bool)
    away[32:] = True
    assert not vis[away].any()
    # visible rows carry noise (true err is ~0 for all of them)
    assert np.abs(obs[vis, 0, s]).std() > 0.02
    # invisible rows are EXACT zeros, like the live adapter emits
    assert (obs[~vis, 0, s] == 0.0).all()


def test_default_config_keeps_xray_and_visible_flag():
    env = DuelVecEnv(4, seed=0)
    obs = env.reset()
    vis_col = OBS_LAYOUT["enemy_visible"][0]
    assert (obs[:, :, vis_col] == 1.0).all()
