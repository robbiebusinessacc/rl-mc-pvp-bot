"""DAgger-style dataset collector for future end-to-end distillation.

Rolls the stub DuelVecEnv, derives side-0's ground-truth viewpoint from env
internals each tick, renders a matching synthetic frame via synth.render_scene,
and yields (frame, obs, action) triples:

    frame:  uint8 (1, FRAME_H, FRAME_W)  -- what a pixel policy would see
    obs:    float32 (OBS_DIM,)           -- privileged obs the teacher saw
    action: int64 (NUM_ACTION_HEADS,)    -- action taken this tick

Stub quality by design: the point is the plumbing (true state -> synth scene
params) so a distillation trainer can consume this the day it exists.
"""
from typing import Callable, Iterator, Optional, Tuple

import numpy as np

from pvpbot.sim.stub import DuelVecEnv
from pvpbot.spec import ACTION_HEAD_SIZES, NUM_ACTION_HEADS
from pvpbot.perception.synth import (
    PX_PER_DEG,
    SceneParams,
    SynthConfig,
    render_scene,
)


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def scene_params_from_env(
    env: DuelVecEnv, env_idx: int = 0, side: int = 0,
    prev_yaw_err: Optional[float] = None,
    prev_pitch_err: Optional[float] = None,
) -> Tuple[SceneParams, float]:
    """Ground-truth stub state -> (SceneParams, pitch_err_deg) for `side`."""
    other = 1 - side
    rel = env.pos[env_idx, other] - env.pos[env_idx, side]
    dist = float(np.linalg.norm(rel) + 1e-6)
    tgt_yaw = np.degrees(np.arctan2(rel[2], rel[0]))
    yaw_err = _wrap_deg(float(tgt_yaw - env.yaw[env_idx, side]))
    # screen velocity from the change in aim error (px/tick)
    vx = 0.0 if prev_yaw_err is None else _wrap_deg(yaw_err - prev_yaw_err) * PX_PER_DEG
    self_pitch = float(env.pitch[env_idx, side])
    pitch_to_enemy = float(np.degrees(np.arctan2(
        -(rel[1] + 0.9) + 1.62, max(dist, 0.5))))  # down-positive
    pitch_err = pitch_to_enemy - self_pitch
    vy = 0.0 if prev_pitch_err is None else (pitch_err - prev_pitch_err) * PX_PER_DEG
    speed = float(np.linalg.norm(env.vel[env_idx, side, [0, 2]]))
    return SceneParams(
        self_pitch_deg=self_pitch,
        yaw_err_deg=yaw_err,
        dist=float(np.clip(dist, 0.6, 24.0)),
        enemy_y=float(max(env.pos[env_idx, other, 1], 0.0)),
        self_hp=float(env.hp[env_idx, side]),
        hurt_flash=bool(env.hurt[env_idx, other] >= 9),  # freshly hit
        screen_vx_px=float(np.clip(vx, -8.0, 8.0)),
        screen_vy_px=float(np.clip(vy, -6.0, 6.0)),
        self_speed=float(np.clip(speed, 0.0, 0.35)),
        occluded=False,
        n_distractors=2,
    ), pitch_err


def _random_policy(rng: np.random.Generator):
    def act(obs: np.ndarray) -> np.ndarray:
        n = obs.shape[0]
        return np.stack(
            [rng.integers(0, k, (n, 2)) for k in ACTION_HEAD_SIZES], axis=-1
        ).astype(np.int64)
    return act


def collect(
    ticks: int = 100,
    seed: int = 0,
    policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    cfg: Optional[SynthConfig] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield `ticks` (frame, obs, action) triples from side 0 of one stub duel.

    policy: obs (n, 2, OBS_DIM) -> actions (n, 2, NUM_ACTION_HEADS); random
    actions if None (DAgger's roll-in policy slot).
    """
    if cfg is None:
        cfg = SynthConfig()
    rng = np.random.default_rng(seed)
    if policy is None:
        policy = _random_policy(rng)
    env = DuelVecEnv(1, seed=seed)
    obs = env.reset()
    prev_yaw_err = None
    prev_pitch_err = None
    for _ in range(ticks):
        params, pitch_err = scene_params_from_env(
            env, 0, 0, prev_yaw_err=prev_yaw_err, prev_pitch_err=prev_pitch_err)
        prev_yaw_err = params.yaw_err_deg
        prev_pitch_err = pitch_err
        frame, _label = render_scene(
            params, np.random.default_rng(int(rng.integers(2 ** 31))), cfg)
        actions = policy(obs)
        assert actions.shape == (1, 2, NUM_ACTION_HEADS)
        from pvpbot.spec import FRAME_SHAPE as _FS
        yield frame.transpose(2, 0, 1).copy(), \
            obs[0, 0].astype(np.float32).copy(), \
            actions[0, 0].astype(np.int64).copy()
        obs, _rew, done, _info = env.step(actions)
        if done[0]:
            prev_yaw_err = None
            prev_pitch_err = None


def collect_arrays(ticks: int = 100, seed: int = 0,
                   cfg: Optional[SynthConfig] = None):
    """Materialize collect() into stacked arrays (frames, obs, actions)."""
    frames, obses, acts = [], [], []
    for f, o, a in collect(ticks=ticks, seed=seed, cfg=cfg):
        frames.append(f)
        obses.append(o)
        acts.append(a)
    return (np.stack(frames), np.stack(obses).astype(np.float32),
            np.stack(acts).astype(np.int64))
