"""Minimal stub implementation of the DuelVecEnv API from pvpbot/spec.py.

Exists so the trainer, eval ladder, and BC modules can develop and test
against the exact env contract before the real physics sim lands.
Physics here are deliberately crude; only the API shape is authoritative.
"""
from typing import Dict, Tuple

import numpy as np

from pvpbot.spec import (
    ACTION_HEAD_SIZES,
    CAMERA_BINS,
    NUM_ACTION_HEADS,
    OBS_DIM,
    OBS_LAYOUT,
)


class DuelVecEnv:
    REACH = 3.0
    MAX_TICKS = 1200  # 60s episode cap

    def __init__(self, num_envs: int, seed: int = 0):
        self.num_envs = num_envs
        self.rng = np.random.default_rng(seed)
        self._alloc()

    def _alloc(self):
        n = self.num_envs
        self.pos = np.zeros((n, 2, 3), dtype=np.float32)
        self.vel = np.zeros((n, 2, 3), dtype=np.float32)
        self.yaw = np.zeros((n, 2), dtype=np.float32)
        self.pitch = np.zeros((n, 2), dtype=np.float32)
        self.hp = np.zeros((n, 2), dtype=np.float32)
        self.hurt = np.zeros((n, 2), dtype=np.int32)
        self.ticks = np.zeros(n, dtype=np.int32)
        self.prev_actions = np.zeros((n, 2, NUM_ACTION_HEADS), dtype=np.int64)

    def _reset_envs(self, mask: np.ndarray):
        k = int(mask.sum())
        if k == 0:
            return
        self.pos[mask] = self.rng.uniform(-4, 4, (k, 2, 3)).astype(np.float32)
        self.pos[mask][:, :, 1] = 0.0
        self.vel[mask] = 0.0
        self.yaw[mask] = self.rng.uniform(0, 360, (k, 2)).astype(np.float32)
        self.pitch[mask] = 0.0
        self.hp[mask] = 20.0
        self.hurt[mask] = 0
        self.ticks[mask] = 0
        self.prev_actions[mask] = 0

    def reset(self) -> np.ndarray:
        self._reset_envs(np.ones(self.num_envs, dtype=bool))
        return self._obs()

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        assert actions.shape == (self.num_envs, 2, NUM_ACTION_HEADS)
        bins = np.asarray(CAMERA_BINS, dtype=np.float32)
        self.yaw = (self.yaw + bins[actions[:, :, 5]]) % 360.0
        self.pitch = np.clip(self.pitch + bins[actions[:, :, 6]], -90, 90)

        fwd = actions[:, :, 0].astype(np.float32) - 1.0
        strafe = actions[:, :, 1].astype(np.float32) - 1.0
        sprint = 1.0 + 0.3 * actions[:, :, 3].astype(np.float32)
        rad = np.deg2rad(self.yaw)
        ax = (fwd * np.cos(rad) - strafe * np.sin(rad)) * 0.1 * sprint
        az = (fwd * np.sin(rad) + strafe * np.cos(rad)) * 0.1 * sprint
        self.vel[:, :, 0] = self.vel[:, :, 0] * 0.546 + ax
        self.vel[:, :, 2] = self.vel[:, :, 2] * 0.546 + az
        on_ground = self.pos[:, :, 1] <= 0.0
        self.vel[:, :, 1] = np.where(
            (actions[:, :, 2] == 1) & on_ground, 0.42, self.vel[:, :, 1] - 0.08
        )
        self.pos += self.vel
        self.pos[:, :, 1] = np.maximum(self.pos[:, :, 1], 0.0)

        diff = self.pos[:, 1] - self.pos[:, 0]
        dist = np.linalg.norm(diff, axis=1)
        swing = actions[:, :, 4] == 1
        in_reach = dist < self.REACH
        # a hit lands if swinging, target not in hurt-time
        target_hurt = self.hurt[:, ::-1]
        hits = swing & in_reach[:, None] & (target_hurt == 0)
        self.hurt = np.maximum(self.hurt - 1, 0)
        self.hurt[:, ::-1][hits] = 10
        dmg = np.zeros_like(self.hp)
        dmg[:, ::-1][hits] = 1.0
        self.hp -= dmg

        rew = hits.astype(np.float32) - dmg
        self.ticks += 1
        dead = self.hp <= 0.0
        done = dead.any(axis=1) | (self.ticks >= self.MAX_TICKS)
        win = np.zeros((self.num_envs, 2), dtype=np.float32)
        win[dead[:, ::-1] & ~dead] = 1.0
        rew += 10.0 * win * done[:, None]

        self.prev_actions = actions.copy()
        info = {"win": win, "dist": dist}
        self._reset_envs(done)
        return self._obs(), rew, done, info

    def _obs(self) -> np.ndarray:
        n = self.num_envs
        obs = np.zeros((n, 2, OBS_DIM), dtype=np.float32)
        for side in (0, 1):
            other = 1 - side
            rel = self.pos[:, other] - self.pos[:, side]
            rad = np.deg2rad(self.yaw[:, side])
            cos, sin = np.cos(rad), np.sin(rad)
            rx = rel[:, 0] * cos + rel[:, 2] * sin
            rz = -rel[:, 0] * sin + rel[:, 2] * cos
            dist = np.linalg.norm(rel, axis=1)
            s, e = OBS_LAYOUT["rel_pos"]
            obs[:, side, s:e] = np.stack([rx, rel[:, 1], rz], axis=1) / 8.0
            s, e = OBS_LAYOUT["dist"]
            obs[:, side, s] = dist / 8.0
            s, e = OBS_LAYOUT["in_reach"]
            obs[:, side, s] = (dist < self.REACH).astype(np.float32)
            s, e = OBS_LAYOUT["self_hp"]
            obs[:, side, s] = self.hp[:, side] / 20.0
            s, e = OBS_LAYOUT["enemy_hp"]
            obs[:, side, s] = self.hp[:, other] / 20.0
            s, e = OBS_LAYOUT["self_hurt"]
            obs[:, side, s] = self.hurt[:, side] / 10.0
            s, e = OBS_LAYOUT["enemy_hurt"]
            obs[:, side, s] = self.hurt[:, other] / 10.0
            tgt_yaw = np.rad2deg(np.arctan2(rel[:, 2], rel[:, 0]))
            err = (tgt_yaw - self.yaw[:, side] + 180.0) % 360.0 - 180.0
            s, e = OBS_LAYOUT["aim_err_yaw"]
            obs[:, side, s] = err / 180.0
            s, e = OBS_LAYOUT["prev_action"]
            obs[:, side, s:e] = self.prev_actions[:, side].astype(np.float32) / np.asarray(
                ACTION_HEAD_SIZES, dtype=np.float32
            )
            s, e = OBS_LAYOUT["enemy_visible"]
            obs[:, side, s] = 1.0  # stub keeps X-ray vision (crude by design)
        return obs
