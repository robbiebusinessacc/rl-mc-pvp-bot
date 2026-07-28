"""Synthetic "human-like" demonstrations from the stub env.

No real human recordings exist yet, so this rolls ``pvpbot.sim.stub.DuelVecEnv``
with a scripted noisy aim-and-approach controller and writes episodes through
``pvpbot.bc.recording`` -- giving BC something real to train on end-to-end,
and doubling as the recording format's integration test.

The controller is deliberately imperfect in human-shaped ways:

  * proportional aim on the yaw error with gain > 1 (overshoot) plus
    Gaussian noise, acting on the error from ``reaction_ticks`` ago;
  * approach logic with occasional hesitation, backing off when too close;
  * persistent strafe direction that flips at random;
  * rare jumps; attacks only when in reach and roughly aligned, with a
    randomized click cooldown mimicking human click cadence.

That yields the label skew real recordings will have (mostly hold-W, ~5-10%
attack ticks), which is exactly what dataset.py's imbalance weights exist
to handle.

CLI::

    python3 -m pvpbot.bc.synth_demos --out-dir data/synth --episodes 16 --seed 0
"""
import argparse
import os
from collections import deque
from typing import Deque, List, Optional

import numpy as np

from pvpbot.sim.stub import DuelVecEnv
from pvpbot.spec import CAMERA_BINS, OBS_LAYOUT, TICK_RATE

from pvpbot.bc.recording import RecordingWriter, mouse_to_bins

_AIM_YAW = OBS_LAYOUT["aim_err_yaw"][0]
_DIST = OBS_LAYOUT["dist"][0]
_IN_REACH = OBS_LAYOUT["in_reach"][0]

SOURCE_TAG = "synthetic-scripted-v1"
MAP_TAG = "stub-flat"


class ScriptedDuelist:
    """Noisy aim-and-approach controller mapping stub obs -> action vector."""

    def __init__(
        self,
        rng: np.random.Generator,
        aim_gain: float = 1.25,
        aim_noise_deg: float = 1.2,
        reaction_ticks: int = 3,
        strafe_switch_p: float = 0.06,
        hesitate_p: float = 0.05,
        jump_p: float = 0.02,
        pitch_noise_p: float = 0.15,
        attack_p: float = 0.9,
        attack_align_deg: float = 15.0,
    ):
        self.rng = rng
        self.aim_gain = aim_gain
        self.aim_noise_deg = aim_noise_deg
        self.reaction_ticks = max(0, int(reaction_ticks))
        self.strafe_switch_p = strafe_switch_p
        self.hesitate_p = hesitate_p
        self.jump_p = jump_p
        self.pitch_noise_p = pitch_noise_p
        self.attack_p = attack_p
        self.attack_align_deg = attack_align_deg
        self._err_buf: Deque[float] = deque(maxlen=self.reaction_ticks + 1)
        self._issued: Deque[float] = deque(maxlen=max(1, self.reaction_ticks))
        self._strafe = 1
        self._click_cooldown = 0

    def reset(self) -> None:
        self._err_buf.clear()
        self._issued.clear()
        self._strafe = int(self.rng.choice([0, 1, 2]))
        self._click_cooldown = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        """obs: one player's (OBS_DIM,) row from the stub env."""
        yaw_err = float(obs[_AIM_YAW]) * 180.0  # de-normalize, degrees
        dist = float(obs[_DIST]) * 8.0
        in_reach = obs[_IN_REACH] > 0.5

        # reaction delay: act on the error from `reaction_ticks` ago, but
        # discount the turns we already issued inside the delay window
        # (humans track their own hand motion; without this the controller
        # rings between the extreme bins forever)
        self._err_buf.append(yaw_err)
        delayed_err = self._err_buf[0]
        effective_err = delayed_err - sum(self._issued)

        # proportional aim with overshoot (gain > 1) + noise, quantized to bins
        cmd = self.aim_gain * effective_err + self.rng.normal(0.0, self.aim_noise_deg)
        cmd = float(np.clip(cmd, -45.0, 45.0))
        pitch_cmd = 0.0
        if self.rng.random() < self.pitch_noise_p:
            pitch_cmd = float(self.rng.normal(0.0, 0.8))
        yaw_idx, pitch_idx = mouse_to_bins(cmd, pitch_cmd)
        self._issued.append(CAMERA_BINS[yaw_idx])

        # approach: close distance, back off when point-blank, hesitate rarely
        if dist > 2.4:
            forward = 2
        elif dist < 1.0:
            forward = 0
        else:
            forward = 1
        if self.rng.random() < self.hesitate_p:
            forward = 1
        sprint = 1 if (forward == 2 and dist > 3.0) else 0

        # persistent strafe with random switches
        if self.rng.random() < self.strafe_switch_p:
            self._strafe = int(self.rng.choice([0, 1, 2], p=[0.4, 0.2, 0.4]))
        strafe = self._strafe

        jump = 1 if self.rng.random() < self.jump_p else 0

        # attack: in reach, roughly aligned, cadence-limited, imperfect
        attack = 0
        if self._click_cooldown > 0:
            self._click_cooldown -= 1
        elif (
            in_reach
            and abs(delayed_err) < self.attack_align_deg
            and self.rng.random() < self.attack_p
        ):
            attack = 1
            self._click_cooldown = int(self.rng.integers(3, 7))

        return np.array(
            [forward, strafe, jump, sprint, attack, yaw_idx, pitch_idx],
            dtype=np.int64,
        )


def generate_demos(
    out_dir: str,
    episodes: int = 8,
    seed: int = 0,
    max_ticks: int = 300,
    record_both_sides: bool = True,
) -> List[str]:
    """Roll scripted duels in the stub env and write episode recordings.

    Deterministic: the same arguments produce identical recorded arrays.
    Records side 0's (obs, action) stream, plus side 1's as a separate file
    when ``record_both_sides`` (both sides are scripted, so both streams are
    valid demonstrations). Returns the list of written paths.
    """
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    if max_ticks < 2:
        raise ValueError("max_ticks must be >= 2")
    os.makedirs(out_dir, exist_ok=True)
    env = DuelVecEnv(num_envs=1, seed=seed)
    players = [
        ScriptedDuelist(np.random.default_rng(seed * 7919 + 1)),
        ScriptedDuelist(
            np.random.default_rng(seed * 7919 + 2),
            aim_gain=1.1,
            aim_noise_deg=2.0,
            reaction_ticks=4,
            attack_p=0.8,
        ),
    ]
    sides = (0, 1) if record_both_sides else (0,)
    obs = env.reset()
    paths: List[str] = []
    for ep in range(episodes):
        writers = {
            side: RecordingWriter(
                os.path.join(out_dir, "ep%03d_s%d.npz" % (ep, side)),
                tick_rate=TICK_RATE,
                source=SOURCE_TAG,
                map_name=MAP_TAG,
                extra_meta={"episode": ep, "side": side, "seed": seed},
            )
            for side in sides
        }
        for p in players:
            p.reset()
        truncated = True
        for _ in range(max_ticks):
            acts = np.stack(
                [players[0].act(obs[0, 0]), players[1].act(obs[0, 1])]
            )
            for side in sides:
                writers[side].append(obs[0, side], acts[side])
            obs, _rew, done, _info = env.step(acts[None])
            if done[0]:
                truncated = False  # env auto-reset; obs is already fresh
                break
        if truncated:
            obs = env.reset()  # don't bleed a half-finished duel into the next file
        for side in sides:
            paths.append(writers[side].finalize())
    return paths


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate synthetic human-like PvP demonstrations."
    )
    ap.add_argument("--out-dir", required=True, help="directory for episode NPZs")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-ticks", type=int, default=300)
    ap.add_argument(
        "--one-side", action="store_true", help="record only side 0 of each duel"
    )
    args = ap.parse_args(argv)
    paths = generate_demos(
        out_dir=args.out_dir,
        episodes=args.episodes,
        seed=args.seed,
        max_ticks=args.max_ticks,
        record_both_sides=not args.one_side,
    )
    total = 0
    for p in paths:
        with np.load(p) as d:
            total += d["obs"].shape[0]
    print("wrote %d recordings (%d ticks) to %s" % (len(paths), total, args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
