"""Throughput benchmark for the vectorized duel env.

Usage:
    python3 -m pvpbot.sim.bench [--envs 4096] [--steps 300] [--seed 0]

Reports env-steps/sec = num_envs * step_calls / wall_time. Target is
>= 1M env-steps/sec at 4096 envs. Action batches are pre-generated so
RNG cost is excluded from the timed loop (the trainer samples actions
on its own budget).
"""
import argparse
import time
from typing import List

import numpy as np

from pvpbot.sim.env import DuelVecEnv
from pvpbot.spec import ACTION_HEAD_SIZES


def _action_pool(
    rng: np.random.Generator, num_envs: int, count: int
) -> List[np.ndarray]:
    return [
        np.stack(
            [rng.integers(0, n, (num_envs, 2)) for n in ACTION_HEAD_SIZES], axis=-1
        )
        for _ in range(count)
    ]


def run(num_envs: int = 4096, steps: int = 300, seed: int = 0, warmup: int = 30):
    """Returns (env_steps_per_sec, seconds_per_step_call)."""
    env = DuelVecEnv(num_envs, seed=seed)
    env.reset()
    pool = _action_pool(np.random.default_rng(seed + 1), num_envs, 32)
    for i in range(warmup):
        env.step(pool[i % len(pool)])
    t0 = time.perf_counter()
    for i in range(steps):
        env.step(pool[i % len(pool)])
    dt = time.perf_counter() - t0
    return num_envs * steps / dt, dt / steps


def main() -> None:
    ap = argparse.ArgumentParser(description="DuelVecEnv throughput benchmark")
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sps, spc = run(args.envs, args.steps, args.seed)
    print("num_envs=%d  step_calls=%d" % (args.envs, args.steps))
    print(
        "%.2fM env-steps/sec  (%.3f ms per step call)"
        % (sps / 1e6, spc * 1e3)
    )
    target = 1_000_000
    print("target 1.00M env-steps/sec: %s" % ("PASS" if sps >= target else "MISS"))


if __name__ == "__main__":
    main()
