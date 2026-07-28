"""End-to-end smoke: short self-play training run on the stub env."""
import os

import numpy as np

from pvpbot.sim.stub import DuelVecEnv as StubEnv
from pvpbot.train import run


def test_env_import_prefers_real_sim():
    # with pvpbot.sim.env merged, make_env must resolve to the real sim,
    # while still satisfying the shared env contract
    from pvpbot.sim.env import DuelVecEnv as RealEnv

    env = run.make_env(4, seed=0)
    assert isinstance(env, RealEnv)
    assert not isinstance(env, StubEnv)
    obs = env.reset()
    assert obs.shape == (4, 2, 48)


def test_cli_defaults_sane():
    args = run.build_parser().parse_args([])
    assert args.num_envs == 4096
    assert int(args.total_steps) == 10 ** 9
    assert args.out == "runs/exp1"
    assert args.chunk_len == 16
    assert args.threads == 12


def test_smoke_training_run(tmp_path):
    out = str(tmp_path / "smoke")
    # 64 envs x 32-step rollouts x 44 updates = 90112 env steps (~1400 ticks,
    # past the stub's 1200-tick episode cap so episodes complete and the
    # league records games against the pool).
    args = run.build_parser().parse_args([
        "--num-envs", "64",
        "--total-steps", "90112",
        "--rollout-len", "32",
        "--chunk-len", "8",
        "--epochs", "2",
        "--minibatches", "4",
        "--out", out,
        "--threads", "6",
        "--seed", "1",
        "--gate-every", "16",
        "--ckpt-every", "16",
        "--log-every", "1000",
    ])
    summary = run.train(args)

    history = summary["history"]
    assert summary["step"] >= 90112
    assert len(history) == summary["updates"] == 44

    # never NaN/inf anywhere in the optimization metrics
    for row in history:
        for key in (
            "loss", "pg_loss", "v_loss", "entropy", "approx_kl",
            "clip_frac", "grad_norm", "steps_per_sec",
        ):
            assert np.isfinite(row[key]), "%s not finite: %r" % (key, row[key])

    # throughput floor (end-to-end incl. PPO updates, tiny 64-env config)
    assert summary["steps_per_sec"] > 10_000, summary["steps_per_sec"]

    # episodes completed; reward / pool win-rate become measurable
    last = history[-1]
    assert last["episodes"] > 0
    assert np.isfinite(last["mean_ep_reward"])
    assert np.isfinite(last["win_rate_pool"])
    assert 0.0 <= last["win_rate_pool"] <= 1.0
    assert last["pool_size"] >= 1

    # artifacts on disk: metrics logs + spec-format checkpoint
    assert os.path.exists(os.path.join(out, "metrics.csv"))
    assert os.path.exists(os.path.join(out, "metrics.jsonl"))
    assert os.path.exists(os.path.join(out, "ckpt_latest.pt"))
    with open(os.path.join(out, "metrics.csv")) as f:
        assert len(f.readlines()) == 45  # header + one row per update

    # resume must pick up where we left off without error
    args2 = run.build_parser().parse_args([
        "--num-envs", "64",
        "--total-steps", str(90112 + 64 * 32 * 2),
        "--rollout-len", "32",
        "--chunk-len", "8",
        "--epochs", "1",
        "--minibatches", "4",
        "--out", out,
        "--threads", "6",
        "--seed", "2",
        "--log-every", "1000",
        "--resume", os.path.join(out, "ckpt_latest.pt"),
    ])
    summary2 = run.train(args2)
    assert summary2["updates"] == 46  # 44 restored + 2 more
    assert summary2["step"] >= 90112 + 64 * 32 * 2
