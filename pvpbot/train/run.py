"""Self-play PPO training CLI.

Usage:
    python3 -m pvpbot.train.run --num-envs 4096 --total-steps 1e9 --out runs/exp1

Uses the real sim (pvpbot.sim.env.DuelVecEnv) when present, else falls back
to the stub (pvpbot.sim.stub.DuelVecEnv). Writes periodic checkpoints in the
spec.py format plus metrics.csv / metrics.jsonl into --out. Resume with
--resume path/to/ckpt.pt.
"""
import argparse
import csv
import json
import os
import time
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import torch

from pvpbot.models import PolicyNet
from pvpbot.spec import ACTION_HEADS, OBS_DIM
from pvpbot.train.league import League
from pvpbot.train.ppo import PPOConfig, PPOTrainer, RolloutBuffer, policy_step

CSV_FIELDS = [
    "update", "step", "steps_per_sec", "mean_ep_reward", "win_rate_pool",
    "elo", "entropy", "loss", "pg_loss", "v_loss", "approx_kl", "clip_frac",
    "grad_norm", "pool_size", "episodes",
]


def make_env(num_envs: int, seed: int = 0, fov_obs: bool = False,
             obs_noise: float = 0.0, obs_delay: int = 0,
             aim_exec_noise: float = 0.0, arena_radius: float = 8.0,
             rel_vel_noise: float = 0.0, rel_pos_y_bias: float = 0.0,
             act_delay: int = 0, act_hold: int = 0, hit_reg: float = 1.0,
             vis_miss: float = 0.0, vis_reacq: float = 0.0,
             cam_assist: int = 0, punish_close: float = 0.0,
             punish_swing: float = 0.0, mask_dead_senses: int = 0,
             arena_square: int = 0):
    """Real sim when available; stub otherwise (drop-in same API)."""
    try:
        from pvpbot.sim.env import DuelVecEnv, RewardConfig, SimConfig
        return DuelVecEnv(num_envs, seed=seed, config=SimConfig(
            fov_limited_obs=fov_obs, obs_noise=obs_noise,
            obs_delay_ticks=obs_delay, aim_exec_noise_deg=aim_exec_noise,
            arena_radius=arena_radius, spawn_radius=min(arena_radius - 0.5, 6.0),
            arena_square=bool(arena_square),
            rel_vel_noise=rel_vel_noise, rel_pos_y_bias=rel_pos_y_bias,
            act_delay_ticks=act_delay, act_hold_ticks=act_hold,
            hit_reg_prob=hit_reg, vis_miss_prob=vis_miss,
            vis_reacq_prob=vis_reacq, cam_assist=bool(cam_assist),
            mask_dead_senses=bool(mask_dead_senses)),
            reward=RewardConfig(punish_close_bonus=punish_close,
                                punish_swing_bonus=punish_swing))
    except ImportError:
        from pvpbot.sim.stub import DuelVecEnv  # crude fallback (X-ray obs)
        return DuelVecEnv(num_envs, seed=seed)


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------
class Collector:
    """Steps the vec env: side 0 = learner, side 1 = league opponents."""

    def __init__(self, env, policy: PolicyNet, trainer: PPOTrainer, league: League):
        self.env = env
        self.policy = policy
        self.trainer = trainer
        self.league = league
        self.device = trainer.device
        self.obs = env.reset()
        self.h0 = policy.initial_state(env.num_envs, device=self.device)
        self.ep_ret = np.zeros(env.num_envs, dtype=np.float32)
        self.ep_returns = deque(maxlen=256)  # completed side-0 episode returns
        self.episodes = 0

    def rollout(self, buf: RolloutBuffer) -> None:
        norm = self.trainer.obs_norm
        policy = self.policy
        league = self.league
        for t in range(buf.T):
            raw0 = self.obs[:, 0]
            obs0 = norm.normalize(raw0)
            norm.update(raw0)
            obs0_t = torch.from_numpy(obs0).to(self.device)
            actions0, logp0, value0, h_new = policy_step(policy, obs0_t, self.h0)
            # scripted opponents get the ungated omniscient side-1 obs
            # (they model mineflayer bots with perfect server knowledge);
            # self-play/checkpoint opponents keep the camera-gated view.
            scripted_obs1 = getattr(self.env, "_obs_ungated", None)
            scripted_obs1 = scripted_obs1[:, 1] if scripted_obs1 is not None else None
            actions1 = league.opponent_actions(
                self.obs[:, 1], policy, norm, scripted_obs=scripted_obs1)
            acts = np.stack([actions0.cpu().numpy(), actions1], axis=1)
            obs, rew, done, info = self.env.step(acts)
            buf.add(
                t, obs0, self.h0, acts[:, 0], logp0.cpu().numpy(),
                value0.cpu().numpy(), rew[:, 0], done.astype(np.float32),
            )
            self.ep_ret += rew[:, 0]
            self.h0 = h_new
            if done.any():
                for i in np.nonzero(done)[0]:
                    self.ep_returns.append(float(self.ep_ret[i]))
                self.episodes += int(done.sum())
                self.ep_ret[done] = 0.0
                league.on_done(done, info["win"])
                mask = torch.from_numpy((~done).astype(np.float32)).to(self.device)
                self.h0 = self.h0 * mask.unsqueeze(-1)
            self.obs = obs
        # bootstrap value for the state after the last step
        obs0_t = torch.from_numpy(norm.normalize(self.obs[:, 0])).to(self.device)
        with torch.no_grad():
            _, last_v, _ = policy(obs0_t, self.h0)
        buf.compute_returns(
            last_v.cpu().numpy(), self.trainer.cfg.gamma, self.trainer.cfg.gae_lambda
        )


# ---------------------------------------------------------------------------
# Checkpointing (spec.py format: {"model": state_dict, "meta": {...}, ...})
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: str,
    policy: PolicyNet,
    trainer: PPOTrainer,
    league: League,
    step: int,
    update: int,
    pin_stage: Optional[int] = None,
) -> None:
    ckpt = {
        "model": policy.state_dict(),
        "meta": {
            "obs_dim": OBS_DIM,
            "action_heads": ACTION_HEADS,
            "step": int(step),
            "elo": float(league.learner_elo),
        },
        "optimizer": trainer.opt.state_dict(),
        "obs_norm": trainer.obs_norm.state_dict(),
        "league": league.state_dict(),
        "update": int(update),
    }
    if pin_stage is not None:
        ckpt["pin_stage"] = int(pin_stage)
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: str,
    policy: PolicyNet,
    trainer: Optional[PPOTrainer] = None,
    league: Optional[League] = None,
):
    """Restore training state. Returns (step, update, pin_stage-or-None)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(ckpt["model"])
    if trainer is not None:
        if "optimizer" in ckpt:
            trainer.opt.load_state_dict(ckpt["optimizer"])
        if "obs_norm" in ckpt:
            trainer.obs_norm.load_state_dict(ckpt["obs_norm"])
    if league is not None and "league" in ckpt:
        league.load_state_dict(ckpt["league"])
    stage = ckpt.get("pin_stage")
    return (int(ckpt["meta"]["step"]), int(ckpt.get("update", 0)),
            None if stage is None else int(stage))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> Dict[str, Any]:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    env = make_env(args.num_envs, seed=args.seed,
                   fov_obs=bool(args.fov_obs), obs_noise=args.obs_noise,
                   obs_delay=args.obs_delay, aim_exec_noise=args.aim_exec_noise,
                   arena_radius=args.arena_radius,
                   rel_vel_noise=args.rel_vel_noise,
                   rel_pos_y_bias=args.rel_pos_y_bias,
                   act_delay=args.act_delay, act_hold=args.act_hold,
                   hit_reg=args.hit_reg, vis_miss=args.vis_miss,
                   vis_reacq=args.vis_reacq, cam_assist=args.cam_assist,
                   punish_close=args.punish_close,
                   punish_swing=args.punish_swing,
                   mask_dead_senses=args.mask_dead_senses,
                   arena_square=args.arena_square)
    policy = PolicyNet()
    ec = args.ent_coef
    move_c = args.ent_coef_move if args.ent_coef_move >= 0 else ec
    cam_c = (args.ent_coef_cam if args.ent_coef_cam >= 0
             else (0.0 if args.cam_assist else ec))
    ent_coefs = None
    if move_c != ec or cam_c != ec:
        # head order: forward, strafe, jump, sprint, attack, yaw, pitch
        ent_coefs = (move_c, move_c, ec, move_c, ec, cam_c, cam_c)
        print("per-head ent coefs: move %.4f jump/attack %.4f cam %.4f"
              % (move_c, ec, cam_c))
    cfg = PPOConfig(
        lr=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip_range=args.clip_range, epochs=args.epochs,
        num_minibatches=args.minibatches, ent_coef=args.ent_coef,
        ent_coefs=ent_coefs,
        vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm,
        chunk_len=args.chunk_len, rollout_len=args.rollout_len,
    )
    trainer = PPOTrainer(policy, cfg, device=args.device)
    league = League(
        args.num_envs, device=args.device, p_self=args.p_self, elo_k=args.elo_k,
        max_pool=args.pool_size, gate_winrate=args.gate_winrate, seed=args.seed,
        pin_name=args.pin_opponent, pin_frac=args.pin_frac,
    )

    step, update, resume_stage = 0, 0, None
    if args.resume:
        step, update, resume_stage = load_checkpoint(
            args.resume, policy, trainer, league)
        print("resumed from %s at step %d (update %d)" % (args.resume, step, update))

    collector = Collector(env, policy, trainer, league)
    buf = RolloutBuffer(cfg.rollout_len, args.num_envs, chunk_len=cfg.chunk_len)
    total_steps = int(float(args.total_steps))
    steps_per_update = cfg.rollout_len * args.num_envs

    csv_path = os.path.join(args.out, "metrics.csv")
    write_header = not os.path.exists(csv_path)
    csv_f = open(csv_path, "a", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if write_header:
        csv_w.writeheader()
    jsonl_f = open(os.path.join(args.out, "metrics.jsonl"), "a")

    history = []
    # adaptive pin curriculum: the frame-perfect P4 (react 0, jitter 0.5)
    # is an unlearnable cliff from scratch (1 h flatline at wr~0.13);
    # start it imprecise and tighten 30% whenever the learner's recent
    # winrate vs the PIN clears the threshold. Floor = the true tier.
    # Stage table measured 2026-07-23 (ckpt 48B diagnostics): single knobs
    # are DEAD vs react:0 (jitter3/cps6/crit30/rot240 alone all ~3-4%);
    # only combined softening yields a learnable start (~17%). Reaction
    # stays 0 at every stage (Robbie's fresh-gear ruling).
    # hit_reg rides the same ladder: at 0.4 (the measured pipeline truth)
    # 60% of landed hits vanish, so movement-quality differences barely
    # move reward and spacing/strafe skills have no gradient to climb
    # (24B-step stall, wr_pin plateau 0.20 after the entropy fix). Start
    # dense, end at the honest 0.4 -- evals and the title set never move.
    # Interleaved: each rung tightens ONE dimension (opponent knobs OR
    # hit_reg), never both. The combined 5-knob step 2->3 of the first
    # table was a cliff (wr 0.42 -> 0.14, negative drift); single-dim
    # rungs keep every step climbable and make stalls diagnosable.
    _OPP = [
        dict(aim_jitter_deg=12.0, cps=6.0, crit_rate=0.3, rot_speed_dps=240.0),
        dict(aim_jitter_deg=8.0, cps=8.0, crit_rate=0.5, rot_speed_dps=360.0),
        dict(aim_jitter_deg=5.0, cps=9.0, crit_rate=0.7, rot_speed_dps=480.0),
        dict(aim_jitter_deg=2.5, cps=10.0, crit_rate=0.85, rot_speed_dps=540.0),
        # summit opponent = the OFFICIAL hacker bar (Robbie calibration
        # 2026-07-24)
        dict(aim_jitter_deg=1.5, cps=11.0, crit_rate=1.0, rot_speed_dps=600.0),
    ]
    # hit_reg column REFIT 2026-07-25 eve: with the truthful pipeline
    # (v11 + damped assist + pitch fix) live conversion sits at the
    # hurt-window cap -- hit theft is nearly gone. Summit 0.85
    # (conservative; the old 0.4 was fitted to the lying pipeline).
    # hit_reg column re-fit CLEAN 2026-07-25 late: live 2.16 hits per
    # in-range-second vs sim-raw 3.54 (same estimator both sides) ->
    # summit 0.61. The 0.85 came through a polluted denominator.
    CUR_STAGES = [
        dict(_OPP[0], hit_reg=0.80),
        dict(_OPP[1], hit_reg=0.80),
        dict(_OPP[1], hit_reg=0.72),
        dict(_OPP[2], hit_reg=0.72),
        dict(_OPP[2], hit_reg=0.66),
        dict(_OPP[3], hit_reg=0.66),
        dict(_OPP[3], hit_reg=0.61),
        dict(_OPP[4], hit_reg=0.61),
        dict(_OPP[4], hit_reg=0.61),  # = the clean-fit faithful config
    ]
    dr_rng = np.random.default_rng(args.seed + 777)

    def _randomize_pin(entry):
        # ranges bracket the OFFICIAL BAR (Robbie calibration): robustness
        # to the family of hackers is the only skill that transfers to
        # the real one
        entry.agent.aim_jitter_deg = float(dr_rng.uniform(1.0, 4.0))
        entry.agent.cps = float(dr_rng.uniform(9.0, 12.0))
        entry.agent.rot_speed_dps = float(dr_rng.uniform(450.0, 750.0))
        entry.agent.crit_rate = float(dr_rng.uniform(0.8, 1.0))
        entry.agent.wtap_rate = float(dr_rng.uniform(0.8, 1.0))
        entry.agent.strafe_rate = float(dr_rng.uniform(0.8, 1.0))
        entry.agent.reaction_ticks = int(dr_rng.integers(0, 2))
        entry.agent.engage_lo = float(dr_rng.uniform(1.8, 2.2))
        entry.agent.engage_hi = float(dr_rng.uniform(2.5, 3.1))
        entry.agent.chase_dist = float(dr_rng.uniform(3.0, 3.6))
        entry.agent.hurt_timed_swing = bool(dr_rng.random() < 0.7)

    def _resolve_pin():
        # NEVER trust a cached index across gate() calls: pool mutation
        # left _pin_idx pointing at a ckpt entry (agent=None) and crashed
        # a 12.5B-step run (2026-07-26). Resolve by name, every time.
        # Match the way League itself binds the pin: pool entries are named
        # "scripted_<agent>", so --pin-opponent P4-Hacker must match
        # scripted_P4-Hacker. An == test here silently disabled pin DR for
        # 1B steps (2026-07-26 -> 27) because dr_on went False without a word.
        want = (getattr(league, "_pin_name", None) or "").lower()
        if not want:
            return None
        for e in league.pool:
            if want in getattr(e, "name", "").lower() and e.agent is not None:
                return e
        return None

    if_dr = bool(args.interface_randomize)

    def _randomize_interface():
        env._hit_reg_p = float(dr_rng.uniform(0.40, 0.75))
        env._aim_exec_nz = float(dr_rng.uniform(2.5, 5.0))
        env._vis_miss = float(dr_rng.uniform(0.03, 0.15))
        env._vis_reacq = float(dr_rng.uniform(0.18, 0.28))

    if if_dr:
        _randomize_interface()
        print("INTERFACE RANDOMIZATION ON: hit_reg/aim-noise/visibility "
              "redrawn every 20 updates (pessimistic..clean)")

    dr_on = bool(args.pin_randomize) and _resolve_pin() is not None
    if args.pin_randomize and not dr_on:
        raise SystemExit(
            "FATAL: --pin-randomize was requested but the pinned opponent %r "
            "could not be resolved in the league pool (%s). Refusing to train "
            "against a FIXED target while reporting nothing -- that is the "
            "exact overfitting condition DR exists to prevent."
            % (getattr(league, "_pin_name", None),
               [getattr(e, "name", "?") for e in league.pool
                if getattr(e, "kind", "") == "scripted"]))
    dr_track = None
    if dr_on:
        e = _resolve_pin()
        _randomize_pin(e)
        dr_track = {"g0": e.games, "w0": e.wins}
        print("pin DOMAIN RANDOMIZATION ON: redraw every 20 updates around the official bar")

    cur_state = None
    if args.pin_curriculum and getattr(league, "_pin_idx", None) is not None:
        entry = league.pool[league._pin_idx]

        def _apply_stage(stage_cfg):
            for k, v in stage_cfg.items():
                if k == "hit_reg":
                    env._hit_reg_p = float(v)  # learner-side artifact anneal
                else:
                    setattr(entry.agent, k, v)

        stage0 = 0
        if resume_stage is not None:
            stage0 = min(int(resume_stage), len(CUR_STAGES) - 1)
        _apply_stage(CUR_STAGES[stage0])
        cur_state = {"entry": entry, "stage": stage0,
                     "g0": entry.games, "w0": entry.wins}
        print("pin curriculum ON: %s stage %d/%d %s (advance at wr %.2f)"
              % (entry.name, stage0 + 1, len(CUR_STAGES), CUR_STAGES[stage0],
                 args.pin_adv_wr))

    t_train_start = time.perf_counter()
    steps_at_start = step
    last_wr_pin = None
    try:
        while step < total_steps:
            t0 = time.perf_counter()
            collector.rollout(buf)
            metrics = trainer.update(buf)
            dt = time.perf_counter() - t0
            step += steps_per_update
            update += 1

            mean_ep = (
                float(np.mean(collector.ep_returns)) if collector.ep_returns
                else float("nan")
            )
            row = {
                "update": update,
                "step": step,
                "steps_per_sec": steps_per_update / dt,
                "mean_ep_reward": mean_ep,
                "win_rate_pool": league.winrate_vs_pool(),
                "elo": league.learner_elo,
                "pool_size": len(league.pool),
                "episodes": collector.episodes,
            }
            row.update(metrics)
            if if_dr and update % 20 == 0:
                _randomize_interface()
            if dr_on and update % 20 == 0:
                e = _resolve_pin()
                if e is None:
                    print("pin DR: scripted pin missing from pool; skipping redraw")
                else:
                    dg = e.games - dr_track["g0"]
                    dw = e.wins - dr_track["w0"]  # entry.wins = OPPONENT wins
                    if dg >= 500:
                        row["wr_pin"] = 1.0 - dw / dg  # vs the randomized family
                    dr_track["g0"], dr_track["w0"] = e.games, e.wins
                    _randomize_pin(e)
            if cur_state is not None and update % 20 == 0:
                e = cur_state["entry"]
                dg = e.games - cur_state["g0"]
                dw = e.wins - cur_state["w0"]      # entry.wins = OPPONENT wins
                if dg >= 2000:
                    wr_pin = 1.0 - dw / dg
                    row["wr_pin"] = wr_pin
                    row["pin_stage"] = cur_state["stage"]
                    row["hit_reg"] = env._hit_reg_p
                    if (wr_pin >= args.pin_adv_wr
                            and cur_state["stage"] < len(CUR_STAGES) - 1):
                        cur_state["stage"] += 1
                        _apply_stage(CUR_STAGES[cur_state["stage"]])
                        print("pin curriculum: wr %.2f -> stage %d/%d %s"
                              % (wr_pin, cur_state["stage"] + 1,
                                 len(CUR_STAGES),
                                 CUR_STAGES[cur_state["stage"]]))
                    cur_state["g0"], cur_state["w0"] = e.games, e.wins
            history.append(row)
            csv_w.writerow(row)
            jsonl_f.write(json.dumps(row) + "\n")

            if "wr_pin" in row:
                last_wr_pin = row["wr_pin"]
            if update % args.log_every == 0:
                csv_f.flush()
                jsonl_f.flush()
                print(
                    "update %5d | step %11d | %8.0f steps/s | ep_rew %7.2f | "
                    "wr_pool %5.2f | wr_bar %s | elo %6.1f | ent %5.2f | loss %7.4f"
                    % (
                        update, step, row["steps_per_sec"], mean_ep,
                        row["win_rate_pool"],
                        "  --" if last_wr_pin is None else "%5.2f" % last_wr_pin,
                        league.learner_elo,
                        metrics["entropy"], metrics["loss"],
                    ),
                    flush=True,
                )
            # seed the pool right away, then gate on cadence
            if update == 1 or update % args.gate_every == 0:
                league.gate(policy, trainer.obs_norm, step)
            if update % args.ckpt_every == 0:
                save_checkpoint(
                    os.path.join(args.out, "ckpt_latest.pt"),
                    policy, trainer, league, step, update,
                    pin_stage=cur_state["stage"] if cur_state else None,
                )
                if args.keep_snapshots:
                    save_checkpoint(
                        os.path.join(args.out, "ckpt_step%012d.pt" % step),
                        policy, trainer, league, step, update,
                        pin_stage=cur_state["stage"] if cur_state else None,
                    )
    finally:
        save_checkpoint(
            os.path.join(args.out, "ckpt_latest.pt"),
            policy, trainer, league, step, update,
            pin_stage=cur_state["stage"] if cur_state else None,
        )
        csv_f.close()
        jsonl_f.close()

    wall = time.perf_counter() - t_train_start
    return {
        "step": step,
        "updates": update,
        "wall_time": wall,
        "steps_per_sec": (step - steps_at_start) / max(wall, 1e-9),
        "history": history,
        "final_elo": league.learner_elo,
        "out": args.out,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Self-play PPO trainer for the duel env."
    )
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--total-steps", type=float, default=1e9,
                   help="total env steps (env.step calls x num_envs)")
    p.add_argument("--out", type=str, default="runs/exp1")
    p.add_argument("--resume", type=str, default="",
                   help="checkpoint path to resume from")
    # PPO
    p.add_argument("--rollout-len", type=int, default=128)
    p.add_argument("--chunk-len", type=int, default=16,
                   help="truncated-BPTT sequence length")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--ent-coef-move", type=float, default=-1.0,
                   help="entropy coef for the latched movement heads "
                        "(forward/strafe/sprint); -1 = use --ent-coef")
    p.add_argument("--ent-coef-cam", type=float, default=-1.0,
                   help="entropy coef for yaw/pitch; -1 = auto "
                        "(0 under --cam-assist, else --ent-coef)")
    p.add_argument("--punish-close", type=float, default=0.0,
                   help="bait-and-punish bonus: extra reward for a hit "
                        "landed while the opponent is committed to "
                        "closing (0 = off)")
    p.add_argument("--interface-randomize", type=int, default=0,
                   help="redraw interface constants every 20 updates "
                        "across the pessimistic..clean range (all five "
                        "clean-world ckpts anti-transferred; the old "
                        "world's exaggerated handicaps were accidental "
                        "interface-DR)")
    p.add_argument("--pin-randomize", type=int, default=0,
                   help="domain-randomize the pin every 20 updates around "
                        "the official bar (kills fixed-target overfitting; "
                        "measured: summit-optimized ckpts anti-transfer "
                        "4-for-4 live)")
    p.add_argument("--arena-square", type=int, default=0,
                   help="square arena (radius = half-side); live truth "
                        "is a 14x14 walled square")
    p.add_argument("--mask-dead-senses", type=int, default=0,
                   help="zero learner obs channels that are dead in the "
                        "live pipeline (enemy_hurt/enemy_hp/hit_dealt)")
    p.add_argument("--punish-swing", type=float, default=0.0,
                   help="swing-punish bonus: extra reward for a hit "
                        "landed within 8 ticks of the opponent's "
                        "swing (0 = off)")
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    # league
    p.add_argument("--p-self", type=float, default=0.5,
                   help="probability an env mirrors the current policy")
    p.add_argument("--elo-k", type=float, default=16.0)
    p.add_argument("--pool-size", type=int, default=20)
    p.add_argument("--gate-winrate", type=float, default=0.55)
    p.add_argument("--gate-every", type=int, default=50,
                   help="try to add a pool checkpoint every N updates")
    # machine / io
    p.add_argument("--threads", type=int, default=12,
                   help="torch.set_num_threads (M4 Max: 12 perf cores)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fov-obs", type=int, default=1,
                   help="1: gate enemy obs by the camera cone (deploy-"
                        "realistic); 0: legacy X-ray obs")
    p.add_argument("--obs-delay", type=int, default=0,
                   help="emit obs from N ticks ago (live sensing lags ~3)")
    p.add_argument("--aim-exec-noise", type=float, default=0.0,
                   help="learner aim-execution jitter deg (live ~5); makes "
                        "the sim faithful so it can't win by perfect-aim kite")
    p.add_argument("--arena-radius", type=float, default=8.0,
                   help="arena disc radius; small = boxed close combat")
    p.add_argument("--rel-vel-noise", type=float, default=0.0,
                   help="learner rel_vel OU noise sigma (live flight-log "
                        "measured ~0.35 after adapter smoothing)")
    p.add_argument("--rel-pos-y-bias", type=float, default=0.0,
                   help="learner rel_pos[1] additive bias (live ~-0.16)")
    p.add_argument("--act-delay", type=int, default=0,
                   help="learner action output lag in ticks (live ~2)")
    p.add_argument("--act-hold", type=int, default=0,
                   help="ticks a movement/sprint bin must persist to take "
                        "effect (kills per-tick velocity PWM; live ~3)")
    p.add_argument("--hit-reg", type=float, default=1.0,
                   help="learner click->hit registration probability "
                        "(live measured ~0.33-0.5 on verified clicks)")
    p.add_argument("--vis-miss", type=float, default=0.0,
                   help="learner per-tick chance a visible enemy reads "
                        "invisible (CNN recall; v9 live ~0.27); with "
                        "--vis-reacq this is p(lose track) of the two-state "
                        "model (live fit 0.235)")
    p.add_argument("--vis-reacq", type=float, default=0.0,
                   help="two-state visibility: per-tick p(reacquire) while "
                        "lost (live fit 0.224); 0 = old IID misses")
    p.add_argument("--cam-assist", type=int, default=0, choices=(0, 1),
                   help="1: env applies deploy's aim/search assist to the "
                        "learner camera (train the DEPLOYED system)")
    p.add_argument("--pin-curriculum", type=int, default=0, choices=(0, 1),
                   help="1: adaptive jitter curriculum on the pinned "
                        "opponent (start wide, tighten on winrate)")
    p.add_argument("--pin-jitter-start", type=float, default=3.0)
    p.add_argument("--pin-jitter-final", type=float, default=0.5)
    p.add_argument("--pin-adv-wr", type=float, default=0.35,
                   help="recent winrate vs the pin that triggers tightening")
    p.add_argument("--pin-opponent", default=None,
                   help="force a fraction of envs vs this named scripted "
                        "opponent (substring match, e.g. P4-Hacker)")
    p.add_argument("--pin-frac", type=float, default=0.0,
                   help="fraction of envs pinned to --pin-opponent")
    p.add_argument("--obs-noise", type=float, default=1.0,
                   help="perception-noise scale (1.0 = measured real "
                        "MAEs, 0 = exact obs)")
    p.add_argument("--ckpt-every", type=int, default=20,
                   help="save checkpoint every N updates")
    p.add_argument("--keep-snapshots", action="store_true",
                   help="also keep per-step checkpoint snapshots")
    p.add_argument("--log-every", type=int, default=1,
                   help="print/flush metrics every N updates")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    summary = train(args)
    print(
        "done: %d steps in %.1fs (%.0f steps/s), %d updates, elo %.1f -> %s"
        % (
            summary["step"], summary["wall_time"], summary["steps_per_sec"],
            summary["updates"], summary["final_elo"], summary["out"],
        )
    )


if __name__ == "__main__":
    main()
