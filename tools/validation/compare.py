#!/usr/bin/env python3
"""Compare a ground-truth duel recording against the NumPy sim.

Replays the schedule's inputs open-loop through the sim env
(pvpbot.sim.env if it exists, else pvpbot.sim.stub), then compares the
per-tick position/velocity/yaw traces against the recording, per schedule
segment: RMSE, max divergence and the tick it happens at.

Usage:
    python3 tools/validation/compare.py \
        --recording rec.jsonl \
        --schedule tools/validation/schedules/basic.json \
        [--env stub|real|auto] [--json-out report.json] \
        [--tick-offset N] [--no-mask-grounded-vy] \
        [--fail-above-pos-rmse X]

COORDINATE CONVENTIONS (the classic silent validation bug -- both frames
produce plausible-looking trajectories, a wrong mapping just rotates the
world and quietly inflates RMSE):

  * The recording is in the MINECRAFT/notchian frame: +X east, +Y up,
    +Z south; yaw 0 faces +Z and increases clockwise viewed from above
    (forward = (-sin yaw, 0, cos yaw)); positions are feet coords;
    velocity is blocks/tick.
  * The sim/stub frame has forward = (cos yaw, 0, sin yaw) and its floor
    at y = 0.
  * Both yaw conventions rotate the same direction in the XZ plane, so a
    constant offset maps between them: sim_yaw = (mc_yaw + 90) % 360.
    Yaw *deltas* are identical in both frames.
  * Positions: x/z transfer unchanged; y is shifted by arena.ground_y
    (sim_y = mc_y - ground_y). Velocities transfer unchanged.

  All conversions live in harness.py (mc_yaw_to_sim / mc_pos_to_sim) and
  everything is compared in the SIM frame.

Known stub artifact: the stub never zeroes vertical velocity while a bot
stands on the floor (vy integrates -0.08/tick forever, position is just
clamped), so grounded vy is meaningless. By default, whenever the
*recording* says onGround, the y-velocity of BOTH traces is masked to 0
before computing velocity RMSE (--no-mask-grounded-vy disables this).
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import harness  # noqa: E402


def _rmse(err: np.ndarray) -> float:
    """RMSE of per-tick error vectors: sqrt(mean over ticks of |err_t|^2)."""
    err = np.asarray(err, dtype=np.float64)
    if err.size == 0:
        return float("nan")
    if err.ndim == 1:
        return float(np.sqrt(np.mean(err ** 2)))
    return float(np.sqrt(np.mean(np.sum(err ** 2, axis=-1))))


def _segment_metrics(
    seg: Dict[str, Any],
    side: int,
    bot: str,
    trace: Dict[str, Any],
    rec_rows: Dict[Any, Dict[str, Any]],
    ground_y: float,
    tick_offset: int,
    mask_grounded_vy: bool,
) -> Dict[str, Any]:
    ticks: List[int] = []
    pos_err: List[np.ndarray] = []
    vel_err: List[np.ndarray] = []
    yaw_err: List[float] = []
    T = trace["num_ticks"]
    for t in range(seg["tick_start"], seg["tick_end"]):
        if not (0 <= t < T):
            continue
        row = rec_rows.get((bot, t + tick_offset))
        if row is None:
            continue
        rec_pos = harness.mc_pos_to_sim(row["pos"], ground_y)
        rec_vel = np.asarray(row["vel"], dtype=np.float64)
        sim_pos = trace["pos"][t, side].copy()
        sim_vel = trace["vel"][t, side].copy()
        if mask_grounded_vy and row.get("onGround"):
            rec_vel = rec_vel.copy()
            rec_vel[1] = 0.0
            sim_vel[1] = 0.0
        rec_yaw_sim = harness.mc_yaw_to_sim(row["yaw"])
        ticks.append(t)
        pos_err.append(rec_pos - sim_pos)
        vel_err.append(rec_vel - sim_vel)
        yaw_err.append(float(harness.wrap_angle_deg(rec_yaw_sim - trace["yaw"][t, side])))

    out: Dict[str, Any] = {
        "id": seg["id"],
        "bot": bot,
        "mechanic": seg["mechanic"],
        "tick_start": seg["tick_start"],
        "tick_end": seg["tick_end"],
        "n_ticks_compared": len(ticks),
    }
    if not ticks:
        out.update(
            pos_rmse=None, vel_rmse=None, yaw_rmse_deg=None,
            max_pos_err=None, max_pos_err_tick=None,
        )
        return out
    pe = np.asarray(pos_err)
    norms = np.linalg.norm(pe, axis=1)
    worst = int(np.argmax(norms))
    out.update(
        pos_rmse=_rmse(pe),
        vel_rmse=_rmse(np.asarray(vel_err)),
        yaw_rmse_deg=_rmse(np.asarray(yaw_err)),
        max_pos_err=float(norms[worst]),
        max_pos_err_tick=int(ticks[worst]),
    )
    return out


def run_compare(
    recording_path: str,
    schedule_path: str,
    env_kind: str = "auto",
    tick_offset: int = 0,
    mask_grounded_vy: bool = True,
    seed: int = 0,
) -> Dict[str, Any]:
    """Full comparison; returns a JSON-serializable report dict."""
    sched = harness.load_schedule(schedule_path)
    rec_rows, hurt_events = harness.read_recording(recording_path)
    collision = bool(sched.get("meta", {}).get("sim_collision", True))
    env, used_kind = harness.make_env(env_kind, seed=seed, collision=collision)
    trace = harness.roll_schedule(env, sched)
    ground_y = sched["arena"]["ground_y"]
    bots = sched["bots"]

    seg_reports = [
        _segment_metrics(
            seg, bots.index(seg["bot"]), seg["bot"], trace, rec_rows,
            ground_y, tick_offset, mask_grounded_vy,
        )
        for seg in sched["segments"]
    ]

    # overall (whole-timeline) metrics per bot, treated as pseudo-segments
    overall: Dict[str, Any] = {}
    for side, bot in enumerate(bots):
        pseudo = {
            "id": "overall_%s" % bot,
            "bot": bot,
            "mechanic": "overall",
            "tick_start": 0,
            "tick_end": trace["num_ticks"],
        }
        m = _segment_metrics(
            pseudo, side, bot, trace, rec_rows, ground_y, tick_offset,
            mask_grounded_vy,
        )
        overall[bot] = m

    scored = [s for s in seg_reports if s["pos_rmse"] is not None]

    def _argmax(key: str) -> Optional[Dict[str, Any]]:
        if not scored:
            return None
        return max(scored, key=lambda s: s[key])

    worst_pos = _argmax("pos_rmse")
    worst_vel = _argmax("vel_rmse")
    combined_pos = [s["pos_rmse"] for s in scored]
    combined_n = [s["n_ticks_compared"] for s in scored]
    combined_pos_rmse = (
        float(
            np.sqrt(
                sum(r * r * n for r, n in zip(combined_pos, combined_n))
                / max(sum(combined_n), 1)
            )
        )
        if scored
        else None
    )

    return {
        "recording": os.path.abspath(recording_path),
        "schedule": os.path.abspath(schedule_path),
        "schedule_name": sched["meta"].get("name"),
        "env": used_kind,
        "tick_offset": tick_offset,
        "mask_grounded_vy": mask_grounded_vy,
        "num_ticks_scheduled": trace["num_ticks"],
        "num_recording_rows": len(rec_rows),
        "num_hurt_events_recorded": len(hurt_events),
        "num_hurt_events_sim": len(trace["hurt_events"]),
        "segments": seg_reports,
        "overall_per_bot": overall,
        "combined_pos_rmse": combined_pos_rmse,
        "worst_segment_pos": None if worst_pos is None else worst_pos["id"],
        "worst_segment_vel": None if worst_vel is None else worst_vel["id"],
    }


def _fmt(v: Optional[float], width: int = 10, prec: int = 6) -> str:
    if v is None:
        return "-".rjust(width)
    return ("%.*f" % (prec, v)).rjust(width)


def format_report(rep: Dict[str, Any]) -> str:
    lines: List[str] = []
    add = lines.append
    add("=== mc-pvp-bot physics validation report ===")
    add("schedule : %s (%s)" % (rep["schedule_name"], rep["schedule"]))
    add("recording: %s" % rep["recording"])
    add(
        "sim env  : %s   ticks scheduled: %d   recording rows: %d"
        % (rep["env"], rep["num_ticks_scheduled"], rep["num_recording_rows"])
    )
    add(
        "hurt events: %d recorded / %d in sim   tick offset: %+d   "
        "grounded-vy masked: %s"
        % (
            rep["num_hurt_events_recorded"],
            rep["num_hurt_events_sim"],
            rep["tick_offset"],
            "yes" if rep["mask_grounded_vy"] else "no",
        )
    )
    add("")
    hdr = "%-14s %-3s %-18s %9s %5s %10s %10s %9s %12s %6s" % (
        "segment", "bot", "mechanic", "ticks", "n",
        "pos_rmse", "vel_rmse", "yaw_rmse", "max_pos_err", "@tick",
    )
    add(hdr)
    add("-" * len(hdr))
    for s in rep["segments"]:
        add(
            "%-14s %-3s %-18s %9s %5d %s %s %s %s %6s"
            % (
                s["id"],
                s["bot"],
                s["mechanic"],
                "%d-%d" % (s["tick_start"], s["tick_end"]),
                s["n_ticks_compared"],
                _fmt(s["pos_rmse"]),
                _fmt(s["vel_rmse"]),
                _fmt(s["yaw_rmse_deg"], width=9, prec=3),
                _fmt(s["max_pos_err"], width=12),
                "-" if s["max_pos_err_tick"] is None else s["max_pos_err_tick"],
            )
        )
    add("-" * len(hdr))
    for bot, m in rep["overall_per_bot"].items():
        add(
            "%-14s %-3s %-18s %9s %5d %s %s %s %s %6s"
            % (
                "overall",
                bot,
                "",
                "%d-%d" % (m["tick_start"], m["tick_end"]),
                m["n_ticks_compared"],
                _fmt(m["pos_rmse"]),
                _fmt(m["vel_rmse"]),
                _fmt(m["yaw_rmse_deg"], width=9, prec=3),
                _fmt(m["max_pos_err"], width=12),
                "-" if m["max_pos_err_tick"] is None else m["max_pos_err_tick"],
            )
        )
    add("")
    add("combined pos RMSE (all segments): %s" % _fmt(rep["combined_pos_rmse"], 0))
    add("worst segment by pos RMSE: %s" % rep["worst_segment_pos"])
    add("worst segment by vel RMSE: %s" % rep["worst_segment_vel"])
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recording", required=True, help="ground-truth JSONL")
    ap.add_argument(
        "--schedule", default=harness.DEFAULT_SCHEDULE, help="schedule JSON"
    )
    ap.add_argument(
        "--env",
        default="auto",
        choices=("stub", "real", "auto"),
        help="which sim to replay through (auto: real if merged, else stub)",
    )
    ap.add_argument("--json-out", default=None, help="also write report JSON here")
    ap.add_argument(
        "--tick-offset",
        type=int,
        default=0,
        help="shift recording ticks by N when aligning (recorder tick t+N "
        "matches sim tick t); use if the recorder start skewed by a tick",
    )
    ap.add_argument(
        "--no-mask-grounded-vy",
        action="store_true",
        help="compare raw vy even when the recording says onGround "
        "(the stub's grounded vy is a known artifact)",
    )
    ap.add_argument(
        "--fail-above-pos-rmse",
        type=float,
        default=None,
        help="exit 1 if combined pos RMSE exceeds this threshold",
    )
    args = ap.parse_args(argv)

    rep = run_compare(
        args.recording,
        args.schedule,
        env_kind=args.env,
        tick_offset=args.tick_offset,
        mask_grounded_vy=not args.no_mask_grounded_vy,
    )
    print(format_report(rep))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(rep, f, indent=2)
        print("\nJSON report written to %s" % os.path.abspath(args.json_out))
    if (
        args.fail_above_pos_rmse is not None
        and rep["combined_pos_rmse"] is not None
        and rep["combined_pos_rmse"] > args.fail_above_pos_rmse
    ):
        print(
            "FAIL: combined pos RMSE %.6f > threshold %.6f"
            % (rep["combined_pos_rmse"], args.fail_above_pos_rmse)
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
