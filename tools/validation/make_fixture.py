#!/usr/bin/env python3
"""Generate synthetic "ground truth" recordings by rolling the stub env.

Two fixtures are produced from one schedule:

  * perfect.jsonl   -- the stub's own trajectory, re-encoded into the exact
    recording format record_duel.js writes (Minecraft frame, notchian yaw).
    compare.py replayed against it must report ~0 RMSE everywhere; this
    proves the harness (input mapping + frame conversions + alignment)
    end-to-end without a Minecraft server.

  * perturbed.jsonl -- the same trajectory with seeded Gaussian velocity
    noise. The noise is also applied to position as a one-tick displacement
    (pos += vel_noise), i.e. "the erroneous velocity persisted for one
    tick", so both pos and vel RMSE are nonzero but segments stay
    statistically independent (no unbounded drift). One segment
    (--boost-segment, default sprint_jump) gets its noise scaled by
    --boost so worst-segment identification is deterministic.

Usage:
    python3 tools/validation/make_fixture.py \
        [--schedule tools/validation/schedules/basic.json] \
        --out-dir DIR [--seed 0] [--noise-vel 0.02] \
        [--boost-segment sprint_jump] [--boost 8.0]
"""
import argparse
import copy
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import harness  # noqa: E402


def _boosted_ticks(sched: Dict[str, Any], boost_segment: str) -> Set[Tuple[str, int]]:
    for seg in sched["segments"]:
        if seg["id"] == boost_segment:
            return {
                (seg["bot"], t)
                for t in range(seg["tick_start"], seg["tick_end"])
            }
    raise ValueError(
        "--boost-segment %r not found in schedule (ids: %s)"
        % (boost_segment, [s["id"] for s in sched["segments"]])
    )


def generate_fixtures(
    schedule_path: str,
    out_dir: str,
    seed: int = 0,
    noise_vel: float = 0.02,
    boost_segment: str = "sprint_jump",
    boost: float = 8.0,
) -> Dict[str, Any]:
    """Write perfect.jsonl and perturbed.jsonl into out_dir; return summary."""
    sched = harness.load_schedule(schedule_path)
    boosted = _boosted_ticks(sched, boost_segment)
    from pvpbot.sim.stub import DuelVecEnv  # fixtures always come from the stub

    env = DuelVecEnv(1, seed=seed)
    trace = harness.roll_schedule(env, sched)
    rows = harness.trace_to_recording_rows(trace, sched)

    os.makedirs(out_dir, exist_ok=True)
    perfect_path = os.path.join(out_dir, "perfect.jsonl")
    harness.write_jsonl(perfect_path, rows)

    rng = np.random.default_rng(seed)
    perturbed: List[Dict[str, Any]] = []
    for row in rows:
        row = copy.deepcopy(row)
        if row.get("event") is None:  # per-tick state rows only
            sigma = noise_vel * (
                boost if (row["bot"], row["tick"]) in boosted else 1.0
            )
            nv = rng.normal(0.0, sigma, 3)
            row["vel"] = (np.asarray(row["vel"]) + nv).tolist()
            row["pos"] = (np.asarray(row["pos"]) + nv).tolist()
        perturbed.append(row)
    perturbed_path = os.path.join(out_dir, "perturbed.jsonl")
    harness.write_jsonl(perturbed_path, perturbed)

    bots = trace["bots"]
    summary = {
        "schedule": os.path.abspath(schedule_path),
        "num_ticks": trace["num_ticks"],
        "num_rows": len(rows),
        "hurt_events": trace["hurt_events"],
        "hit_ticks": [ev["tick"] for ev in trace["hurt_events"]],
        "final_hp": {
            b: float(trace["hp"][-1, i]) for i, b in enumerate(bots)
        },
        "final_pos_mc": {
            b: harness.sim_pos_to_mc(
                trace["pos"][-1, i], sched["arena"]["ground_y"]
            ).tolist()
            for i, b in enumerate(bots)
        },
        "boost_segment": boost_segment,
        "perfect_path": perfect_path,
        "perturbed_path": perturbed_path,
    }
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--schedule", default=harness.DEFAULT_SCHEDULE)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-vel", type=float, default=0.02)
    ap.add_argument("--boost-segment", default="sprint_jump")
    ap.add_argument("--boost", type=float, default=8.0)
    args = ap.parse_args(argv)

    s = generate_fixtures(
        args.schedule,
        args.out_dir,
        seed=args.seed,
        noise_vel=args.noise_vel,
        boost_segment=args.boost_segment,
        boost=args.boost,
    )
    print("schedule      : %s" % s["schedule"])
    print("ticks         : %d  (%d rows)" % (s["num_ticks"], s["num_rows"]))
    print("hits landed   : %d at ticks %s" % (len(s["hit_ticks"]), s["hit_ticks"]))
    for b, hp in sorted(s["final_hp"].items()):
        print(
            "bot %s         : final hp %.1f  final pos (MC) %s"
            % (b, hp, ["%.3f" % v for v in s["final_pos_mc"][b]])
        )
    print("perfect       : %s" % s["perfect_path"])
    print("perturbed     : %s  (boosted segment: %s)" % (s["perturbed_path"], s["boost_segment"]))
    if not s["hit_ticks"]:
        print(
            "WARNING: no hits landed -- the attack window never overlapped "
            "reach; retune the schedule",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
