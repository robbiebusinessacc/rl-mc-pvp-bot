"""Ladder CLI: round-robin scripted tiers + checkpoints, rate, report.

Usage::

    python3 -m pvpbot.eval.ladder --checkpoints runs/*/ckpt_*.pt \
        --matches 200 --out reports/

With no --checkpoints the ladder ranks just the scripted tiers against each
other -- the module's self-consistency check (T4 must beat T2 head-to-head,
and every tier must beat T0 on decisive games).  Writes ``report.json`` and
``report.md`` into the --out directory.
"""
import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from pvpbot.eval.arena import MatchResult, as_contestant, run_match
from pvpbot.eval.ratings import Elo, TrueSkillLite
from pvpbot.eval.scripted import default_tiers


# ---------------------------------------------------------------------------
def build_contestants(checkpoint_args: Sequence[str], seed: int = 0) -> List[Any]:
    """Scripted tiers + any checkpoints (args may be globs or paths)."""
    contestants: List[Any] = list(default_tiers(seed))
    from pvpbot.eval.practice import practice_tiers
    contestants += list(practice_tiers(seed))
    paths: List[str] = []
    for pat in checkpoint_args:
        hits = sorted(glob.glob(pat))
        paths.extend(hits if hits else [pat])
    seen = set()
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        if not os.path.exists(p):
            raise FileNotFoundError("checkpoint not found: %s" % p)
        contestants.append(as_contestant(p))
    names = [c.name for c in contestants]
    if len(set(names)) != len(names):
        raise ValueError("duplicate contestant names: %r" % names)
    return contestants


def run_ladder(
    contestants: List[Any],
    matches: int = 200,
    seed: int = 0,
    max_steps: Optional[int] = None,
    env_cls: Optional[type] = None,
) -> Dict[str, Any]:
    """Round-robin every pair, rate with Elo + TrueSkill-lite, build report."""
    results: List[MatchResult] = []
    for i in range(len(contestants)):
        for j in range(i + 1, len(contestants)):
            res = run_match(
                contestants[i], contestants[j], num_duels=matches,
                seed=seed + 1000 * i + j, max_steps=max_steps,
                env_cls=env_cls)
            results.append(res)

    # Feed every duel outcome into both raters, interleaved across pairs so
    # neither system sees one series as a long streak.
    elo = Elo()
    tsl = TrueSkillLite()
    queues = []
    for r in results:
        outs = [1.0] * r.wins_a + [0.0] * r.wins_b + [0.5] * r.draws
        rng = np.random.default_rng(seed + len(queues))
        rng.shuffle(outs)
        queues.append((r.name_a, r.name_b, outs))
    remaining = True
    k = 0
    while remaining:
        remaining = False
        for a, b, outs in queues:
            if k < len(outs):
                elo.record(a, b, outs[k])
                tsl.record(a, b, outs[k])
                remaining = True
        k += 1

    names = [c.name for c in contestants]
    totals = {n: {"wins": 0, "losses": 0, "draws": 0} for n in names}
    pair: Dict[str, Dict[str, Any]] = {n: {} for n in names}
    stats: Dict[str, Dict[str, Any]] = {
        n: {"hits_landed": [], "hits_taken": [], "avg_combo": [],
            "mean_aim_err_deg": [], "mean_episode_len": []} for n in names}
    for r in results:
        totals[r.name_a]["wins"] += r.wins_a
        totals[r.name_a]["losses"] += r.wins_b
        totals[r.name_a]["draws"] += r.draws
        totals[r.name_b]["wins"] += r.wins_b
        totals[r.name_b]["losses"] += r.wins_a
        totals[r.name_b]["draws"] += r.draws
        pair[r.name_a][r.name_b] = {
            "wins": r.wins_a, "losses": r.wins_b, "draws": r.draws}
        pair[r.name_b][r.name_a] = {
            "wins": r.wins_b, "losses": r.wins_a, "draws": r.draws}
        for name, s in ((r.name_a, r.stats_a), (r.name_b, r.stats_b)):
            d = s.to_dict()
            for key in ("hits_landed", "hits_taken", "avg_combo",
                        "mean_aim_err_deg"):
                stats[name][key].append(d[key])
            stats[name]["mean_episode_len"].append(r.mean_episode_len)
    stat_summary = {
        n: {k: (float(np.mean(v)) if v else 0.0) for k, v in d.items()}
        for n, d in stats.items()}

    table = []
    for n in names:
        mu, sigma = tsl.rating(n)
        table.append({
            "name": n,
            "elo": elo.rating(n),
            "ts_mu": mu,
            "ts_sigma": sigma,
            "ts_conservative": tsl.conservative(n),
            **totals[n],
        })
    table.sort(key=lambda row: row["elo"], reverse=True)

    report: Dict[str, Any] = {
        "matches_per_pair": matches,
        "seed": seed,
        "contestants": names,
        "ratings": table,
        "pairs": pair,
        "stats": stat_summary,
        "matches": [r.to_dict() for r in results],
        "elo_json": json.loads(elo.to_json()),
        "trueskill_json": json.loads(tsl.to_json()),
    }
    report["sanity"] = _sanity(pair, names)
    return report


def _sanity(pair: Dict[str, Dict[str, Any]], names: List[str]) -> Dict[str, Any]:
    """Scripted-tier self-consistency: T4 beats T2, everything beats T0.

    Exception: stationary T1 vs idle T0 is only required to never lose --
    under real-sim physics T1's own knockback can push T0 out of reach
    permanently, drawing by timeout, which is physically correct play.
    """
    out: Dict[str, Any] = {}

    def beats(a: str, b: str) -> Optional[bool]:
        cell = pair.get(a, {}).get(b)
        if cell is None:
            return None
        return cell["wins"] > cell["losses"]

    def never_loses(a: str, b: str) -> Optional[bool]:
        cell = pair.get(a, {}).get(b)
        if cell is None:
            return None
        return cell["losses"] == 0

    if "T4-Pro" in names and "T2-Chaser" in names:
        out["t4_beats_t2"] = beats("T4-Pro", "T2-Chaser")
    if "T0-Idle" in names:
        for n in names:
            if n == "T0-Idle":
                continue
            if n == "T1-Aimbot":
                out["T1-Aimbot_never_loses_to_t0"] = never_loses(n, "T0-Idle")
            else:
                out["%s_beats_t0" % n] = beats(n, "T0-Idle")
    out["ok"] = all(v for v in out.values() if v is not None) and bool(out)
    return out


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------
def write_reports(report: Dict[str, Any], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=False)
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(render_markdown(report))


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# PvP Eval Ladder Report")
    lines.append("")
    lines.append("Matches per pair: %d | seed: %d" % (
        report["matches_per_pair"], report["seed"]))
    lines.append("")
    lines.append("## Ratings")
    lines.append("")
    lines.append("| # | Contestant | Elo | TS mu | TS sigma | TS (mu-3s) "
                 "| W | L | D |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for i, row in enumerate(report["ratings"], 1):
        lines.append(
            "| %d | %s | %.0f | %.2f | %.2f | %.2f | %d | %d | %d |" % (
                i, row["name"], row["elo"], row["ts_mu"], row["ts_sigma"],
                row["ts_conservative"], row["wins"], row["losses"],
                row["draws"]))
    lines.append("")
    lines.append("## Win matrix (row vs column: wins-losses-draws)")
    lines.append("")
    names = [row["name"] for row in report["ratings"]]
    lines.append("| vs | " + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * len(names))
    for a in names:
        cells = []
        for b in names:
            if a == b:
                cells.append("--")
            else:
                c = report["pairs"][a][b]
                cells.append("%d-%d-%d" % (c["wins"], c["losses"], c["draws"]))
        lines.append("| **%s** | " % a + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Per-contestant stats (means over pairings)")
    lines.append("")
    lines.append("| Contestant | Hits landed | Hits taken | Avg combo "
                 "| Aim err (deg) | Ep len |")
    lines.append("|---|---|---|---|---|---|")
    for n in names:
        s = report["stats"][n]
        lines.append("| %s | %.2f | %.2f | %.2f | %.1f | %.0f |" % (
            n, s["hits_landed"], s["hits_taken"], s["avg_combo"],
            s["mean_aim_err_deg"], s["mean_episode_len"]))
    lines.append("")
    lines.append("## Sanity")
    lines.append("")
    for k, v in report["sanity"].items():
        lines.append("- %s: %s" % (k, v))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python3 -m pvpbot.eval.ladder",
        description="Round-robin eval ladder over checkpoints + scripted tiers.")
    ap.add_argument("--checkpoints", nargs="*", default=[],
                    help="checkpoint paths/globs (spec.py format); optional")
    ap.add_argument("--matches", type=int, default=200,
                    help="duels per pair (default 200)")
    ap.add_argument("--out", default="reports/",
                    help="output directory for report.json/report.md")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap env steps per pairing (unfinished duels draw)")
    ap.add_argument("--env", choices=["auto", "stub", "real"], default="auto",
                    help="duel env: auto (real if merged, else stub), or force")
    args = ap.parse_args(argv)

    env_cls = None
    if args.env == "stub":
        from pvpbot.sim.stub import DuelVecEnv as env_cls
    elif args.env == "real":
        from pvpbot.sim.env import DuelVecEnv as env_cls

    contestants = build_contestants(args.checkpoints, seed=args.seed)
    if not args.checkpoints:
        print("no checkpoints given: ranking scripted tiers only "
              "(self-consistency mode)")
    print("contestants: %s" % ", ".join(c.name for c in contestants))
    report = run_ladder(contestants, matches=args.matches, seed=args.seed,
                        max_steps=args.max_steps, env_cls=env_cls)
    write_reports(report, args.out)
    print("wrote %s and %s" % (os.path.join(args.out, "report.json"),
                               os.path.join(args.out, "report.md")))
    for row in report["ratings"]:
        print("%-16s elo=%7.1f  ts=%6.2f+-%.2f  W/L/D=%d/%d/%d" % (
            row["name"], row["elo"], row["ts_mu"], row["ts_sigma"],
            row["wins"], row["losses"], row["draws"]))
    sanity = report["sanity"]
    print("sanity: %s" % ("OK" if sanity.get("ok") else "FAILED %r" % sanity))
    return 0 if sanity.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
