#!/usr/bin/env python3
"""Render `docs/assets/final-ladder-standings.svg`: the ten-way round-robin result.

WHAT IT PLOTS
  Left panel  - horizontal Elo bar chart for all ten contestants of the real-physics
                ladder, sorted descending, with each contestant's W-L-D printed at the
                end of its bar and a dashed reference line at the seed rating that the
                Elo book was initialised with (read from the report, not hardcoded).
  Right panel - mean hits landed per duel (x) against mean hits taken per duel (y),
                one marker per contestant, marker area scaled linearly by that
                contestant's mean combo length, with a y = x diagonal. A contestant
                below the diagonal deals more than it receives.

  Bars and markers are coloured by family: the learned checkpoint in policy violet,
  the P-tier ports of PracticeBotPvP's difficulty tables in perception amber, the
  T-tier hand-written scripted opponents in mid grey. Those three hexes come from the
  docset's fixed seven-colour palette.

DATA
  docs/assets/data/ladder-report.json - COMMITTED (reports/ is gitignored; the
  ladder JSON is copied in beside the other figure sidecars), so this re-renders from a fresh
  clone. Keys consumed: `matches_per_pair`, `seed`, `pairs` (to count pairings),
  `elo_json.initial` and `elo_json.k` (the seed-rating line), `ratings` (elo, wins,
  losses, draws per contestant) and `stats` (hits_landed, hits_taken, avg_combo,
  mean_aim_err_deg per contestant). Every number drawn is read from that file; nothing
  is smoothed, resampled, or aggregated across contestants.

RUN
  python3 tools/figures/fig_ladder_standings.py            # writes the SVG
  python3 tools/figures/fig_ladder_standings.py --png /tmp/preview.png   # raster copy
                                                            # for eyeballing only
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO_ROOT, "docs", "assets", "data", "ladder-report.json")
OUT = os.path.join(REPO_ROOT, "docs", "assets", "final-ladder-standings.svg")

# --------------------------------------------------------------------------------------
# palette (the seven-hex docset palette; only three roles appear here)
# --------------------------------------------------------------------------------------
C_POLICY = "#5b3fa8"      # policy: the learned checkpoint
C_PERCEPTION = "#a8621b"  # perception amber: the P-tier practice-bot ports
C_SCRIPT = "#8b8b8b"      # mid grey: the T-tier scripted opponents
INK = "#4a4a4a"           # axis and tick text
GRID = "#8b8b8b"



def _learned_name(names):
    """The learned contestant, whatever the checkpoint file was called."""
    for n in names:
        if n.lower().startswith("ckpt"):
            return n
    raise SystemExit("no contestant name starts with 'ckpt'")


def family_colour(name: str) -> str:
    """Colour a contestant by which family it belongs to."""
    if name.lower().startswith("ckpt"):
        return C_POLICY
    if name.startswith("P"):
        return C_PERCEPTION
    return C_SCRIPT


def load(path: str) -> dict:
    if not os.path.exists(path):
        sys.exit(
            "ERROR: missing data file\n"
            "  expected: {}\n"
            "  This figure reads the committed ladder report. If it is absent, "
            "regenerate it with the eval ladder\n"
            "  (pvpbot/eval/ladder.py) writing to reports/milestones/, then re-run "
            "this script.".format(path)
        )
    with open(path, "r") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------------------
# label placement for the right panel. The five strongest scripted opponents sit inside
# a 1.6 x 1.1 hit box, so their names are pushed outward and joined back to their marker
# with a hairline leader. Positions are in data coordinates: (x, y, horizontal-align).
# --------------------------------------------------------------------------------------
LABEL_POS = {
    "T0-Idle":     (0.60, 11.60, "left"),
    "P2-Medium":   (3.50, 11.55, "left"),
    "P1-Easy":     (4.75, 10.90, "left"),
    "T2-Chaser":   (11.85, 10.95, "left"),
    "T3-Strafer":  (8.95, 10.30, "right"),
    "P3-Hard":     (12.80, 9.60, "left"),
    "P4-Hacker":   (11.55, 5.50, "left"),
    "T4-Pro":      (12.55, 6.55, "left"),
    "T1-Aimbot":   (9.40, 6.55, "center"),
    "ckpt_latest": (11.85, 0.34, "right"),
}
# names whose label needs a leader line back to its marker
LEADERED = {"T2-Chaser", "T3-Strafer", "P3-Hard", "P4-Hacker", "T4-Pro", "T1-Aimbot"}


def build(report: dict, out_path: str, png_path: str = "") -> dict:
    ratings = report["ratings"]
    stats = report["stats"]
    elo_book = report["elo_json"]

    seed_rating = float(elo_book["initial"])
    elo_k = float(elo_book["k"])
    per_pair = int(report["matches_per_pair"])
    seed = report["seed"]

    # count unique (unordered) pairings straight from the pair table
    pairings = {tuple(sorted((a, b))) for a, row in report["pairs"].items() for b in row}
    n_pairings = len(pairings)
    n_duels = n_pairings * per_pair

    rows = sorted(ratings, key=lambda r: r["elo"], reverse=True)

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(11.0, 4.4), gridspec_kw={"width_ratios": [1.18, 1.0], "wspace": 0.26}
    )
    fig.patch.set_alpha(0.0)

    # ---------------------------------------------------------------- LEFT: Elo bars
    ax_l.set_facecolor("none")
    names = [r["name"] for r in rows]
    elos = [r["elo"] for r in rows]
    ypos = list(range(len(rows)))

    ax_l.barh(
        ypos,
        elos,
        height=0.66,
        color=[family_colour(n) for n in names],
        edgecolor="none",
        zorder=3,
    )

    for y, r in zip(ypos, rows):
        # nudge a W-L-D label clear of the dashed seed line when the bar ends just short
        # of it, so the line does not run through the digits (affects one row here)
        tx = r["elo"] + 26
        if r["elo"] < seed_rating < r["elo"] + 240:
            tx = seed_rating + 26
        ax_l.text(
            tx,
            y,
            "{}-{}-{}".format(r["wins"], r["losses"], r["draws"]),
            va="center",
            ha="left",
            fontsize=8,
            color=INK,
            zorder=4,
        )

    ax_l.axvline(seed_rating, color=INK, ls=(0, (4, 3)), lw=1.0, alpha=0.85, zorder=2)
    ax_l.text(
        seed_rating + 22,
        -1.02,
        "seed rating {:.0f}".format(seed_rating),
        fontsize=8,
        color=INK,
        va="center",
        ha="left",
    )

    ax_l.set_yticks(ypos)
    ax_l.set_yticklabels(names, fontsize=9)
    ax_l.set_ylim(len(rows) - 0.35, -1.45)
    ax_l.set_xlim(0, 1990)
    ax_l.set_xticks([0, 400, 800, 1200, 1600])
    ax_l.set_xlabel("Elo rating  (K = {:.0f}, seeded at {:.0f})".format(elo_k, seed_rating),
                    fontsize=9, color=INK)
    ax_l.xaxis.grid(True, color=GRID, alpha=0.2, lw=0.8)
    ax_l.set_axisbelow(True)

    handles = [
        Line2D([], [], marker="s", ls="", ms=7, mfc=C_POLICY, mec="none",
               label="learned checkpoint"),
        Line2D([], [], marker="s", ls="", ms=7, mfc=C_PERCEPTION, mec="none",
               label="P-tier practice-bot ports"),
        Line2D([], [], marker="s", ls="", ms=7, mfc=C_SCRIPT, mec="none",
               label="T-tier scripted opponents"),
    ]
    leg = ax_l.legend(
        handles=handles, loc="lower right", frameon=False, fontsize=8,
        handletextpad=0.5, borderaxespad=0.6, labelspacing=0.45,
    )
    for txt in leg.get_texts():
        txt.set_color(INK)

    # ------------------------------------------------- RIGHT: hits landed vs hits taken
    ax_r.set_facecolor("none")
    ax_r.set_xlim(-0.55, 14.75)
    ax_r.set_ylim(-0.75, 13.25)
    ax_r.plot([0, 12.35], [0, 12.35], color=GRID, alpha=0.4, lw=1.0, zorder=1)

    def area(combo: float) -> float:
        """Marker area in points^2, linear in mean combo length."""
        return 24.0 + 42.0 * combo

    for name in [r["name"] for r in rows]:
        s = stats[name]
        x, y = s["hits_landed"], s["hits_taken"]
        col = family_colour(name)
        ax_r.scatter(
            [x], [y], s=area(s["avg_combo"]), color=col, alpha=0.78,
            edgecolors=col, linewidths=0.8, zorder=3,
        )
        # hand-placed if known, otherwise a default up-right offset
        lx, ly, ha = LABEL_POS.get(name, (x + 0.35, y + 0.40, "left"))
        if name in LEADERED:
            ax_r.plot([x, lx], [y, ly], color=GRID, lw=0.55, alpha=0.6, zorder=2)
        ax_r.text(lx, ly, name, fontsize=8, color=INK, ha=ha, va="center", zorder=5)

    # the y = x guide label, rotated to the true on-screen slope of the line
    ang = ax_r.transData.transform((10.0, 10.0)) - ax_r.transData.transform((0.0, 0.0))
    ax_r.text(
        11.90, 11.90, "y = x", fontsize=8, color=GRID,
        ha="center", va="bottom", rotation=math.degrees(math.atan2(ang[1], ang[0])),
    )
    ax_r.text(
        6.20, 4.90, "below this line: lands more than it takes",
        fontsize=8, color=GRID, ha="left", va="center",
    )

    # the learned checkpoint is the only contestant far below the diagonal
    LEARNED = _learned_name(list(stats)); ck = stats[LEARNED]
    runner_up = rows[1]["name"]
    runner_up_aim = stats[runner_up]["mean_aim_err_deg"]
    ax_r.annotate(
        "lands {:.1f}, takes {:.1f}, combos {:.1f} — while\n"
        "aiming WORSE than the runner-up ({:.1f}° vs {:.1f}°)".format(
            ck["hits_landed"], ck["hits_taken"], ck["avg_combo"],
            ck["mean_aim_err_deg"], runner_up_aim,
        ),
        xy=(ck["hits_landed"] - 0.62, ck["hits_taken"] + 0.42),
        xytext=(3.70, 3.15),
        fontsize=8.5, color=INK, ha="left", va="center", linespacing=1.5,
        arrowprops=dict(arrowstyle="-", color=GRID, lw=0.9, shrinkB=0,
                        connectionstyle="angle3,angleA=0,angleB=58"),
        zorder=5,
    )

    size_handles = [
        Line2D([], [], marker="o", ls="", mfc=INK, mec="none", alpha=0.38,
               ms=(area(c) ** 0.5), label="{:g}".format(c))
        for c in (2.5, 5.0, round(ck["avg_combo"], 1))
    ]
    leg2 = ax_r.legend(
        handles=size_handles, loc="upper left", bbox_to_anchor=(0.02, 0.74),
        frameon=False, fontsize=8, labelspacing=1.25, handletextpad=1.1,
        borderaxespad=0.0, title="marker area = mean combo (hits)",
    )
    leg2.get_title().set_fontsize(8)
    leg2.get_title().set_color(INK)
    for txt in leg2.get_texts():
        txt.set_color(INK)

    ax_r.set_xlabel("mean hits landed per duel", fontsize=9, color=INK)
    ax_r.set_ylabel("mean hits taken per duel", fontsize=9, color=INK)
    ax_r.grid(True, color=GRID, alpha=0.2, lw=0.8)
    ax_r.set_axisbelow(True)

    # ------------------------------------------------------------------ shared cosmetics
    for ax in (ax_l, ax_r):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=INK, labelsize=8, length=3, width=0.8)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color(INK)

    stamp = (
        "docs/assets/data/ladder-report.json  —  {} duels per pair, seed {}, {} pairings "
        "({} duels), rendered {}".format(
            per_pair, seed, n_pairings, n_duels, _dt.date.today().isoformat()
        )
    )
    fig.text(0.5, -0.035, stamp, ha="center", va="top", fontsize=7,
             color=GRID, family="monospace")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, format="svg", transparent=True, bbox_inches="tight")
    if png_path:
        fig.savefig(png_path, format="png", dpi=170, transparent=False,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)

    return {
        "rows": rows,
        "stats": stats,
        "n_pairings": n_pairings,
        "n_duels": n_duels,
        "per_pair": per_pair,
        "seed": seed,
        "seed_rating": seed_rating,
        "runner_up": (runner_up, runner_up_aim),
        "best_aim": min((s["mean_aim_err_deg"], n) for n, s in stats.items()
                        if not n.lower().startswith("ckpt") and s["hits_landed"] > 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--png", default="", help="also write a raster preview here (debug)")
    args = ap.parse_args()

    report = load(DATA)
    info = build(report, OUT, args.png)

    print("wrote {} ({} bytes)".format(OUT, os.path.getsize(OUT)))
    print("{} contestants, {} pairings x {} duels = {} duels, seed {}".format(
        len(info["rows"]), info["n_pairings"], info["per_pair"], info["n_duels"],
        info["seed"]))
    print("{:<14} {:>7}  {:>10}  {:>6} {:>6} {:>6} {:>6}".format(
        "contestant", "elo", "W-L-D", "land", "taken", "combo", "aim"))
    for r in info["rows"]:
        s = info["stats"][r["name"]]
        print("{:<14} {:>7.1f}  {:>10}  {:>6.2f} {:>6.2f} {:>6.2f} {:>5.1f}°".format(
            r["name"], r["elo"],
            "{}-{}-{}".format(r["wins"], r["losses"], r["draws"]),
            s["hits_landed"], s["hits_taken"], s["avg_combo"], s["mean_aim_err_deg"]))
    print("seed-rating line at {:.0f}; annotation compares against runner-up {} "
          "at {:.2f}°".format(info["seed_rating"], info["runner_up"][0],
                              info["runner_up"][1]))
    print("(sharpest aim of any non-learned contestant that lands hits: {} at {:.2f}°)"
          .format(info["best_aim"][1], info["best_aim"][0]))


if __name__ == "__main__":
    main()
