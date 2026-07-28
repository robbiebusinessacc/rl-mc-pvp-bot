#!/usr/bin/env python3
"""Render docs/assets/perception-learning-curve.svg -- the perception CNN's
learning curve in physical units, plus a first-vs-last slope chart of all
twelve PERCEPTION_LAYOUT output channels.

DATA (gitignored, not present in a fresh clone)
    runs/perception/metrics.jsonl
        One JSON object per eval, written by pvpbot/perception/train.py in
        APPEND mode, so the file concatenates every perception run ever
        launched against that path.  Run boundaries are detected here as
        step-counter resets (step[i] <= step[i-1]); only the FINAL contiguous
        run is plotted, and the figure stamps how many records were read and
        how many were discarded as belonging to earlier runs.

        Every "mae/*" key is already in physical units -- degrees, hit points,
        pixels, blocks/tick -- scaled by _SLOT_UNITS in
        pvpbot/perception/train.py:112-125.  Nothing here rescales them.

LEFT PANEL
    Four MAE series versus training step on a log y-axis in degrees.  Raw
    per-eval points are drawn as light markers; the solid/dashed line through
    them is a centered rolling MEDIAN over 5 evals (1,000 training steps).
    The median is cosmetic only -- it is drawn on top of every raw point, so
    the eval-to-eval noise the run actually has stays visible, including the
    upward spike in the last eval.

RIGHT PANEL
    All twelve output channels, first eval versus last eval of the same run,
    each normalized to its own first value = 100%.  Improvement slopes down.
    No smoothing at all on this panel: it is the literal first and last row.

WHAT THE PLOTTED RANGE IS NOT
    The first record in the whole FILE reads 23.29 deg on
    mae/aim_err_yaw_deg_vis, but it belongs to an earlier, colder run and is
    discarded here.  Within the final run the headline series runs 11.41 ->
    9.38 deg, and those are the only endpoints this figure states.  Anything
    quoting "23.3 -> 9.4" is spanning two different runs.

SOURCES FOR THE ANNOTATIONS
    stack=2 (v13 conv1 is (32, 6, 8, 8))  runs/perception/perception_v13.pt
    single-frame eval batches             pvpbot/perception/train.py:139
    frozen-window replication             pvpbot/models.py:95
    reserved has loss weight 0.0          pvpbot/perception/train.py:51

Run from anywhere:
    python3 tools/figures/fig_perception_curve.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO_ROOT, "runs", "perception", "metrics.jsonl")
OUT = os.path.join(REPO_ROOT, "docs", "assets", "perception-learning-curve.svg")

# --------------------------------------------------------------------------
# Palette -- the only seven hexes used anywhere in these docs
# --------------------------------------------------------------------------
C_SIM = "#1f6f4a"
C_PERCEPTION = "#a8621b"
C_POLICY = "#5b3fa8"
C_LIVE = "#1d5c93"
C_DEGRADED = "#9b1c31"
INK = "#4a4a4a"
FAINT = "#8b8b8b"

# --------------------------------------------------------------------------
# Left panel: which MAE series, in which colour
# --------------------------------------------------------------------------
SERIES = [
    ("mae/aim_err_yaw_deg_vis", "aim yaw error, enemy visible", C_PERCEPTION, "-", 2.6, 1.00),
    ("mae/aim_err_yaw_deg", "aim yaw error, all frames", C_PERCEPTION, "--", 1.4, 0.90),
    ("mae/aim_err_pitch_deg_vis", "aim pitch error, enemy visible", C_LIVE, "-", 1.7, 1.00),
    ("mae/self_pitch_deg", "own camera pitch error", C_POLICY, "-", 1.7, 1.00),
]

# --------------------------------------------------------------------------
# Right panel: all twelve PERCEPTION_LAYOUT slots (pvpbot/spec.py:68-81),
# in layout order, with the metrics key and a short display name + unit.
# --------------------------------------------------------------------------
CHANNELS = [
    ("mae/aim_err_yaw_deg", "aim_err_yaw", "deg"),
    ("mae/aim_err_pitch_deg", "aim_err_pitch", "deg"),
    ("mae/bbox_height_frac", "bbox_height", "frac"),
    ("mae/visible_p", "visible", "p"),
    ("mae/self_pitch_deg", "self_pitch", "deg"),
    ("mae/self_hp_hp", "self_hp", "hp"),
    ("mae/hurt_flash_p", "hurt_flash", "p"),
    ("mae/rel_screen_vx_px", "rel_screen_vx", "px"),
    ("mae/rel_screen_vy_px", "rel_screen_vy", "px"),
    ("mae/self_speed_bpt", "self_speed", "blocks/tick"),
    ("mae/enemy_on_ground_p", "enemy_on_ground", "p"),
    ("mae/reserved", "reserved", ""),
]

# Channels whose last eval is at or above their first eval get the degraded
# colour.  Membership is DECIDED FROM THE DATA below, not hard-coded.


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
def load_records(path: str):
    if not os.path.exists(path):
        sys.exit(
            "ERROR: {} not found.\n"
            "This figure is rendered from perception training telemetry that is\n"
            "gitignored, so a fresh clone will not have it. A comparable run comes from\n"
            "    python3 -m pvpbot.perception.train --steps 9000 \\\n"
            "        --metrics runs/perception/metrics.jsonl\n"
            "though the plotted run also passed --init and --real-data (the exact flags\n"
            "are recorded in the meta block of runs/perception/perception_v13.pt).\n"
            "The committed SVG at {} is the only copy of that run."
            .format(path, os.path.relpath(OUT, REPO_ROOT))
        )
    recs = []
    for lineno, line in enumerate(open(path, "r"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError as exc:
            sys.exit("ERROR: {}:{} is not valid JSON ({})".format(path, lineno, exc))
    if not recs:
        sys.exit("ERROR: {} is empty.".format(path))
    return recs


def final_run(recs):
    """Split on step-counter resets; return (final_run, n_read, n_discarded)."""
    start = 0
    for i in range(1, len(recs)):
        if recs[i]["step"] <= recs[i - 1]["step"]:
            start = i
    return recs[start:], len(recs), start


def rolling_median(y, window=5):
    y = np.asarray(y, dtype=float)
    half = window // 2
    out = np.empty_like(y)
    for i in range(len(y)):
        lo = max(0, i - half)
        hi = min(len(y), i + half + 1)
        out[i] = np.median(y[lo:hi])
    return out


def fmt_pair(a, b):
    """Format a first/last pair at a shared precision set by the smaller value."""
    m = min(abs(a), abs(b))
    dp = 2 if m >= 1.0 else (3 if m >= 0.1 else 4)
    return "{0:.{2}f} -> {1:.{2}f}".format(a, b, dp)


def label_column(values, lo, hi):
    """Evenly spaced label slots in `values` order, spanning [lo, hi].

    Twelve endpoints between 34% and 135% cannot be labelled in place without
    collisions, so labels go in a fixed column joined to their endpoint by a
    leader line. Rank order is preserved; the vertical position of the LABEL
    carries no quantity, which is why every label repeats its own number.
    """
    order = np.argsort(-np.asarray(values, dtype=float))  # largest at the top
    slots = np.linspace(hi, lo, len(values))
    out = np.empty(len(values), dtype=float)
    for slot, idx in zip(slots, order):
        out[idx] = slot
    return out


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def main() -> None:
    recs = load_records(DATA)
    run, n_read, n_discarded = final_run(recs)
    steps = np.array([r["step"] for r in run], dtype=float)
    eval_n = int(run[-1].get("eval_n", 0))

    missing = [k for k, _, _ in CHANNELS if k not in run[0] or k not in run[-1]]
    missing += [k for k, _, _, _, _, _ in SERIES if k not in run[0]]
    if missing:
        sys.exit("ERROR: final run is missing required keys: {}".format(sorted(set(missing))))

    print("read {} records from {}".format(n_read, os.path.relpath(DATA, REPO_ROOT)))
    print("discarded {} records belonging to earlier runs (step-counter resets)"
          .format(n_discarded))
    print("final run: {} records, step {:.0f} -> {:.0f}, {} synthetic frames per eval"
          .format(len(run), steps[0], steps[-1], eval_n))

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.24, 1.0], "wspace": 0.24}
    )
    fig.patch.set_alpha(0.0)
    for ax in (axL, axR):
        ax.set_facecolor("none")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(FAINT)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=INK, labelsize=8, length=3, width=0.8)

    # ---------------- LEFT: MAE versus training step ----------------
    x_end = steps[-1]
    axL.set_xlim(0, x_end * 1.155)

    print("\nleft panel -- final-run endpoints (degrees):")
    for key, label, colour, style, lw, alpha in SERIES:
        y = np.array([r[key] for r in run], dtype=float)
        axL.plot(steps, y, marker="o", ms=2.4, lw=0, color=colour, alpha=0.32 * alpha,
                 zorder=2)
        axL.plot(steps, rolling_median(y), ls=style, lw=lw, color=colour, alpha=alpha,
                 label=label, zorder=3, solid_capstyle="round")
        # last raw eval, marked and given its value in place
        axL.plot([x_end], [y[-1]], "o", ms=4.4 if lw > 2 else 3.4, color=colour, zorder=4)
        axL.text(x_end * 1.022, y[-1],
                 "{:.1f} deg".format(y[-1]) if lw > 2 else "{:.1f}".format(y[-1]),
                 color=colour, fontsize=7.8, va="center", ha="left",
                 fontweight="bold" if lw > 2 else "normal")
        print("  {:<30s} {:6.2f} -> {:5.2f}  (best {:5.2f} at step {:.0f})"
              .format(label, y[0], y[-1], y.min(), steps[int(np.argmin(y))]))

    head = np.array([r["mae/aim_err_yaw_deg_vis"] for r in run], dtype=float)
    axL.set_yscale("log")
    axL.set_ylim(2.25, 18.6)
    axL.set_yticks([2.5, 3, 4, 5, 6, 8, 10, 12])
    axL.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "{:g}".format(v)))
    axL.yaxis.set_minor_formatter(NullFormatter())
    axL.set_xticks([0, 2000, 4000, 6000, 8000])
    axL.set_xticklabels(["0", "2,000", "4,000", "6,000", "8,000"])
    axL.grid(True, which="major", color=FAINT, alpha=0.22, lw=0.6)
    axL.set_axisbelow(True)
    axL.set_xlabel("training step (one eval every 200 steps)", color=INK, fontsize=8.5)
    axL.set_ylabel("mean absolute error (degrees, log scale)", color=INK, fontsize=8.5)

    # headline series: mark and label the first raw eval
    axL.plot([steps[0]], [head[0]], "o", ms=4.2, color=C_PERCEPTION, zorder=4)
    axL.annotate(
        "{:.1f} deg".format(head[0]),
        xy=(steps[0], head[0]), xytext=(steps[0] + 190, head[0] * 1.09),
        color=C_PERCEPTION, fontsize=8.6, fontweight="bold", va="bottom", ha="left",
    )
    best_i = int(np.argmin(head))
    # the last eval is a noisy tick upward; name the run's lowest eval so the
    # 9.4 endpoint is not read as the best the run reached
    axL.annotate(
        "enemy-visible yaw:\nlowest eval {:.1f} deg at step {:,.0f}".format(
            head[best_i], steps[best_i]),
        xy=(steps[best_i], head[best_i]),
        xytext=(steps[best_i] - 480, 6.05),
        color=C_PERCEPTION, fontsize=7.2, ha="right", va="center", alpha=0.9,
        linespacing=1.5,
        arrowprops=dict(arrowstyle="-|>", lw=0.7, color=C_PERCEPTION, alpha=0.65,
                        mutation_scale=7, shrinkA=2, shrinkB=3),
    )
    axL.text(150, 18.0, "dots: every eval\nline: rolling median over 5 evals",
             color=FAINT, fontsize=6.8, va="top", ha="left", linespacing=1.5)
    leg = axL.legend(loc="upper right", fontsize=7.3, frameon=False, handlelength=2.4,
                     borderaxespad=0.15, labelspacing=0.30, handletextpad=0.6)
    for txt in leg.get_texts():
        txt.set_color(INK)

    # ---------------- RIGHT: first-vs-last slope chart ----------------
    firsts, lasts, names = [], [], []
    for key, name, unit in CHANNELS:
        firsts.append(float(run[0][key]))
        lasts.append(float(run[-1][key]))
        names.append((name, unit))
    firsts = np.array(firsts)
    lasts = np.array(lasts)
    pct = 100.0 * lasts / firsts

    print("\nright panel -- first vs last eval, all twelve channels:")
    ymin = min(pct.min(), 100.0) - 9.0
    ymax = max(pct.max(), 100.0) + 6.0
    pad = (ymax - ymin) * 0.035
    label_y = label_column(pct, ymin + pad, ymax - pad)

    x_slope, x_leader, x_text = 1.0, 1.32, 1.36
    axR.plot([-0.34, x_slope + 0.04], [100.0, 100.0], color=FAINT, lw=0.8,
             ls=(0, (4, 3)), alpha=0.8, zorder=1)
    axR.text(-0.60, 100.0 + (ymax - ymin) * 0.075, "worse", color=C_DEGRADED,
             fontsize=7.0, ha="left", va="bottom")
    axR.text(-0.60, 100.0 - (ymax - ymin) * 0.075, "better", color=C_SIM,
             fontsize=7.0, ha="left", va="top")

    for i, (name, unit) in enumerate(names):
        worse = pct[i] >= 100.0
        colour = C_DEGRADED if worse else C_SIM
        lw = 2.0 if worse else 1.25
        alpha = 0.95 if worse else 0.78
        axR.plot([0, x_slope], [100.0, pct[i]], color=colour, lw=lw, alpha=alpha,
                 solid_capstyle="round", zorder=3 if worse else 2)
        axR.plot([0, x_slope], [100.0, pct[i]], "o", ms=3.0, color=colour, alpha=alpha,
                 zorder=4)
        axR.plot([x_slope + 0.03, x_leader], [pct[i], label_y[i]], lw=0.6, color=colour,
                 alpha=0.5, zorder=2)
        unit_txt = " {}".format(unit) if unit else ""
        axR.text(
            x_text, label_y[i],
            "{}  {:.0f}%   {}{}".format(name, pct[i], fmt_pair(firsts[i], lasts[i]), unit_txt),
            color=colour, fontsize=6.9, va="center", ha="left",
            fontweight="bold" if worse else "normal",
        )
        print("  {:<18s} {:9.4f} -> {:9.4f} {:<12s} {:6.1f}%  {}"
              .format(name, firsts[i], lasts[i], unit, pct[i], "WORSE" if worse else ""))

    axR.set_xlim(-0.66, 2.62)
    axR.set_ylim(ymin, ymax)
    axR.set_xticks([0, x_slope])
    axR.set_xticklabels(
        ["first eval\nstep {:,.0f}".format(steps[0]), "last eval\nstep {:,.0f}".format(steps[-1])],
        fontsize=7.8,
    )
    axR.set_yticks([40, 60, 80, 100, 120, 140])
    axR.set_yticklabels(["40%", "60%", "80%", "100%", "120%", "140%"])
    axR.set_ylabel("MAE relative to its own first eval", color=INK, fontsize=8.5)
    axR.grid(True, axis="y", color=FAINT, alpha=0.22, lw=0.6)
    axR.set_axisbelow(True)
    axR.spines["bottom"].set_visible(False)
    axR.tick_params(axis="x", length=0)
    for tick in axR.get_xticklabels():
        tick.set_horizontalalignment("center")

    note = (
        "rel_screen_vx regresses and rel_screen_vy improves least partly because the synthetic eval set is built\n"
        "from single frames (pvpbot/perception/train.py:139), so a stacked model is scored on a temporally frozen\n"
        "window (pvpbot/models.py:95). reserved carries loss weight 0.0 and is never trained (train.py:51)."
    )
    fig.text(0.505, -0.045, note, color=INK, fontsize=6.8, ha="left", va="top",
             linespacing=1.55)

    stamp = ("runs/perception/metrics.jsonl, {} records read, {} discarded as earlier runs, "
             "final run of {} records ({} synthetic frames per eval), rendered {}"
             .format(n_read, n_discarded, len(run), eval_n, date.today().isoformat()))
    fig.text(0.0, -0.185, stamp, color=FAINT, fontsize=6.6, ha="left", va="top")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, format="svg", transparent=True, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("\nwrote {} ({:,} bytes)".format(os.path.relpath(OUT, REPO_ROOT),
                                           os.path.getsize(OUT)))
    print("stamp: {}".format(stamp))


if __name__ == "__main__":
    main()
