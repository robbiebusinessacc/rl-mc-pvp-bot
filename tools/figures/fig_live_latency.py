#!/usr/bin/env python3
"""Render `docs/assets/live-latency-breakdown.svg`: where one live 50 ms tick goes.

WHAT IT PLOTS
  Left panel  - one horizontal stacked bar spanning the 50 ms tick budget, split into
                the four measured stages of the deploy loop (capture, encode, policy,
                inject) followed by an unfilled `idle` segment: the sleep the loop
                takes to hit its next deadline. Each segment is labelled with its mean
                in ms and its share of the 50 ms budget. A dashed rule marks the
                budget itself.
  Right panel  - the per-stage distribution over every tick in the recording, drawn on
                a log-x axis because inject spans four decades (microsecond key events
                in combat, ~165 ms when it drives a respawn click). Box = p25-p75,
                whiskers = p1-p99, white rule = p50, diamond = p99; individual ticks
                above p99 are drawn as points so the inject tail is visible as tail
                rather than as a summary statistic.

  Colours are the four subsystem hexes of the docset palette, one per stage, identical
  in both panels. All text is mid/dark grey and the figure is saved transparent, so it
  reads on GitHub's light and dark themes.

WHAT `tick` MEANS HERE
  The recorded `latency_ms.tick` is t4 - t0, i.e. the sum of the four stages and NOT
  the wall-clock period (pvpbot/deploy/loop.py, the `lat` dict built after t4). The
  loop then sleeps to `next_deadline` (TICK_PERIOD = 50 ms). The `idle` segment is
  therefore 50 ms minus the mean tick, and it is checked against the mean spacing of
  the recorded `t` field, which is printed at build time.

DATA
  pvpbot-flight.jsonl - GITIGNORED live-run telemetry, one JSON object per tick. This
  script reads EVERY row at build time; no count, duration or percentile is hardcoded,
  and the row count plus session duration are stamped into the figure footer. A fresh
  clone will not have this file: the script then exits with a message naming it, and
  the committed SVG is the only copy of the result.

  No smoothing, no downsampling, no outlier removal: a single live session is short
  enough to summarise exactly. The only selection made anywhere is cosmetic - the
  right panel draws individual points only for ticks above each stage's own p99,
  because plotting every tick would paint a solid bar over the box.

RUN
  python3 tools/figures/fig_live_latency.py                              # writes the SVG
  python3 tools/figures/fig_live_latency.py --png /tmp/preview.png       # raster copy
                                                                        # for eyeballing
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# -- paths ------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(_HERE))
DATA = os.path.join(REPO, "pvpbot-flight.jsonl")
OUT = os.path.join(REPO, "docs", "assets", "live-latency-breakdown.svg")

# -- constants that mirror the loop -----------------------------------------
# pvpbot/deploy/loop.py: TICK_RATE -> TICK_BUDGET_MS = 1000 / TICK_RATE
BUDGET_MS = 50.0
STAGES = ("capture", "encode", "policy", "inject")

# -- palette (the docset's seven fixed hexes) -------------------------------
C = {
    "capture": "#1d5c93",   # live   : screen grab
    "encode": "#a8621b",    # percep : CNN + adapter
    "policy": "#5b3fa8",    # policy : PolicyNet forward
    "inject": "#1f6f4a",    # sim/IO : mouse + keyboard events
}

# the matching stroke hexes of the same palette entries, used for the median
# rule and the p99 marker edge: a near-white rule reads as a block on a 2 px
# wide box under GitHub's dark theme, these do not
S = {
    "capture": "#0e3153",
    "encode": "#5f370f",
    "policy": "#31215c",
    "inject": "#0d3f29",
}
IDLE = "#8b8b8b"
BUDGET_C = "#9b1c31"
INK = "#4a4a4a"
GRID = "#8b8b8b"


def load(path):
    """Return (rows, ticks_read). Exits with a clear message if the file is absent."""
    if not os.path.exists(path):
        sys.exit(
            "missing data file: %s\n"
            "This figure is rendered from live-run telemetry that is gitignored, so a\n"
            "fresh clone will not have it. The committed SVG at %s is the rendered\n"
            "result; regenerate it only on a machine that has actually run\n"
            "`python3 -m pvpbot.deploy.run` with a flight recorder attached."
            % (path, os.path.relpath(OUT, REPO))
        )
    rows = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        sys.exit("data file %s is empty" % path)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--png", default=None, help="also write a raster preview here")
    args = ap.parse_args()

    rows = load(DATA)
    n = len(rows)
    t = np.array([r["t"] for r in rows], dtype=float)
    duration_s = float(t[-1] - t[0])
    lat = {s: np.array([r["latency_ms"][s] for r in rows], dtype=float) for s in STAGES}
    tick = np.array([r["latency_ms"]["tick"] for r in rows], dtype=float)
    settled = np.array([int(r["settled"]) for r in rows], dtype=bool)

    means = {s: float(lat[s].mean()) for s in STAGES}
    tick_mean = float(tick.mean())
    tick_p95 = float(np.percentile(tick, 95))
    idle_mean = BUDGET_MS - tick_mean
    over = int((tick > BUDGET_MS).sum())
    over_settled = int((tick > BUDGET_MS)[settled].sum())
    n_settled = int(settled.sum())
    period_ms = float(np.diff(t).mean() * 1000.0)

    # -- console receipts ---------------------------------------------------
    print("read %d ticks from %s" % (n, os.path.relpath(DATA, REPO)))
    print("session duration %.1f s   mean recorded period %.2f ms" % (duration_s, period_ms))
    for s in STAGES:
        v = lat[s]
        print(
            "  %-8s mean %7.3f ms  p50 %7.3f  p99 %7.3f  max %8.3f  (%.1f%% of budget)"
            % (s, means[s], np.percentile(v, 50), np.percentile(v, 99), v.max(),
               100.0 * means[s] / BUDGET_MS)
        )
    print("  %-8s %7.3f ms (50 - mean tick)" % ("idle", idle_mean))
    print("tick mean %.3f ms  p95 %.3f ms  capture = %.1f%% of the work"
          % (tick_mean, tick_p95, 100.0 * means["capture"] / tick_mean))
    print("over budget: %d of %d ticks; %d of %d settled-in-combat ticks"
          % (over, n, over_settled, n_settled))

    # -- figure -------------------------------------------------------------
    fig = plt.figure(figsize=(11, 3.6))
    fig.patch.set_alpha(0.0)
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.22, 1.0], left=0.035, right=0.985,
        bottom=0.345, top=0.945, wspace=0.16,
    )
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    for ax in (axL, axR):
        ax.patch.set_alpha(0.0)
        # `which="both"`: minor ticks otherwise keep matplotlib's default black,
        # which vanishes on a dark GitHub theme
        ax.tick_params(which="both", colors=INK, labelsize=8.5, length=3)
        ax.tick_params(which="minor", length=1.8)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.spines["bottom"].set_linewidth(0.8)

    # ---------------- LEFT: mean composition of one tick -------------------
    BAR_H = 0.5
    x = 0.0
    spans = []
    for s in STAGES:
        w = means[s]
        axL.add_patch(Rectangle((x, -BAR_H / 2), w, BAR_H, facecolor=C[s],
                                edgecolor="none", zorder=3))
        spans.append((s, x, w))
        x += w
    # idle: the sleep to the next deadline, drawn hollow
    axL.add_patch(Rectangle((x, -BAR_H / 2), idle_mean, BAR_H, facecolor="none",
                            edgecolor=IDLE, linewidth=1.0, zorder=3))
    spans.append(("idle", x, idle_mean))

    def seg_label(name, mid, y, ha="center"):
        axL.text(mid, y, name, ha=ha, va="top", fontsize=9, color=INK)
        axL.text(mid, y - 0.30, "%.2f ms · %.0f%%" % (
            means.get(name, idle_mean), 100.0 * means.get(name, idle_mean) / BUDGET_MS),
            ha=ha, va="top", fontsize=8.5, color=INK)

    # wide segments get a label straight below; the three hairline segments get
    # fanned labels on leader lines in their own colour
    seg_label("capture", means["capture"] / 2.0, -0.36)
    axL.text(spans[4][1] + idle_mean / 2.0, -0.36, "idle · sleep to deadline",
             ha="center", va="top", fontsize=9, color=INK)
    axL.text(spans[4][1] + idle_mean / 2.0, -0.66, "%.2f ms · %.0f%%"
             % (idle_mean, 100.0 * idle_mean / BUDGET_MS),
             ha="center", va="top", fontsize=8.5, color=INK)

    fan = {"encode": (16.0, -1.06), "policy": (24.6, -1.52), "inject": (33.6, -1.06)}
    for s in ("encode", "policy", "inject"):
        _, x0, w = [sp for sp in spans if sp[0] == s][0]
        lx, ly = fan[s]
        axL.annotate(
            "", xy=(x0 + w / 2.0, -BAR_H / 2), xytext=(lx, ly + 0.06),
            arrowprops=dict(arrowstyle="-", color=C[s], linewidth=0.9,
                            shrinkA=1.5, shrinkB=1.5), zorder=2,
        )
        axL.text(lx, ly, "%s  %.2f ms · %.1f%%" % (s, means[s],
                 100.0 * means[s] / BUDGET_MS),
                 ha="center", va="top", fontsize=8.5, color=INK)

    # work bracket over the four measured stages
    axL.plot([0.0, 0.0, tick_mean, tick_mean], [0.36, 0.46, 0.46, 0.36],
             color=INK, linewidth=0.8, solid_joinstyle="miter")
    axL.text(tick_mean / 2.0, 0.53,
             "work %.2f ms — capture is %.0f%% of it"
             % (tick_mean, 100.0 * means["capture"] / tick_mean),
             ha="center", va="bottom", fontsize=8.5, color=INK)
    axL.text(BUDGET_MS - 0.6, 0.53, "recorded mean period %.2f ms" % period_ms,
             ha="right", va="bottom", fontsize=8, color=INK)

    axL.axvline(BUDGET_MS, color=BUDGET_C, linewidth=1.2, linestyle=(0, (5, 3)),
                zorder=4)
    axL.text(BUDGET_MS + 0.8, -0.35, "TICK BUDGET", rotation=90, ha="left",
             va="center", fontsize=8, color=INK)

    axL.set_xlim(0, 54.2)
    axL.set_ylim(-1.95, 1.05)
    axL.set_yticks([])
    axL.set_xticks([0, 10, 20, 30, 40, 50])
    axL.set_xlabel("milliseconds inside one 50 ms tick\n"
                   "mean of every recorded tick · idle = budget minus work",
                   fontsize=9, color=INK, labelpad=4, linespacing=1.4)
    axL.xaxis.grid(True, color=GRID, alpha=0.25, linewidth=0.7)
    axL.set_axisbelow(True)

    # ---------------- RIGHT: per-stage distribution, log x -----------------
    ys = {"capture": 3, "encode": 2, "policy": 1, "inject": 0}
    for s in STAGES:
        v = lat[s]
        y = ys[s]
        p1, p25, p50, p75, p99 = np.percentile(v, [1, 25, 50, 75, 99])
        axR.plot([p1, p99], [y, y], color=C[s], linewidth=1.0, zorder=3)
        for cap in (p1, p99):
            axR.plot([cap, cap], [y - 0.14, y + 0.14], color=C[s], linewidth=1.0,
                     zorder=3)
        axR.add_patch(Rectangle((p25, y - 0.21), max(p75 - p25, 1e-9), 0.42,
                                facecolor=C[s], edgecolor="none", zorder=4))
        axR.plot([p50, p50], [y - 0.21, y + 0.21], color=S[s], linewidth=1.3,
                 zorder=5)
        axR.plot([p99], [y], marker="D", markersize=3.6, color=C[s],
                 markeredgecolor=S[s], markeredgewidth=0.6, zorder=6)
        tail = v[v > p99]
        if tail.size:
            rng = np.random.default_rng(0)
            axR.scatter(tail, y + rng.uniform(-0.13, 0.13, tail.size), s=5,
                        color=C[s], alpha=0.45, linewidths=0, zorder=4)
        # right-aligned so the label ends on its own p99 marker and never
        # collides with the budget rule
        axR.text(p99 * 1.06, y + 0.34, "p50 %.2f  ·  p99 %.2f" % (p50, p99),
                 ha="right", va="center", fontsize=7.5, color=INK)

    axR.axvline(BUDGET_MS, color=BUDGET_C, linewidth=1.0, linestyle=(0, (5, 3)),
                zorder=2)
    axR.text(BUDGET_MS * 1.14, 3.88, "50 ms budget", ha="left", va="center",
             fontsize=7.5, color=INK)

    inj_hi = float(lat["inject"].max())
    axR.annotate(
        "respawn clicks:\ndeliberate blocking sleeps",
        xy=(inj_hi * 0.84, 0.16), xytext=(24.0, 0.68),
        ha="right", va="center", fontsize=7.5, color=INK, linespacing=1.35,
        arrowprops=dict(arrowstyle="-", color=IDLE, linewidth=0.8, shrinkA=3,
                        shrinkB=3, connectionstyle="arc3,rad=-0.2"),
    )
    cap_hi = float(lat["capture"].max())
    axR.annotate(
        "first tick:\ncold window grab",
        xy=(cap_hi, 3.0), xytext=(cap_hi * 1.5, 3.0),
        ha="left", va="center", fontsize=7.5, color=INK, linespacing=1.35,
        arrowprops=dict(arrowstyle="-", color=IDLE, linewidth=0.8, shrinkA=2,
                        shrinkB=3),
    )
    axR.text(0.0032, -0.52,
             "left tails: mouse grab un-settled — policy and injector skipped",
             ha="left", va="center", fontsize=7, color=GRID)

    axR.set_xscale("log")
    axR.set_xlim(0.0028, 1600)
    axR.set_ylim(-0.75, 4.0)
    axR.set_yticks([ys[s] for s in STAGES])
    axR.set_yticklabels(list(STAGES), fontsize=9, color=INK)
    axR.tick_params(axis="y", length=0)
    axR.set_xticks([0.01, 0.1, 1, 10, 100])
    axR.set_xticklabels(["0.01", "0.1", "1", "10", "100"])
    axR.set_xlabel(
        "per-stage latency (ms, log scale)\n"
        "box p25–p75 · whiskers p1–p99 · points above p99",
        fontsize=9, color=INK, labelpad=4, linespacing=1.4,
    )
    axR.xaxis.grid(True, which="major", color=GRID, alpha=0.25, linewidth=0.7)
    axR.xaxis.grid(True, which="minor", color=GRID, alpha=0.12, linewidth=0.5)
    axR.set_axisbelow(True)

    # ---------------- footer ----------------------------------------------
    stamp = _dt.date.today().isoformat()
    fig.text(
        0.5, 0.10,
        "mean tick %.2f ms · p95 %.2f ms · over the 50 ms budget on %d of %s ticks, "
        "and on %d of %s settled-in-combat ticks"
        % (tick_mean, tick_p95, over, format(n, ","), over_settled,
           format(n_settled, ",")),
        ha="center", va="center", fontsize=8.5, color=INK,
    )
    fig.text(
        0.5, 0.028,
        "pvpbot-flight.jsonl, %s ticks over %.1f s, rendered %s"
        % (format(n, ","), duration_s, stamp),
        ha="center", va="center", fontsize=8, color=GRID, family="monospace",
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, format="svg", transparent=True, bbox_inches="tight")
    print("wrote %s (%.1f kB)" % (os.path.relpath(OUT, REPO),
                                  os.path.getsize(OUT) / 1024.0))
    if args.png:
        fig.savefig(args.png, format="png", dpi=160, transparent=False,
                    facecolor="white", bbox_inches="tight")
        print("wrote %s" % args.png)
    plt.close(fig)


if __name__ == "__main__":
    main()
