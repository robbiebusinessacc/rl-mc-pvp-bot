"""Animated live-run cockpit rendered from pvpbot-flight.jsonl (gitignored).

One GIF frame per recorded live tick -- 100 ticks, 4.95 s, 50 ms/frame, so the
clip runs at exactly the 20 Hz the loop ran at. Every value drawn comes straight
out of the flight recorder written by pvpbot/deploy/loop.py: obs slots via
OBS_LAYOUT, action indices via ACTION_HEADS/CAMERA_BINS, perception via
PERCEPTION_LAYOUT, and the four per-stage latency timers.

Honest framing of the radar panel: obs["rel_pos"] is the ObsAssembler's
ESTIMATE of the enemy, in the bot's own yaw-aligned frame. On the 33 ticks
where the CNN sees the enemy it is rebuilt from the perceived aim angles plus
the distance inverted from bbox height; on the other 67 it is the last bearing
counter-rotated by the yaw the bot itself commanded on the PREVIOUS tick
(ObsAssembler._integrate_action runs before the new perception is folded in).
That one-tick lag is verified against this log at render time and the residual
is printed into the sidecar. Nothing here is server ground truth and nothing is
a world-frame path.

Provenance: the logged action is the FINAL injected action, after humanize, the
click gate and the aim/crit assists -- the bot commanded it; it is not a raw
policy sample.

Outputs
  docs/assets/live-cockpit.gif        1000x580, 100 frames, duration=50, loop
  docs/assets/data/live-cockpit.json  every derived number the frame displays

Usage:  python3 tools/figures/anim_live_cockpit.py [--out docs/assets/live-cockpit.gif]
"""
import argparse
import io
import json
import math
import os
import sys
import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Wedge

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from pvpbot.spec import (OBS_LAYOUT, ACTION_HEADS, CAMERA_BINS,  # noqa: E402
                         PERCEPTION_LAYOUT)

# head order is spec'd, not assumed: this figure indexes action[] by position.
assert [h[0] for h in ACTION_HEADS] == ["forward", "strafe", "jump", "sprint",
                                        "attack", "yaw", "pitch"]

FLIGHT = os.path.join(REPO, "pvpbot-flight.jsonl")
OUT_GIF = os.path.join(REPO, "docs", "assets", "live-cockpit.gif")
OUT_JSON = os.path.join(REPO, "docs", "assets", "data", "live-cockpit.json")

# Menlo is the house face for every figure; fall back so a fresh clone on a
# non-mac box still renders rather than dying in the font manager.
MONO = "Menlo"
try:
    import matplotlib.font_manager as _fm
    if "Menlo" not in {f.name for f in _fm.fontManager.ttflist}:
        MONO = "DejaVu Sans Mono"
except Exception:  # pragma: no cover
    MONO = "DejaVu Sans Mono"
plt.rcParams["font.family"] = MONO

BG, PANEL, GRID, INK, DIM = "#12161c", "#1a212a", "#2b3542", "#ccd6e2", "#66747f"
CYAN, AMBER, RED, GREEN, VIOLET = "#3fd0d8", "#e8a33d", "#e2564a", "#6ecf94", "#a184e8"
OFF = "#242e3a"

STAGE_BADGE = "1→4"          # sensor -> adapter -> controller -> injection
COMPONENT = "live deploy loop"
SESSION = "session A"

if not os.path.exists(FLIGHT):
    sys.exit("source clip no longer on disk: %s\n"
             "  (the live flight recorder log is gitignored; run the deploy "
             "loop with --record to regenerate it)" % FLIGHT)

rows = [json.loads(l) for l in open(FLIGHT)]
N = len(rows)
obs = np.array([r["obs"] for r in rows], dtype=np.float32)
act = np.array([r["action"] for r in rows], dtype=np.int64)
per = np.array([r["percep"] for r in rows], dtype=np.float32)
STG = ("capture", "encode", "policy", "inject")
SCOL = {"capture": DIM, "encode": CYAN, "policy": GREEN, "inject": VIOLET}
lat = np.array([[r["latency_ms"][k] for k in STG] for r in rows], dtype=np.float32)
tick_ms = np.array([r["latency_ms"]["tick"] for r in rows], dtype=np.float32)
tt = np.array([r["t"] for r in rows], dtype=np.float64)


def S(n):
    a, b = OBS_LAYOUT[n]
    return obs[:, a:b]


rel = S("rel_pos") * 8.0
dist = S("dist").ravel() * 8.0
vis = S("enemy_visible").ravel()
aey = S("aim_err_yaw").ravel() * 180.0
aep = S("aim_err_pitch").ravel() * 90.0
in_reach = S("in_reach").ravel()
PLAB = [k for k in PERCEPTION_LAYOUT]

SEEN = vis > 0.5
# before the first sighting the assembler has no bearing at all: rel_pos is
# exactly zero and dist is still the adapter's 4.0 m initial value.
HAS_EST = np.hypot(rel[:, 0], rel[:, 2]) > 1e-6
DUR = float(tt[-1] - tt[0])
CPS = float(SEEN.size and (act[:, 4] == 1).sum() / DUR)
MED_TICK = float(np.median(tick_ms[1:]))          # tick 0 is the cold-start outlier
DIST_MIN = float(dist.min())
RENDER_DATE = datetime.date.today().isoformat()

HDR = "%s · %d ticks · %.2f s · median tick %.2f ms" % (
    SESSION, N, DUR, MED_TICK)
FOOT_L = "live, through the pixel pipeline · %s" % HDR
FOOT_R = "pvpbot-flight.jsonl (%d rows) · rendered %s" % (N, RENDER_DATE)


def _run_lengths(mask):
    out, run = [], 0
    for v in mask:
        if v:
            run += 1
        elif run:
            out.append(run); run = 0
    if run:
        out.append(run)
    return out


def held_bearing_residual():
    """Verify the one-tick lag in the held-bearing rotation, both sign choices."""
    idx = [i for i in range(1, N) if not SEEN[i]]
    res = {}
    for lag, key in ((1, "prev_action"), (0, "same_action")):
        for sgn, skey in ((-1.0, "neg"), (1.0, "pos")):
            errs = []
            for i in idx:
                a = math.radians(sgn * CAMERA_BINS[act[i - lag, 5]])
                f, u, s = rel[i - 1]
                pf = f * math.cos(a) - s * math.sin(a)
                ps = f * math.sin(a) + s * math.cos(a)
                errs.append(math.hypot(pf - rel[i, 0], ps - rel[i, 2]))
            res["%s_%s" % (key, skey)] = errs
    best = min(res, key=lambda k: float(np.median(res[k])))
    return best, res, len(idx)


def panel(fig, rect, title=None, sub=None):
    ax = fig.add_axes(rect)
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(GRID); sp.set_linewidth(0.8)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        fig.text(rect[0], rect[1] + rect[3] + 0.016, title, color=INK,
                 fontsize=8, fontweight="bold")
    if sub:
        fig.text(rect[0] + rect[2], rect[1] + rect[3] + 0.016, sub, color=DIM,
                 fontsize=6.4, ha="right")
    return ax


def draw(i, fig):
    fig.clf()
    fig.patch.set_facecolor(BG)
    seen = SEEN[i]

    # ---- chrome: 1 px border, header strip, footer strip ------------------
    fig.add_artist(Rectangle((0.0006, 0.0011), 0.9988, 0.9978, transform=fig.transFigure,
                             fc="none", ec=GRID, lw=1.0, zorder=20))
    fig.add_artist(Rectangle((0, 0.9483), 1, 0.0517, transform=fig.transFigure,
                             fc=PANEL, ec="none", zorder=10))
    fig.add_artist(plt.Line2D([0, 1], [0.9483, 0.9483], transform=fig.transFigure,
                              color=GRID, lw=0.9, zorder=11))
    fig.add_artist(Rectangle((0, 0), 1, 0.0414, transform=fig.transFigure,
                             fc=PANEL, ec="none", zorder=10))
    fig.add_artist(plt.Line2D([0, 1], [0.0414, 0.0414], transform=fig.transFigure,
                              color=GRID, lw=0.9, zorder=11))

    fig.add_artist(FancyBboxPatch((0.0165, 0.9585), 0.0700, 0.0320,
                                  boxstyle="round,pad=0.004", transform=fig.transFigure,
                                  fc="#232d39", ec=GRID, lw=0.9, zorder=12))
    fig.text(0.0515, 0.9742, "STAGE " + STAGE_BADGE, color=INK, fontsize=7.6,
             fontweight="bold", ha="center", va="center", zorder=13)
    fig.text(0.0955, 0.9742,
             "%s  ·  tick %04d  ·  t=%05.2f s" % (COMPONENT, rows[i]["tick"],
                                                            tt[i] - tt[0]),
             color=DIM, fontsize=7.6, va="center", zorder=13)
    fig.text(0.984, 0.9742,
             "pvpbot-flight.jsonl  ·  %s  ·  20 Hz  ·  state %s  ·  "
             "mouse grabbed" % (SESSION, rows[i]["state"]),
             color=DIM, fontsize=7.0, ha="right", va="center", zorder=13)

    fig.text(0.016, 0.0198, FOOT_L, color=DIM, fontsize=6.3, va="center", zorder=13)
    fig.text(0.984, 0.0198, FOOT_R, color=DIM, fontsize=6.3, ha="right",
             va="center", zorder=13)

    # ---- title ------------------------------------------------------------
    fig.text(0.028, 0.898, "LIVE LOOP", color=INK, fontsize=13, fontweight="bold")
    fig.text(0.148, 0.902, "screen pixels → CNN → GRU policy → mouse + keys",
             color=CYAN, fontsize=8.5)
    fig.text(0.972, 0.902, "one GIF frame = one live tick",
             color=DIM, fontsize=7.4, ha="right")

    # ---- ego radar -------------------------------------------------------
    R = [0.028, 0.398, 0.262, 0.452]
    ax = panel(fig, R, "EGO RADAR", "the policy's estimate")
    ax.set_xlim(-8.7, 8.7); ax.set_ylim(-8.7, 8.7); ax.set_aspect("equal")
    ax.add_patch(Wedge((0, 0), 8.7, 30, 150, fc="#202b35", ec="none"))
    for r in (2, 4, 6, 8):
        ax.add_patch(Circle((0, 0), r, fill=False, ec=GRID, lw=0.6))
        ax.text(-0.72 * r + 0.10, -0.72 * r + 0.16, "%d m" % r, color=DIM, fontsize=6.2)
    ax.add_patch(Circle((0, 0), 3.0, fill=False, ec=RED, lw=1.1, ls=(0, (3, 3)), alpha=0.9))
    ax.text(0.25, 3.10, "reach 3.0 m", color=RED, fontsize=6.2)
    for s in (-1, 1):
        a = math.radians(60) * s
        ax.plot([0, 8.7 * math.sin(a)], [0, 8.7 * math.cos(a)], color="#3d4d5c", lw=0.8)
    ax.text(-8.3, 7.9, "120° camera cone", color="#5d7285", fontsize=6.0)
    # 16-tick comet, faded in 4 discrete age bands: a continuous ramp would
    # re-tint every trail dot on every frame and triples the GIF's inter-frame
    # delta for no visible gain.
    for j in range(max(0, i - 16), i):
        if not HAS_EST[j]:
            continue
        al = 0.50 - 0.10 * ((i - j - 1) // 4)
        ax.plot([rel[j, 2]], [rel[j, 0]], marker="o", ms=3.0,
                color=(CYAN if SEEN[j] else AMBER), alpha=al, mec="none")
    c = CYAN if seen else AMBER
    if HAS_EST[i]:
        if not seen:
            # A held run is a pure rotation, so the estimate sweeps an arc of
            # EXACTLY constant range (dist is bit-identical across every held
            # run in this log). Connecting the held points draws that arc.
            j = i
            while j > 0 and not SEEN[j - 1]:
                j -= 1
            k0 = max(0, j - 1)
            ax.plot(rel[k0:i + 1, 2], rel[k0:i + 1, 0], color=AMBER, lw=1.0,
                    ls=(0, (2, 2)), alpha=0.75, zorder=3)
        ax.plot([0, rel[i, 2]], [0, rel[i, 0]], color=c, lw=0.9, alpha=0.45, zorder=4)
        ax.plot([rel[i, 2]], [rel[i, 0]], marker="o", ms=12, color=c, mec=BG,
                mew=1.1, zorder=6)
    ax.plot([0], [0], marker="^", ms=12, color=GREEN, zorder=7)
    ax.text(-8.3, -8.35, "learner (fixed at centre, facing up)", color=GREEN, fontsize=6.0)

    # radar status, in the slack below the square panel
    if not HAS_EST[i]:
        # ticks 0-1: the assembler has never had a sighting, so there is no
        # bearing at all. A blip drawn at the origin would read as "the enemy is
        # on top of me"; there is simply nothing to draw yet.
        fig.text(0.028, 0.355, "NO ESTIMATE YET", color=DIM, fontsize=9.4,
                 fontweight="bold", va="center")
        fig.text(0.028, 0.320, "no sighting since the loop started",
                 color=DIM, fontsize=6.8, va="center")
        fig.text(0.028, 0.297, "dist_est at its 4.00 m initial value",
                 color=DIM, fontsize=6.4, va="center")
    else:
        fig.text(0.028, 0.355, "CNN SIGHTING" if seen else "HELD BEARING · unseen",
                 color=c, fontsize=9.4, fontweight="bold", va="center")
        fig.text(0.028, 0.320,
                 "range %4.2f m · in_reach %d · bearing %+6.1f°"
                 % (dist[i], int(in_reach[i]),
                    math.degrees(math.atan2(rel[i, 2], rel[i, 0]))),
                 color=DIM, fontsize=6.8, va="center")
        fig.text(0.028, 0.297,
                 ("range + bearing refreshed from this frame" if seen else
                  "range held · bearing rotated by yaw command"),
                 color=(CYAN if seen else DIM), fontsize=6.4, va="center")
    fig.text(0.028, 0.276,
             "never in reach this window · closest %.2f m" % DIST_MIN,
             color=DIM, fontsize=6.4, va="center")

    # ---- aim scope -------------------------------------------------------
    R = [0.300, 0.591, 0.150, 0.259]
    ax = panel(fig, R, "AIM ERROR", "degrees")
    ax.set_xlim(-33, 33); ax.set_ylim(-33, 33); ax.set_aspect("equal")
    d_ = max(dist[i], 0.5)
    gc = GREEN if seen else DIM           # a gate sized from a held range is stale
    bw = math.degrees(math.atan2(0.35, d_)); bh = math.degrees(math.atan2(1.05, d_))
    ax.add_patch(Rectangle((-bw, -bh), 2 * bw, 2 * bh, fc="none", ec=gc,
                           lw=0.9, ls=(0, (2, 2))))
    ax.axhline(0, color=GRID, lw=0.5); ax.axvline(0, color=GRID, lw=0.5)
    for dx in ((-6, -2), (2, 6)):
        ax.plot(dx, [0, 0], color=INK, lw=1.3)
        ax.plot([0, 0], dx, color=INK, lw=1.3)
    for j in range(max(0, i - 12), i + 1):
        if SEEN[j]:
            al = 0.90 - 0.20 * ((i - j) // 4)
            ax.plot([aey[j]], [aep[j]], marker="o", ms=(11 if j == i else 4),
                    color=CYAN, alpha=al, mec="none")
    ax.text(-31, 27.5, "hitbox gate", color=gc, fontsize=6.2)
    ax.text(-31, -31, "±30°", color=DIM, fontsize=6.0)
    fig.text(0.300, 0.556,
             ("yaw %+5.1f°   pitch %+5.1f°" % (aey[i], aep[i])) if seen
             else "-- no sighting --", color=(CYAN if seen else DIM),
             fontsize=8.0, va="center")
    fig.text(0.300, 0.526,
             "gate ±%.1f° × ±%.1f° at %.2f m" % (bw, bh, d_),
             color=gc, fontsize=6.4, va="center")

    # ---- camera bin ladders ---------------------------------------------
    R = [0.470, 0.495, 0.222, 0.355]
    ax = panel(fig, R, "CAMERA HEADS", "11 μ-law bins each")
    ax.set_xlim(-0.15, 11.15); ax.set_ylim(0, 3.05)
    for row, (hid, lbl) in enumerate([(6, "pitch"), (5, "yaw")]):
        y = 0.62 + row * 1.18
        for b in range(11):
            on = act[i, hid] == b
            ax.add_patch(Rectangle((b + 0.07, y), 0.86, 0.50,
                                   fc=(VIOLET if on else OFF),
                                   ec=(VIOLET if on else GRID), lw=0.6))
        ax.text(0.1, y + 0.66, "%-5s %+5.0f °/tick" % (lbl, CAMERA_BINS[act[i, hid]]),
                color=VIOLET, fontsize=7.4)
    for b, v in enumerate(CAMERA_BINS):
        ax.text(b + 0.5, 0.14, "%+.0f" % v, color=DIM, fontsize=5.6, ha="center")
    ax.text(0.1, 2.72, "commanded bin lit · ±30 °/tick = 600 °/s",
            color=DIM, fontsize=6.0)

    # ---- latency ---------------------------------------------------------
    R = [0.712, 0.655, 0.260, 0.195]
    ax = panel(fig, R, "TICK BUDGET", "50 ms @ 20 Hz")
    left = 0.0
    for k, v in zip(STG, lat[i]):
        ax.barh(0.62, v, left=left, color=SCOL[k], height=0.30)
        left += v
    ax.add_patch(Rectangle((0, 0.44), 50, 0.36, fc="none", ec=RED, lw=0.9, ls=(0, (3, 3))))
    ax.set_xlim(-1.5, 54); ax.set_ylim(0, 1.45)
    ax.text(0.5, 1.10, "%5.2f ms used  ·  %.0f%% of the 50 ms budget"
            % (lat[i].sum(), lat[i].sum() / 50 * 100), color=INK, fontsize=7.4)
    ax.text(49.5, 0.90, "50 ms", color=RED, fontsize=6.2, ha="right")
    x0 = 0.6
    for k, v in zip(STG, lat[i]):
        ax.text(x0, 0.12, "%s %.2f" % (k[:3], v), color=SCOL[k], fontsize=6.0)
        x0 += 13.0
    if i == 0:
        ax.text(0.6, 0.30, "tick 0 cold start · first CNN forward",
                color=INK, fontsize=5.8)

    # ---- perception ------------------------------------------------------
    R = [0.712, 0.295, 0.260, 0.310]
    ax = panel(fig, R, "PERCEPTION", "12 floats · visible gate 0.5")
    yy = np.arange(12)[::-1]
    ax.barh(yy, per[i], color=CYAN, height=0.62, left=0)
    ax.axvline(0, color=GRID, lw=0.6)
    # the adapter's sighting test, pvpbot/perception/adapter.py:181
    vrow = yy[PLAB.index("visible")]
    ax.plot([0.5, 0.5], [vrow - 0.46, vrow + 0.46], color=INK, lw=1.0, zorder=5)
    ax.text(0.52, vrow + 0.62, "0.5", color=INK, fontsize=5.0, va="center")
    ax.set_xlim(-1.78, 1.06); ax.set_ylim(-0.75, 11.75)
    for k, lb in enumerate(PLAB):
        ax.text(-1.74, yy[k], lb, color=DIM, fontsize=6.1, va="center", ha="left")
    ax.axvline(-0.64, color=GRID, lw=0.5)
    ax.text(-0.60, 11.30, "-1", color=DIM, fontsize=5.0)
    ax.text(1.02, 11.30, "+1", color=DIM, fontsize=5.0, ha="right")

    # ---- keys ------------------------------------------------------------
    R = [0.300, 0.295, 0.392, 0.155]
    ax = panel(fig, R, "ACTION HEADS", "forward · strafe · jump · sprint · attack")
    ax.set_xlim(0, 11.4); ax.set_ylim(0, 1.55)
    a = act[i]
    keys = [("W", a[0] == 2, 0.82), ("A", a[1] == 0, 0.82), ("S", a[0] == 0, 0.82),
            ("D", a[1] == 2, 0.82), ("JUMP", a[2] == 1, 1.45),
            ("SPRINT", a[3] == 1, 1.75), ("ATTACK", a[4] == 1, 1.75)]
    x = 0.3
    for name, on, w in keys:
        col = (RED if name == "ATTACK" else VIOLET) if on else OFF
        ax.add_patch(FancyBboxPatch((x, 0.44), w, 0.68, boxstyle="round,pad=0.055",
                                    fc=col, ec=(col if on else GRID), lw=0.6))
        ax.text(x + w / 2, 0.79, name, color=(BG if on else DIM), fontsize=7.6,
                ha="center", va="center", fontweight="bold")
        x += w + 0.24
    ax.text(0.3, 0.16, "the bot commanded these · attack fired on %d of %d ticks "
            "= %.1f clicks/s" % (int((act[:, 4] == 1).sum()), N, CPS),
            color=DIM, fontsize=6.2)

    # ---- timeline raster -------------------------------------------------
    ax = fig.add_axes([0.028, 0.090, 0.944, 0.168])
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(GRID); sp.set_linewidth(0.8)
    tracks = [("CNN sees", SEEN, CYAN),
              ("attack", act[:, 4] == 1, RED),
              ("jump", act[:, 2] == 1, GREEN),
              ("sprint", act[:, 3] == 1, VIOLET)]
    for k, (lbl, mask, col) in enumerate(tracks):
        y = len(tracks) - 1 - k
        lbl = "%s %d/%d" % (lbl, int(mask.sum()), N)
        for j in np.where(mask)[0]:
            al = 1.0 if j <= i else 0.14
            # antialiased=False: 400 sub-pixel-wide raster cells would otherwise
            # carry two blended fringe colours each and cost ~8% of the file.
            ax.add_patch(Rectangle((j, y + 0.20), 0.92, 0.60, fc=col, ec="none",
                                   alpha=al, antialiased=False))
        ax.text(-1.5, y + 0.5, lbl, color=DIM, fontsize=6.4, ha="right", va="center")
    ax.axvline(i + 0.5, color=INK, lw=1.2)
    ax.set_xlim(-16, N + 0.5); ax.set_ylim(0, len(tracks))
    ax.set_yticks([])
    ax.set_xticks([0, 20, 40, 60, 80, 99])
    ax.set_xticklabels(["0 s", "1 s", "2 s", "3 s", "4 s", "5 s"], fontsize=6.2)
    ax.tick_params(colors=DIM, length=2, pad=1)
    fig.text(0.972, 0.268, "%d-tick raster · playhead at tick %03d"
             % (N, rows[i]["tick"]), color=DIM, fontsize=6.4, ha="right")


def sidecar(out_gif, path):
    key, res, n_held = held_bearing_residual()
    ybin = {("%+d" % v): int((act[:, 5] == b).sum()) for b, v in enumerate(CAMERA_BINS)}
    pbin = {("%+d" % v): int((act[:, 6] == b).sum()) for b, v in enumerate(CAMERA_BINS)}
    yaw_deg = np.abs([CAMERA_BINS[b] for b in act[:, 5]])
    d = {
        "id": "live-cockpit",
        "title": "Live loop cockpit -- one frame per real logged tick",
        "generator": "tools/figures/anim_live_cockpit.py",
        "source": "pvpbot-flight.jsonl",
        "source_rows": N,
        "session": SESSION,
        "rendered": RENDER_DATE,
        "stage_badge": STAGE_BADGE,
        "gif": {
            "path": "docs/assets/live-cockpit.gif",
            "width": 1000, "height": 580, "frames": N,
            "frame_duration_ms": 50, "fps": 20, "loop": True,
            "bytes": os.path.getsize(out_gif) if os.path.exists(out_gif) else None,
        },
        "window": {
            "ticks": N, "duration_s": round(DUR, 3),
            "dt_median_s": round(float(np.median(np.diff(tt))), 4),
            "dt_min_s": round(float(np.diff(tt).min()), 4),
            "dt_max_s": round(float(np.diff(tt).max()), 4),
            "state": sorted({r["state"] for r in rows}),
            "grabbed_all": bool(all(r["grabbed"] for r in rows)),
            "settled_all": bool(all(r["settled"] for r in rows)),
        },
        "latency_ms": {
            "note": "tick 0 excluded as a cold-start outlier",
            "tick0": {k: float(v) for k, v in
                      zip(STG + ("tick",), list(lat[0]) + [float(tick_ms[0])])},
            "median": {k: round(float(np.median(lat[1:, j])), 3)
                       for j, k in enumerate(STG)},
            "p95": {k: round(float(np.percentile(lat[1:, j], 95)), 3)
                    for j, k in enumerate(STG)},
            "max": {k: round(float(lat[1:, j].max()), 3) for j, k in enumerate(STG)},
            "tick_median": round(MED_TICK, 3),
            "tick_p95": round(float(np.percentile(tick_ms[1:], 95)), 3),
            "tick_max": round(float(tick_ms[1:].max()), 3),
            "budget_ms": 50.0,
            "median_pct_of_budget": round(MED_TICK / 50.0 * 100, 2),
            "ticks_over_budget": int((tick_ms > 50.0).sum()),
            "encode_share_of_median_tick_pct": round(
                float(np.median(lat[1:, 1])) / MED_TICK * 100, 1),
        },
        "perception": {
            "sightings": int(SEEN.sum()),
            "held": int((~SEEN).sum()),
            "sighting_rate": round(float(SEEN.mean()), 3),
            "longest_held_run_ticks": int(max(_run_lengths(~SEEN))),
            "longest_sighting_run_ticks": int(max(_run_lengths(SEEN))),
            "dist_min_m": round(DIST_MIN, 3),
            "dist_max_m": round(float(dist.max()), 3),
            "in_reach_ticks": int(in_reach.sum()),
            "aim_err_yaw_deg_when_seen": {
                "mean_abs": round(float(np.abs(aey[SEEN]).mean()), 2),
                "min": round(float(aey[SEEN].min()), 2),
                "max": round(float(aey[SEEN].max()), 2)},
            "aim_err_pitch_deg_when_seen": {
                "mean_abs": round(float(np.abs(aep[SEEN]).mean()), 2),
                "min": round(float(aep[SEEN].min()), 2),
                "max": round(float(aep[SEEN].max()), 2)},
            "hurt_flash_max": round(float(per[:, 6].max()), 4),
        },
        "commanded_actions": {
            "note": "final injected action, after humanize / click gate / assists",
            "forward": {"back": int((act[:, 0] == 0).sum()),
                        "none": int((act[:, 0] == 1).sum()),
                        "forward": int((act[:, 0] == 2).sum())},
            "strafe": {"left": int((act[:, 1] == 0).sum()),
                       "none": int((act[:, 1] == 1).sum()),
                       "right": int((act[:, 1] == 2).sum())},
            "jump_ticks": int((act[:, 2] == 1).sum()),
            "sprint_ticks": int((act[:, 3] == 1).sum()),
            "attack_ticks": int((act[:, 4] == 1).sum()),
            "clicks_per_s": round(CPS, 2),
            "camera_yaw_bin_counts_deg_per_tick": ybin,
            "camera_pitch_bin_counts_deg_per_tick": pbin,
            "mean_abs_commanded_yaw_deg_per_tick": round(float(yaw_deg.mean()), 2),
            "mean_abs_commanded_yaw_deg_per_s": round(float(yaw_deg.mean()) * 20, 1),
            "total_commanded_yaw_travel_deg": round(float(yaw_deg.sum()), 1),
        },
        "held_bearing_lag_check": {
            "model": "rel_pos[i] == rel_pos[i-1] rotated by CAMERA_BINS[action[i-1][5]]",
            "held_ticks_tested": n_held,
            "best_variant": key,
            "residual_m": {k: {"median": round(float(np.median(v)), 6),
                               "max": round(float(np.max(v)), 6)}
                           for k, v in res.items()},
        },
        "obs_reserved_all_zero": bool(np.all(obs[:, OBS_LAYOUT["reserved"][0]:
                                                 OBS_LAYOUT["reserved"][1]] == 0.0)),
        "enemy_hp_constant": bool(np.allclose(S("enemy_hp").ravel(),
                                              S("enemy_hp").ravel()[0])),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(d, fh, indent=2)
        fh.write("\n")
    return d


def main():
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_GIF)
    ap.add_argument("--json", default=OUT_JSON)
    ap.add_argument("--colors", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig = plt.figure(figsize=(10.0, 5.8), dpi=100)
    frames = []
    for i in range(N):
        draw(i, fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=BG)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
    # ONE global palette: median-cut a montage of the raw frames for the
    # antialias blends, then append the eleven brand hexes verbatim. Appending
    # them (rather than trusting a swatch strip to survive median-cut, which at
    # this colour count it does not) is what keeps the lit violet camera bin,
    # the red 50 ms box and the amber held-bearing blip on their exact hex.
    brand = [BG, PANEL, GRID, INK, DIM, CYAN, AMBER, RED, GREEN, VIOLET, OFF]
    nbase = a.colors - len(brand)
    mont = Image.new("RGB", (frames[0].width, frames[0].height * 8))
    for k, j in enumerate(range(0, N, max(1, N // 8))[:8]):
        mont.paste(frames[j], (0, k * frames[0].height))
    base = mont.quantize(colors=nbase, method=Image.MEDIANCUT, dither=Image.NONE)
    pl = list(base.getpalette()[:nbase * 3])
    for h in brand:
        pl += [int(h[1 + 2 * j:3 + 2 * j], 16) for j in range(3)]
    # Pad to exactly `colors` entries, never to 256: the GIF colour-table size
    # sets the LZW minimum code size, and a 256-entry table costs 8 bits per
    # pixel instead of 6 (+10% on this clip, measured).
    pl += [0, 0, 0] * (a.colors - len(pl) // 3)
    pal = Image.new("P", (1, 1))
    pal.putpalette(pl)
    qs = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
    # disposal=1 (leave) lets Pillow emit inter-frame diffs -- the same frames
    # cost 4.07 MB at disposal=2 and 0.56 MB here, and the background never
    # changes so there is nothing to ghost. (A hand-rolled transparent-sentinel
    # delta was measured too: 0.564 MB, i.e. no better, so this stays simple.)
    qs[0].save(a.out, save_all=True, append_images=qs[1:], duration=50,
               loop=0, optimize=True, disposal=1)

    # Never ship a GIF that has not been decoded back and compared.
    from PIL import ImageSequence
    dec = [np.asarray(f.convert("RGB"), dtype=np.uint8)
           for f in ImageSequence.Iterator(Image.open(a.out))]
    bad = [k for k in range(min(len(dec), len(qs)))
           if not np.array_equal(dec[k], np.asarray(qs[k].convert("RGB")))]
    if len(dec) != len(qs) or bad:
        sys.exit("GIF did not round-trip: %d frames written, %d read back, "
                 "first mismatches %s" % (len(qs), len(dec), bad[:5]))
    d = sidecar(a.out, a.json)
    print("%s  %d frames  %.3f MB" % (a.out, len(qs), os.path.getsize(a.out) / 1e6))
    print("%s  held-bearing residual max %.2e m over %d held ticks (%s)"
          % (a.json,
             d["held_bearing_lag_check"]["residual_m"][
                 d["held_bearing_lag_check"]["best_variant"]]["max"],
             d["held_bearing_lag_check"]["held_ticks_tested"],
             d["held_bearing_lag_check"]["best_variant"]))


if __name__ == "__main__":
    main()
