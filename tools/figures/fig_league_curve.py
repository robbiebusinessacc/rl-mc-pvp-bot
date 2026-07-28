#!/usr/bin/env python3
"""Render docs/assets/league-training-curve-annotated.svg.

WHAT IT PLOTS
    Top panel  - league-internal Elo of the self-play learner against cumulative
                 environment steps, for both branches of the fov1 run: the
                 original branch A and the branch forked from
                 runs/fov1/ckpt_live_set2.pt at step 44.63e9. Each branch is
                 drawn as a per-bucket min/max band of the raw series
                 (alpha 0.15) plus a rolling median over ROLL updates. A rug
                 along the top edge marks every "resumed from ... at step N"
                 line in runs/fov1/train.log. Saved checkpoints are overlaid as
                 dots at (meta.step, meta.elo), ringed in their branch colour.
    Bottom panel - league pool size as a step plot, annotated at its two
                 plateaus (20 frozen self-copies, then 20 + 4 scripted).

DATA (all gitignored; a GitHub cloner will not have it)
    runs/fov1/metrics.csv   columns update, step, elo, pool_size
    runs/fov1/train.log     lines matching ^resumed from .* at step N
    runs/fov1/*.pt          checkpoint meta {"step", "elo"} (optional overlay)

MANDATORY PREPROCESSING (the CSV is append-mode across many resumed sessions)
    1. Split into branches at the single large drop in the update counter.
       Small drops (<= ~20 updates) are ordinary resume rewinds and stay inside
       a branch; the one large drop is the fork.
    2. Dedupe within each branch keeping the LAST row per update value. This
       collapses both the resume rewinds and the region where two trainer
       processes appended concurrently (last write wins).
    3. Assert step is strictly monotone within each branch after dedupe.
    Raw row count, per-branch deduped counts and resume-marker count are
    printed and stamped into the figure footer.

SMOOTHING
    Rolling median over ROLL = 200 updates (~0.84e9 steps). Median, not mean, so
    the single-update Elo collapses that follow a pool refresh do not drag the
    line. The raw series stays visible underneath as a 0.15-alpha band drawn
    from per-bucket minima and maxima, so every excursion survives; the band is
    an exact envelope of the raw data, not a decimation of it. The envelope and
    the step-compressed pool trace exist only to keep the committed SVG small.

RUN
    python3 tools/figures/fig_league_curve.py        # from the repo root
"""

import csv
import datetime
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_CSV = os.path.join(REPO, "runs", "fov1", "metrics.csv")
DATA_LOG = os.path.join(REPO, "runs", "fov1", "train.log")
CKPT_GLOB = os.path.join(REPO, "runs", "fov1", "*.pt")
OUT = os.path.join(REPO, "docs", "assets", "league-training-curve-annotated.svg")

ROLL = 200                 # rolling-median window, in updates
FORK_CKPT = "ckpt_live_set2.pt"

# --------------------------------------------------------------------------
# Palette (the seven repo hexes only) + neutral text/grid tones
# --------------------------------------------------------------------------
C_SIM = "#1f6f4a"          # sim green   - branch A
C_PERCEPTION = "#a8621b"   # amber       - forked branch B
C_POLICY = "#5b3fa8"       # purple      - pool size
C_CONTRACT_FILL = "#e8e8e8"  # checkpoint blobs
C_CONTRACT_EDGE = "#8b8b8b"
# The style guide allows #4a4a4a or #8b8b8b for figure text. #4a4a4a scores
# 2.1:1 against the GitHub dark background and is unreadable there, so every
# glyph in this figure uses #8b8b8b (3.4:1 on light, 5.6:1 on dark). Emphasis
# comes from size and from the series colours, not from a darker grey.
TEXT = "#8b8b8b"
MUTED = "#8b8b8b"


def die(msg):
    sys.stderr.write("fig_league_curve: %s\n" % msg)
    sys.exit(1)


# --------------------------------------------------------------------------
# Load + preprocess
# --------------------------------------------------------------------------
def load_metrics():
    if not os.path.exists(DATA_CSV):
        die("missing %s\n"
            "  This figure is rendered from a gitignored training run. A fresh\n"
            "  clone will not have runs/. Produce it with:\n"
            "    python3 -m pvpbot.train.run --out runs/fov1 ...\n"
            "  or use the committed SVG at docs/assets/." % DATA_CSV)
    with open(DATA_CSV, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        die("%s is empty" % DATA_CSV)
    return rows


def split_branches(rows):
    """Split at the single largest drop in the update counter."""
    best_mag, best_idx = 0, None
    prev = None
    for i, r in enumerate(rows):
        u = int(r["update"])
        if prev is not None and prev - u > best_mag:
            best_mag, best_idx = prev - u, i
        prev = u
    if best_idx is None or best_mag < 100:
        # No fork present in this file: everything is one branch.
        return rows, [], 0
    return rows[:best_idx], rows[best_idx:], best_mag


def dedupe_last_per_update(rows):
    """Keep the LAST row seen for each update value, ordered by update."""
    keep = {}
    for r in rows:
        keep[int(r["update"])] = r
    order = sorted(keep)
    return [keep[u] for u in order]


def columns(rows):
    step = np.array([int(r["step"]) for r in rows], dtype=np.float64) / 1e9
    elo = np.array([float(r["elo"]) for r in rows], dtype=np.float64)
    pool = np.array([int(r["pool_size"]) for r in rows], dtype=np.float64)
    upd = np.array([int(r["update"]) for r in rows], dtype=np.int64)
    return step, elo, pool, upd


def rolling_median(y, win):
    """Centred rolling median; edges fall back to an expanding median."""
    n = len(y)
    if n == 0:
        return y
    win = min(win, n)
    out = np.empty(n, dtype=np.float64)
    half = win // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.median(y[lo:hi])
    return out


def envelope(x, y, nbuckets=900):
    """Per-bucket (min, max) of y. Drawn as a band this reproduces every
    excursion in the raw series exactly -- unlike decimation, which would drop
    the one-update Elo collapses this figure exists to show -- while emitting
    ~900 vertices instead of ~24,000."""
    n = len(x)
    if n <= nbuckets:
        return x, y, y
    edges = np.linspace(0, n, nbuckets + 1).astype(int)
    xs, lo, hi = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        xs.append(x[a:b].mean())
        lo.append(y[a:b].min())
        hi.append(y[a:b].max())
    return np.array(xs), np.array(lo), np.array(hi)


def compress_step(x, y):
    """Keep only the points where a step series changes value, plus the ends."""
    if len(x) < 3:
        return x, y
    keep = np.ones(len(x), dtype=bool)
    keep[1:-1] = y[1:-1] != y[:-2]
    return x[keep], y[keep]


def load_resumes():
    """Every '^resumed from <path> at step <N>' line in train.log."""
    if not os.path.exists(DATA_LOG):
        sys.stderr.write("fig_league_curve: WARNING no %s, rug omitted\n" % DATA_LOG)
        return [], []
    pat = re.compile(r"^resumed from (\S+) at step (\d+)")
    steps, srcs = [], []
    with open(DATA_LOG, errors="replace") as fh:
        for line in fh:
            m = pat.match(line)
            if m:
                srcs.append(os.path.basename(m.group(1)))
                steps.append(int(m.group(2)) / 1e9)
    return steps, srcs


def load_checkpoints():
    """(name, step_B, elo, update) for every ckpt whose meta loads cheaply."""
    try:
        import torch
    except Exception:
        sys.stderr.write("fig_league_curve: torch unavailable, ckpt dots skipped\n")
        return []
    out = []
    for path in sorted(glob.glob(CKPT_GLOB)):
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            meta = ck.get("meta", {})
            out.append((os.path.basename(path),
                        float(meta["step"]) / 1e9,
                        float(meta["elo"]),
                        int(ck.get("update", 0))))
        except Exception as exc:  # unreadable / partially written ckpt
            sys.stderr.write("fig_league_curve: skipping %s (%s)\n"
                             % (os.path.basename(path), exc))
    return out


def assign_branch(ck_update, ck_elo, upd_a, elo_a, upd_b, elo_b):
    """Which branch a checkpoint belongs to: the one whose Elo at that update
    is closest to the checkpoint's recorded meta.elo. Steps overlap across the
    fork, so step alone cannot disambiguate."""
    def probe(upd, elo):
        idx = np.searchsorted(upd, ck_update)
        if idx >= len(upd) or upd[idx] != ck_update:
            return None
        return abs(elo[idx] - ck_elo)
    da, db = probe(upd_a, elo_a), probe(upd_b, elo_b)
    if da is None and db is None:
        return None
    if db is None:
        return "A"
    if da is None:
        return "B"
    return "A" if da <= db else "B"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    rows = load_metrics()
    n_raw = len(rows)
    raw_a, raw_b, fork_mag = split_branches(rows)
    A = dedupe_last_per_update(raw_a)
    B = dedupe_last_per_update(raw_b)
    n_a, n_b = len(A), len(B)

    step_a, elo_a, pool_a, upd_a = columns(A)
    if n_b:
        step_b, elo_b, pool_b, upd_b = columns(B)
    else:
        step_b = elo_b = pool_b = np.array([])
        upd_b = np.array([], dtype=np.int64)

    # -- mandatory monotonicity check -------------------------------------
    for name, st in (("A", step_a), ("B", step_b)):
        if len(st) > 1 and not np.all(np.diff(st) > 0):
            bad = int(np.argmin(np.diff(st)))
            die("branch %s step is NOT monotone after dedupe at index %d "
                "(%.6fB -> %.6fB)" % (name, bad, st[bad], st[bad + 1]))

    resume_steps, resume_srcs = load_resumes()
    n_resume = len(resume_steps)

    print("raw rows                : %d" % n_raw)
    print("fork: update drop of    : %d  (branch B starts at step %.3fB)"
          % (fork_mag, step_b[0] if n_b else float("nan")))
    print("branch A deduped rows   : %d   step %.3fB - %.3fB   update %d - %d"
          % (n_a, step_a[0], step_a[-1], upd_a[0], upd_a[-1]))
    print("branch B deduped rows   : %d   step %.3fB - %.3fB   update %d - %d"
          % (n_b, step_b[0] if n_b else 0, step_b[-1] if n_b else 0,
             upd_b[0] if n_b else 0, upd_b[-1] if n_b else 0))
    print("step monotone per branch: True")
    print("resume markers          : %d  (sources: %s)"
          % (n_resume, ", ".join(sorted(set(resume_srcs)))))

    med_a = rolling_median(elo_a, ROLL)
    med_b = rolling_median(elo_b, ROLL) if n_b else np.array([])

    print("branch A rolling median : start %.0f  peak %.0f @ %.2fB  end %.0f"
          % (med_a[0], med_a.max(), step_a[int(np.argmax(med_a))], med_a[-1]))
    if n_b:
        print("branch B rolling median : start %.0f  peak %.0f @ %.2fB  end %.0f"
              % (med_b[0], med_b.max(), step_b[int(np.argmax(med_b))], med_b[-1]))
    print("branch A raw elo min/max: %.0f / %.0f" % (elo_a.min(), elo_a.max()))
    print("pool_size plateaus      : %s" % sorted(set(int(v) for v in pool_a)))

    cks = load_checkpoints()

    # ----------------------------------------------------------------------
    # Figure
    # ----------------------------------------------------------------------
    fig = plt.figure(figsize=(11, 4.6))
    gs = GridSpec(2, 1, height_ratios=[3.0, 1.0], hspace=0.16,
                  left=0.062, right=0.985, top=0.945, bottom=0.355)
    ax = fig.add_subplot(gs[0])
    axp = fig.add_subplot(gs[1], sharex=ax)

    for a in (ax, axp):
        a.set_facecolor("none")
        a.grid(True, color=MUTED, alpha=0.22, linewidth=0.6)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(MUTED)
            a.spines[side].set_linewidth(0.8)
        a.tick_params(colors=TEXT, labelsize=8.5, length=3, width=0.8)

    # -- Elo -------------------------------------------------------------
    ex, elo_, ehi = envelope(step_a, elo_a)
    ax.fill_between(ex, elo_, ehi, color=C_SIM, alpha=0.15, linewidth=0, zorder=2)
    mx, _, mmed = envelope(step_a, med_a, 1400)
    ax.plot(mx, mmed, color=C_SIM, linewidth=1.7, zorder=4,
            label="branch A  (median of %d updates)" % ROLL)
    if n_b:
        bx, blo, bhi = envelope(step_b, elo_b, 450)
        ax.fill_between(bx, blo, bhi, color=C_PERCEPTION, alpha=0.15,
                        linewidth=0, zorder=3)
        nx, _, nmed = envelope(step_b, med_b, 700)
        ax.plot(nx, nmed, color=C_PERCEPTION, linewidth=1.7, zorder=5,
                label="branch B  forked from %s" % FORK_CKPT)

    ylo = 600.0
    yhi = max(elo_a.max(), elo_b.max() if n_b else 0) * 1.18
    ax.set_ylim(ylo, yhi)
    span = yhi - ylo

    # -- resume rug along the top edge ------------------------------------
    rug_top = yhi
    rug_bot = yhi - 0.028 * span
    for s in resume_steps:
        ax.plot([s, s], [rug_bot, rug_top], color=MUTED, linewidth=0.9,
                alpha=0.85, solid_capstyle="butt", zorder=6, clip_on=False)
    if resume_steps:
        ax.text(0.6, rug_bot - 0.038 * span,
                "each tick above: one 'resumed from' line in train.log (%d)" % n_resume,
                color=MUTED, fontsize=7.6, ha="left", va="center")

    # -- checkpoint dots ---------------------------------------------------
    # Steps overlap across the fork, so the ring is coloured by the branch the
    # checkpoint's own meta.elo matches at its own update number.
    n_ck = 0
    for name, cstep, celo, cupd in cks:
        if not (ylo <= celo <= yhi):
            print("  ckpt off-axis, skipped: %s (elo %.0f)" % (name, celo))
            continue
        br = assign_branch(cupd, celo, upd_a, elo_a, upd_b, elo_b)
        edge = {"A": C_SIM, "B": C_PERCEPTION}.get(br, C_CONTRACT_EDGE)
        print("  ckpt %-26s step %6.2fB  update %6d  elo %7.1f  -> branch %s"
              % (name, cstep, cupd, celo, br or "?"))
        ax.plot([cstep], [celo], marker="o", markersize=4.0,
                markerfacecolor=C_CONTRACT_FILL, markeredgecolor=edge,
                markeredgewidth=0.9, linestyle="none", zorder=7)
        n_ck += 1
    if n_ck:
        ax.plot([], [], marker="o", markersize=4.0, linestyle="none",
                markerfacecolor=C_CONTRACT_FILL, markeredgecolor=C_CONTRACT_EDGE,
                markeredgewidth=0.8,
                label="saved checkpoint (%d) at meta.step / meta.elo,\nring colour is its branch" % n_ck)
    print("checkpoint dots drawn   : %d of %d" % (n_ck, len(cks)))

    # -- fork marker -------------------------------------------------------
    if n_b:
        fork_x = step_b[0]
        for a in (ax, axp):
            a.axvline(fork_x, color=C_PERCEPTION, linestyle=(0, (4, 3)),
                      linewidth=0.9, alpha=0.75, zorder=1)
        ax.annotate("fork at %.2fB steps\n(%s, Elo %.0f)"
                    % (fork_x, FORK_CKPT, elo_b[0]),
                    xy=(fork_x, elo_b[0]),
                    xytext=(fork_x - 25.0, ylo + 0.16 * span),
                    color=TEXT, fontsize=7.8, ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color=C_PERCEPTION,
                                    linewidth=0.9, shrinkA=1, shrinkB=3))

    # -- annotations on the Elo curve --------------------------------------
    # First plateau: highest rolling median before the fork region.
    pre = step_a < 40.0
    ia = int(np.argmax(np.where(pre, med_a, -np.inf)))
    ax.annotate("first plateau %.0f at %.1fB" % (med_a[ia], step_a[ia]),
                xy=(step_a[ia], med_a[ia]), xytext=(step_a[ia] - 3.0, med_a[ia] + 1150),
                color=TEXT, fontsize=7.8, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="->", color=C_SIM, linewidth=0.9,
                                shrinkA=1, shrinkB=3))

    # Start of the run: fixed initial rating.
    ax.text(2.2, ylo + 0.048 * span, "the league opens every learner at 1000",
            color=TEXT, fontsize=7.8, ha="left", va="center")

    # Deepest post-peak trough on branch A.
    tail = np.arange(ia, len(med_a))
    it = int(tail[np.argmin(med_a[tail])])
    ax.annotate("pool turnover drags the rating to %.0f\nwhile training continues uninterrupted"
                % med_a[it],
                xy=(step_a[it], med_a[it]),
                xytext=(step_a[it] + 9.0, ylo + 0.20 * span),
                color=TEXT, fontsize=7.8, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=C_SIM, linewidth=0.9,
                                shrinkA=1, shrinkB=3))

    ax.annotate("branch A ends %.0f at %.1fB" % (med_a[-1], step_a[-1]),
                xy=(step_a[-1], med_a[-1]),
                xytext=(step_a[-1], ylo + 0.295 * span),
                color=TEXT, fontsize=7.8, ha="right", va="center",
                arrowprops=dict(arrowstyle="->", color=C_SIM, linewidth=0.9,
                                shrinkA=1, shrinkB=3))

    if n_b:
        ib = int(np.argmax(med_b))
        a_here = float(np.interp(step_b[ib], step_a, med_a))
        print("branch A median at branch B peak step (%.2fB): %.0f  -> gap %.0f"
              % (step_b[ib], a_here, med_b[ib] - a_here))
        ax.annotate("branch B peaks at %.0f — %.0f points above branch A\n"
                    "at the same step count, on the same task"
                    % (med_b[ib], med_b[ib] - a_here),
                    xy=(step_b[ib], med_b[ib]),
                    xytext=(step_b[ib] + 3.0, yhi - 0.072 * span),
                    color=TEXT, fontsize=7.8, ha="left", va="top",
                    arrowprops=dict(arrowstyle="->", color=C_PERCEPTION,
                                    linewidth=0.9, shrinkA=1, shrinkB=3))

    ax.set_ylabel("league-internal Elo\n(rating points)", color=TEXT, fontsize=9)
    ax.tick_params(labelbottom=False)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.635, 0.805),
                    frameon=False, fontsize=8.0, labelcolor=TEXT,
                    handlelength=2.0, borderaxespad=0.0)
    for t in leg.get_texts():
        t.set_color(TEXT)

    # -- pool size ---------------------------------------------------------
    px, py = compress_step(step_a, pool_a)
    axp.step(px, py, where="post", color=C_POLICY, linewidth=1.4, zorder=4)
    if n_b:
        qx, qy = compress_step(step_b, pool_b)
        axp.step(qx, qy, where="post", color=C_POLICY, linewidth=1.4,
                 alpha=0.45, zorder=3)
    print("vertices drawn          : elo band %d+%d, medians %d+%d, pool %d+%d"
          % (len(ex), len(bx) if n_b else 0, len(mx), len(nx) if n_b else 0,
             len(px), len(qx) if n_b else 0))
    axp.set_ylim(-1.0, 27.0)
    axp.set_ylabel("league pool\n(opponents)", color=TEXT, fontsize=9)
    axp.set_xlabel("cumulative environment steps (billions)", color=TEXT, fontsize=9)

    # plateau annotations, placed inside the panel under the trace
    i20 = int(np.argmax(pool_a >= 20))
    i24 = int(np.argmax(pool_a >= 24)) if (pool_a >= 24).any() else None
    axp.annotate("20 frozen self-copies from %.1fB" % step_a[i20],
                 xy=(16.0, 20), xytext=(16.0, 8.0),
                 color=TEXT, fontsize=7.6, ha="center", va="center",
                 arrowprops=dict(arrowstyle="->", color=C_POLICY, linewidth=0.9,
                                 shrinkA=3, shrinkB=2))
    if i24 is not None:
        axp.annotate("+4 scripted opponents at %.2fB" % step_a[i24],
                     xy=(step_a[i24] + 1.0, 24), xytext=(step_a[i24] + 8.0, 8.0),
                     color=TEXT, fontsize=7.6, ha="left", va="center",
                     arrowprops=dict(arrowstyle="->", color=C_POLICY, linewidth=0.9,
                                     shrinkA=3, shrinkB=2))
    axp.set_yticks([0, 20, 24])
    axp.tick_params(axis="y", labelsize=7.5)

    ax.set_xlim(-1.5, max(step_a[-1], step_b[-1] if n_b else 0) * 1.015)

    # ----------------------------------------------------------------------
    # Mandatory caption block + footer stamp, as figure text
    # ----------------------------------------------------------------------
    caption = (
        "This Elo is LEAGUE-INTERNAL — a rating against a self-replacing pool of the "
        "learner's own past selves, which inflates as the pool turns over.\n"
        "It is NOT comparable to the fixed-ladder Elo of 1710 in reports/. "
        "Do not put the two on one axis."
    )
    fig.text(0.062, 0.225, caption, color=TEXT, fontsize=8.6, ha="left", va="top",
             linespacing=1.5)

    stamp = ("runs/fov1/metrics.csv (%d raw rows, deduped to %d + %d) + train.log "
             "(%d resumes), rendered %s"
             % (n_raw, n_a, n_b, n_resume,
                datetime.date.today().isoformat()))
    fig.text(0.062, 0.085, stamp, color=MUTED, fontsize=7.4, ha="left", va="bottom",
             family="monospace")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, format="svg", transparent=True, bbox_inches="tight")
    # Optional raster copy for eyeballing the layout; never written into the
    # repo unless PVPBOT_FIG_PNG names a path outside it.
    png = os.environ.get("PVPBOT_FIG_PNG")
    if png:
        fig.savefig(png, format="png", dpi=110, bbox_inches="tight",
                    facecolor="#ffffff")
        print("wrote %s (debug raster)" % png)
    plt.close(fig)
    print("wrote %s (%d bytes)" % (OUT, os.path.getsize(OUT)))


if __name__ == "__main__":
    main()
