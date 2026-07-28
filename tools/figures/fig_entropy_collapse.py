#!/usr/bin/env python3
"""Render docs/assets/per-head-entropy-collapse.svg -- the per-head policy
entropy dumbbell.

WHAT IT PLOTS
    One row per categorical action head in ``pvpbot.spec.ACTION_HEADS``. Each
    row is a dumbbell: a hollow marker at the head's entropy in the FIRST
    record of the training metrics stream, a filled marker at its entropy in
    the LAST record, joined by a line. The x axis is entropy expressed as a
    share of that head's own theoretical maximum, so heads with different
    cardinalities (2, 3 and 11 bins) are comparable on one scale.

    The maxima come from ``ACTION_HEAD_SIZES`` in ``pvpbot/spec.py``:
    ln(3) for forward and strafe, ln(2) for jump/sprint/attack, ln(11) for yaw
    and pitch. ``pvpbot/train/ppo.py:84`` computes the logged per-head entropy
    with ``log_softmax``, so the units are nats and a uniform head sits at
    exactly ln(n).

    Rows are sorted by the FINAL share, ascending, which puts the two camera
    heads at the top.

DATA
    runs/fov1/metrics.jsonl -- the per-update PPO metrics stream. The
    ``entropy_forward`` / ``_strafe`` / ``_jump`` / ``_sprint`` / ``_attack`` /
    ``_yaw`` / ``_pitch`` keys exist only in the JSONL; metrics.csv drops them.
    This path is gitignored, so a fresh clone will not have it and this script
    will exit with an error rather than inventing values. The committed SVG is
    the only copy of the result.

    The file is opened in append mode by every training run, so it grows and is
    not a fixed-length artifact. The record count is printed on stdout and
    stamped into the figure at render time; do not quote it from prose.

WHAT IS AND IS NOT DONE TO THE DATA
    Nothing is smoothed, binned or downsampled: the dumbbell reads exactly two
    records, the first and the last that parse as JSON. A live training run may
    be mid-write on the final line, so an unparseable trailing line is skipped
    and the last complete record is used instead.

    The whole file is still scanned, for two reasons: to count records for the
    stamp, and to report the branch structure on stdout. The stream contains
    one hard fork -- a run resumed from an earlier checkpoint with a different
    entropy-coefficient setting and kept appending to the same file -- plus
    many short backward jumps where a run restarted from its own last
    checkpoint. ``split_branches`` reports both. No entropy-versus-step curve
    is drawn here: the two branches end at different entropies, so a single
    curve panel next to a first-record/last-record dumbbell would contradict
    it. The branch summary is printed instead, for whoever writes that figure.

NOT PLOTTED, DELIBERATELY
    ``mean_ep_reward`` and ``win_rate_pool`` are in the same file and are not
    drawn anywhere. The reward function changed during the run and the
    opponent sampler is Elo-matched, so both series move for reasons that have
    nothing to do with policy quality.

RUN
    python3 tools/figures/fig_entropy_collapse.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "runs" / "fov1" / "metrics.jsonl"
OUT = REPO / "docs" / "assets" / "per-head-entropy-collapse.svg"

# pvpbot/spec.py ACTION_HEADS -- name and cardinality, in head order.
ACTION_HEADS: Tuple[Tuple[str, int], ...] = (
    ("forward", 3),
    ("strafe", 3),
    ("jump", 2),
    ("sprint", 2),
    ("attack", 2),
    ("yaw", 11),
    ("pitch", 11),
)

# Palette: the seven-hex docs palette. Only two of them are used here.
POLICY = "#5b3fa8"   # the learner
MID = "#8b8b8b"      # neutral mid-tone, legible on light and dark
INK = "#4a4a4a"      # body text

# A backward jump in `update` larger than this is a fork onto a different
# lineage; anything smaller is a run restarting from its own last checkpoint.
FORK_UPDATE_DROP = 50


def load_records(path: Path) -> List[Dict]:
    """Parse every complete JSON line. A live run may be mid-write on the
    last line, so unparseable lines are counted and skipped, not fatal."""
    if not path.exists():
        sys.exit(
            "ERROR: {} not found.\n"
            "This figure is rendered from the PPO metrics stream of the fov1\n"
            "training run. runs/ is gitignored, so a fresh clone does not have\n"
            "it and this figure cannot be regenerated without re-training.\n"
            "The committed SVG at {} is the only copy.".format(
                path.relative_to(REPO) if path.is_absolute() else path,
                OUT.relative_to(REPO),
            )
        )
    records: List[Dict] = []
    skipped = 0
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
    if skipped:
        print("  note: skipped {} unparseable line(s) "
              "(file is being appended to live)".format(skipped))
    if len(records) < 2:
        sys.exit("ERROR: {} holds {} usable record(s); need at least 2."
                 .format(path, len(records)))
    return records


def require_entropy_keys(rec: Dict, which: str) -> None:
    missing = [n for n, _ in ACTION_HEADS if "entropy_" + n not in rec]
    if missing:
        sys.exit(
            "ERROR: the {} record of {} has no per-head entropy keys {}.\n"
            "Only metrics.jsonl carries them; metrics.csv does not."
            .format(which, DATA, ["entropy_" + m for m in missing])
        )


def split_branches(records: List[Dict]) -> List[Tuple[int, int]]:
    """Return [(start, stop), ...] row-index spans, one per lineage.

    A new span starts wherever `update` falls by more than FORK_UPDATE_DROP,
    which is a resume from a much earlier checkpoint rather than a restart.
    """
    bounds = [0]
    for i in range(1, len(records)):
        if records[i]["update"] < records[i - 1]["update"] - FORK_UPDATE_DROP:
            bounds.append(i)
    bounds.append(len(records))
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def dedupe_by_update(rows: List[Dict]) -> List[Dict]:
    """Keep the last row per `update` within one branch, in update order.

    Short restarts replay a handful of updates; the replayed rows are the ones
    that actually happened, so the later write wins.
    """
    last: Dict[int, Dict] = {}
    for r in rows:
        last[r["update"]] = r
    return [last[u] for u in sorted(last)]


def report_branches(records: List[Dict]) -> None:
    spans = split_branches(records)
    print("  branches (fork = `update` drop of more than {}):"
          .format(FORK_UPDATE_DROP))
    for k, (a, b) in enumerate(spans):
        rows = dedupe_by_update(records[a:b])
        first, last = rows[0], rows[-1]
        print("    branch {}: rows[{}:{}]  {} raw -> {} after dedupe  "
              "update {}..{}  step {:.2f}B..{:.2f}B  "
              "entropy_yaw {:.4f} -> {:.4f} nats".format(
                  chr(ord("A") + k), a, b, b - a, len(rows),
                  first["update"], last["update"],
                  first["step"] / 1e9, last["step"] / 1e9,
                  first["entropy_yaw"], last["entropy_yaw"]))


def render(records: List[Dict],
           out: Path) -> Dict[str, Tuple[float, float, float, float]]:
    """Draw the dumbbell. Returns {head: (nats_first, nats_last, %_first,
    %_last)} in plotted order, for the stdout sanity print."""
    first, last = records[0], records[-1]
    require_entropy_keys(first, "first")
    require_entropy_keys(last, "last")

    rows = []
    for name, card in ACTION_HEADS:
        hmax = math.log(card)
        e0 = float(first["entropy_" + name])
        e1 = float(last["entropy_" + name])
        rows.append({
            "name": name, "card": card, "hmax": hmax,
            "e0": e0, "e1": e1,
            "p0": 100.0 * e0 / hmax, "p1": 100.0 * e1 / hmax,
        })
    # Sorted by the FINAL share, ascending -- the camera heads land on top.
    rows.sort(key=lambda r: r["p1"])
    n = len(rows)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    for i, r in enumerate(rows):
        y = n - 1 - i  # index 0 (smallest final share) at the top
        r["y"] = y
        ax.plot([r["p1"], r["p0"]], [y, y],
                color=MID, alpha=0.5, linewidth=2.0, solid_capstyle="round",
                zorder=2)
        ax.plot([r["p0"]], [y], marker="o", markersize=8.5,
                markerfacecolor="none", markeredgecolor=MID,
                markeredgewidth=1.5, linestyle="none", zorder=3)
        ax.plot([r["p1"]], [y], marker="o", markersize=8.5,
                color=POLICY, linestyle="none", zorder=4)

    ax.set_xlim(0, 105)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_yticks([r["y"] for r in rows])
    ax.set_yticklabels(["{} ({})".format(r["name"], r["card"]) for r in rows])
    ax.set_xlabel(
        "policy entropy, % of that head's own uniform maximum "
        "(ln 2 = 0.693, ln 3 = 1.099, ln 11 = 2.398 nats)",
        fontsize=8.5, color=INK, labelpad=7)

    ax.grid(axis="x", color=MID, alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MID)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="x", colors=MID, labelsize=8.5, length=3)
    ax.tick_params(axis="y", length=0, labelsize=9.5)
    for lbl in ax.get_xticklabels():
        lbl.set_color(INK)
    for lbl in ax.get_yticklabels():
        lbl.set_color(INK)

    # Raw nats, one per row, in a column just outside the axes on the right.
    right = blended_transform_factory(ax.transAxes, ax.transData)
    for r in rows:
        ax.text(1.025, r["y"], "{:.3f} → {:.3f} nats".format(r["e0"], r["e1"]),
                transform=right, ha="left", va="center",
                fontsize=8.5, color=INK, family="monospace")

    # Leader on the yaw row. The text sits in the empty lower-left triangle
    # (no row's final marker is left of ~45%) and the leader climbs the empty
    # left margin to the yaw marker without crossing any dumbbell.
    yaw = next(r for r in rows if r["name"] == "yaw")
    lead_x = yaw["p1"]
    ax.plot([lead_x, lead_x], [yaw["y"] - 0.38, 1.92],
            color=MID, alpha=0.7, linewidth=0.9, zorder=1)
    ax.text(2.0, 1.02,
            "the camera heads collapse hardest — which is\n"
            "why the deploy loop overrides them with an\n"
            "explicit aim assist (docs/05)",
            fontsize=8.5, color=INK, ha="left", va="center",
            linespacing=1.45, zorder=5)

    handles = [
        Line2D([], [], marker="o", markersize=8.5, linestyle="none",
               markerfacecolor="none", markeredgecolor=MID,
               markeredgewidth=1.5, label="first record in the file"),
        Line2D([], [], marker="o", markersize=8.5, linestyle="none",
               color=POLICY, label="last record in the file"),
    ]
    leg = ax.legend(handles=handles, loc="lower center",
                    bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False,
                    handletextpad=0.4, columnspacing=2.0, fontsize=9,
                    borderpad=0.1)
    for txt in leg.get_texts():
        txt.set_color(INK)

    stamp = "runs/fov1/metrics.jsonl, {} records, rendered {}".format(
        len(records), time.strftime("%Y-%m-%d"))
    fig.text(0.0, -0.245, stamp, transform=ax.transAxes,
             fontsize=7, color=MID, ha="left", va="top", family="monospace")

    out.parent.mkdir(parents=True, exist_ok=True)
    # The committed artifact is SVG; a .png path renders a raster preview for
    # eyeballing the layout and is not committed.
    fig.savefig(out, format=(out.suffix.lstrip(".") or "svg"),
                transparent=True, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return {r["name"]: (r["e0"], r["e1"], r["p0"], r["p1"]) for r in rows}


def main() -> None:
    print("reading {}".format(DATA))
    records = load_records(DATA)
    print("  {} records".format(len(records)))
    print("  first record: update {}, step {:.3f}B".format(
        records[0]["update"], records[0]["step"] / 1e9))
    print("  last  record: update {}, step {:.3f}B".format(
        records[-1]["update"], records[-1]["step"] / 1e9))
    report_branches(records)

    vals = render(records, OUT)
    print("plotted, sorted by final share ascending:")
    for name, (e0, e1, p0, p1) in vals.items():
        print("    {:<8} {:.4f} -> {:.4f} nats   {:.1f}% -> {:.1f}% of max"
              .format(name, e0, e1, p0, p1))
    print("wrote {} ({} bytes)".format(OUT, OUT.stat().st_size))


if __name__ == "__main__":
    main()
