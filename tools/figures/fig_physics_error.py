#!/usr/bin/env python3
"""Figure: sim-vs-real-physics-error.

Renders docs/assets/sim-vs-real-physics-error.svg: how far the NumPy duel sim
(pvpbot/sim/env.py) drifts from a real PaperSpigot 1.8.8 server replaying the
same open-loop input schedule.

Unlike every other figure in tools/figures/, this one reads NOTHING from the
gitignored runs/ or reports/ trees. Both of its inputs are committed:

    tools/validation/schedules/*.json      the input scripts
    tools/validation/recordings/*.jsonl    the ground truth captured by
                                           tools/validation/recorder/record_duel.js

so it re-renders from a fresh clone with no Minecraft client, no server and no
network. The replay is seeded and deterministic, so every rendered path is
identical run to run; only matplotlib's own SVG metadata date and its random
clip-path ids change between runs.

MAIN PANEL - per-tick absolute position error, in millimetres, for both bots
over the 420-tick `basic` course. Computed in this process: harness.py rolls
schedules/basic.json through pvpbot.sim.env.DuelVecEnv, the recording rows are
converted from the Minecraft frame to the sim frame by harness.mc_pos_to_sim,
and the error is the Euclidean norm of the difference at every tick. 420 ticks
is short enough to plot raw: NO smoothing, NO downsampling, every tick drawn.
The curve is a staircase because error is injected at acceleration events and
then held, not accumulated continuously.

INSET - combined position RMSE, in blocks, for all eight committed recordings,
from compare.run_compare(..., env_kind="real"), the same code path as
`python3 tools/validation/compare.py`. Three of the eight sit above one block;
they are the combat replays, and the cause is documented ground truth
(tools/validation/RUNBOOK.md: mineflayer does not model client-side player
pushing, so its bots phase through each other).

Run standalone:
    python3 tools/figures/fig_physics_error.py
or import render() from tools/figures/make_figures.py.
"""
import datetime
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

# --- paths -----------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))  # tools/figures -> repo root

DATA_RECORDINGS = os.path.join(REPO_ROOT, "tools", "validation", "recordings")
DATA_SCHEDULES = os.path.join(REPO_ROOT, "tools", "validation", "schedules")
VALIDATION_DIR = os.path.join(REPO_ROOT, "tools", "validation")

MAIN_RECORDING = os.path.join(DATA_RECORDINGS, "basic_1.8.8.jsonl")
MAIN_SCHEDULE = os.path.join(DATA_SCHEDULES, "basic.json")

OUT_PATH = os.path.join(REPO_ROOT, "docs", "assets", "sim-vs-real-physics-error.svg")

# --- palette (the seven repo hexes; nothing else) ---------------------------
C_SIM = "#1f6f4a"       # sim / physics / ground truth
C_POLICY = "#5b3fa8"    # the second trace
C_DEGRADED = "#9b1c31"  # known-failure annotation
C_TEXT = "#4a4a4a"
C_MUTED = "#8b8b8b"

BLOCK_TO_MM = 1000.0  # one Minecraft block is one metre, so blocks -> mm is x1000
DEGRADED_ABOVE_BLOCKS = 1.0


def _require(path, what):
    if not os.path.exists(path):
        raise SystemExit(
            "missing %s: %s\n"
            "This figure reads only committed files under tools/validation/; if "
            "the path above does not exist the clone is incomplete. Nothing in "
            "runs/ or reports/ is needed." % (what, path)
        )


def _import_validation():
    """Import tools/validation/{harness,compare}.py with the repo on sys.path."""
    for p in (REPO_ROOT, VALIDATION_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    import compare  # noqa: E402
    import harness  # noqa: E402

    return harness, compare


# --- data ------------------------------------------------------------------

def per_tick_error_mm(harness):
    """Per-tick |sim - real| position error in mm for the `basic` course.

    Returns (err_mm[T, 2], bot_ids, segment_bands, num_rows_read).
    segment_bands is the ordered list of bot-A segments as
    (tick_start, tick_end, label).
    """
    sched = harness.load_schedule(MAIN_SCHEDULE)
    collision = bool(sched.get("meta", {}).get("sim_collision", True))
    env, used_kind = harness.make_env("real", seed=0, collision=collision)
    if used_kind != "real":
        raise SystemExit("pvpbot.sim.env did not import; refusing to plot the stub")
    trace = harness.roll_schedule(env, sched)

    rec_rows, _hurt = harness.read_recording(MAIN_RECORDING)
    ground_y = sched["arena"]["ground_y"]
    bots = sched["bots"]
    T = trace["num_ticks"]

    err = np.full((T, len(bots)), np.nan, dtype=np.float64)
    for side, bot in enumerate(bots):
        for t in range(T):
            row = rec_rows.get((bot, t))
            if row is None:
                continue
            real_pos = harness.mc_pos_to_sim(row["pos"], ground_y)
            err[t, side] = np.linalg.norm(real_pos - trace["pos"][t, side])

    # Band label: the segment's mechanic, unless the mechanic name does not
    # appear in the segment id (jump_arc -> jump_launch / jump_air, coast ->
    # disengage), in which case the id is the more specific name.
    bands = []
    for seg in sched["segments"]:
        if seg["bot"] != bots[0]:
            continue
        label = seg["mechanic"] if seg["mechanic"] in seg["id"] else seg["id"]
        bands.append((seg["tick_start"], seg["tick_end"], label))
    bands.sort()

    return err * BLOCK_TO_MM, bots, bands, len(rec_rows)


def all_recording_rmse(compare):
    """Combined pos RMSE (blocks) for every committed recording.

    Each recording's meta line names the schedule it replays, so the pairing is
    read from the data rather than guessed from the filename.
    """
    rows = []
    for fn in sorted(os.listdir(DATA_RECORDINGS)):
        if not fn.endswith(".jsonl"):
            continue
        rec_path = os.path.join(DATA_RECORDINGS, fn)
        with open(rec_path, "r") as f:
            meta = json.loads(f.readline())
        if meta.get("event") != "meta" or "schedule" not in meta:
            raise SystemExit("%s has no meta line naming its schedule" % rec_path)
        sched_path = os.path.join(DATA_SCHEDULES, meta["schedule"] + ".json")
        _require(sched_path, "schedule for %s" % fn)
        rep = compare.run_compare(rec_path, sched_path, env_kind="real")
        if rep["env"] != "real":
            raise SystemExit("compare fell back to the stub; refusing to plot")
        rows.append(
            {
                "name": fn[: -len(".jsonl")],
                "schedule": meta["schedule"],
                "rmse": float(rep["combined_pos_rmse"]),
                "ticks": int(rep["num_ticks_scheduled"]),
                "rows": int(rep["num_recording_rows"]),
            }
        )
    rows.sort(key=lambda r: r["rmse"])
    return rows


# --- figure ----------------------------------------------------------------

def _style_axes(ax):
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=C_MUTED, labelcolor=C_TEXT, width=0.8, length=3)


def render(out_path=OUT_PATH, verbose=True):
    _require(MAIN_RECORDING, "ground-truth recording")
    _require(MAIN_SCHEDULE, "input schedule")
    _require(DATA_RECORDINGS, "recordings directory")
    harness, compare = _import_validation()

    err_mm, bots, bands, n_rows = per_tick_error_mm(harness)
    ladder = all_recording_rmse(compare)

    T = err_mm.shape[0]
    ticks = np.arange(T)
    worst_flat = int(np.nanargmax(err_mm))
    worst_t, worst_side = np.unravel_index(worst_flat, err_mm.shape)
    worst_mm = float(err_mm[worst_t, worst_side])

    fig = plt.figure(figsize=(11.0, 4.2))
    fig.patch.set_alpha(0.0)

    ax = fig.add_axes([0.048, 0.180, 0.560, 0.655])
    _style_axes(ax)

    # alternating segment bands + rotated mechanic labels along the top
    xy_trans = blended_transform_factory(ax.transData, ax.transAxes)
    for i, (t0, t1, label) in enumerate(bands):
        if i % 2 == 0:
            ax.axvspan(t0, t1, color=C_MUTED, alpha=0.08, lw=0, zorder=0)
        ax.text(
            0.5 * (t0 + t1), 1.02, label, transform=xy_trans, rotation=90,
            ha="center", va="bottom", fontsize=7, color=C_MUTED,
        )

    ax.plot(ticks, err_mm[:, 0], color=C_SIM, lw=1.4, zorder=4,
            label="bot A  ·  runs all thirteen segments")
    ax.plot(ticks, err_mm[:, 1], color=C_POLICY, lw=1.4, zorder=3,
            label="bot B  ·  walks in, then holds still")

    # the top of the curve is a plateau, so quote its spread as well as the
    # argmax tick -- otherwise "@ tick 305" reads as a sharp event
    plateau = err_mm[:, worst_side] >= 0.99 * worst_mm
    p_ticks = ticks[plateau]
    p_spread = float(np.nanmax(err_mm[plateau, worst_side])
                     - np.nanmin(err_mm[plateau, worst_side]))

    ax.plot([worst_t], [worst_mm], marker="o", ms=5.5, mfc="none",
            mec=C_SIM, mew=1.2, zorder=5)
    ax.annotate(
        "max %.1f mm @ tick %d\nflat within %.2f mm from tick %d"
        % (worst_mm, worst_t, p_spread, int(p_ticks[0])),
        xy=(worst_t, worst_mm), xytext=(worst_t - 22, 12.3),
        ha="right", va="top", fontsize=7.5, color=C_TEXT, linespacing=1.4,
        arrowprops=dict(arrowstyle="-", color=C_MUTED, lw=0.8,
                        shrinkA=1, shrinkB=4),
    )

    ax.set_xlim(0, T)
    ax.set_ylim(0, 16)
    ax.set_xticks([0, 60, 120, 180, 240, 300, 360, 420])
    ax.set_yticks([0, 4, 8, 12, 16])
    ax.set_xlabel("game tick  (one tick = 50 ms)", fontsize=8.5, color=C_TEXT)
    ax.set_ylabel("position error (millimetres)", fontsize=8.5, color=C_TEXT)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color=C_MUTED, alpha=0.22, lw=0.7)
    ax.set_axisbelow(True)

    leg = ax.legend(loc="upper left", frameon=False, fontsize=7.5,
                    handlelength=1.6, borderpad=0.1, labelspacing=0.35)
    for txt in leg.get_texts():
        txt.set_color(C_TEXT)

    # --- corner note (upper right of the figure) ---------------------------
    fig.text(
        0.665, 0.955,
        "1 block = 1000 mm; the whole course stays under 15 mm",
        fontsize=8.5, color=C_TEXT, ha="left", va="top", weight="bold",
    )
    fig.text(
        0.665, 0.885,
        "Both bots replay a fixed keystroke script open-loop for %d ticks\n"
        "(%.1f s) with no feedback, so nothing corrects an error once made.\n"
        "The steps are acceleration events; between them the error holds flat."
        % (T, T / 20.0),
        fontsize=7.2, color=C_MUTED, ha="left", va="top", linespacing=1.5,
    )

    # --- inset: every committed recording ----------------------------------
    axi = fig.add_axes([0.775, 0.300, 0.200, 0.415])
    _style_axes(axi)
    axi.set_xscale("log")

    names = [r["name"] for r in ladder]
    vals = [r["rmse"] for r in ladder]
    ypos = np.arange(len(vals))[::-1]  # smallest at the top
    colors = [C_DEGRADED if v > DEGRADED_ABOVE_BLOCKS else C_SIM for v in vals]

    axi.axvspan(DEGRADED_ABOVE_BLOCKS, 90.0, color=C_MUTED, alpha=0.09, lw=0)
    axi.barh(ypos, vals, height=0.62, color=colors, lw=0, zorder=2)
    axi.axvline(DEGRADED_ABOVE_BLOCKS, color=C_MUTED, lw=0.8, alpha=0.55,
                ls=(0, (3, 2)), zorder=1)
    axi.text(DEGRADED_ABOVE_BLOCKS * 1.25, len(vals) - 0.62,
             "over 1 block", fontsize=6.6, color=C_MUTED, ha="left", va="center")

    for y, v in zip(ypos, vals):
        txt = "%.4f" % v if v < 0.01 else ("%.3f" % v if v < 1 else "%.1f" % v)
        axi.text(v * 1.18, y, txt, fontsize=6.8, color=C_TEXT,
                 ha="left", va="center", zorder=3)

    axi.set_yticks(ypos)
    axi.set_yticklabels(names, fontsize=6.8, color=C_TEXT)
    axi.set_ylim(-0.7, len(vals) - 0.3)
    axi.set_xlim(0.004, 90.0)
    axi.set_xticks([0.01, 0.1, 1.0, 10.0])
    axi.set_xticklabels(["0.01", "0.1", "1", "10"], fontsize=6.8)
    axi.tick_params(axis="y", length=0)
    axi.set_xlabel("combined position RMSE (blocks, log scale)",
                   fontsize=7.4, color=C_TEXT, labelpad=2)
    axi.grid(axis="x", color=C_MUTED, alpha=0.22, lw=0.7)
    axi.set_axisbelow(True)

    n_bad = sum(1 for v in vals if v > DEGRADED_ABOVE_BLOCKS)
    worst_ok = max(v for v in vals if v <= DEGRADED_ABOVE_BLOCKS)
    fig.text(
        0.665, 0.760,
        "All %d recordings in tools/validation/recordings/" % len(vals),
        fontsize=7.6, color=C_TEXT, ha="left", va="top", weight="bold",
    )
    fig.text(
        0.665, 0.180,
        "The %d red bars are open-loop combat replay desyncs: mineflayer bots\n"
        "phase through each other, so the sim separates them and the recording\n"
        "does not (tools/validation/RUNBOOK.md). The other %d stay at or under\n"
        "%.3f blocks." % (n_bad, len(vals) - n_bad, worst_ok),
        fontsize=7.2, color=C_MUTED, ha="left", va="top", linespacing=1.5,
    )

    # --- footer stamp ------------------------------------------------------
    stamp_date = datetime.date.today().isoformat()
    fig.text(
        0.048, 0.058,
        "tools/validation/recordings/ (committed) replayed through "
        "pvpbot/sim/env.py, rendered %s" % stamp_date,
        fontsize=6.5, color=C_MUTED, ha="left", va="bottom",
    )
    fig.text(
        0.048, 0.012,
        "main panel: basic_1.8.8.jsonl vs schedules/basic.json, %d rows over "
        "%d ticks  ·  inset: compare.py combined pos RMSE, %d recordings  ·  "
        "no gitignored data, reproducible from a fresh clone"
        % (n_rows, T, len(vals)),
        fontsize=6.5, color=C_MUTED, ha="left", va="bottom",
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # format follows the extension so the same call can emit a PNG proof sheet
    fig.savefig(out_path, transparent=True, bbox_inches="tight", dpi=150)
    plt.close(fig)

    if verbose:
        print("wrote %s" % out_path)
        print("main panel: %s ticks, %d recording rows" % (T, n_rows))
        for side, bot in enumerate(bots):
            col = err_mm[:, side]
            print(
                "  bot %s: max %.3f mm @ tick %d | mean %.3f mm | final %.3f mm"
                % (bot, np.nanmax(col), int(np.nanargmax(col)),
                   np.nanmean(col), col[-1])
            )
        print("inset: combined pos RMSE (blocks)")
        for r in ladder:
            print("  %-22s %-13s %9.4f  (%d ticks)"
                  % (r["name"], r["schedule"], r["rmse"], r["ticks"]))
    return out_path


if __name__ == "__main__":
    render()
