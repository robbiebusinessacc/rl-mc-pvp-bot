#!/usr/bin/env python3
"""Figure: obs-provenance-strip.

Two stacked 48-cell strips over one shared index axis, answering "what does the
sim know, and what can a camera actually recover?".

  TOP strip    SIM (privileged) -- DuelVecEnv writes slots 0-31 from ground
               truth; 32-47 are the reserved tail and are never filled.
  BOTTOM strip LIVE (through a camera) -- each slot coloured by how the live
               ObsAssembler produced it: MEASURED (straight from a CNN output),
               DERIVED (computed from a measurement), DEAD-RECKONED (integrated
               from the bot's own issued action, no sensor at all), or DEAD.

DATA SOURCE -- this figure is GENERATED FROM SOURCE, never hand-drawn, so that
an edit to the observation contract or to the adapter breaks the build instead
of quietly producing a wrong picture:

  * pvpbot/spec.py                  imported directly (``from pvpbot.spec
                                    import OBS_LAYOUT``) for the 48-float slot
                                    boundaries and their names.
  * pvpbot/perception/adapter.py    parsed as text for its ``obs[...] = ...``
                                    write sites; each site is matched back to
                                    the ``s, e = OBS_LAYOUT["<name>"]`` line
                                    that precedes it.

No gitignored data is read: both inputs are committed source files, so a fresh
clone can regenerate this figure. Nothing is smoothed, sampled or averaged --
every cell is one observation index.

BUILD-TIME ASSERTIONS (all fatal):
  1. the adapter parse finds exactly 19 write sites;
  2. those 19 names are exactly the OBS_LAYOUT keys minus "reserved";
  3. the hardcoded provenance mapping covers every OBS_LAYOUT key;
  4. the expanded per-index classes cover 0..47 with no gap and no overlap;
  5. the class counts are exactly MEASURED 6 / DERIVED 12 / DEAD-RECKONED 14 /
     DEAD 16, and the six MEASURED indices are {11, 15, 18, 22, 23, 31}.

Run from the repo root:

    python3 tools/figures/fig_obs_provenance_strip.py

Writes docs/assets/obs-provenance-strip.svg (transparent, bbox_inches='tight').
Set PVPBOT_FIG_PNG=<path> to also drop a flattened raster proof for eyeballing,
and PVPBOT_FIG_BG=#0d1117 to check that proof against GitHub's dark canvas.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Paths. The two "data files" are committed source, not run artifacts.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "pvpbot" / "spec.py"
ADAPTER_PATH = REPO_ROOT / "pvpbot" / "perception" / "adapter.py"
OUT_PATH = REPO_ROOT / "docs" / "assets" / "obs-provenance-strip.svg"

for _p in (SPEC_PATH, ADAPTER_PATH):
    if not _p.is_file():
        raise SystemExit(
            "fig_obs_provenance_strip: missing source file {}\n"
            "This figure is generated from pvpbot/spec.py and "
            "pvpbot/perception/adapter.py. Run it from a full checkout of the "
            "repo:  python3 tools/figures/fig_obs_provenance_strip.py".format(_p)
        )

sys.path.insert(0, str(REPO_ROOT))
from pvpbot.spec import OBS_DIM, OBS_LAYOUT  # noqa: E402

# ---------------------------------------------------------------------------
# Palette -- the seven-hex docset palette, nothing else.
# ---------------------------------------------------------------------------
C_SIM = "#1f6f4a"          # sim / ground truth
C_MEASURED = "#1d5c93"     # live (deploy path) blue
C_DERIVED = "#a8621b"      # perception orange
C_RECKONED = "#5b3fa8"     # policy purple (the bot's own issued action)
C_DEAD = "#d0d0d0"         # permanently zero, never consumed
INK = "#4a4a4a"            # darker neutral; only used on a light fill, since
                           # #4a4a4a on GitHub dark (#0d1117) is 2.1:1 contrast
GREY = "#8b8b8b"           # mid-tone text / rules, legible on both themes

CLASS_COLOR = {
    "MEASURED": C_MEASURED,
    "DERIVED": C_DERIVED,
    "DEAD-RECKONED": C_RECKONED,
    "DEAD": C_DEAD,
}
EXPECTED_COUNTS = {"MEASURED": 6, "DERIVED": 12, "DEAD-RECKONED": 14, "DEAD": 16}
EXPECTED_MEASURED_IDX = {11, 15, 18, 22, 23, 31}
EXPECTED_WRITE_SITES = 19

# ---------------------------------------------------------------------------
# Live provenance of every OBS_LAYOUT slot, one entry per key.
#
# Each comment names the adapter write site that produces the value. Line
# numbers are as read on 2026-07-27; the parser below reports the LIVE line
# numbers at build time, so drift shows up in stdout rather than silently.
# ---------------------------------------------------------------------------
PROVENANCE = {
    # -- DERIVED: computed from a measurement, not read off the screen ------
    "rel_pos": "DERIVED",              # adapter.py:299  obs[s:e] = self.rel_pos / 8.0
                                       #   trig on aim angles + dist_est (l.246-251)
    "rel_vel": "DERIVED",              # adapter.py:301  obs[s:e] = self.rel_vel
                                       #   EMA finite difference of rel_pos (l.254)
    "in_reach": "DERIVED",             # adapter.py:310  visible and dist_est < _REACH
    "self_hurt": "DERIVED",            # adapter.py:312  timer off a >=3 hp median drop (l.287)
    "enemy_hurt": "DERIVED",           # adapter.py:314  timer off hurt_flash rising edge (l.275)
    "enemy_hp": "DERIVED",             # adapter.py:318  -1 hp per flash, regen (l.277,283)
    "ticks_since_hit_dealt": "DERIVED",   # adapter.py:326  counter off enemy_hurt edge
    "ticks_since_hit_taken": "DERIVED",   # adapter.py:328  counter off self_hurt trigger
    # -- MEASURED: written straight from a PerceptionCNN output -------------
    "dist": "MEASURED",                # adapter.py:308  dist_from_bbox_height(P.bbox_height)
    "self_hp": "MEASURED",             # adapter.py:316  P.self_hp, median-filtered
    "enemy_on_ground": "MEASURED",     # adapter.py:322  p[_P["enemy_on_ground"]] > 0.5
    "aim_err_yaw": "MEASURED",         # adapter.py:330  P.aim_err_yaw, direct
    "aim_err_pitch": "MEASURED",       # adapter.py:332  P.aim_err_pitch, sign-flipped
    "enemy_visible": "MEASURED",       # adapter.py:334  p[_P["visible"]] > 0.5
    # -- DEAD-RECKONED: integrated from the bot's own issued action ---------
    "self_vel": "DEAD-RECKONED",       # adapter.py:303  stub-physics integrator (l.157-158)
    "self_pitch_sincos": "DEAD-RECKONED",  # adapter.py:306  issued pitch bins, perception
                                       #   correction gated off above 40 deg (l.217)
    "self_on_ground": "DEAD-RECKONED",  # adapter.py:320  jump/gravity integrator (l.164)
    "self_sprinting": "DEAD-RECKONED",  # adapter.py:324  sprint head of issued action (l.155)
    "prev_action": "DEAD-RECKONED",    # adapter.py:337  issued action indices / head sizes
    # -- DEAD: never written after the zero-fill ----------------------------
    "reserved": "DEAD",                # adapter.py:296  obs = np.zeros(OBS_DIM); stays zero
}

# Bands bracketed above the top strip. Label placement is measured at draw
# time: letterspaced if it fits the band, plain if that fits, otherwise lifted
# to a second row with a leader line (which is what VISIBLE, one cell wide,
# always needs).
BANDS = [
    ("GEOMETRY", 0, 12),
    ("COMBAT STATE", 12, 22),
    ("AIM", 22, 24),
    ("MOTOR MEMORY", 24, 31),
    ("VISIBLE", 31, 32),
    ("RESERVED", 32, 48),
]

# The six MEASURED slots, named in the gap between the strips so the headline
# claim is checkable from the picture alone. row 0 sits just above the LIVE
# strip, row 1 above that; the split keeps the names from colliding.
MEASURED_CALLOUTS = [
    ((11,), "dist", 0),
    ((15,), "self_hp", 1),
    ((18,), "enemy_on_ground", 0),
    ((22, 23), "aim_err_yaw/pitch", 1),
    ((31,), "enemy_visible", 0),
]


# ---------------------------------------------------------------------------
# Parse the adapter's write sites.
# ---------------------------------------------------------------------------
_SLOT_RE = re.compile(r'^\s*s,\s*e\s*=\s*OBS_LAYOUT\["(?P<name>\w+)"\]')
_WRITE_RE = re.compile(r"^\s*obs\[(?P<sl>s:e|s)\]\s*=")


def parse_write_sites(path: Path):
    """-> list of (obs_layout_key, line_number, 'slice'|'scalar')."""
    sites = []
    pending = None
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        m = _SLOT_RE.match(line)
        if m:
            pending = m.group("name")
            continue
        w = _WRITE_RE.match(line)
        if w:
            if pending is None:
                raise SystemExit(
                    "fig_obs_provenance_strip: {}:{} writes obs[] with no "
                    "preceding `s, e = OBS_LAYOUT[...]` binding; the parser "
                    "cannot attribute it.".format(path.name, lineno)
                )
            sites.append((pending, lineno, "slice" if w.group("sl") == "s:e" else "scalar"))
    return sites


def build_index_classes(sites):
    """Expand PROVENANCE over OBS_LAYOUT and check every invariant."""
    if len(sites) != EXPECTED_WRITE_SITES:
        raise SystemExit(
            "fig_obs_provenance_strip: parsed {} obs[] write sites in {}, "
            "expected {}. The adapter changed -- update PROVENANCE and the "
            "expected counts, then re-render.".format(
                len(sites), ADAPTER_PATH.name, EXPECTED_WRITE_SITES)
        )

    written = [n for n, _, _ in sites]
    if len(set(written)) != len(written):
        raise SystemExit(
            "fig_obs_provenance_strip: an OBS_LAYOUT slot is written twice: "
            "{}".format(sorted(n for n in set(written) if written.count(n) > 1))
        )
    expected_written = set(OBS_LAYOUT) - {"reserved"}
    if set(written) != expected_written:
        raise SystemExit(
            "fig_obs_provenance_strip: adapter write sites do not match the "
            "contract. missing={} unexpected={}".format(
                sorted(expected_written - set(written)),
                sorted(set(written) - expected_written))
        )
    if set(PROVENANCE) != set(OBS_LAYOUT):
        raise SystemExit(
            "fig_obs_provenance_strip: PROVENANCE does not cover OBS_LAYOUT. "
            "missing={} unexpected={}".format(
                sorted(set(OBS_LAYOUT) - set(PROVENANCE)),
                sorted(set(PROVENANCE) - set(OBS_LAYOUT)))
        )

    idx_class = {}
    for name, (start, stop) in OBS_LAYOUT.items():
        for i in range(start, stop):
            if i in idx_class:
                raise SystemExit(
                    "fig_obs_provenance_strip: OBS_LAYOUT slots overlap at "
                    "index {}".format(i)
                )
            idx_class[i] = PROVENANCE[name]
    if set(idx_class) != set(range(OBS_DIM)):
        raise SystemExit(
            "fig_obs_provenance_strip: classification covers {} indices, not "
            "0..{}. gaps={}".format(
                len(idx_class), OBS_DIM - 1,
                sorted(set(range(OBS_DIM)) - set(idx_class)))
        )

    counts = {k: 0 for k in EXPECTED_COUNTS}
    for cls in idx_class.values():
        counts[cls] += 1
    if counts != EXPECTED_COUNTS:
        raise SystemExit(
            "fig_obs_provenance_strip: provenance class counts changed: got "
            "{}, expected {}. An adapter edit moved a slot between classes -- "
            "fix the figure before shipping it.".format(counts, EXPECTED_COUNTS)
        )
    measured = {i for i, c in idx_class.items() if c == "MEASURED"}
    if measured != EXPECTED_MEASURED_IDX:
        raise SystemExit(
            "fig_obs_provenance_strip: MEASURED indices are {}, expected "
            "{}".format(sorted(measured), sorted(EXPECTED_MEASURED_IDX))
        )
    return idx_class, counts


# ---------------------------------------------------------------------------
# Geometry of the drawing (data coordinates; 1 x-unit == 1 observation slot).
# ---------------------------------------------------------------------------
CELL_PAD = 0.09           # horizontal gap each side of a cell
Y_TOP_B, Y_TOP_T = 1.30, 1.82
Y_BOT_B, Y_BOT_T = 0.34, 0.86
X_LEFT = -10.6            # room for the two strip labels
Y_BAND0 = 1.92            # bracket rule above the top strip
Y_BAND_ROW = (0.09, 0.32)  # label offsets above the bracket rule
Y_CALLOUT = (0.90, 1.06)  # measured-slot names, in the gap between strips
Y_TICKS = 0.20            # index numbers, below the bottom strip
Y_LEGEND = -0.30
Y_CAP1 = -0.68
Y_CAP2 = -0.92


def _sp(text):
    """Poor-man's letterspacing (matplotlib 3.9 Text has no letterspacing)."""
    return " ".join(text)


def _data_width(fig, ax, artist):
    """Rendered width of a text artist, in observation-index units."""
    fig.canvas.draw()
    bb = artist.get_window_extent(renderer=fig.canvas.get_renderer())
    inv = ax.transData.inverted()
    return abs(inv.transform((bb.x1, bb.y0))[0] - inv.transform((bb.x0, bb.y0))[0])


def cell(ax, i, y0, y1, color, hatched=False):
    if hatched:
        ax.add_patch(mpatches.Rectangle(
            (i + CELL_PAD, y0), 1 - 2 * CELL_PAD, y1 - y0,
            facecolor=color, edgecolor=GREY, linewidth=0.5,
            hatch="////", zorder=2))
    else:
        ax.add_patch(mpatches.Rectangle(
            (i + CELL_PAD, y0), 1 - 2 * CELL_PAD, y1 - y0,
            facecolor=color, edgecolor="none", linewidth=0, zorder=2))


def render(idx_class, counts, date_str):
    plt.rcParams["hatch.linewidth"] = 0.5
    fig, ax = plt.subplots(figsize=(11.0, 3.2))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.set_xlim(X_LEFT, OBS_DIM + 0.4)
    ax.set_ylim(Y_CAP2 - 0.22, Y_BAND0 + 0.46)
    ax.set_axis_off()

    # --- band boundary hairlines, broken across the callout gap so they do
    #     not strike through the measured-slot names -------------------------
    for x in (12, 22, 24, 31, 32):
        for y0, y1 in ((Y_BOT_B - 0.07, Y_BOT_T + 0.03),
                       (Y_TOP_B - 0.03, Y_BAND0 - 0.02)):
            ax.plot([x, x], [y0, y1], color=GREY, lw=0.6, alpha=0.28, zorder=1)

    # --- cells --------------------------------------------------------------
    for i in range(OBS_DIM):
        dead = idx_class[i] == "DEAD"
        # TOP: sim ground truth everywhere the contract is defined
        cell(ax, i, Y_TOP_B, Y_TOP_T, C_DEAD if dead else C_SIM, hatched=dead)
        # BOTTOM: coloured by live provenance
        cell(ax, i, Y_BOT_B, Y_BOT_T, CLASS_COLOR[idx_class[i]], hatched=dead)

    # --- strip labels, left gutter -----------------------------------------
    def strip_label(y0, y1, title, sub):
        yc = 0.5 * (y0 + y1)
        ax.text(-0.7, yc + 0.10, title, ha="right", va="center",
                fontsize=9.6, color=GREY, fontweight="bold")
        ax.text(-0.7, yc - 0.14, sub, ha="right", va="center",
                fontsize=7.0, color=GREY, family="DejaVu Sans Mono")

    strip_label(Y_TOP_B, Y_TOP_T, "SIM (privileged)", "pvpbot/sim/env.py")
    strip_label(Y_BOT_B, Y_BOT_T, "LIVE (through a camera)",
                "pvpbot/perception/adapter.py")

    # --- GROUND TRUTH annotation over the green run ------------------------
    ax.text(16.0, 0.5 * (Y_TOP_B + Y_TOP_T), _sp("GROUND TRUTH"),
            ha="center", va="center", fontsize=7.6, color="#ffffff",
            fontweight="bold", zorder=4,
            bbox=dict(facecolor=C_SIM, edgecolor="none", pad=2.2))
    ax.text(40.0, 0.5 * (Y_TOP_B + Y_TOP_T), _sp("never filled"),
            ha="center", va="center", fontsize=6.8, color=INK, zorder=4,
            bbox=dict(facecolor=C_DEAD, edgecolor="none", pad=2.0))

    # --- band brackets above the top strip ---------------------------------
    y = Y_BAND0
    for name, start, stop in BANDS:
        ax.plot([start + CELL_PAD, stop - CELL_PAD], [y, y],
                color=GREY, lw=0.7, alpha=0.85)
        for x in (start + CELL_PAD, stop - CELL_PAD):
            ax.plot([x, x], [y, y - 0.055], color=GREY, lw=0.7, alpha=0.85)
        xc = 0.5 * (start + stop)
        room = (stop - start) - 0.4
        t = ax.text(xc, y + Y_BAND_ROW[0], _sp(name), ha="center",
                    va="bottom", fontsize=7.0, color=GREY, fontweight="bold")
        if _data_width(fig, ax, t) > room:          # letterspaced too wide
            t.set_text(name)
            if _data_width(fig, ax, t) > room:      # plain still too wide
                t.set_y(y + Y_BAND_ROW[1])
                ax.plot([xc, xc], [y + 0.02, y + Y_BAND_ROW[1] - 0.04],
                        color=GREY, lw=0.6, alpha=0.6)

    # --- names of the six MEASURED slots, in the inter-strip gap -----------
    for idxs, label, row in MEASURED_CALLOUTS:
        xc = sum(i + 0.5 for i in idxs) / len(idxs)
        ylab = Y_CALLOUT[row]
        ax.text(xc, ylab, label, ha="center", va="bottom", fontsize=6.9,
                color=GREY, family="DejaVu Sans Mono")
        for i in idxs:
            ax.plot([i + 0.5, i + 0.5], [Y_BOT_T + 0.012, ylab - 0.015],
                    color=GREY, lw=0.7, alpha=0.75)

    # --- index ruler --------------------------------------------------------
    for i in range(0, OBS_DIM, 4):
        ax.plot([i + 0.5, i + 0.5], [Y_BOT_B - 0.015, Y_BOT_B - 0.065],
                color=GREY, lw=0.7, alpha=0.8)
        ax.text(i + 0.5, Y_TICKS - 0.055, str(i), ha="center", va="top",
                fontsize=7.2, color=GREY)
    ax.text(OBS_DIM + 0.3, Y_TICKS - 0.055, "obs index", ha="left", va="top",
            fontsize=7.2, color=GREY, style="italic")

    # --- legend -------------------------------------------------------------
    order = ["MEASURED", "DERIVED", "DEAD-RECKONED", "DEAD"]
    x = 0.0
    for cls in order:
        hatched = cls == "DEAD"
        ax.add_patch(mpatches.Rectangle(
            (x, Y_LEGEND), 1.5, 0.26, facecolor=CLASS_COLOR[cls],
            edgecolor=GREY if hatched else "none",
            linewidth=0.5 if hatched else 0,
            hatch="////" if hatched else None))
        ax.text(x + 2.0, Y_LEGEND + 0.13,
                "{} — {}".format(cls, counts[cls]),
                ha="left", va="center", fontsize=8.0, color=GREY)
        x += 12.0

    # --- caption + provenance ----------------------------------------------
    ax.text(0.0, Y_CAP1,
            "Same dtype, same length, same policy weights. Six of the 48 "
            "numbers are things the bot can actually see.",
            ha="left", va="center", fontsize=8.4, color=GREY)
    ax.text(0.0, Y_CAP2,
            "generated from pvpbot/spec.py + pvpbot/perception/adapter.py, "
            "{}".format(date_str),
            ha="left", va="center", fontsize=7.4, color=GREY,
            family="DejaVu Sans Mono")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, format="svg", transparent=True, bbox_inches="tight")
    if os.environ.get("PVPBOT_FIG_PNG"):
        png = os.environ["PVPBOT_FIG_PNG"]
        if png == "1":
            png = str(OUT_PATH.with_suffix(".png"))
        # PVPBOT_FIG_BG lets a proof be flattened onto GitHub's dark canvas
        # (#0d1117) to check that nothing disappears in either theme.
        bg = os.environ.get("PVPBOT_FIG_BG", "white")
        fig.patch.set_alpha(1.0)
        fig.patch.set_facecolor(bg)
        fig.savefig(png, format="png", dpi=200, transparent=False,
                    facecolor=bg, bbox_inches="tight")
        print("wrote proof {}".format(png))
    plt.close(fig)


def main():
    sites = parse_write_sites(ADAPTER_PATH)
    idx_class, counts = build_index_classes(sites)
    date_str = _dt.date.today().isoformat()
    render(idx_class, counts, date_str)

    print("adapter write sites parsed: {}".format(len(sites)))
    for name, lineno, kind in sites:
        start, stop = OBS_LAYOUT[name]
        print("  adapter.py:{:<4} {:<22} obs[{}:{}]  {:<6}  {}".format(
            lineno, name, start, stop, kind, PROVENANCE[name]))
    print("class counts: {}".format(
        {k: counts[k] for k in ("MEASURED", "DERIVED", "DEAD-RECKONED", "DEAD")}))
    print("MEASURED indices: {}".format(
        sorted(i for i, c in idx_class.items() if c == "MEASURED")))
    print("row of 48 classes: {}".format(
        "".join({"MEASURED": "M", "DERIVED": "D", "DEAD-RECKONED": "R",
                 "DEAD": "."}[idx_class[i]] for i in range(OBS_DIM))))
    print("wrote {} ({} bytes)".format(OUT_PATH, OUT_PATH.stat().st_size))


if __name__ == "__main__":
    main()
