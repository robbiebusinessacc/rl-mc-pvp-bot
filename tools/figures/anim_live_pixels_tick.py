#!/usr/bin/env python3
"""live-pixels-tick.gif -- the live loop, one tick at a time: pixels in, keystrokes out.

Renders 56 consecutive LIVE ticks of the deployed loop: the real Minecraft 1.8
frame the loop captured, the box the loop's own perception estimate projects
back onto those pixels, the seven categorical action heads it injected, and the
four-stage latency breakdown of that same tick against the 50 ms budget.

Everything drawn is read straight off the flight recorder written by
pvpbot/deploy/loop.py -- nothing is re-simulated and no model is re-run here.

PROVENANCE, EXACTLY
  * The recorded action is the FINAL INJECTED action. In loop.py the chain is
    policy -> humanize -> click-discipline gate -> aim-assist override of heads
    5 and 6 -> crit-assist override of heads 2 and 4, and only then is it
    recorded. Every label in this figure says "commanded", never "chose".
  * The recorded percep is the 12-float estimate the deployed PerceptionCNN
    produced from that frame. It is not server ground truth and the live
    checkpoint is not named: these values come from the flight recorder.
  * The cyan box is that estimate projected back onto the pixels, not a label
    and not an annotation of where the enemy really was.

SOURCE (gitignored capture output; not present in a fresh clone)
  frames_191133/a0/frames.npy   (3375, 96, 170, 3) uint8, one per settled tick
  cal_flight_191133.jsonl       4000 rows, 3375 with settled=1
  Pairing rule: s = [r for r in rows if r["settled"]]; s[i] <-> frames[i].

Usage:  python3 tools/figures/anim_live_pixels_tick.py
        python3 tools/figures/anim_live_pixels_tick.py --frames-npy P --log P
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from pvpbot.spec import ACTION_HEADS, CAMERA_BINS, PERCEPTION_LAYOUT  # noqa: E402
from pvpbot.perception.synth import dist_from_bbox_height  # noqa: E402

# --------------------------------------------------------------------------
# Paths. The source clip is gitignored; these are the locations it was
# recorded to. Override with --frames-npy / --log.
# --------------------------------------------------------------------------
_SCRATCH_CANDIDATES = (
    os.path.join(REPO, "runs", "live", "191133"),
)
FRAMES_NAME = os.path.join("frames_191133", "a0", "frames.npy")
LOG_NAME = "cal_flight_191133.jsonl"
OUT_GIF = os.path.join(REPO, "docs", "assets", "live-pixels-tick.gif")
OUT_JSON = os.path.join(REPO, "docs", "assets", "data", "live-pixels-tick.json")

# --------------------------------------------------------------------------
# Clip. Settled-tick index 270 is log tick 326 -- the tick traced field by
# field in the README and in docs/01-interfaces.md.
# --------------------------------------------------------------------------
CLIP_START = 270
# 52 ticks, not 56: at 4x the real capture costs ~40 KB/frame and 56 frames
# measured 2.48 MB against a 2.5 MB ceiling. Frames come off the END of the
# clip, never out of the middle -- single-tick attack flags and hurt flashes
# vanish silently if you stride-drop.
CLIP_N = 52

# --------------------------------------------------------------------------
# Locked palette -- identical hexes to tools/figures/anim_live_cockpit.py.
# Colour roles are semantic and never reused:
#   CYAN   what the machine produced (perception estimate, sensor stage)
#   AMBER  what exists independently of the machine (the screen-capture stage)
#   RED    damage events
#   GREEN  the policy stage of the tick budget
#   VIOLET input injection and actuation (the commanded action, inject stage)
#   DIM    a held or stale value
# --------------------------------------------------------------------------
BG = (0x12, 0x16, 0x1C)
PANEL = (0x1A, 0x21, 0x2A)
GRID = (0x2B, 0x35, 0x42)
INK = (0xCC, 0xD6, 0xE2)
DIM = (0x66, 0x74, 0x7F)
CYAN = (0x3F, 0xD0, 0xD8)
AMBER = (0xE8, 0xA3, 0x3D)
RED = (0xE2, 0x56, 0x4A)
GREEN = (0x6E, 0xCF, 0x94)
VIOLET = (0xA1, 0x84, 0xE8)
BRAND = [BG, PANEL, GRID, INK, DIM, CYAN, AMBER, RED, GREEN, VIOLET]

STAGE_COL = {"capture": AMBER, "encode": CYAN, "policy": GREEN, "inject": VIOLET}
STAGES = ("capture", "encode", "policy", "inject")


def mix(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


CELL_OFF = mix(BG, GRID, 0.55)      # unlit action cell
CELL_HOLD = mix(BG, DIM, 0.30)      # neutral / no-op cell
VIOLET_DK = mix(BG, VIOLET, 0.35)
CYAN_DK = mix(BG, CYAN, 0.30)
LINE_CYAN = mix(BG, CYAN, 0.75)

# --------------------------------------------------------------------------
# Camera model, measured against these exact captures.
#   principal point (84.5, 50.0) in the 170x96 frame -- the game's own
#   crosshair; row 50 not 48 because the capture includes the macOS title bar.
#   focal 42.0 px  => ~127 deg horizontal / ~98 deg vertical FOV.
# Self-test at import: aim_err_pitch = 0 must land exactly on the crosshair row
# for any self_pitch. A sign-flipped up-vector passes at level pitch only.
# --------------------------------------------------------------------------
CX0, CY0, FOCAL = 84.5, 50.0, 42.0
FRAME_W, FRAME_H = 170, 96
UPSCALE = 4
IMG_W, IMG_H = FRAME_W * UPSCALE, FRAME_H * UPSCALE   # 680 x 384


def project(yaw_n: float, pitch_n: float, self_pitch_n: float):
    """Perception angles -> pixel in the native 170x96 frame."""
    a = math.radians(yaw_n * 180.0)
    b = math.radians(pitch_n * 90.0)
    p = math.radians(self_pitch_n * 90.0)
    pb = p + b
    den = math.cos(pb) * math.cos(p) * math.cos(a) + math.sin(pb) * math.sin(p)
    if abs(den) < 1e-6:
        den = 1e-6 if den >= 0 else -1e-6
    u = FOCAL * math.cos(pb) * math.sin(a) / den
    v = FOCAL * (math.sin(pb) * math.cos(p) - math.cos(pb) * math.sin(p) * math.cos(a)) / den
    return CX0 + u, CY0 + v


for _p in (-0.9, -0.3, 0.0, 0.25, 0.8):
    assert abs(project(0.0, 0.0, _p)[1] - CY0) < 1e-9, "projection up-vector sign error"

# --------------------------------------------------------------------------
# Canvas geometry. Explicit pixel rects only: nothing may move between frames.
# --------------------------------------------------------------------------
W, H = 1000, 534
HDR_H = 30
BAND_Y0, BAND_Y1 = 30, 416          # image band (386 px)
IMG_X0 = (W - IMG_W) // 2           # 160
IMG_Y0 = BAND_Y0 + 1                # 31
HEADS_Y0, HEADS_Y1 = 416, 466
BUD_Y0, BUD_Y1 = 466, 510
FTR_Y0 = 510

FONT_PATH = "/System/Library/Fonts/Menlo.ttc"


def _font(size, bold=False):
    try:
        return ImageFont.truetype(FONT_PATH, size, index=1 if bold else 0)
    except OSError:
        return ImageFont.load_default()


F_HDR = _font(13, True)
F_HDR_S = _font(10)
F_T = _font(10, True)
F_B = _font(9)
F_S = _font(9)
F_XS = _font(8)
F_VAL = _font(11, True)
F_BIG = _font(14, True)


def txt(d, xy, s, font, fill, anchor="la"):
    d.text(xy, s, font=font, fill=fill, anchor=anchor)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def _find(paths, name):
    for base in paths:
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return None


def load(args):
    fp = args.frames_npy or _find(_SCRATCH_CANDIDATES, FRAMES_NAME)
    lp = args.log or _find(_SCRATCH_CANDIDATES, LOG_NAME)
    missing = []
    if not fp or not os.path.exists(fp):
        missing.append("frames .npy (" + FRAMES_NAME + ")")
    if not lp or not os.path.exists(lp):
        missing.append("flight log (" + LOG_NAME + ")")
    if missing:
        sys.stderr.write(
            "source clip no longer on disk: " + ", ".join(missing) + "\n"
            "  This clip is gitignored live-capture data from session B and is not\n"
            "  part of a fresh clone. Searched:\n    "
            + "\n    ".join(_SCRATCH_CANDIDATES)
            + "\n  Pass --frames-npy and --log to point at a relocated copy.\n"
            "  The committed docs/assets/live-pixels-tick.gif and its sidecar\n"
            "  docs/assets/data/live-pixels-tick.json remain valid.\n"
        )
        raise SystemExit(2)

    rows = [json.loads(l) for l in open(lp) if l.strip()]
    settled = [r for r in rows if r.get("settled")]
    frames = np.load(fp, mmap_mode="r")
    if len(settled) != frames.shape[0]:
        sys.stderr.write(
            "settled ticks (%d) do not pair 1:1 with dumped frames (%d); "
            "wrong session?\n" % (len(settled), frames.shape[0])
        )
        raise SystemExit(2)
    return rows, settled, frames


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------
def draw_header(d, r, n_settled, tick_secs):
    d.rectangle([0, 0, W - 1, HDR_H - 1], fill=PANEL)
    d.line([0, HDR_H - 1, W - 1, HDR_H - 1], fill=GRID)
    x = 10
    d.rectangle([x, 8, x + 46, 22], fill=CYAN_DK, outline=CYAN)
    txt(d, (x + 23, 15), "0-4", F_T, CYAN, "mm")
    txt(d, (x + 56, 15), "LIVE", F_HDR, INK, "lm")
    txt(d, (x + 96, 16), "screen -> CNN -> GRU policy -> mouse + keys"
        "   ·   20 fps, 1 GIF frame = 1 live tick", F_HDR_S, DIM, "lm")
    txt(d, (W - 10, 15),
        "tick %04d · t=%.2f s · %s settled ticks / %.1f s"
        % (r["tick"], r["t"], format(n_settled, ","), tick_secs),
        F_HDR_S, INK, "rm")


def draw_left_rail(d):
    """Static input-side spec. Costs nothing per frame: it never changes."""
    x = 10
    y = 40

    def stage(num, name, col):
        nonlocal y
        d.rectangle([x, y, x + 16, y + 14], fill=mix(BG, col, 0.30), outline=col)
        txt(d, (x + 8, y + 7), num, F_T, col, "mm")
        txt(d, (x + 23, y + 7), name, F_T, INK, "lm")
        y += 21

    stage("0", "SCREEN", AMBER)
    for line in ("uint8[3,96,170]", "48,960 bytes", "20 Hz screen grab",
                 "drawn 4x nearest"):
        txt(d, (x + 3, y), line, F_S, DIM)
        y += 12
    y += 10
    stage("1", "SENSOR", CYAN)
    for line in ("PerceptionCNN", "3,500,204 params", "-> float32[12]"):
        txt(d, (x + 3, y), line, F_S, DIM)
        y += 12

    y += 16
    d.line([x, y, x + 138, y], fill=GRID)
    y += 10
    txt(d, (x, y), "ON THE FRAME", F_T, INK)
    y += 17
    d.rectangle([x, y - 1, x + 13, y + 8], outline=CYAN)
    txt(d, (x + 19, y + 4), "cyan box + line", F_S, CYAN, "lm")
    y += 15
    for line in ("the loop's own estimate of",
                 "the enemy, projected back",
                 "onto the pixels it read.",
                 "Not a label: this is what",
                 "the deployed loop believed."):
        txt(d, (x, y), line, F_S, DIM)
        y += 12

    y += 11
    d.rectangle([x, y - 1, x + 13, y + 8], outline=RED, fill=mix(BG, RED, 0.35))
    txt(d, (x + 19, y + 4), "red border", F_S, RED, "lm")
    y += 15
    for line in ("percep hurt_flash > 0.5:",
                 "the enemy's damage flash,",
                 "read off the pixels",
                 "(adapter.py:198)"):
        txt(d, (x, y), line, F_S, DIM)
        y += 12


def draw_frame_panel(d, per, hitflash):
    """Paste the real capture and draw the loop's own belief onto it."""
    x0, y0 = IMG_X0, IMG_Y0
    d.rectangle([x0 - 1, y0 - 1, x0 + IMG_W, y0 + IMG_H], outline=GRID)

    vis = per[PERCEPTION_LAYOUT["visible"]] > 0.5
    if not vis:
        return
    px, py = project(per[PERCEPTION_LAYOUT["aim_err_yaw"]],
                     per[PERCEPTION_LAYOUT["aim_err_pitch"]],
                     per[PERCEPTION_LAYOUT["self_pitch"]])
    bh = float(per[PERCEPTION_LAYOUT["bbox_height"]]) * FRAME_H
    bw = 0.36 * bh
    cx, cy = x0 + px * UPSCALE, y0 + py * UPSCALE
    hx, hy = x0 + CX0 * UPSCALE, y0 + CY0 * UPSCALE

    # cyan line, crosshair -> belief centre (the live aim error, drawn)
    d.line([hx, hy, cx, cy], fill=LINE_CYAN, width=1)
    d.ellipse([hx - 2, hy - 2, hx + 2, hy + 2], outline=LINE_CYAN)

    l = cx - bw * UPSCALE / 2.0
    r_ = cx + bw * UPSCALE / 2.0
    t = cy - bh * UPSCALE / 2.0
    b = cy + bh * UPSCALE / 2.0
    # clip to the image rect so the box can never bleed into the chrome
    l2, r2 = max(l, x0), min(r_, x0 + IMG_W - 1)
    t2, b2 = max(t, y0), min(b, y0 + IMG_H - 1)
    if r2 > l2 and b2 > t2:
        d.rectangle([l2, t2, r2, b2], outline=CYAN, width=2)
        # corner ticks so the box survives quantisation over busy pixels
        k = 9
        for (ax, ay, dx, dy) in ((l2, t2, 1, 1), (r2, t2, -1, 1),
                                 (l2, b2, 1, -1), (r2, b2, -1, -1)):
            d.line([ax, ay, ax + dx * k, ay], fill=CYAN, width=3)
            d.line([ax, ay, ax, ay + dy * k], fill=CYAN, width=3)

    if hitflash:
        for i in range(3):
            d.rectangle([x0 + i, y0 + i, x0 + IMG_W - 1 - i, y0 + IMG_H - 1 - i],
                        outline=RED)


def draw_right_rail(d, per, hitflash, track_atk, track_hit, i):
    x = 848
    y = 38
    txt(d, (x, y), "PERCEP  float32[12]", F_T, INK)
    y += 13
    txt(d, (x, y), "from the flight recorder", F_S, DIM)
    y += 17

    P = PERCEPTION_LAYOUT
    bbox = float(per[P["bbox_height"]])
    vis = per[P["visible"]] > 0.5
    grn = per[P["enemy_on_ground"]] > 0.5
    rows = [
        ("aim_err_yaw", per[P["aim_err_yaw"]],
         "%+.2f deg" % (per[P["aim_err_yaw"]] * 180.0), CYAN),
        ("aim_err_pitch", per[P["aim_err_pitch"]],
         "%+.2f deg" % (per[P["aim_err_pitch"]] * 90.0), CYAN),
        ("bbox_height", bbox, "%.2f blocks" % dist_from_bbox_height(bbox), CYAN),
        ("visible", per[P["visible"]], "on screen" if vis else "not seen",
         CYAN if vis else DIM),
        ("self_hp  (hearts HUD)", per[P["self_hp"]],
         "%.1f hp" % (per[P["self_hp"]] * 20.0), CYAN),
        ("hurt_flash", per[P["hurt_flash"]],
         "ENEMY HIT" if hitflash else "under 0.5", RED if hitflash else DIM),
        ("enemy_on_ground", per[P["enemy_on_ground"]],
         "grounded" if grn else "airborne", CYAN if grn else DIM),
    ]
    for name, raw, unit, col in rows:
        txt(d, (x, y), name, F_S, DIM)
        y += 12
        txt(d, (x + 3, y), "%+.5f" % raw, F_VAL, col)
        txt(d, (x + 70, y + 2), "= " + unit, F_S, mix(BG, col, 0.75))
        y += 16

    y += 6
    d.line([x, y, x + 144, y], fill=GRID)
    y += 9
    for num, name, col, sub in (("2", "ADAPTER", CYAN, "-> float32[48]"),
                                ("3", "CONTROLLER", CYAN, "GRU -> 34 logits")):
        d.rectangle([x, y, x + 16, y + 14], fill=mix(BG, col, 0.30), outline=col)
        txt(d, (x + 8, y + 7), num, F_T, col, "mm")
        txt(d, (x + 23, y + 7), name, F_T, INK, "lm")
        txt(d, (x + 3, y + 17), sub, F_S, DIM)
        y += 32

    # raster of the whole clip, with a playhead: click cadence and hits
    n = len(track_atk)
    y += 3
    txt(d, (x, y), "THIS CLIP  %d ticks" % n, F_XS, DIM)
    y += 12
    d.rectangle([x, y + 1, x + 7, y + 8], fill=VIOLET)
    txt(d, (x + 11, y), "attack", F_XS, mix(BG, VIOLET, 0.9))
    d.rectangle([x + 56, y + 1, x + 63, y + 8], fill=RED)
    txt(d, (x + 67, y), "hurt_flash", F_XS, mix(BG, RED, 0.9))
    y += 13
    cw = 144.0 / n
    y_top = y
    for track, col in ((track_atk, VIOLET), (track_hit, RED)):
        d.rectangle([x, y, x + 144, y + 9], fill=mix(BG, GRID, 0.55))
        for k, v in enumerate(track):
            if v:
                d.rectangle([x + k * cw, y, x + (k + 1) * cw - 0.6, y + 9], fill=col)
        y += 12
    px = x + (i + 0.5) * cw
    d.line([px, y_top - 3, px, y - 1], fill=INK)


# --- action heads ---------------------------------------------------------
CELL_W, CELL_G = 24, 2
GROUP_G = 12
_GROUPS = [
    ("FORWARD", 0, ["BACK", "·", "FWD"], 1),
    ("STRAFE", 1, ["LEFT", "·", "RGHT"], 1),
    ("JUMP", 2, ["·", "JUMP"], 0),
    ("SPRINT", 3, ["·", "SPRT"], 0),
    ("ATTACK", 4, ["·", "ATK"], 0),
]
_CAM_LAB = ["-30", "-15", "-7", "-3", "-1", "0", "+1", "+3", "+7", "+15", "+30"]


def _group_w(n):
    return n * CELL_W + (n - 1) * CELL_G


def heads_layout():
    widths = [_group_w(len(g[2])) for g in _GROUPS] + [_group_w(11), _group_w(11)]
    total = sum(widths) + GROUP_G * (len(widths) - 1) + 14   # +14 extra before YAW
    x = (W - total) // 2
    xs = []
    for i, w in enumerate(widths):
        xs.append(x)
        x += w + GROUP_G + (14 if i == len(_GROUPS) - 1 else 0)
    return xs


_HX = heads_layout()


def draw_heads(d, act):
    d.rectangle([0, HEADS_Y0, W - 1, HEADS_Y1 - 1], fill=BG)
    d.line([0, HEADS_Y0, W - 1, HEADS_Y0], fill=GRID)
    cy0 = HEADS_Y0 + 13
    cy1 = cy0 + 22

    hx = _HX[0]
    d.rectangle([hx, HEADS_Y0 + 3, hx + 15, HEADS_Y0 + 12],
                fill=mix(BG, VIOLET, 0.30), outline=VIOLET)
    txt(d, (hx + 8, HEADS_Y0 + 8), "4", F_XS, VIOLET, "mm")
    txt(d, (hx + 21, HEADS_Y0 + 8), "INJECTION", F_XS, INK, "lm")
    txt(d, (hx + 84, HEADS_Y0 + 8),
        "the action the bot commanded this tick  ·  7 categorical heads  ·  34 logits",
        F_XS, DIM, "lm")

    def cell(cx, lab, lit, neutral):
        if lit and not neutral:
            fill, out, tc = VIOLET, VIOLET, BG
        elif lit:
            fill, out, tc = CELL_HOLD, DIM, INK
        else:
            fill, out, tc = BG, CELL_OFF, mix(BG, DIM, 0.62)
        d.rectangle([cx, cy0, cx + CELL_W, cy1], fill=fill, outline=out)
        txt(d, (cx + CELL_W / 2, (cy0 + cy1) / 2 + 1), lab, F_XS, tc, "mm")

    for gi, (name, head, labels, neutral) in enumerate(_GROUPS):
        x = _HX[gi]
        sel = int(act[head])
        for ci, lab in enumerate(labels):
            cell(x + ci * (CELL_W + CELL_G), lab, ci == sel, ci == neutral)
        txt(d, (x + _group_w(len(labels)) / 2, cy1 + 4), name, F_S,
            VIOLET if sel != neutral else DIM, "ma")

    for li, (head, name) in enumerate(((5, "YAW"), (6, "PITCH"))):
        x = _HX[len(_GROUPS) + li]
        sel = int(act[head])
        for ci in range(11):
            cell(x + ci * (CELL_W + CELL_G), _CAM_LAB[ci], ci == sel, ci == 5)
        gw = _group_w(11)
        lit = sel != 5
        txt(d, (x + gw / 2, cy1 + 4),
            "%s  %+g deg/tick" % (name, CAMERA_BINS[sel]), F_S,
            VIOLET if lit else DIM, "ma")


# --- tick budget ----------------------------------------------------------
BAR_X0, BAR_X1 = 130, 826
BAR_MS = 50.0
BAR_Y0, BAR_Y1 = BUD_Y0 + 14, BUD_Y0 + 30


def draw_budget(d, lat):
    d.rectangle([0, BUD_Y0, W - 1, BUD_Y1 - 1], fill=BG)
    d.line([0, BUD_Y0, W - 1, BUD_Y0], fill=GRID)
    ppm = (BAR_X1 - BAR_X0) / BAR_MS

    txt(d, (10, BUD_Y0 + 3), "TICK BUDGET", F_T, INK)
    txt(d, (10, BAR_Y0 + 5), "one 50 ms tick", F_XS, DIM)

    # dashed 50 ms box -- the empty part of it is the message
    for x in range(BAR_X0, BAR_X1, 6):
        d.line([x, BAR_Y0, min(x + 3, BAR_X1), BAR_Y0], fill=GRID)
        d.line([x, BAR_Y1, min(x + 3, BAR_X1), BAR_Y1], fill=GRID)
    d.line([BAR_X0, BAR_Y0, BAR_X0, BAR_Y1], fill=GRID)
    d.line([BAR_X1, BAR_Y0, BAR_X1, BAR_Y1], fill=GRID)
    for m in (0, 10, 20, 30, 40):
        gx = BAR_X0 + m * ppm
        d.line([gx, BAR_Y1 + 1, gx, BAR_Y1 + 3], fill=GRID)
        txt(d, (gx, BAR_Y1 + 3), "%d" % m, F_XS, DIM, "ma")
    txt(d, (BAR_X1, BAR_Y1 + 3), "50", F_XS, DIM, "ma")
    txt(d, (BAR_X1 - 26, BAR_Y0 + 4), "ms", F_XS, DIM, "lm")

    # true-to-scale segments; 1 px separators so the sub-millisecond policy and
    # inject slivers stay countable next to a 20 ms capture
    x = float(BAR_X0)
    edges = []
    for st in STAGES:
        w = float(lat[st]) * ppm
        d.rectangle([x, BAR_Y0 + 1, max(x + w, x + 1), BAR_Y1 - 1], fill=STAGE_COL[st])
        x += w
        edges.append(x)
    for e in edges[:-1]:
        d.line([e, BAR_Y0 + 1, e, BAR_Y1 - 1], fill=BG)
    total = sum(float(lat[s]) for s in STAGES)
    d.line([x, BAR_Y0 - 3, x, BAR_Y1 + 3], fill=INK)

    # legend with this tick's values, coloured per stage
    lx = 130
    for st in STAGES:
        d.rectangle([lx, BUD_Y0 + 3, lx + 9, BUD_Y0 + 11], fill=STAGE_COL[st])
        s = "%s %.3f" % (st, lat[st])
        txt(d, (lx + 13, BUD_Y0 + 2), s, F_S, mix(BG, STAGE_COL[st], 0.88))
        lx += 13 + int(F_S.getlength(s)) + 18

    txt(d, (BAR_X1 + 14, BAR_Y0 - 4), "%.3f ms" % total, F_BIG, INK)
    txt(d, (BAR_X1 + 14, BAR_Y0 + 14), "used of the 50 ms tick", F_S, DIM)


def draw_footer(d, stamp, source_stamp):
    d.rectangle([0, FTR_Y0, W - 1, H - 1], fill=PANEL)
    d.line([0, FTR_Y0, W - 1, FTR_Y0], fill=GRID)
    txt(d, (10, FTR_Y0 + 12), stamp, F_S, INK, "lm")
    txt(d, (W - 10, FTR_Y0 + 12), source_stamp, F_S, DIM, "rm")


# --------------------------------------------------------------------------
def compose(frame_rgb, rec, ftr_stamp, src_stamp, n_settled, tick_secs,
            tr_atk, tr_hit, i):
    img = Image.new("RGB", (W, H), BG)
    up = Image.fromarray(np.ascontiguousarray(frame_rgb)).resize(
        (IMG_W, IMG_H), Image.NEAREST)
    img.paste(up, (IMG_X0, IMG_Y0))
    d = ImageDraw.Draw(img)

    per = np.asarray(rec["percep"], dtype=np.float64)
    act = np.asarray(rec["action"], dtype=np.int64)
    P = PERCEPTION_LAYOUT
    hitflash = bool(per[P["hurt_flash"]] > 0.5 and per[P["visible"]] > 0.5)

    draw_header(d, rec, n_settled, tick_secs)
    draw_left_rail(d)
    draw_frame_panel(d, per, hitflash)
    draw_right_rail(d, per, hitflash, tr_atk, tr_hit, i)
    draw_heads(d, act)
    draw_budget(d, rec["latency_ms"])
    draw_footer(d, ftr_stamp, src_stamp)
    d.rectangle([0, 0, W - 1, H - 1], outline=GRID)
    return img


def build_palette(raw_frames, n_img_colors=128):
    """One global palette: median-cut the RAW captures, then pin the UI colours.

    Median-cutting composed frames loses the red hurt tint and collapses amber
    to grey; median-cutting the raw captures keeps the game footage honest and
    the explicit swatch list keeps the chrome exact.
    """
    montage = np.concatenate(list(raw_frames), axis=0)
    base = Image.fromarray(montage).quantize(
        colors=n_img_colors, method=Image.MEDIANCUT, dither=Image.NONE)
    pal = list(base.getpalette()[: n_img_colors * 3])

    ui = list(BRAND) + [CELL_OFF, CELL_HOLD, VIOLET_DK, CYAN_DK, LINE_CYAN,
                        (255, 255, 255), (0, 0, 0)]
    # antialiasing ramps for every text colour used on both backgrounds
    for bgc in (BG, PANEL):
        for fg in (INK, DIM, CYAN, AMBER, RED, GREEN, VIOLET):
            for t in (0.18, 0.36, 0.55, 0.74, 0.88):
                ui.append(mix(bgc, fg, t))
    seen, ui_u = set(), []
    for c in ui:
        if c not in seen:
            seen.add(c)
            ui_u.append(c)
    ui_u = ui_u[: 256 - n_img_colors]
    for c in ui_u:
        pal.extend(c)
    pal.extend([0, 0, 0] * (256 - len(pal) // 3))
    p = Image.new("P", (1, 1))
    p.putpalette(pal[:768])
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-npy", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--out", default=OUT_GIF)
    ap.add_argument("--sidecar", default=OUT_JSON)
    ap.add_argument("--start", type=int, default=CLIP_START)
    ap.add_argument("--n", type=int, default=CLIP_N)
    ap.add_argument("--colors", type=int, default=128)
    ap.add_argument("--dump", default=None, help="also write PNGs of every 8th frame here")
    args = ap.parse_args()

    np.random.seed(0)
    rows, settled, frames = load(args)
    n_settled = len(settled)
    tick_secs = n_settled / 20.0

    a, b = args.start, args.start + args.n
    clip = settled[a:b]
    raw = np.asarray(frames[a:b])

    lat = np.array([[r["latency_ms"][s] for s in STAGES] for r in settled], dtype=np.float64)
    med = {s: float(np.median(lat[:, i])) for i, s in enumerate(STAGES)}
    med_tick = float(np.median([r["latency_ms"]["tick"] for r in settled]))

    ftr = ("session B · %s settled ticks / %.1f s · median capture %.2f / "
           "encode %.2f / policy %.2f / inject %.2f / tick %.2f ms"
           % (format(n_settled, ","), tick_secs, med["capture"], med["encode"],
              med["policy"], med["inject"], med_tick))
    src = ("%s · %s rows · rendered %s"
           % (LOG_NAME, format(len(rows), ","), _dt.date.today().isoformat()))

    P = PERCEPTION_LAYOUT
    tr_atk = [int(r["action"][4]) == 1 for r in clip]
    tr_hit = [bool(r["percep"][P["hurt_flash"]] > 0.5
                   and r["percep"][P["visible"]] > 0.5) for r in clip]

    imgs = []
    for i, rec in enumerate(clip):
        imgs.append(compose(raw[i], rec, ftr, src, n_settled, tick_secs,
                            tr_atk, tr_hit, i))

    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
        for i in range(0, len(imgs), 8):
            imgs[i].save(os.path.join(args.dump, "f_%03d.png" % i))
        imgs[-1].save(os.path.join(args.dump, "f_last.png"))

    pal = build_palette(raw, args.colors)
    q = [im.quantize(palette=pal, dither=Image.NONE) for im in imgs]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    q[0].save(args.out, save_all=True, append_images=q[1:], duration=50,
              loop=0, optimize=True, disposal=1)

    size = os.path.getsize(args.out)

    # ---- sidecar: every number the caption is allowed to quote --------
    P = PERCEPTION_LAYOUT
    t0 = clip[0]
    per0 = t0["percep"]
    acts = np.array([r["action"] for r in clip], dtype=np.int64)
    pers = np.array([r["percep"] for r in clip], dtype=np.float64)
    hits = int(((pers[:, P["hurt_flash"]] > 0.5) & (pers[:, P["visible"]] > 0.5)).sum())
    side = {
        "id": "live-pixels-tick",
        "rendered": _dt.date.today().isoformat(),
        "source_log_rows": len(rows),
        "settled_ticks": n_settled,
        "settled_seconds_at_20hz": round(tick_secs, 2),
        "log_wall_span_s": round(settled[-1]["t"] - settled[0]["t"], 2),
        "clip": {
            "settled_index_start": a, "settled_index_end": b - 1,
            "log_tick_start": t0["tick"], "log_tick_end": clip[-1]["tick"],
            "t_start_s": t0["t"], "t_end_s": clip[-1]["t"],
            "frames": len(clip), "fps": 20, "duration_s": round(len(clip) / 20.0, 2),
        },
        "frame0_tick": {
            "tick": t0["tick"], "t_s": t0["t"], "state": t0["state"],
            "action": [int(v) for v in t0["action"]],
            "action_decoded": {
                "forward": ["back", "none", "forward"][t0["action"][0]],
                "strafe": ["left", "none", "right"][t0["action"][1]],
                "jump": int(t0["action"][2]), "sprint": int(t0["action"][3]),
                "attack": int(t0["action"][4]),
                "yaw_deg_per_tick": CAMERA_BINS[t0["action"][5]],
                "pitch_deg_per_tick": CAMERA_BINS[t0["action"][6]],
            },
            "percep": per0,
            "percep_units": {
                "aim_err_yaw_deg": round(per0[P["aim_err_yaw"]] * 180.0, 2),
                "aim_err_pitch_deg": round(per0[P["aim_err_pitch"]] * 90.0, 2),
                "bbox_height": per0[P["bbox_height"]],
                "range_blocks": round(dist_from_bbox_height(per0[P["bbox_height"]]), 2),
                "self_hp": round(per0[P["self_hp"]] * 20.0, 2),
            },
            "latency_ms": t0["latency_ms"],
            "latency_stage_sum_ms": round(sum(t0["latency_ms"][s] for s in STAGES), 3),
        },
        "clip_stats": {
            "attack_ticks": int((acts[:, 4] == 1).sum()),
            "sprint_ticks": int((acts[:, 3] == 1).sum()),
            "jump_ticks": int((acts[:, 2] == 1).sum()),
            "back_ticks": int((acts[:, 0] == 0).sum()),
            "yaw_bin_counts": np.bincount(acts[:, 5], minlength=11).tolist(),
            "pitch_bin_counts": np.bincount(acts[:, 6], minlength=11).tolist(),
            "yaw_bins_used": int((np.bincount(acts[:, 5], minlength=11) > 0).sum()),
            "pitch_bins_used": int((np.bincount(acts[:, 6], minlength=11) > 0).sum()),
            "hurt_flash_ticks": hits,
            "visible_ticks": int((pers[:, P["visible"]] > 0.5).sum()),
            "bbox_height_min": round(float(pers[:, P["bbox_height"]].min()), 4),
            "bbox_height_max": round(float(pers[:, P["bbox_height"]].max()), 4),
            "range_blocks_min": round(dist_from_bbox_height(
                float(pers[:, P["bbox_height"]].max())), 2),
            "range_blocks_max": round(dist_from_bbox_height(
                float(pers[:, P["bbox_height"]].min())), 2),
        },
        "session_b_medians_ms": dict(med, tick=med_tick),
        "camera_bins_deg_per_tick": list(CAMERA_BINS),
        "action_heads": [list(h) for h in ACTION_HEADS],
        "projection": {"cx": CX0, "cy": CY0, "focal_px": FOCAL,
                       "frame": [FRAME_W, FRAME_H], "upscale": UPSCALE},
        "gif": {"path": os.path.relpath(args.out, REPO), "bytes": size,
                "mb": round(size / 1048576.0, 3),
                "size": [W, H], "frames": len(q), "duration_ms": 50,
                "colors": args.colors},
    }
    os.makedirs(os.path.dirname(args.sidecar), exist_ok=True)
    with open(args.sidecar, "w") as fh:
        json.dump(side, fh, indent=1, sort_keys=False)

    print("%s  %dx%d  %d frames  %.2f MB" % (args.out, W, H, len(q), size / 1048576.0))
    print("clip settled[%d:%d] = log ticks %d..%d  t=%.2f..%.2f s"
          % (a, b, t0["tick"], clip[-1]["tick"], t0["t"], clip[-1]["t"]))
    print("frame0 action", t0["action"], "latency", t0["latency_ms"])
    print("attack %d/%d  hurt_flash %d  yaw bins used %d/11  pitch bins used %d/11"
          % (side["clip_stats"]["attack_ticks"], len(clip), hits,
             side["clip_stats"]["yaw_bins_used"], side["clip_stats"]["pitch_bins_used"]))
    if size > 2.5 * 1048576:
        print("OVER BUDGET: %.2f MB > 2.5 MB" % (size / 1048576.0))


if __name__ == "__main__":
    main()
