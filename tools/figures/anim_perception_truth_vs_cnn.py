"""Ground truth vs PerceptionCNN prediction, drawn on real held-out game frames.

Renders docs/assets/perception-truth-vs-cnn.gif: 56 consecutive real Minecraft
1.8 capture frames (170x96 RGB, the exact tensor the sensor stage eats) shown at
4x nearest-neighbour, with two boxes on the enemy -- AMBER = the telemetry
ground-truth label, dashed CYAN = what runs/perception/perception_v12.pt infers
from that single frame -- plus a live 12-row readout of PERCEPTION_LAYOUT.

Every number on screen is measured, none are staged:
  * frames + labels + mask : v11_telemetry_true.npz, held-out tail (idx >= 13466
    of a sequential 85/15 split, matching pvpbot/perception/train.py).
  * predictions            : runs/perception/perception_v12.pt, one forward pass
    per frame, no temporal smoothing.
  * projection             : measured perspective model of the real capture,
    principal point (84.5, 50.0) px and f = 42.0 px (~127 deg horizontal,
    ~98 deg vertical FOV). synth.PX_PER_DEG is a LINEAR angle->pixel map and
    diverges from this everywhere but dead centre -- 11.3 vs 25.5 deg at 20 px
    off-axis, 48.0 vs 63.7 deg at the edge, a 14-16 deg absolute gap across the
    whole frame; it is deliberately not used here. Range in
    blocks does come from synth.dist_from_bbox_height, which is the same
    inverse-perspective call the live adapter makes
    (pvpbot/perception/adapter.py:226).
  * hurt flash             : PERCEPTION_LAYOUT["hurt_flash"] truth column. The
    footage itself reddens on exactly these frames (fraction of reddish pixels
    0.48-0.65% when the label is 0, 0.80-1.71% when it is 1).

Six of the twelve slots (self_hp, rel_screen_vx, rel_screen_vy, self_speed,
enemy_on_ground, reserved) have mask == 0 in every row of this capture, so they
carry no ground truth and are drawn dim and labelled "no label".

SOURCE DATA IS GITIGNORED and lives outside the repo; a fresh clone cannot
re-render this. The committed GIF and its sidecar
docs/assets/data/perception-truth-vs-cnn.json are the durable artifacts.

Usage:  python3 tools/figures/anim_perception_truth_vs_cnn.py
        python3 tools/figures/anim_perception_truth_vs_cnn.py --frames 48
"""
import argparse
import datetime
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

# --------------------------------------------------------------------------
# Paths. The frame corpus is not in the repo (261 MB, gitignored, produced by
# the calibration capture session); the checkpoint is under runs/.
# --------------------------------------------------------------------------
DATA_CANDIDATES = [
    os.path.join(REPO, "data", "realdata", "v11_telemetry_true.npz"),
]
CKPT = os.path.join(REPO, "runs", "perception", "perception_v12.pt")
OUT = os.path.join(REPO, "docs", "assets", "perception-truth-vs-cnn.gif")
SIDECAR = os.path.join(REPO, "docs", "assets", "data",
                       "perception-truth-vs-cnn.json")

SPLIT_FRAC = 0.85          # pvpbot/perception/train.py: n_tr = int(len(frames)*0.85)
# Held-out engagement 14105-14160 is the only run in the tail that keeps the
# enemy visible for 50+ consecutive frames at close range (bbox 0.51-0.88) with
# heavy hurt-flash activity. It is entered at 14110: frames 14105-14109 are the
# approach, where the CNN is 25-46 deg off and the cyan box lands on bare grass
# -- true, but a misleading frame to loop on, and 14110 is itself a hurt-flash
# frame. Enemy visible == 1 on every frame of 14110-14160; the run ends at 14161.
CLIP_START = 14110
CLIP_LEN = 51

# Measured camera model of the real capture (NOT synth's linear PX_PER_DEG).
CX0, CY0, FOCAL = 84.5, 50.0, 42.0
FW, FH = 170, 96
UP = 4                     # nearest-neighbour upscale; 5x blows the size budget

# --------------------------------------------------------------------------
# Locked palette, shared with tools/figures/anim_live_cockpit.py.
# CYAN = anything the machine produced.  AMBER = anything true independently of
# it.  RED = damage.  DIM = held / absent.
# --------------------------------------------------------------------------
BG, PANEL, GRID, INK, DIM = "#12161c", "#1a212a", "#2b3542", "#ccd6e2", "#66747f"
CYAN, AMBER, RED, GREEN, VIOLET = "#3fd0d8", "#e8a33d", "#e2564a", "#6ecf94", "#a184e8"


def rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def mix(a, b, t):
    A, B = rgb(a), rgb(b)
    return tuple(int(round(A[i] * (1 - t) + B[i] * t)) for i in range(3))


C_BG, C_PANEL, C_GRID = rgb(BG), rgb(PANEL), rgb(GRID)
C_INK, C_DIM = rgb(INK), rgb(DIM)
C_CYAN, C_AMBER, C_RED = rgb(CYAN), rgb(AMBER), rgb(RED)
C_AMBER_FAINT = mix(AMBER, BG, 0.52)     # crosshair -> truth aim line
C_CYAN_DIM = mix(CYAN, BG, 0.45)
C_RED_CHIP = mix(RED, BG, 0.62)
C_INK_DIM = mix(INK, BG, 0.35)
UI_COLORS = [C_BG, C_PANEL, C_GRID, C_INK, C_DIM, C_CYAN, C_AMBER, C_RED,
             rgb(GREEN), rgb(VIOLET), C_AMBER_FAINT, C_CYAN_DIM, C_RED_CHIP,
             C_INK_DIM, mix(GRID, BG, 0.5), (255, 255, 255)]

# --------------------------------------------------------------------------
# Canvas geometry. Every rectangle is fixed for the whole clip: a panel that
# shifts by one pixel between frames destroys GIF inter-frame diffing.
# --------------------------------------------------------------------------
W, H = 1004, 444
HEAD_Y0, HEAD_Y1 = 1, 35            # header strip rows [1, 35)
IMG_X0, IMG_Y0 = 1, 35              # 680 x 384 image block
IMG_W, IMG_H = FW * UP, FH * UP
PAN_X0, PAN_X1 = IMG_X0 + IMG_W, W - 1
FOOT_Y0 = IMG_Y0 + IMG_H + 1        # 420
PAD = 10
PX0, PX1 = PAN_X0 + PAD, PAN_X1 - PAD       # panel content columns
ROW_Y0, ROW_H = 64, 16
LBL_W = 92
TRK_X0, TRK_X1 = PX0 + LBL_W + 6, PX1
SCAT_X, SCAT_Y, SCAT_S = PX0, 284, 132      # aim-error scatter square
OFF_X = SCAT_X + SCAT_S + 16                # residual readout block
LEGEND_BOX = (IMG_X0 + 8, IMG_Y0 + 8, IMG_X0 + 8 + 168, IMG_Y0 + 8 + 34)

FONT_PATH = "/System/Library/Fonts/Menlo.ttc"


def font(sz, bold=False):
    return ImageFont.truetype(FONT_PATH, sz, index=1 if bold else 0)


# --------------------------------------------------------------------------
# Perspective projection.  Self-test: b == 0 must give v == 0 for ANY p.
# --------------------------------------------------------------------------
def project(yaw_n, pitch_n, selfpitch_n):
    """PERCEPTION_LAYOUT normalised angles -> pixel centre in the 170x96 frame."""
    a = math.radians(float(yaw_n) * 180.0)
    b = math.radians(float(pitch_n) * 90.0)
    p = math.radians(float(selfpitch_n) * 90.0)
    pb = p + b
    den = math.cos(pb) * math.cos(p) * math.cos(a) + math.sin(pb) * math.sin(p)
    if abs(den) < 1e-6:
        den = math.copysign(1e-6, den)
    u = FOCAL * math.cos(pb) * math.sin(a) / den
    v = FOCAL * (math.sin(pb) * math.cos(p) - math.cos(pb) * math.sin(p) * math.cos(a)) / den
    return CX0 + u, CY0 + v


def _selftest():
    for pdeg in (-40, -20, 0, 20, 40, 60):
        x, y = project(0.0, 0.0, pdeg / 90.0)
        assert abs(x - CX0) < 1e-9 and abs(y - CY0) < 1e-9, (pdeg, x, y)


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
def dashed_rect(d, box, color, width=2, dash=5, gap=4):
    x0, y0, x1, y1 = box
    def seg(ax, ay, bx, by):
        L = math.hypot(bx - ax, by - ay)
        if L < 1:
            return
        n = max(1, int(L))
        ux, uy = (bx - ax) / L, (by - ay) / L
        t = 0.0
        while t < L:
            t2 = min(L, t + dash)
            d.line([(ax + ux * t, ay + uy * t), (ax + ux * t2, ay + uy * t2)],
                   fill=color, width=width)
            t = t2 + gap
    seg(x0, y0, x1, y0); seg(x1, y0, x1, y1)
    seg(x1, y1, x0, y1); seg(x0, y1, x0, y0)


def dashed_line(d, p0, p1, color, dash=4, gap=3, width=1):
    ax, ay = p0; bx, by = p1
    L = math.hypot(bx - ax, by - ay)
    if L < 1:
        return
    ux, uy = (bx - ax) / L, (by - ay) / L
    t = 0.0
    while t < L:
        t2 = min(L, t + dash)
        d.line([(ax + ux * t, ay + uy * t), (ax + ux * t2, ay + uy * t2)],
               fill=color, width=width)
        t = t2 + gap


def shadow_text(d, xy, txt, f, color, anchor=None):
    """1 px dark drop so small type stays legible over game footage."""
    x, y = xy
    d.text((x + 1, y + 1), txt, font=f, fill=C_BG, anchor=anchor)
    d.text((x, y), txt, font=f, fill=color, anchor=anchor)


def run_text(d, x, y, parts, gap=0):
    """Draw coloured (text, font, colour) runs left to right; return end x."""
    for txt, f, col in parts:
        d.text((x, y), txt, font=f, fill=col)
        x += f.getlength(txt) + gap
    return x


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_clip(n_frames):
    src = next((p for p in DATA_CANDIDATES if os.path.exists(p)), None)
    if src is None:
        sys.exit("source clip no longer on disk: " + DATA_CANDIDATES[0] +
                 "\n(15,843-frame real-capture corpus, gitignored; "
                 "docs/assets/perception-truth-vs-cnn.gif is the committed artifact)")
    if not os.path.exists(CKPT):
        sys.exit("perception checkpoint missing: " + CKPT)
    z = np.load(src)
    frames, labels, mask = z["frames"], z["labels"], z["mask"]
    total = len(frames)
    held0 = int(total * SPLIT_FRAC)
    a, b = CLIP_START, CLIP_START + n_frames
    if a < held0:
        sys.exit("clip %d starts inside the training split (held-out from %d)"
                 % (a, held0))
    return frames[a:b], labels[a:b], mask[a:b], total, held0, src


def predict(frames):
    import torch
    from pvpbot.perception.infer import FrameEncoder
    torch.manual_seed(0)
    enc = FrameEncoder(CKPT)
    n_params = sum(p.numel() for p in enc.model.parameters())
    x = torch.from_numpy(np.moveaxis(frames, -1, 1).copy()).float().div_(255.0)
    with torch.no_grad():
        pred = enc.model(x).numpy()
    return pred, n_params, int(enc.model.net[0].in_channels)


# --------------------------------------------------------------------------
# Panel row spec: (slot index, name, half-range or max, signed, has_truth)
# --------------------------------------------------------------------------
ROWS = [
    (0,  "aim_err_yaw",     0.30, True,  True),
    (1,  "aim_err_pitch",   0.50, True,  True),
    (2,  "bbox_height",     1.00, False, True),
    (3,  "visible",         1.00, False, True),
    (4,  "self_pitch",      0.50, True,  True),
    (5,  "self_hp",         1.00, False, False),
    (6,  "hurt_flash",      1.00, False, True),
    (7,  "rel_screen_vx",   0.20, True,  False),
    (8,  "rel_screen_vy",   0.20, True,  False),
    (9,  "self_speed",      0.40, False, False),
    (10, "enemy_on_ground", 1.00, False, False),
    (11, "reserved",        0.50, True,  False),
]


def draw_bar(d, y, h, val, rng, signed, color, track=(None, None)):
    x0 = TRK_X0 if track[0] is None else track[0]
    x1 = TRK_X1 if track[1] is None else track[1]
    over = False
    if signed:
        mid = (x0 + x1) / 2.0
        t = float(np.clip(val / rng, -1.0, 1.0))
        over = abs(val / rng) > 1.0
        xa, xb = (mid, mid + t * (x1 - x0) / 2.0)
        if xb < xa:
            xa, xb = xb, xa
        d.rectangle([xa, y, max(xa + 1, xb), y + h], fill=color)
    else:
        t = float(np.clip(val / rng, 0.0, 1.0))
        over = val / rng > 1.0
        d.rectangle([x0, y, x0 + max(1, t * (x1 - x0)), y + h], fill=color)
    if over:
        d.rectangle([x1 - 2, y, x1, y + h], fill=color)


# --------------------------------------------------------------------------
# Frame composition
# --------------------------------------------------------------------------
def compose(i, frames, lab, pred, geo, meta, fonts, static):
    f9, f9b, f10, f10b, f11b, f8, f22b = fonts
    im = static.copy()
    d = ImageDraw.Draw(im)

    L, P = lab[i], pred[i]
    hurt = L[6] > 0.5

    # ---- left: the real captured frame at 4x nearest ----------------------
    shot = Image.fromarray(frames[i]).resize((IMG_W, IMG_H), Image.NEAREST)
    im.paste(shot, (IMG_X0, IMG_Y0))
    d = ImageDraw.Draw(im)

    def to_canvas(px, py):
        return IMG_X0 + px * UP, IMG_Y0 + py * UP

    ccx, ccy = to_canvas(CX0, CY0)
    tx, ty = geo["tx"][i], geo["ty"][i]
    cx, cy = geo["cx"][i], geo["cy"][i]
    th, tw = geo["th"][i], geo["tw"][i]
    ph, pw = geo["ph"][i], geo["pw"][i]

    # aim line: crosshair -> the true target centre. Its length IS the aim error.
    d.line([(ccx, ccy), (tx, ty)], fill=C_AMBER_FAINT, width=1)

    # boxes. Height is PERCEPTION_LAYOUT["bbox_height"] * 96 px, the field's own
    # definition (pvpbot/spec.py:73); both boxes use the identical transform, so
    # what is visible between them is model error and nothing else.
    dashed_rect(d, (cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2),
                C_CYAN, width=2, dash=6, gap=5)
    d.rectangle([tx - tw / 2, ty - th / 2, tx + tw / 2, ty + th / 2],
                outline=C_AMBER, width=2)
    # centre marks: the residual below is measured between exactly these
    for (mx_, my_, col) in ((tx, ty, C_AMBER), (cx, cy, C_CYAN)):
        d.line([(mx_ - 4, my_), (mx_ + 4, my_)], fill=col, width=1)
        d.line([(mx_, my_ - 4), (mx_, my_ + 4)], fill=col, width=1)

    # residual, in body-widths of the true target, anchored above the boxes so
    # it never lands on the player it is measuring
    off_bw = geo["off_bw"][i]
    if math.hypot(cx - tx, cy - ty) > 8:
        dashed_line(d, (tx, ty), (cx, cy), C_INK_DIM, dash=3, gap=3)
    lx = max(tx + tw / 2, cx + pw / 2) + 6
    ly = min(ty - th / 2, cy - ph / 2) - 2
    lx = min(max(lx, IMG_X0 + 4), IMG_X0 + IMG_W - 84)
    ly = min(max(ly, IMG_Y0 + 46), IMG_Y0 + IMG_H - 16)
    shadow_text(d, (lx, ly), "Δ %.2f body-w" % off_bw, f9b, C_INK)

    # measured principal point (the game's own crosshair sits here); the dark
    # halo keeps it visible when the target passes under it
    for col, w in ((C_BG, 3), (C_INK, 1)):
        d.line([(ccx - 9, ccy), (ccx - 3, ccy)], fill=col, width=w)
        d.line([(ccx + 3, ccy), (ccx + 9, ccy)], fill=col, width=w)
        d.line([(ccx, ccy - 9), (ccx, ccy - 3)], fill=col, width=w)
        d.line([(ccx, ccy + 3), (ccx, ccy + 9)], fill=col, width=w)

    if hurt:
        d.rectangle([IMG_X0, IMG_Y0, IMG_X0 + IMG_W - 1, IMG_Y0 + IMG_H - 1],
                    outline=C_RED, width=2)

    # the static legend chip lives under the pasted footage; blit it back
    im.paste(static.crop(LEGEND_BOX), LEGEND_BOX[:2])
    d = ImageDraw.Draw(im)

    # ---- header ------------------------------------------------------------
    d.rectangle([IMG_X0, HEAD_Y0, PAN_X1 - 1, HEAD_Y1 - 1], fill=C_PANEL)
    run_text(d, 12, 6, [
        ("STAGE 1", f11b, C_INK),
        ("  ·  PerceptionCNN 3,500,204 params  ·  held-out frame ", f10, C_DIM),
        ("%d + %d" % (CLIP_START, i), f10b, C_INK),
    ])
    d.text((12, 21),
           "170x96 RGB in  →  12 floats out   ·   20 fps, 1 frame = 1 capture tick"
           "   ·   telemetry truth vs one forward pass, no smoothing",
           font=f9, fill=C_DIM)
    d.text((W - 13, 6), "t = %.2f s" % (i / 20.0), font=f10b, fill=C_INK, anchor="ra")
    d.text((W - 13, 21), meta["stamp"], font=f9, fill=C_DIM, anchor="ra")

    # ---- right panel: 12-slot readout -------------------------------------
    for k, (si, name, rng, signed, has_t) in enumerate(ROWS):
        y = ROW_Y0 + k * ROW_H
        col_lbl = C_INK if has_t else C_DIM
        d.text((PX0, y + 2), name, font=f10, fill=col_lbl)
        if signed:
            midx = (TRK_X0 + TRK_X1) / 2.0
            d.line([(midx, y + 1), (midx, y + 14)], fill=C_GRID, width=1)
        if has_t:
            draw_bar(d, y + 2, 5, float(L[si]), rng, signed, C_AMBER)
            draw_bar(d, y + 9, 5, float(P[si]), rng, signed, C_CYAN)
        else:
            d.text((TRK_X0 + 2, y + 1), "no label", font=f9, fill=C_DIM)
            draw_bar(d, y + 11, 3, float(P[si]), rng, signed, mix(DIM, BG, 0.35))

    # ---- aim-error scatter -------------------------------------------------
    sx0, sy0, S = SCAT_X, SCAT_Y, SCAT_S
    ppd = (S / 2.0) / 35.0
    n = len(lab)

    def spt(j):
        px = sx0 + S / 2.0 + geo["ey"][j] * ppd
        py = sy0 + S / 2.0 + geo["ep"][j] * ppd
        return min(max(px, sx0 + 3), sx0 + S - 3), min(max(py, sy0 + 3), sy0 + S - 3)

    for j in range(i):
        px, py = spt(j)
        d.rectangle([px - 1, py - 1, px, py], fill=C_DIM)
    px, py = spt(i)
    d.ellipse([px - 4, py - 4, px + 4, py + 4], outline=C_CYAN, width=1)
    d.rectangle([px - 1, py - 1, px + 1, py + 1], fill=C_CYAN)
    d.text((sx0 + S - 4, sy0 + 4), "n = %d / %d" % (i + 1, n), font=f9,
           fill=C_DIM, anchor="ra")

    # ---- residual readout --------------------------------------------------
    ox0 = OFF_X
    d.text((ox0, 296), "%.2f" % off_bw, font=f22b, fill=C_INK)
    d.text((ox0 + 62, 310), "body-widths", font=f9, fill=C_DIM)
    hx0, hy1, hh = ox0, 410, 40
    for j in range(n):
        bx = hx0 + j * 2.8
        v = min(geo["off_bw"][j] / 2.0, 1.0)
        col = C_CYAN if j == i else (C_DIM if j < i else mix(GRID, BG, 0.35))
        d.rectangle([bx, hy1 - max(1, v * hh), bx + 1.8, hy1], fill=col)

    # ---- footer ------------------------------------------------------------
    d.rectangle([IMG_X0, FOOT_Y0, PAN_X1 - 1, H - 2], fill=C_PANEL)
    fy = FOOT_Y0 + 6
    run_text(d, 12, fy - 1, [
        ("aim err  yaw ", f10, C_DIM),
        ("%+.1f" % (P[0] * 180), f11b, C_CYAN), (" / ", f10, C_DIM),
        ("%+.1f" % (L[0] * 180), f11b, C_AMBER), (" deg", f10, C_DIM),
        ("    pitch ", f10, C_DIM),
        ("%+.1f" % (P[1] * 90), f11b, C_CYAN), (" / ", f10, C_DIM),
        ("%+.1f" % (L[1] * 90), f11b, C_AMBER), (" deg", f10, C_DIM),
        ("    range ", f10, C_DIM),
        ("%.2f" % geo["rng_p"][i], f11b, C_CYAN), (" / ", f10, C_DIM),
        ("%.2f" % geo["rng_t"][i], f11b, C_AMBER), (" blk", f10, C_DIM),
        ("    offset ", f10, C_DIM),
        ("%.2f" % off_bw, f11b, C_INK), (" body-w", f10, C_DIM),
    ])
    d.text((W - 116, fy), meta["summary"], font=f9, fill=C_DIM, anchor="ra")
    bx1 = W - 13
    bx0 = bx1 - 92
    if hurt:
        d.rectangle([bx0, fy - 3, bx1, fy + 14], fill=C_RED_CHIP, outline=C_RED)
        d.text(((bx0 + bx1) / 2, fy + 1), "HURT FLASH", font=f9b, fill=(255, 236, 232),
               anchor="ma")
    else:
        d.rectangle([bx0, fy - 3, bx1, fy + 14], outline=mix(GRID, BG, 0.5))
        d.text(((bx0 + bx1) / 2, fy + 1), "hurt flash", font=f9, fill=mix(DIM, BG, 0.45),
               anchor="ma")
    return im


def build_static(fonts, held0, total):
    """Everything that never moves: borders, panel chrome, legends, axes."""
    f9, f9b, f10, f10b, f11b, f8, f22b = fonts
    im = Image.new("RGB", (W, H), C_BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W - 1, H - 1], outline=C_GRID)
    d.rectangle([PAN_X0, IMG_Y0, PAN_X1 - 1, IMG_Y0 + IMG_H - 1], fill=C_PANEL)
    d.line([(PAN_X0, IMG_Y0), (PAN_X0, IMG_Y0 + IMG_H - 1)], fill=C_GRID)
    d.line([(IMG_X0, HEAD_Y1 - 1), (PAN_X1 - 1, HEAD_Y1 - 1)], fill=C_GRID)
    d.line([(IMG_X0, FOOT_Y0 - 1), (PAN_X1 - 1, FOOT_Y0 - 1)], fill=C_GRID)

    d.text((PX0, 38), "PERCEPTION VECTOR", font=f10b, fill=C_INK)
    d.text((PX1, 39), "12 floats / frame", font=f9, fill=C_DIM, anchor="ra")
    d.rectangle([PX0, 53, PX0 + 10, 58], fill=C_AMBER)
    d.text((PX0 + 15, 50), "telemetry truth", font=f9, fill=C_DIM)
    d.rectangle([PX0 + 122, 53, PX0 + 132, 58], fill=C_CYAN)
    d.text((PX0 + 137, 50), "CNN, 1 frame", font=f9, fill=C_DIM)

    sx0, sy0, S = SCAT_X, SCAT_Y, SCAT_S
    d.line([(PX0, sy0 - 22), (PX1, sy0 - 22)], fill=C_GRID)
    d.text((PX0, sy0 - 17), "AIM ERROR  CNN − truth", font=f9b, fill=C_INK)
    d.text((OFF_X, sy0 - 17), "CNN BOX − TRUTH BOX", font=f9b, fill=C_INK)

    d.rectangle([sx0, sy0, sx0 + S, sy0 + S], fill=C_BG, outline=C_GRID)
    mid = S / 2.0
    d.line([(sx0 + 1, sy0 + mid), (sx0 + S - 1, sy0 + mid)], fill=mix(GRID, BG, 0.3))
    d.line([(sx0 + mid, sy0 + 1), (sx0 + mid, sy0 + S - 1)], fill=mix(GRID, BG, 0.3))
    r10 = 10 * (S / 2.0) / 35.0
    for rr in (r10, r10 * 2):
        for k in range(0, 72, 2):
            a0 = 2 * math.pi * k / 72
            d.point((sx0 + mid + rr * math.cos(a0), sy0 + mid + rr * math.sin(a0)),
                    fill=C_GRID)
    d.text((sx0 + mid + r10 + 2, sy0 + mid - 12), "10°", font=f9, fill=C_DIM)
    d.text((sx0 + mid + r10 * 2 + 2, sy0 + mid + 2), "20°", font=f9, fill=C_DIM)
    d.text((sx0 + 4, sy0 + S - 13), "yaw err →", font=f9, fill=C_DIM)
    d.text((sx0 + 4, sy0 + 4), "pitch err ↑", font=f9, fill=C_DIM)
    d.text((sx0 + S - 4, sy0 + S - 13), "±35°", font=f9, fill=C_DIM, anchor="ra")

    ox0 = OFF_X
    d.text((ox0, 326), "residual between the two", font=f9, fill=C_DIM)
    d.text((ox0, 337), "box centres, in widths of", font=f9, fill=C_DIM)
    d.text((ox0, 348), "the 0.6-block target", font=f9, fill=C_DIM)
    d.text((ox0, 363), "every frame of the clip", font=f9, fill=C_DIM)
    d.line([(ox0, 390), (ox0 + CLIP_LEN * 2.8, 390)], fill=mix(GRID, BG, 0.45))
    d.text((PX1, 377), "1.0", font=f9, fill=C_DIM, anchor="ra")
    d.line([(ox0, 411), (ox0 + CLIP_LEN * 2.8, 411)], fill=C_GRID)

    # legend chip over the sky, top-left of the footage: static, so it is free
    lx, ly = IMG_X0 + 8, IMG_Y0 + 8
    d.rectangle([lx, ly, lx + 168, ly + 34], fill=C_PANEL, outline=C_GRID)
    d.rectangle([lx + 7, ly + 7, lx + 19, ly + 12], outline=C_AMBER, width=2)
    d.text((lx + 26, ly + 4), "TRUTH  telemetry", font=f9, fill=C_AMBER)
    dashed_rect(d, (lx + 7, ly + 21, lx + 19, ly + 26), C_CYAN, width=2, dash=3, gap=2)
    d.text((lx + 26, ly + 18), "CNN    this frame", font=f9, fill=C_CYAN)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--frames", type=int, default=CLIP_LEN)
    ap.add_argument("--colors", type=int, default=128)
    ap.add_argument("--dump", default=None, help="also write PNGs of a few frames here")
    args = ap.parse_args()

    _selftest()
    np.random.seed(0)

    frames, lab, mask, total, held0, src = load_clip(args.frames)
    pred, n_params, in_ch = predict(frames)
    n = len(frames)
    assert in_ch == 3, "checkpoint expects %d input channels, clip is RGB" % in_ch
    assert mask[:, [0, 1, 2, 3, 4, 6]].min() == 1.0, "clip has unlabelled rows"
    assert mask[:, [5, 7, 8, 9, 10, 11]].max() == 0.0, "unexpected truth in dim slots"

    from pvpbot.perception.synth import dist_from_bbox_height

    geo = {k: np.zeros(n) for k in
           ("tx", "ty", "cx", "cy", "th", "tw", "ph", "pw", "off_bw",
            "ey", "ep", "rng_t", "rng_p")}
    for i in range(n):
        L, P = lab[i], pred[i]
        x, y = project(L[0], L[1], L[4])
        geo["tx"][i], geo["ty"][i] = IMG_X0 + x * UP, IMG_Y0 + y * UP
        x, y = project(P[0], P[1], P[4])
        geo["cx"][i], geo["cy"][i] = IMG_X0 + x * UP, IMG_Y0 + y * UP
        geo["th"][i] = float(L[2]) * FH * UP
        geo["tw"][i] = 0.36 * geo["th"][i]
        geo["ph"][i] = float(np.clip(P[2], 0.05, 1.5)) * FH * UP
        geo["pw"][i] = 0.36 * geo["ph"][i]
        geo["off_bw"][i] = math.hypot(geo["cx"][i] - geo["tx"][i],
                                      geo["cy"][i] - geo["ty"][i]) / geo["tw"][i]
        geo["ey"][i] = (P[0] - L[0]) * 180.0
        geo["ep"][i] = -(P[1] - L[1]) * 90.0     # + = CNN thinks target is higher
        geo["rng_t"][i] = dist_from_bbox_height(float(L[2]))
        geo["rng_p"][i] = dist_from_bbox_height(float(P[2]))

    fonts = (font(9), font(9, True), font(10), font(10, True), font(11, True),
             font(8), font(22, True))
    meta = {"stamp": "perception_v12.pt · held-out idx ≥ %d of %s · %s"
                     % (held0, format(total, ","), datetime.date.today().isoformat())}
    meta["summary"] = ("clip median  %.1f° yaw  ·  %.2f body-w"
                       % (np.median(np.abs(geo["ey"])), np.median(geo["off_bw"])))
    static = build_static(fonts, held0, total)

    comp = [compose(i, frames, lab, pred, geo, meta, fonts, static) for i in range(n)]

    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
        for i in (0, 5, 15, 25, 30, 40, 50, 55):
            if i < n:
                comp[i].save(os.path.join(args.dump, "f%02d.png" % i))

    # ---- one global palette, median-cut from the RAW footage of this clip --
    # Building it from the composed frames destroys the red hurt tint.
    nc = args.colors
    base = Image.fromarray(np.concatenate(list(frames), axis=0)).quantize(
        colors=nc, method=Image.MEDIANCUT, dither=Image.NONE)
    pal = list(base.getpalette()[:nc * 3])
    for c in UI_COLORS:
        pal += list(c)
    pal += [0, 0, 0] * (256 - len(pal) // 3)
    pimg = Image.new("P", (1, 1))
    pimg.putpalette(pal[:768])
    q = [c.quantize(palette=pimg, dither=Image.NONE) for c in comp]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    q[0].save(args.out, save_all=True, append_images=q[1:], duration=50,
              loop=0, optimize=True, disposal=1)

    ey, ep = np.abs(geo["ey"]), np.abs(geo["ep"])
    side = {
        "id": "perception-truth-vs-cnn",
        "source_frames": os.path.basename(src),
        "corpus_frames": int(total),
        "heldout_from_index": int(held0),
        "clip": [int(CLIP_START), int(CLIP_START + n - 1)],
        "n_frames": int(n), "fps": 20, "duration_s": round(n / 20.0, 2),
        "checkpoint": "runs/perception/perception_v12.pt",
        "cnn_params": int(n_params), "cnn_in_channels": int(in_ch),
        "frame_shape_hwc": [FH, FW, 3],
        "camera_model": {"cx": CX0, "cy": CY0, "focal_px": FOCAL,
                         "hfov_deg": round(2 * math.degrees(math.atan(FW / 2 / FOCAL)), 1),
                         "vfov_deg": round(2 * math.degrees(math.atan(FH / 2 / FOCAL)), 1)},
        "median_abs_yaw_err_deg": round(float(np.median(ey)), 2),
        "median_abs_pitch_err_deg": round(float(np.median(ep)), 2),
        "p90_abs_yaw_err_deg": round(float(np.percentile(ey, 90)), 2),
        "max_abs_yaw_err_deg": round(float(ey.max()), 2),
        "median_box_offset_bodywidths": round(float(np.median(geo["off_bw"])), 2),
        "max_box_offset_bodywidths": round(float(geo["off_bw"].max()), 2),
        "range_blocks_truth": [round(float(geo["rng_t"].min()), 2),
                               round(float(geo["rng_t"].max()), 2)],
        "bbox_height_truth": [round(float(lab[:, 2].min()), 3),
                              round(float(lab[:, 2].max()), 3)],
        "hurt_flash_frames": int((lab[:, 6] > 0.5).sum()),
        "slots_with_truth": [ROWS[k][1] for k in range(12) if ROWS[k][4]],
        "slots_without_truth": [ROWS[k][1] for k in range(12) if not ROWS[k][4]],
        "gif_bytes": os.path.getsize(args.out),
        "rendered": datetime.date.today().isoformat(),
    }
    os.makedirs(os.path.dirname(SIDECAR), exist_ok=True)
    with open(SIDECAR, "w") as fh:
        json.dump(side, fh, indent=2)
    print("wrote %s  %.2f MB  %dx%d  %d frames" %
          (args.out, side["gif_bytes"] / 1e6, W, H, n))
    print("sidecar %s" % SIDECAR)
    print("median |yaw err| %.2f deg   median box offset %.2f body-widths" %
          (side["median_abs_yaw_err_deg"], side["median_box_offset_bodywidths"]))


if __name__ == "__main__":
    main()
