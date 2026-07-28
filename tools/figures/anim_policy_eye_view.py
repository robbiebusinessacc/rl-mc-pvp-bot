"""Full-bleed policy eye view: the trained policy fighting P4-Hacker.

Same rollout as ``tools/figures/anim_duel_vs_p4hacker.py`` -- this script
imports ``pick_duel`` and the projection helpers from it rather than
re-deriving either -- but renders only the first-person panel, full canvas,
so it works as a hero image at the top of the README.

Source (all of it gitignored; a fresh clone will not have ``runs/``):

  runs/fov1/ckpt_32.8B_faithful78.pt   PolicyNet checkpoint, meta step
                                       32,799,457,280, driven as side 0
  pvpbot.eval.practice.PracticeHacker  side 1 (P4 tier: 600 deg/s aim,
                                       0-tick reaction, 11 cps)
  pvpbot.sim.env.DuelVecEnv            14x14 square arena, full-information
                                       observations for both sides

Every value drawn is per-tick ground truth snapshotted BEFORE each env.step
(env.pos / env.vel / env.yaw / env.pitch / env.hp / env.hurt / env.on_ground /
env.sprinting), plus the (2, 7) action array fed to that step and the HP delta
the step resolved.  The view itself is a pinhole projection from the policy's
own eye (pos + 1.62 m, its own yaw/pitch, Minecraft's default 70 deg vertical
FOV): a geometric reconstruction of the simulator, not a screenshot of
Minecraft and not the live pixel pipeline.

Usage:
    python3 tools/figures/anim_policy_eye_view.py
    python3 tools/figures/anim_policy_eye_view.py --cache /tmp/duel.npz
"""
import argparse
import datetime as _dt
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Polygon, Rectangle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

SIBLING = os.path.join(HERE, "anim_duel_vs_p4hacker.py")
if not os.path.exists(SIBLING):
    raise SystemExit("sibling generator no longer on disk: %s\n"
                     "anim_policy_eye_view.py reuses its pick_duel() and "
                     "projection helpers." % SIBLING)

# pick_duel / basis / proj / cube / visible_runs and the palette all come from
# the four-panel generator, so both GIFs render the same duel identically.
from anim_duel_vs_p4hacker import (  # noqa: E402
    AMBER, ARENA_R, BG, CYAN, DIM, EYE_H, FLASH, FOCAL, GRID, HORIZ, INK,
    NEAR, PANEL, RED, WALLTOP, CUBE_E, CUBE_F, arena_perimeter, basis, cube,
    pick_duel, proj, visible_runs)

GREEN = "#6ecf94"          # locked palette: a landed hit
BLADE, HILT = "#e6edf3", "#7c5a3a"
# The camera here is the policy's own head, so it pans every tick and the whole
# background moves with it -- the one lever left on GIF size is how many
# background pixels change palette index between frames.  A low-contrast,
# non-antialiased ground grid quantises into the ground plane instead of
# smearing a halo of intermediate colours across the frame.
GRID_LINE, WALL_FILL = "#212b37", "#26313f"
SKY_C, GROUND_C = "#12161c", "#1a212a"
AA = False
FAR = 0.5          # clip geometry closer than this: a wall vertex at z ~ 0.12
                   # projects to enormous coordinates and streaks a line clean
                   # across the frame, above the horizon
GFAR = 5.5         # and start the ground grid this far out.  Near-field grid
                   # lines sweep most of the panel between ticks, and this is
                   # the whole GIF budget: measured on this clip, the same
                   # lines cost 1.98 MB drawn from 0.5 m and 1.67 MB from
                   # 5.5 m.  Everything else (colour count, grid spacing,
                   # canvas size) moved the file by under 5 %.

CKPT = os.path.join(REPO, "runs", "fov1", "ckpt_32.8B_faithful78.pt")
OUT = os.path.join(REPO, "docs", "assets", "policy-eye-view.gif")
SIDECAR = os.path.join(REPO, "docs", "assets", "data", "policy-eye-view.json")

# canvas: 840 x 520, a 26 px header strip and a 24 px footer strip
W, H, HEAD_H, FOOT_H = 840, 520, 26, 24
EYE_H_PX = H - HEAD_H - FOOT_H                   # 470
AR = W / float(EYE_H_PX)                         # half-width in NDC, 1.787
HOLD = 16                                        # held frames on the K.O.


# --------------------------------------------------------------------------
def hexlerp(a, b, u):
    ca = np.array([int(a[i:i + 2], 16) for i in (1, 3, 5)], float)
    cb = np.array([int(b[i:i + 2], 16) for i in (1, 3, 5)], float)
    c = np.clip(ca + (cb - ca) * float(np.clip(u, 0.0, 1.0)), 0, 255)
    return "#%02x%02x%02x" % tuple(int(round(v)) for v in c)


def draw_eye(ax, d, t, cum, ko=False):
    """One frame of the first-person view, full canvas."""
    pos, yaw, pitch = d["pos"], d["yaw"], d["pitch"]
    hp, hurt, dmg, act = d["hp"], d["hurt"], d["dmg"], d["act"]
    eye = pos[t, 0].astype(np.float64) + np.array([0.0, EYE_H, 0.0])
    f, r, u = basis(yaw[t, 0], pitch[t, 0])
    ax.clear()
    ax.set_facecolor(SKY_C)
    ax.set_xlim(-AR, AR)
    ax.set_ylim(-1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    # ground: a FILLED plane below the y=0 plane's vanishing line, which is the
    # horizontal line at -tan(pitch)*focal.  Filled, never wireframe -- an
    # earlier wireframe cut of this clip cost 2.48 MB against 1.64 MB filled.
    yh = -np.tan(np.radians(pitch[t, 0])) * FOCAL
    ax.add_patch(Rectangle((-AR, -1.4), 2 * AR, min(yh, 1.4) + 1.4,
                           facecolor=GROUND_C, ec="none", zorder=0))
    ax.plot([-AR, AR], [yh, yh], color=HORIZ, lw=1.1, zorder=1,
            antialiased=AA)
    for g in np.arange(-12, 12.1, 4.0):          # sparse 4 m ground grid
        for a, b in ((np.array([-14., 0., g]), np.array([14., 0., g])),
                     (np.array([g, 0., -14.]), np.array([g, 0., 14.]))):
            pa, pb = a.copy(), b.copy()
            za, zb = (pa - eye) @ f, (pb - eye) @ f
            if za < GFAR and zb < GFAR:
                continue
            if za < GFAR:
                pa = pa + (GFAR - za) / (zb - za) * (pb - pa)
            elif zb < GFAR:
                pb = pb + (GFAR - zb) / (za - zb) * (pa - pb)
            P, _ = proj(np.stack([pa, pb]), eye, f, r, u)
            ax.plot(P[:, 0], P[:, 1], color=GRID_LINE, lw=1.0, zorder=2,
                    antialiased=AA)

    # arena wall: the 14x14 walled square the bot actually plays in
    # (arena_square=True, half-side 7.0), drawn as a filled band
    bot = arena_perimeter(ARENA_R)
    Pb, zb = proj(bot, eye, f, r, u)
    Pt, _ = proj(bot + np.array([0, 1.25, 0]), eye, f, r, u)
    ok = (zb > FAR) & (np.abs(Pb[:, 1]) < 6.0) & (np.abs(Pt[:, 1]) < 6.0)
    for i, j in visible_runs(ok):
        if j - i < 2:
            continue
        ax.add_patch(Polygon(np.concatenate([Pb[i:j + 1], Pt[i:j + 1][::-1]]),
                             closed=True, facecolor=WALL_FILL, ec="none",
                             zorder=3, antialiased=AA))
        ax.plot(Pt[i:j + 1, 0], Pt[i:j + 1, 1], color=WALLTOP, lw=1.2, zorder=4,
                antialiased=AA)

    # opponent: its true 0.6 x 1.8 m hitbox column plus a head cube at +1.25.
    # AMBER = a thing that exists independently of the machine; it reddens as
    # its HP fraction falls under 0.25 and goes solid RED on the hurt tick.
    ex, ey, ez = pos[t, 1]
    frac = float(hp[t, 1]) / 20.0
    col = AMBER if frac >= 0.25 else hexlerp(AMBER, RED, (0.25 - frac) / 0.25)
    if hurt[t, 1] >= 8:            # the first 3 ticks of the 10-tick hurt
        col = RED                  # window, so the entity stays amber between
    if ko:
        col = RED
    # solid faces, not a translucent prism: the union of the six faces is
    # exactly the box silhouette whatever the draw order, and a flat fill
    # quantises to one palette index instead of a stack of alpha blends
    edge = hexlerp(col, "#ffffff", 0.36)
    head = hexlerp(col, "#ffffff", 0.20)
    for c3, hw, h, fc in (((ex, ey, ez), 0.3, 1.8, col),
                          ((ex, ey + 1.25, ez), 0.26, 0.52, head)):
        P, z = proj(cube(*c3, hw, h), eye, f, r, u)
        if (z > NEAR).all():
            for q in CUBE_F:
                ax.add_patch(Polygon(P[list(q)], closed=True, facecolor=fc,
                                     ec="none", zorder=6))
            for a, b in CUBE_E:
                ax.plot(P[[a, b], 0], P[[a, b], 1], color=edge, lw=1.3,
                        zorder=7)
    Pn, zn = proj(np.array([[ex, ey + 2.15, ez]]), eye, f, r, u)
    Pn[0, 1] = min(Pn[0, 1], 0.70)      # keep it clear of the readout rows
    if zn[0] > NEAR and abs(Pn[0, 0]) < AR:
        ax.text(Pn[0, 0], Pn[0, 1], "P4-HACKER  %.1f" % max(hp[t, 1], 0.0),
                color=INK, fontsize=9.0, ha="center", family="monospace",
                zorder=8, bbox=dict(boxstyle="square,pad=0.28", fc="#0b0f15",
                                    ec="none", alpha=0.80))

    # landed hit: a white burst ring out of the column centre over 3 frames,
    # with the damage the step actually resolved
    for back in range(3):
        tt = t - back
        if tt >= 0 and dmg[tt, 1] > 0:
            Ph, zh = proj(np.array([[pos[tt, 1, 0], pos[tt, 1, 1] + 0.9,
                                     pos[tt, 1, 2]]]), eye, f, r, u)
            if zh[0] > NEAR:
                rr = (0.06 + 0.045 * back) * FOCAL / max(zh[0], 0.5)
                ax.add_patch(Circle((Ph[0, 0], Ph[0, 1]), rr, fill=False,
                                    ec=FLASH, lw=2.6 - 0.6 * back,
                                    alpha=max(0.0, 0.95 - 0.26 * back),
                                    zorder=8))
                ax.text(Ph[0, 0] + rr + 0.06, Ph[0, 1] + 0.02,
                        "-%.1f" % dmg[tt, 1], color=INK, fontsize=11.5,
                        ha="left", va="center", family="monospace",
                        weight="bold", alpha=max(0.0, 1.0 - 0.30 * back),
                        zorder=9, bbox=dict(boxstyle="square,pad=0.22",
                                            fc="#0b0f15", ec="none",
                                            alpha=0.72))

    # held sword, lower right: it lunges up-left toward the crosshair
    # on exactly the ticks the attack head fires, and snaps back the next frame
    swing = act[t, 0, 4] == 1 and not ko
    hx, hy, ang = ((AR - 0.24, -1.02, 122) if not swing
                   else (AR - 0.38, -0.90, 132))
    a0 = np.radians(ang)
    dxu, dyu = np.cos(a0), np.sin(a0)
    px, py = -dyu, dxu
    ax.plot([hx, hx + 0.14 * dxu], [hy, hy + 0.14 * dyu], color=HILT, lw=7.0,
            solid_capstyle="round", zorder=8)
    ax.plot([hx + 0.16 * dxu - 0.085 * px, hx + 0.16 * dxu + 0.085 * px],
            [hy + 0.16 * dyu - 0.085 * py, hy + 0.16 * dyu + 0.085 * py],
            color="#9aa7b4", lw=5.0, solid_capstyle="round", zorder=8)
    ax.plot([hx + 0.17 * dxu, hx + 0.60 * dxu],
            [hy + 0.17 * dyu, hy + 0.60 * dyu], color=BLADE, lw=6.5,
            solid_capstyle="round", zorder=8)
    if swing:
        ax.add_patch(Arc((hx, hy), 1.24, 1.24, angle=0, theta1=134, theta2=176,
                         ec=BLADE, lw=2.0, alpha=0.5, zorder=8))
        ax.text(AR - 0.06, -0.955, "ATTACK", color=BLADE, fontsize=9.0,
                ha="right", family="monospace", weight="bold", zorder=9,
                bbox=dict(boxstyle="square,pad=0.30", fc="#0b0f15",
                          ec="none", alpha=0.74))

    # self HP: ten hearts driven by env.hp[side 0]
    for i in range(10):
        v = hp[t, 0]
        c = RED if v >= (i + 1) * 2 - .01 else ("#8a2b34" if v > i * 2 + .01
                                                else "#242d38")
        ax.add_patch(Circle((-AR + 0.11 + i * 0.105, -0.870), 0.037,
                            facecolor=c, ec="none", zorder=9))
    ax.text(-AR + 0.075, -0.960, "SELF HP  %.1f / 20" % max(hp[t, 0], 0.0),
            color=DIM, fontsize=8.4, family="monospace", zorder=9,
            bbox=dict(boxstyle="square,pad=0.30", fc="#0b0f15", ec="none",
                      alpha=0.74))

    # crosshair, dead centre, dark-outlined so it survives over the amber column
    for xs, ys in (((-.062, -.020), (0, 0)), ((.020, .062), (0, 0)),
                   ((0, 0), (-.062, -.020)), ((0, 0), (.020, .062))):
        ax.plot(xs, ys, color="#0b0f15", lw=4.0, zorder=9,
                solid_capstyle="butt")
        ax.plot(xs, ys, color=BLADE, lw=1.8, zorder=9, solid_capstyle="butt")

    # one full-screen red frame on exactly the tick the policy takes damage
    if dmg[t, 0] > 0:
        ax.add_patch(Rectangle((-AR, -1), 2 * AR, 2, facecolor=RED, alpha=0.25,
                               zorder=10))

    # readouts.  Every HUD string gets a dark plate: at close range the
    # opponent's column reaches the top of the frame and would otherwise be
    # read through the text (it also freezes those pixels between frames).
    plate = dict(boxstyle="square,pad=0.30", fc="#0b0f15", ec="none",
                 alpha=0.74)
    gap = float(np.linalg.norm(pos[t, 1, [0, 2]] - pos[t, 0, [0, 2]]))
    ax.text(AR - 0.05, 0.905, "yaw %5.1f deg   pitch %5.1f deg   gap %.2f m"
            % (yaw[t, 0] % 360.0, pitch[t, 0], gap), color=INK, fontsize=8.4,
            ha="right", family="monospace", zorder=11, bbox=plate)
    ax.text(AR - 0.05, 0.820, "hits landed %2d  ·  taken %2d"
            % (cum[t, 0], cum[t, 1]), color=DIM, fontsize=8.4, ha="right",
            family="monospace", zorder=11, bbox=plate)
    if not ko:
        ax.text(-AR + 0.05, 0.820, "enemy = its true 0.6 x 1.8 m hitbox  ·  "
                "70 deg vertical FOV", color=DIM, fontsize=7.8,
                family="monospace", zorder=11, bbox=plate)

    # target off the edge of the frame (or behind the head): put a chevron on
    # the side the observation vector says it is on
    Pc, zc = proj(np.array([[ex, ey + 0.9, ez]]), eye, f, r, u)
    aim = float(d["obs0"][t, 22]) * 180.0
    if zc[0] <= NEAR or abs(Pc[0, 0]) > AR - 0.05:
        s = 1.0 if aim > 0 else -1.0
        xe = s * (AR - 0.11)
        ye = float(np.clip(Pc[0, 1] if zc[0] > NEAR else 0.0, -0.55, 0.55))
        ax.add_patch(Polygon([(xe + s * 0.085, ye), (xe - s * 0.045, ye + 0.075),
                              (xe - s * 0.045, ye - 0.075)], closed=True,
                             facecolor=AMBER, ec="none", zorder=11))
        ax.text(xe - s * 0.09, ye, "TARGET %+.0f deg" % aim, color=AMBER,
                fontsize=8.4, va="center", ha="right" if s > 0 else "left",
                family="monospace", zorder=11)

    # the moment, annotated on the frame
    tag, tcol = None, DIM
    if ko:
        tag = None
    elif dmg[t, 0] > 0:
        tag, tcol = "DAMAGE TAKEN  -%.1f" % dmg[t, 0], RED
    elif t >= 1 and dmg[t - 1, 1] > 0:
        tag, tcol = "HIT LANDED  -%.1f%s" % (
            dmg[t - 1, 1], "  CRIT x1.5" if dmg[t - 1, 1] > 2.0 else ""), GREEN
    elif dmg[t, 1] > 0:
        tag, tcol = "HIT LANDED  -%.1f%s" % (
            dmg[t, 1], "  CRIT x1.5" if dmg[t, 1] > 2.0 else ""), GREEN
    else:
        if abs(aim) > 25.0:
            tag, tcol = "ACQUIRING TARGET   aim err %+6.1f deg" % aim, CYAN
        else:
            tag, tcol = "TRACKING   aim err %+5.1f deg" % aim, DIM
    if tag:
        ax.text(-AR + 0.05, 0.905, tag, color=tcol, fontsize=8.8,
                family="monospace", weight="bold", zorder=11, bbox=plate)
    if ko:
        ax.text(0.0, 0.68, "K.O.", color=RED, fontsize=30, ha="center",
                va="center", family="monospace", weight="bold", zorder=12)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--colors", type=int, default=48)
    ap.add_argument("--cache", default=None,
                    help="npz to memoise the picked duel into (render "
                         "iteration only; the rollout itself takes ~4 s)")
    a = ap.parse_args()
    ckpt = a.ckpt if os.path.isabs(a.ckpt) else os.path.join(REPO, a.ckpt)
    if not os.path.exists(ckpt) and not (a.cache and os.path.exists(a.cache)):
        raise SystemExit(
            "source checkpoint no longer on disk: %s\n"
            "runs/ is gitignored, so a fresh clone cannot re-render this GIF."
            % os.path.relpath(ckpt, REPO))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    if a.cache and os.path.exists(a.cache):
        blob = np.load(a.cache)
        d = {k: blob[k] for k in blob.files if k != "_pick"}
        pick = {k: int(v) for k, v in zip(("seed", "env_index", "ticks"),
                                          blob["_pick"])}
        meta_step = int(blob["_pick"][3])
    else:
        import torch
        meta_step = int(torch.load(ckpt, map_location="cpu",
                                   weights_only=False)["meta"]["step"])
        # the four-panel generator returns either the duel dict or
        # (duel dict, provenance) depending on its revision
        res = pick_duel(ckpt)
        d, info = res if isinstance(res, tuple) else (res, {})
        pick = {"seed": int(info.get("seed", -1)),
                "env_index": int(info.get("env_index", -1)),
                "ticks": int(info.get("ticks", d["pos"].shape[0] - 1))}
        if a.cache:
            np.savez(a.cache, _pick=np.array(
                [pick["seed"], pick["env_index"], pick["ticks"], meta_step],
                np.int64), **{k: v for k, v in d.items()
                              if isinstance(v, np.ndarray)})

    n = d["pos"].shape[0]
    dmg = d["dmg"]
    # hits RESOLVED BEFORE this tick: the drawn state (hearts, nametag HP) is
    # the pre-step snapshot, so an inclusive count would run one hit ahead of it
    cum = np.stack([np.cumsum(dmg[:, 1] > 0), np.cumsum(dmg[:, 0] > 0)], 1)
    landed, taken = int(cum[-1, 0]), int(cum[-1, 1])
    cum = np.concatenate([np.zeros((1, 2), np.int64), cum[:-1]], 0)
    crits = int((dmg[:, 1] > 2.0).sum())

    prov = ("seed %d env %d" % (pick["seed"], pick["env_index"])
            if pick["seed"] >= 0 else "sim rollout")

    fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.0, FOOT_H / float(H), 1.0, EYE_H_PX / float(H)])
    ax.set_facecolor(SKY_C)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    # header / footer strips, drawn once so no label can shift by a pixel
    fig.patches.append(Rectangle((0.0, (H - HEAD_H) / float(H)), 1.0,
                                 HEAD_H / float(H), transform=fig.transFigure,
                                 facecolor=PANEL, ec="none", zorder=5))
    fig.patches.append(Rectangle((0.0, 0.0), 1.0, FOOT_H / float(H),
                                 transform=fig.transFigure, facecolor=PANEL,
                                 ec="none", zorder=5))
    hdr = fig.text(0.008, (H - HEAD_H / 2.0) / float(H), "", color=INK,
                   fontsize=9.6, va="center", family="monospace",
                   weight="bold", zorder=6)
    fig.text(0.992, (H - HEAD_H / 2.0) / float(H),
             "meta step %s · %s" % ("{:,}".format(meta_step), prov),
             color=DIM, fontsize=8.4, va="center", ha="right",
             family="monospace", zorder=6)
    fig.text(0.008, FOOT_H / 2.0 / float(H),
             "ckpt_32.8B_faithful78 vs P4-Hacker · 14x14 square arena · "
             "20 fps = 1 frame per 50 ms tick", color=DIM, fontsize=8.0,
             va="center", family="monospace", zorder=6)
    fig.text(0.992, FOOT_H / 2.0 / float(H),
             "rendered %s" % _dt.date.today().isoformat(), color=DIM,
             fontsize=8.0, va="center", ha="right", family="monospace",
             zorder=6)

    frames = []
    for t in list(range(n)) + [n - 1] * HOLD:
        hdr.set_text("M · POLICY EYE VIEW · tick %04d · t=%5.2f s"
                     % (t, t / 20.0))
        draw_eye(ax, d, t, cum, ko=(t == n - 1))
        fig.canvas.draw()
        f = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        b = np.array([int(GRID[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)
        f[0, :] = f[-1, :] = b                    # 1 px border into the frame
        f[:, 0] = f[:, -1] = b
        frames.append(f)

    # median-cut on the montage alone drops rare-but-load-bearing colours (the
    # steel blade is a few hundred px/frame and comes out khaki); pin them.
    anchors = [FLASH, BLADE, INK, HILT, CYAN, AMBER, RED, GREEN, BG, PANEL,
               SKY_C, GROUND_C, GRID, GRID_LINE, WALL_FILL, DIM, HORIZ, WALLTOP]
    rgb = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in anchors],
                   np.uint8)
    swatch = np.zeros((24, W, 3), np.uint8)
    step = W // len(anchors)
    for i, c in enumerate(rgb):
        swatch[:, i * step:(i + 1) * step] = c
    swatch[:, len(rgb) * step:] = rgb[-1]
    montage = np.concatenate(frames[::7] + [swatch], 0)
    pal = Image.fromarray(montage).quantize(colors=a.colors,
                                            method=Image.MEDIANCUT)
    qs = [Image.fromarray(f).quantize(palette=pal, dither=Image.NONE)
          for f in frames]
    # disposal=1 lets PIL emit inter-frame diffs, and PIL collapses the held
    # K.O. run into one long-duration frame, so the ending is nearly free.
    qs[0].save(a.out, save_all=True, append_images=qs[1:], duration=50, loop=0,
               optimize=True, disposal=1)
    size_mb = os.path.getsize(a.out) / 1e6

    os.makedirs(os.path.dirname(SIDECAR), exist_ok=True)
    with open(SIDECAR, "w") as fh:
        json.dump({
            "id": "policy-eye-view",
            "output": os.path.relpath(a.out, REPO),
            "rendered": _dt.date.today().isoformat(),
            "checkpoint": os.path.relpath(ckpt, REPO),
            "checkpoint_meta_step": meta_step,
            "opponent": "pvpbot.eval.practice.PracticeHacker(13)",
            "env": "pvpbot.sim.env.DuelVecEnv, SimConfig(arena_square=True, arena_radius=7.0)",
            "rollout": {"seed": pick["seed"], "env_index": pick["env_index"],
                        "duel_ticks": n - 1, "synthetic_post_kill_ticks": 1,
                        "held_frames": HOLD},
            "frames": len(qs), "fps": 20, "ms_per_frame": 50,
            "seconds": round(len(qs) / 20.0, 2),
            "width_px": W, "height_px": H, "colors": a.colors,
            "size_mb": round(size_mb, 3),
            "hits_landed_by_policy": landed, "hits_taken_by_policy": taken,
            "critical_hits_landed": crits,
            "policy_hp_at_kill": round(float(d["hp"][-1, 0]), 1),
            "opponent_hp_start": 20.0,
            "attack_head_fired_ticks": int((d["act"][:n - 1, 0, 4] == 1).sum()),
            "gap_min_m": round(float(np.min(np.linalg.norm(
                d["pos"][:, 1, [0, 2]] - d["pos"][:, 0, [0, 2]], axis=1))), 2),
            "gap_max_m": round(float(np.max(np.linalg.norm(
                d["pos"][:, 1, [0, 2]] - d["pos"][:, 0, [0, 2]], axis=1))), 2),
        }, fh, indent=2)
        fh.write("\n")

    print("%s  %d frames  %dx%d  %.2f MB  |  landed %d (crit %d), taken %d, "
          "policy ends %.1f hp"
          % (os.path.relpath(a.out, REPO), len(qs), W, H, size_mb, landed,
             crits, taken, d["hp"][-1, 0]))


if __name__ == "__main__":
    main()
