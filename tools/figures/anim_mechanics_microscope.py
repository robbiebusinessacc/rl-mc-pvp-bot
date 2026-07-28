"""Two 1.8 combat rules under a microscope, one GIF frame per game tick.

Renders docs/assets/mechanics-microscope.gif -- a 5 fps clip (duration=200,
one frame per 50 ms Minecraft tick, so 4x slower than real time) in two acts:

  ACT 1  CRITICAL HIT     Two DuelVecEnv envs stepped in lockstep.  Both
                          attackers stand at the same spot, aim at the same
                          target through the same 11-bin camera head, and
                          swing on the same tick.  Env 1 jumps at tick 7, so
                          on the swing tick it is airborne with vel_y < 0 and
                          its hit is worth x1.5.  Nothing else differs.
  ACT 2  HURT WINDOW      Both attackers land a normal 1.6 hit at tick 4,
                          opening the 10-tick invulnerability window, then
                          swing again at tick 10 while the window is still
                          open.  The grounded swing connects and deals 0.0.
                          The falling swing is worth 2.4 against a window
                          opened by 1.6, so 0.8 -- exactly the excess -- goes
                          through, and the window is NOT re-armed.

Everything drawn is real state pulled off a live DuelVecEnv instance after
each step(): pos, vel, yaw, pitch, hp, hurt, on_ground, last_dmg.  Every
constant in the labels is read off the same instance (env._dmg_base 1.6,
env._dmg_crit 2.4, env._hurt_ticks 10, env._reach 3.0, env._eye_h 1.62,
env._kb_h/_kb_v 0.4, env._jump_v 0.42, env._gravity 0.08), never hard-coded.

Source of truth for the rules being shown:
  crit         pvpbot/sim/env.py:677    crit = (~on_ground) & (vel_y < 0.0)
  hurt window  pvpbot/sim/env.py:671    fresh / excess / no re-arm

There is no gitignored input: the animation is generated from pvpbot/sim/env.py
itself, so this script runs on a fresh clone.

Usage:  python3 tools/figures/anim_mechanics_microscope.py
        [--out docs/assets/mechanics-microscope.gif] [--colors 96]
"""
import argparse
import datetime as _dt
import io
import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

try:
    from pvpbot.sim.env import DuelVecEnv, SimConfig
    from pvpbot.spec import CAMERA_BINS, OBS_LAYOUT
except Exception as exc:  # pragma: no cover - fresh-clone guard
    sys.exit(
        "cannot import the simulator, which is this animation's only source: "
        "expected pvpbot/sim/env.py under %s (%s)" % (REPO, exc)
    )

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from PIL import Image

# ---------------------------------------------------------------------------
# locked palette -- identical literals to tools/figures/anim_live_cockpit.py
# ---------------------------------------------------------------------------
BG, PANEL, GRID, INK, DIM = "#12161c", "#1a212a", "#2b3542", "#ccd6e2", "#66747f"
CYAN, AMBER, RED, GREEN, VIOLET = "#3fd0d8", "#e8a33d", "#e2564a", "#6ecf94", "#a184e8"
BRAND = (BG, PANEL, GRID, INK, DIM, CYAN, AMBER, RED, GREEN, VIOLET)

MONO = "Menlo"
matplotlib.rcParams["font.family"] = MONO
matplotlib.rcParams["font.monospace"] = [MONO, "DejaVu Sans Mono"]

W_PX, H_PX = 1100, 600
DPI = 100

FWD, STR, JMP, SPR, ATK, YAW, PIT = range(7)
BINS = np.array(CAMERA_BINS, dtype=np.float64)
NBIN = len(CAMERA_BINS)

# panel geometry (figure coords); axes are 462 x 308 px, so 1.5:1 and the
# x/y limits below are exactly 5.8 x 3.8667 blocks -> set_aspect('equal') is
# a no-op and no panel can shift by a pixel between frames.
AX_RECTS = ([0.050, 0.335, 0.420, 0.51333], [0.525, 0.335, 0.420, 0.51333])
YLIM = (-0.35, 3.51667)
XLIM_ACT1 = (-3.10, 2.70)
XLIM_ACT2 = (-2.50, 3.30)

Y_HDR_1, Y_HDR_2 = 0.963, 0.927
Y_ONLY = 0.897
Y_TITLE = 0.872
Y_HP, Y_WIN = 0.2933, 0.2433
Y_ROW, Y_RC = 0.1867, 0.1233
Y_DMG = 0.0667
Y_FOOT = 0.0135
# readout column offsets inside a panel (Menlo advance = 0.6 em, so a 7.0 pt
# key runs 0.0053 figure-widths per character)
C1, C2, C3, C4 = 0.000, 0.100, 0.205, 0.318


# ---------------------------------------------------------------------------
# scenario driver
# ---------------------------------------------------------------------------
def _place(env, ax_x, tx_x):
    """Overwrite every piece of episode state after reset()'s randomisation."""
    env.pos[:] = 0.0
    env.pos[:, 0, 0] = ax_x
    env.pos[:, 1, 0] = tx_x
    env.vel[:] = 0.0
    env.yaw[:, 0] = 0.0
    env.yaw[:, 1] = 180.0
    env.pitch[:] = 0.0
    env.hp[:] = 20.0
    env.hurt[:] = 0
    env.on_ground[:] = True
    env.sprinting[:] = False
    env.sprint_blocked[:] = False
    env.last_dmg[:] = 0.0
    env.since_dealt[:] = 100.0
    env.since_taken[:] = 100.0
    env.ticks[:] = 0
    env.prev_actions[:] = 0
    return env._obs()


def _connects(env, e):
    """Exact ray-vs-AABB reach test, mirroring pvpbot/sim/env.py:615-665.

    Returns True when env ``e``'s attacker crosshair ray intersects the
    target's 0.6 x 1.8 box within reach -- i.e. when a swing on this tick
    would register, whether or not it deals damage.
    """
    a = env.pos[e, 0]
    t = env.pos[e, 1]
    dx, dz = float(t[0] - a[0]), float(t[2] - a[2])
    eye_y = float(a[1]) + env._eye_h
    ax_ = max(abs(dx) - env._half_w, 0.0)
    az_ = max(abs(dz) - env._half_w, 0.0)
    ay_ = max(max(float(t[1]) - eye_y, eye_y - (float(t[1]) + env._box_h)), 0.0)
    if math.sqrt(ax_ * ax_ + ay_ * ay_ + az_ * az_) > env._reach:
        return False
    yr = math.radians(float(env.yaw[e, 0]))
    pr = math.radians(float(env.pitch[e, 0]))
    v = (math.cos(pr) * math.cos(yr), math.sin(pr), math.cos(pr) * math.sin(yr))
    oy = float(t[1]) - eye_y
    off = (dx, oy + 0.5 * env._box_h, dz)
    half = (env._click_hw, 0.5 * env._box_h + env._click_b, env._click_hw)
    tmin, tmax = -np.inf, np.inf
    for vi, oi, hi in zip(v, off, half):
        if abs(vi) < 1e-8:
            if abs(oi) > hi + 1e-6:
                return False
            continue
        t0, t1 = (oi - hi) / vi, (oi + hi) / vi
        if t0 > t1:
            t0, t1 = t1, t0
        tmin, tmax = max(tmin, t0), min(tmax, t1)
    return bool(tmax >= max(tmin, 0.0) and tmin <= env._reach)


def _reach_dist(env, e):
    a, t = env.pos[e, 0], env.pos[e, 1]
    dx, dz = float(t[0] - a[0]), float(t[2] - a[2])
    eye_y = float(a[1]) + env._eye_h
    ax_ = max(abs(dx) - env._half_w, 0.0)
    az_ = max(abs(dz) - env._half_w, 0.0)
    ay_ = max(max(float(t[1]) - eye_y, eye_y - (float(t[1]) + env._box_h)), 0.0)
    return math.sqrt(ax_ * ax_ + ay_ * ay_ + az_ * az_)


def run_scenario(attacker_x, target_x, nticks, jumps, swings):
    """Step ONE DuelVecEnv(2) so both panels share a clock; env 0 = variant A,
    env 1 = variant B.  Every frame records POST-step state, which is exactly
    the state that tick's combat resolution saw (camera and movement run
    before combat; only velocities change afterwards)."""
    env = DuelVecEnv(2, seed=0, config=SimConfig())
    env.reset()
    obs = _place(env, attacker_x, target_x)
    frames = []
    for t in range(nticks):
        a = np.zeros((2, 2, 7), dtype=np.int64)
        a[:, :, FWD] = 1
        a[:, :, STR] = 1
        a[:, :, YAW] = a[:, :, PIT] = int(np.argmin(np.abs(BINS)))
        # aim exactly as pvpbot/eval/scripted.py _aim does: snap the residual
        # error from the observation vector onto the nearest of 11 camera bins
        ey = obs[:, 0, OBS_LAYOUT["aim_err_yaw"][0]] * 180.0
        ep = obs[:, 0, OBS_LAYOUT["aim_err_pitch"][0]] * 90.0
        a[:, 0, YAW] = np.abs(BINS[None, :] - ey[:, None]).argmin(1)
        a[:, 0, PIT] = np.abs(BINS[None, :] - ep[:, None]).argmin(1)
        for e, ts in jumps.items():
            if t in ts:
                a[e, 0, JMP] = 1
        for e, ts in swings.items():
            if t in ts:
                a[e, 0, ATK] = 1
        hp_pre = env.hp.copy()
        obs, _r, _d, _i = env.step(a)
        frames.append(dict(
            tick=t,
            pos=env.pos.copy(), pitch=env.pitch.copy(), hp=env.hp.copy(),
            hurt=env.hurt.copy(), og=env.on_ground.copy(),
            vy=env.vel[:, :, 1].copy(), last=env.last_dmg.copy(),
            dmg=(hp_pre - env.hp)[:, 1].copy(),
            swing=a[:, 0, ATK].copy(), pit_bin=a[:, 0, PIT].copy(),
            hit=np.array([_connects(env, 0), _connects(env, 1)]),
            rd=np.array([_reach_dist(env, 0), _reach_dist(env, 1)]),
            ring=np.full(2, -1, dtype=np.int64),
            ring_xy=np.zeros((2, 2), dtype=np.float64),
        ))
    # 3-tick fading marker pinned to the world point where a swing connected,
    # so the knockback is visible as the target leaving that point
    for e in (0, 1):
        age, xy = 99, (0.0, 0.0)
        for f in frames:
            if f["swing"][e] and f["hit"][e]:
                age = 0
                xy = (float(f["pos"][e, 1, 0]),
                      float(f["pos"][e, 1, 1]) + 1.62 - 0.72)
            if age <= 2:
                f["ring"][e] = age
                f["ring_xy"][e] = xy
            age += 1
    return env, frames


# ---------------------------------------------------------------------------
# drawing helpers (all figure-coordinate, so nothing jitters between frames)
# ---------------------------------------------------------------------------
def fig_rect(fig, x, y, w, h, fc, ec=None, lw=0.8, z=1, alpha=1.0):
    r = Rectangle((x, y), w, h, transform=fig.transFigure, facecolor=fc,
                  edgecolor=ec or "none", linewidth=lw, zorder=z, alpha=alpha)
    fig.add_artist(r)
    return r


def key_val(fig, x, y, key, val, vcol=INK, ksize=6.6, vsize=10.0):
    fig.text(x, y + 0.0155, key, color=DIM, fontsize=ksize)
    fig.text(x, y - 0.008, val, color=vcol, fontsize=vsize, fontweight="bold")


def seg_bar(fig, x, y, n_lit, n=10, w=0.0118, h=0.019, col=RED):
    for i in range(n):
        on = i < n_lit
        fig_rect(fig, x + i * (w + 0.0020), y, w, h,
                 col if on else "#20262f", col if on else GRID, 0.6, z=3)


def draw_scene(fig, rect, xlim, S, e, dims):
    ax = fig.add_axes(rect)
    ax.set_facecolor(PANEL)
    ax.set_xlim(*xlim)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal", adjustable="box")
    for sp in ax.spines.values():
        sp.set_color(GRID)
        sp.set_linewidth(0.9)
    xt = np.arange(math.ceil(xlim[0]), math.floor(xlim[1]) + 1, 1.0)
    ax.set_xticks(xt)
    ax.set_xticklabels([])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["0", "1", "2", "3 m"])
    ax.tick_params(colors=DIM, labelsize=6.6, length=2.5, width=0.7, pad=2.0)
    ax.grid(True, color="#232b35", lw=0.6, zorder=0)
    # x ruler drawn INSIDE the sub-floor band so it cannot collide with the
    # readout rows below the panel
    for xv in xt:
        ax.text(xv, -0.215, "%d" % xv, color=DIM, fontsize=6.2, ha="center",
                va="center", zorder=3)

    # ground: solid line at y=0 plus a hatched sub-floor band
    ax.add_patch(Rectangle((xlim[0], YLIM[0]), xlim[1] - xlim[0], -YLIM[0],
                           facecolor="#161c24", edgecolor="none", zorder=1,
                           hatch="////"))
    ax.add_patch(Rectangle((xlim[0], YLIM[0]), xlim[1] - xlim[0], -YLIM[0],
                           facecolor="none", edgecolor=GRID, lw=0.0,
                           hatch="////", zorder=1))
    ax.plot(xlim, [0, 0], color=GRID, lw=1.6, zorder=2)

    axp, typ = S["pos"][e, 0], S["pos"][e, 1]
    hw, bh, eh = 0.3, 1.8, 1.62
    hurting = S["hurt"][e, 1] > 0
    took = S["dmg"][e] > 1e-6

    # target (AMBER = exists independently of the machine)
    tcol = RED if took else AMBER
    ax.add_patch(Rectangle((typ[0] - hw, typ[1]), 2 * hw, bh, facecolor=tcol,
                           alpha=0.34 if took else 0.20, edgecolor=tcol,
                           lw=1.6, zorder=6))
    ax.plot([typ[0]], [typ[1] + eh], marker="o", ms=3.4, color=tcol, zorder=7)
    if hurting:
        ax.add_patch(Rectangle((typ[0] - hw - 0.07, typ[1] - 0.07),
                               2 * hw + 0.14, bh + 0.14, facecolor="none",
                               edgecolor=RED, lw=1.1, ls=(0, (2, 2)), zorder=7))

    # attacker (CYAN)
    ax.add_patch(Rectangle((axp[0] - hw, axp[1]), 2 * hw, bh, facecolor=CYAN,
                           alpha=0.20, edgecolor=CYAN, lw=1.6, zorder=6))
    eye = (axp[0], axp[1] + eh)
    ax.plot([eye[0]], [eye[1]], marker="o", ms=4.0, color=CYAN, zorder=8)

    # crosshair ray, length = reach 3.0, along (cos pitch, sin pitch)
    pr = math.radians(float(S["pitch"][e, 0]))
    end = (eye[0] + 3.0 * math.cos(pr), eye[1] + 3.0 * math.sin(pr))
    connected = bool(S["swing"][e]) and bool(S["hit"][e])
    if connected:
        ax.plot([eye[0], end[0]], [eye[1], end[1]], color=GREEN, lw=2.4,
                zorder=9, solid_capstyle="round")
        cy = typ[1] + eh - 0.72
        ax.plot([typ[0]], [cy], marker="o", ms=14, mfc="none", mec=GREEN,
                mew=2.0, zorder=10)
        ax.text(typ[0] + 0.44, cy + 0.30, "SWING CONNECTS", color=GREEN,
                fontsize=7.8, fontweight="bold", ha="left", va="center",
                zorder=10)
        ax.text(typ[0] + 0.44, cy + 0.09, "%.2f m ≤ reach 3.0" % S["rd"][e],
                color=GREEN, fontsize=6.9, ha="left", va="center", zorder=10)
    else:
        ax.plot([eye[0], end[0]], [eye[1], end[1]], color=DIM, lw=1.2,
                ls=(0, (3, 3)), zorder=5)
    age = int(S["ring"][e])
    if 0 <= age <= 2:
        ax.plot([S["ring_xy"][e, 0]], [S["ring_xy"][e, 1]], marker="o",
                ms=14 + 11 * age, mfc="none", mec=GREEN, mew=2.0 - 0.5 * age,
                alpha=1.0 - 0.32 * age, zorder=10)
    # reach cap at the end of the 3.0 m crosshair ray
    ax.plot([end[0]], [end[1]], marker=(2, 0, math.degrees(pr)), ms=6,
            color=GREEN if connected else DIM, mew=1.4, zorder=5)

    # 11-bin camera head, drawn inside the panel where the aiming happens
    lit = int(S["pit_bin"][e])
    for i in range(NBIN):
        on = i == lit
        ax.add_patch(Rectangle((0.755 + i * 0.0193, 0.905), 0.0165, 0.045,
                               transform=ax.transAxes, zorder=9,
                               facecolor=CYAN if on else PANEL,
                               edgecolor=CYAN if on else GRID, lw=0.6))
    ax.text(0.967, 0.958, "pitch %+.1f°  ·  head bin %+g°/tick"
            % (float(S["pitch"][e, 0]), CAMERA_BINS[lit]), color=DIM,
            fontsize=7.0, ha="right", va="bottom", transform=ax.transAxes,
            zorder=9)

    if not S["og"][e, 0]:
        # ghost floor line + a vel_y arrow on the free side of the box
        ax.plot([axp[0] - hw, axp[0] + hw], [0, 0], color=CYAN, lw=0.9,
                ls=(0, (1, 2)), zorder=6)
        vy = float(S["vy"][e, 0])
        sgn = 1.0 if vy >= 0 else -1.0
        xa = axp[0] - hw - 0.20
        ax.annotate("", xy=(xa, axp[1] + 0.90 + 0.42 * sgn),
                    xytext=(xa, axp[1] + 0.90 - 0.42 * sgn),
                    arrowprops=dict(arrowstyle="-|>", color=CYAN, lw=1.4,
                                    shrinkA=0, shrinkB=0), zorder=8)

    if dims:
        # true-scale key: panel 0's attacker never leaves the floor, so the
        # dimension marks below never move and never cross the ray
        xd = axp[0] - hw - 0.22
        ax.annotate("", xy=(xd, axp[1]), xytext=(xd, axp[1] + bh),
                    arrowprops=dict(arrowstyle="<->", color=DIM, lw=0.9,
                                    shrinkA=0, shrinkB=0), zorder=4)
        ax.annotate("", xy=(axp[0] - hw, axp[1] + bh + 0.13),
                    xytext=(axp[0] + hw, axp[1] + bh + 0.13),
                    arrowprops=dict(arrowstyle="<->", color=DIM, lw=0.9,
                                    shrinkA=0, shrinkB=0), zorder=4)
        ax.plot([axp[0] - hw - 0.14, axp[0] + hw + 0.14],
                [axp[1] + eh] * 2, color=DIM, lw=0.7, ls=(0, (2, 2)), zorder=4)
        for j, s in enumerate((
                "TRUE SCALE  ·  1 m grid",
                "hitbox  0.6 x 1.8 m",
                "eye  1.62 m   (dashed)",
                "crosshair ray  =  3.0 m reach")):
            ax.text(0.022, 0.962 - j * 0.045, s,
                    color=INK if j == 0 else DIM, fontsize=6.9,
                    fontweight="bold" if j == 0 else "normal",
                    ha="left", va="center", transform=ax.transAxes, zorder=9)
    return ax


# ---------------------------------------------------------------------------
def chrome(fig, act, tick, hdr2, only_line, foot):
    fig.patch.set_facecolor(BG)
    fig_rect(fig, 0, 0.9133, 1, 0.0867, PANEL, z=0)
    fig_rect(fig, 0, 0.9128, 1, 0.0006, GRID, z=1)
    fig_rect(fig, 0, 0, 1, 0.040, PANEL, z=0)

    clock = ("" if tick is None else
             "  ·  tick %02d  ·  t=%.2f s" % (tick, tick * 0.05))
    fig.text(0.022, Y_HDR_1, "STAGE M  ·  DuelVecEnv  ·  %s%s" % (act, clock),
             color=INK, fontsize=9.6, fontweight="bold", va="center")
    fig.text(0.022, Y_HDR_2, hdr2, color=AMBER, fontsize=9.8,
             fontweight="bold", va="center")

    fig.add_artist(FancyBboxPatch((0.752, 0.9315), 0.222, 0.0345,
                                  boxstyle="round,pad=0.004,rounding_size=0.006",
                                  transform=fig.transFigure, facecolor="#20262f",
                                  edgecolor=CYAN, lw=0.9, zorder=2))
    fig.text(0.863, 0.9487, "1 FRAME = 1 TICK  ·  4x SLOWER", color=CYAN,
             fontsize=8.6, fontweight="bold", ha="center", va="center", zorder=3)

    fig.text(0.5, Y_ONLY, only_line, color=INK, fontsize=9.4, ha="center",
             va="center", fontweight="bold")
    fig.text(0.022, Y_FOOT, foot, color=DIM, fontsize=6.5, va="center")


def draw_act(fig, act, S, cfg):
    fig.clf()
    tick = S["tick"]
    chrome(fig, cfg["act_name"], tick, cfg["hdr2"],
           cfg["only"](S), cfg["foot"])

    for e in (0, 1):
        rect = AX_RECTS[e]
        x0, wd = rect[0], rect[2]
        draw_scene(fig, rect, cfg["xlim"], S, e, dims=(e == 0))
        fig.text(x0, Y_TITLE, cfg["titles"][e], color=INK, fontsize=10.2,
                 fontweight="bold", va="center")

        # --- target HP -----------------------------------------------------
        hp = float(S["hp"][e, 1])
        lost = 20.0 - hp
        fig.text(x0, Y_HP, "TARGET HP", color=DIM, fontsize=7.4, va="center")
        bx, bw = x0 + 0.062, 0.190
        fig_rect(fig, bx, Y_HP - 0.0105, bw, 0.021, "#20262f", GRID, 0.7, z=2)
        fig_rect(fig, bx, Y_HP - 0.0105, bw * hp / 20.0, 0.021, AMBER, z=3)
        if lost > 1e-6:
            fig_rect(fig, bx + bw * hp / 20.0, Y_HP - 0.0105,
                     bw * lost / 20.0, 0.021, RED, z=3)
        fig.text(bx + bw + 0.009, Y_HP, "%5.2f / 20" % hp, color=AMBER,
                 fontsize=9.4, fontweight="bold", va="center")
        if lost > 1e-6:
            fig.text(bx + bw + 0.092, Y_HP, "−%.2f" % lost, color=RED,
                     fontsize=9.4, fontweight="bold", va="center")

        # --- 10-tick hurt window -------------------------------------------
        hu = int(S["hurt"][e, 1])
        fig.text(x0, Y_WIN, "HURT WINDOW", color=DIM, fontsize=7.4, va="center")
        seg_bar(fig, x0 + 0.068, Y_WIN - 0.0095, hu)
        fig.text(x0 + 0.068 + 10 * 0.0138 + 0.007, Y_WIN, "%d" % hu,
                 color=RED if hu else DIM, fontsize=9.4, fontweight="bold",
                 va="center")
        fig.text(x0 + 0.252, Y_WIN, "lastDamage", color=DIM, fontsize=7.4,
                 va="center")
        fig.text(x0 + 0.318, Y_WIN, "%.1f" % float(S["last"][e, 1]),
                 color=INK if S["last"][e, 1] else DIM, fontsize=9.4,
                 fontweight="bold", va="center")

        # --- readouts (one row, four columns) ------------------------------
        og = bool(S["og"][e, 0])
        vy = float(S["vy"][e, 0])
        falling = (not og) and vy < 0.0
        sw = bool(S["swing"][e])
        key_val(fig, x0 + C1, Y_ROW, "on_ground",
                "1" if og else "0  AIRBORNE", INK if og else CYAN)
        key_val(fig, x0 + C2, Y_ROW, "vel_y after gravity", "%+.3f" % vy,
                RED if falling else INK)
        key_val(fig, x0 + C3, Y_ROW, "eye→box  reach 3.0 m",
                "%.2f m" % float(S["rd"][e]),
                GREEN if S["rd"][e] <= 3.0 else DIM)
        key_val(fig, x0 + C4, Y_ROW, "attack head",
                "1  SWING" if sw else "0", GREEN if sw else DIM)

        rc_txt, rc_col = cfg["rowc"](S, e)
        fig.text(x0, Y_RC, rc_txt, color=rc_col, fontsize=8.4, va="center",
                 fontweight="bold")

        for dx, txt, col, sz in cfg["dmg"](S, e):
            fig.text(x0 + dx, Y_DMG, txt, color=col, fontsize=sz,
                     fontweight="bold", va="center")


# ---------------------------------------------------------------------------
def title_card(fig, foot):
    fig.clf()
    chrome(fig, "ACT 2 — THE 10-TICK HURT WINDOW", None,
           "hurt_ticks = 10  ·  a hit inside the window is not simply ignored",
           "", foot)
    fig.text(0.5, 0.560, "ACT 2 — THE 10-TICK HURT WINDOW", color=INK,
             fontsize=26, fontweight="bold", ha="center", va="center")
    fig.text(0.5, 0.470,
             "a swing inside the window deals only its EXCESS over lastDamage",
             color=AMBER, fontsize=13.5, ha="center", va="center")
    fig.text(0.5, 0.415, "— and does not re-arm it —", color=DIM,
             fontsize=11.5, ha="center", va="center")
    fig_rect(fig, 0.34, 0.345, 0.32, 0.0035, RED, z=2)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        REPO, "docs/assets/mechanics-microscope.gif"))
    ap.add_argument("--colors", type=int, default=96)
    ap.add_argument("--dump", default=None, help="also write PNG frames here")
    args = ap.parse_args()

    np.random.seed(0)
    env1, A1 = run_scenario(-2.2, 0.0, 30, {1: [7]}, {0: [12], 1: [12]})
    env2, A2 = run_scenario(-1.5, 0.0, 21, {1: [5]}, {0: [4, 10], 1: [4, 10]})
    E = env1  # constants are identical on both instances

    stamp = _dt.date.today().isoformat()
    FOOT = ("DuelVecEnv(2, seed=0, SimConfig()) · hand-placed state, both envs "
            "in lockstep · pvpbot/sim/env.py:677 (crit) · pvpbot/sim/env.py:671 "
            "(hurt window) · rendered " + stamp)

    dmg_base, dmg_crit = float(E._dmg_base), float(E._dmg_crit)
    hurt_ticks, reach = int(E._hurt_ticks), float(E._reach)

    # ---- ACT 1 ------------------------------------------------------------
    def a1_only(S):
        if S["tick"] >= 22:
            return ("SAME swing tick, SAME reach, SAME target — %.1f vs %.1f "
                    "is 0.8 damage from vel_y alone" % (dmg_crit, dmg_base))
        return ("ONLY DIFFERENCE: whether the attacker is falling on the swing "
                "tick — everything else is identical")

    def a1_rowc(S, e):
        og, vy = bool(S["og"][e, 0]), float(S["vy"][e, 0])
        if (not og) and vy < 0.0:
            return ("crit test  (~on_ground) & (vel_y < 0)  →  TRUE  "
                    "·  x1.5", RED)
        return ("crit test  (~on_ground) & (vel_y < 0)  →  false  "
                "·  x1.0", DIM)

    def a1_dmg(S, e):
        if S["tick"] < 12:
            return []
        if e == 0:
            return [(0.0, "tick 12   −%.1f   normal hit" % dmg_base, INK, 12.6)]
        return [(0.0, "tick 12   −%.1f   CRIT x1.5" % dmg_crit, RED, 12.6)]

    cfg1 = dict(
        act_name="ACT 1 — CRITICAL HIT",
        hdr2=("8.0 sword damage x (1 − 20 armor points x 4%%) = %.1f normal"
              "     ·     x1.5 crit = %.1f" % (dmg_base, dmg_crit)),
        only=a1_only, rowc=a1_rowc, dmg=a1_dmg, foot=FOOT, xlim=XLIM_ACT1,
        titles=("ENV 0 — grounded attacker, swings at tick 12",
                "ENV 1 — jumps at tick 7, swings at tick 12"),
    )

    # ---- ACT 2 ------------------------------------------------------------
    def a2_only(S):
        if S["tick"] >= 15:
            return ("the window closed on schedule — the excess hit never "
                    "re-armed it")
        if S["tick"] >= 10:
            return ("BOTH swings connect at tick 10, %d ticks into a window "
                    "opened by %.1f" % (hurt_ticks - int(S["hurt"][0, 1]), dmg_base))
        return ("both attackers land a normal %.1f at tick 4, opening the "
                "%d-tick window" % (dmg_base, hurt_ticks))

    def a2_rowc(S, e):
        if S["tick"] < 10:
            return ("window open · a swing worth <= lastDamage is absorbed "
                    "whole", DIM)
        if e == 0:
            return ("%.1f > lastDamage %.1f ?  false  →  0.0 through"
                    % (dmg_base, dmg_base), DIM)
        return ("%.1f > lastDamage %.1f ?  TRUE  →  %.1f − %.1f = 0.8 "
                "through" % (dmg_crit, dmg_base, dmg_crit, dmg_base), RED)

    def a2_dmg(S, e):
        if S["tick"] < 4:
            return []
        if S["tick"] < 10:
            return [(0.0, "tick 4   −%.1f   fresh hit" % dmg_base, INK, 12.6)]
        first = (0.0, "tick 4  −%.1f" % dmg_base, DIM, 9.5)
        if e == 0:
            return [first, (0.105, "tick 10  −0.0  ABSORBED", DIM, 12.6)]
        return [first, (0.105, "tick 10  −0.8  EXCESS", RED, 12.6)]

    cfg2 = dict(
        act_name="ACT 2 — HURT WINDOW",
        hdr2=("a hit inside the %d-tick window deals only its EXCESS over "
              "lastDamage · %.1f − %.1f = 0.8" % (hurt_ticks, dmg_crit,
                                                            dmg_base)),
        only=a2_only, rowc=a2_rowc, dmg=a2_dmg, foot=FOOT, xlim=XLIM_ACT2,
        titles=("ENV 0 — grounded for both swings",
                "ENV 1 — jumps at 5, falling at tick 10"),
    )

    # ---- render -----------------------------------------------------------
    fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI)
    raw = []

    def grab():
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=BG, dpi=DPI)
        buf.seek(0)
        raw.append(Image.open(buf).convert("RGB"))

    for S in A1:
        draw_act(fig, 1, S, cfg1)
        grab()
    for _ in range(4):
        title_card(fig, FOOT)
        grab()
    for S in A2:
        draw_act(fig, 2, S, cfg2)
        grab()
    plt.close(fig)

    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
        for i, im in enumerate(raw):
            im.save(os.path.join(args.dump, "f_%03d.png" % i))

    # ONE global palette: median-cut the raw frames for (colors - 10) entries,
    # then FORCE the ten brand hexes in as the last ten.  A swatch strip is not
    # enough on its own -- measured, median-cut collapsed CYAN #3fd0d8 and
    # GREEN #6ecf94 onto a single (92,190,190) entry, which destroys the
    # "machine vs landed hit" colour split.  Reserving slots is the fix.
    nfree = args.colors - len(BRAND)
    mont = np.concatenate([np.asarray(im) for im in raw[::3]], axis=0)
    base = Image.fromarray(mont).quantize(colors=nfree, method=Image.MEDIANCUT,
                                          dither=Image.NONE)
    pl = list(base.getpalette()[:nfree * 3])
    for hx in BRAND:
        pl += [int(hx[j:j + 2], 16) for j in (1, 3, 5)]
    pl += [0, 0, 0] * (256 - args.colors)
    pal = Image.new("P", (1, 1))
    pal.putpalette(pl[:768])
    qs = [im.quantize(palette=pal, dither=Image.NONE) for im in raw]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    qs[0].save(args.out, save_all=True, append_images=qs[1:], duration=200,
               loop=0, optimize=True, disposal=1)
    size = os.path.getsize(args.out)

    side = os.path.join(REPO, "docs/assets/data/mechanics-microscope.json")
    os.makedirs(os.path.dirname(side), exist_ok=True)
    json.dump({
        "id": "mechanics-microscope",
        "rendered": stamp,
        "source": "pvpbot/sim/env.py (DuelVecEnv(2, seed=0, SimConfig()))",
        "frames": len(qs), "fps": 5.0, "duration_ms": 200,
        "size_bytes": size, "dimensions": [W_PX, H_PX], "colors": args.colors,
        "constants": {
            "dmg_base": dmg_base, "dmg_crit": dmg_crit,
            "hurt_ticks": hurt_ticks, "reach": reach,
            "eye_h": float(E._eye_h), "half_w": float(E._half_w),
            "box_h": float(E._box_h), "kb_h": float(E._kb_h),
            "kb_v": float(E._kb_v), "jump_v": float(E._jump_v),
            "gravity": float(E._gravity),
        },
        "act1": {
            "ticks": len(A1), "swing_tick": 12, "jump_tick_env1": 7,
            "attacker_x": -2.2, "target_x": 0.0,
            "vel_y_after_gravity_env1": [round(float(f["vy"][1, 0]), 4)
                                         for f in A1[6:14]],
            "damage": [float(A1[12]["dmg"][0]), float(A1[12]["dmg"][1])],
            "target_hp_after": [float(A1[-1]["hp"][0, 1]),
                                float(A1[-1]["hp"][1, 1])],
        },
        "act2": {
            "ticks": len(A2), "swing_ticks": [4, 10], "jump_tick_env1": 5,
            "attacker_x": -1.5, "target_x": 0.0,
            "first_hit_damage": [float(A2[4]["dmg"][0]), float(A2[4]["dmg"][1])],
            "second_hit_damage": [float(A2[10]["dmg"][0]),
                                  float(A2[10]["dmg"][1])],
            "hurt_after_second": [int(A2[10]["hurt"][0, 1]),
                                  int(A2[10]["hurt"][1, 1])],
            "last_dmg_after_second": [float(A2[10]["last"][0, 1]),
                                      float(A2[10]["last"][1, 1])],
        },
    }, open(side, "w"), indent=1)

    print("wrote %s  %d frames  %dx%d  %.2f MB" %
          (args.out, len(qs), W_PX, H_PX, size / 1e6))
    print("sidecar %s" % side)


if __name__ == "__main__":
    main()
