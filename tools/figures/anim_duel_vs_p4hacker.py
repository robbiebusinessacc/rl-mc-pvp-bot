"""Animated duel: the trained policy fighting P4-Hacker in the 1.8 duel sim.

Everything drawn is per-tick ground truth read straight out of
``pvpbot.sim.env.DuelVecEnv`` while a ``pvpbot.models.PolicyNet`` checkpoint
(side 0) fights ``pvpbot.eval.practice.PracticeHacker`` (side 1):

  env.pos / env.vel / env.yaw / env.pitch / env.hp / env.hurt /
  env.on_ground / env.sprinting                    -- snapshotted BEFORE each
                                                      env.step (the env
                                                      auto-resets on the done
                                                      tick, so post-step state
                                                      belongs to a fresh duel)
  hp_before - env.hp                               -- damage resolved this tick
  the (2, 7) action array fed to env.step          -- the action heads

Panels
  * POLICY EYE VIEW  - a pinhole projection of the same ground truth from the
    policy's eye (pos + 1.62 m, its own yaw/pitch, 70 deg vertical FOV). It is
    a geometric reconstruction of the sim, not a screenshot of Minecraft and
    not the live pixel pipeline. aim_residual() re-checks it against the env's
    own aim_err_yaw / aim_err_pitch channels on EVERY rendered tick and stamps
    the worst disagreement into the footer, so the number in the caption and
    the number in the frame cannot drift apart.
  * TOP-DOWN        - true 0.6 m hitbox footprints, facing, the 3.0 m reach
    ring (dashed out of reach, SOLID in), velocity (knockback shows up here),
    swing arcs, damage numbers. "In reach" is the env's own eye-to-AABB test
    (reach_dist(), mirroring pvpbot/sim/env.py:615-623), not a centre-to-centre
    gap -- the two differ by up to 0.42 m and only the former agrees with the
    ticks on which hits actually land.
  * ACTION HEADS    - the 7 categorical heads of spec.ACTION_HEADS, live.
  * TIMELINE        - both HP tracks, that same reach distance against the
    3.0 m line, and a marker per landed hit.

Source data is gitignored (runs/ is not committed), so a fresh clone cannot
re-render this: the script exits with a named-file message instead. The GIF and
its sidecar docs/assets/data/duel-vs-p4hacker.json are the committed record.

Usage:
    python3 tools/figures/anim_duel_vs_p4hacker.py \
        [--ckpt runs/fov1/ckpt_32.8B_faithful78.pt] \
        [--out docs/assets/duel-vs-p4hacker.gif]
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Polygon, Rectangle
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import torch

from pvpbot.eval.practice import PracticeHacker
from pvpbot.models import PolicyNet
from pvpbot.sim.env import DuelVecEnv, SimConfig
from pvpbot.spec import ACTION_HEADS, OBS_LAYOUT
from pvpbot.train.ppo import RunningNorm

# palette shared with the other figures in tools/figures
BG, PANEL, GRID, INK, DIM = "#12161c", "#1a212a", "#2b3542", "#ccd6e2", "#66747f"
CYAN, AMBER, RED = "#3fd0d8", "#e8a33d", "#e2564a"
SKY, GROUND, HORIZ, GLINE = "#0e131a", "#171f29", "#31404f", "#1f2a36"
WALL, WALLTOP, SLOT, FLASH = "#242f3d", "#3a4859", "#232c38", "#ffffff"

# stage badge: this artifact is a bench test of the sim mirror (DuelVecEnv),
# not of anything on the live wire (0 screen, 1 sensor, 2 adapter,
# 3 controller, 4 injection).
STAGE = "M"
COMPONENT = "DuelVecEnv 1.8 combat mirror"
HDR_PX, FTR_PX = 26, 26   # header / footer strip heights, rendered into frame

EYE_H = 1.62            # SimConfig.eye_height
VFOV = 70.0             # Minecraft's default vertical FOV
FOCAL = 1.0 / np.tan(np.radians(VFOV) / 2.0)
NEAR = 0.12
AR = 1.66               # eye-view panel aspect (half-width in NDC)
ARENA_R = 7.0           # SimConfig.arena_radius as HALF-SIDE: the live arena is
                        # a 14x14 walled square (pvpbot/sim/env.py:181), so the
                        # rollout sets arena_square=True and the wall is drawn
                        # with corners, not as a disc.
REACH = 3.0             # SimConfig.reach

# The duel is rolled out on the arena the bot actually plays in: a 14x14 walled
# square. arena_square=True switches env.py's position clamp from a radial one
# (a disc) to an axis-independent one, so the corners exist -- and per
# pvpbot/sim/env.py:182, corners are what punish a backpedalling policy.
ARENA_CFG = SimConfig(arena_square=True, arena_radius=ARENA_R)


def arena_perimeter(half, per_edge=60):
    """Closed 14x14 wall path in world XZ, densely sampled.

    Straight edges stay straight under perspective projection, but the sampling
    has to be dense anyway so near-plane clipping and the visible-run splitting
    behave exactly as they did for the disc. Corners land on exact samples.
    """
    t = np.linspace(0.0, 4.0, 4 * per_edge + 1)
    seg, fr = np.minimum(np.floor(t).astype(int), 3), t % 1.0
    span = 2.0 * half
    ones = np.ones_like(fr)
    x = np.select([seg == 0, seg == 1, seg == 2, seg == 3],
                  [-half + span * fr, half * ones, half - span * fr, -half * ones])
    z = np.select([seg == 0, seg == 1, seg == 2, seg == 3],
                  [-half * ones, -half + span * fr, half * ones, half - span * fr])
    return np.stack([x, np.zeros_like(x), z], 1)


AIM_YAW = OBS_LAYOUT["aim_err_yaw"][0]

CUBE_E = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
          (0, 4), (1, 5), (2, 6), (3, 7)]
CUBE_F = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6),
          (0, 2, 6, 4), (1, 3, 7, 5)]


# --------------------------------------------------------------------------
# rollout
# --------------------------------------------------------------------------
class Learner:
    """A spec-format checkpoint driven as side 0, GRU state carried per env."""

    def __init__(self, path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.meta = blob["meta"]
        obs_dim = int(self.meta.get("obs_dim", 48))
        self.net = PolicyNet(obs_dim=obs_dim)
        self.net.load_state_dict(blob["model"])
        self.net.eval()
        self.norm = None
        if "obs_norm" in blob:            # trainer ships the obs RunningNorm
            self.norm = RunningNorm(obs_dim)
            self.norm.load_state_dict(blob["obs_norm"])
        self.state = None

    def begin(self, n):
        self.state = self.net.initial_state(n)

    def on_done(self, done):
        self.state[torch.from_numpy(np.ascontiguousarray(done, bool))] = 0.0

    def act(self, obs):
        if self.state is None or self.state.shape[0] != obs.shape[0]:
            self.begin(obs.shape[0])
        x = self.norm.normalize(obs) if self.norm is not None else obs
        with torch.no_grad():
            a, self.state = self.net.act(
                torch.from_numpy(np.ascontiguousarray(x, np.float32)), self.state)
        return a.numpy().astype(np.int64)


KEYS = ("pos", "vel", "yaw", "pitch", "hp", "hurt", "on_ground", "sprinting",
        "act", "obs0", "dmg")


def rollout(ckpt, num_envs, seed, max_ticks=1200):
    env = DuelVecEnv(num_envs, seed=seed, config=ARENA_CFG)
    learner, opp = Learner(ckpt), PracticeHacker(13)
    obs = env.reset()
    learner.begin(num_envs)
    opp.begin(num_envs)
    pending = np.ones(num_envs, bool)
    ep_len = np.zeros(num_envs, np.int64)
    outcome = np.zeros(num_envs, np.int8)
    rec = {k: [] for k in KEYS}
    for _ in range(max_ticks):
        if not pending.any():
            break
        # scripted opponents model server-side mineflayer bots: they get the
        # ungated omniscient side-1 obs, exactly as pvpbot/train/run.py does
        ung = getattr(env, "_obs_ungated", None)
        obs1 = ung[:, 1] if ung is not None else obs[:, 1]
        acts = np.stack([learner.act(obs[:, 0]), opp.act(obs1)], axis=1)
        rec["pos"].append(env.pos.copy()); rec["vel"].append(env.vel.copy())
        rec["yaw"].append(env.yaw.copy()); rec["pitch"].append(env.pitch.copy())
        rec["hp"].append(env.hp.copy()); rec["hurt"].append(env.hurt.copy())
        rec["on_ground"].append(env.on_ground.copy())
        rec["sprinting"].append(env.sprinting.copy())
        rec["act"].append(acts.copy()); rec["obs0"].append(obs[:, 0].copy())
        hp_pre = env.hp.copy()
        obs, _, done, info = env.step(acts)
        dmg = hp_pre - env.hp
        fin = pending & done
        if fin.any():                      # env already reset: rebuild the
            w = info["win"][fin]           # killing blow from the win flag
            k = np.zeros((int(fin.sum()), 2), np.float32)
            k[w[:, 0] > 0.5, 1] = hp_pre[fin][w[:, 0] > 0.5, 1]
            k[w[:, 1] > 0.5, 0] = hp_pre[fin][w[:, 1] > 0.5, 0]
            dmg[fin] = k
            outcome[fin & (info["win"][:, 0] > 0.5)] = 1
            outcome[fin & (info["win"][:, 1] > 0.5)] = -1
        dmg[~pending] = 0.0
        rec["dmg"].append(dmg)
        ep_len[pending] += 1
        pending &= ~done
        learner.on_done(done)
        opp.on_done(done)
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["ep_len"], out["outcome"] = ep_len, outcome
    return out


def pick_duel(ckpt, seeds=12, num_envs=32):
    """Score every decisive duel for legibility, keep the best one.

    PolicyNet.act SAMPLES its heads, so torch's global RNG is part of the
    trajectory: seed it, and keep the winning rollout rather than replaying it.
    """
    best, best_r = None, None
    for seed in range(seeds):
        torch.manual_seed(1234 + seed)
        r = rollout(ckpt, num_envs, seed)
        for e in range(num_envs):
            n = int(r["ep_len"][e])
            if r["outcome"][e] != 1 or not (150 <= n <= 230):
                continue
            pos, dmg = r["pos"][:n, e], r["dmg"][:n, e]
            landed, taken = int((dmg[:, 1] > 0).sum()), int((dmg[:, 0] > 0).sum())
            spread = float(max(np.ptp(pos[:, :, 0]), np.ptp(pos[:, :, 2])))
            # want: decisive, plenty of two-way exchanges, a fight that stays
            # compact enough to frame, and about 9 s long
            score = (landed + 1.6 * taken - 0.06 * abs(n - 180)
                     - 0.8 * max(0.0, spread - 9.0))
            if best is None or score > best[0]:
                best, best_r = (score, seed, e, n, landed, taken), r
    if best is None:
        raise SystemExit("no decisive duel found in the scored range")
    score, seed, e, n, landed, taken = best
    d = {k: best_r[k][:n, e] for k in KEYS}
    # append one synthetic post-kill tick: the killing blow resolves inside
    # the final env.step(), so the pre-step HP snapshot never reaches 0
    for k in KEYS:
        d[k] = np.concatenate([d[k], d[k][-1:]], 0)
    d["hp"][-1, 1] = 0.0
    d["hurt"][-1, 1] = 10
    d["dmg"][-1] = 0.0
    d["reach"] = reach_dist(d["pos"].astype(np.float64))
    print("duel: seed %d env %d | %d ticks (%.2f s) | policy landed %d, took %d"
          % (seed, e, n, n / 20.0, landed, taken))
    return d, {"seed": seed, "env_index": e, "ticks": n,
               "hits_landed": landed, "hits_taken": taken}


# --------------------------------------------------------------------------
# projection helpers
# --------------------------------------------------------------------------
def basis(yaw_deg, pitch_deg):
    """Camera axes for the sim's convention: facing = (cos yaw, 0, sin yaw),
    pitch positive = looking up."""
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    cy, sy, cp, sp = np.cos(y), np.sin(y), np.cos(p), np.sin(p)
    return (np.array([cp * cy, sp, cp * sy]),      # forward
            np.array([-sy, 0.0, cy]),              # screen right
            np.array([-cy * sp, cp, -sy * sp]))    # screen up


def proj(pts, eye, f, r, u):
    rel = np.asarray(pts, np.float64) - eye
    z = rel @ f
    zz = np.where(np.abs(z) < 1e-9, 1e-9, z)
    return np.stack([(rel @ r) / zz * FOCAL, (rel @ u) / zz * FOCAL], 1), z


def cube(cx, cy, cz, hw, h):
    return np.array([(cx + dx, cy + dy, cz + dz)
                     for dy in (0.0, h) for dz in (-hw, hw) for dx in (-hw, hw)],
                    np.float64)


def visible_runs(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j)); i = j + 1
        else:
            i += 1
    return out


def reach_dist(pos):
    """Each side's eye -> opponent-AABB distance, the sim's own reach metric.

    pvpbot/sim/env.py:615-623 -- a swing connects when this is <= 3.0 m, NOT
    when the centre-to-centre gap is. The difference is up to 0.42 m on a
    diagonal approach, which is the difference between the reach ring reading
    "out of reach" on a tick where a hit demonstrably lands, and reading true.
    """
    out = np.zeros(pos.shape[:2], np.float64)
    for s in (0, 1):
        o = 1 - s
        dx = pos[:, o, 0] - pos[:, s, 0]
        dz = pos[:, o, 2] - pos[:, s, 2]
        ax_ = np.maximum(np.abs(dx) - 0.3, 0.0)      # 0.6 m box half-width
        az_ = np.maximum(np.abs(dz) - 0.3, 0.0)
        eye_y = pos[:, s, 1] + EYE_H
        ay_ = np.maximum(np.maximum(pos[:, o, 1] - eye_y,
                                    eye_y - (pos[:, o, 1] + 1.8)), 0.0)
        out[:, s] = np.sqrt(ax_ * ax_ + ay_ * ay_ + az_ * az_)
    return out


def aim_residual(d, n):
    """Cross-check the eye-view projection against the env's own aim channels.

    ``aim_err_yaw`` / ``aim_err_pitch`` (OBS_LAYOUT slots 22, 23) are the angles
    from the policy's crosshair to the enemy hitbox centre. With el = pitch + b
    and a = yaw error, that same point lands at
        sx = FOCAL * cos(el) sin(a) / D
        sy = FOCAL * (cos(p) sin(el) - sin(p) cos(el) cos(a)) / D
        D  = cos(el) cos(p) cos(a) + sin(el) sin(p)
    in the camera basis built by basis(). If draw_eye's projection is right the
    two agree; the max disagreement over the duel is stamped into the footer.
    """
    pos, yaw, pitch, obs = d["pos"], d["yaw"], d["pitch"], d["obs0"]
    worst = 0.0
    for t in range(n):
        eye = pos[t, 0].astype(np.float64) + np.array([0.0, EYE_H, 0.0])
        f, r, u = basis(yaw[t, 0], pitch[t, 0])
        # hitbox centre = feet + 0.5 * hitbox_height (SimConfig, 1.8 m box)
        P, _ = proj((pos[t, 1].astype(np.float64)
                     + np.array([0.0, 0.9, 0.0]))[None, :], eye, f, r, u)
        a = np.radians(float(obs[t, AIM_YAW]) * 180.0)
        b = np.radians(float(obs[t, AIM_YAW + 1]) * 90.0)
        p = np.radians(float(pitch[t, 0]))
        el = p + b
        den = np.cos(el) * np.cos(p) * np.cos(a) + np.sin(el) * np.sin(p)
        sx = FOCAL * np.cos(el) * np.sin(a) / den
        sy = FOCAL * (np.cos(p) * np.sin(el)
                      - np.sin(p) * np.cos(el) * np.cos(a)) / den
        worst = max(worst, abs(P[0, 0] - sx), abs(P[0, 1] - sy))
    return float(worst)


def step_camera(pos, half, margin=1.15):
    """Top-down camera: static until a fighter nears the edge, then snaps.
    A still background is also what keeps the GIF's inter-frame deltas small."""
    c = 0.5 * (pos[0, 0, [0, 2]] + pos[0, 1, [0, 2]]).astype(np.float64)
    out = np.zeros((pos.shape[0], 2))
    for t in range(pos.shape[0]):
        pts = pos[t, :, :][:, [0, 2]]
        if np.any(np.abs(pts - c) > half - margin):
            c = 0.5 * (pts[0] + pts[1])
        out[t] = c
    return out


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------
def draw_eye(ax, d, t, ko=False):
    pos, yaw, pitch = d["pos"], d["yaw"], d["pitch"]
    hp, hurt, dmg, act = d["hp"], d["hurt"], d["dmg"], d["act"]
    eye = pos[t, 0].astype(np.float64) + np.array([0.0, EYE_H, 0.0])
    f, r, u = basis(yaw[t, 0], pitch[t, 0])
    ax.clear(); ax.set_facecolor(SKY)
    ax.set_xlim(-AR, AR); ax.set_ylim(-1, 1)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    # the vanishing line of the y=0 plane is a horizontal line at -tan(pitch)*f
    yh = -np.tan(np.radians(pitch[t, 0])) * FOCAL
    ax.add_patch(Rectangle((-AR, -1.3), 2 * AR, min(yh, 1.3) + 1.3,
                           facecolor=GROUND, ec="none", zorder=0))
    ax.plot([-AR, AR], [yh, yh], color=HORIZ, lw=1.0, zorder=1)
    for g in np.arange(-16, 16.1, 4.0):
        for a, b in ((np.array([-16., 0., g]), np.array([16., 0., g])),
                     (np.array([g, 0., -16.]), np.array([g, 0., 16.]))):
            pa, pb = a.copy(), b.copy()
            za, zb = (pa - eye) @ f, (pb - eye) @ f
            if za < NEAR and zb < NEAR:
                continue
            if za < NEAR:
                pa = pa + (NEAR - za) / (zb - za) * (pb - pa)
            elif zb < NEAR:
                pb = pb + (NEAR - zb) / (za - zb) * (pa - pb)
            P, _ = proj(np.stack([pa, pb]), eye, f, r, u)
            ax.plot(P[:, 0], P[:, 1], color=GLINE, lw=0.9, zorder=2)
    bot = arena_perimeter(ARENA_R)
    Pb, zb = proj(bot, eye, f, r, u)
    Pt, _ = proj(bot + np.array([0, 1.25, 0]), eye, f, r, u)
    for i, j in visible_runs(zb > NEAR):
        if j - i < 2:
            continue
        ax.add_patch(Polygon(np.concatenate([Pb[i:j + 1], Pt[i:j + 1][::-1]]),
                             closed=True, facecolor=WALL, ec="none", zorder=3))
        ax.plot(Pt[i:j + 1, 0], Pt[i:j + 1, 1], color=WALLTOP, lw=1.1, zorder=4)
    ex, ey, ez = pos[t, 1]
    col = RED if hurt[t, 1] > 0 else AMBER
    for c3, hw, h, al in (((ex, ey, ez), 0.3, 1.8, 0.42),
                          ((ex, ey + 1.25, ez), 0.26, 0.52, 0.78)):
        P, z = proj(cube(*c3, hw, h), eye, f, r, u)
        if (z > NEAR).all():
            for q in CUBE_F:
                ax.add_patch(Polygon(P[list(q)], closed=True, facecolor=col,
                                     alpha=al, ec="none", zorder=6))
            for a, b in CUBE_E:
                ax.plot(P[[a, b], 0], P[[a, b], 1], color=col, lw=1.4, zorder=7)
    Pn, zn = proj(np.array([[ex, ey + 2.15, ez]]), eye, f, r, u)
    if zn[0] > NEAR and abs(Pn[0, 0]) < AR:
        # clamp below the panel title/subtitle: at close range with the camera
        # pitched up the nametag otherwise lands on top of them
        ny = min(float(Pn[0, 1]), 0.42 if ko else 0.70)
        ax.text(Pn[0, 0], ny, "P4-HACKER  %.1f" % max(hp[t, 1], 0.0),
                color="#e8eef5", fontsize=6.5, ha="center", family="monospace",
                zorder=8, bbox=dict(boxstyle="square,pad=0.25", fc="#0b0f15",
                                    ec="none", alpha=0.75))
    for back in range(4):
        tt = t - back
        if tt >= 0 and dmg[tt, 1] > 0:
            Ph, zh = proj(np.array([[pos[tt, 1, 0], pos[tt, 1, 1] + 0.9,
                                     pos[tt, 1, 2]]]), eye, f, r, u)
            if zh[0] > NEAR:
                ax.add_patch(Circle(
                    (Ph[0, 0], Ph[0, 1]),
                    (0.05 + 0.035 * back) * FOCAL / max(zh[0], 0.5),
                    fill=False, ec=FLASH, lw=2.4 - 0.5 * back,
                    alpha=max(0.0, 0.9 - 0.22 * back), zorder=8))
    swinging = act[t, 0, 4] == 1
    hx, hy, ang = (1.40, -1.06, 44) if not swinging else (1.16, -0.96, 18)
    a0 = np.radians(90 - ang)
    ax.plot([hx, hx + 0.40 * np.cos(a0)], [hy, hy + 0.40 * np.sin(a0)],
            color="#e6edf3", lw=5, solid_capstyle="round", zorder=8)
    ax.plot([hx - .065 * np.cos(a0 + 1.57), hx + .065 * np.cos(a0 + 1.57)],
            [hy - .065 * np.sin(a0 + 1.57), hy + .065 * np.sin(a0 + 1.57)],
            color="#7c5a3a", lw=4.0, solid_capstyle="round", zorder=8)
    for i in range(10):
        v = hp[t, 0]
        c = RED if v >= (i + 1) * 2 - .01 else ("#8a2b34" if v > i * 2 + .01
                                                else "#242d38")
        ax.add_patch(Circle((-1.54 + i * 0.10, -0.88), 0.031, facecolor=c,
                            ec="none", zorder=9))
    for xs, ys in (((-.048, -.015), (0, 0)), ((.015, .048), (0, 0)),
                   ((0, 0), (-.048, -.015)), ((0, 0), (.015, .048))):
        ax.plot(xs, ys, color="#e6edf3", lw=1.5, zorder=9)
    if dmg[t, 0] > 0:
        ax.add_patch(Rectangle((-AR, -1), 2 * AR, 2, facecolor=RED, alpha=0.20,
                               zorder=10))
    ax.text(-1.62, 0.90, "POLICY EYE VIEW", color=INK, fontsize=8.5,
            family="monospace", weight="bold")
    ax.text(-1.62, 0.80, "geometric reconstruction from sim ground truth",
            color=DIM, fontsize=6.4, family="monospace")
    ax.text(1.62, 0.90, "yaw %5.1f  pitch %5.1f"
            % (yaw[t, 0] % 360.0, pitch[t, 0]), color=DIM, fontsize=6.8,
            ha="right", family="monospace")
    if ko:
        # low enough that the glyph tops clear the panel subtitle at y=0.80
        ax.text(0, 0.56, "K.O.", color=FLASH, fontsize=24, ha="center",
                family="monospace", weight="bold", zorder=12)


def draw_top(ax, d, t, cx, cz, half):
    pos, yaw, hurt = d["pos"], d["yaw"], d["hurt"]
    spr, dmg, act, vel = d["sprinting"], d["dmg"], d["act"], d["vel"]
    ax.clear(); ax.set_facecolor(PANEL)
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cz - half, cz + half)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for g in np.arange(-16, 16.1, 2.0):
        ax.plot([-18, 18], [g, g], color=GRID, lw=0.5, zorder=0)
        ax.plot([g, g], [-18, 18], color=GRID, lw=0.5, zorder=0)
    ax.add_patch(Rectangle((-ARENA_R, -ARENA_R), 2 * ARENA_R, 2 * ARENA_R,
                           fill=False, ec="#39465a", lw=1.8, zorder=1))
    t0 = max(0, t - 24)
    for s, c in ((0, CYAN), (1, AMBER)):
        xs, zs = pos[t0:t + 1, s, 0], pos[t0:t + 1, s, 2]
        for k in range(len(xs) - 1):
            ax.plot(xs[k:k + 2], zs[k:k + 2], color=c, lw=1.8, zorder=2,
                    alpha=0.06 + 0.44 * (k / max(len(xs) - 1, 1)))
    rd = d["reach"]
    for s, c in ((1, AMBER), (0, CYAN)):
        x, y, z = pos[t, s]
        yw = yaw[t, s]
        inr = rd[t, s] <= REACH
        # the 3.0 m reach ring goes dashed -> SOLID the tick the gap is inside
        # reach, which is the only geometry that decides whether a swing lands
        ax.add_patch(Circle((x, z), REACH, fill=False, ec=c,
                            lw=1.7 if inr else 0.9,
                            ls="-" if inr else (0, (4, 4)),
                            alpha=0.75 if inr else 0.34, zorder=3))
        rr = np.radians(yw)
        ax.plot([x, x + REACH * np.cos(rr)], [z, z + REACH * np.sin(rr)],
                color=c, lw=1.1, alpha=0.75, zorder=4)
        if y > 0.02:                       # airborne: the shadow stays put
            ax.add_patch(Circle((x, z), 0.26, facecolor="#0b0e13", ec="none",
                                alpha=0.6, zorder=4))
        by = z - y * 0.5                   # slight 3/4 lift so jumps read
        ax.add_patch(Rectangle((x - .3, by - .3), .6, .6, facecolor=c,
                               ec="#0b0e13", lw=.9, zorder=6))
        ax.add_patch(Polygon(
            [(x + .68 * np.cos(rr), by + .68 * np.sin(rr)),
             (x + .34 * np.cos(rr + 2.5), by + .34 * np.sin(rr + 2.5)),
             (x + .34 * np.cos(rr - 2.5), by + .34 * np.sin(rr - 2.5))],
            facecolor=c, ec="none", zorder=6))
        v = vel[t, s]
        if np.hypot(v[0], v[2]) > 0.03:
            ax.arrow(x, by, v[0] * 7, v[2] * 7, width=0.05, head_width=0.20,
                     color=c, alpha=0.9 if spr[t, s] else 0.4,
                     length_includes_head=True, zorder=5)
        if act[t, s, 4] == 1:
            ax.add_patch(Arc((x, by), 2.0, 2.0, angle=yw, theta1=-40, theta2=40,
                             ec=c, lw=2.0, alpha=0.95, zorder=7))
        if hurt[t, s] > 0:
            ax.add_patch(Circle((x, by), .5, facecolor=RED, ec="none",
                                alpha=.07 + .02 * hurt[t, s], zorder=7))
    for back in range(5):
        tt = t - back
        if tt < 0:
            continue
        for s in (0, 1):
            if dmg[tt, s] > 0:
                x, z = pos[tt, s, 0], pos[tt, s, 2]
                ax.add_patch(Circle((x, z), .5 + .6 * back, fill=False,
                                    ec=FLASH, lw=2.4 - .45 * back,
                                    alpha=max(0., .9 - .2 * back), zorder=8))
                if back == 0:
                    ax.text(x, z + 1.0, "-%.1f" % dmg[tt, s], color=FLASH,
                            fontsize=8, ha="center", family="monospace",
                            weight="bold", zorder=9)
    tbb = dict(boxstyle="square,pad=0.22", fc=PANEL, ec="none", alpha=0.82)
    ax.text(0.03, 0.955, "TOP-DOWN  2 m grid", color=DIM, fontsize=6.8,
            family="monospace", transform=ax.transAxes, zorder=10, bbox=tbb)
    ax.text(0.03, 0.028, "ring = 3.0 m reach   dashed OUT / solid IN",
            color=DIM, fontsize=6.0, family="monospace", zorder=10,
            transform=ax.transAxes,
            bbox=dict(boxstyle="square,pad=0.22", fc=PANEL, ec="none",
                      alpha=0.82))
    ax.text(0.97, 0.955, "reach %4.2f m" % rd[t, 0],
            color=CYAN if rd[t, 0] <= REACH else INK, fontsize=7.2,
            ha="right", family="monospace", transform=ax.transAxes,
            zorder=10, bbox=tbb)


def draw_heads(ax, d, t):
    hp, hurt, spr, og, act = (d["hp"], d["hurt"], d["sprinting"],
                              d["on_ground"], d["act"])
    ax.clear(); ax.set_facecolor(PANEL)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])

    def bar(y0, s, name, sub, c):
        frac = max(0.0, hp[t, s] / 20.0)
        ax.text(0.05, y0 + 0.115, name, color=c, fontsize=9,
                family="monospace", weight="bold")
        ax.text(0.95, y0 + 0.115, "%4.1f" % max(hp[t, s], 0.0), color=INK,
                fontsize=9, ha="right", family="monospace", weight="bold")
        ax.text(0.05, y0 + 0.070, sub, color=DIM, fontsize=6.2,
                family="monospace")
        ax.add_patch(Rectangle((0.05, y0), 0.90, 0.050, facecolor=SLOT,
                               ec="none"))
        ax.add_patch(Rectangle((0.05, y0), 0.90 * frac, 0.050, facecolor=c,
                               ec="none"))
        chips = []
        if spr[t, s]:
            chips.append("SPRINT")
        if not og[t, s]:
            chips.append("AIR")
        if hurt[t, s] > 0:
            chips.append("HURT %d" % hurt[t, s])
        if act[t, s, 4] == 1:
            chips.append("SWING")
        ax.text(0.05, y0 - 0.055, "  ".join(chips) or " ",
                color=RED if hurt[t, s] > 0 else DIM, fontsize=6.8,
                family="monospace")

    bar(0.80, 0, "POLICY", "PolicyNet GRU, self-play PPO league", CYAN)
    bar(0.56, 1, "P4-HACKER", "scripted: 600 deg/s, 11 cps, react 0", AMBER)
    ax.text(0.05, 0.445, "POLICY ACTION HEADS", color=DIM, fontsize=6.8,
            family="monospace")
    for i, (nm, k) in enumerate(ACTION_HEADS):
        yy = 0.375 - i * 0.055
        ax.text(0.05, yy, nm.upper(), color=DIM, fontsize=6.8,
                family="monospace")
        x0, w = 0.38, 0.57 / k
        for b in range(k):
            ax.add_patch(Rectangle((x0 + b * w, yy - 0.004), w * 0.8, 0.026,
                                   facecolor=CYAN if act[t, 0, i] == b else SLOT,
                                   ec="none"))


def draw_timeline(ax, d, t):
    hp, dmg, pos = d["hp"], d["dmg"], d["pos"]
    n = hp.shape[0]
    gap = d["reach"][:, 0]      # the sim's own eye -> hitbox reach distance
    ax.clear(); ax.set_facecolor(PANEL)
    # left gutter so the row labels never sit on top of the hit markers
    gut = 0.30 * (n - 1)
    ax.set_xlim(-gut, n - 1); ax.set_ylim(-0.06, 1.62)
    ax.set_xticks([]); ax.set_yticks([])
    ax.axvline(0, color=GRID, lw=0.8)
    ax.plot(np.arange(t + 1), gap[:t + 1] / 8.0 + 0.02, color="#5b6b7e", lw=1.0)
    ax.plot(np.arange(t, n), gap[t:] / 8.0 + 0.02, color="#5b6b7e", lw=1.0,
            alpha=0.18)
    # drawn from x=0 so it never crosses the label gutter
    ax.plot([0, n - 1], [REACH / 8.0 + 0.02] * 2, color="#3d4a5a", lw=0.7,
            ls=(0, (3, 3)))
    ax.plot(np.arange(t + 1), hp[:t + 1, 0] / 20.0, color=CYAN, lw=1.6)
    ax.plot(np.arange(t + 1), hp[:t + 1, 1] / 20.0, color=AMBER, lw=1.6)
    ax.plot(np.arange(t, n), hp[t:, 0] / 20.0, color=CYAN, lw=1.0, alpha=0.18)
    ax.plot(np.arange(t, n), hp[t:, 1] / 20.0, color=AMBER, lw=1.0, alpha=0.18)
    for s, c, yy in ((0, CYAN, 1.48), (1, AMBER, 1.30)):
        idx = np.nonzero(dmg[:, 1 - s] > 0)[0]
        ax.scatter(idx[idx <= t], np.full((idx <= t).sum(), yy), s=14, color=c,
                   marker="v")
        ax.scatter(idx[idx > t], np.full((idx > t).sum(), yy), s=14,
                   color="#28313e", marker="v")
    ax.axvline(t, color=INK, lw=1.0, alpha=0.85)
    # static labels only, plus the two running hit tallies: HP and gap already
    # read out live in the heads and top-down panels, and every extra glyph
    # that changes per frame is inter-frame delta the GIF has to pay for
    gx = -0.28 * (n - 1)
    for s, c, yy in (("POLICY hits", CYAN, 1.48),
                     ("P4-HACKER hits", AMBER, 1.30),
                     ("HP  20 -> 0", DIM, 1.06),
                     ("reach dist", "#7d8ea1", 0.50),
                     ("eye -> hitbox", "#7d8ea1", 0.36),
                     ("dashed = 3.0 m reach", "#5b6b7e", 0.22),
                     ("TIMELINE  whole duel", DIM, 0.13)):
        ax.text(gx, yy, s, color=c, fontsize=6.4, family="monospace",
                va="center")
    ax.text(gx + 0.265 * (n - 1), 1.48, "%2d" % int((dmg[:t + 1, 1] > 0).sum()),
            color=CYAN, fontsize=6.4, family="monospace", va="center",
            ha="right", weight="bold")
    ax.text(gx + 0.265 * (n - 1), 1.30, "%2d" % int((dmg[:t + 1, 0] > 0).sum()),
            color=AMBER, fontsize=6.4, family="monospace", va="center",
            ha="right", weight="bold")


def draw_header(ax, W, t):
    """Persistent strip: stage badge, component under test, tick and clock."""
    ax.clear(); ax.set_facecolor(PANEL)
    ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(Rectangle((9, 0.17), 19, 0.66, facecolor=GRID, ec="none"))
    ax.text(18.5, 0.51, STAGE, color=INK, fontsize=9.5, ha="center",
            va="center", family="monospace", weight="bold")
    ax.text(36, 0.51, "STAGE %s · %s · tick %04d · t=%5.2f s"
            % (STAGE, COMPONENT, t, t / 20.0), color=INK, fontsize=8.2,
            va="center", family="monospace")
    ax.text(W - 9, 0.51, "1 GIF frame = 1 sim tick · 20 fps · real time",
            color=DIM, fontsize=6.8, ha="right", va="center",
            family="monospace")


def draw_footer(ax, W, cond, check, prov, stampline):
    """Persistent strip: the condition, and the input stamp (steps + date)."""
    ax.clear(); ax.set_facecolor(PANEL)
    ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    ax.text(9, 0.70, cond, color="#93a2b2", fontsize=6.2, va="center",
            family="monospace")
    ax.text(W - 9, 0.70, check, color="#93a2b2", fontsize=6.2, va="center",
            ha="right", family="monospace")
    ax.text(9, 0.28, prov, color=DIM, fontsize=6.0, va="center",
            family="monospace")
    ax.text(W - 9, 0.28, stampline, color=DIM, fontsize=6.0, va="center",
            ha="right", family="monospace")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/fov1/ckpt_32.8B_faithful78.pt")
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "assets",
                                                  "duel-vs-p4hacker.gif"))
    ap.add_argument("--colors", type=int, default=64)
    ap.add_argument("--width", type=float, default=8.24)
    ap.add_argument("--panels", type=float, default=5.28,
                    help="height of the four-panel block, inches; the header "
                         "and footer strips are added on top of it")
    a = ap.parse_args()
    ckpt = a.ckpt if os.path.isabs(a.ckpt) else os.path.join(REPO, a.ckpt)
    if not os.path.exists(ckpt):
        raise SystemExit(
            "source checkpoint no longer on disk: %s\n"
            "runs/ is gitignored; docs/assets/duel-vs-p4hacker.gif and "
            "docs/assets/data/duel-vs-p4hacker.json are the committed record."
            % ckpt)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    d, sel = pick_duel(ckpt)
    n = d["pos"].shape[0]
    half = 5.0
    cam = step_camera(d["pos"], half)
    resid = aim_residual(d, n - 1)      # last row is the synthetic post-kill tick
    meta = torch.load(ckpt, map_location="cpu", weights_only=False)["meta"]
    stamp = datetime.date.today().isoformat()

    cond = ("14x14 square arena, full-information observations · seed %d, "
            "env %d · %d ticks · %d hits landed / %d taken · K.O."
            % (sel["seed"], sel["env_index"], sel["ticks"],
               sel["hits_landed"], sel["hits_taken"]))
    check = "eye view vs env aim_err: max %.1e NDC" % resid
    prov = ("%s (%s env steps) vs pvpbot.eval.practice.PracticeHacker"
            % (os.path.relpath(ckpt, REPO), "{:,}".format(int(meta["step"]))))
    stampline = "all %d ticks checked · rendered %s" % (sel["ticks"], stamp)

    # canvas: the four-panel block keeps its verified pixel geometry; the
    # header and footer strips are added above and below it.
    W = int(round(a.width * 100))
    PH = int(round(a.panels * 100))
    H = PH + HDR_PX + FTR_PX
    fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
    fig.patch.set_facecolor(BG)

    def rect(l, b, w, h):
        """Panel rect given in the ORIGINAL panel-block fractions."""
        return [l, (FTR_PX + b * PH) / H, w, h * PH / H]

    eye = fig.add_axes(rect(0.006, 0.408, 0.622, 0.585))
    top = fig.add_axes(rect(0.636, 0.408, 0.358, 0.585))
    hud = fig.add_axes(rect(0.636, 0.012, 0.358, 0.383))
    tml = fig.add_axes(rect(0.006, 0.012, 0.622, 0.383))
    for ax in (eye, top, hud, tml):
        ax.set_facecolor(PANEL); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
    hdr = fig.add_axes([0.0, (H - HDR_PX) / H, 1.0, HDR_PX / H])
    ftr = fig.add_axes([0.0, 0.0, 1.0, FTR_PX / H])
    for ax in (hdr, ftr):
        for sp in ax.spines.values():
            sp.set_visible(False)
    draw_footer(ftr, W, cond, check, prov, stampline)

    bd = np.array([int(GRID[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)
    frames = []
    for i, t in enumerate(list(range(n)) + [n - 1] * 16):
        draw_eye(eye, d, t, ko=(t == n - 1))
        draw_top(top, d, t, cam[t][0], cam[t][1], half)
        draw_heads(hud, d, t)
        draw_timeline(tml, d, t)
        draw_header(hdr, W, t)
        fig.canvas.draw()
        f = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        f[0, :] = f[-1, :] = f[:, 0] = f[:, -1] = bd   # 1 px frame border
        frames.append(f)

    # ONE global palette. Median-cut alone drops rare-but-load-bearing colours:
    # pure white is the hit rings, the damage numbers and the K.O. text, and
    # over a 28-frame montage it is ~0.01% of the pixels, so it gets merged into
    # the nearest grey. A swatch strip is not enough at this montage size --
    # median-cut the frames for a reduced budget and APPEND the brand hexes, so
    # every one of them survives exactly (verified below on the written file).
    # in priority order: the locked brand palette first, then this figure's own
    # shades. Anchors nearer than L1 12 to an already-kept anchor are dropped --
    # two near-identical entries make the lookup ambiguous and BOTH end up
    # snapping to whichever the cache found first.
    want = [BG, PANEL, GRID, INK, DIM, CYAN, AMBER, RED, "#ffffff",
            "#e6edf3", "#7c5a3a", "#8a2b34", WALLTOP, HORIZ, WALL, SKY,
            GROUND, SLOT, GLINE, "#0b0e13"]
    anchors, rgb = [], []
    for h in want:
        c = [int(h[i:i + 2], 16) for i in (1, 3, 5)]
        if rgb and np.abs(np.array(rgb, np.int16) - np.array(c)).sum(1).min() <= 12:
            continue
        anchors.append(h); rgb.append(c)
    anc = np.array(rgb, np.int16)
    montage = np.concatenate(frames[::7], 0)
    base = Image.fromarray(montage).quantize(colors=a.colors,
                                             method=Image.MEDIANCUT)
    bp = np.array(base.getpalette()[:a.colors * 3], np.int16).reshape(-1, 3)
    # drop median-cut entries that sit on top of an anchor: Pillow's
    # palette lookup is a cached approximate nearest, so a near-duplicate
    # steals the mapping and #12161c lands 5/255 off its own swatch
    keep = [c for c in bp if np.abs(anc - c).sum(1).min() > 12]
    keep = keep[: a.colors - len(rgb)]
    plist = [int(v) for c in keep for v in c] + [v for c in rgb for v in c]
    plist += list(rgb[anchors.index(BG)]) * (256 - len(plist) // 3)
    pal = Image.new("P", (1, 1))
    pal.putpalette(plist[:768])
    qs = [Image.fromarray(f).quantize(palette=pal, dither=Image.NONE)
          for f in frames]
    # disposal=1 lets PIL emit inter-frame diffs; the panels' backgrounds are
    # static (the top-down camera only snaps), so the deltas stay small.
    qs[0].save(a.out, save_all=True, append_images=qs[1:], duration=50, loop=0,
               optimize=True, disposal=1)
    from PIL import ImageSequence
    done_im = Image.open(a.out)
    nstored = sum(1 for _ in ImageSequence.Iterator(done_im))
    # decode the written file back and assert every brand hex round-trips
    # through the palette LOOKUP (not merely that it is present in the table)
    probe = Image.new("P", (1, 1))
    probe.putpalette(Image.open(a.out).getpalette())
    bad = []
    for h, c in zip(anchors, rgb):
        sw = np.zeros((1, 1, 3), np.uint8)
        sw[:] = c
        out = np.asarray(Image.fromarray(sw).quantize(
            palette=probe, dither=Image.NONE).convert("RGB"))[0, 0]
        if list(int(v) for v in out) != c:
            bad.append("%s->#%02x%02x%02x" % (h, out[0], out[1], out[2]))
    print("palette: %d colours, brand hexes round-trip exact: %s"
          % (a.colors, "yes" if not bad else "NO -> %s" % bad))
    write_sidecar(a, d, sel, meta, ckpt, resid, stamp, W, H,
                  len(qs), nstored)
    print("%s  %d frames  %.2f MB" % (a.out, len(qs), os.path.getsize(a.out) / 1e6))


def write_sidecar(a, d, sel, meta, ckpt, resid, stamp, W, H, nframes, nstored):
    """docs/assets/data/<id>.json -- the numbers a caption is allowed to quote.

    runs/ is gitignored, so once the GIF is committed this file is the only
    remaining record of what was measured to produce it.
    """
    n = sel["ticks"]
    pos, dmg, act = d["pos"][:n], d["dmg"][:n], d["act"][:n]
    rd = d["reach"][:n, 0]
    gap = np.linalg.norm(pos[:, 1, [0, 2]] - pos[:, 0, [0, 2]], axis=1)
    land = dmg[:, 1][dmg[:, 1] > 0]
    took = dmg[:, 0][dmg[:, 0] > 0]
    opp_kb = float(np.hypot(d["vel"][:n, 1, 0], d["vel"][:n, 1, 2]).max())
    out = {
        "id": "duel-vs-p4hacker",
        "title": "The same duel, instrumented in four synchronised panels",
        "stage_badge": STAGE,
        "output": os.path.relpath(a.out, REPO),
        "generator": "tools/figures/anim_duel_vs_p4hacker.py",
        "rendered": stamp,
        "source": {
            "checkpoint": os.path.relpath(ckpt, REPO),
            "checkpoint_env_steps": int(meta["step"]),
            "opponent": "pvpbot.eval.practice.PracticeHacker",
            "env": "pvpbot.sim.env.DuelVecEnv",
            "condition": "SimConfig(arena_square=True, arena_radius=7.0) -- the 14x14 walled square, full-information observations",
            "seed": sel["seed"], "env_index": sel["env_index"],
        },
        "duel": {
            "ticks": n,
            "seconds": round(n / 20.0, 2),
            "outcome": "policy wins by K.O.",
            "hits_landed_by_policy": int(sel["hits_landed"]),
            "hits_taken_by_policy": int(sel["hits_taken"]),
            "damage_per_landed_hit": [round(float(v), 1) for v in land],
            "damage_per_taken_hit": [round(float(v), 1) for v in took],
            "crit_hits_landed": int((land > 2.0).sum()),
            "policy_hp_at_ko": round(float(d["hp"][n - 1, 0]), 1),
            "centre_gap_min_m": round(float(gap.min()), 2),
            "centre_gap_max_m": round(float(gap.max()), 2),
            "reach_dist_min_m": round(float(rd.min()), 2),
            "reach_dist_max_m": round(float(rd.max()), 2),
            "ticks_inside_3m_reach": int((rd <= REACH).sum()),
            "reach_metric": ("eye -> opponent AABB, pvpbot/sim/env.py:615-623; "
                             "a swing connects at <= 3.0 m"),
            "policy_attack_head_ticks": int((act[:, 0, 4] == 1).sum()),
            "policy_sprint_ticks": int((act[:, 0, 3] == 1).sum()),
            "policy_jump_ticks": int((act[:, 0, 2] == 1).sum()),
            "policy_airborne_ticks": int((~d["on_ground"][:n, 0]).sum()),
            "yaw_head_bins_used": int(len(set(act[:, 0, 5].tolist()))),
            "opponent_peak_knockback_blocks_per_tick": round(opp_kb, 3),
        },
        "verification": {
            "what": ("eye-view pinhole projection of the enemy hitbox centre "
                     "vs the closed form implied by the env's own "
                     "aim_err_yaw / aim_err_pitch obs channels, every tick"),
            "max_residual_ndc": float("%.3g" % resid),
            # the eye panel is 0.622 * W px wide and spans 2 * AR NDC units
            "max_residual_px_in_eye_panel":
                float("%.2g" % (resid * 0.622 * W / (2.0 * AR))),
            "ticks_checked": n,
        },
        "encoding": {
            "width": W, "height": H, "frames_rendered": nframes,
            "frames_stored_after_pillow_dedup": nstored,
            "fps": 20, "frame_duration_ms": 50, "palette_colors": a.colors,
            "disposal": 1, "bytes": os.path.getsize(a.out),
        },
    }
    p = os.path.join(REPO, "docs", "assets", "data", "duel-vs-p4hacker.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    print("sidecar: %s" % os.path.relpath(p, REPO))


if __name__ == "__main__":
    main()
