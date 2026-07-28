"""The same seeded duel, fought by three checkpoints from three run dirs.

Three top-down arena panels under one shared tick clock.  Each panel is a real
rollout of ``pvpbot.sim.env.DuelVecEnv(1, seed=3, config=SimConfig())`` -- the
default full-information physics -- with a PolicyNet checkpoint on side 0 and
``pvpbot.eval.scripted.Pro`` (T4) on side 1:

    runs/integration-check/ckpt_latest.pt      5,242,880 env-steps
    runs/overnight/ckpt_latest.pt            83,886,080 env-steps
    runs/fov1/ckpt_live_34.3B.pt         34,644,951,040 env-steps

Because the env seed is identical, all three duels open from the *same* spawn:
policy at (4.61, 2.75) yaw 33.9 deg, T4-Pro at (0.38, 4.56) yaw 155.9 deg.
Frame 0 shows that, so the difference between the three panels cannot be
written off as luck.

Everything drawn is per-tick ground truth snapshotted BEFORE each env.step()
(the env auto-resets on the done tick, so post-step state belongs to a fresh
duel):  env.pos, env.yaw, env.hp, env.hurt, env.sprinting, plus the attack
head of the action array actually fed to step().  A hit is a drop in env.hp
between consecutive ticks; the killing blow is recovered from info["win"].

The 64-duel records printed in each panel header come from
``pvpbot.eval.arena.run_match(ckpt, Pro(0), 64, seed=7)`` -- 14x14 square arena,
sides swapped per env slot -- and are recomputed at render time unless
--skip-records is passed.  No per-run Elo is shown anywhere: the ``elo`` in a
checkpoint's meta is league-internal to its own run directory (the 5.2M one is
the initial 1000) and is not comparable across run directories.

Sources are gitignored (runs/ is not committed), so a fresh clone cannot
re-render this; the script exits with a named-file message instead of a stack
trace, and the derived numbers are mirrored into
docs/assets/data/policy-across-training.json.

Output is one GIF frame per 50 ms sim tick at duration=50 -- true 20 fps,
real time, no strided frames -- so no single-tick hit or swing can vanish
between frames.

Usage:
    python3 tools/figures/anim_policy_across_training.py
    python3 tools/figures/anim_policy_across_training.py --stride 2 --duration 100
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
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle, Wedge
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

# ---------------------------------------------------------------------------
# palette -- locked, shared with tools/figures/anim_live_cockpit.py
# ---------------------------------------------------------------------------
BG, PANEL, GRID, INK, DIM = "#12161c", "#1a212a", "#2b3542", "#ccd6e2", "#66747f"
CYAN, AMBER, RED, GREEN, VIOLET = "#3fd0d8", "#e8a33d", "#e2564a", "#6ecf94", "#a184e8"

MONO = "Menlo"

# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
CKPTS = [
    ("runs/integration-check/ckpt_latest.pt", "integration-check"),
    ("runs/overnight/ckpt_latest.pt", "overnight"),
    ("runs/fov1/ckpt_live_34.3B.pt", "fov1 / ckpt_live_34.3B"),
]
ENV_SEED = 3          # DuelVecEnv seed -> the identical spawn in all 3 panels
TORCH_SEED = 0        # PolicyNet.act samples off the GLOBAL torch RNG
RECORD_SEED = 7       # seed for the 64-duel arena records in the headers
RECORD_DUELS = 64
MAX_TICKS = 600
OUT = os.path.join(REPO, "docs", "assets", "policy-across-training.gif")
SIDECAR = os.path.join(REPO, "docs", "assets", "data", "policy-across-training.json")

ARENA_R = 7.0         # SimConfig.arena_radius as HALF-SIDE (14x14 square)

# The live arena is a 14x14 walled square (pvpbot/sim/env.py:181), so every
# rollout and every recorded duel here runs with arena_square=True. Both the
# config and the run_match env subclass are built inside main(), because
# DuelVecEnv/SimConfig are imported lazily there so --help works without torch.

REACH = 3.0           # SimConfig.reach, blocks
HALF_W = 0.3          # hitbox half-width, blocks
RING_TICKS = 6        # ticks a hit ring stays up (0.30 s)


def ordinal(n):
    return "%d%s" % (n, "th" if 10 <= n % 100 < 20
                     else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th"))


def die(msg):
    sys.stderr.write("anim_policy_across_training: %s\n" % msg)
    raise SystemExit(2)


def human_steps(n):
    if n >= 1e9:
        return "%.1fB" % (n / 1e9)
    if n >= 1e6:
        return "%.1fM" % (n / 1e6)
    return "%d" % n


# ---------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------
def rollout(contestant, torch, Pro, DuelVecEnv, SimConfig):
    """One duel: contestant (side 0) vs scripted T4-Pro (side 1)."""
    torch.manual_seed(TORCH_SEED)          # act() samples off the global RNG
    opp = Pro(0)
    env = DuelVecEnv(1, seed=ENV_SEED,
                     config=SimConfig(arena_square=True,
                                      arena_radius=ARENA_R))
    obs = env.reset()
    contestant.begin(1)
    opp.begin(1)

    pos, yaw, hp, hurt, spr, atk = [], [], [], [], [], []
    win = np.zeros(2, dtype=np.float32)
    for _ in range(MAX_TICKS):
        acts = np.stack([contestant.act(obs[:, 0]), opp.act(obs[:, 1])], axis=1)
        # snapshot BEFORE stepping -- step() auto-resets finished envs
        pos.append(env.pos[0].copy())
        yaw.append(env.yaw[0].copy())
        hp.append(env.hp[0].copy())
        hurt.append(env.hurt[0].copy())
        spr.append(env.sprinting[0].copy())
        atk.append(acts[0, :, 4] == 1)
        obs, _, done, info = env.step(acts)
        contestant.on_done(done)
        opp.on_done(done)
        if done[0]:
            win = np.asarray(info["win"][0], dtype=np.float32).copy()
            break

    tr = {k: np.asarray(v) for k, v in
          dict(pos=pos, yaw=yaw, hp=hp, hurt=hurt, spr=spr, atk=atk).items()}
    # damage resolved on tick t is visible as the hp drop into tick t+1
    dmg = np.vstack([np.zeros((1, 2), np.float32), -np.diff(tr["hp"], axis=0)])
    tr["dmg"] = np.maximum(dmg, 0.0)
    tr["win"] = win
    tr["T"] = len(tr["hp"])
    # killing blow: hp on the done tick is the pre-step value, so the fatal
    # damage is whatever the loser had left
    loser = 0 if win[1] > 0.5 else 1
    tr["fatal_side"] = loser
    tr["hits_landed"] = int((tr["dmg"][:, 1] > 1e-4).sum()) + (1 if loser == 1 else 0)
    tr["hits_taken"] = int((tr["dmg"][:, 0] > 1e-4).sum()) + (1 if loser == 0 else 0)
    return tr


# ---------------------------------------------------------------------------
# per-panel drawing
# ---------------------------------------------------------------------------
def hit_frames(tr, ticks_of_frame, side):
    """Frame indices on which a hit against `side` should flash.

    Frames are strided, so every tick inside a frame's group is OR-ed in:
    a single-tick hit event can never fall between two rendered frames.
    """
    out = {}
    for fi, grp in enumerate(ticks_of_frame):
        for t in grp:
            if t < tr["T"] and tr["dmg"][t, side] > 1e-4:
                out.setdefault(fi, []).append(t)
    # the killing blow lands on the final tick and never shows as an hp drop
    if tr["fatal_side"] == side:
        out.setdefault(len(ticks_of_frame) - 1, []).append(tr["T"] - 1)
    return out


def draw_arena(ax, tr, ti, rings_g, rings_r, fi, ring_hold):
    ax.clear()
    ax.set_facecolor(BG)
    ax.set_xlim(-9, 9)
    ax.set_ylim(-9, 9)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
        sp.set_linewidth(0.9)

    # arena floor (static, filled -- cheap for GIF inter-frame diffs)
    ax.add_patch(Rectangle((-ARENA_R, -ARENA_R), 2 * ARENA_R, 2 * ARENA_R,
                           facecolor=PANEL, edgecolor=GRID, lw=1.1, zorder=0))
    # inner 8x8 reference box, concentric with the wall -- a dashed CIRCLE here
    # reads as a second, round boundary and there is no such thing on a square
    # arena.
    ax.add_patch(Rectangle((-4.0, -4.0), 8.0, 8.0, facecolor="none",
                           edgecolor=GRID, lw=0.6, ls=(0, (3, 4)), zorder=1))

    live = min(ti, tr["T"] - 1)
    P = tr["pos"][:live + 1]
    Y = tr["yaw"][live]

    # opponent trail: short and faint
    if live >= 2:
        q = P[max(0, live - 20):, 1]
        seg = np.stack([q[:-1, [0, 2]], q[1:, [0, 2]]], axis=1)
        a = np.linspace(0.05, 0.30, len(seg))
        lc = LineCollection(seg, colors=[(0.910, 0.639, 0.239, x) for x in a],
                            linewidths=1.1, zorder=2)
        ax.add_collection(lc)

    # policy trail: FULL history, alpha ramped 0.10 -> 0.65 over the duel.
    # The ramp is indexed by ABSOLUTE tick, not by position in the visible
    # history: a segment therefore keeps the same colour once drawn, so the
    # old trail contributes nothing to the GIF's inter-frame delta (ramping
    # over the visible history instead recolours every trail pixel every
    # frame and cost 0.20 MB when measured).
    if live >= 2:
        q = P[:, 0]
        seg = np.stack([q[:-1, [0, 2]], q[1:, [0, 2]]], axis=1)
        a = 0.10 + 0.55 * np.arange(len(seg)) / max(1.0, tr["T"] - 2.0)
        lc = LineCollection(seg, colors=[(0.247, 0.816, 0.847, x) for x in a],
                            linewidths=1.6, zorder=3)
        ax.add_collection(lc)

    # reach wedges + aim rays + hitbox footprints
    dead = tr["fatal_side"] if ti >= tr["T"] - 1 else -1
    for side, col in ((1, AMBER), (0, CYAN)):
        x, z = P[live, side, 0], P[live, side, 2]
        yw = float(Y[side])
        fc = DIM if side == dead else col
        ax.add_patch(Wedge((x, z), REACH, yw - 7.0, yw + 7.0, facecolor=fc,
                           alpha=0.30, edgecolor="none", zorder=4))
        ax.plot([x, x + REACH * np.cos(np.radians(yw))],
                [z, z + REACH * np.sin(np.radians(yw))],
                color=fc, lw=0.9, alpha=0.55, zorder=5, solid_capstyle="butt")
        ax.add_patch(Rectangle((x - HALF_W, z - HALF_W), 2 * HALF_W, 2 * HALF_W,
                               facecolor=fc, edgecolor="#0b0e12", lw=0.7,
                               zorder=7))

    # hit rings -- green when the policy lands one, red when it takes one.
    # The ring tracks the victim (knockback moves it ~0.9 blocks), so it always
    # reads as "this fighter is being hit".
    for rings, side, col in ((rings_g, 1, GREEN), (rings_r, 0, RED)):
        for f0, tks in rings.items():
            age = fi - f0
            if 0 <= age < ring_hold:
                u = age / max(1.0, ring_hold - 1.0)
                x, z = P[live, side, 0], P[live, side, 2]
                # full-strength brand colour, never alpha-faded: an alpha ring
                # blends to a washed #6fb09c and stops reading as GREEN
                ax.add_patch(Circle((x, z), 0.85 + 1.15 * u, facecolor="none",
                                    edgecolor=col, lw=3.0 - 1.7 * u, zorder=9))
                if age == 0 and side != dead:
                    ax.add_patch(Rectangle(
                        (x - HALF_W, z - HALF_W), 2 * HALF_W, 2 * HALF_W,
                        facecolor=col, edgecolor=col, lw=1.6, zorder=10))

    # corner readouts -- outside the r=8 disc, so they never touch a trail
    gap = float(np.hypot(P[live, 0, 0] - P[live, 1, 0],
                         P[live, 0, 2] - P[live, 1, 2]))
    # how close the policy is to a wall. The clamp is axis-independent
    # (pvpbot/sim/env.py:556-560), so the binding distance is Chebyshev:
    # a radial readout would report 9.9 in a corner of a half-side-7.0 box.
    rim = float(max(abs(P[live, 0, 0]), abs(P[live, 0, 2])))
    wall = ARENA_R - rim
    ax.text(-8.7, -8.45, "gap %4.2f blk" % gap, color=INK, fontsize=7.2)
    ax.text(8.7, -8.45, "policy wall %.1f blk" % wall,
            color=RED if wall < 1.0 else CYAN, fontsize=7.2, ha="right")


def draw_raster(ax, tr, ti, tmax, ticks_of_frame, fi):
    ax.clear()
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, tmax)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
        sp.set_linewidth(0.8)
    shown = min(ti, tr["T"] - 1)
    for t in range(tr["T"]):
        if t > shown:
            break
        if tr["dmg"][t, 1] > 1e-4:
            ax.add_patch(Rectangle((t - 0.8, 0.53), 2.8, 0.43, facecolor=GREEN,
                                   edgecolor="none"))
        if tr["dmg"][t, 0] > 1e-4:
            ax.add_patch(Rectangle((t - 0.8, 0.04), 2.8, 0.43, facecolor=RED,
                                   edgecolor="none"))
    if tr["T"] - 1 <= shown:                       # killing blow
        c = GREEN if tr["fatal_side"] == 1 else RED
        y0 = 0.53 if tr["fatal_side"] == 1 else 0.04
        ax.add_patch(Rectangle((tr["T"] - 1.8, y0), 2.8, 0.43, facecolor=c,
                               edgecolor="none"))
    ax.axvline(min(ti, tr["T"] - 1), color=INK, lw=0.9, alpha=0.75)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--width", type=int, default=1100)
    ap.add_argument("--height", type=int, default=580)
    ap.add_argument("--colors", type=int, default=48)
    ap.add_argument("--hold", type=int, default=30)
    ap.add_argument("--duration", type=int, default=50)
    ap.add_argument("--skip-records", action="store_true",
                    help="skip the 64-duel arena sweep and reuse the sidecar")
    args = ap.parse_args()

    for rel, _ in CKPTS:
        p = os.path.join(REPO, rel)
        if not os.path.exists(p):
            die("source checkpoint no longer on disk: %s  "
                "(runs/ is gitignored; this figure cannot be re-rendered from "
                "a fresh clone)" % rel)

    try:
        import torch  # noqa: F401
    except ImportError:
        die("torch is required to roll out the checkpoints (pip install torch)")

    from pvpbot.eval.arena import CheckpointContestant, run_match
    from pvpbot.eval.scripted import Pro
    from pvpbot.sim.env import DuelVecEnv, SimConfig

    arena_cfg = SimConfig(arena_square=True, arena_radius=ARENA_R)

    class SquareArenaEnv(DuelVecEnv):
        """run_match() builds its env as cls(n, seed=seed) with no config
        hook, so the 14x14 square arena is baked in here."""

        def __init__(self, num_envs, seed=0, **kw):
            super().__init__(num_envs, seed=seed, config=arena_cfg)


    # ---- rollouts ---------------------------------------------------------
    trs, metas, records = [], [], []
    for rel, short in CKPTS:
        c = CheckpointContestant(os.path.join(REPO, rel))
        metas.append(dict(c.meta))
        if args.skip_records:
            records.append(None)
        else:
            torch.manual_seed(RECORD_SEED)     # act() samples off the global RNG
            r = run_match(c, Pro(0), RECORD_DUELS, seed=RECORD_SEED,
                          env_cls=SquareArenaEnv)
            records.append((r.wins_a, r.wins_b, r.draws,
                            r.stats_a.mean_aim_err_deg))
        trs.append(rollout(c, torch, Pro, DuelVecEnv, SimConfig))
        print("%-22s step %14d  %3d ticks  %s  landed %2d  taken %2d"
              % (short, metas[-1]["step"], trs[-1]["T"],
                 "WIN " if trs[-1]["win"][0] > 0.5 else "LOSS",
                 trs[-1]["hits_landed"], trs[-1]["hits_taken"]))

    if args.skip_records and os.path.exists(SIDECAR):
        old = json.load(open(SIDECAR))
        records = [(p["record_w"], p["record_l"], p["record_d"],
                    p["mean_aim_err_deg"]) for p in old["panels"]]
    elif args.skip_records:
        die("--skip-records needs a previous sidecar at docs/assets/data/"
            "policy-across-training.json")

    # spawn must be identical across panels -- that is the whole comparison
    sp0 = trs[0]["pos"][0]
    for tr in trs[1:]:
        if not np.allclose(sp0, tr["pos"][0], atol=1e-6):
            die("spawns diverged between panels; the identical-spawn claim on "
                "frame 0 would be false")

    tmax = max(tr["T"] for tr in trs)
    ticks_of_frame = [list(range(t0, min(t0 + args.stride, tmax)))
                      for t0 in range(0, tmax, args.stride)]
    nframes = len(ticks_of_frame)
    rings_g = [hit_frames(tr, ticks_of_frame, 1) for tr in trs]
    rings_r = [hit_frames(tr, ticks_of_frame, 0) for tr in trs]
    ring_hold = max(2, int(round(float(RING_TICKS) / args.stride)))
    gaps_med = []
    for tr in trs:                      # median ticks between LANDED hits
        g = np.diff(np.where(tr["dmg"][:, 1] > 1e-4)[0])
        gaps_med.append(int(round(float(np.median(g)))) if len(g) else 0)

    # playback speed relative to the 20 Hz sim; stride 1 @ 50 ms is real time
    speed = args.stride * 50.0 / args.duration
    if abs(speed - 1.0) < 0.02:
        timing = ("one frame per 50 ms tick · 20 fps · real time"
                  if args.stride == 1 else
                  "one frame per %d ticks held %d ms · real time"
                  % (args.stride, args.duration))
    else:
        timing = ("every %s tick held %d ms · %.2fx %s than the 20 Hz sim"
                  % (ordinal(args.stride), args.duration, max(speed, 1 / speed),
                     "faster" if speed > 1 else "slower"))

    # ---- figure scaffolding (explicit rects; nothing moves between frames)--
    W, H = args.width, args.height
    plt.rcParams["font.family"] = "monospace"
    plt.rcParams["font.monospace"] = [MONO, "DejaVu Sans Mono"]
    fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
    fig.patch.set_facecolor(BG)

    pw = 0.308
    ph = pw * W / H                      # square in pixels: no aspect shrink
    x0s = [0.032 + 0.3175 * k for k in range(3)]
    AR_B = 0.150
    RA_B, RA_H = 0.096, 0.028
    axes_a = [fig.add_axes([x, AR_B, pw, ph]) for x in x0s]
    axes_r = [fig.add_axes([x, RA_B, pw, RA_H]) for x in x0s]
    for ax in axes_a:
        ax.set_aspect("equal", adjustable="box")
    AR_T = AR_B + ph                     # 0.732 at 1100x580

    stamp = datetime.datetime.now().strftime("%Y-%m-%d")
    frames = []

    for fi in range(nframes + args.hold):
        f = min(fi, nframes - 1)
        ti = ticks_of_frame[f][-1]
        for t in fig.texts[:]:
            t.remove()
        del fig.patches[:]

        steps_all = [float(m["step"]) for m in metas]
        # ---- header strip -------------------------------------------------
        fig.patches.append(Rectangle((0, 0.898), 1, 0.102, transform=fig.transFigure,
                                     facecolor=PANEL, edgecolor=GRID, lw=0.8,
                                     zorder=0))
        fig.text(0.013, 0.933, "M", color=AMBER, fontsize=12.5, fontweight="bold")
        fig.text(0.030, 0.944, "SIM MIRROR", color=INK, fontsize=10.5,
                 fontweight="bold")
        fig.text(0.030, 0.917, "DuelVecEnv · one seeded duel, three checkpoints "
                 "from three run directories", color=DIM, fontsize=7.2)
        fig.text(0.986, 0.944, "tick %04d · t=%5.2f s" % (ti, ti * 0.05),
                 color=INK, fontsize=11.5, ha="right", fontweight="bold")
        if ti < 30:
            fig.text(0.986, 0.917, "IDENTICAL SPAWN   policy (%.2f, %.2f) yaw "
                     "%.1f°   T4-Pro (%.2f, %.2f) yaw %.1f°"
                     % (sp0[0, 0], sp0[0, 2], trs[0]["yaw"][0, 0],
                        sp0[1, 0], sp0[1, 2], trs[0]["yaw"][0, 1]),
                     color=GREEN, fontsize=7.2, ha="right", fontweight="bold")
        else:
            fig.text(0.986, 0.917, "same spawn, same env seed %d, same opponent "
                     "in all three panels" % ENV_SEED,
                     color=DIM, fontsize=7.2, ha="right")

        # ---- per-panel header + arena + raster -----------------------------
        for k, tr in enumerate(trs):
            x = x0s[k]
            live = min(ti, tr["T"] - 1)
            done = ti >= tr["T"] - 1
            hp = tr["hp"][live].copy()
            if done:                       # loser is dead on the final tick
                hp[tr["fatal_side"]] = 0.0
            landed = int((tr["dmg"][:live + 1, 1] > 1e-4).sum())
            taken = int((tr["dmg"][:live + 1, 0] > 1e-4).sum())
            if done:
                landed, taken = tr["hits_landed"], tr["hits_taken"]

            fig.text(x, 0.858, "%s env-steps" % human_steps(metas[k]["step"]),
                     color=INK, fontsize=11.5, fontweight="bold")
            # the unit suffix carries the whole story (M -> M -> B) while the
            # mantissa falls, so state the multiplier outright and draw it too.
            mult = float(metas[k]["step"]) / float(metas[0]["step"])
            fig.text(x + 0.207, 0.858,
                     "×1" if k == 0 else "×%s" % format(int(round(mult)), ","),
                     color=AMBER if k == 2 else DIM, fontsize=10.0,
                     fontweight="bold", ha="right")
            # shared log scale across the three panels: equal bar length would
            # imply equal training, and these differ by 6,608x.
            lo, hi = np.log10(steps_all[0]), np.log10(steps_all[-1])
            frac = (np.log10(float(metas[k]["step"])) - lo) / (hi - lo)
            fig.patches.append(Rectangle((x, 0.846), 0.207, 0.0055,
                                         transform=fig.transFigure,
                                         facecolor=GRID, ec="none", zorder=1))
            fig.patches.append(Rectangle((x, 0.846), 0.207 * max(frac, 0.012),
                                         0.0055, transform=fig.transFigure,
                                         facecolor=CYAN if k == 2 else DIM,
                                         ec="none", zorder=2))
            w, l, d, aim = records[k]
            fig.text(x, 0.826, "%d duels vs T4-Pro  %d-%d-%d  ·  mean aim err "
                     "%.1f°" % (RECORD_DUELS, w, l, d, aim),
                     color=DIM, fontsize=6.8)

            for j, (side, col, nm) in enumerate(((0, CYAN, "POLICY"),
                                                 (1, AMBER, "T4-PRO"))):
                yb = 0.797 - 0.026 * j
                fig.text(x, yb, nm, color=col, fontsize=6.8)
                fig.patches.append(Rectangle((x + 0.045, yb - 0.001), 0.196, 0.013,
                                             transform=fig.transFigure,
                                             facecolor=BG, edgecolor=GRID,
                                             lw=0.6, zorder=1))
                frac = max(0.0, float(hp[side]) / 20.0)
                if frac > 0:
                    fig.patches.append(Rectangle((x + 0.045, yb - 0.001),
                                                 0.196 * frac, 0.013,
                                                 transform=fig.transFigure,
                                                 facecolor=col, edgecolor="none",
                                                 zorder=2))
                fig.text(x + 0.247, yb, "%4.1f" % hp[side], color=col,
                         fontsize=6.8, fontweight="bold")

            fig.text(x, 0.745, "hits landed", color=DIM, fontsize=6.8)
            fig.text(x + 0.080, 0.743, "%02d" % landed, color=GREEN,
                     fontsize=8.8, fontweight="bold")
            fig.text(x + 0.109, 0.745, "taken", color=DIM, fontsize=6.8)
            fig.text(x + 0.152, 0.743, "%02d" % taken, color=RED, fontsize=8.8,
                     fontweight="bold")
            if done:
                won = tr["win"][0] > 0.5
                fig.text(x + pw, 0.741, "WIN" if won else "LOSS",
                         color=GREEN if won else RED, fontsize=14,
                         ha="right", fontweight="bold")

            draw_arena(axes_a[k], tr, ti, rings_g[k], rings_r[k], f, ring_hold)
            draw_raster(axes_r[k], tr, ti, tmax, ticks_of_frame, f)

        fig.text(x0s[0], RA_B + RA_H + 0.007, "hit raster · %d ticks" % tmax,
                 color=DIM, fontsize=6.6)
        for k in range(3):
            g = gaps_med[k]
            fig.text(x0s[k] + pw, RA_B + RA_H + 0.007,
                     "landed hits every %d tk (med)" % g if g else
                     "one landed hit, no cadence", color=DIM, fontsize=6.6,
                     ha="right")

        # ---- footer: legend + condition ------------------------------------
        fig.patches.append(Rectangle((0, 0.0), 1, 0.078, transform=fig.transFigure,
                                     facecolor=PANEL, edgecolor=GRID, lw=0.8,
                                     zorder=0))
        ly = 0.052
        lx = 0.032
        for col, lab in ((CYAN, "policy"), (AMBER, "T4-Pro (scripted T4)")):
            fig.patches.append(Rectangle((lx, ly - 0.002), 0.0090, 0.015,
                                         transform=fig.transFigure,
                                         facecolor=col, edgecolor="none", zorder=2))
            fig.text(lx + 0.014, ly, lab, color=INK, fontsize=7.4)
            lx += 0.024 + 0.0066 * len(lab)
        fig.patches.append(Wedge((lx + 0.002, ly + 0.005), 0.013, -22, 22,
                                 transform=fig.transFigure, facecolor=CYAN,
                                 alpha=0.45, edgecolor="none", zorder=2))
        fig.text(lx + 0.020, ly, "wedge = 3.0-block reach, ±7°", color=INK,
                 fontsize=7.4)
        lx += 0.020 + 0.0066 * 28
        fig.patches.append(Rectangle((lx + 0.002, ly + 0.0015), 0.020, 0.0055,
                                     transform=fig.transFigure, facecolor=CYAN,
                                     ec="none", zorder=2))
        fig.text(lx + 0.027, ly, "header bar = env-steps, log scale", color=INK,
                 fontsize=7.4)
        lx += 0.027 + 0.0066 * 33
        for col, lab in ((GREEN, "hit landed"), (RED, "hit taken")):
            fig.patches.append(Circle((lx + 0.006, ly + 0.005), 0.0070,
                                      transform=fig.transFigure, facecolor="none",
                                      edgecolor=col, lw=1.7, zorder=2))
            fig.text(lx + 0.018, ly, "ring = %s" % lab, color=INK, fontsize=7.4)
            lx += 0.026 + 0.0066 * len("ring = " + lab)

        fig.text(0.032, 0.030, "DuelVecEnv(1, seed=%d, 14x14 square arena) · "
                 "default 1.8 physics, full-information observations · %s"
                 % (ENV_SEED, timing), color=DIM, fontsize=6.8)
        fig.text(0.032, 0.011, "  ·  ".join(r for r, _ in CKPTS),
                 color=DIM, fontsize=6.8)
        fig.text(0.986, 0.011, "tools/figures/anim_policy_across_training.py · "
                 "rendered %s" % stamp, color=DIM, fontsize=6.8, ha="right")

        fig.canvas.draw()
        a = np.array(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
        # 1 px border so the raster reads as a deliberate screen in both
        # GitHub light and dark themes
        bd = [int(GRID[i:i + 2], 16) for i in (1, 3, 5)]
        a[0, :] = bd
        a[-1, :] = bd
        a[:, 0] = bd
        a[:, -1] = bd
        frames.append(Image.fromarray(a))

    plt.close(fig)

    # ---- one global palette, brand hexes pinned in -------------------------
    # median-cut the raw frames for (colors - 10) entries, then APPEND the ten
    # brand hexes verbatim.  Appending a swatch strip to the montage is not
    # enough at 48 colours: median-cut merges RED into AMBER and every hit
    # ring comes out orange.
    brand = [BG, PANEL, GRID, INK, DIM, CYAN, AMBER, RED, GREEN, VIOLET]
    nc = args.colors - len(brand)
    mont = np.concatenate([np.asarray(f) for f in frames[::5]], axis=0)
    base = Image.fromarray(mont).quantize(colors=nc, method=Image.MEDIANCUT,
                                          dither=Image.NONE)
    pal = list(base.getpalette()[:nc * 3])
    for hx in brand:
        pal += [int(hx[j:j + 2], 16) for j in (1, 3, 5)]
    # Pad to exactly 64 entries, never 256: GIF colour tables are written at
    # the next power of two, and a 256-entry table costs 768 bytes on EVERY
    # frame (198 KB over this clip) against 192 bytes for a 64-entry one.
    # Index 63 is a magenta no pixel can quantise onto -> the transparent slot.
    bgc = [int(BG[j:j + 2], 16) for j in (1, 3, 5)]
    while len(pal) // 3 < 63:
        pal += bgc
    pal += [255, 0, 255]
    TRANS = 63
    pal_img = Image.new("P", (1, 1))
    pal_img.putpalette(pal[:192])
    q = [f.quantize(palette=pal_img, dither=Image.NONE) for f in frames]

    # Inter-frame diff: every pixel identical to the previous frame becomes the
    # reserved transparent index, and disposal=1 leaves it showing through.
    # Pillow's own optimiser only crops to one bounding box per frame, which
    # here is the whole canvas (the clock is top-right, the raster bottom-left),
    # so this is the difference between 1.31 MB and the budget.
    idx = [np.array(f) for f in q]
    diffed = [q[0]]
    for i in range(1, len(idx)):
        b = idx[i].copy()
        b[idx[i] == idx[i - 1]] = TRANS
        im = Image.fromarray(b)
        im.putpalette(pal[:192])
        diffed.append(im)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    diffed[0].save(args.out, save_all=True, append_images=diffed[1:],
                   duration=args.duration, loop=0, optimize=True, disposal=1,
                   transparency=TRANS)
    size = os.path.getsize(args.out)
    print("wrote %s  %dx%d  %d frames (%d live + %d held)  %.2f MB"
          % (os.path.relpath(args.out, REPO), W, H, len(q), nframes,
             args.hold, size / 1e6))

    # ---- sidecar -----------------------------------------------------------
    os.makedirs(os.path.dirname(SIDECAR), exist_ok=True)
    panels = []
    for k, tr in enumerate(trs):
        gaps = np.diff(np.where(tr["dmg"][:, 1] > 1e-4)[0])
        panels.append(dict(
            checkpoint=CKPTS[k][0], short=CKPTS[k][1],
            env_steps=int(metas[k]["step"]),
            label="%s env-steps" % human_steps(metas[k]["step"]),
            duel_ticks=int(tr["T"]),
            result="win" if tr["win"][0] > 0.5 else "loss",
            hits_landed=int(tr["hits_landed"]), hits_taken=int(tr["hits_taken"]),
            median_landed_gap_ticks=(float(np.median(gaps)) if len(gaps) else None),
            record_w=int(records[k][0]), record_l=int(records[k][1]),
            record_d=int(records[k][2]),
            mean_aim_err_deg=round(float(records[k][3]), 2)))
    json.dump(dict(
        id="policy-across-training", rendered=stamp,
        gif=os.path.relpath(args.out, REPO),
        width=W, height=H, frames=len(q), live_frames=nframes,
        held_frames=args.hold, stride_ticks=args.stride,
        frame_duration_ms=args.duration, bytes=size,
        env=("DuelVecEnv(1, seed=%d, config=SimConfig("
             "arena_square=True, arena_radius=7.0))" % ENV_SEED),
        opponent="pvpbot.eval.scripted.Pro (T4)",
        torch_seed=TORCH_SEED,
        records="pvpbot.eval.arena.run_match(ckpt, Pro(0), %d, seed=%d)"
                % (RECORD_DUELS, RECORD_SEED),
        spawn=dict(policy=[round(float(sp0[0, 0]), 4), round(float(sp0[0, 2]), 4)],
                   policy_yaw_deg=round(float(trs[0]["yaw"][0, 0]), 3),
                   opponent=[round(float(sp0[1, 0]), 4), round(float(sp0[1, 2]), 4)],
                   opponent_yaw_deg=round(float(trs[0]["yaw"][0, 1]), 3)),
        panels=panels), open(SIDECAR, "w"), indent=1)
    print("wrote %s" % os.path.relpath(SIDECAR, REPO))


if __name__ == "__main__":
    main()
