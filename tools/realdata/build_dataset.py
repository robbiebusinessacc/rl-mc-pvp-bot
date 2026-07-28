"""Align captured real frames with target-bot ground truth into a labeled
perception dataset (NPZ: frames u8 (N,64,114), labels f32 (N,12),
mask f32 (N,12)).

Label conventions deliberately mirror pvpbot/perception/synth.py so the
same model serves both data sources and dist_from_bbox_height stays the
single inverse-perspective authority:
  * aim errors: synth-frame signed angles (/180, /90; pitch DOWN-positive
    per synth's "+ = below crosshair"; the adapter flips into the sim
    frame), computed from server ground truth (camera feet pos + 1.62 eye
    height, target center at feet + 0.9).
  * bbox_height: synth's bbox_height_frac(dist).
  * rel_screen_v*: px/frame via synth PX_PER_DEG, normalized by frame dims.
  * visible == 0 zeroes all enemy-derived labels (synth convention). The
    visibility test uses the REAL client FOV (vertical --fov, horizontal
    from the window aspect), not the synth camera.
  * mask: slots with no usable ground truth in a row get 0 loss weight
    (e.g. rel_screen on the first frame after an alignment gap).

Mineflayer angle conventions (logged raw): notchian_yaw_deg = deg(pi - cy),
mc_pitch_deg (down-positive) = -deg(cpt); sim pitch = -mc_pitch.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_THIS)))

from pvpbot.perception.synth import (  # noqa: E402
    FRAME_H,
    FRAME_W,
    PX_PER_DEG,
    bbox_height_frac,
)
from pvpbot.spec import PERCEPTION_LAYOUT  # noqa: E402

_L = PERCEPTION_LAYOUT
EYE = 1.62
CENTER = 0.9
HURT_FLASH_MS = 450.0


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def solve_head_yaw(flight_path, frame_times, rows, row_t, hurt_t, lag_ms):
    """True head-yaw track from hit anchors + integrated yaw commands.

    The bot log's ``cy`` is the camera player's BODY yaw (mineflayer's view
    of a remote player), which sits up to ~90 deg from the crosshair while
    strafing -- labels built from it are fiction in combat data. Physics
    gives a better track: at every landed hit the crosshair was ON the box
    (aim err ~ 0), so head_yaw(t_hit) = bearing(t_hit); between anchors,
    integrate the yaw deltas the flight log actually injected (verified
    1:1 next-frame camera response), spreading each segment's closure
    residual linearly (absorbs gain error and drift).

    Returns (yaw_at(t_ms) -> deg or None). None outside anchored coverage.
    """
    from pvpbot.spec import CAMERA_BINS  # noqa: PLC0415

    bins = np.asarray(CAMERA_BINS, dtype=np.float64)
    fr = [json.loads(l) for l in open(flight_path)]
    settled_idx = [i for i, r in enumerate(fr) if r.get("settled")]
    if len(settled_idx) < 10 or len(frame_times) < 10:
        return lambda t: None
    m = min(len(settled_idx), len(frame_times))
    # epoch time of every flight tick: settled ticks pair 1:1 with dumped
    # frames (same loop, same order); interpolate the rest
    tick_rel = np.asarray([r["t"] for r in fr], dtype=np.float64)
    ep = np.interp(tick_rel,
                   tick_rel[settled_idx[:m]], np.asarray(frame_times[:m]))
    dyaw = bins[[r["action"][5] for r in fr]]
    cum = np.concatenate([[0.0], np.cumsum(dyaw)])  # cum[i] = yaw before tick i

    def cum_at(t_ms):
        return float(np.interp(t_ms, ep, cum[:-1]))

    def bearing_at(t_ms):
        j = float(np.interp(t_ms - lag_ms, row_t, np.arange(len(rows))))
        k = int(round(j))
        k = max(0, min(len(rows) - 1, k))
        r = rows[k]
        d0 = r["sp"][0] - r["cp"][0]
        d2 = r["sp"][2] - r["cp"][2]
        return math.degrees(math.atan2(-d0, d2))

    anchors = [(t, bearing_at(t)) for t in hurt_t if ep[0] <= t <= ep[-1]]
    if len(anchors) < 2:
        return lambda t: None

    ANCHOR_MS = 2000.0  # trust integration only this close to an anchor

    def yaw_at(t_ms):
        near = min(abs(t_ms - a[0]) for a in anchors)
        if near > ANCHOR_MS:
            return None                # drift risk: label noise > signal
        k = 0
        while k + 1 < len(anchors) and anchors[k + 1][0] < t_ms:
            k += 1
        t0, y0 = anchors[k]
        if k + 1 < len(anchors):
            t1, y1 = anchors[k + 1]
            raw = y0 + (cum_at(t_ms) - cum_at(t0))
            raw1 = y0 + (cum_at(t1) - cum_at(t0))
            resid = wrap180(y1 - raw1)
            fr_ = (t_ms - t0) / max(t1 - t0, 1e-9)
            return wrap180(raw + resid * min(max(fr_, 0.0), 1.0))
        return wrap180(y0 + (cum_at(t_ms) - cum_at(t0)))

    return yaw_at


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--bot-log", required=True)
    ap.add_argument("--flight-log", default=None,
                    help="deploy flight log for the same round; enables the "
                         "hit-anchored head-yaw solver (the bot log's cy is "
                         "BODY yaw -- wrong by up to ~90 deg while strafing)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stack", type=int, default=1,
                    help="emit S-frame temporal windows (S,H,W,C) per sample "
                         "for motion-aware perception (v10)")
    ap.add_argument("--fov", type=float, default=90.0,
                    help="client vertical FOV while collecting (no sprint!)")
    ap.add_argument("--aspect", type=float, default=854.0 / 480.0)
    ap.add_argument("--max-gap-ms", type=float, default=40.0)
    ap.add_argument("--target-lag-ms", type=float, default=0.0,
                    help="match the TARGET state this many ms before the "
                         "frame: the client renders remote entities ~3 "
                         "ticks (150 ms) behind server truth")
    args = ap.parse_args()

    frames = np.load(os.path.join(args.frames_dir, "frames.npy"))
    times = np.load(os.path.join(args.frames_dir, "times.npy"))

    rows, hurts = [], []
    for line in open(args.bot_log):
        r = json.loads(line)
        if r.get("ev") == "hurt":
            hurts.append(r)
        elif "sp" in r:  # tick row; skips other event rows (e.g. "swing")
            rows.append(r)
    if not rows:
        raise SystemExit("bot log has no tick rows")
    row_t = np.asarray([r["t"] for r in rows], dtype=np.float64)
    hurt_t = np.asarray([h["t"] for h in hurts], dtype=np.float64)

    vfov = args.fov
    hfov = 2.0 * math.degrees(math.atan(math.tan(math.radians(vfov / 2.0))
                                        * args.aspect))

    n = len(frames)
    labels = np.zeros((n, len(_L)), dtype=np.float32)
    mask = np.ones((n, len(_L)), dtype=np.float32)
    keep = np.zeros(n, dtype=bool)
    prev = {}  # index -> (aim_yaw_deg_mc, aim_pitch_deg_mc, t_ms) for vel

    head_yaw_at = None
    if args.flight_log:
        head_yaw_at = solve_head_yaw(args.flight_log, times, rows, row_t,
                                     hurt_t, args.target_lag_ms)

    def nearest(t):
        j = int(np.searchsorted(row_t, t))
        cand = [k for k in (j - 1, j) if 0 <= k < len(rows)]
        if not cand:
            return None
        k = min(cand, key=lambda k: abs(row_t[k] - t))
        return k if abs(row_t[k] - t) <= args.max_gap_ms else None

    for i in range(n):
        k = nearest(times[i])            # camera state: fresh (client-auth)
        kt = nearest(times[i] - args.target_lag_ms)  # target: render-lagged
        if k is None or kt is None:
            continue
        r = rows[k]
        rt = rows[kt]
        keep[i] = True

        cam_eye = np.asarray(r["cp"]) + np.asarray([0.0, EYE, 0.0])
        tgt_c = np.asarray(rt["sp"]) + np.asarray([0.0, CENTER, 0.0])
        d = tgt_c - cam_eye
        dist = float(np.linalg.norm(d))
        hdist = float(math.hypot(d[0], d[2]))

        if head_yaw_at is not None:
            hy = head_yaw_at(times[i])
            if hy is None:
                keep[i] = False     # no anchored yaw truth: drop the frame
                continue
            cam_yaw_mc = hy
        else:
            cam_yaw_mc = wrap180(180.0 - math.degrees(r["cy"]))
        cam_pitch_mc = -math.degrees(r["cpt"])          # down-positive
        yaw_dir_mc = math.degrees(math.atan2(-d[0], d[2]))
        pitch_dir_mc = math.degrees(math.atan2(-d[1], hdist))
        err_yaw_mc = wrap180(yaw_dir_mc - cam_yaw_mc)
        err_pitch_mc = pitch_dir_mc - cam_pitch_mc

        visible = (abs(err_yaw_mc) < hfov / 2.0
                   and abs(err_pitch_mc) < vfov / 2.0
                   and dist <= 16.0)

        lab = labels[i]
        lab[_L["self_pitch"]] = -cam_pitch_mc / 90.0    # sim: up-positive
        if head_yaw_at is not None:
            # combat data: the camera player takes damage constantly and we
            # have no ground truth for it -- labeling 1.0 teaches the model
            # to ALWAYS report full HP (v8 shipped with exactly that lobotomy:
            # live self_hp pinned at 0.997 across a six-death round)
            mask[i, _L["self_hp"]] = 0.0
        else:
            lab[_L["self_hp"]] = 1.0                    # idle camera: never damaged
        spd_dt = (row_t[k] - row_t[k - 1]) / 1000.0 if k > 0 else 0.05
        if k > 0 and 0.01 < spd_dt < 0.2:
            dp = np.asarray(rows[k]["cp"]) - np.asarray(rows[k - 1]["cp"])
            lab[_L["self_speed"]] = float(
                np.linalg.norm(dp[[0, 2]]) / spd_dt * 0.05)  # blocks/tick
        else:
            mask[i, _L["self_speed"]] = 0.0
        if visible:
            # synth conventions: yaw + = right of crosshair (mc diff
            # transfers directly), pitch + = BELOW crosshair (mc
            # down-positive, NO flip -- the adapter flips into sim frame)
            lab[_L["aim_err_yaw"]] = err_yaw_mc / 180.0
            lab[_L["aim_err_pitch"]] = err_pitch_mc / 90.0
            lab[_L["bbox_height"]] = bbox_height_frac(dist)
            lab[_L["visible"]] = 1.0
            lab[_L["enemy_on_ground"]] = float(rt["g"])
            recent = hurt_t[(hurt_t <= times[i])
                            & (hurt_t > times[i] - HURT_FLASH_MS)]
            lab[_L["hurt_flash"]] = 1.0 if len(recent) else 0.0
            if i - 1 in prev:
                py, pp, pt = prev[i - 1]
                dtf = (times[i] - pt) / 1000.0 / 0.05    # frames of 50 ms
                if 0.5 < dtf < 3.0:
                    lab[_L["rel_screen_vx"]] = (
                        wrap180(err_yaw_mc - py) * PX_PER_DEG / dtf / FRAME_W)
                    lab[_L["rel_screen_vy"]] = (
                        (err_pitch_mc - pp) * PX_PER_DEG / dtf / FRAME_H)
                else:
                    mask[i, _L["rel_screen_vx"]] = 0.0
                    mask[i, _L["rel_screen_vy"]] = 0.0
            else:
                mask[i, _L["rel_screen_vx"]] = 0.0
                mask[i, _L["rel_screen_vy"]] = 0.0
            prev[i] = (err_yaw_mc, err_pitch_mc, times[i])
        # invisible: enemy-derived labels stay 0 per synth convention

    if args.stack > 1:
        # temporal windows: sample i becomes frames [i-S+1 .. i] stacked on
        # a new leading axis. Only windows whose inter-frame gaps are all
        # <= 75 ms qualify (settled-frame dumps skip menu ticks, leaving
        # holes) -- a stack spanning a hole would teach false motion.
        S = args.stack
        ok = keep.copy()
        for i in np.nonzero(keep)[0]:
            if i < S - 1 or np.any(np.diff(times[i - S + 1: i + 1]) > 75.0):
                ok[i] = False
        idx = np.nonzero(ok)[0]
        stacked = np.stack([frames[idx - o] for o in range(S - 1, -1, -1)], 1)
        print("stack=%d: %d of %d labeled frames have full windows"
              % (S, len(idx), int(keep.sum())))
        frames, labels, mask = stacked, labels[idx], mask[idx]
    else:
        frames, labels, mask = frames[keep], labels[keep], mask[keep]
    np.savez_compressed(args.out, frames=frames, labels=labels, mask=mask)
    vis = labels[:, _L["visible"]] > 0.5
    print("dataset: %d frames (%d dropped by alignment), %.0f%% visible, "
          "%d hurt-flash positives -> %s"
          % (len(frames), int((~keep).sum()), 100.0 * vis.mean(),
             int(labels[:, _L["hurt_flash"]].sum()), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
