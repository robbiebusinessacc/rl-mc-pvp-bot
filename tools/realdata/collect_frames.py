"""Capture downscaled real Minecraft frames while driving the camera.

Runs alongside tools/realdata/target_bot.js: the target bot supplies moving
ground truth; this script captures FRAME_SHAPE grayscale frames at ~20 Hz
from the (focused) Minecraft window while injecting smooth random camera
motion and walking bursts into the client, and periodically toggles damage
on the target via the server-console FIFO so hurt_flash gets positive
examples.

Outputs into --out DIR: frames.npy (N,64,114) u8, times.npy (N,) float ms.

Abort at any time with: touch /tmp/pvpbot-stop
"""
import argparse
import os
import random
import subprocess
import sys
import time

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_THIS)))

from pvpbot.deploy.capture import QuartzFrameSource  # noqa: E402
from pvpbot.deploy.input_inject import QuartzInputSink  # noqa: E402

STOP = "/tmp/pvpbot-stop"


def focus_minecraft() -> None:
    subprocess.run(
        ["osascript", "-e",
         'tell application "System Events" to set frontmost of '
         'first process whose name contains "java" to true'],
        check=False, capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=600)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--px-per-degree", type=float, default=6.667)
    ap.add_argument("--console-fifo", default=None,
                    help="server console FIFO for periodic hurt-flash pulses")
    ap.add_argument("--target-name", default="PvPTarget")
    ap.add_argument("--pitch-min", type=float, default=-35.0)
    ap.add_argument("--pitch-max", type=float, default=25.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(0)
    src = QuartzFrameSource(window_title="Minecraft")
    sink = QuartzInputSink()
    focus_minecraft()
    time.sleep(1.0)

    try:
        from AppKit import NSWorkspace
        _ws = NSWorkspace.sharedWorkspace()

        def game_frontmost():
            try:
                name = (_ws.frontmostApplication().localizedName() or "").lower()
                return "java" in name or "minecraft" in name
            except Exception:
                return False
    except ImportError:
        def game_frontmost():
            return False  # cannot verify -> never inject

    frames, times = [], []
    dt = 1.0 / args.fps
    n_ticks = int(args.seconds * args.fps)
    yaw_rate = 0.0          # deg/tick, OU process
    est_pitch = 0.0         # integrated estimate, for clamping only
    w_ticks = 0             # walking burst countdown
    strafe_key = None
    strafe_ticks = 0
    next_pulse = time.time() + 8.0
    pulse_damage = True

    menu_streak = 0

    def press_esc():
        import Quartz.CoreGraphics as CG
        for down in (True, False):
            CG.CGEventPost(CG.kCGHIDEventTap,
                           CG.CGEventCreateKeyboardEvent(None, 53, down))
            time.sleep(0.05)

    try:
        for i in range(n_ticks):
            t_next = time.time() + dt
            if os.path.exists(STOP):
                print("stop sentinel found; ending collection")
                break

            # focus loss opens the 1.8 game menu and dims the world; detect
            # (daylight gameplay frames average >100; the dimmed menu sits in
            # a tight ~50-70 band, measured 57.7) and auto-recover so a stray
            # click elsewhere doesn't silently ruin the session
            if frames and 45.0 < float(frames[-1].mean()) < 72.0:
                menu_streak += 1
            else:
                menu_streak = 0
            if menu_streak >= 3:
                print("menu detected at tick %d -- refocusing" % i)
                focus_minecraft()
                time.sleep(0.4)
                press_esc()
                time.sleep(0.4)
                menu_streak = 0
            elif i % 300 == 299:      # periodic focus re-assert (~15 s)
                focus_minecraft()

            # never inject into anything but the game (a client crash
            # once leaked keystrokes into the user's other apps)
            if not game_frontmost():
                sink.release_all()
                time.sleep(0.5)
                focus_minecraft()
                continue
            # camera: OU yaw rate + bounded pitch random walk
            yaw_rate += -0.15 * yaw_rate + rng.gauss(0.0, 2.0)
            yaw_rate = max(-9.0, min(9.0, yaw_rate))
            dpitch = rng.gauss(0.0, 0.8) - 0.02 * est_pitch
            lo, hi = args.pitch_min - est_pitch, args.pitch_max - est_pitch
            dpitch = max(lo, min(hi, dpitch))
            est_pitch += dpitch
            sink.move_mouse(int(round(yaw_rate * args.px_per_degree)),
                            int(round(dpitch * args.px_per_degree)))

            # movement: W bursts (never sprint: sprint changes the FOV),
            # occasional A/D strafes
            if w_ticks > 0:
                w_ticks -= 1
                if w_ticks == 0:
                    sink.key("w", False)
            elif rng.random() < 0.02:
                w_ticks = rng.randint(10, 40)
                sink.key("w", True)
            if strafe_ticks > 0:
                strafe_ticks -= 1
                if strafe_ticks == 0 and strafe_key:
                    sink.key(strafe_key, False)
                    strafe_key = None
            elif rng.random() < 0.01:
                strafe_key = rng.choice(["a", "d"])
                strafe_ticks = rng.randint(5, 20)
                sink.key(strafe_key, True)

            # periodic hurt-flash pulses on the target
            if args.console_fifo and time.time() >= next_pulse:
                cmd = ("effect %s 7 1 0" if pulse_damage
                       else "effect %s 6 1 1") % args.target_name
                try:
                    with open(args.console_fifo, "w") as f:
                        f.write(cmd + "\n")
                except OSError:
                    pass
                pulse_damage = not pulse_damage
                next_pulse = time.time() + 8.0

            frames.append(src.get_frame())
            times.append(time.time() * 1000.0)

            lag = t_next - time.time()
            if lag > 0:
                time.sleep(lag)
    finally:
        sink.release_all()
        src.close()

    np.save(os.path.join(args.out, "frames.npy"),
            np.stack(frames).astype(np.uint8))
    np.save(os.path.join(args.out, "times.npy"), np.asarray(times))
    print("saved %d frames -> %s" % (len(frames), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
