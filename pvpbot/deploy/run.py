"""CLI for the live deployment loop.

Dry run (no Quartz, no Minecraft -- mock capture + mock input)::

    python3 -m pvpbot.deploy.run --dry-run --ticks 100
    python3 -m pvpbot.deploy.run --dry-run --checkpoint runs/ppo/latest.pt

Live (requires SETUP.md completed: permissions, pyobjc, calibration)::

    python3 -m pvpbot.deploy.run --checkpoint runs/ppo/latest.pt \
        --px-per-degree 12.9 --ticks 2400

Utilities::

    python3 -m pvpbot.deploy.run --calibrate     # print calibration procedure
    python3 -m pvpbot.deploy.run --nudge 1000    # inject one relative mouse move
    python3 -m pvpbot.deploy.capture --check     # capture readiness probe

Stop a live run at any time with:  touch /tmp/pvpbot-stop
"""
import argparse
import logging
import os
import sys
import time

from pvpbot.deploy import input_inject
from pvpbot.deploy.capture import MockFrameSource
from pvpbot.deploy.input_inject import Calibration, MockInputSink
from pvpbot.deploy.loop import (
    DEATH_FLAG_PATH,
    KILL_SWITCH_PATH,
    TickLoop,
    TorchPolicy,
)


def _activate_game(pid: int) -> bool:
    """Bring the game app to the foreground (pyobjc AppKit)."""
    try:
        from AppKit import (  # noqa: PLC0415
            NSApplicationActivateIgnoringOtherApps,
            NSRunningApplication,
        )

        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        if app is not None:
            app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
            return True
    except Exception:
        pass
    return False


def _prepare_game(input_sink, pid: int) -> None:
    """Focus the game and un-pause it: activate, then Esc (pid-addressed)
    if the cursor is free (1.8 auto-pauses on focus loss). If the cursor
    is STILL free afterwards, the player is on the death screen -- raise
    the death flag so the loop's respawn logic takes over."""
    _activate_game(pid)
    time.sleep(0.9)
    if input_sink.cursor_visible():
        input_sink.key("esc", True)
        time.sleep(0.05)
        input_sink.key("esc", False)
        time.sleep(0.9)
    if input_sink.cursor_visible():
        try:
            open(DEATH_FLAG_PATH, "w").close()
        except OSError:
            pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m pvpbot.deploy.run",
        description="mc-pvp-bot live deployment loop",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="mock capture (noise) + mock input; no Quartz needed")
    p.add_argument("--checkpoint", default=None,
                   help="PolicyNet checkpoint (.pt); random-init policy if omitted")
    p.add_argument("--ticks", type=int, default=100,
                   help="number of ticks to run (default 100 = 5 s)")
    p.add_argument("--record", default=None,
                   help="flight-recorder JSONL path (default: pvpbot-flight.jsonl; "
                        "'none' disables)")
    p.add_argument("--window-title", default="Minecraft",
                   help="substring match for the game window (live mode)")
    p.add_argument("--px-per-degree", type=float, default=None,
                   help="calibration for both axes (see --calibrate)")
    p.add_argument("--px-per-degree-x", type=float, default=None)
    p.add_argument("--px-per-degree-y", type=float, default=None)
    p.add_argument("--deterministic", action="store_true",
                   help="argmax the policy instead of sampling")
    p.add_argument("--kill-switch", default=KILL_SWITCH_PATH,
                   help="sentinel file that stops the loop when it exists")
    p.add_argument("--calibrate", action="store_true",
                   help="print the manual degrees->pixels calibration procedure")
    p.add_argument("--nudge", type=int, default=None, metavar="PX",
                   help="inject a single relative horizontal mouse move of PX "
                        "pixels (calibration helper; needs Quartz)")
    p.add_argument("--seed", type=int, default=0, help="dry-run noise seed")
    p.add_argument("--dump-frames", default=None, metavar="DIR",
                   help="save per-tick downscaled frames + wall-clock times "
                        "(combat-domain data for perception fine-tuning)")
    p.add_argument("--crit-assist", type=int, default=0, choices=(0, 1),
                   help="1 (default): jump-crit before swinging in reach "
                        "(1.5x damage; matches P4-Hacker's 100%% crit rate)")
    p.add_argument("--click-discipline", type=int, default=0, choices=(0, 1),
                   help="1 (default): suppress swings unless crosshair is on "
                        "the hitbox and in reach (1.8 swings reset sprint)")
    p.add_argument("--aim-assist", type=int, default=1, choices=(0, 1),
                   help="1 (default): bounded-rate camera homing on the "
                        "perceived aim error while the enemy is visible "
                        "(wurst-style, same mechanism as the practice-bot "
                        "competition); 0: policy aims unassisted")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def make_calibration(args: argparse.Namespace) -> Calibration:
    x = args.px_per_degree_x or args.px_per_degree
    y = args.px_per_degree_y or args.px_per_degree
    if x is None and y is None:
        return Calibration()
    return Calibration(
        px_per_degree_x=x or Calibration.px_per_degree_x,
        px_per_degree_y=y or Calibration.px_per_degree_y,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.calibrate:
        print(input_inject.__doc__)
        return 0

    if args.nudge is not None:
        from pvpbot.deploy.input_inject import QuartzInputSink  # noqa: PLC0415

        try:
            sink = QuartzInputSink()
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print("Injecting relative mouse move of %d px in 3 s -- focus Minecraft."
              % args.nudge)
        import time as _time
        _time.sleep(3.0)
        sink.move_mouse(args.nudge, 0)
        print("Done. Read the yaw delta off the F3 overlay "
              "(px_per_degree = %d / |yaw1 - yaw0|)." % args.nudge)
        return 0

    record = args.record
    if record is None:
        record = "pvpbot-flight.jsonl"
    elif record.lower() == "none":
        record = None

    policy = TorchPolicy(checkpoint=args.checkpoint, deterministic=args.deterministic)
    calibration = make_calibration(args)

    if args.dry_run:
        frame_source = MockFrameSource.noise(seed=args.seed)
        input_sink = MockInputSink()
        print("dry-run: MockFrameSource(noise) + MockInputSink, %d ticks at 20 Hz"
              % args.ticks)
    else:
        from pvpbot.deploy.capture import QuartzFrameSource  # noqa: PLC0415
        from pvpbot.deploy.input_inject import QuartzInputSink  # noqa: PLC0415

        try:
            frame_source = QuartzFrameSource(window_title=args.window_title)
            game_pid = frame_source.window_owner_pid()
            if game_pid is None:
                print("refusing LIVE mode: cannot resolve the game window's "
                      "owner pid (keyboard events would need the global HID "
                      "tap, which can leak into other apps).", file=sys.stderr)
                return 1
            input_sink = QuartzInputSink(target_pid=game_pid)
        except RuntimeError as exc:
            print("live mode unavailable:\n%s" % exc, file=sys.stderr)
            print("hint: use --dry-run, and see pvpbot/deploy/SETUP.md",
                  file=sys.stderr)
            return 1
        print("LIVE mode against window %r (events addressed to pid %d). "
              "Kill switch: touch %s"
              % (args.window_title, game_pid, args.kill_switch))
        print("Reminder: consenting opponents on private servers only "
              "(pvpbot/deploy/SETUP.md).")

    # Supervised match (live): transient stops (focus blip, grab timeout)
    # re-focus the game and resume the remaining ticks. Two focus losses
    # within a minute mean the USER is taking control -- yield for real.
    # Kill-switch and a vanished window always end the match.
    remaining = args.ticks
    attempt = 0
    focus_stops = []
    result = None
    flight_paths = []
    while remaining > 0:
        rec_path = record
        if record and attempt > 0:
            rec_path = "%s.r%d" % (record, attempt)
        if rec_path:
            flight_paths.append(rec_path)
        loop = TickLoop(
            frame_source=frame_source,
            input_sink=input_sink,
            policy=policy,
            calibration=calibration,
            recorder_path=rec_path,
            kill_switch_path=args.kill_switch,
            clear_death_flag=(attempt == 0),
            aim_assist=bool(args.aim_assist),
            click_discipline=bool(args.click_discipline),
            crit_assist=bool(args.crit_assist),
            dump_frames_dir=(os.path.join(args.dump_frames, "a%d" % attempt)
                             if args.dump_frames else None),
        )
        # AFTER TickLoop construction: the constructor clears a stale
        # death flag, and _prepare_game may legitimately raise a fresh
        # one (death screen at match start)
        if not args.dry_run:
            _prepare_game(input_sink, game_pid)
        result = loop.run(remaining)
        remaining -= max(result.ticks_run, 1)
        reason = result.stop_reason
        if (args.dry_run or reason.startswith("completed")
                or "kill-switch" in reason or "exhausted" in reason):
            break
        if "frontmost" in reason:
            now = time.monotonic()
            focus_stops = [t for t in focus_stops if now - t < 60.0] + [now]
            if len(focus_stops) >= 2:
                print("two focus losses within 60 s -- user is taking "
                      "control, ending the match")
                break
        attempt += 1
        if attempt >= 10:
            print("too many restarts; ending the match")
            break
        print("loop stopped (%s) with %d ticks left -- refocusing and "
              "resuming" % (reason, remaining))

    print()
    print("ticks run   : %d" % (args.ticks - remaining))
    print("final state : %s" % result.state.value)
    print("stop reason : %s" % result.stop_reason)
    for p in flight_paths:
        print("flight log  : %s" % p)
    print()
    print(result.stats.format_table())

    if args.dry_run and isinstance(input_sink, MockInputSink):
        moves = input_sink.events_of("move")
        clicks = input_sink.events_of("click")
        keys = input_sink.events_of("key")
        print()
        print("mock input sink: %d mouse moves, %d clicks, %d key events"
              % (len(moves), len(clicks), len(keys)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
