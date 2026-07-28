"""Shared library for the physics-validation tooling (tools/validation).

Used by compare.py, make_fixture.py and the tests. Owns:

  * schedule loading + schema validation (schedules/*.json)
  * schedule-input -> ACTION_HEADS mapping (incl. camera-delta binning)
  * Minecraft <-> sim coordinate/yaw conversions (see the big comment below)
  * rolling a schedule through a DuelVecEnv and capturing a per-tick trace
  * recording JSONL read/write (the format record_duel.js emits)

Python 3.9 / numpy only. No network, no Minecraft.

---------------------------------------------------------------------------
COORDINATE CONVENTIONS -- READ BEFORE TOUCHING ANY MATH IN THIS PACKAGE.
This is the classic source of silent validation bugs: both conventions
"look plausible" and a wrong mapping still produces smooth trajectories,
just rotated -- RMSE goes quietly wrong instead of loudly wrong.

Minecraft / notchian frame (what recorder/record_duel.js logs; mineflayer's
internal yaw is converted back to notchian before logging):
    +X = east, +Y = up, +Z = south.  Positions are feet coordinates.
    yaw   0 deg -> facing +Z (south)
    yaw  90 deg -> facing -X (west)
    yaw 180 deg -> facing -Z (north)
    yaw 270 deg -> facing +X (east)
    forward unit vector = (-sin(yaw), 0, +cos(yaw))
    pitch: positive = looking DOWN.
    velocity: blocks per tick (prismarine-physics native unit).

Sim / stub frame (pvpbot.sim.stub.DuelVecEnv, and the contract for the
real pvpbot.sim.env.DuelVecEnv):
    abstract (x, y, z), ground plane at y = 0.
    forward unit vector = (+cos(yaw), 0, +sin(yaw))
    velocity: blocks per tick (same unit -- no scaling needed).

Both conventions rotate the same way in the XZ plane (X -> Z -> -X -> -Z as
yaw grows), so a CONSTANT OFFSET maps one to the other and per-tick yaw
*deltas* are identical in both frames:

    sim_yaw = (mc_yaw + 90) % 360
    check: mc 0   (+Z) -> sim 90  -> (cos90,  sin90)  = (0, 1)  = +Z  OK
           mc 90  (-X) -> sim 180 -> (cos180, sin180) = (-1, 0) = -X  OK
    strafe check (left must stay left): mc yaw 0 faces +Z, MC-left is +X
    (east); sim yaw 90 with strafe=-1 gives ax = -(-1)*sin90 = +1 = +X. OK

Positions: X and Z transfer unchanged; only Y is shifted so the arena
surface (arena.ground_y in the schedule, MC feet-level) maps to sim y = 0:

    sim_pos = (mc_x, mc_y - ground_y, mc_z)

Velocities transfer unchanged (same axes, same units).
---------------------------------------------------------------------------
"""
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pvpbot.spec import (  # noqa: E402
    ACTION_HEADS,
    CAMERA_BINS,
    NUM_ACTION_HEADS,
)

HEAD_INDEX = {name: i for i, (name, _) in enumerate(ACTION_HEADS)}
CAMERA_ZERO_BIN = CAMERA_BINS.index(0.0)

MC_TO_SIM_YAW_OFFSET_DEG = 90.0

BOOL_INPUT_KEYS = ("forward", "back", "left", "right", "jump", "sprint", "attack")
DELTA_INPUT_KEYS = ("yaw_delta_deg", "pitch_delta_deg")
ALLOWED_INPUT_KEYS = set(BOOL_INPUT_KEYS) | set(DELTA_INPUT_KEYS)

# Every schedule must exercise at least these mechanics (segment "mechanic"
# field). Extra mechanics ("coast", "approach", ...) are allowed.
REQUIRED_MECHANICS = frozenset(
    {
        "walk",
        "sprint",
        "sprint_jump",
        "stop",
        "turn_180",
        "jump_arc",
        "attack",
        "knockback_observe",
    }
)

DEFAULT_SCHEDULE = os.path.join(_THIS_DIR, "schedules", "basic.json")


# ---------------------------------------------------------------------------
# Yaw / frame conversions
# ---------------------------------------------------------------------------

def mc_yaw_to_sim(mc_yaw_deg: float) -> float:
    """Notchian yaw (0 = +Z, clockwise from above) -> stub yaw (0 = +X)."""
    return (float(mc_yaw_deg) + MC_TO_SIM_YAW_OFFSET_DEG) % 360.0


def sim_yaw_to_mc(sim_yaw_deg: float) -> float:
    """Inverse of mc_yaw_to_sim."""
    return (float(sim_yaw_deg) - MC_TO_SIM_YAW_OFFSET_DEG) % 360.0


def wrap_angle_deg(a: np.ndarray) -> np.ndarray:
    """Wrap angles to [-180, 180)."""
    return (np.asarray(a, dtype=np.float64) + 180.0) % 360.0 - 180.0


def mc_pos_to_sim(pos_xyz, ground_y: float) -> np.ndarray:
    p = np.asarray(pos_xyz, dtype=np.float64).copy()
    p[..., 1] -= float(ground_y)
    return p


def sim_pos_to_mc(pos_xyz, ground_y: float) -> np.ndarray:
    p = np.asarray(pos_xyz, dtype=np.float64).copy()
    p[..., 1] += float(ground_y)
    return p


# ---------------------------------------------------------------------------
# Input -> action mapping
# ---------------------------------------------------------------------------

def nearest_camera_bin(delta_deg: float) -> int:
    """Nearest CAMERA_BINS index for a per-tick camera delta (degrees).

    Out-of-range deltas clip to the extreme bins. Exact ties resolve to the
    lowest index (np.argmin behavior) -- e.g. 2.0 is equidistant from bins
    1.0 and 3.0 and maps to index 6 (bin 1.0).
    """
    bins = np.asarray(CAMERA_BINS, dtype=np.float64)
    return int(np.argmin(np.abs(bins - float(delta_deg))))


def inputs_to_action(inputs: Dict[str, Any]) -> np.ndarray:
    """Map one schedule input dict to a (NUM_ACTION_HEADS,) int64 action.

    Head order/semantics from pvpbot.spec.ACTION_HEADS:
      forward: 0=back 1=none 2=forward   (forward+back cancel to 1)
      strafe:  0=left 1=none 2=right     (left+right cancel to 1)
      jump/sprint/attack: 0/1
      yaw/pitch: nearest CAMERA_BINS index for *_delta_deg (default 0 -> bin
      index CAMERA_ZERO_BIN). Yaw deltas are identical in MC and sim frames
      (constant-offset mapping), so no sign flip happens here.
    """
    unknown = set(inputs) - ALLOWED_INPUT_KEYS
    if unknown:
        raise ValueError("unknown input keys: %s" % sorted(unknown))
    a = np.zeros(NUM_ACTION_HEADS, dtype=np.int64)
    a[HEAD_INDEX["forward"]] = 1 + int(bool(inputs.get("forward"))) - int(
        bool(inputs.get("back"))
    )
    a[HEAD_INDEX["strafe"]] = 1 + int(bool(inputs.get("right"))) - int(
        bool(inputs.get("left"))
    )
    a[HEAD_INDEX["jump"]] = int(bool(inputs.get("jump")))
    a[HEAD_INDEX["sprint"]] = int(bool(inputs.get("sprint")))
    a[HEAD_INDEX["attack"]] = int(bool(inputs.get("attack")))
    a[HEAD_INDEX["yaw"]] = nearest_camera_bin(inputs.get("yaw_delta_deg", 0.0))
    a[HEAD_INDEX["pitch"]] = nearest_camera_bin(inputs.get("pitch_delta_deg", 0.0))
    return a


IDLE_ACTION = inputs_to_action({})


# ---------------------------------------------------------------------------
# Schedule loading / validation
# ---------------------------------------------------------------------------

def load_schedule(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        sched = json.load(f)
    errors = validate_schedule(sched)
    if errors:
        raise ValueError(
            "invalid schedule %s:\n  %s" % (path, "\n  ".join(errors))
        )
    return sched


def validate_schedule(sched: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable schema errors (empty list == valid)."""
    errs: List[str] = []
    if not isinstance(sched, dict):
        return ["schedule root must be a JSON object"]
    for key in ("meta", "arena", "bots", "segments"):
        if key not in sched:
            errs.append("missing top-level key %r" % key)
    if errs:
        return errs

    bots = sched["bots"]
    if (
        not isinstance(bots, list)
        or len(bots) != 2
        or len(set(bots)) != 2
        or not all(isinstance(b, str) for b in bots)
    ):
        errs.append("'bots' must be a list of two distinct string ids")
        return errs

    arena = sched["arena"]
    if not isinstance(arena.get("ground_y"), (int, float)):
        errs.append("arena.ground_y must be a number (MC feet-level of the floor)")
    spawn = arena.get("spawn", {})
    for b in bots:
        sp = spawn.get(b)
        if not isinstance(sp, dict):
            errs.append("arena.spawn missing entry for bot %r" % b)
            continue
        pos = sp.get("pos")
        if (
            not isinstance(pos, list)
            or len(pos) != 3
            or not all(isinstance(v, (int, float)) for v in pos)
        ):
            errs.append("arena.spawn[%r].pos must be [x, y, z] numbers" % b)
        if not isinstance(sp.get("yaw_mc_deg"), (int, float)):
            errs.append("arena.spawn[%r].yaw_mc_deg must be a number" % b)

    segments = sched["segments"]
    if not isinstance(segments, list) or not segments:
        errs.append("'segments' must be a non-empty list")
        return errs

    seen_ids = set()
    per_bot: Dict[str, List[Tuple[int, int, str]]] = {b: [] for b in bots}
    for i, seg in enumerate(segments):
        where = "segments[%d]" % i
        if not isinstance(seg, dict):
            errs.append("%s must be an object" % where)
            continue
        sid = seg.get("id")
        if not isinstance(sid, str) or not sid:
            errs.append("%s.id must be a non-empty string" % where)
            sid = where
        if sid in seen_ids:
            errs.append("duplicate segment id %r" % sid)
        seen_ids.add(sid)
        bot = seg.get("bot")
        if bot not in bots:
            errs.append("%s(%s).bot %r not in bots %s" % (where, sid, bot, bots))
        ts, te = seg.get("tick_start"), seg.get("tick_end")
        if not (isinstance(ts, int) and isinstance(te, int) and 0 <= ts < te):
            errs.append(
                "%s(%s) needs int ticks with 0 <= tick_start < tick_end" % (where, sid)
            )
        elif bot in per_bot:
            per_bot[bot].append((ts, te, sid))
        if not isinstance(seg.get("mechanic"), str):
            errs.append("%s(%s).mechanic must be a string" % (where, sid))
        inputs = seg.get("inputs")
        if not isinstance(inputs, dict):
            errs.append("%s(%s).inputs must be an object" % (where, sid))
            continue
        unknown = set(inputs) - ALLOWED_INPUT_KEYS
        if unknown:
            errs.append("%s(%s) unknown input keys %s" % (where, sid, sorted(unknown)))
        for k in BOOL_INPUT_KEYS:
            if k in inputs and not isinstance(inputs[k], bool):
                errs.append("%s(%s).inputs.%s must be a boolean" % (where, sid, k))
        for k in DELTA_INPUT_KEYS:
            if k in inputs and not isinstance(inputs[k], (int, float)):
                errs.append("%s(%s).inputs.%s must be a number" % (where, sid, k))

    # no per-bot overlaps: same bot cannot have two active segments on a tick
    for b, spans in per_bot.items():
        spans.sort()
        for (s0, e0, id0), (s1, e1, id1) in zip(spans, spans[1:]):
            if s1 < e0:
                errs.append(
                    "bot %r segments %r and %r overlap (ticks %d..%d)"
                    % (b, id0, id1, s1, min(e0, e1))
                )

    # focused schedules (e.g. combat-only) may opt out of full coverage
    if sched.get("meta", {}).get("coverage") != "partial":
        mechanics = {
            seg.get("mechanic")
            for seg in segments
            if isinstance(seg, dict) and isinstance(seg.get("mechanic"), str)
        }
        missing = REQUIRED_MECHANICS - mechanics
        if missing:
            errs.append("schedule does not cover mechanics: %s" % sorted(missing))
    return errs


def schedule_num_ticks(sched: Dict[str, Any]) -> int:
    return max(seg["tick_end"] for seg in sched["segments"])


def inputs_track(sched: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    """Per-tick input dict for each bot side: track[tick][side] -> inputs.

    Ticks not covered by any segment for a bot are idle ({}).
    """
    T = schedule_num_ticks(sched)
    bots = sched["bots"]
    track: List[List[Dict[str, Any]]] = [[{} for _ in bots] for _ in range(T)]
    for seg in sched["segments"]:
        side = bots.index(seg["bot"])
        for t in range(seg["tick_start"], seg["tick_end"]):
            track[t][side] = seg["inputs"]
    return track


def build_action_track(sched: Dict[str, Any]) -> np.ndarray:
    """(num_ticks, 2, NUM_ACTION_HEADS) int64 actions for the whole schedule."""
    track = inputs_track(sched)
    T = len(track)
    actions = np.tile(IDLE_ACTION, (T, 2, 1))
    for t in range(T):
        for side in (0, 1):
            if track[t][side]:
                actions[t, side] = inputs_to_action(track[t][side])
    return actions


def segment_for(sched: Dict[str, Any], bot: str, tick: int) -> Optional[str]:
    for seg in sched["segments"]:
        if seg["bot"] == bot and seg["tick_start"] <= tick < seg["tick_end"]:
            return seg["id"]
    return None


# ---------------------------------------------------------------------------
# Env construction / schedule replay
# ---------------------------------------------------------------------------

def make_env(kind: str = "auto", seed: int = 0, collision: bool = True):
    """Return (env, used_kind). kind: 'stub' | 'real' | 'auto'.

    'real' imports pvpbot.sim.env (the drop-in physics sim); 'stub' imports
    pvpbot.sim.stub; 'auto' prefers real and falls back to the stub.
    """
    if kind not in ("stub", "real", "auto"):
        raise ValueError("env kind must be stub|real|auto, got %r" % kind)
    if kind in ("real", "auto"):
        try:
            from pvpbot.sim.env import DuelVecEnv, SimConfig  # type: ignore

            # validation courses run tens of blocks in the open; the training
            # arena wall (position clamp at arena_radius) must not intrude
            cfg = SimConfig(
                arena_radius=1e6,
                spawn_radius=6.0,
                collision_push=0.10 if collision else 0.0,
            )
            return DuelVecEnv(1, seed=seed, config=cfg), "real"
        except ImportError:
            if kind == "real":
                raise RuntimeError(
                    "--env real requested but pvpbot.sim.env is not importable "
                    "(the real sim has not merged yet); use --env stub or auto"
                )
    from pvpbot.sim.stub import DuelVecEnv

    return DuelVecEnv(1, seed=seed), "stub"


def set_initial_state(env, sched: Dict[str, Any]) -> None:
    """Force the env's single duel into the schedule's arena spawn state.

    Relies on the DuelVecEnv state arrays (pos/vel/yaw/pitch/hp/hurt/ticks/
    prev_actions) that the stub exposes; the real sim is contractually a
    drop-in replacement. Spawn positions/yaws in the schedule are in the
    MINECRAFT frame and are converted here (see module docstring).
    """
    bots = sched["bots"]
    ground_y = sched["arena"]["ground_y"]
    try:
        env.pos[:] = 0.0
        env.vel[:] = 0.0
        env.pitch[:] = 0.0
        env.hp[:] = 20.0
        env.hurt[:] = 0
        env.ticks[:] = 0
        env.prev_actions[:] = 0
        for side, b in enumerate(bots):
            sp = sched["arena"]["spawn"][b]
            env.pos[0, side] = mc_pos_to_sim(sp["pos"], ground_y)
            env.yaw[0, side] = mc_yaw_to_sim(sp["yaw_mc_deg"])
    except AttributeError as e:
        raise RuntimeError(
            "env does not expose the stub state arrays needed for replay "
            "(pos/vel/yaw/pitch/hp/hurt/ticks/prev_actions): %s" % e
        )


def roll_schedule(env, sched: Dict[str, Any]) -> Dict[str, Any]:
    """Replay the schedule open-loop through env; return a per-tick trace.

    Trace arrays are in the SIM frame. Row t is the state at the END of game
    tick t, produced by the inputs the schedule holds during tick t (same
    semantics as record_duel.js rows).
    """
    if env.num_envs != 1:
        raise ValueError("replay requires a single-env DuelVecEnv (num_envs=1)")
    env.reset()
    set_initial_state(env, sched)
    actions = build_action_track(sched)
    T = actions.shape[0]
    bots = sched["bots"]

    pos = np.zeros((T, 2, 3), dtype=np.float64)
    vel = np.zeros((T, 2, 3), dtype=np.float64)
    yaw = np.zeros((T, 2), dtype=np.float64)
    pitch = np.zeros((T, 2), dtype=np.float64)
    hp = np.zeros((T, 2), dtype=np.float64)
    on_ground = np.zeros((T, 2), dtype=bool)
    hurt_events: List[Dict[str, Any]] = []

    for t in range(T):
        hp_before = env.hp[0].copy()
        _, _, done, _ = env.step(actions[t : t + 1])
        if bool(done[0]):
            raise RuntimeError(
                "episode terminated at tick %d during replay (death or "
                "timeout); schedule and env state are out of sync" % t
            )
        pos[t] = env.pos[0]
        vel[t] = env.vel[0]
        yaw[t] = env.yaw[0]
        pitch[t] = env.pitch[0]
        hp[t] = env.hp[0]
        on_ground[t] = env.pos[0, :, 1] <= 0.0
        for side in (0, 1):
            if env.hp[0, side] < hp_before[side]:
                hurt_events.append(
                    {
                        "tick": t,
                        "victim": bots[side],
                        "attacker": bots[1 - side],
                        "pre_vel": (vel[t - 1, side] if t > 0 else np.zeros(3))
                        .tolist(),
                        "post_vel": vel[t, side].tolist(),
                    }
                )

    return {
        "bots": bots,
        "num_ticks": T,
        "pos": pos,
        "vel": vel,
        "yaw": yaw,
        "pitch": pitch,
        "hp": hp,
        "on_ground": on_ground,
        "hurt_events": hurt_events,
    }


# ---------------------------------------------------------------------------
# Recording JSONL (format shared with recorder/record_duel.js)
# ---------------------------------------------------------------------------

def normalized_inputs(inputs: Dict[str, Any]) -> Dict[str, bool]:
    return {k: bool(inputs.get(k)) for k in BOOL_INPUT_KEYS}


def trace_to_recording_rows(
    trace: Dict[str, Any], sched: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Convert a sim-frame trace into recording rows in the MINECRAFT frame,
    exactly the shape record_duel.js writes -- so compare.py's conversion
    path is exercised even by synthetic fixtures.
    """
    ground_y = sched["arena"]["ground_y"]
    track = inputs_track(sched)
    bots = trace["bots"]
    rows: List[Dict[str, Any]] = []
    hurt_by_tick: Dict[int, List[Dict[str, Any]]] = {}
    for ev in trace["hurt_events"]:
        hurt_by_tick.setdefault(ev["tick"], []).append(ev)
    for t in range(trace["num_ticks"]):
        for side, bot in enumerate(bots):
            rows.append(
                {
                    "tick": t,
                    "bot": bot,
                    "pos": sim_pos_to_mc(trace["pos"][t, side], ground_y).tolist(),
                    "vel": trace["vel"][t, side].tolist(),
                    "yaw": sim_yaw_to_mc(trace["yaw"][t, side]),
                    "pitch": float(trace["pitch"][t, side]),
                    "onGround": bool(trace["on_ground"][t, side]),
                    "health": float(trace["hp"][t, side]),
                    "inputs": normalized_inputs(track[t][side]),
                }
            )
        for ev in hurt_by_tick.get(t, []):
            rows.append(
                {
                    "event": "hurt",
                    "tick": ev["tick"],
                    "victim": ev["victim"],
                    "attacker": ev["attacker"],
                    "pre_vel": ev["pre_vel"],
                    "post_vel": ev["post_vel"],
                }
            )
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_recording(
    path: str,
) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], List[Dict[str, Any]]]:
    """Read a recording JSONL.

    Returns (tick_rows, hurt_events) where tick_rows maps (bot, tick) -> row.
    Rows with an "event" field other than per-tick state are collected as
    events ("hurt") or skipped ("meta").
    """
    tick_rows: Dict[Tuple[str, int], Dict[str, Any]] = {}
    hurt_events: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                raise ValueError("%s:%d is not valid JSON" % (path, lineno))
            ev = row.get("event")
            if ev == "hurt":
                hurt_events.append(row)
            elif ev is None:
                tick_rows[(row["bot"], int(row["tick"]))] = row
            # other events ("meta", ...) are informational; ignore
    return tick_rows, hurt_events
