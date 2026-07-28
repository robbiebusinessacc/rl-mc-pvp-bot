# Duel input schedules

A schedule is a deterministic, open-loop input script for a two-bot duel.
The **same** JSON file drives both sides of the validation harness:

* `recorder/record_duel.js` replays it against a real 1.8.9 server via
  mineflayer and logs ground-truth physics per tick, and
* `compare.py` / `make_fixture.py` replay it through the NumPy sim
  (`pvpbot.sim.env`, falling back to `pvpbot.sim.stub`) by mapping each
  tick's inputs onto `pvpbot.spec.ACTION_HEADS`.

Because both replays consume identical inputs, any difference between the
recorded trace and the sim trace is a *physics* difference, not an input
difference.

## Conventions

* One tick = 1/20 s (`pvpbot.spec.TICK_RATE`).
* All positions and yaws in a schedule are in the **Minecraft frame**:
  feet coordinates, notchian yaw degrees (0 = +Z/south, 90 = -X/west,
  increasing clockwise from above). The sim side converts with
  `sim_yaw = (mc_yaw + 90) % 360` and `sim_y = mc_y - arena.ground_y`
  (see the convention block at the top of `harness.py`).
* `yaw_delta_deg` / `pitch_delta_deg` are per-tick camera deltas. They are
  identical in both frames (constant-offset yaw mapping), and the sim bins
  them to the nearest `CAMERA_BINS` value (max 30 deg/tick — a 180 turn
  therefore takes 6 ticks at 30 deg/tick).
* A tick not covered by any segment for a bot is **idle** (no inputs,
  camera delta 0).
* Segments for the same bot must not overlap. Segments of different bots
  are independent tracks.

## File format

```jsonc
{
  "meta":  { "name": ..., "tick_rate": 20, "description": ... },
  "arena": {
    "ground_y": 4.0,              // MC feet-level of the flat floor (superflat surface)
    "spawn": { "<bot>": { "pos": [x, y, z], "yaw_mc_deg": deg } }
  },
  "bots": ["A", "B"],             // order defines sim side index 0/1
  "segments": [
    {
      "id": "walk_out",           // unique
      "bot": "A",
      "tick_start": 0,            // inclusive
      "tick_end": 40,             // exclusive
      "mechanic": "walk",         // which mechanic this segment validates
      "inputs": {                 // held constant for every tick in range
        "forward": true,          // booleans: forward, back, left, right,
        "sprint": false,          //           jump, sprint, attack
        "yaw_delta_deg": 0.0,     // numbers: yaw_delta_deg, pitch_delta_deg
        "pitch_delta_deg": 0.0
      }
    }
  ]
}
```

`attack: true` means "swing this tick if a target is in reach": the
recorder attacks the nearest other player within 3 blocks; the sim's
attack head likewise only lands hits in reach. This keeps the script
open-loop while still producing hits without exact range prediction.

Every schedule must cover at least the mechanics in
`harness.REQUIRED_MECHANICS`: `walk`, `sprint`, `sprint_jump`, `stop`,
`turn_180`, `jump_arc`, `attack`, `knockback_observe`. Extra labels
(`coast`, `approach`, ...) are fine.

## `basic.json` walkthrough

Arena: flat superflat floor (surface at y=4). A spawns at z=0.5 facing
south (+Z, toward B); B spawns 16 blocks away at z=16.5 facing north.

| ticks   | bot | segment       | validates                                        |
|---------|-----|---------------|--------------------------------------------------|
| 0–40    | A   | walk_out      | pure walk acceleration + terminal walk speed     |
| 40–70   | A   | stop_decay    | friction decay to standstill (stop)              |
| 70–110  | A   | sprint_out    | sprint acceleration + terminal sprint speed      |
| 110–140 | A   | coast_1       | friction again (buffer between mechanics)        |
| 140–200 | A   | sprint_jump   | repeated sprint-jumps (jump held while sprinting)|
| 200–230 | A   | coast_2       | settle before the turn                           |
| 230–236 | A   | turn_180      | 180° turn while moving (30°/tick × 6 ticks)      |
| 236–300 | A   | walk_back     | walk on the reversed heading (back toward B)     |
| 300–303 | A   | jump_launch   | standing jump impulse                            |
| 303–330 | A   | jump_air      | full jump arc + landing, no horizontal input     |
| 330–380 | A   | approach      | walk into B's reach envelope                     |
| 380–410 | A   | attack_window | attack input held; swings land when in reach     |
| 410–420 | A   | disengage     | tail buffer                                      |
| 0–20    | B   | b_walk_in     | second walk sample (independent bot)             |
| 20–420  | B   | b_hold_victim | stationary target; knockback observation via `hurt` events |

Timing was tuned against the stub's kinematics so that A ends roughly
1 block from B when the attack window opens; hits land at ticks ~380/391/402
(1.8 hurt-time gating: one hit per 11 ticks in the stub).

Knockback itself is **recorded** (the `hurt` events carry pre/post victim
velocity) but the stub applies none, so the `b_hold_victim` /
`attack_window` segments are the ones expected to diverge against real
ground truth until the real sim implements knockback.
