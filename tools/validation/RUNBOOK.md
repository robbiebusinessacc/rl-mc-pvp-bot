# Physics validation runbook — real ground truth

How to produce a real 1.8 ground-truth recording and compare it against the
sim. Everything here is manual/one-time; the automated tests never touch a
server or the network (they use the stub-generated fixtures from
`make_fixture.py` instead).

## 1. Prerequisites

* Java 8 (1.8.8 servers do not run on modern JVMs).
* Node.js >= 18 and npm.
* This repo, with `python3` (3.9) + numpy for the compare step.

## 2. Server setup (PaperSpigot 1.8.8, offline mode, flat world)

1. Download the PaperSpigot 1.8.8 jar — file name is typically
   `paper-1.8.8-445.jar` (final 1.8.8 build). Sources: the PaperMC legacy
   archive (https://papermc.io/legacy) or a mirror such as
   https://getbukkit.org/download/spigot for plain Spigot 1.8.8.
   **Link/name only here — nothing in this repo downloads it.**
   Protocol note: 1.8.0–1.8.9 all speak protocol 47, so a 1.8.8 server
   works with the recorder's `--version 1.8.9` client.
2. Put the jar in an empty directory, run it once to generate files, accept
   `eula.txt` (`eula=true`).
3. `server.properties` — the lines that matter:

   ```properties
   online-mode=false
   level-type=FLAT
   generate-structures=false
   spawn-monsters=false
   spawn-animals=false
   spawn-npcs=false
   difficulty=1
   gamemode=0
   force-gamemode=true
   pvp=true
   spawn-protection=0
   view-distance=6
   max-players=10
   white-list=false
   motd=mc-pvp-bot validation arena
   ```

   `difficulty=1` (not 0): peaceful blocks all player-vs-player damage
   in some 1.8 builds and disables the hunger/regen behavior we want to
   observe as-is. `spawn-protection=0` is required or the bots cannot be
   knocked around near world spawn.

   The default FLAT preset is bedrock + 2 dirt + grass: surface (feet
   level) at **y = 4**, which is what `arena.ground_y` in
   `schedules/basic.json` assumes. If you use a custom preset, update
   `ground_y` and the spawn `pos` y values to the new surface height.
4. Start: `java -jar paper-1.8.8-445.jar nogui`
5. From the server console (these affect physics/health traces):

   ```
   gamerule doDaylightCycle false
   gamerule doMobSpawning false
   gamerule naturalRegeneration false
   op ValBotA
   op ValBotB
   ```

   `naturalRegeneration false` keeps recorded health monotonic so damage
   values can be read directly from the trace. OP is needed for the bots'
   self-teleport (`/tp`) into the arena.

## 3. Recorder setup and run

```sh
cd tools/validation/recorder
npm install                       # installs mineflayer (network needed once)
node record_duel.js \
    --host 127.0.0.1 --port 25565 --version 1.8.9 \
    --schedule ../schedules/basic.json \
    --out /tmp/duel_rec.jsonl
```

The two bots join as `ValBotA` / `ValBotB` (`--name-prefix` changes this —
re-op accordingly), teleport to the schedule's arena spawns, wait
`--settle-ms` (default 3000 ms) for chunks/teleports to settle, then run
the 420-tick (21 s) schedule and exit. Progress goes to stderr, data to
the `--out` JSONL.

Sanity-check the recording: ~2×420 tick rows plus a handful of
`{"event":"hurt",...}` rows, and `ValBotB`'s health should drop during
ticks ~380–410.

## 4. Compare against the sim

```sh
python3 tools/validation/compare.py \
    --recording /tmp/duel_rec.jsonl \
    --schedule tools/validation/schedules/basic.json \
    --env stub \
    --json-out /tmp/duel_report.json
```

Use `--env real` once `pvpbot/sim/env.py` has merged (or leave the default
`auto`, which prefers the real sim). If the whole trace looks rigidly
shifted by one tick, re-run with `--tick-offset 1` (or `-1`) — the two bot
connections start counting on a shared signal but a one-tick start skew is
possible.

## 5. What this schedule does and does not validate

Validated by `schedules/basic.json`:

* walk acceleration + terminal walk speed (`walk_out`, `b_walk_in`)
* ground friction / stop decay (`stop_decay`, `coast_*`)
* sprint acceleration + terminal sprint speed (`sprint_out`)
* repeated sprint-jumps (`sprint_jump`)
* turning while moving, 30°/tick camera-bin turn (`turn_180`)
* standing jump impulse + full arc + landing (`jump_launch`/`jump_air`)
* attack timing under 1.8 hurt-time gating (`attack_window`; the stub
  lands one hit per 11 ticks)
* knockback impulse observation (`hurt` events carry the victim's pre/post
  velocity). NOTE: the stub applies **no** knockback, so real-vs-stub will
  legitimately diverge in `b_hold_victim` after the first hit — that gap
  is a to-do for the real sim, not a harness bug.

Known stub artifacts to expect in real comparisons (documented in
`compare.py`): grounded vertical velocity is meaningless in the stub
(masked by default), sprint applies even without forward input, and there
is no player-player collision.

NOT yet validated (needs new schedules and/or equipment setup):

* armor damage reduction values (bots join with empty inventories; a
  schedule + console `/give`/`/replaceitem` setup for armor tiers is
  needed)
* critical hits (attacking while falling) and sword damage tiers
* sword blocking, fishing-rod/projectile knockback
* hunger/sprint-food interaction, potion effects, water/ladder movement

## 6. Offline note

`npm install` and the server jar download are the only network steps and
happen outside the repo's test path. `python3 -m pytest tests/ -q` uses
only the stub-generated fixtures and `node --check` (no packages, no
server, no network).

## 7. Real-run results (2026-07-20, Spigot 1.8.8, this repo's schedules)

Ground-truth recordings live in `tools/validation/recordings/*.jsonl` (basic,
combat, sprintkb) — compare.py can be rerun against them offline. Findings
that were fed back into `pvpbot/sim/env.py`:

* **Friction timing**: vanilla picks a tick's horizontal friction from the
  START-of-tick on-ground state (jump tick = ground friction, landing tick =
  air drag). Fixing this took the sprint-jump segment from 2.74 to 0.009
  pos RMSE; the whole 420-tick movement course now replays at ~0.01 RMSE.
* **1.8 diamond sword deals 8 damage** (1.6 hp through 80% armor), not the
  1.9-era 7.
* **Player collision**: server shoves overlapping players ~0.10*sqrt(d) per
  tick (vanilla 0.05 applied from both entities' updates); B's measured
  pre-hit drift matched 0.10. `SimConfig.collision_push`, disable per
  schedule with meta `"sim_collision": false`.
* **Sprint knockback**: additive (no second halving on horizontal): launch
  0.9 h with one ground-friction bite then 0.91 air decay; vertical 0.46.
  Total slide 4.94 blocks, pop 1.49.

Equip for combat schedules (console, while bots are settling):
`give ValBotA minecraft:diamond_sword 1` and `replaceitem entity <bot>
slot.armor.{head,chest,legs,feet} minecraft:diamond_{helmet,chestplate,
leggings,boots} 1`.

Known ground-truth caveats:
* mineflayer does NOT model client-side player pushing — its bots phase
  through each other where real clients cannot (combat recording diverges
  from the sim after the second hit for exactly this reason). Basic
  (movement) comparisons therefore run with `"sim_collision": false`.
* mineflayer attacks by server rule (<= 6 blocks), not the client's 3-block
  crosshair ray, so reach/aim-cone limits cannot be validated this way.
  The sim's padded cone lands a borderline hit at ~3.3 blocks that a real
  client would not throw — revisit `aim_pad`/`aim_slack_deg` with a
  client-side recording if max-range behavior ever matters.
* Recorder `hurt` events carry zero pre/post velocities (packet timing);
  read knockback from the per-tick trace instead.
