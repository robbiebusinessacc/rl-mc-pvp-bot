# Competitor Research & Milestones

## How real rule-based PvP bots work (research summary)

The bots players actually practice against (pvp.land bot fights, practice
plugins) are rule-based state machines with humanized parameters:

- **Difficulty tables** (PracticeBotPvP): Easy/Medium/Hard/Hacker rows
  scaling crit rate (10/40/80/100%), w-tap/combo rate (10/50/90/100%),
  strafe rate (30/60/85/100%) and reaction speed (1.5x slower to 3x
  faster than "normal").
- **Aim model** (Wurst AimAssist): rotate toward the target at a capped
  rotation speed (default 600 deg/s; configurable 10-3600) inside a FOV
  gate — bounded-rate homing with smoothing, not teleport aim.
- **Techniques**, tick-accurate: crits = jump, ~7 ticks to falling state,
  swing while falling (1.5x); w-tap = hit -> release W 1 tick -> re-press
  (sprint-knockback reset); circle/zigzag strafe with random direction
  switches; managed spacing (back up when too close, chase when far);
  random hops every 20-60 ticks; 8-14 CPS is the competitive click band.

Implemented as `pvpbot/eval/practice.py` tiers P1-Easy .. P4-Hacker
(spacing adapted to our honest 3.0 client reach; the plugin defaults to a
generous 3.5). They feed both the eval ladder and the training league.

## Current standings (real-physics sim, 60 duels/pair, re-run 2026-07-28)

| Rank | Contestant | Elo | W-L-D |
|---|---|---|---|
| 1 | **ckpt 45.3B** | 1612 | **514-5-21** |
| 2 | P3-Hard | 1235 | 300-79-161 |
| 3 | P4-Hacker | 1210 | 303-75-162 |
| 4 | T2-Chaser | 1156 | 230-91-219 |
| 5 | T4-Pro | 1079 | 230-168-142 |
| 6 | T3-Strafer | 1043 | 203-167-170 |
| 7 | T1-Aimbot | 1037 | 161-156-223 |
| 8 | P1-Easy | 673 | 110-430-0 |
| 9 | P2-Medium | 541 | 70-470-0 |
| 10 | T0-Idle | 415 | 0-480-60 |

The learner now beats every contestant in the field: 60-0 against T0-Idle, T2-Chaser,
T3-Strafer, P1-Easy and P2-Medium, 59-1 against P3-Hard, 58-2 against P4-Hacker
(98.7% over a separate 1,000-duel head-to-head across five seeds), and 39-0-21
against T1-Aimbot, whose remaining draws are timeouts rather than losses.

Earlier checkpoints in this run could not finish a stationary target: they closed to
point blank, where the target sits below the crosshair, and their pitch ran away
upward — 0-0-60 against both immobile tiers. That is fixed. Mean aim error halved
(23.4° to 12.1°), hits taken per duel more than halved (7.99 to 3.16), average combo
nearly tripled (1.98 to 5.52), and duels now end in 274 ticks instead of 497.

(P1/P2 rank low in the full round-robin because the strong field punishes their
crit-fishing jumps; head to head P1-Easy beats P2-Medium 57-3-0, because Medium's
higher crit rate makes it jump more and jumping breaks sprint.)

## Milestones

- [x] **M1 (sim mechanics):** beat every practice tier including
  P4-Hacker at >90% winrate. — Cleared 2026-07-21, and it holds on the current
  engine for both checkpoints on record: `ckpt_32.8B_faithful78` is 128-0-0
  against each of the seven mobile contestants (896-0-0 overall, seed 5), and
  the 45.3B checkpoint takes 237 of 240 duels against the practice field (98.8%) — 60-0 vs P1-Easy and
  P2-Medium, 59-1 vs P3-Hard, 58-2 vs P4-Hacker — plus 98.7% against P4-Hacker
  over a separate 1,000-duel head-to-head across five seeds.
- [x] **M2 (live perception):** crosshair-on-hitbox >=25% of engaged
  ticks live. — CLEARED 2026-07-22: 54.1% / 48.5% in consecutive rounds
  after the pitch-axis inversion fix (0d8ed39) + wurst-style aim assist.
  (History: 10.6% -> 24.1% -> 54.1%.)
- [x] **M3 (live combat):** sustained kills (>=1/min) against an ARMORED,
  fighting-back target through the real pixel pipeline. — CLEARED
  2026-07-22 vs P2-Medium in full diamond: 4-1 and 3-1 in consecutive
  rounds (3.5 kills/min), fully autonomous respawn, zero leaks.
  vs P3-Hard: 2/min kill rate but 4-18 — trades lost to 150 ms sensor
  lag; obs-delay-adapted policy training overnight (18B -> 21B,
  --obs-delay 3).
- [ ] **M3.5 (the decisive bar):** WIN a live duel against the P4-Hacker
  port (`target_bot.js --fight hacker` mirrors practice.py tier-for-tier:
  720 deg/s aim, 0-tick reaction, 13 CPS, hurt-timed swings, 100%
  wtap/crit/strafe), both sides in full diamond, >=5 min, more kills than
  deaths. P4's spec is mechanically beyond most humans — beating it live
  through pixels is the proof the sim dominance transfers.
- [ ] **M4 (the video's bar):** duel a consenting human and land
  competitive exchanges.

Sources: PracticeBotPvP (Modrinth/Spigot), Wurst wiki AimAssist,
mc-servers.io PvP guide, pvp.land server descriptions.
