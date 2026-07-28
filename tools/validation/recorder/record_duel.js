#!/usr/bin/env node
/*
 * record_duel.js -- two-bot ground-truth physics recorder for mc-pvp-bot.
 *
 * Connects TWO mineflayer bots to a local offline-mode 1.8.x server,
 * teleports them to the flat arena defined in the schedule JSON, then
 * drives them through the schedule's deterministic per-tick input segments
 * (walk / sprint / sprint-jump / turns / jump / attack) while logging every
 * physics tick to JSONL. The exact same schedule file is replayed through
 * the NumPy sim by tools/validation/compare.py, so recorded-vs-simulated
 * divergence is purely a physics difference.
 *
 * Usage (after `npm install` in this directory -- see ../RUNBOOK.md):
 *   node record_duel.js --host 127.0.0.1 --port 25565 --version 1.8.9 \
 *       --schedule ../schedules/basic.json --out duel_recording.jsonl \
 *       [--name-prefix ValBot] [--settle-ms 3000]
 *
 * The server must be offline-mode and both bot accounts must be OP'd
 * (console: `op ValBotA`, `op ValBotB`) so the /tp commands work.
 *
 * Output JSONL rows:
 *   per tick, per bot:
 *     {tick, bot, pos:[x,y,z], vel:[x,y,z], yaw, pitch, onGround, health,
 *      inputs:{forward,back,left,right,jump,sprint,attack}}
 *   damage/knockback events (logged by the primary observer bot only, to
 *   avoid duplicates -- both connections see every entityHurt):
 *     {event:"hurt", tick, victim, attacker, pre_vel:[..], post_vel:[..]}
 *   plus one leading {event:"meta", ...} row.
 *
 * COORDINATE / YAW CONVENTIONS (must match harness.py -- the classic
 * silent validation bug lives here):
 *   - Logged positions are Minecraft feet coordinates (bot.entity.position).
 *   - Logged velocity is bot.entity.velocity, blocks per tick.
 *   - Logged yaw is NOTCHIAN degrees in [0, 360): 0 = +Z (south),
 *     90 = -X (west), increasing clockwise viewed from above.
 *     mineflayer stores yaw in radians with the OPPOSITE sense and a 180
 *     offset (mineflayer_yaw_rad = PI - notchian_yaw_rad), so we convert
 *     on every read/write below. Getting this wrong rotates the whole
 *     world and silently wrecks every RMSE downstream.
 *   - Logged pitch is notchian degrees, positive = looking DOWN
 *     (mineflayer pitch is positive-up, so it is negated).
 *   - Schedule yaw_delta_deg is a notchian per-tick delta; notchian and
 *     sim yaw differ by a constant offset so deltas transfer 1:1.
 *
 * Tick semantics: row `tick` is the state AFTER physics ran for that tick,
 * with `inputs` = the controls that were held DURING that tick. Controls
 * for tick t+1 are applied at the end of the tick-t handler because
 * mineflayer's 'physicsTick' fires after movement is integrated. The two
 * bots run on independent connections; their tick counters both start on
 * the shared "go" signal, so cross-bot skew is at most ~1 tick (compare.py
 * has --tick-offset if a systematic skew shows up).
 */
'use strict';

const fs = require('fs');
const path = require('path');

let mineflayer;
try {
  mineflayer = require('mineflayer');
} catch (err) {
  console.error(
    'mineflayer is not installed. Run `npm install` in tools/validation/recorder/ first.'
  );
  process.exit(1);
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
function parseArgs(argv) {
  const args = {
    host: '127.0.0.1',
    port: 25565,
    version: '1.8.9',
    schedule: path.join(__dirname, '..', 'schedules', 'basic.json'),
    out: 'duel_recording.jsonl',
    namePrefix: 'ValBot',
    settleMs: 3000,
  };
  for (let i = 2; i < argv.length; i += 2) {
    const key = argv[i];
    const val = argv[i + 1];
    switch (key) {
      case '--host': args.host = val; break;
      case '--port': args.port = parseInt(val, 10); break;
      case '--version': args.version = val; break;
      case '--schedule': args.schedule = val; break;
      case '--out': args.out = val; break;
      case '--name-prefix': args.namePrefix = val; break;
      case '--settle-ms': args.settleMs = parseInt(val, 10); break;
      default:
        console.error('unknown argument: ' + key);
        process.exit(2);
    }
  }
  return args;
}

// ---------------------------------------------------------------------------
// Yaw / pitch conversions (see header comment)
// ---------------------------------------------------------------------------
const RAD2DEG = 180 / Math.PI;
const DEG2RAD = Math.PI / 180;

function mod360(deg) {
  return ((deg % 360) + 360) % 360;
}
function toNotchianYawDeg(mineflayerYawRad) {
  return mod360(180 - mineflayerYawRad * RAD2DEG);
}
function toMineflayerYawRad(notchianYawDeg) {
  return (180 - notchianYawDeg) * DEG2RAD;
}
function toNotchianPitchDeg(mineflayerPitchRad) {
  return -mineflayerPitchRad * RAD2DEG;
}
function toMineflayerPitchRad(notchianPitchDeg) {
  return -notchianPitchDeg * DEG2RAD;
}

// ---------------------------------------------------------------------------
// Schedule handling (format documented in ../schedules/README.md)
// ---------------------------------------------------------------------------
const BOOL_KEYS = ['forward', 'back', 'left', 'right', 'jump', 'sprint', 'attack'];
const CONTROL_KEYS = ['forward', 'back', 'left', 'right', 'jump', 'sprint'];

function loadSchedule(file) {
  const sched = JSON.parse(fs.readFileSync(file, 'utf8'));
  if (!sched.bots || sched.bots.length !== 2) {
    throw new Error('schedule must define exactly two bots');
  }
  return sched;
}

function numTicks(sched) {
  return sched.segments.reduce((m, s) => Math.max(m, s.tick_end), 0);
}

/** Precompute per-tick inputs for one bot id; uncovered ticks are idle. */
function buildInputTrack(sched, botId) {
  const T = numTicks(sched);
  const track = new Array(T).fill(null).map(() => ({}));
  for (const seg of sched.segments) {
    if (seg.bot !== botId) continue;
    for (let t = seg.tick_start; t < seg.tick_end; t++) track[t] = seg.inputs;
  }
  return track;
}

function normalizedInputs(inp) {
  const out = {};
  for (const k of BOOL_KEYS) out[k] = !!inp[k];
  return out;
}

// ---------------------------------------------------------------------------
// Recorder
// ---------------------------------------------------------------------------
function main() {
  const args = parseArgs(process.argv);
  const sched = loadSchedule(args.schedule);
  const T = numTicks(sched);
  const stream = fs.createWriteStream(args.out);
  const log = (row) => stream.write(JSON.stringify(row) + '\n');

  log({
    event: 'meta',
    schedule: sched.meta && sched.meta.name,
    schedule_file: path.resolve(args.schedule),
    host: args.host,
    port: args.port,
    version: args.version,
    num_ticks: T,
    started_at: new Date().toISOString(),
  });

  const drivers = new Map(); // bot id -> driver
  let started = false;

  function usernameFor(botId) {
    return args.namePrefix + botId;
  }
  function botIdFor(username) {
    for (const id of sched.bots) if (usernameFor(id) === username) return id;
    return null;
  }

  function makeDriver(botId, isPrimary) {
    const spawn = sched.arena.spawn[botId];
    const track = buildInputTrack(sched, botId);
    const bot = mineflayer.createBot({
      host: args.host,
      port: args.port,
      username: usernameFor(botId),
      version: args.version,
    });

    const d = {
      id: botId,
      bot,
      tick: -1,
      done: false,
      spawned: false,
      // camera target held in NOTCHIAN degrees; deltas accumulate here
      targetYawDeg: spawn.yaw_mc_deg,
      targetPitchDeg: 0,
      currentInputs: {}, // inputs active during the tick being integrated
    };

    function applyInputs(inp) {
      for (const k of CONTROL_KEYS) bot.setControlState(k, !!inp[k]);
      const dYaw = inp.yaw_delta_deg || 0;
      const dPitch = inp.pitch_delta_deg || 0;
      if (dYaw !== 0 || dPitch !== 0) {
        d.targetYawDeg = mod360(d.targetYawDeg + dYaw);
        d.targetPitchDeg = Math.max(-90, Math.min(90, d.targetPitchDeg + dPitch));
        bot.look(
          toMineflayerYawRad(d.targetYawDeg),
          toMineflayerPitchRad(d.targetPitchDeg),
          true // force: snap instantly, no smooth interpolation
        );
      }
      if (inp.attack) {
        const target = nearestOtherPlayer(bot);
        if (target && clientRayHits(bot, target, d)) {
          bot.attack(target);
        }
      }
      d.currentInputs = inp;
    }

    // Client-faithful attack gate (MCP-919 EntityRenderer.getMouseOver):
    // ray from the EYE along the view vector, intersected with the target
    // AABB expanded by Entity.getCollisionBorderSize() = 0.1, hit param
    // <= 3.0 blocks. The old center-distance <= 3.0 gate was neither the
    // client rule nor the server rule, so recorded hit ground truth never
    // matched either world.
    function clientRayHits(b, target, d) {
      const EYE = 1.62;
      const HALF_W = 0.3 + 0.1;
      const BOX_H = 1.8;
      const BORDER = 0.1;
      const REACH = 3.0;
      const yawN = (d.targetYawDeg * Math.PI) / 180.0; // notchian rad
      const pitN = (d.targetPitchDeg * Math.PI) / 180.0;
      // notchian: yaw 0 = +Z, 90 = -X; pitch down-positive
      const vx = -Math.sin(yawN) * Math.cos(pitN);
      const vy = -Math.sin(pitN);
      const vz = Math.cos(yawN) * Math.cos(pitN);
      const ox = b.entity.position.x;
      const oy = b.entity.position.y + EYE;
      const oz = b.entity.position.z;
      const tx = target.position.x;
      const ty = target.position.y;
      const tz = target.position.z;
      const lo = [tx - HALF_W, ty - BORDER, tz - HALF_W];
      const hi = [tx + HALF_W, ty + BOX_H + BORDER, tz + HALF_W];
      const o = [ox, oy, oz];
      const v = [vx, vy, vz];
      let tmin = -Infinity;
      let tmax = Infinity;
      for (let i = 0; i < 3; i++) {
        if (Math.abs(v[i]) < 1e-9) {
          if (o[i] < lo[i] || o[i] > hi[i]) return false;
        } else {
          let t0 = (lo[i] - o[i]) / v[i];
          let t1 = (hi[i] - o[i]) / v[i];
          if (t0 > t1) { const tmp = t0; t0 = t1; t1 = tmp; }
          tmin = Math.max(tmin, t0);
          tmax = Math.min(tmax, t1);
        }
      }
      return tmax >= Math.max(tmin, 0) && tmin <= REACH;
    }

    function nearestOtherPlayer(b) {
      let best = null;
      let bestDist = Infinity;
      for (const key of Object.keys(b.entities)) {
        const e = b.entities[key];
        if (!e || e.type !== 'player' || e === b.entity) continue;
        const dist = b.entity.position.distanceTo(e.position);
        if (dist < bestDist) {
          bestDist = dist;
          best = e;
        }
      }
      return best;
    }

    function logTickRow() {
      const e = bot.entity;
      log({
        tick: d.tick,
        bot: botId,
        pos: [e.position.x, e.position.y, e.position.z],
        vel: [e.velocity.x, e.velocity.y, e.velocity.z],
        yaw: toNotchianYawDeg(e.yaw),
        pitch: toNotchianPitchDeg(e.pitch),
        onGround: !!e.onGround,
        health: bot.health,
        inputs: normalizedInputs(d.currentInputs),
      });
    }

    function finish() {
      if (d.done) return;
      d.done = true;
      for (const k of CONTROL_KEYS) bot.setControlState(k, false);
      maybeFinishAll();
    }

    bot.once('spawn', () => {
      d.spawned = true;
      console.error('[' + botId + '] spawned as ' + usernameFor(botId));
      // Teleport to the arena spawn with the schedule's facing. 1.8 /tp
      // takes notchian yaw/pitch degrees. Requires OP.
      const p = spawn.pos;
      bot.chat(
        '/tp ' + usernameFor(botId) + ' ' +
        p[0] + ' ' + p[1] + ' ' + p[2] + ' ' +
        spawn.yaw_mc_deg + ' 0'
      );
      maybeStartAll();
    });

    bot.on('physicsTick', () => {
      if (!started || d.done) return;
      d.tick += 1;
      if (d.tick >= T) {
        finish();
        return;
      }
      // State AFTER physics for tick d.tick, with the inputs that were
      // active during it; then stage inputs for the next tick.
      logTickRow();
      const next = d.tick + 1;
      applyInputs(next < T ? track[next] : {});
    });

    if (isPrimary) {
      // Damage/knockback observation. Only the primary bot's connection
      // logs hurt events so each event appears exactly once. The victim's
      // velocity right after the hurt tick carries the knockback impulse
      // (the server sends it as an entity-velocity packet).
      bot.on('entityHurt', (entity) => {
        if (!started || !entity || entity.type !== 'player') return;
        const victimId = entity.username ? botIdFor(entity.username) : null;
        if (!victimId) return;
        const atTick = d.tick;
        const pre = [entity.velocity.x, entity.velocity.y, entity.velocity.z];
        bot.once('physicsTick', () => {
          log({
            event: 'hurt',
            tick: atTick,
            victim: victimId,
            attacker: sched.bots.find((b) => b !== victimId),
            pre_vel: pre,
            post_vel: [entity.velocity.x, entity.velocity.y, entity.velocity.z],
          });
        });
      });
    }

    bot.on('kicked', (reason) => {
      console.error('[' + botId + '] kicked: ' + JSON.stringify(reason));
      process.exitCode = 1;
      finish();
    });
    bot.on('error', (err) => {
      console.error('[' + botId + '] error: ' + err.message);
      process.exitCode = 1;
      finish();
    });

    d.applyInputs = applyInputs;
    d.track = track;
    d.finish = finish;
    return d;
  }

  function maybeStartAll() {
    if (started) return;
    for (const d of drivers.values()) if (!d.spawned) return;
    // Give the teleports + chunk loads time to settle, then fire the
    // shared "go" signal: both tick counters start on their next
    // physicsTick, and tick-0 inputs are pre-applied so the first
    // integrated tick already uses them.
    setTimeout(() => {
      console.error('starting schedule (' + T + ' ticks)');
      for (const d of drivers.values()) d.applyInputs(d.track[0] || {});
      started = true;
    }, args.settleMs);
  }

  function maybeFinishAll() {
    for (const d of drivers.values()) if (!d.done) return;
    // Small grace period so trailing hurt events get their post-velocity.
    setTimeout(() => {
      stream.end(() => {
        for (const d of drivers.values()) d.bot.quit();
        console.error('recording written to ' + path.resolve(args.out));
        process.exit(process.exitCode || 0);
      });
    }, 1000);
  }

  sched.bots.forEach((botId, i) => {
    drivers.set(botId, makeDriver(botId, i === 0));
  });

  // Hard safety timeout: schedule length + generous margin.
  const maxMs = (T / 20) * 1000 + args.settleMs + 60000;
  setTimeout(() => {
    console.error('safety timeout hit; aborting');
    process.exit(1);
  }, maxMs).unref();
}

main();
