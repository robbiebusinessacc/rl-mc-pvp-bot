/*
 * Ground-truth target for real-frame perception data collection, and
 * (with --fight) a live port of the pvpbot/eval/practice.py tiers so the
 * learned policy can be tested against P4-Hacker on the real server.
 *
 * A mineflayer bot joins the local 1.8.9 server and logs per-tick ground
 * truth to JSONL:
 *
 *   {t, sp:[x,y,z], sv:[vx,vy,vz], g, cp:[x,y,z], cy, cpt}
 *     t   wall-clock ms (Date.now(), same clock as the python capturer)
 *     sp/sv/g  self (target) feet position, velocity, onGround
 *     cp  camera player's feet position
 *     cy/cpt   camera player's RAW mineflayer yaw/pitch (radians;
 *              notchian_deg = deg(PI - cy), mc_pitch_deg = -deg(cpt))
 *   {ev:"hurt", t}  when the target takes damage (drives hurt_flash labels)
 *   {ev:"swing", t} when the target attacks the camera player (ray-gated,
 *                   so at <3.05 blocks these effectively all land)
 *
 * Movement: without --fight, varied data-collection patterns at 1.5-12
 * blocks. With --fight easy|medium|hard|hacker, the PracticeBot state
 * machine (mirrors practice.py number-for-number): bounded-rate aim with
 * reaction delay, managed spacing, strafing, hops, crit jumps, CPS-capped
 * ray-aware clicking, w-taps, and (hacker) hurt-window-timed swings.
 * --fight 1 keeps the legacy simple swing-back (1.4s cadence).
 *
 * Gear is staged server-side (replaceitem + keepInventory); this bot just
 * holds hotbar slot 0.
 *
 * Run:  NODE_PATH=../validation/recorder/node_modules node target_bot.js \
 *         --host 127.0.0.1 --port 25565 --cam PainEXE --seconds 600 \
 *         [--fight hacker] --out /tmp/target_log.jsonl
 */
'use strict';
const fs = require('fs');
const mineflayer = require('mineflayer');

const args = {};
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i += 2) args[argv[i].replace(/^--/, '')] = argv[i + 1];

const CAM = args.cam || 'PainEXE';
const FIGHT = args.fight || '';
const SECONDS = parseInt(args.seconds || '600', 10);
const out = fs.createWriteStream(args.out || 'target_log.jsonl');

// practice.py difficulty table, verbatim (deg/s, ticks, clicks/s, rates)
const TIERS = {
  easy:   { rot: 120, react: 6, cps: 4,  crit: 0.10, wtap: 0.10, strafe: 0.30,
            jitter: 4.0, lo: 1.8, hi: 2.7, chase: 3.2, hurtTimed: false },
  medium: { rot: 240, react: 3, cps: 8,  crit: 0.40, wtap: 0.50, strafe: 0.60,
            jitter: 2.5, lo: 1.8, hi: 2.7, chase: 3.2, hurtTimed: false },
  hard:   { rot: 480, react: 1, cps: 12, crit: 0.80, wtap: 0.90, strafe: 0.85,
            jitter: 1.5, lo: 1.8, hi: 2.7, chase: 3.2, hurtTimed: false },
  // react: 0 (frame-perfect). The react:1 "spec reading" was beaten
  // consistently by Robbie in a fresh-equal-gear duel (2026-07-23); the
  // real practice-server hacker beats most humans, and react:0 is the
  // port that matched that feel. Human calibration outranks spec reading.
  // OFFICIAL BAR (Robbie calibration 2026-07-24): frame-perfect react,
  // rot 600 / jitter 1.5 / cps 11 = matches the real practice-server
  // hacker feel. Old 720/0.5/13 exceeded it; react:1 was below it.
  hacker: { rot: 600, react: 0, cps: 11, crit: 1.00, wtap: 1.00, strafe: 1.00,
            jitter: 1.5, lo: 2.0, hi: 2.8, chase: 3.2, hurtTimed: true },
};
// --tweak 'k=v,k=v' overrides tier params (bar-calibration duels)
if (args.tweak && TIERS[FIGHT]) {
  for (const kv of String(args.tweak).split(',')) {
    const [k, v] = kv.split('=');
    if (k && v !== undefined) TIERS[FIGHT][k.trim()] = parseFloat(v);
  }
  console.log('[target] tier tweaks: ' + args.tweak);
}
const TIER = TIERS[FIGHT] || null;
// 'idle' = stand perfectly still, never attack: the stationary-target
// conversion probe (measures click->hit registration with motion removed).
const IDLE = FIGHT === 'idle';
if (FIGHT && !TIER && !IDLE && FIGHT !== '1' && FIGHT !== 'true') {
  console.error('unknown --fight tier: ' + FIGHT);
  process.exit(2);
}

const bot = mineflayer.createBot({
  host: args.host || '127.0.0.1',
  port: parseInt(args.port || '25565', 10),
  username: args.name || 'PvPTarget',
  version: args.version || '1.8.9',
});

bot.on('error', (e) => { console.error('bot error:', e.message); });
bot.on('entityHurt', (e) => {
  if (bot.entity && e.id === bot.entity.id) {
    out.write(JSON.stringify({ ev: 'hurt', t: Date.now() }) + '\n');
  }
  const cam = bot.players[CAM] && bot.players[CAM].entity;
  if (cam && e.id === cam.id) { camHurtTicks = 10; camJustHurt = true; } // 1.8 hurt window ~10 ticks
});

const MODES = ['approach', 'retreat', 'strafeL', 'strafeR', 'idle', 'jumpy', 'wander'];
let attackCd = 0;
let mode = 'approach';
let modeTicks = 0;
let wanderYaw = 0;

// practice-bot state (mirrors PracticeBot._alloc for n=1)
let camHurtTicks = 0;
let camJustHurt = false;
let clickCd = 0;
let strafeDir = Math.random() < 0.5 ? -1 : 1;
let strafeTimer = 10 + Math.floor(Math.random() * 40);
let strafing = TIER ? Math.random() < TIER.strafe : false;
let wtapPhase = 0;
let critPhase = 0;
let hopTimer = 20 + Math.floor(Math.random() * 40);
const errBuf = [];   // reaction delay buffer of [yawErr, pitchErr] (rad)

function gauss() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}
function wrapAngle(a) {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}
function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

function clearControls() {
  ['forward', 'back', 'left', 'right', 'jump', 'sprint'].forEach((c) => bot.setControlState(c, false));
}

// One practice-bot tick (mirrors PracticeBot._act row-for-row for n=1).
function practiceTick(cam) {
  const T = TIER;
  const D2R = Math.PI / 180;
  const eye = cam.position.offset(0, 1.62, 0);
  const selfEye = bot.entity.position.offset(0, 1.62, 0);
  const dx = eye.x - selfEye.x, dy = eye.y - selfEye.y, dz = eye.z - selfEye.z;
  const distH = Math.sqrt(dx * dx + dz * dz);
  const dist = bot.entity.position.distanceTo(cam.position);
  const onGround = bot.entity.onGround;

  // current aim error to the target eye (mineflayer look convention)
  const wantYaw = Math.atan2(-dx, -dz);
  const wantPitch = Math.atan2(dy, distH);
  const yawNow = wrapAngle(wantYaw - bot.entity.yaw);
  const pitchNow = wantPitch - bot.entity.pitch;

  // reaction delay: rotation homes on errors from `react` ticks ago
  let yawErr = yawNow, pitchErr = pitchNow;
  if (T.react > 0) {
    errBuf.push([yawNow, pitchNow]);
    if (errBuf.length > T.react) { const d = errBuf.shift(); yawErr = d[0]; pitchErr = d[1]; }
    else { yawErr = 0; pitchErr = 0; }
  }

  // bounded-rate aim (wurst-style) + tracking jitter
  const cap = (T.rot / 20) * D2R;
  const jit = gauss() * T.jitter * D2R;
  const dYaw = clamp(yawErr + jit, -cap, cap);
  const dPitch = clamp(pitchErr + 0.5 * jit, -cap, cap);
  bot.look(bot.entity.yaw + dYaw, clamp(bot.entity.pitch + dPitch, -1.5, 1.5), true);

  clearControls();

  // distance management: back up under lo, walk in beyond hi, chase-sprint
  if (dist > T.hi) {
    bot.setControlState('forward', true);
    if (dist > T.chase) bot.setControlState('sprint', true);
  } else if (dist < T.lo) {
    bot.setControlState('back', true);
  }

  // strafing with random direction switches
  strafeTimer -= 1;
  if (strafeTimer <= 0) {
    strafeDir = Math.random() < 0.5 ? -1 : 1;
    strafing = Math.random() < T.strafe;
    strafeTimer = 10 + Math.floor(Math.random() * 40);
  }
  if (strafing && dist < T.chase) {
    bot.setControlState(strafeDir > 0 ? 'right' : 'left', true);
  }

  // natural hops + crit sequence (jump, swing while falling)
  hopTimer -= 1;
  const canCrit = onGround && dist > 1.8 && dist < T.hi && camHurtTicks <= 0
    && critPhase === 0 && Math.random() < T.crit * 0.04;
  if (canCrit) critPhase = 9;
  if (critPhase === 9) bot.setControlState('jump', true);
  else if (hopTimer <= 0 && onGround && critPhase === 0) {
    bot.setControlState('jump', true);
    hopTimer = 20 + Math.floor(Math.random() * 40);
  }
  if (critPhase > 0) critPhase -= 1;

  // clicking: CPS-capped, ray-aware, optionally hurt-window-timed
  camHurtTicks = Math.max(camHurtTicks - 1, 0);
  clickCd = Math.max(clickCd - 1, 0);
  const dSafe = Math.max(dist, 0.5);
  const halfW = Math.atan2(0.35, dSafe);
  const halfH = Math.atan2(1.05, dSafe);
  const aimed = Math.abs(yawNow) < halfW && Math.abs(pitchNow) < halfH;
  const inReach = dist < 3.05;
  let want = aimed && inReach && clickCd === 0;
  if (T.hurtTimed) want = want && camHurtTicks <= 0;
  if (T.crit > 0 && critPhase > 0 && critPhase <= 4 && !onGround) {
    want = want || (aimed && inReach && clickCd === 0);
  }
  if (want) {
    try { bot.attack(cam); } catch (e) {}
    out.write(JSON.stringify({ ev: 'swing', t: Date.now() }) + '\n');
    clickCd = Math.max(Math.round(20 / T.cps), 1);
  }

  // w-tap on a LANDED hit (cam hurt rising edge, mirrors practice.py):
  // release forward 1 tick to reset sprint, then re-press
  if (camJustHurt) {
    camJustHurt = false;
    if (Math.random() < T.wtap) wtapPhase = 2;
  }

  // w-tap phases override forward state
  if (wtapPhase === 2) {
    bot.setControlState('forward', false);
    bot.setControlState('sprint', false);
  } else if (wtapPhase === 1) {
    bot.setControlState('forward', true);
    bot.setControlState('sprint', true);
  }
  if (wtapPhase > 0) wtapPhase -= 1;
}

function collectTick(cam) {
  const dx = cam.position.x - bot.entity.position.x;
  const dz = cam.position.z - bot.entity.position.z;
  const dist = Math.sqrt(dx * dx + dz * dz);

  if (--modeTicks <= 0) {
    mode = MODES[Math.floor(Math.random() * MODES.length)];
    modeTicks = 40 + Math.floor(Math.random() * 80); // 2-6 s
    wanderYaw = Math.random() * Math.PI * 2;
  }
  // distance guardrails override the chosen mode
  let m = mode;
  if (dist > 11) m = 'approach';
  else if (dist < 1.5) m = 'retreat';

  const eye = cam.position.offset(0, 1.62, 0);
  clearControls();
  switch (m) {
    case 'approach':
      bot.lookAt(eye, true); bot.setControlState('forward', true); break;
    case 'retreat':
      bot.lookAt(eye, true); bot.setControlState('back', true); break;
    case 'strafeL':
      bot.lookAt(eye, true); bot.setControlState('left', true); break;
    case 'strafeR':
      bot.lookAt(eye, true); bot.setControlState('right', true); break;
    case 'jumpy':
      bot.lookAt(eye, true);
      bot.setControlState('forward', dist > 3);
      bot.setControlState('jump', true);
      break;
    case 'wander':
      bot.look(wanderYaw, 0, true); bot.setControlState('forward', true); break;
    case 'idle':
    default:
      bot.lookAt(eye, true); break;
  }

  // legacy simple fight-back: swing at ~1.4s cadence in sword range
  if (FIGHT === '1' || FIGHT === 'true') {
    attackCd -= 1;
    const d3 = bot.entity.position.distanceTo(cam.position);
    if (attackCd <= 0 && d3 < 3.0) {
      try { bot.attack(cam); } catch (e) {}
      attackCd = 28;
    }
  }
}

bot.once('spawn', () => {
  console.error('[target] spawned as ' + bot.username
    + (TIER ? ' | fight tier: ' + FIGHT : ''));
  try { bot.setQuickBarSlot(0); } catch (e) {}
  const t0 = Date.now();
  const timer = setInterval(() => {
    if (Date.now() - t0 > SECONDS * 1000) {
      clearInterval(timer);
      clearControls();
      out.end(() => process.exit(0));
      return;
    }
    const cam = bot.players[CAM] && bot.players[CAM].entity;
    if (!cam || !bot.entity) return;

    if (TIER) practiceTick(cam);
    else if (IDLE) { /* stationary probe: no movement, no attack */ }
    else collectTick(cam);

    out.write(JSON.stringify({
      t: Date.now(),
      sp: [bot.entity.position.x, bot.entity.position.y, bot.entity.position.z],
      sv: [bot.entity.velocity.x, bot.entity.velocity.y, bot.entity.velocity.z],
      g: bot.entity.onGround ? 1 : 0,
      cp: [cam.position.x, cam.position.y, cam.position.z],
      cy: cam.yaw,
      cyh: (cam.headYaw !== undefined && cam.headYaw !== null) ? cam.headYaw : cam.yaw,
      cpt: cam.pitch,
    }) + '\n');
  }, 50);
});
