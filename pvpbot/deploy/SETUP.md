# Live deployment setup (macOS)

Runbook for taking a trained checkpoint into a real Minecraft 1.8.9 window.
Nothing here is needed for tests or `--dry-run` — those run fully offline
with mocks.

## 0. Ethics rule (non-negotiable)

Run the bot **only on private servers, against opponents who have
explicitly consented** to sparring with a bot. Never connect it to public
servers, ranked queues, or matches with unaware players — that is cheating,
violates server rules and the Minecraft EULA spirit, and taints any data
you record. The flight recorder logs every action precisely so sessions are
auditable; keep it enabled.

## 1. Install the optional Quartz dependency

The system Python 3.9 (`python3` = 3.9.6) needs the pyobjc Quartz bindings
for capture + injection:

```sh
python3 -m pip install --user pyobjc-framework-Quartz
# or, if you use uv:
uv pip install pyobjc-framework-Quartz
```

Verify:

```sh
python3 -m pvpbot.deploy.capture --check
```

## 2. Grant macOS permissions to your terminal

Both permissions go to the app that launches `python3` (Terminal, iTerm2,
VS Code, …). Quit and relaunch the terminal after granting.

- **Screen Recording** (window capture):
  System Settings → Privacy & Security → Screen & System Audio Recording →
  enable your terminal app. (On macOS 12 and earlier: System Preferences →
  Security & Privacy → Privacy → Screen Recording.)
- **Accessibility** (posting keyboard/mouse events):
  System Settings → Privacy & Security → Accessibility → enable your
  terminal app.

Re-run `python3 -m pvpbot.deploy.capture --check` — it should report the
screen-recording permission OK. The first real `CGEventPost` may still pop
a one-time prompt for input monitoring; accept it.

## 3. Minecraft 1.8.9 client settings

- **Windowed mode**, not fullscreen (window capture needs a window). A
  fixed size around 1280×720 is ideal; do not resize mid-session — the
  capture is downscaled to 114×64 either way, but perception was trained on
  a 16:9-ish aspect.
- The window title must contain **"Minecraft"** (vanilla 1.8.9 titles
  itself `Minecraft 1.8.9`; launchers/forks may differ — pass
  `--window-title` if so).
- Options → Controls → **Raw Input: OFF** if your client has that toggle
  (vanilla 1.8.9 predates it and always reads OS cursor deltas, which is
  what our relative-move injection produces).
- **Mouse Sensitivity**: pick a value and never change it — the
  degrees→pixels calibration below is only valid for one sensitivity.
  100% is a fine choice.
- **FOV**: pick one (Normal/70 recommended) and keep it fixed; perception's
  bbox-height→distance estimate assumes a constant FOV.
- Turn off view bobbing (Options → Video Settings) to reduce optic-flow
  noise in perception.
- Keybinds must match the injected keys: W/A/S/D movement, Space jump,
  **Sprint bound to X** (Options -> Controls -> Sprint). The sink posts
  X, never Left Ctrl: macOS turns Ctrl+left-click into a RIGHT click,
  so a Ctrl-sprinting bot blocks with its sword instead of attacking.
  Left-click attack.

## 4. Calibration (degrees → pixels)

The policy emits camera deltas in degrees/tick; the sink injects pixels.
Measure your pixels-per-degree once per sensitivity/DPI setup:

```sh
python3 -m pvpbot.deploy.run --calibrate   # prints the full procedure
```

**Measured on this machine (2026-07-20, vanilla 1.8.9, current sensitivity,
854x508 window): 1000 px = 150.0 deg exactly -> `--px-per-degree 6.667`.**
Both axes verified live: 30 deg commanded flick landed exactly, screen-down
= pitch-down. Re-measure only if sensitivity, DPI, or the OS changes.

Short version: note yaw from the F3 overlay, run
`python3 -m pvpbot.deploy.run --nudge 1000` with Minecraft focused, read
the new yaw, then `px_per_degree = 1000 / |Δyaw|`. Pass it to the runner
with `--px-per-degree`. Sanity check: a commanded 30° flick should land
within ~1° on the F3 readout.

## 5. Private server

Host a local or LAN 1.8.9 server (e.g. a vanilla or Spigot 1.8.8/1.8.9 jar
on `localhost`), flat arena, `/gamerule naturalRegeneration false` matches
the sim's damage model more closely. Give both players identical swords and
no armor to match training. Only consenting humans join.

## 6. Run

```sh
# smoke test the whole loop with mocks first:
python3 -m pvpbot.deploy.run --dry-run --ticks 100

# live, with a trained checkpoint:
python3 -m pvpbot.deploy.run --checkpoint runs/ppo/latest.pt \
    --px-per-degree <measured> --ticks 2400
```

Focus the Minecraft window immediately after launching. To stop at any
time:

```sh
touch /tmp/pvpbot-stop
```

The loop checks that sentinel every tick (50 ms), releases all held keys
and buttons, and writes the flight recorder (`pvpbot-flight.jsonl` by
default) — one JSON line per tick with obs, perception, action, and
per-stage latencies, ready for post-match analysis or BC data mining.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `RuntimeError: Quartz (pyobjc) is required…` | Step 1. |
| `--check` shows permission MISSING | Step 2; fully quit + relaunch the terminal. |
| `No on-screen window matching 'Minecraft'` | Client running? Windowed? Try `--window-title` with your launcher's title. |
| Mouse moves but the camera doesn't | The Minecraft window isn't focused, or a raw-input mod is eating OS deltas. |
| Aim consistently over/under-shoots | Re-run calibration; sensitivity or DPI changed. |
| `tick exceeded budget` warnings | Close other apps; capture is the usual culprit — shrink the game window. |
