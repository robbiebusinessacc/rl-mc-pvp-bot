# mc-pvp-bot — Integration Contracts

Local-only (Apple M4 Max, 16-core / 64 GB, no cloud) Minecraft sword-PvP bot:
vectorized physics sim → self-play PPO on privileged state → pixel perception
CNN → BC human prior with KL regularization → live macOS deployment.

## Hard rules for module branches

1. **Never modify shared files**: `pvpbot/spec.py`, `pvpbot/models.py`,
   `pvpbot/sim/stub.py`, `SPEC.md`, `README.md`, `requirements.txt`,
   `.gitignore`. Import from them.
2. Work **only** inside your assigned directory plus `tests/test_<module>_*.py`.
3. Target **Python 3.9** (`python3` = 3.9.6 system). Available: numpy 2.0.2,
   torch 2.8.0 (MPS works), pytest 8.4.2. No new required dependencies —
   optional extras must degrade gracefully with a clear skip/error message.
4. Tests must run **offline** (no network, no Minecraft, no screen capture) via
   `python3 -m pytest tests/ -q`. Mock anything external.
5. Commit all work to your branch before finishing.

## Module map

| Directory | Owner agent | Deliverable |
|---|---|---|
| `pvpbot/sim/` | sim-engine | 1.8-accurate vectorized `DuelVecEnv` (`env.py`), ≥1M steps/s @ 4096 envs |
| `pvpbot/train/` | selfplay-trainer | PPO + self-play league, checkpointing, CLI |
| `pvpbot/perception/` | perception | synthetic data gen, CNN training (CPU+MPS), obs-assembly adapter |
| `pvpbot/bc/` | bc-prior | human recording format, BC training, KL-to-prior for PPO, humanization |
| `pvpbot/deploy/` | deploy-live | macOS capture → perception → policy → input injection, dry-run mode |
| `pvpbot/eval/` | eval-ladder | scripted opponents, Elo/TrueSkill ladder, reports |
| `tools/validation/` | validation-tools | mineflayer ground-truth recorder + sim trajectory comparer |

## Contracts

All interface constants and layouts live in `pvpbot/spec.py`:
observation layout (`OBS_LAYOUT`, 48 floats), action space (`ACTION_HEADS`,
7 categorical heads incl. binned camera deltas), perception output
(`PERCEPTION_LAYOUT`, 12 floats), env API and checkpoint format (docstring at
bottom of `spec.py`). Network architectures live in `pvpbot/models.py`
(`PolicyNet`, `PerceptionCNN`) — train these, don't redefine them.

`pvpbot/sim/stub.py` implements the env API with crude physics: build and test
against it until the real sim merges. The real sim must be a drop-in
replacement (same class name `DuelVecEnv`, importable as
`from pvpbot.sim.env import DuelVecEnv`).
