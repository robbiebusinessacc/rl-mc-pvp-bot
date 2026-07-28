"""Arena: runs K duels between two contestants on the vectorized duel env.

A "contestant" is anything with::

    act(obs: np.ndarray (N, 48)) -> np.ndarray (N, 7) int64

plus optional ``begin(num_envs)`` and ``on_done(done_mask)`` lifecycle hooks
(scripted tiers and the checkpoint wrapper both have them; bare callables are
wrapped).  Checkpoints saved in the spec.py format
(``torch.save({"model": state_dict, "meta": {...}}, path)``) are loaded into
the shared ``PolicyNet`` with its GRU state handled per env slot (reset on
episode boundaries).

The env is imported as ``pvpbot.sim.env.DuelVecEnv`` when the real sim is
merged, falling back to ``pvpbot.sim.stub.DuelVecEnv``.

Per-match stats collected from each side's observation stream:

* hits landed / taken -- detected as drops of the enemy_hp / self_hp slots
  between consecutive observations; the killing blow on a decisive game is
  credited from ``info["win"]`` (the post-done obs belongs to a fresh
  episode, so the final drop is not otherwise visible).
* avg combo length -- a combo is a maximal run of >= 1 hits dealt without
  taking one (taking a hit closes the combo; episode end closes any open
  combo).  On a tick with a mutual exchange, the taken hit is processed
  first (the exchange breaks the previous combo, the dealt hit starts a
  new one).
* mean aim error (degrees) -- mean of |aim_err_yaw| * 180 over all ticks.
* episode length -- ticks from reset to done.

Only each env's FIRST episode counts (the env auto-resets; envs that finish
keep simulating but stop being recorded), so ``num_duels`` envs yield exactly
``num_duels`` results.  Side assignment alternates per env slot so neither
contestant systematically plays side 0.
"""
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from pvpbot.spec import NUM_ACTION_HEADS, OBS_DIM, OBS_LAYOUT

try:  # real sim lands on a parallel branch as pvpbot/sim/env.py
    from pvpbot.sim.env import DuelVecEnv  # type: ignore
except ImportError:  # pragma: no cover - depends on merge state
    from pvpbot.sim.stub import DuelVecEnv

_SELF_HP = OBS_LAYOUT["self_hp"][0]
_ENEMY_HP = OBS_LAYOUT["enemy_hp"][0]
_AIM_YAW = OBS_LAYOUT["aim_err_yaw"][0]
_HP_EPS = 1e-4


# ---------------------------------------------------------------------------
# Contestant wrappers
# ---------------------------------------------------------------------------
class FnContestant:
    """Wraps a bare ``f(obs)->actions`` callable in the contestant protocol."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], name: str = None):
        self._fn = fn
        self.name = name or getattr(fn, "__name__", "fn")

    def begin(self, num_envs: int) -> None:
        pass

    def on_done(self, done: np.ndarray) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self._fn(obs), dtype=np.int64)


class CheckpointContestant:
    """PolicyNet checkpoint (spec.py format) with recurrent state handling."""

    def __init__(self, path: str, deterministic: bool = False,
                 name: str = None):
        import torch  # local import: keep arena importable without torch use
        from pvpbot.models import PolicyNet

        self._torch = torch
        blob = torch.load(path, map_location="cpu")
        if "model" not in blob or "meta" not in blob:
            raise ValueError(
                "checkpoint %r missing 'model'/'meta' (spec.py format)" % path)
        self.meta: Dict[str, Any] = blob["meta"]
        obs_dim = int(self.meta.get("obs_dim", OBS_DIM))
        self.net = PolicyNet(obs_dim=obs_dim)
        self.net.load_state_dict(blob["model"])
        self.net.eval()
        # trainer checkpoints carry the obs RunningNorm the policy was
        # trained behind; feeding raw obs instead silently cripples policies
        # whose norm is far from identity (an FOV-trained ckpt went from
        # winning in training to 0/500 on the ladder before this was applied)
        self.norm = None
        if "obs_norm" in blob:
            from pvpbot.train.ppo import RunningNorm

            self.norm = RunningNorm(obs_dim)
            self.norm.load_state_dict(blob["obs_norm"])
        self.deterministic = deterministic
        self.name = name or os.path.splitext(os.path.basename(path))[0]
        self._state = None

    def begin(self, num_envs: int) -> None:
        self._state = self.net.initial_state(num_envs)

    def on_done(self, done: np.ndarray) -> None:
        mask = self._torch.from_numpy(np.ascontiguousarray(done, dtype=bool))
        self._state[mask] = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self._state is None or self._state.shape[0] != obs.shape[0]:
            self.begin(obs.shape[0])
        if self.norm is not None:
            obs = self.norm.normalize(np.ascontiguousarray(obs, np.float32))
        t_obs = self._torch.from_numpy(np.ascontiguousarray(obs, np.float32))
        actions, self._state = self.net.act(
            t_obs, self._state, deterministic=self.deterministic)
        return actions.cpu().numpy().astype(np.int64)


def as_contestant(x: Any) -> Any:
    """Coerce a ScriptedPolicy / checkpoint path / callable into a contestant."""
    if hasattr(x, "act") and hasattr(x, "begin"):
        return x
    if isinstance(x, str):
        return CheckpointContestant(x)
    if callable(x):
        return FnContestant(x)
    raise TypeError("cannot use %r as a contestant" % (x,))


# ---------------------------------------------------------------------------
# Match result
# ---------------------------------------------------------------------------
@dataclass
class SideStats:
    hits_landed: float = 0.0        # mean per duel
    hits_taken: float = 0.0         # mean per duel
    avg_combo: float = 0.0          # mean combo length (>=1-hit runs)
    mean_aim_err_deg: float = 0.0   # mean |aim_err_yaw| in degrees

    def to_dict(self) -> Dict[str, float]:
        return {
            "hits_landed": self.hits_landed,
            "hits_taken": self.hits_taken,
            "avg_combo": self.avg_combo,
            "mean_aim_err_deg": self.mean_aim_err_deg,
        }


@dataclass
class MatchResult:
    name_a: str
    name_b: str
    num_duels: int
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0
    mean_episode_len: float = 0.0
    stats_a: SideStats = field(default_factory=SideStats)
    stats_b: SideStats = field(default_factory=SideStats)

    @property
    def decisive(self) -> int:
        return self.wins_a + self.wins_b

    def score_a(self) -> float:
        """Series score for a in [0, 1] (draws count half)."""
        n = self.wins_a + self.wins_b + self.draws
        return 0.5 if n == 0 else (self.wins_a + 0.5 * self.draws) / n

    def to_dict(self) -> Dict[str, Any]:
        return {
            "a": self.name_a, "b": self.name_b, "num_duels": self.num_duels,
            "wins_a": self.wins_a, "wins_b": self.wins_b, "draws": self.draws,
            "mean_episode_len": self.mean_episode_len,
            "stats_a": self.stats_a.to_dict(),
            "stats_b": self.stats_b.to_dict(),
        }


# ---------------------------------------------------------------------------
# Match runner
# ---------------------------------------------------------------------------
class _SideTracker:
    """Accumulates per-side stats over the first episode of each env."""

    def __init__(self, n: int):
        self.hits_dealt = np.zeros(n, dtype=np.int64)
        self.hits_taken = np.zeros(n, dtype=np.int64)
        self.combo = np.zeros(n, dtype=np.int64)
        self.combos: List[int] = []
        self.aim_sum = 0.0
        self.aim_ticks = 0

    def tick(self, pending: np.ndarray, obs_side: np.ndarray) -> None:
        errs = np.abs(obs_side[pending, _AIM_YAW]) * 180.0
        self.aim_sum += float(errs.sum())
        self.aim_ticks += int(pending.sum())

    def hits(self, dealt: np.ndarray, taken: np.ndarray) -> None:
        # taken first: a mutual exchange breaks the running combo, then the
        # dealt hit starts a new one.
        closing = taken & (self.combo > 0)
        self.combos.extend(self.combo[closing].tolist())
        self.combo[taken] = 0
        self.hits_taken[taken] += 1
        self.combo[dealt] += 1
        self.hits_dealt[dealt] += 1

    def finish(self, idx: np.ndarray, killing_blow: np.ndarray,
               killed: np.ndarray) -> None:
        """Close out envs `idx` (bool mask); credit dealt/taken killing blows."""
        kb = idx & killing_blow
        self.hits_dealt[kb] += 1
        self.combo[kb] += 1
        self.hits_taken[idx & killed] += 1
        open_combo = idx & (self.combo > 0)
        self.combos.extend(self.combo[open_combo].tolist())
        self.combo[idx] = 0

    def summary(self, num_duels: int) -> SideStats:
        return SideStats(
            hits_landed=float(self.hits_dealt.sum()) / max(num_duels, 1),
            hits_taken=float(self.hits_taken.sum()) / max(num_duels, 1),
            avg_combo=(float(np.mean(self.combos)) if self.combos else 0.0),
            mean_aim_err_deg=self.aim_sum / max(self.aim_ticks, 1),
        )


def run_match(
    contestant_a: Any,
    contestant_b: Any,
    num_duels: int = 100,
    seed: int = 0,
    env_cls: Optional[type] = None,
    max_steps: Optional[int] = None,
    swap_sides: bool = True,
) -> MatchResult:
    """Run `num_duels` duels of a vs b; returns a MatchResult.

    max_steps caps the number of env steps; envs still pending at the cap
    are recorded as draws.  Defaults to the env's MAX_TICKS (+ margin).
    """
    a = as_contestant(contestant_a)
    b = as_contestant(contestant_b)
    cls = env_cls or DuelVecEnv
    env = cls(num_duels, seed=seed)
    n = num_duels
    if max_steps is None:
        max_steps = int(getattr(env, "MAX_TICKS", 1200)) + 8

    idx = np.arange(n)
    side_a = (idx % 2).astype(np.int64) if swap_sides else np.zeros(n, np.int64)
    side_b = 1 - side_a

    obs = env.reset()
    a.begin(n)
    b.begin(n)

    pending = np.ones(n, dtype=bool)
    result = MatchResult(name_a=a.name, name_b=b.name, num_duels=n)
    ep_len = np.zeros(n, dtype=np.int64)
    ep_len_final = np.zeros(n, dtype=np.int64)
    track_a = _SideTracker(n)
    track_b = _SideTracker(n)

    steps = 0
    while pending.any() and steps < max_steps:
        obs_a = obs[idx, side_a]
        obs_b = obs[idx, side_b]
        track_a.tick(pending, obs_a)
        track_b.tick(pending, obs_b)

        actions = np.zeros((n, 2, NUM_ACTION_HEADS), dtype=np.int64)
        actions[idx, side_a] = a.act(obs_a)
        actions[idx, side_b] = b.act(obs_b)

        new_obs, rew, done, info = env.step(actions)
        steps += 1
        ep_len[pending] += 1

        new_a = new_obs[idx, side_a]
        new_b = new_obs[idx, side_b]
        live = pending & ~done  # obs after done belongs to a fresh episode
        dealt_a = live & (obs_a[:, _ENEMY_HP] - new_a[:, _ENEMY_HP] > _HP_EPS)
        taken_a = live & (obs_a[:, _SELF_HP] - new_a[:, _SELF_HP] > _HP_EPS)
        dealt_b = live & (obs_b[:, _ENEMY_HP] - new_b[:, _ENEMY_HP] > _HP_EPS)
        taken_b = live & (obs_b[:, _SELF_HP] - new_b[:, _SELF_HP] > _HP_EPS)
        track_a.hits(dealt_a, taken_a)
        track_b.hits(dealt_b, taken_b)

        finish = pending & done
        if finish.any():
            win = np.asarray(info["win"])
            win_a = finish & (win[idx, side_a] > 0.5)
            win_b = finish & (win[idx, side_b] > 0.5)
            result.wins_a += int(win_a.sum())
            result.wins_b += int(win_b.sum())
            result.draws += int((finish & ~win_a & ~win_b).sum())
            track_a.finish(finish, win_a, killed=win_b)
            track_b.finish(finish, win_b, killed=win_a)
            ep_len_final[finish] = ep_len[finish]
            pending &= ~done

        a.on_done(done)
        b.on_done(done)
        obs = new_obs

    if pending.any():  # hit the step cap: count as draws
        result.draws += int(pending.sum())
        none = np.zeros(n, dtype=bool)
        track_a.finish(pending, none, killed=none)
        track_b.finish(pending, none, killed=none)
        ep_len_final[pending] = ep_len[pending]

    result.mean_episode_len = float(ep_len_final.mean()) if n else 0.0
    result.stats_a = track_a.summary(n)
    result.stats_b = track_b.summary(n)
    return result
