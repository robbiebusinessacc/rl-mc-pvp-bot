"""Self-play opponent league with Elo bookkeeping.

Side 0 of every env is the learning agent. Side 1 is driven per env by an
opponent chosen from:
  - the current policy itself ("mirror" self-play, probability ``p_self``),
  - frozen past checkpoints of the learner, sampled with probability
    proportional to a softmax over negative Elo distance to the learner
    (closer Elo -> more likely),
  - scripted baselines from ``pvpbot.eval`` when that module exists
    (imported lazily and optionally; absent on this branch -> degrade to
    checkpoints only).

Elo is updated from ``info["win"]`` on episode ends (draws count 0.5).
Mirror self-play games do not move Elo (the learner would play itself).

Gating: ``gate()`` is called every N updates by the training loop. A new
checkpoint enters the pool if the pool has no checkpoints yet, or if the
learner's recent win rate against the pool reaches ``gate_winrate``.
When the pool is full the lowest-Elo checkpoint is evicted.
"""
import copy
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from pvpbot.models import PolicyNet
from pvpbot.spec import ACTION_HEAD_SIZES, NUM_ACTION_HEADS, OBS_DIM
from pvpbot.train.ppo import RunningNorm, sample_multi_discrete

SELF_PLAY = -1  # sentinel opponent id: mirror the current policy


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------
def elo_expected(ra: float, rb: float) -> float:
    """Expected score of A against B."""
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def elo_update(ra: float, rb: float, score: float, k: float = 16.0):
    """Update both ratings after A scores ``score`` (1 win, 0.5 draw, 0 loss)."""
    ea = elo_expected(ra, rb)
    return ra + k * (score - ea), rb + k * ((1.0 - score) - (1.0 - ea))


# ---------------------------------------------------------------------------
# Scripted baselines (optional, from pvpbot.eval)
# ---------------------------------------------------------------------------
def discover_scripted() -> List[Any]:
    """Best-effort lazy import of scripted opponents from pvpbot.eval.

    Accepts any object exposing ``act(obs: np.ndarray (B, OBS_DIM)) ->
    np.ndarray (B, NUM_ACTION_HEADS)``. Returns [] when pvpbot.eval is
    missing or exposes nothing usable (the normal case on this branch).
    """
    import importlib

    found = []  # type: List[Any]
    for modname in ("pvpbot.eval", "pvpbot.eval.opponents", "pvpbot.eval.scripted"):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for attr in (
            "get_scripted_opponents",
            "scripted_opponents",
            "SCRIPTED_OPPONENTS",
            "BASELINES",
        ):
            obj = getattr(mod, attr, None)
            if obj is None:
                continue
            try:
                items = list(obj()) if callable(obj) else list(obj)
            except Exception:
                continue
            for it in items:
                if hasattr(it, "act"):
                    found.append(it)
            if found:
                return found
    return found


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------
class PoolEntry:
    """One frozen opponent: a past checkpoint or a scripted baseline."""

    def __init__(
        self,
        name: str,
        kind: str,  # "ckpt" | "scripted"
        elo: float,
        net: Optional[PolicyNet] = None,
        norm: Optional[RunningNorm] = None,
        agent: Optional[Any] = None,
    ):
        self.name = name
        self.kind = kind
        self.elo = float(elo)
        self.net = net
        self.norm = norm
        self.agent = agent
        self.games = 0
        self.wins = 0.0  # score accumulated by this entry (vs the learner)
        self.broken = False  # scripted entry whose act() raised


class League:
    def __init__(
        self,
        num_envs: int,
        obs_dim: int = OBS_DIM,
        device: str = "cpu",
        p_self: float = 0.5,
        elo_k: float = 16.0,
        elo_scale: float = 200.0,
        max_pool: int = 20,
        gate_winrate: float = 0.55,
        min_gate_games: int = 32,
        initial_elo: float = 1000.0,
        recent_window: int = 512,
        include_scripted: bool = True,
        seed: int = 0,
        max_active: int = 3,
        pin_name: Optional[str] = None,
        pin_frac: float = 0.0,
    ):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.device = torch.device(device)
        self.p_self = p_self
        self.elo_k = elo_k
        self.elo_scale = elo_scale
        self.max_pool = max_pool
        self.gate_winrate = gate_winrate
        self.min_gate_games = min_gate_games
        self.learner_elo = float(initial_elo)
        self.rng = np.random.default_rng(seed)
        # Every distinct pool member acting in a rollout costs one extra
        # policy forward per tick (a separate small-batch device round-trip,
        # painful on MPS: throughput decayed 356k -> 20k steps/s as the pool
        # filled). Cap how many pool members are simultaneously active;
        # the active subset rotates on every gate() so diversity comes from
        # time, not from concurrency.
        self.max_active = max_active
        self._active = []  # type: List[int]
        self.pool = []  # type: List[PoolEntry]
        self.assign = np.full(num_envs, SELF_PLAY, dtype=np.int64)
        self.h_opp = torch.zeros(num_envs, PolicyNet.CORE, device=self.device)
        # recent learner scores vs pool opponents (1 win / 0.5 draw / 0 loss)
        self.recent = deque(maxlen=recent_window)  # type: deque
        self.games_vs_pool = 0
        if include_scripted:
            for agent in discover_scripted():
                name = getattr(agent, "name", type(agent).__name__)
                elo = float(getattr(agent, "elo", initial_elo))
                self.pool.append(
                    PoolEntry(name="scripted_" + str(name), kind="scripted", elo=elo, agent=agent)
                )
        # Optionally PIN a fraction of envs to one named opponent regardless
        # of Elo proximity. The learner's Elo climbs far above the scripted
        # tiers, so the Elo-softmax sampler almost never draws P4-Hacker --
        # yet beating P4 live is the goal. Pinning forces sustained P4
        # exposure so the policy learns its specific counter.
        self._pin_frac = float(pin_frac)
        self._pin_name = str(pin_name) if pin_name else ""
        self._pin_idx = None
        if self._pin_name and pin_frac > 0.0:
            for i, e in enumerate(self.pool):
                if self._pin_name.lower() in e.name.lower():
                    self._pin_idx = i
                    break
        self.resample(np.arange(num_envs))

    # -- sampling -----------------------------------------------------------
    def _pool_probs(self) -> np.ndarray:
        """Softmax over negative Elo distance to the learner (broken excluded)."""
        elos = np.array([e.elo for e in self.pool], dtype=np.float64)
        logits = -np.abs(elos - self.learner_elo) / self.elo_scale
        for i, e in enumerate(self.pool):
            if e.broken:
                logits[i] = -np.inf
        if not np.isfinite(logits).any():
            return np.zeros(len(self.pool))
        logits -= logits.max()
        p = np.exp(logits)
        return p / p.sum()

    def _refresh_active(self) -> None:
        """Re-draw the small set of pool members allowed to act concurrently."""
        usable = [i for i, e in enumerate(self.pool) if not e.broken]
        if len(usable) <= self.max_active:
            self._active = usable
            return
        probs = self._pool_probs()
        p = probs[usable]
        s = p.sum()
        p = p / s if s > 0 else np.full(len(usable), 1.0 / len(usable))
        self._active = [int(i) for i in self.rng.choice(
            usable, size=self.max_active, replace=False, p=p)]

    def resample(self, env_ids: np.ndarray) -> None:
        """Assign fresh opponents to the given envs and reset their hidden state."""
        env_ids = np.asarray(env_ids)
        if env_ids.size == 0:
            return
        active = [i for i in self._active if i < len(self.pool)
                  and not self.pool[i].broken]
        if not active:
            self._refresh_active()
            active = self._active
        if active:
            probs = self._pool_probs()
            p = probs[active]
            s = p.sum()
            p = p / s if s > 0 else np.full(len(active), 1.0 / len(active))
            use_self = self.rng.random(env_ids.size) < self.p_self
            picks = np.asarray(active)[
                self.rng.choice(len(active), size=env_ids.size, p=p)]
            self.assign[env_ids] = np.where(use_self, SELF_PLAY, picks)
        else:
            self.assign[env_ids] = SELF_PLAY
        # override: pin a fraction of these envs to the pinned opponent
        # (bypasses both self-play and Elo sampling). Not gated on the
        # active set -- the pin must always be available to act.
        if self._pin_idx is not None and self._pin_frac > 0.0:
            pin_mask = self.rng.random(env_ids.size) < self._pin_frac
            self.assign[env_ids[pin_mask]] = self._pin_idx
            if self._pin_idx not in self._active:
                self._active = self._active + [self._pin_idx]
        self.h_opp[torch.as_tensor(env_ids, device=self.device)] = 0.0

    # -- acting -------------------------------------------------------------
    @torch.no_grad()
    def opponent_actions(
        self, raw_obs: np.ndarray, policy: PolicyNet, obs_norm: RunningNorm,
        scripted_obs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Batched side-1 actions for all envs, grouped per opponent.

        raw_obs: (num_envs, obs_dim) un-normalized side-1 observations (the
            camera-gated learner view; used for self-play / checkpoint
            opponents which are also camera-limited).
        scripted_obs: optional ungated OMNISCIENT side-1 obs used for
            scripted opponents -- they model real mineflayer bots with
            perfect server-side awareness. Falls back to raw_obs if None.
        """
        if scripted_obs is None:
            scripted_obs = raw_obs
        n = raw_obs.shape[0]
        out = np.zeros((n, NUM_ACTION_HEADS), dtype=np.int64)

        def run_policy(net: PolicyNet, norm: RunningNorm, env_idx: np.ndarray) -> None:
            # Pad the batch to a power-of-two bucket: league slice sizes vary
            # every tick, and torch-MPS compiles + caches a kernel graph per
            # NEW shape -- unbounded shapes made throughput decay from ~350k
            # to ~20k steps/s within minutes. Bucketing bounds the shape set.
            k = env_idx.size
            pad = 1 << (k - 1).bit_length() if k > 1 else 1
            pad = min(pad, n)
            obs_np = np.zeros((pad, raw_obs.shape[1]), dtype=np.float32)
            obs_np[:k] = norm.normalize(raw_obs[env_idx])
            obs_t = torch.from_numpy(obs_np).to(self.device)
            idx_t = torch.as_tensor(env_idx, device=self.device)
            h = torch.zeros(pad, self.h_opp.shape[1], device=self.device)
            h[:k] = self.h_opp[idx_t]
            logits, _, h_new = net(obs_t, h)
            acts, _ = sample_multi_discrete(logits)
            self.h_opp[idx_t] = h_new[:k]
            out[env_idx] = acts[:k].cpu().numpy()

        mirror = np.nonzero(self.assign == SELF_PLAY)[0]
        if mirror.size:
            run_policy(policy, obs_norm, mirror)
        for pid in np.unique(self.assign[self.assign >= 0]):
            entry = self.pool[int(pid)]
            env_idx = np.nonzero(self.assign == pid)[0]
            if entry.kind == "ckpt":
                run_policy(entry.net, entry.norm, env_idx)
            else:  # scripted -- omniscient obs (models a mineflayer bot)
                try:
                    acts = np.asarray(entry.agent.act(scripted_obs[env_idx]), dtype=np.int64)
                    acts = acts.reshape(env_idx.size, NUM_ACTION_HEADS)
                    for h_i, size in enumerate(ACTION_HEAD_SIZES):
                        np.clip(acts[:, h_i], 0, size - 1, out=acts[:, h_i])
                    out[env_idx] = acts
                except Exception:
                    entry.broken = True  # drop from future sampling
                    self.assign[env_idx] = SELF_PLAY
                    run_policy(policy, obs_norm, env_idx)
        return out

    # -- outcomes -----------------------------------------------------------
    def on_done(self, done: np.ndarray, win: np.ndarray) -> None:
        """Update Elo from finished episodes, then resample their opponents.

        done: (num_envs,) bool; win: (num_envs, 2) from info["win"].
        """
        idxs = np.nonzero(done)[0]
        for env in idxs:
            pid = int(self.assign[env])
            if pid < 0:
                continue  # mirror self-play: no Elo signal
            entry = self.pool[pid]
            if entry.broken:
                continue
            if win[env, 0] > 0.5:
                score = 1.0
            elif win[env, 1] > 0.5:
                score = 0.0
            else:
                score = 0.5
            self.learner_elo, entry.elo = elo_update(
                self.learner_elo, entry.elo, score, self.elo_k
            )
            entry.games += 1
            entry.wins += 1.0 - score
            self.recent.append(score)
            self.games_vs_pool += 1
        self.resample(idxs)

    def winrate_vs_pool(self) -> float:
        """Mean recent learner score vs pool opponents (nan when no games)."""
        if not self.recent:
            return float("nan")
        return float(np.mean(self.recent))

    # -- gating -------------------------------------------------------------
    def num_ckpts(self) -> int:
        return sum(1 for e in self.pool if e.kind == "ckpt")

    def gate(self, policy: PolicyNet, obs_norm: RunningNorm, step: int) -> bool:
        """Snapshot the current policy into the pool if it passes the gate.

        Rule: always accept while the pool has no checkpoints; afterwards
        accept only when the recent win rate vs the pool is >= gate_winrate
        (with at least min_gate_games recent games; too few games -> accept,
        so a starved pool still refreshes).
        """
        if self.num_ckpts() > 0 and len(self.recent) >= self.min_gate_games:
            if self.winrate_vs_pool() < self.gate_winrate:
                return False
        had_usable = any(not e.broken for e in self.pool)
        net = PolicyNet(obs_dim=self.obs_dim)
        net.load_state_dict(copy.deepcopy(policy.state_dict()))
        net.to(self.device)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.pool.append(
            PoolEntry(
                name="ckpt_%d" % int(step),
                kind="ckpt",
                elo=self.learner_elo,
                net=net,
                norm=obs_norm.copy(),
            )
        )
        while self.num_ckpts() > self.max_pool:
            ckpts = [(i, e) for i, e in enumerate(self.pool) if e.kind == "ckpt"]
            evict = min(ckpts[:-1], key=lambda ie: ie[1].elo)[0]  # keep newest
            self._evict(evict)
        self._refresh_active()
        if not had_usable:
            # cold start: the pool just became non-empty; give the currently
            # running (all-mirror) episodes real opponents right away
            self.resample(np.arange(self.num_envs))
        return True

    def _evict(self, pool_idx: int) -> None:
        del self.pool[pool_idx]
        hit = self.assign == pool_idx
        self.assign[self.assign > pool_idx] -= 1
        self._active = [i - 1 if i > pool_idx else i
                        for i in self._active if i != pool_idx]
        if hit.any():
            self.resample(np.nonzero(hit)[0])

    # -- (de)serialization --------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        entries = []
        for e in self.pool:
            rec = {
                "name": e.name,
                "kind": e.kind,
                "elo": e.elo,
                "games": e.games,
                "wins": e.wins,
                "broken": e.broken,
            }  # type: Dict[str, Any]
            if e.kind == "ckpt":
                rec["model"] = e.net.state_dict()
                rec["norm"] = e.norm.state_dict()
            entries.append(rec)
        return {
            "learner_elo": self.learner_elo,
            "entries": entries,
            "recent": list(self.recent),
            "games_vs_pool": self.games_vs_pool,
        }

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.learner_elo = float(sd["learner_elo"])
        self.recent = deque(sd.get("recent", []), maxlen=self.recent.maxlen)
        self.games_vs_pool = int(sd.get("games_vs_pool", 0))
        scripted_live = {e.name: e for e in self.pool if e.kind == "scripted"}
        self.pool = []
        for rec in sd["entries"]:
            if rec["kind"] == "ckpt":
                net = PolicyNet(obs_dim=self.obs_dim)
                net.load_state_dict(rec["model"])
                net.to(self.device)
                net.eval()
                for p in net.parameters():
                    p.requires_grad_(False)
                norm = RunningNorm(self.obs_dim)
                norm.load_state_dict(rec["norm"])
                e = PoolEntry(rec["name"], "ckpt", rec["elo"], net=net, norm=norm)
            else:
                live = scripted_live.get(rec["name"])
                if live is None:
                    continue  # baseline no longer available: drop it
                e = live
                e.elo = float(rec["elo"])
            e.games = int(rec["games"])
            e.wins = float(rec["wins"])
            e.broken = bool(rec.get("broken", False))
            self.pool.append(e)
        # A save from before a baseline existed must not drop it forever:
        # re-append live scripted entries the save didn't know about, then
        # re-bind the pin by NAME (saved pool order differs from __init__'s).
        have = {e.name for e in self.pool}
        for name in sorted(scripted_live):
            if name not in have:
                self.pool.append(scripted_live[name])
        if self._pin_name:
            self._pin_idx = None
            for i, e in enumerate(self.pool):
                if self._pin_name.lower() in e.name.lower():
                    self._pin_idx = i
                    break
            if self._pin_idx is None:
                print("WARNING: pinned opponent %r missing after resume"
                      % self._pin_name)
        n_scripted = sum(1 for e in self.pool if e.kind == "scripted")
        print("league resumed: %d ckpt + %d scripted%s" % (
            len(self.pool) - n_scripted, n_scripted,
            (", pin=%s" % self.pool[self._pin_idx].name)
            if self._pin_idx is not None else ""))
        self.assign[:] = SELF_PLAY
        self.h_opp.zero_()
        self._refresh_active()
        self.resample(np.arange(self.num_envs))
