"""PPO for the recurrent, multi-discrete PolicyNet.

Design notes
------------
- Multi-discrete policy: one categorical per action head; the joint log-prob
  is the sum of per-head log-probs. Entropy is computed per head (summed for
  the bonus, logged individually).
- GAE(lambda) advantages with bootstrap masking at episode ends. The env
  auto-resets, so ``done[t] == 1`` cuts credit between step t and t+1.
- Recurrence / truncated BPTT: the rollout buffer stores the GRU hidden state
  at every chunk boundary (``chunk_len`` steps, default 16). The update
  replays each (chunk, env) pair as an independent sequence, starting from the
  stored (detached) initial state and re-zeroing the hidden state at done
  steps -- exactly mirroring what the collector did.
- Observation normalization: running mean/var (parallel Welford), clip +/-10.
  The buffer stores the *normalized* obs that were fed to the policy during
  collection, so the replay sees identical inputs.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from pvpbot.models import PolicyNet
from pvpbot.spec import ACTION_HEADS, NUM_ACTION_HEADS, OBS_DIM


# ---------------------------------------------------------------------------
# Observation normalization
# ---------------------------------------------------------------------------
class RunningNorm:
    """Running mean/var normalizer using the parallel (Chan) Welford update."""

    def __init__(self, dim: int, clip: float = 10.0, eps: float = 1e-8):
        self.dim = dim
        self.clip = clip
        self.eps = eps
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.dim)
        n = x.shape[0]
        if n == 0:
            return
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        delta = batch_mean - self.mean
        tot = self.count + n
        m2 = self.var * self.count + batch_var * n + np.square(delta) * (self.count * n / tot)
        self.mean = self.mean + delta * (n / tot)
        self.var = m2 / tot
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x) - self.mean) / np.sqrt(self.var + self.eps)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def copy(self) -> "RunningNorm":
        out = RunningNorm(self.dim, self.clip, self.eps)
        out.mean = self.mean.copy()
        out.var = self.var.copy()
        out.count = self.count
        return out

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "mean": torch.from_numpy(self.mean.copy()),
            "var": torch.from_numpy(self.var.copy()),
            "count": torch.tensor(float(self.count), dtype=torch.float64),
        }

    def load_state_dict(self, sd: Dict[str, torch.Tensor]) -> None:
        self.mean = sd["mean"].numpy().astype(np.float64).copy()
        self.var = sd["var"].numpy().astype(np.float64).copy()
        self.count = float(sd["count"])


# ---------------------------------------------------------------------------
# Multi-discrete distribution math
# ---------------------------------------------------------------------------
def heads_log_prob_entropy(
    logits_list: List[torch.Tensor], actions: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Joint log-prob (sum over heads) and per-head entropy.

    logits_list[i]: (..., n_i); actions: (..., NUM_ACTION_HEADS) int64.
    Returns (logp (...,), entropy (..., NUM_ACTION_HEADS)).
    """
    logps = []
    ents = []
    for i, logits in enumerate(logits_list):
        logp_all = torch.log_softmax(logits, dim=-1)
        a = actions[..., i].unsqueeze(-1)
        logps.append(torch.gather(logp_all, -1, a).squeeze(-1))
        ents.append(-(logp_all.exp() * logp_all).sum(dim=-1))
    return torch.stack(logps, dim=-1).sum(dim=-1), torch.stack(ents, dim=-1)


def sample_multi_discrete(
    logits_list: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample one action per head. Returns (actions (B, H) int64, logp (B,))."""
    acts = []
    logps = []
    for logits in logits_list:
        logp_all = torch.log_softmax(logits, dim=-1)
        a = torch.multinomial(logp_all.exp(), 1)
        logps.append(torch.gather(logp_all, -1, a).squeeze(-1))
        acts.append(a.squeeze(-1))
    return torch.stack(acts, dim=-1), torch.stack(logps, dim=-1).sum(dim=-1)


@torch.no_grad()
def policy_step(
    policy: PolicyNet, obs: torch.Tensor, h: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One batched inference step: (actions, logp, value, new_hidden)."""
    logits, value, h_new = policy(obs, h)
    actions, logp = sample_multi_discrete(logits)
    return actions, logp, value, h_new


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------
def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """GAE(lambda) on time-major arrays.

    rewards/values/dones: (T, N); dones[t] == 1 means the episode ended at
    step t (no bootstrap from t+1). last_values: (N,) value estimate for the
    obs after the final step. Returns (advantages, returns), both (T, N).
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    T = rewards.shape[0]
    adv = np.zeros_like(rewards)
    lastgae = np.zeros_like(np.asarray(last_values, dtype=np.float32))
    for t in reversed(range(T)):
        nonterm = 1.0 - dones[t]
        next_v = last_values if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        lastgae = delta + gamma * lam * nonterm * lastgae
        adv[t] = lastgae
    return adv, adv + values


# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------
class RolloutBuffer:
    """Time-major rollout storage with hidden states kept at chunk starts."""

    def __init__(
        self,
        T: int,
        N: int,
        obs_dim: int = OBS_DIM,
        chunk_len: int = 16,
        core_dim: int = PolicyNet.CORE,
    ):
        if T % chunk_len != 0:
            raise ValueError("rollout_len must be a multiple of chunk_len")
        self.T = T
        self.N = N
        self.obs_dim = obs_dim
        self.chunk_len = chunk_len
        self.num_chunks = T // chunk_len
        self.obs = np.zeros((T, N, obs_dim), dtype=np.float32)
        self.actions = np.zeros((T, N, NUM_ACTION_HEADS), dtype=np.int64)
        self.logp = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros((T, N), dtype=np.float32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=np.float32)
        self.h_init = np.zeros((self.num_chunks, N, core_dim), dtype=np.float32)
        self.advantages = None  # type: Optional[np.ndarray]
        self.returns = None  # type: Optional[np.ndarray]

    def add(
        self,
        t: int,
        obs: np.ndarray,
        h: torch.Tensor,
        actions: np.ndarray,
        logp: np.ndarray,
        value: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
    ) -> None:
        self.obs[t] = obs
        if t % self.chunk_len == 0:
            self.h_init[t // self.chunk_len] = h.detach().cpu().numpy()
        self.actions[t] = actions
        self.logp[t] = logp
        self.values[t] = value
        self.rewards[t] = reward
        self.dones[t] = done

    def compute_returns(self, last_values: np.ndarray, gamma: float, lam: float) -> None:
        self.advantages, self.returns = compute_gae(
            self.rewards, self.values, self.dones, last_values, gamma, lam
        )


# ---------------------------------------------------------------------------
# PPO trainer
# ---------------------------------------------------------------------------
@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    epochs: int = 2
    num_minibatches: int = 8
    ent_coef: float = 0.01
    # per-head entropy coefs (len NUM_ACTION_HEADS); None = ent_coef on all.
    # Heads whose actions the env overrides (camera under cam_assist) or
    # latches earn entropy bonus with no dynamics consequence, which bleeds
    # max-entropy pressure into the consequential decisions -- weight them
    # separately.
    ent_coefs: Optional[Tuple[float, ...]] = None
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    chunk_len: int = 16
    rollout_len: int = 128
    norm_adv: bool = True
    clip_vloss: bool = True
    adam_eps: float = 1e-5


class PPOTrainer:
    """Clipped-objective PPO with truncated BPTT over stored chunks."""

    def __init__(
        self,
        policy: PolicyNet,
        cfg: Optional[PPOConfig] = None,
        device: str = "cpu",
        obs_dim: int = OBS_DIM,
    ):
        self.policy = policy
        self.cfg = cfg if cfg is not None else PPOConfig()
        self.device = torch.device(device)
        self.policy.to(self.device)
        self.opt = torch.optim.Adam(
            policy.parameters(), lr=self.cfg.lr, eps=self.cfg.adam_eps
        )
        self.obs_norm = RunningNorm(obs_dim)

    # -- data prep ----------------------------------------------------------
    def _prep(self, buf: RolloutBuffer) -> Dict[str, torch.Tensor]:
        """Reshape buffer into (num_chunks, chunk_len, N, ...) tensors."""
        if buf.advantages is None:
            raise RuntimeError("call buf.compute_returns() before update()")
        C, L, N = buf.num_chunks, buf.chunk_len, buf.N
        dev = self.device

        def t(x, dtype):
            return torch.as_tensor(x, dtype=dtype, device=dev)

        adv = t(buf.advantages, torch.float32).view(C, L, N)
        if self.cfg.norm_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return {
            "obs": t(buf.obs, torch.float32).view(C, L, N, buf.obs_dim),
            "actions": t(buf.actions, torch.int64).view(C, L, N, NUM_ACTION_HEADS),
            "old_logp": t(buf.logp, torch.float32).view(C, L, N),
            "old_values": t(buf.values, torch.float32).view(C, L, N),
            "dones": t(buf.dones, torch.float32).view(C, L, N),
            "adv": adv,
            "returns": t(buf.returns, torch.float32).view(C, L, N),
            "h_init": t(buf.h_init, torch.float32),  # (C, N, CORE)
        }

    def _sequence_forward(
        self, data: Dict[str, torch.Tensor], c_idx: torch.Tensor, n_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Replay chunks (c_idx[i], n_idx[i]) with truncated BPTT.

        Returns per-sequence-step tensors shaped (B, L[, H]).
        """
        L = data["obs"].shape[1]
        # advanced indexing on dims 0 and 2 puts (B, L) in front: (B, L, ...)
        obs = data["obs"][c_idx, :, n_idx]
        actions = data["actions"][c_idx, :, n_idx]
        dones = data["dones"][c_idx, :, n_idx]
        h = data["h_init"][c_idx, n_idx].clone()
        lp_steps, ent_steps, val_steps = [], [], []
        for t in range(L):
            logits, v, h = self.policy(obs[:, t], h)
            lp, ent = heads_log_prob_entropy(logits, actions[:, t])
            lp_steps.append(lp)
            ent_steps.append(ent)
            val_steps.append(v)
            # reset hidden state where the episode ended at step t
            h = h * (1.0 - dones[:, t]).unsqueeze(-1)
        return {
            "logp": torch.stack(lp_steps, dim=1),  # (B, L)
            "entropy": torch.stack(ent_steps, dim=1),  # (B, L, H)
            "values": torch.stack(val_steps, dim=1),  # (B, L)
        }

    def _losses(
        self, data: Dict[str, torch.Tensor], c_idx: torch.Tensor, n_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        out = self._sequence_forward(data, c_idx, n_idx)
        old_logp = data["old_logp"][c_idx, :, n_idx]
        old_val = data["old_values"][c_idx, :, n_idx]
        adv = data["adv"][c_idx, :, n_idx]
        ret = data["returns"][c_idx, :, n_idx]

        log_ratio = out["logp"] - old_logp
        ratio = log_ratio.exp()
        pg1 = -adv * ratio
        pg2 = -adv * ratio.clamp(1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
        pg_loss = torch.max(pg1, pg2).mean()

        values = out["values"]
        if cfg.clip_vloss:
            v_clip = old_val + (values - old_val).clamp(-cfg.clip_range, cfg.clip_range)
            v_loss = 0.5 * torch.max((values - ret) ** 2, (v_clip - ret) ** 2).mean()
        else:
            v_loss = 0.5 * ((values - ret) ** 2).mean()

        ent_per_head = out["entropy"].mean(dim=(0, 1))  # (H,)
        ent_total = out["entropy"].sum(dim=-1).mean()
        if cfg.ent_coefs is not None:
            ent_bonus = (ent_per_head * ent_per_head.new_tensor(cfg.ent_coefs)).sum()
        else:
            ent_bonus = cfg.ent_coef * ent_total
        loss = pg_loss + cfg.vf_coef * v_loss - ent_bonus
        with torch.no_grad():
            approx_kl = (ratio - 1.0 - log_ratio).mean()
            clip_frac = ((ratio - 1.0).abs() > cfg.clip_range).float().mean()
        return {
            "loss": loss,
            "pg_loss": pg_loss,
            "v_loss": v_loss,
            "entropy": ent_total,
            "ent_per_head": ent_per_head,
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
        }

    # -- public API ---------------------------------------------------------
    @torch.no_grad()
    def evaluate_loss(self, buf: RolloutBuffer, max_batch: int = 4096) -> float:
        """Composite PPO loss over the full buffer with current parameters."""
        data = self._prep(buf)
        S = buf.num_chunks * buf.N
        idx = torch.arange(S, device=self.device)
        total, n_batches = 0.0, 0
        for start in range(0, S, max_batch):
            part = idx[start : start + max_batch]
            c_idx = torch.div(part, buf.N, rounding_mode="floor")
            n_idx = part % buf.N
            losses = self._losses(data, c_idx, n_idx)
            total += float(losses["loss"])
            n_batches += 1
        return total / max(n_batches, 1)

    def update(self, buf: RolloutBuffer) -> Dict[str, float]:
        """Run cfg.epochs of clipped PPO over the buffer. Returns metrics."""
        cfg = self.cfg
        data = self._prep(buf)
        S = buf.num_chunks * buf.N
        mb_size = max(1, S // cfg.num_minibatches)

        agg = {
            "loss": 0.0, "pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0,
            "approx_kl": 0.0, "clip_frac": 0.0, "grad_norm": 0.0,
        }
        ent_heads = torch.zeros(NUM_ACTION_HEADS)
        n_mb = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(S, device=self.device)
            for start in range(0, S, mb_size):
                idx = perm[start : start + mb_size]
                c_idx = torch.div(idx, buf.N, rounding_mode="floor")
                n_idx = idx % buf.N
                losses = self._losses(data, c_idx, n_idx)
                self.opt.zero_grad(set_to_none=True)
                losses["loss"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), cfg.max_grad_norm
                )
                self.opt.step()

                agg["loss"] += float(losses["loss"].detach())
                agg["pg_loss"] += float(losses["pg_loss"].detach())
                agg["v_loss"] += float(losses["v_loss"].detach())
                agg["entropy"] += float(losses["entropy"].detach())
                agg["approx_kl"] += float(losses["approx_kl"])
                agg["clip_frac"] += float(losses["clip_frac"])
                agg["grad_norm"] += float(grad_norm)
                ent_heads += losses["ent_per_head"].detach().cpu()
                n_mb += 1

        metrics = {k: v / max(n_mb, 1) for k, v in agg.items()}
        for i, (name, _) in enumerate(ACTION_HEADS):
            metrics["entropy_" + name] = float(ent_heads[i]) / max(n_mb, 1)
        return metrics
