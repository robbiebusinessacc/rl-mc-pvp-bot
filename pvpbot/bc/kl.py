"""KL-to-BC-prior auxiliary loss for the PPO trainer.

Adds ``kl_coef * mean_B[ sum_heads KL(pi_RL(.|s) || pi_BC(.|s)) ]`` to the
PPO objective, anchoring the RL policy to human-like behavior. The BC prior
is a frozen ``PolicyNet`` (pvpbot/models.py); this module depends only on
``pvpbot.models`` and ``pvpbot.spec`` -- never on ``pvpbot.train`` -- so the
trainer branch can import it freely.

Usage from the PPO trainer
--------------------------
::

    from pvpbot.bc.kl import KLPriorLoss

    prior = KLPriorLoss.from_checkpoint("bc_policy.pt", device=device)
    bc_state = prior.initial_state(num_envs, device=device)   # (N, CORE)

    # --- during rollout/update, wherever you already have this step's
    #     per-head RL logits for obs (B, OBS_DIM):
    kl, bc_state = prior(rl_logits, obs, bc_state)
    loss = ppo_loss + kl_coef * kl

    # --- on env resets, zero the prior's hidden state rows like your own:
    bc_state = KLPriorLoss.reset_state(bc_state, done)

Contract details:

* ``rl_logits`` is the list of per-head tensors ``[(B, n_0), ..., (B, n_6)]``
  in ``ACTION_HEADS`` order -- the same logits the PPO loss uses. Gradients
  flow through them and only them: the BC net runs under ``torch.no_grad``
  and its logits are constants in the graph.
* ``obs`` must be the same (B, OBS_DIM) float32 batch the RL policy saw.
* ``bc_state`` is the frozen prior's own GRU state; thread it through time
  exactly like the RL policy's state. The returned state is detached. If you
  evaluate KL on shuffled/flattened PPO minibatches instead of in rollout
  order, either store per-step bc states during the rollout (preferred, same
  as you store RL states) or pass zeros and accept a slightly stale prior --
  the prior is a regularizer, not a critic, so this is tolerable.
* Direction: KL(RL || BC) is mode-seeking -- it lets the RL policy commit to
  one human-plausible action rather than smearing mass over everything the
  prior tolerates, and it is finite even where the prior is (numerically)
  near-zero.

``KLPriorLoss.from_policy`` is a convenience wrapper for callers that have
obs + both hidden states but no precomputed logits (e.g. an evaluation
probe); it runs the RL policy forward itself and returns both new states.
"""
import copy
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from pvpbot.models import PolicyNet
from pvpbot.spec import ACTION_HEAD_SIZES, NUM_ACTION_HEADS


def kl_categorical(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """Elementwise KL(P || Q) between categorical distributions from logits.

    logits_p, logits_q: (..., n). Returns (...,) -- summed over the class dim.
    Computed in log-space for numerical stability.
    """
    logp = F.log_softmax(logits_p, dim=-1)
    logq = F.log_softmax(logits_q, dim=-1)
    return (logp.exp() * (logp - logq)).sum(dim=-1)


class KLPriorLoss:
    """mean over batch of sum over heads of KL(pi_RL || pi_BC), pluggable.

    Holds a private frozen deep-copy of the BC policy (the caller's network
    is left untouched): eval mode, ``requires_grad=False`` on every
    parameter, all forward passes under ``torch.no_grad``.
    """

    def __init__(self, bc_policy: PolicyNet):
        self.policy = copy.deepcopy(bc_policy)
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)

    # -- construction / device management ---------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "KLPriorLoss":
        """Load a BC checkpoint in the spec.py format and wrap it."""
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
        if "model" not in ckpt:
            raise ValueError("%s: not a spec.py checkpoint (missing 'model')" % path)
        net = PolicyNet()
        net.load_state_dict(ckpt["model"])
        if device is not None:
            net = net.to(device)
        prior = cls.__new__(cls)  # skip the deepcopy; net is already private
        prior.policy = net.eval()
        for p in prior.policy.parameters():
            p.requires_grad_(False)
        return prior

    def to(self, device: Union[str, torch.device]) -> "KLPriorLoss":
        self.policy = self.policy.to(device)
        return self

    # -- hidden state helpers ----------------------------------------------

    def initial_state(
        self, batch: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        return self.policy.initial_state(batch, device=device)

    @staticmethod
    def reset_state(state: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """Zero the state rows where ``done`` is truthy. done: (B,) bool/float."""
        keep = (~done.bool()).to(state.dtype).unsqueeze(-1)
        return state * keep

    # -- the loss ----------------------------------------------------------

    def __call__(
        self,
        rl_logits: List[torch.Tensor],
        obs: torch.Tensor,
        bc_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(kl scalar, new bc_state). Gradients flow into rl_logits only."""
        if len(rl_logits) != NUM_ACTION_HEADS:
            raise ValueError(
                "expected %d per-head logit tensors, got %d"
                % (NUM_ACTION_HEADS, len(rl_logits))
            )
        for k, (lg, n) in enumerate(zip(rl_logits, ACTION_HEAD_SIZES)):
            if lg.shape[-1] != n:
                raise ValueError(
                    "head %d logits have %d classes, expected %d"
                    % (k, lg.shape[-1], n)
                )
        with torch.no_grad():
            bc_logits, _, new_state = self.policy(obs, bc_state)
        kl = torch.zeros((), device=obs.device)
        for lg_rl, lg_bc in zip(rl_logits, bc_logits):
            kl = kl + kl_categorical(lg_rl, lg_bc).mean()
        return kl, new_state.detach()

    def from_policy(
        self,
        rl_policy: PolicyNet,
        obs: torch.Tensor,
        rl_state: torch.Tensor,
        bc_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convenience: runs the RL policy itself.

        Returns (kl, new_rl_state, new_bc_state); gradients flow through the
        RL policy's forward pass only.
        """
        rl_logits, _, new_rl_state = rl_policy(obs, rl_state)
        kl, new_bc_state = self(rl_logits, obs, bc_state)
        return kl, new_rl_state, new_bc_state
