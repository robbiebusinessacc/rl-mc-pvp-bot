"""Behavior-clone PolicyNet from recorded episodes.

Per-head cross-entropy with the inverse-frequency class weights from
dataset.py (computed on the *training* split only), truncated BPTT over
fixed-length chunks with hidden-state burn-in, and an episode-level
validation split reporting per-head accuracy AND per-head macro-F1.

Why macro-F1: under the label skew of PvP play, accuracy is misleading -- a
policy collapsed to "hold W, never click" scores ~92% accuracy on the attack
head while being useless (the hold-W collapse; see dataset.py). Macro-F1
averages F1 over classes, so the rare class counts as much as the majority
class; the collapsed policy gets ~0.48 while anything that actually attacks
scores well above it. ``baseline_macro_f1`` in the returned metrics is the
majority-class predictor's score -- beat that or the head learned nothing.

Checkpoints use the spec.py format::

    torch.save({"model": state_dict, "meta": {...}}, path)
    meta includes: obs_dim, action_heads, step (+ kind="bc", val metrics)

Note: BC provides no gradient to the value head, so it stays at its random
init in the checkpoint; PPO fine-tuning trains its own value function.

CLI::

    python3 -m pvpbot.bc.train_bc --data-dir data/synth --out bc_policy.pt \\
        --epochs 10 --seq-len 32 --burn-in 8
"""
import argparse
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pvpbot.models import PolicyNet
from pvpbot.spec import ACTION_HEADS, OBS_DIM, TICK_RATE

from pvpbot.bc.dataset import (
    BCSequenceDataset,
    RecordingLike,
    compute_head_class_weights,
    split_recordings,
)
from pvpbot.bc.recording import list_recordings, load_recording

HEAD_NAMES = tuple(name for name, _ in ACTION_HEADS)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def macro_f1(pred: np.ndarray, target: np.ndarray, num_classes: int) -> float:
    """Macro-averaged F1 over classes with nonzero support in ``target``.

    Zero-support classes are excluded from the average (they say nothing
    about the classifier); a class that is present but never predicted
    contributes F1=0. This matches the intuition that a majority-class
    predictor on a 90/10 binary head scores ~0.47, not ~0.95.
    """
    pred = np.asarray(pred)
    target = np.asarray(target)
    f1s: List[float] = []
    for c in range(num_classes):
        tp = int(np.sum((pred == c) & (target == c)))
        fp = int(np.sum((pred == c) & (target != c)))
        fn = int(np.sum((pred != c) & (target == c)))
        if tp + fn == 0:
            continue
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2.0 * tp / denom)
    return float(np.mean(f1s)) if f1s else 0.0


def _majority_baseline_f1(target: np.ndarray, num_classes: int) -> float:
    """Macro-F1 of the always-predict-majority-class baseline on ``target``."""
    target = np.asarray(target)
    counts = np.bincount(target, minlength=num_classes)
    majority = int(np.argmax(counts))
    return macro_f1(np.full_like(target, majority), target, num_classes)


@torch.no_grad()
def evaluate(
    policy: PolicyNet,
    loader: DataLoader,
    burn_in: int,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Per-head accuracy, macro-F1, and majority-baseline macro-F1."""
    policy.eval()
    preds: List[List[np.ndarray]] = [[] for _ in HEAD_NAMES]
    targets: List[List[np.ndarray]] = [[] for _ in HEAD_NAMES]
    for batch in loader:
        obs = batch["obs"].to(device)
        acts = batch["actions"].to(device)
        b, w = obs.shape[0], obs.shape[1]
        state = policy.initial_state(b, device=device)
        for t in range(w):
            logits, _, state = policy(obs[:, t], state)
            if t < burn_in:
                continue
            for k, lg in enumerate(logits):
                preds[k].append(lg.argmax(dim=-1).cpu().numpy())
                targets[k].append(acts[:, t, k].cpu().numpy())
    out: Dict[str, Dict[str, float]] = {
        "acc": {}, "macro_f1": {}, "baseline_macro_f1": {}
    }
    for k, (name, size) in enumerate(ACTION_HEADS):
        p = np.concatenate(preds[k])
        t = np.concatenate(targets[k])
        out["acc"][name] = float(np.mean(p == t))
        out["macro_f1"][name] = macro_f1(p, t, size)
        out["baseline_macro_f1"][name] = _majority_baseline_f1(t, size)
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _chunk_loss(
    policy: PolicyNet,
    obs: torch.Tensor,       # (B, W, OBS_DIM)
    acts: torch.Tensor,      # (B, W, heads)
    mask: torch.Tensor,      # (B, W)
    burn_in: int,
    weights: Sequence[torch.Tensor],
    tbptt: int = 0,
) -> torch.Tensor:
    """Masked weighted per-head CE over one chunk, with burn-in + TBPTT.

    Burn-in ticks run without gradient purely to warm the GRU state, which is
    then detached; with ``tbptt > 0`` the state is additionally detached every
    ``tbptt`` scored steps (truncated backprop within the chunk).
    """
    b, w = obs.shape[0], obs.shape[1]
    state = policy.initial_state(b, device=obs.device)
    if burn_in > 0:
        with torch.no_grad():
            for t in range(burn_in):
                _, _, state = policy(obs[:, t], state)
        state = state.detach()
    total = obs.new_zeros(())
    denom = mask[:, burn_in:].sum().clamp(min=1.0)
    for i, t in enumerate(range(burn_in, w)):
        if tbptt > 0 and i > 0 and i % tbptt == 0:
            state = state.detach()
        logits, _, state = policy(obs[:, t], state)
        m = mask[:, t]
        step_loss = obs.new_zeros(())
        for k, lg in enumerate(logits):
            ce = F.cross_entropy(
                lg, acts[:, t, k], weight=weights[k], reduction="none"
            )
            step_loss = step_loss + (ce * m).sum()
        total = total + step_loss
    return total / (denom * len(ACTION_HEADS))


def train_bc(
    data_dir: Optional[str] = None,
    recordings: Optional[Sequence[RecordingLike]] = None,
    out_path: str = "bc_policy.pt",
    epochs: int = 10,
    seq_len: int = 32,
    burn_in: int = 8,
    stride: Optional[int] = None,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    weight_smoothing: float = 1.0,
    max_weight: Optional[float] = None,
    tbptt: int = 0,
    seed: int = 0,
    device: Union[str, torch.device] = "cpu",
    select: str = "best",
    lr_decay: bool = True,
    log: Optional[Callable[[str], None]] = print,
) -> Dict[str, object]:
    """Train PolicyNet by BC. Returns {"val": metrics, "checkpoint": path, ...}.

    Provide either ``data_dir`` (a directory of recording NPZs) or an explicit
    ``recordings`` sequence. Class weights are computed from the training
    split only, so validation metrics are honest.

    ``select="best"`` (default) checkpoints the epoch with the highest mean
    per-head val macro-F1 rather than the last one -- the weighted objective
    makes rare-class argmax decisions borderline, so per-epoch metrics are
    noisy and the final epoch is often not the best. ``select="last"``
    disables this.
    """
    if select not in ("best", "last"):
        raise ValueError("select must be 'best' or 'last'")
    if recordings is None:
        if data_dir is None:
            raise ValueError("provide data_dir or recordings")
        paths = list_recordings(data_dir)
        if not paths:
            raise ValueError("no recordings found in %s" % data_dir)
        recordings = [load_recording(p) for p in paths]
    device = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))

    train_recs, val_recs = split_recordings(recordings, val_frac=val_frac, seed=seed)
    train_ds = BCSequenceDataset(
        train_recs, seq_len=seq_len, burn_in=burn_in, stride=stride
    )
    val_ds = BCSequenceDataset(
        val_recs, seq_len=seq_len, burn_in=burn_in, stride=stride
    )
    weights_np = compute_head_class_weights(
        train_recs, smoothing=weight_smoothing, max_weight=max_weight
    )
    weights = [torch.from_numpy(w).to(device) for w in weights_np]

    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, generator=gen
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    policy = PolicyNet(obs_dim=OBS_DIM).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    # cosine decay to ~0: the weighted objective makes rare-class argmax
    # decisions borderline, so a constant lr keeps flipping them; annealing
    # settles the operating point instead of bouncing around it
    sched = None
    if lr_decay:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    if log:
        log(
            "BC: %d train / %d val episodes, %d train chunks (%d ticks), "
            "window=%d (burn_in=%d + seq_len=%d)"
            % (
                len(train_recs), len(val_recs), len(train_ds),
                train_ds.num_ticks, burn_in + seq_len, burn_in, seq_len,
            )
        )

    step = 0
    history: List[Dict[str, object]] = []
    best_score = float("-inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = -1
    best_val: Optional[Dict[str, Dict[str, float]]] = None
    t0 = time.time()
    for epoch in range(epochs):
        policy.train()
        epoch_loss, n_batches = 0.0, 0
        for batch in train_loader:
            obs = batch["obs"].to(device)
            acts = batch["actions"].to(device)
            mask = batch["loss_mask"].to(device)
            loss = _chunk_loss(policy, obs, acts, mask, burn_in, weights, tbptt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            opt.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
            step += 1
        if sched is not None:
            sched.step()
        val = evaluate(policy, val_loader, burn_in, device)
        history.append(
            {"epoch": epoch, "train_loss": epoch_loss / max(1, n_batches), "val": val}
        )
        score = float(np.mean(list(val["macro_f1"].values())))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_val = val
            best_state = {
                k: v.detach().cpu().clone() for k, v in policy.state_dict().items()
            }
        if log:
            log(
                "epoch %d  loss %.4f  attack acc %.3f  attack macro-F1 %.3f "
                "(baseline %.3f)  [%.1fs]"
                % (
                    epoch,
                    epoch_loss / max(1, n_batches),
                    val["acc"]["attack"],
                    val["macro_f1"]["attack"],
                    val["baseline_macro_f1"]["attack"],
                    time.time() - t0,
                )
            )

    if select == "best" and best_state is not None:
        policy.load_state_dict(best_state)
        final_val = best_val
        selected_epoch = best_epoch
    else:
        final_val = history[-1]["val"] if history else evaluate(
            policy, val_loader, burn_in, device
        )
        selected_epoch = epochs - 1
    meta = {
        "obs_dim": OBS_DIM,
        "action_heads": [list(h) for h in ACTION_HEADS],
        "step": step,
        "kind": "bc",
        "tick_rate": TICK_RATE,
        "seq_len": seq_len,
        "burn_in": burn_in,
        "selected_epoch": selected_epoch,
        "head_class_weights": [w.tolist() for w in weights_np],
        "val": final_val,
    }
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save({"model": policy.state_dict(), "meta": meta}, out_path)
    if log:
        log(
            "saved checkpoint (epoch %d weights) to %s" % (selected_epoch, out_path)
        )
    return {
        "val": final_val,
        "checkpoint": out_path,
        "history": history,
        "steps": step,
        "selected_epoch": selected_epoch,
        "head_class_weights": weights_np,
    }


def load_bc_checkpoint(
    path: str, device: Union[str, torch.device] = "cpu"
) -> PolicyNet:
    """Load a spec.py-format BC checkpoint into a fresh PolicyNet (eval mode)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model" not in ckpt or "meta" not in ckpt:
        raise ValueError("%s: not a spec.py checkpoint" % path)
    net = PolicyNet(obs_dim=int(ckpt["meta"].get("obs_dim", OBS_DIM)))
    net.load_state_dict(ckpt["model"])
    return net.to(device).eval()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Behavior-clone PolicyNet.")
    ap.add_argument("--data-dir", required=True, help="directory of recording NPZs")
    ap.add_argument("--out", default="bc_policy.pt", help="checkpoint output path")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--burn-in", type=int, default=8)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--weight-smoothing", type=float, default=1.0)
    ap.add_argument("--max-weight", type=float, default=None)
    ap.add_argument("--tbptt", type=int, default=0,
                    help="detach GRU state every N scored steps (0 = whole chunk)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    ap.add_argument("--select", choices=("best", "last"), default="best",
                    help="checkpoint the best-val epoch (default) or the last")
    ap.add_argument("--no-lr-decay", action="store_true",
                    help="disable cosine lr annealing")
    args = ap.parse_args(argv)
    results = train_bc(
        data_dir=args.data_dir,
        out_path=args.out,
        epochs=args.epochs,
        seq_len=args.seq_len,
        burn_in=args.burn_in,
        stride=args.stride,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        weight_smoothing=args.weight_smoothing,
        max_weight=args.max_weight,
        tbptt=args.tbptt,
        seed=args.seed,
        device=args.device,
        select=args.select,
        lr_decay=not args.no_lr_decay,
    )
    val = results["val"]
    print("final per-head validation metrics:")
    for name in HEAD_NAMES:
        print(
            "  %-8s acc %.3f  macro-F1 %.3f  (majority baseline %.3f)"
            % (
                name,
                val["acc"][name],
                val["macro_f1"][name],
                val["baseline_macro_f1"][name],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
