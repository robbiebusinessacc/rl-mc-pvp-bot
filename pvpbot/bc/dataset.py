"""Chunked sequence dataset over many recordings, for recurrent BC.

Why the class-imbalance machinery exists (the "hold-W collapse")
----------------------------------------------------------------
In sword PvP the label distribution per head is savagely skewed: a human
holds W on the vast majority of ticks, attacks on maybe 5-10% of them, and
jumps even less. Plain per-head cross-entropy is minimized by predicting the
marginal distribution, so a naively trained clone converges to "walk forward,
never click" -- exactly the failure the reference video author hit: his BC
bot held W into the opponent and rarely swung, because the rare-but-decisive
actions (attack, jump) contribute almost nothing to the unweighted loss.

The fix implemented here: per-head, per-class inverse-frequency loss weights
computed from the *training* recordings (sklearn's ``class_weight="balanced"``
heuristic): ``w_c = N / (K * count_c)`` for a head with K classes and N total
ticks. Under the data distribution the expected weight is exactly 1, so the
overall loss scale is unchanged while rare classes (attack=1, jump=1) are
boosted in proportion to their rarity. Pair this with macro-F1 validation
(see train_bc) -- plain accuracy still looks great on a collapsed policy.

Chunking / burn-in
------------------
PolicyNet is recurrent (GRU core), so samples are fixed-length windows of
``burn_in + seq_len`` consecutive ticks. The first ``burn_in`` ticks warm up
the hidden state from zeros and are excluded from the loss via ``loss_mask``;
the trainer should run them without gradient and detach the state before the
scored region (truncated BPTT). Episodes shorter than one window are skipped.
The train/val split is done at the *episode* level (``split_recordings``) so
overlapping windows never leak across the split.
"""
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from pvpbot.spec import ACTION_HEADS, ACTION_HEAD_SIZES

from pvpbot.bc.recording import Recording, load_recording

RecordingLike = Union[Recording, str, "os.PathLike[str]"]


def _as_recording(item: RecordingLike) -> Recording:
    if isinstance(item, Recording):
        return item
    return load_recording(item)


def _actions_of(item: Union[Recording, np.ndarray]) -> np.ndarray:
    if isinstance(item, Recording):
        return item.actions
    return np.asarray(item)


# ---------------------------------------------------------------------------
# Imbalance statistics
# ---------------------------------------------------------------------------

def compute_action_class_counts(
    recordings: Iterable[Union[Recording, np.ndarray]]
) -> List[np.ndarray]:
    """Per-head class counts over all ticks. Returns one int64 array per head."""
    counts = [np.zeros(n, dtype=np.int64) for n in ACTION_HEAD_SIZES]
    total = 0
    for item in recordings:
        actions = _actions_of(item)
        total += actions.shape[0]
        for k, n in enumerate(ACTION_HEAD_SIZES):
            counts[k] += np.bincount(actions[:, k], minlength=n).astype(np.int64)
    if total == 0:
        raise ValueError("no ticks in provided recordings")
    return counts


def compute_head_class_weights(
    recordings: Iterable[Union[Recording, np.ndarray]],
    smoothing: float = 1.0,
    max_weight: Optional[float] = None,
) -> List[np.ndarray]:
    """Inverse-frequency loss weights per head: ``w_c = N / (K * count_c)``.

    This is the "balanced" heuristic; it satisfies ``sum_c w_c * count_c = N``,
    i.e. the expected weight under the data distribution is 1, so it rescales
    rare classes without changing the overall loss magnitude. This is the
    direct antidote to the hold-W collapse described in the module docstring.

    smoothing   -- Laplace count added to every class before inverting; keeps
                   never-observed classes finite. With ``smoothing=0`` a
                   zero-count class gets weight 0.0 (it can never appear as a
                   target, so its CE weight is irrelevant).
    max_weight  -- optional cap for pathologically rare classes.

    Returns one float32 array of shape (head_size,) per head, in
    ``ACTION_HEADS`` order.
    """
    if smoothing < 0:
        raise ValueError("smoothing must be >= 0")
    counts = compute_action_class_counts(recordings)
    weights: List[np.ndarray] = []
    for c in counts:
        c = c.astype(np.float64) + float(smoothing)
        n_total = c.sum()
        k = len(c)
        w = np.zeros(k, dtype=np.float64)
        nz = c > 0
        w[nz] = n_total / (k * c[nz])
        if max_weight is not None:
            w = np.minimum(w, float(max_weight))
        weights.append(w.astype(np.float32))
    return weights


def split_recordings(
    recordings: Sequence[RecordingLike], val_frac: float = 0.2, seed: int = 0
) -> Tuple[List[Recording], List[Recording]]:
    """Deterministic episode-level train/val split (no window leakage).

    Guarantees at least one episode on each side; requires >= 2 recordings.
    """
    recs = [_as_recording(r) for r in recordings]
    if len(recs) < 2:
        raise ValueError("need at least 2 recordings to split train/val")
    if not (0.0 < val_frac < 1.0):
        raise ValueError("val_frac must be in (0, 1)")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(recs))
    n_val = int(round(len(recs) * val_frac))
    n_val = max(1, min(n_val, len(recs) - 1))
    val_idx = set(order[:n_val].tolist())
    train = [recs[i] for i in range(len(recs)) if i not in val_idx]
    val = [recs[i] for i in range(len(recs)) if i in val_idx]
    return train, val


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BCSequenceDataset(Dataset):
    """Fixed-length windows of consecutive ticks across many recordings.

    Each sample is a dict of torch tensors:
      obs        (burn_in + seq_len, OBS_DIM)          float32
      actions    (burn_in + seq_len, NUM_ACTION_HEADS) int64
      loss_mask  (burn_in + seq_len,)                  float32
                 0.0 on the first ``burn_in`` ticks (hidden-state warm-up,
                 no loss), 1.0 afterwards.

    stride defaults to ``max(1, seq_len // 2)`` (overlapping windows) to get
    more samples out of short recordings; set ``stride=seq_len + burn_in``
    for disjoint windows.
    """

    def __init__(
        self,
        recordings: Sequence[RecordingLike],
        seq_len: int = 32,
        burn_in: int = 8,
        stride: Optional[int] = None,
    ):
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if burn_in < 0:
            raise ValueError("burn_in must be >= 0")
        self.seq_len = int(seq_len)
        self.burn_in = int(burn_in)
        self.window = self.seq_len + self.burn_in
        self.stride = int(stride) if stride is not None else max(1, seq_len // 2)
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        self.recordings: List[Recording] = [_as_recording(r) for r in recordings]
        if not self.recordings:
            raise ValueError("no recordings provided")
        self._index: List[Tuple[int, int]] = []
        self.num_skipped = 0
        for ri, rec in enumerate(self.recordings):
            t = rec.num_ticks
            if t < self.window:
                self.num_skipped += 1
                continue
            for start in range(0, t - self.window + 1, self.stride):
                self._index.append((ri, start))
        if not self._index:
            raise ValueError(
                "no recording is long enough for a %d-tick window "
                "(burn_in=%d + seq_len=%d)" % (self.window, self.burn_in, self.seq_len)
            )
        mask = np.ones(self.window, dtype=np.float32)
        mask[: self.burn_in] = 0.0
        self._mask = torch.from_numpy(mask)

    @property
    def num_ticks(self) -> int:
        return sum(r.num_ticks for r in self.recordings)

    def class_weights(
        self, smoothing: float = 1.0, max_weight: Optional[float] = None
    ) -> List[np.ndarray]:
        """Imbalance weights over this dataset's recordings (see module doc)."""
        return compute_head_class_weights(
            self.recordings, smoothing=smoothing, max_weight=max_weight
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ri, start = self._index[i]
        rec = self.recordings[ri]
        stop = start + self.window
        return {
            "obs": torch.from_numpy(
                np.ascontiguousarray(rec.obs[start:stop], dtype=np.float32)
            ),
            "actions": torch.from_numpy(
                np.ascontiguousarray(rec.actions[start:stop])
            ),
            "loss_mask": self._mask.clone(),
        }


HEAD_NAMES = tuple(name for name, _ in ACTION_HEADS)
