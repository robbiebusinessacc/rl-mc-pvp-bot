"""Behavior-cloning prior + humanization.

Submodules:
  recording   -- on-disk NPZ episode format, writer/loader, mouse<->bin quantization
  dataset     -- chunked sequence dataset + class-imbalance weights for BC
  train_bc    -- CLI / function to behavior-clone PolicyNet from recordings
  kl          -- KL(pi_RL || pi_BC) auxiliary loss for the PPO trainer
  humanize    -- deployment-time reaction delay, mouse smoothing, click jitter
  synth_demos -- synthetic "human-like" demonstrations rolled from the stub env
"""
from pvpbot.bc.recording import (
    Recording,
    RecordingWriter,
    bins_to_mouse,
    iter_recordings,
    list_recordings,
    load_recording,
    mouse_to_bins,
)
from pvpbot.bc.dataset import (
    BCSequenceDataset,
    compute_head_class_weights,
    split_recordings,
)
from pvpbot.bc.kl import KLPriorLoss, kl_categorical
from pvpbot.bc.humanize import ClickJitter, MouseSmoother, ReactionDelay

__all__ = [
    "Recording",
    "RecordingWriter",
    "bins_to_mouse",
    "iter_recordings",
    "list_recordings",
    "load_recording",
    "mouse_to_bins",
    "BCSequenceDataset",
    "compute_head_class_weights",
    "split_recordings",
    "KLPriorLoss",
    "kl_categorical",
    "ClickJitter",
    "MouseSmoother",
    "ReactionDelay",
]
