"""Self-play RL trainer: PPO + opponent league + training CLI.

Modules:
    ppo.py    -- PPO core: GAE, multi-discrete log-prob/entropy, running obs
                 normalization, rollout buffer with truncated-BPTT chunking,
                 the PPOTrainer update step.
    league.py -- self-play opponent pool with Elo bookkeeping and gating.
    run.py    -- CLI entry point, rollout collector, checkpointing, logging.
"""
