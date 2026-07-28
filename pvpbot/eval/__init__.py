"""Evaluation module: scripted opponents, arena match runner, rating systems,
ladder CLI and promotion gate.

Public surface:
    pvpbot.eval.scripted  -- ScriptedPolicy tiers T0..T4
    pvpbot.eval.ratings   -- Elo and TrueSkill-lite
    pvpbot.eval.arena     -- run_match() duel runner + checkpoint loading
    pvpbot.eval.ladder    -- round-robin CLI (python3 -m pvpbot.eval.ladder)
    pvpbot.eval.gate      -- promotion gate (exact two-sided binomial test)
"""


def get_scripted_opponents():
    """League baselines: practice-server-style bots (see practice.py)."""
    from pvpbot.eval.practice import practice_tiers

    tiers = practice_tiers(seed=7)
    for t in tiers:
        t.elo = float(t.intended_elo)
    return tiers
