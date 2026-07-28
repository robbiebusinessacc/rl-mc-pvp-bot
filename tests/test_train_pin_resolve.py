"""The pinned opponent must resolve against the pool's ACTUAL entry names.

League stores scripted baselines as "scripted_<name>" while --pin-opponent is
given as the bare name ("P4-Hacker"). An exact-equality lookup in run.py's
_resolve_pin silently returned None for 1B steps (2026-07-26 -> 27): pin domain
randomization switched itself off and the learner ground away against a FIXED
official bar -- precisely the overfitting condition DR was added to prevent --
while reporting nothing. These tests pin the naming contract on both sides.
"""
from pvpbot.train.league import League, discover_scripted


def _resolve(league, want):
    """Mirror of run.py::_resolve_pin (substring, case-insensitive, live agent)."""
    want = (want or "").lower()
    if not want:
        return None
    for e in league.pool:
        if want in getattr(e, "name", "").lower() and e.agent is not None:
            return e
    return None


def test_scripted_pool_entries_are_prefixed():
    """If this ever stops holding, the bare-name pin flag needs revisiting."""
    league = League(8, pin_name="P4-Hacker", pin_frac=0.5)
    scripted = [e.name for e in league.pool if e.kind == "scripted"]
    assert scripted, "no scripted baselines discovered"
    assert all(n.startswith("scripted_") for n in scripted), scripted


def test_bare_pin_name_resolves_to_prefixed_entry():
    league = League(8, pin_name="P4-Hacker", pin_frac=0.5)
    entry = _resolve(league, league._pin_name)
    assert entry is not None, "pin failed to resolve -- DR would silently disable"
    assert entry.name == "scripted_P4-Hacker"
    assert entry.agent is not None


def test_exact_equality_would_have_missed_it():
    """Guards the specific regression: == against the bare name finds nothing."""
    league = League(8, pin_name="P4-Hacker", pin_frac=0.5)
    assert not [e for e in league.pool if e.name == "P4-Hacker"]


def test_league_binds_pin_index_to_same_entry():
    """run.py's resolver and League's own binding must agree."""
    league = League(8, pin_name="P4-Hacker", pin_frac=0.5)
    assert league._pin_idx is not None
    assert league.pool[league._pin_idx] is _resolve(league, league._pin_name)


def test_pin_survives_state_dict_roundtrip():
    league = League(8, pin_name="P4-Hacker", pin_frac=0.5)
    other = League(8, pin_name="P4-Hacker", pin_frac=0.5)
    other.load_state_dict(league.state_dict())
    assert other._pin_idx is not None
    entry = _resolve(other, other._pin_name)
    assert entry is not None and entry.agent is not None
    assert other.pool[other._pin_idx] is entry


def test_every_discovered_baseline_is_pinnable_by_bare_name():
    league = League(8, pin_name="P4-Hacker", pin_frac=0.5)
    for agent in discover_scripted():
        bare = getattr(agent, "name", None) or type(agent).__name__
        assert _resolve(league, bare) is not None, bare
