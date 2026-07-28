"""Elo and TrueSkill-lite: convergence order on synthetic data, JSON I/O."""
import numpy as np

from pvpbot.eval.ratings import Elo, TrueSkillLite


def _synthetic_results(rng):
    """A beats B 90%, B beats C 90%, A beats C 99% -> true order A > B > C."""
    games = []
    for _ in range(200):
        games.append(("A", "B", 1.0 if rng.random() < 0.9 else 0.0))
        games.append(("B", "C", 1.0 if rng.random() < 0.9 else 0.0))
        games.append(("A", "C", 1.0 if rng.random() < 0.99 else 0.0))
    rng.shuffle(games)
    return games


def test_elo_expected_symmetric_at_start():
    elo = Elo()
    assert abs(elo.expected("x", "y") - 0.5) < 1e-12


def test_elo_zero_sum_and_direction():
    elo = Elo(k=32)
    elo.record("a", "b", 1.0)
    assert elo.rating("a") > elo.initial > elo.rating("b")
    assert abs((elo.rating("a") - elo.initial)
               + (elo.rating("b") - elo.initial)) < 1e-9


def test_elo_converges_in_right_order():
    rng = np.random.default_rng(0)
    elo = Elo()
    for a, b, s in _synthetic_results(rng):
        elo.record(a, b, s)
    assert elo.rating("A") > elo.rating("B") > elo.rating("C")
    assert elo.rating("A") - elo.rating("C") > 200  # big true gap shows up


def test_elo_record_series_matches_totals():
    elo = Elo()
    elo.record_series("p", "q", wins_a=8, wins_b=2, draws=2)
    assert elo.rating("p") > elo.rating("q")


def test_elo_json_roundtrip():
    elo = Elo(k=24, initial=1200)
    elo.record("a", "b", 1.0)
    elo.record("a", "c", 0.5)
    back = Elo.from_json(elo.to_json())
    assert back.k == 24 and back.initial == 1200
    assert back.ratings == elo.ratings


def test_trueskill_converges_in_right_order():
    rng = np.random.default_rng(1)
    ts = TrueSkillLite()
    for a, b, s in _synthetic_results(rng):
        ts.record(a, b, s)
    mus = {n: ts.rating(n)[0] for n in "ABC"}
    assert mus["A"] > mus["B"] > mus["C"]
    for n in "ABC":  # uncertainty must shrink with games played
        assert ts.rating(n)[1] < TrueSkillLite.SIGMA0 * 0.5
    assert ts.conservative("A") > ts.conservative("C")


def test_trueskill_win_moves_mu_correct_direction():
    ts = TrueSkillLite()
    ts.record("w", "l", 1.0)
    (mw, sw), (ml, sl) = ts.rating("w"), ts.rating("l")
    assert mw > TrueSkillLite.MU0 > ml
    assert sw < TrueSkillLite.SIGMA0 and sl < TrueSkillLite.SIGMA0


def test_trueskill_draws_keep_equals_close():
    ts = TrueSkillLite()
    for _ in range(50):
        ts.record("d1", "d2", 0.5)
    (m1, s1), (m2, s2) = ts.rating("d1"), ts.rating("d2")
    assert abs(m1 - m2) < 0.5
    assert s1 < TrueSkillLite.SIGMA0 and s2 < TrueSkillLite.SIGMA0


def test_trueskill_upset_moves_more_than_expected_win():
    ts = TrueSkillLite()
    ts.ratings["strong"] = (35.0, 2.0)
    ts.ratings["weak"] = (15.0, 2.0)
    ts.record("weak", "strong", 1.0)  # huge upset
    up_move = ts.rating("weak")[0] - 15.0
    ts2 = TrueSkillLite()
    ts2.ratings["strong"] = (35.0, 2.0)
    ts2.ratings["weak"] = (15.0, 2.0)
    ts2.record("strong", "weak", 1.0)  # expected result
    exp_move = ts2.rating("strong")[0] - 35.0
    assert up_move > exp_move >= 0.0


def test_trueskill_json_roundtrip():
    ts = TrueSkillLite()
    ts.record("a", "b", 1.0)
    ts.record("a", "c", 0.5)
    back = TrueSkillLite.from_json(ts.to_json())
    assert back.beta == ts.beta and back.tau == ts.tau
    assert back.ratings == ts.ratings
