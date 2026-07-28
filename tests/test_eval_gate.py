"""Promotion gate: exact binomial p-values on hand-computed cases."""
import pytest

from pvpbot.eval.gate import (
    binomial_pmf,
    binomial_two_sided_pvalue,
    evaluate_gate,
)


def test_pmf_hand_values():
    assert abs(binomial_pmf(5, 10, 0.5) - 252 / 1024) < 1e-12
    assert abs(binomial_pmf(0, 10, 0.5) - 1 / 1024) < 1e-12
    assert binomial_pmf(-1, 10, 0.5) == 0.0
    assert binomial_pmf(11, 10, 0.5) == 0.0


def test_two_sided_pvalues_hand_computed():
    # k=8, n=10: 2 * (C(10,0)+C(10,1)+C(10,2)) / 2^10 = 112/1024
    assert abs(binomial_two_sided_pvalue(8, 10) - 112 / 1024) < 1e-12
    # k=9, n=10: 2 * (1 + 10) / 1024
    assert abs(binomial_two_sided_pvalue(9, 10) - 22 / 1024) < 1e-12
    # k=10, n=10: both extreme points only
    assert abs(binomial_two_sided_pvalue(10, 10) - 2 / 1024) < 1e-12
    # symmetric: k=0 identical to k=n
    assert abs(binomial_two_sided_pvalue(0, 10)
               - binomial_two_sided_pvalue(10, 10)) < 1e-15
    # dead-center observation: everything is "as extreme"
    assert binomial_two_sided_pvalue(5, 10) == 1.0
    # k=15, n=20: 2 * (15504+4845+1140+190+20+1) / 2^20 = 43400/1048576
    assert abs(binomial_two_sided_pvalue(15, 20) - 43400 / 1048576) < 1e-12


def test_pvalue_edge_cases_and_validation():
    assert binomial_two_sided_pvalue(0, 0) == 1.0
    with pytest.raises(ValueError):
        binomial_two_sided_pvalue(11, 10)
    with pytest.raises(ValueError):
        binomial_two_sided_pvalue(5, 10, p=0.0)
    # asymmetric null still a valid probability, near 1 at the mode
    p = binomial_two_sided_pvalue(9, 10, p=0.9)
    assert 0.9 < p <= 1.0


def test_gate_promotes_significant_winner():
    g = evaluate_gate(candidate_wins=9, incumbent_wins=1)
    assert g.promote is True
    assert abs(g.p_value - 22 / 1024) < 1e-12
    assert g.n_decisive == 10 and g.win_rate == 0.9


def test_gate_rejects_insignificant_lead():
    g = evaluate_gate(candidate_wins=8, incumbent_wins=2)  # p = 0.109 > 0.05
    assert g.promote is False
    assert abs(g.p_value - 112 / 1024) < 1e-12


def test_gate_rejects_losing_candidate_even_if_significant():
    g = evaluate_gate(candidate_wins=1, incumbent_wins=9)
    assert g.promote is False and g.p_value < 0.05


def test_gate_requires_min_decisive_games():
    g = evaluate_gate(candidate_wins=6, incumbent_wins=0, min_decisive=10)
    assert g.promote is False  # only 6 decisive games
    g2 = evaluate_gate(candidate_wins=12, incumbent_wins=0, min_decisive=10)
    assert g2.promote is True and g2.p_value == pytest.approx(2 / 4096)


def test_gate_draws_excluded_from_test():
    with_draws = evaluate_gate(9, 1, draws=100)
    without = evaluate_gate(9, 1, draws=0)
    assert with_draws.p_value == without.p_value
    assert with_draws.promote is without.promote is True
    assert with_draws.draws == 100


def test_gate_no_games():
    g = evaluate_gate(0, 0)
    assert g.promote is False and g.p_value == 1.0 and g.win_rate == 0.5


def test_gate_alpha_threshold():
    assert evaluate_gate(15, 5, alpha=0.05).promote is True   # p = 0.0414
    assert evaluate_gate(15, 5, alpha=0.04).promote is False


def test_gate_serializable():
    d = evaluate_gate(9, 1).to_dict()
    assert d["promote"] is True and d["wins"] == 9 and "p_value" in d
