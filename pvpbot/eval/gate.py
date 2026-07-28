"""Promotion gate for the trainer's league.

Decides whether a candidate checkpoint should replace the incumbent based on
head-to-head results, using an exact two-sided binomial test implemented from
scratch (no scipy).

Test definition (the "minlike" exact method): under H0 the candidate wins
each decisive game with probability ``p`` (default 0.5).  The p-value is the
total probability of all outcomes ``i`` in [0, n] whose point probability
does not exceed that of the observed win count ``k``:

    p_value = sum_{i: P(X=i) <= P(X=k)} P(X=i),  X ~ Binomial(n, p)

For p = 0.5 this coincides with the symmetric tail definition
``P(|X - n/2| >= |k - n/2|)``, which makes hand computation easy, e.g.
k=8, n=10: 2 * (C(10,0)+C(10,1)+C(10,2)) / 2^10 = 112/1024 = 0.109375.

Draws are excluded from the test (standard sign-test practice); the gate
promotes only if the candidate won strictly more decisive games than the
incumbent AND the two-sided p-value is below alpha.
"""
import math
from dataclasses import dataclass
from typing import Dict

__all__ = ["binomial_pmf", "binomial_two_sided_pvalue", "GateResult",
           "evaluate_gate"]


def binomial_pmf(k: int, n: int, p: float) -> float:
    if k < 0 or k > n:
        return 0.0
    return math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))


def binomial_two_sided_pvalue(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test p-value (minlike method)."""
    if n <= 0:
        return 1.0
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    if not 0.0 < p < 1.0:
        raise ValueError("need 0 < p < 1")
    pk = binomial_pmf(k, n, p)
    cutoff = pk * (1.0 + 1e-9)  # float-safe inclusion of equal-mass outcomes
    total = 0.0
    for i in range(n + 1):
        pi = binomial_pmf(i, n, p)
        if pi <= cutoff:
            total += pi
    return min(total, 1.0)


@dataclass
class GateResult:
    promote: bool
    p_value: float
    wins: int
    losses: int
    draws: int
    n_decisive: int
    win_rate: float           # of decisive games; 0.5 if none
    alpha: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "promote": self.promote, "p_value": self.p_value,
            "wins": self.wins, "losses": self.losses, "draws": self.draws,
            "n_decisive": self.n_decisive, "win_rate": self.win_rate,
            "alpha": self.alpha,
        }


def evaluate_gate(candidate_wins: int, incumbent_wins: int, draws: int = 0,
                  alpha: float = 0.05, min_decisive: int = 10) -> GateResult:
    """Should the candidate replace the incumbent?

    Promotes iff: at least `min_decisive` decisive games were played, the
    candidate won more of them than the incumbent, and the exact two-sided
    binomial test rejects "even match" at level `alpha`.
    """
    w, l = int(candidate_wins), int(incumbent_wins)
    if w < 0 or l < 0 or draws < 0:
        raise ValueError("counts must be non-negative")
    n = w + l
    p_value = binomial_two_sided_pvalue(w, n) if n > 0 else 1.0
    win_rate = (w / n) if n > 0 else 0.5
    promote = n >= min_decisive and w > l and p_value < alpha
    return GateResult(promote=promote, p_value=p_value, wins=w, losses=l,
                      draws=int(draws), n_decisive=n, win_rate=win_rate,
                      alpha=alpha)
