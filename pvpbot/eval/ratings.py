"""Rating systems over pairwise match results: Elo and TrueSkill-lite.

Both are implemented from scratch (no external rating libraries) and are
JSON-serializable via ``to_json`` / ``from_json``.

Scores use the chess convention: 1.0 = first player wins, 0.0 = loses,
0.5 = draw.

Elo
---
Classic logistic Elo: expected score E = 1 / (1 + 10^((Rb - Ra)/400)),
update R += K * (score - E).  Order of results matters slightly, so batch
feeding should shuffle/interleave outcomes (the ladder does).

TrueSkill-lite
--------------
Each player is a Gaussian N(mu, sigma^2), initialised mu0=25, sigma0=25/3
(the classic TrueSkill scale).  For a decisive game the posterior is
approximated by moment matching against the truncated Gaussian on the
performance difference (the standard 2-player TrueSkill update):

    c   = sqrt(2*beta^2 + sigma_w^2 + sigma_l^2)
    t   = (mu_w - mu_l) / c
    v   = pdf(t) / cdf(t)          (truncation mean shift)
    w   = v * (v + t)              (truncation variance shrink)
    mu_w += sigma_w^2 / c * v ;  mu_l -= sigma_l^2 / c * v
    sigma_i^2 *= 1 - sigma_i^2 / c^2 * w    (then + tau^2 dynamics noise)

Draws are handled "lite": a draw is treated as half a win each way computed
from the pre-game state (mu moves cancel to a pull toward each other,
sigma still shrinks).  ``conservative()`` returns mu - 3*sigma, the usual
displayed rank score.
"""
import json
import math
from typing import Dict, List, Tuple

__all__ = ["Elo", "TrueSkillLite"]


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class Elo:
    def __init__(self, k: float = 16.0, initial: float = 1000.0):
        self.k = float(k)
        self.initial = float(initial)
        self.ratings: Dict[str, float] = {}

    def rating(self, name: str) -> float:
        return self.ratings.setdefault(name, self.initial)

    def expected(self, a: str, b: str) -> float:
        ra, rb = self.rating(a), self.rating(b)
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def record(self, a: str, b: str, score_a: float) -> None:
        """score_a: 1.0 a wins, 0.5 draw, 0.0 b wins."""
        ea = self.expected(a, b)
        delta = self.k * (score_a - ea)
        self.ratings[a] += delta
        self.ratings[b] -= delta

    def record_series(self, a: str, b: str, wins_a: int, wins_b: int,
                      draws: int = 0) -> None:
        """Feed a series interleaved (deterministically) to reduce order bias."""
        outcomes: List[float] = [1.0] * wins_a + [0.0] * wins_b + [0.5] * draws
        # deterministic interleave: stride through the list
        n = len(outcomes)
        if n == 0:
            return
        stride = max(1, int(math.sqrt(n)))
        order = [i for s in range(stride) for i in range(s, n, stride)]
        for i in order:
            self.record(a, b, outcomes[i])

    def to_json(self) -> str:
        return json.dumps(
            {"type": "elo", "k": self.k, "initial": self.initial,
             "ratings": self.ratings}, sort_keys=True)

    @classmethod
    def from_json(cls, blob: str) -> "Elo":
        d = json.loads(blob)
        obj = cls(k=d["k"], initial=d["initial"])
        obj.ratings = {str(k): float(v) for k, v in d["ratings"].items()}
        return obj


class TrueSkillLite:
    MU0 = 25.0
    SIGMA0 = 25.0 / 3.0

    def __init__(self, beta: float = None, tau: float = None):
        self.beta = self.SIGMA0 / 2.0 if beta is None else float(beta)
        self.tau = self.SIGMA0 / 100.0 if tau is None else float(tau)
        self.ratings: Dict[str, Tuple[float, float]] = {}  # name -> (mu, sigma)

    def rating(self, name: str) -> Tuple[float, float]:
        return self.ratings.setdefault(name, (self.MU0, self.SIGMA0))

    def conservative(self, name: str) -> float:
        mu, sigma = self.rating(name)
        return mu - 3.0 * sigma

    def _deltas(self, winner: str, loser: str) -> Tuple[float, float, float, float]:
        """(d_mu_w, d_mu_l, var_factor_w, var_factor_l) for winner beats loser."""
        mw, sw = self.rating(winner)
        ml, sl = self.rating(loser)
        c2 = 2.0 * self.beta ** 2 + sw ** 2 + sl ** 2
        c = math.sqrt(c2)
        t = (mw - ml) / c
        cdf = max(_norm_cdf(t), 1e-12)
        v = _norm_pdf(t) / cdf
        w = v * (v + t)
        return (sw ** 2 / c * v, -(sl ** 2 / c * v),
                max(1.0 - sw ** 2 / c2 * w, 1e-4),
                max(1.0 - sl ** 2 / c2 * w, 1e-4))

    def _apply(self, name: str, d_mu: float, var_factor: float) -> None:
        mu, sigma = self.rating(name)
        sigma2 = sigma ** 2 * var_factor + self.tau ** 2
        self.ratings[name] = (mu + d_mu, math.sqrt(sigma2))

    def record(self, a: str, b: str, score_a: float) -> None:
        """score_a: 1.0 a wins, 0.5 draw, 0.0 b wins."""
        if score_a == 0.5:
            # draw-lite: half a win each way from the pre-game state
            da_w, db_l, fa_w, fb_l = self._deltas(a, b)
            db_w, da_l, fb_w, fa_l = self._deltas(b, a)
            self._apply(a, 0.5 * (da_w + da_l), math.sqrt(fa_w * fa_l))
            self._apply(b, 0.5 * (db_w + db_l), math.sqrt(fb_w * fb_l))
            return
        winner, loser = (a, b) if score_a > 0.5 else (b, a)
        d_mu_w, d_mu_l, f_w, f_l = self._deltas(winner, loser)
        self._apply(winner, d_mu_w, f_w)
        self._apply(loser, d_mu_l, f_l)

    def record_series(self, a: str, b: str, wins_a: int, wins_b: int,
                      draws: int = 0) -> None:
        outcomes: List[float] = [1.0] * wins_a + [0.0] * wins_b + [0.5] * draws
        n = len(outcomes)
        if n == 0:
            return
        stride = max(1, int(math.sqrt(n)))
        for s in range(stride):
            for i in range(s, n, stride):
                self.record(a, b, outcomes[i])

    def to_json(self) -> str:
        return json.dumps(
            {"type": "trueskill-lite", "beta": self.beta, "tau": self.tau,
             "ratings": {k: list(v) for k, v in self.ratings.items()}},
            sort_keys=True)

    @classmethod
    def from_json(cls, blob: str) -> "TrueSkillLite":
        d = json.loads(blob)
        obj = cls(beta=d["beta"], tau=d["tau"])
        obj.ratings = {str(k): (float(v[0]), float(v[1]))
                       for k, v in d["ratings"].items()}
        return obj
