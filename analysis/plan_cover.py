"""
Cover of a stock vector for one zone, exact under the lognormal-Poisson model,
and the plans that meet a cover target.

    Y_p | u, w_p ~ Poisson(lam_p exp(u + w_p - (tau^2 + omega^2)/2))
    u ~ N(0, tau^2) shared by the zone, w_p ~ N(0, omega^2) per reference

Both latent terms sit on one lattice so that every u + w is a lattice point
(the device used in stock_vector.py), which lets references have different
means. Mission-level noise that is independent across references can be folded
into omega: pass omega = sqrt(omega^2 + eps^2).

The same plans can be computed from predictive draws (plan_*_draws), which is
how a fitted model is used.
"""
import math

import numpy as np
from scipy.stats import poisson

SD_ZERO = 0.01     # a latent sd below this is treated as zero
PER_SD = 16        # lattice points per standard deviation of the smaller term
HALF_WIDTH = 8.0   # lattice half-width in standard deviations


def _side(sd, h):
    if sd < SD_ZERO:
        return np.zeros(1, int), np.ones(1)
    k = int(math.ceil(HALF_WIDTH * sd / h))
    j = np.arange(-k, k + 1)
    wt = np.exp(-0.5 * (j * h / sd) ** 2)
    return j, wt / wt.sum()


class Zone:
    def __init__(self, lams, tau, omega, smax=None):
        lams = np.atleast_1d(np.asarray(lams, float))
        self.lams, self.P = lams, len(lams)
        self.tau, self.omega = float(tau), float(omega)
        sds = [s for s in (tau, omega) if s >= SD_ZERO]
        h = min(sds) / PER_SD if sds else 1.0
        ju, self.wu = _side(tau, h)
        jw, ww = _side(omega, h)
        s2 = (tau if tau >= SD_ZERO else 0) ** 2 + (omega if omega >= SD_ZERO else 0) ** 2
        if smax is None:
            tot = lams.sum()
            var = tot + tot ** 2 * (math.exp(s2) - 1)
            smax = int(tot + 12 * math.sqrt(var) + 30)
        self.smax = smax
        v = (np.arange(ju[0] + jw[0], ju[-1] + jw[-1] + 1)) * h
        s = np.arange(smax + 1)
        off = jw[0]
        self.G, self.g = [], []
        for lam in lams:
            mu = lam * np.exp(v - s2 / 2)
            F = poisson.cdf(s[None, :], mu[:, None])
            f = poisson.pmf(s[None, :], mu[:, None])
            G = np.zeros((len(ju), smax + 1))
            g = np.zeros((len(ju), smax + 1))
            for k, wk in zip(jw, ww):
                rows = ju - ju[0] + (k - off)
                G += wk * F[rows]
                g += wk * f[rows]
            self.G.append(G)
            self.g.append(g)

    def _col(self, p, s):
        return self.G[p][:, min(int(s), self.smax)]

    def marginal(self, p, s):
        return float(self.wu @ self._col(p, s))

    def joint(self, stocks):
        prod = np.ones(len(self.wu))
        for p, s in enumerate(stocks):
            prod *= self._col(p, s)
        return float(self.wu @ prod)

    def total_cdf(self):
        acc = np.zeros(self.smax + 1)
        for i, wi in enumerate(self.wu):
            pmf = self.g[0][i]
            for p in range(1, self.P):
                pmf = np.convolve(pmf, self.g[p][i])[:self.smax + 1]
            acc += wi * pmf
        return np.cumsum(acc)

    def aggregate(self, S):
        return float(self.total_cdf()[min(int(S), self.smax)])

    # plans --------------------------------------------------------------------
    def plan_marginal(self, level):
        """Each reference at its own marginal quantile."""
        out = []
        for p in range(self.P):
            c = self.wu @ self.G[p]
            out.append(int(np.argmax(c >= level)))
        return np.array(out)

    def plan_joint(self, level, cost=None):
        return greedy_joint(self.joint, self.plan_marginal(level), level, cost)

    def plan_common(self, level):
        for s in range(self.smax + 1):
            if self.joint([s] * self.P) >= level:
                return np.full(self.P, s)
        return np.full(self.P, self.smax)

    def plan_aggregate(self, level):
        return int(np.argmax(self.total_cdf() >= level))


def greedy_joint(cover, start, level, cost=None, max_steps=100000):
    """Marginal analysis: add the unit with the largest cover gain per unit cost."""
    s = np.array(start, int)
    cost = np.ones(len(s)) if cost is None else np.asarray(cost, float)
    c = cover(s)
    for _ in range(max_steps):
        if c >= level:
            break
        best, best_gain, best_c = None, -1.0, c
        for p in range(len(s)):
            s[p] += 1
            cp = cover(s)
            s[p] -= 1
            gain = (math.log(max(cp, 1e-300)) - math.log(max(c, 1e-300))) / cost[p]
            if gain > best_gain:
                best, best_gain, best_c = p, gain, cp
        s[best] += 1
        c = best_c
    return s


# from predictive draws ---------------------------------------------------------
def plan_marginal_draws(Y, level):
    return np.array([int(np.ceil(np.quantile(Y[:, p], level, method="inverted_cdf")))
                     for p in range(Y.shape[1])])


def joint_cover_draws(Y, stocks):
    return float(np.mean(np.all(Y <= np.asarray(stocks)[None, :], axis=1)))


def plan_joint_draws(Y, level, cost=None):
    return greedy_joint(lambda s: joint_cover_draws(Y, s), plan_marginal_draws(Y, level),
                        level, cost)


def plan_joint_draws_robust(Y, level, cost=None, max_drops=500):
    """Joint plan from draws that stays fast when the draws are heavy-tailed.

    Raise all references to a common marginal level until the joint target is
    met (bisection), then remove units one at a time, the costliest first,
    while the target still holds."""
    cost = np.ones(Y.shape[1]) if cost is None else np.asarray(cost, float)
    lo, hi = level, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if joint_cover_draws(Y, plan_marginal_draws(Y, mid)) >= level:
            hi = mid
        else:
            lo = mid
    s = plan_marginal_draws(Y, hi)
    for _ in range(max_drops):
        best = None
        for p in np.argsort(-cost):
            if s[p] == 0:
                continue
            s[p] -= 1
            if joint_cover_draws(Y, s) >= level:
                best = p
            s[p] += 1
            if best is not None:
                break
        if best is None:
            break
        s[best] -= 1
    return s


def plan_aggregate_draws(Y, level):
    return int(np.ceil(np.quantile(Y.sum(axis=1), level, method="inverted_cdf")))
