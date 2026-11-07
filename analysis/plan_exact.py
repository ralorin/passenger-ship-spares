"""
Unit-minimal joint plans, certified for the law they are computed from.

opt_draws      min sum(s) subject to the share of predictive draws with
               Y <= s in every column being at least the target. Iterative
               deepening on the units above the marginal quantiles; a branch is
               cut when the draws still coverable fall below the target. The
               result is optimal for the empirical law of the draws, or, when
               the node limit is reached, the best plan known with a lower bound.
opt_product    the same problem when the joint law is the product of given
               marginal distribution functions (independent references). The
               log of the cover is separable, so a dynamic programme over the
               units above the marginal quantiles is exact.
"""
import math

import numpy as np

EPS = 1e-12


def _need(level, R):
    return int(math.ceil(level * R - 1e-9))


def opt_draws(Y, level, ub=None, node_limit=300000):
    """Returns (plan, certified, lower_bound_units, nodes)."""
    Y = np.asarray(Y)
    R, P = Y.shape
    k = _need(level, R)
    q = np.sort(Y, axis=0)[k - 1].astype(int)
    D = np.maximum(Y - q[None, :], 0)
    rows, cnt = np.unique(D, axis=0, return_counts=True)
    free = int(cnt[~rows.any(axis=1)].sum())
    if free >= k:
        return q.copy(), True, int(q.sum()), 0
    keep = rows.any(axis=1)
    rows, cnt = rows[keep], cnt[keep]
    need = k - free
    # branch first on the columns that exceed their quantile most often
    order = np.argsort(-(cnt[:, None] * (rows > 0)).sum(axis=0))
    rows = rows[:, order]
    E_ub = int((np.asarray(ub) - q).sum()) if ub is not None else int(rows.sum(axis=1).max()) * P
    nodes = [0]

    def last_col(mask, budget):
        v, c = rows[mask, -1], cnt[mask]
        if c.sum() < need:
            return None
        o = np.argsort(v, kind="stable")
        cs = np.cumsum(c[o])
        e = int(v[o][np.searchsorted(cs, need)])
        return e if e <= budget else None

    def dfs(j, mask, budget, e):
        nodes[0] += 1
        if nodes[0] > node_limit:
            raise TimeoutError
        if j == P - 1:
            x = last_col(mask, budget)
            if x is None:
                return None
            return e + [x]
        col = rows[mask, j]
        vals = np.unique(np.r_[0, col[col <= budget]])
        for x in vals:
            m2 = mask.copy()
            m2[mask] = col <= x
            if cnt[m2].sum() < need:
                continue
            r = dfs(j + 1, m2, budget - int(x), e + [int(x)])
            if r is not None:
                return r
        return None

    full = np.ones(len(rows), bool)
    for E in range(0, E_ub + 1):
        try:
            r = dfs(0, full, E, [])
        except TimeoutError:
            return (np.asarray(ub, int).copy() if ub is not None else None), False, int(q.sum()) + E, nodes[0]
        if r is not None:
            e = np.zeros(P, int)
            e[order] = r
            return q + e, True, int(q.sum()) + E, nodes[0]
    return np.asarray(ub, int).copy(), True, int(np.sum(ub)), nodes[0]


def opt_product(logF, level, e_max=400):
    """logF[p][s] = log Pr(Y_p <= s) for s = 0..len-1 (last value treated as
    holding beyond). Returns the unit-minimal plan with sum_p logF >= log level."""
    target = math.log(level) - EPS
    P = len(logF)
    q = np.array([int(np.argmax(np.asarray(L) >= target)) if np.any(np.asarray(L) >= target)
                  else len(L) - 1 for L in logF])

    def val(p, s):
        L = logF[p]
        return L[min(s, len(L) - 1)]

    best = np.zeros(1)                     # best[E]: max sum of logF with E extra units
    arg = []
    for p in range(P):
        opts = np.array([val(p, q[p] + e) for e in range(e_max + 1)])
        new = np.full(len(best) + e_max, -np.inf)
        choice = np.zeros(len(new), int)
        for e, v in enumerate(opts):
            cand = best + v
            sl = slice(e, e + len(best))
            better = cand > new[sl]
            new[sl] = np.where(better, cand, new[sl])
            choice[sl] = np.where(better, e, choice[sl])
        arg.append(choice)
        best = new
        ok = np.flatnonzero(best >= target)
        # the earliest feasible total can only fall as references are added: keep all
    ok = np.flatnonzero(best >= target)
    if not len(ok):
        raise ValueError("target not reachable within e_max")
    E = int(ok[0])
    e = np.zeros(P, int)
    for p in range(P - 1, -1, -1):
        e[p] = arg[p][E]
        E -= e[p]
    return q + e


def ecdf_log(col, smax):
    col = np.asarray(col)
    c = np.bincount(np.minimum(col, smax + 1), minlength=smax + 2)[:smax + 1].cumsum() / len(col)
    with np.errstate(divide="ignore"):
        return np.log(c)
