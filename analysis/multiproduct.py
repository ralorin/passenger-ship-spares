"""
A ship carries several products: how the allocation of latent spread changes
the plan.

Exact numerical calculation; no device data are used.

A ship embarks spares for P product populations in the same zone and the
decision is the total over them, so the dependence between populations of one
ship enters the arithmetic. Two allocations of the same total latent variance
sigma^2 = tau^2 + omega^2 are compared:

  A  ship-wide only (tau = sigma, omega = 0). One deviation u is shared by
     every population of the ship, so it does not average out when products
     are summed.
         Y | u ~ Poisson(Lambda e^{u - sigma^2/2})
  B  interaction only (tau = 0, omega = sigma). Each (ship, product)
     population deviates independently, so the sum concentrates.
         Y_p | w_p ~ Poisson(lambda_p e^{w_p - sigma^2/2}), w_p iid
  and the intermediate mixtures with both terms.

Both give the same marginal distribution for a single population, and the same
mean for the total; they differ only in the dependence between products of one
ship. That dependence changes the number of spares a multi-product ship must
embark, and the two structures are hard to tell apart from prediction alone,
so the manuscript treats the allocation as unidentified until a fleet
measures it.

The count distribution of the total is computed exactly (to quadrature and
truncation error, both reported): for each node of the ship-wide term the
product populations are conditionally independent, so their pmfs are convolved
and the result averaged over the ship-wide term.

    python multiproduct.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data")
os.makedirs(OUT, exist_ok=True)

LEVELS = (90, 95, 99)
NODES = 801          # per latent term; convergence checked in main()
TAIL = 1e-13


def _grid(sigma, nodes):
    """Simpson nodes and normalised weights for N(0, sigma^2) on +-10 sigma."""
    if sigma <= 0:
        return np.zeros(1), np.ones(1)
    n = int(nodes) | 1
    u = np.linspace(-10.0 * sigma, 10.0 * sigma, n)
    w = np.ones(n)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    w = w * norm.pdf(u, scale=sigma)
    return u, w / w.sum()


def _pois_pmf(lam, nmax):
    k = np.arange(nmax + 1)
    return poisson.pmf(k, lam)


def _lattice(tau, omega, nodes):
    """Common lattice for u and w, so that u+w stays on it.

    With one step for both terms the Poisson pmf at a given rate is computed
    once, not once per (u, w) pair; otherwise the cost is quadratic in the
    node count.
    """
    span = 10.0 * max(tau, omega, 1e-12)
    n = int(nodes) | 1
    h = 2.0 * span / (n - 1)
    def side(s):
        if s <= 0:
            return np.array([0]), np.array([1.0])
        m = int(np.ceil(10.0 * s / h))
        idx = np.arange(-m, m + 1)
        w = np.ones(len(idx))
        if len(idx) >= 3:
            w[1:-1:2], w[2:-1:2] = 4.0, 2.0
        w = w * norm.pdf(idx * h, scale=s)
        return idx, w / w.sum()
    iu, wu = side(tau)
    iw, ww = side(omega)
    return h, iu, wu, iw, ww


def total_pmf(lams, tau, omega, nmax=None, nodes=NODES):
    """pmf of the total failure count of P populations on one ship.

    lams  expected count of each population under the shared covariates;
    tau   ship-wide latent sd; omega  (ship, product) interaction latent sd.
    The mean-preserving centring uses the total latent variance tau^2+omega^2.
    """
    lams = np.asarray(lams, float)
    tot = lams.sum()
    s2 = tau ** 2 + omega ** 2
    if nmax is None:
        nmax = int(max(40, np.ceil(tot * np.exp(3.5 * np.sqrt(s2)) + 12 * np.sqrt(tot) + 40)))
    h, iu, wu, iw, ww = _lattice(tau, omega, nodes)
    # pmf of one population at every lattice value of the latent sum, per lambda
    lo, hi = iu.min() + iw.min(), iu.max() + iw.max()
    shift = -lo
    table = {}
    for lam in np.unique(lams):
        table[lam] = np.stack([_pois_pmf(lam * np.exp(j * h - s2 / 2), nmax)
                               for j in range(lo, hi + 1)])
    out = np.zeros(nmax + 1)
    for ui, wui in zip(iu, wu):
        conv = np.zeros(nmax + 1)
        conv[0] = 1.0
        for lam in lams:
            rows = table[lam][iw + ui + shift]          # (len(iw), nmax+1)
            pm = ww @ rows                              # mix over the interaction
            conv = np.convolve(conv, pm)[:nmax + 1]
        out += wui * conv
    return out, float(1.0 - out.sum())


def quantile_from_pmf(pmf, p):
    c = np.cumsum(pmf)
    idx = np.argmax(c >= p / 100.0)
    return int(idx) if c[idx] >= p / 100.0 else int(len(pmf) - 1)


def cover_from_pmf(pmf, q):
    return float(np.cumsum(pmf)[min(q, len(pmf) - 1)])


def main():
    sigma = 0.6            # total latent sd, an assumed value (see the manuscript)
    total_mean = 8.0       # expected failures of the whole zone over the mission
    rows = []
    for P in (1, 2, 4, 8):
        lams = np.full(P, total_mean / P)
        for split in (("A: ship-wide only", sigma, 0.0),
                      ("mixed: equal split", sigma / np.sqrt(2), sigma / np.sqrt(2)),
                      ("B: interaction only", 0.0, sigma)):
            name, tau, om = split
            pmf, tail = total_pmf(lams, tau, om)
            r = dict(P=P, structure=name, tau=float(tau), omega=float(om),
                     total_mean=total_mean, truncation=tail)
            for p in LEVELS:
                r[f"q{p}"] = quantile_from_pmf(pmf, p)
            rows.append((r, pmf))
    T = pd.DataFrame([r for r, _ in rows])

    # what a plan built under the wrong structure delivers
    cross = []
    pm = {(r["P"], r["structure"]): p for r, p in rows}
    for P in (1, 2, 4, 8):
        for assumed in ("A: ship-wide only", "B: interaction only"):
            for truth in ("A: ship-wide only", "B: interaction only"):
                q = T[(T.P == P) & (T.structure == assumed)].q99.iloc[0]
                cross.append(dict(P=P, planned_under=assumed, truth=truth,
                                  q99=int(q), delivered=cover_from_pmf(pm[(P, truth)], int(q))))
    X = pd.DataFrame(cross)

    # convergence: halve the node count and re-read the 99% quantiles
    conv = []
    for P in (1, 8):
        for name, tau, om in (("A: ship-wide only", sigma, 0.0),
                              ("B: interaction only", 0.0, sigma)):
            a, _ = total_pmf(np.full(P, total_mean / P), tau, om, nodes=NODES)
            b, _ = total_pmf(np.full(P, total_mean / P), tau, om, nodes=(NODES - 1) // 2 | 1)
            conv.append(dict(P=P, structure=name,
                             q99_nodes=quantile_from_pmf(a, 99),
                             q99_half=quantile_from_pmf(b, 99),
                             max_abs_pmf_diff=float(np.abs(a - b).max())))
    C = pd.DataFrame(conv)

    T.to_csv(os.path.join(OUT, "multiproduct_plans.csv"), index=False)
    X.to_csv(os.path.join(OUT, "multiproduct_cross.csv"), index=False)
    json.dump(dict(sigma=sigma, total_mean=total_mean, nodes=NODES,
                   plans=T.to_dict("records"), cross=X.to_dict("records"),
                   convergence=C.to_dict("records"),
                   max_truncation=float(T.truncation.abs().max())),
              open(os.path.join(OUT, "multiproduct.json"), "w"), indent=1, default=float)

    print(f"sigma={sigma}, total expected failures={total_mean}\n")
    print(T.pivot(index="P", columns="structure", values="q99").to_string())
    print("\n99% plans at every level:")
    print(T[["P", "structure", "q90", "q95", "q99", "truncation"]].to_string(index=False))
    print("\nplan built under one structure, evaluated under the other (99%):")
    print(X.to_string(index=False))
    print("\nconvergence (node count halved):")
    print(C.to_string(index=False))


if __name__ == "__main__":
    main()
