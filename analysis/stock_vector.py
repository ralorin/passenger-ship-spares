"""
Spares are held per product reference, not as a single interchangeable pool.

Analytical / exact numerical. No device data enter.

The decision of the manuscript is a VECTOR of stocks, one entry per product
reference, because a spare inverter does not replace a failed charger. Two
coverage criteria therefore have to be distinguished, and they are not the same
problem:

  JOINT     Pr( Y_p <= s_p for every p ) >= level
            the mission is covered only if every reference is covered. This is
            the criterion the shipboard decision requires when references are
            not interchangeable.

  AGGREGATE Pr( sum_p Y_p <= S ) >= level
            legitimate only when the references ARE interchangeable (a common
            module, or a kit drawn against any of them).

Under the structure of the manuscript, for ship s and reference p,

    Y_p | u, w_p ~ Poisson(lambda_p exp(u + w_p - sigma^2/2)),
    u ~ N(0, tau^2) shared by the whole ship,  w_p ~ N(0, omega^2) independent,
    sigma^2 = tau^2 + omega^2 held fixed as the allocation changes.

The two criteria respond to the allocation in OPPOSITE directions:

  * aggregate: a shared deviation never averages out, so the ship-wide
    allocation needs the most stock;
  * joint: a shared deviation makes the references move together, which RAISES
    the probability that all are simultaneously covered, so the ship-wide
    allocation needs the least stock.

Reporting only the aggregate result would therefore give a fleet the wrong sign.

Implementation. Both latent terms are discretised on a COMMON lattice, so every
value of u+w is a lattice point and the Poisson cdf is tabulated once, as
F[j, s] = F_Pois(s; lambda e^{v_j - sigma^2/2}). The per-reference conditional
cdf G[i, s] = E_w F(s | u_i + w) is then one matrix product, and each criterion
is a weighted sum over the shared term. Without this the cost is quadratic in
the node count and the calculation is infeasible at the accuracy required.

    python stock_vector.py
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
NODES = 801


def _side(sd, h):
    """Simpson weights on the lattice of step h for N(0, sd^2), +-10 sd."""
    if sd <= 0:
        return np.zeros(1, int), np.ones(1)
    m = int(np.ceil(10.0 * sd / h))
    idx = np.arange(-m, m + 1)
    w = np.ones(len(idx))
    if len(idx) >= 3:
        w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    w = w * norm.pdf(idx * h, scale=sd)
    return idx, w / w.sum()


class Zone:
    """Conditional cdf table for one reference of a zone, on a shared lattice."""

    def __init__(self, lam, tau, omega, smax, nodes=NODES):
        s2 = tau ** 2 + omega ** 2
        span = 10.0 * max(tau, omega, 1e-12)
        self.h = 2.0 * span / ((int(nodes) | 1) - 1)
        self.iu, self.wu = _side(tau, self.h)
        self.iw, self.ww = _side(omega, self.h)
        lo = self.iu.min() + self.iw.min()
        hi = self.iu.max() + self.iw.max()
        s = np.arange(smax + 1)
        v = np.arange(lo, hi + 1) * self.h
        rate = lam * np.exp(v - s2 / 2)
        # F[j, s]
        F = poisson.cdf(s[None, :], rate[:, None])
        # G[i, s] = sum_k ww[k] F[iu_i + iw_k - lo, s]
        rows = (self.iu[:, None] + self.iw[None, :]) - lo
        self.G = np.einsum("k,iks->is", self.ww, F[rows])
        self.smax = smax

    def marginal_cover(self, s):
        return float(self.wu @ self.G[:, min(s, self.smax)])

    def joint_cover_common(self, s, P):
        """Pr(all P references <= s), references identical in lambda."""
        return float(self.wu @ (self.G[:, min(s, self.smax)] ** P))

    def smallest(self, level, P):
        """Smallest common per-reference stock meeting the joint criterion."""
        col = self.G ** P
        cov = self.wu @ col
        idx = np.argmax(cov >= level / 100.0)
        return (int(idx), float(cov[idx])) if cov[idx] >= level / 100.0 \
            else (int(self.smax), float(cov[-1]))

    def smallest_marginal(self, level):
        cov = self.wu @ self.G
        idx = np.argmax(cov >= level / 100.0)
        return (int(idx), float(cov[idx])) if cov[idx] >= level / 100.0 \
            else (int(self.smax), float(cov[-1]))


def main():
    sigma = 0.6            # assumed total latent sd, as elsewhere in the paper
    total_mean = 8.0       # assumed expected failures for the whole zone
    structures = [("ship-wide", sigma, 0.0),
                  ("equal split", sigma / np.sqrt(2), sigma / np.sqrt(2)),
                  ("interaction", 0.0, sigma)]

    rows = []
    for P in (1, 2, 4, 8):
        lam = total_mean / P
        smax = int(max(20, np.ceil(lam * np.exp(4.5 * sigma) + 14 * np.sqrt(lam) + 25)))
        for name, tau, om in structures:
            z = Zone(lam, tau, om, smax)
            for lv in LEVELS:
                sm, cm = z.smallest_marginal(lv)
                jm = z.joint_cover_common(sm, P)
                sj, cj = z.smallest(lv, P)
                rows.append(dict(P=P, structure=name, tau=float(tau), omega=float(om),
                                 level=lv, lam_per_ref=float(lam),
                                 s_marginal=sm, marginal_cover=cm,
                                 joint_cover_of_marginal_plan=jm,
                                 s_joint=sj, joint_cover=cj,
                                 total_marginal=int(P * sm), total_joint=int(P * sj)))
        print(f"P={P} done", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "stock_vector.csv"), index=False)

    conv = []
    for P in (1, 8):
        lam = total_mean / P
        smax = int(max(20, np.ceil(lam * np.exp(4.5 * sigma) + 14 * np.sqrt(lam) + 25)))
        for name, tau, om in (structures[0], structures[2]):
            a = Zone(lam, tau, om, smax).smallest(99, P)[0]
            b = Zone(lam, tau, om, smax, nodes=(NODES - 1) // 2 | 1).smallest(99, P)[0]
            conv.append(dict(P=P, structure=name, s_joint=a, s_joint_half=b))
    json.dump(dict(sigma=sigma, total_mean=total_mean, nodes=NODES,
                   rows=T.to_dict("records"), convergence=conv),
              open(os.path.join(OUT, "stock_vector.json"), "w"), indent=1, default=float)

    pd.set_option("display.width", 200)
    q = T[T.level == 99]
    print("\n99% level. s_marg: per-reference marginal quantile, and the JOINT "
          "cover it actually delivers; s_joint: smallest common stock meeting "
          "the joint criterion.\n")
    print(q[["P", "structure", "s_marginal", "joint_cover_of_marginal_plan",
             "s_joint", "total_marginal", "total_joint"]].to_string(index=False))
    print("\nconvergence (node count halved):")
    print(pd.DataFrame(conv).to_string(index=False))


if __name__ == "__main__":
    main()
