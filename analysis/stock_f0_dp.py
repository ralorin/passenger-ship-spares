"""
Certificates for the interrupted searches of stock_exact.py at f = 0.

With the ship-wide share at zero the references are independent, so the joint
cover is the product of the marginal covers and problem (3) is separable. With
integer costs (1 or 5 in the grid) it is solved exactly by dynamic programming
over the budget: V_p(b) is the largest sum of log marginal covers of the first
p references at cost at most b. The optimum is the least b with
V_P(b) >= log(level); the gap log(level) - V_P(b - 1) > 0 certifies that no
cheaper vector is feasible. The script also checks the lower bounds of
stock_exact.py against these optima.

    python stock_f0_dp.py
"""
import json
import math
import os

import numpy as np
import pandas as pd

from plan_cover import Zone
from stock_exact import LEVEL, split

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "v2")


def dp_optimum(z, cost, level, bmax):
    P = z.P
    logF = [np.log(np.maximum(z.wu @ G, 1e-300)) for G in z.G]
    lo = z.plan_marginal(level)
    V = np.zeros(bmax + 1)
    arg = []
    for p in range(P):
        c = int(cost[p])
        Vn = np.full(bmax + 1, -np.inf)
        An = np.full(bmax + 1, -1)
        for s in range(lo[p], len(logF[p])):
            if c * s > bmax:
                break
            cand = np.full(bmax + 1, -np.inf)
            cand[c * s:] = V[:bmax + 1 - c * s] + logF[p][s]
            better = cand > Vn
            Vn[better], An[better] = cand[better], s
        V = Vn
        arg.append(An)
    feas = np.flatnonzero(V >= math.log(level))
    if not feas.size:
        return None
    b = int(feas[0])
    s, rem = [], b
    for p in reversed(range(P)):
        # recover the stocks: the choice of reference p at budget rem
        sp = int(arg[p][rem])
        s.append(sp)
        rem -= int(cost[p]) * sp
    s = np.array(s[::-1])
    margin = math.log(level) - (V[b - 1] if b > 0 else -np.inf)
    return s, b, float(margin)


def main():
    T = pd.read_csv(os.path.join(OUT, "stock_exact_part1.csv"))
    rows = []
    for _, r in T[(~T.proved.astype(bool)) & (T.f == 0)].iterrows():
        P = int(r.P)
        a = np.ones(P) if r.means == "equal" else 2.0 ** np.arange(P)
        lams = r.total * a / a.sum()
        c = np.ones(P) if r.costs == "equal" else 5.0 ** (np.arange(P) % 2)
        z = Zone(lams, *split(0.0))
        s, b, margin = dp_optimum(z, c, LEVEL, int(r.cost_exact) + 5)
        J = z.joint(s)
        rows.append(dict(P=P, total=r.total, means=r.means, costs=r.costs, f=0.0,
                         best_found=float(r.cost_exact), lower_bound=float(r.lower_bound),
                         dp_cost=float(c @ s), dp_vector=s.tolist(), dp_cover=J,
                         log_margin_below=margin, certified=bool(margin > 1e-9 and J >= LEVEL)))
        print(rows[-1])
    with open(os.path.join(OUT, "stock_f0_dp.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
